#!/usr/bin/env python3
"""Corrected trainable spectral-mixture experiment for medium-ERA5.

This script keeps the spectral-mixture shape learned from a dense calibration
subset, but treats Route-B-facing hyperparameters as validation choices rather
than blindly accepting the dense GP optimum. It also supports fitting the
spectral shape on the medium-feature residual, which better matches the
structured Route B decomposition.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import gpytorch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("gpytorch is required for this experiment") from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_era5_medium_trainable_spectral_mixture as base_sm


DEFAULT_OUTDIR = (
    ROOT
    / "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready"
    / "trainable_spectral_mixture_medium_corrected"
)


class ResidualProductSpectralMixtureGP(gpytorch.models.ExactGP):
    """Product SM kernel without a feature covariance, for residual fitting."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        num_mixtures: int,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.temporal_kernel = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=num_mixtures,
            ard_num_dims=1,
            active_dims=(0,),
        )
        self.spatial_kernel = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=num_mixtures,
            ard_num_dims=2,
            active_dims=(1, 2),
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        kt = self.temporal_kernel(x).to_dense()
        ks = self.spatial_kernel(x).to_dense()
        cov = kt * ks
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), cov)


def dense_residual_gp_nlml(
    model: ResidualProductSpectralMixtureGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    jitter: float,
) -> torch.Tensor:
    kt = model.temporal_kernel(train_x).to_dense()
    ks = model.spatial_kernel(train_x).to_dense()
    cov = kt * ks
    resid = train_y - model.mean_module(train_x)
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    noise = torch.clamp(likelihood.noise.reshape(()), min=torch.as_tensor(1e-8, dtype=cov.dtype, device=cov.device))
    diag_scale = torch.clamp(torch.mean(torch.diagonal(cov)).detach().abs(), min=torch.as_tensor(1.0, dtype=cov.dtype, device=cov.device))
    base_cov = cov + noise * eye
    jitter_value = float(jitter) * diag_scale
    chol = None
    for _ in range(8):
        try:
            chol = torch.linalg.cholesky(base_cov + jitter_value * eye)
            break
        except RuntimeError:
            jitter_value = jitter_value * 10.0
    if chol is None:
        raise RuntimeError("residual dense GP covariance is not positive definite")
    alpha = torch.cholesky_solve(resid.unsqueeze(-1), chol).squeeze(-1)
    logdet = 2.0 * torch.sum(torch.log(torch.diagonal(chol)))
    n = train_y.numel()
    return 0.5 * (torch.dot(resid, alpha) + logdet + n * math.log(2.0 * math.pi)) / n


@dataclass(frozen=True)
class Candidate:
    name: str
    target: str
    fit_seed: int
    ell_t: float
    sigma: float
    kernel_variance: float


def _extract_sm_params(model: Any, args: argparse.Namespace, info: dict[str, Any], target: str, final_loss: float) -> dict[str, Any]:
    temporal_weights = model.temporal_kernel.mixture_weights.detach().cpu().numpy().reshape(-1)
    temporal_means = model.temporal_kernel.mixture_means.detach().cpu().numpy().reshape(int(args.num_mixtures), -1)[:, 0]
    temporal_scales = model.temporal_kernel.mixture_scales.detach().cpu().numpy().reshape(int(args.num_mixtures), -1)[:, 0]
    spatial_weights = model.spatial_kernel.mixture_weights.detach().cpu().numpy().reshape(-1)
    spatial_means = model.spatial_kernel.mixture_means.detach().cpu().numpy().reshape(int(args.num_mixtures), -1)
    spatial_scales = model.spatial_kernel.mixture_scales.detach().cpu().numpy().reshape(int(args.num_mixtures), -1)
    temporal_amp = max(float(temporal_weights.sum()), 1e-8)
    spatial_amp = max(float(spatial_weights.sum()), 1e-8)
    return {
        "schema": "corrected_trainable_product_spectral_mixture_v2",
        "source": "gpytorch SpectralMixtureKernel product fitted on ERA5 calibration subset; Route-B hyperparameters selected separately",
        "fit_target": target,
        "num_mixtures": int(args.num_mixtures),
        "kernel_variance": float(temporal_amp * spatial_amp),
        "routeb_noise": float("nan"),
        "model_ell_t": float("nan"),
        "temporal_weights": (temporal_weights / temporal_amp).tolist(),
        "temporal_means": np.maximum(temporal_means, 1e-8).tolist(),
        "temporal_scales": np.maximum(temporal_scales, 1e-8).tolist(),
        "spatial_weights": (spatial_weights / spatial_amp).tolist(),
        "spatial_means": np.maximum(spatial_means, 1e-8).tolist(),
        "spatial_scales": np.maximum(spatial_scales, 1e-8).tolist(),
        "training": {
            **info,
            "target": target,
            "iterations": int(args.training_iters),
            "learning_rate": float(args.lr),
            "grad_clip": float(args.grad_clip),
            "cholesky_jitter": float(args.cholesky_jitter),
            "beta_ridge": float(args.beta_ridge),
            "final_negative_mll": float(final_loss),
        },
    }


def fit_one_shape(args: argparse.Namespace, *, target: str, seed: int) -> tuple[dict[str, Any], list[dict[str, float]]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.set_default_dtype(torch.float64)
    x_np, y_np, info = base_sm.build_training_subset(args)
    residual_np, _ = base_sm.ridge_residual(y_np, x_np[:, 3:], ridge=float(args.beta_ridge))
    if target == "residual":
        y_fit_np = residual_np
    elif target == "raw":
        y_fit_np = y_np
    else:
        raise ValueError(f"unknown fit target: {target}")
    train_x = torch.as_tensor(x_np, dtype=torch.float64)
    train_y = torch.as_tensor(y_fit_np, dtype=torch.float64)
    train_y = train_y - train_y.mean()

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
    )
    initial_noise = max(float(np.var(y_fit_np)) * 0.05, 1e-8)
    likelihood.initialize(noise=initial_noise)
    if target == "residual":
        model: Any = ResidualProductSpectralMixtureGP(train_x, train_y, likelihood, num_mixtures=int(args.num_mixtures))
    else:
        model = base_sm.ProductSpectralMixtureGP(train_x, train_y, likelihood, num_mixtures=int(args.num_mixtures))
    base_sm.initialise_sm_kernel(model.temporal_kernel, train_x[:, :1], train_y, dim=1)
    base_sm.initialise_sm_kernel(model.spatial_kernel, train_x[:, 1:], train_y, dim=2)

    model.train()
    likelihood.train()
    params_by_id = {id(param): param for param in list(model.parameters()) + list(likelihood.parameters())}
    optimizer = torch.optim.Adam(list(params_by_id.values()), lr=float(args.lr))
    trace: list[dict[str, float]] = []
    final_loss = float("nan")
    for iteration in range(1, int(args.training_iters) + 1):
        optimizer.zero_grad(set_to_none=True)
        if target == "residual":
            loss = dense_residual_gp_nlml(model, likelihood, train_x, train_y, jitter=float(args.cholesky_jitter))
        else:
            loss = base_sm.dense_exact_gp_nlml(model, likelihood, train_x, train_y, jitter=float(args.cholesky_jitter))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params_by_id.values()), max_norm=float(args.grad_clip))
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
        if iteration == 1 or iteration % int(args.log_every) == 0 or iteration == int(args.training_iters):
            trace.append(
                {
                    "fit_seed": float(seed),
                    "iteration": float(iteration),
                    "negative_mll": final_loss,
                    "noise": float(likelihood.noise.detach().cpu().item()),
                }
            )
    params = _extract_sm_params(model, args, info, target, final_loss)
    params["training"]["fit_seed"] = int(seed)
    params["training"]["fit_noise_variance"] = float(likelihood.noise.detach().cpu().item())
    return params, trace


def run_routeb_candidate(
    args: argparse.Namespace,
    *,
    candidate: Candidate,
    param_path: Path,
    run_dir: Path,
    routeb_methods: list[str],
    split_seeds: list[int],
    save_per_location: bool = False,
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_hipposvgp_era5_routeb.py"),
        "--outdir",
        str(run_dir),
        "--root",
        str(args.root),
        "--calibration-tasks",
        *args.calibration_tasks,
        "--online-tasks",
        *args.online_tasks,
        "--variable-index",
        str(args.variable_index),
        "--split",
        str(args.split),
        "--block-size",
        str(args.block_size),
        "--routeb-methods",
        *routeb_methods,
        "--eval-modes",
        "seen_history",
        "--phi-mode",
        "medium_era5",
        "--ohsvgp-heldout-eval",
        "--heldout-split-seeds",
        *[str(v) for v in split_seeds],
        "--seeds",
        str(args.seed),
        "--mt",
        str(args.mt),
        "--ms",
        str(args.ms),
        "--kernel-type",
        "spectral_mixture",
        "--spectral-mixture-param-path",
        str(param_path),
        "--temporal-backend",
        "analytic_hippo_rff",
        "--temporal-rff-sample-size",
        str(args.temporal_rff_sample_size),
        "--temporal-rff-seed",
        str(args.temporal_rff_seed),
        "--prediction-mode",
        "streaming_sylvester",
        "--prediction-chunk-size",
        str(args.prediction_chunk_size),
        "--hyperparam-fit-mode",
        "none",
        "--ell-t-fit-mode",
        "none",
        "--model-ell-t",
        f"{candidate.ell_t:.12g}",
        "--routeb-noise",
        f"{candidate.sigma:.12g}",
        "--kernel-variance",
        f"{candidate.kernel_variance:.12g}",
    ]
    if save_per_location:
        cmd.extend(["--save-per-location-predictions", "--per-location-indices", str(args.per_location_index)])
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def read_structured_summary(run_dir: Path) -> dict[str, float]:
    path = run_dir / "era5_routeb_summary.csv"
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["method"] == "structured_joint" and r["eval_mode"] == "seen_history")
    return {
        "rmse": float(row["rmse"]),
        "nll": float(row["nll"]),
        "coverage90": float(row["coverage90"]),
        "ece": float(row["ece"]),
        "avg_var": float(row["avg_var"]),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_final(run_dir: Path, out_path: Path) -> None:
    path = run_dir / "era5_routeb_summary.csv"
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["eval_mode"] == "seen_history"]
    out_rows = []
    for method in ["no_transfer", "mean_field", "structured_joint"]:
        picked = [r for r in rows if r["method"] == method]
        out = {"method": method}
        for metric in ["rmse", "nll", "coverage90", "ece", "avg_var"]:
            vals = np.asarray([float(r[metric]) for r in picked], dtype=float)
            out[f"{metric}_mean"] = float(vals.mean())
            out[f"{metric}_sd"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        out_rows.append(out)
    write_rows(out_path, out_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--calibration-tasks", nargs="+", default=["task_1"])
    parser.add_argument("--online-tasks", nargs="+", default=["task_2"])
    parser.add_argument("--variable-index", type=int, default=0)
    parser.add_argument("--split", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fit-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-mixtures", type=int, default=3)
    parser.add_argument("--fit-max-time", type=int, default=30)
    parser.add_argument("--fit-max-locations", type=int, default=30)
    parser.add_argument("--training-iters", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--beta-ridge", type=float, default=1e-3)
    parser.add_argument("--cholesky-jitter", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--mt", type=int, default=16)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--temporal-rff-sample-size", type=int, default=512)
    parser.add_argument("--temporal-rff-seed", type=int, default=0)
    parser.add_argument("--prediction-chunk-size", type=int, default=8192)
    parser.add_argument("--heldout-split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--validation-split-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--per-location-index", type=int, default=486)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_traces: list[dict[str, float]] = []
    best_by_target: dict[str, Path] = {}
    best_param_by_target: dict[str, dict[str, Any]] = {}
    fit_rows: list[dict[str, Any]] = []
    for target in ["raw", "residual"]:
        target_dir = args.outdir / f"fit_{target}"
        target_dir.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")
        for fit_seed in args.fit_seeds:
            param_path = target_dir / f"trainable_sm_{target}_seed{fit_seed}.json"
            trace_path = target_dir / f"trainable_sm_{target}_seed{fit_seed}_trace.csv"
            if args.force or not param_path.exists():
                params, trace = fit_one_shape(args, target=target, seed=int(fit_seed))
                param_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
                write_rows(trace_path, trace)
            else:
                params = json.loads(param_path.read_text(encoding="utf-8"))
                with trace_path.open("r", newline="", encoding="utf-8") as f:
                    trace = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
            all_traces.extend(trace)
            final_loss = float(params["training"]["final_negative_mll"])
            fit_rows.append(
                {
                    "target": target,
                    "fit_seed": fit_seed,
                    "final_negative_mll": final_loss,
                    "fit_noise_variance": params["training"].get("fit_noise_variance", float("nan")),
                    "param_path": str(param_path.relative_to(args.outdir)),
                }
            )
            if final_loss < best_loss:
                best_loss = final_loss
                best_by_target[target] = param_path
                best_param_by_target[target] = params
    write_rows(args.outdir / "fit_multistart_summary.csv", fit_rows)
    write_rows(args.outdir / "fit_training_trace_all.csv", all_traces)

    candidates = [
        Candidate("raw_shape_fixed_routeb_hypers", "raw", int(args.fit_seeds[0]), 0.2, 0.05, 0.25),
        Candidate("residual_shape_fixed_routeb_hypers", "residual", int(args.fit_seeds[0]), 0.2, 0.05, 0.25),
        Candidate("residual_shape_fixed_routeb_hypers_sigma0075", "residual", int(args.fit_seeds[0]), 0.2, 0.075, 0.25),
    ]

    validation_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        param_path = best_by_target[candidate.target]
        run_dir = args.outdir / "validation" / candidate.name
        summary_path = run_dir / "era5_routeb_summary.csv"
        if args.force or not summary_path.exists():
            run_routeb_candidate(
                args,
                candidate=candidate,
                param_path=param_path,
                run_dir=run_dir,
                routeb_methods=["structured_joint"],
                split_seeds=args.validation_split_seeds,
            )
        metrics = read_structured_summary(run_dir)
        validation_rows.append(
            {
                "candidate": candidate.name,
                "target": candidate.target,
                "ell_t": candidate.ell_t,
                "sigma": candidate.sigma,
                "kernel_variance": candidate.kernel_variance,
                **metrics,
                "param_path": str(param_path.relative_to(args.outdir)),
                "run_dir": str(run_dir.relative_to(args.outdir)),
            }
        )
    write_rows(args.outdir / "validation_candidate_summary.csv", validation_rows)
    best_row = min(validation_rows, key=lambda r: (float(r["rmse"]), float(r["nll"])))
    best_candidate = next(c for c in candidates if c.name == best_row["candidate"])
    best_param_path = best_by_target[best_candidate.target]

    final_params = json.loads(best_param_path.read_text(encoding="utf-8"))
    final_params["model_ell_t"] = best_candidate.ell_t
    final_params["routeb_noise"] = best_candidate.sigma
    final_params["kernel_variance"] = best_candidate.kernel_variance
    final_params["selection"] = {
        "selected_candidate": best_candidate.name,
        "selection_metric": "lowest validation split-0 structured-joint RMSE, NLL as tie-breaker",
        "validation_summary": "validation_candidate_summary.csv",
    }
    final_param_path = args.outdir / "corrected_trainable_sm_selected_params.json"
    final_param_path.write_text(json.dumps(final_params, indent=2), encoding="utf-8")

    final_run_dir = args.outdir / "routeb_medium_trainable_spectral_mixture_corrected"
    if args.force or not (final_run_dir / "era5_routeb_summary.csv").exists():
        run_routeb_candidate(
            args,
            candidate=best_candidate,
            param_path=final_param_path,
            run_dir=final_run_dir,
            routeb_methods=["no_transfer", "mean_field", "structured_joint"],
            split_seeds=args.heldout_split_seeds,
        )
    aggregate_final(final_run_dir, args.outdir / "corrected_trainable_sm_medium_seen_history_3split_summary.csv")

    single_run_dir = args.outdir / "routeb_medium_trainable_spectral_mixture_corrected_singlepoint"
    if args.force or not (single_run_dir / "era5_routeb_per_location_predictions.csv").exists():
        run_routeb_candidate(
            args,
            candidate=best_candidate,
            param_path=final_param_path,
            run_dir=single_run_dir,
            routeb_methods=["structured_joint"],
            split_seeds=[args.heldout_split_seeds[0]],
            save_per_location=True,
        )

    print(
        json.dumps(
            {
                "outdir": str(args.outdir),
                "selected_candidate": best_candidate.name,
                "selected_params": str(final_param_path),
                "validation_summary": str(args.outdir / "validation_candidate_summary.csv"),
                "final_summary": str(args.outdir / "corrected_trainable_sm_medium_seen_history_3split_summary.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
