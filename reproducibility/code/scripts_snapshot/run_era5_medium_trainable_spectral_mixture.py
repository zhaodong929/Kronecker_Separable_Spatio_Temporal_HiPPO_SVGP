#!/usr/bin/env python3
"""Train a spectral-mixture kernel on ERA5 calibration data, then freeze it.

The fitted kernel is used by the medium-ERA5 Route B safe-lag protocol. This is
intended to be the formal trainable SM experiment, not the earlier fixed-kernel
diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import gpytorch
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise SystemExit("gpytorch is required for the trainable spectral-mixture experiment") from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_hipposvgp_era5_routeb as routeb
from stvgp_kronecker.data.hipposvgp_era5 import load_hipposvgp_era5
from stvgp_kronecker.joint_ssgp_kron.kron_utils import vec_f


DEFAULT_OUTDIR = (
    ROOT
    / "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready"
    / "trainable_spectral_mixture_medium"
)


class ProductSpectralMixtureGP(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood, num_mixtures: int) -> None:
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
        self.feature_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel(active_dims=tuple(range(3, train_x.shape[1])))
        )
        self.covar_module = gpytorch.kernels.ProductKernel(self.temporal_kernel, self.spatial_kernel) + self.feature_kernel

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def ridge_residual(y: np.ndarray, phi: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    precision = phi.T @ phi + float(ridge) * np.eye(phi.shape[1])
    coef = np.linalg.solve(precision, phi.T @ y)
    return y - phi @ coef, coef


def build_training_subset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = load_hipposvgp_era5(
        args.root,
        tasks=args.calibration_tasks,
        variable_index=args.variable_index,
        prefer_scaled=True,
        split=args.split,
        first_n_locations=None,
        random_n_locations=None,
        seed=int(args.seed),
        max_time=None,
    )
    calibration_dataset, _ = routeb.normalise_time_dataset(raw)
    calibration_dataset = routeb.augment_dataset_phi(calibration_dataset, phi_mode="medium_era5")
    subset = routeb.subset_dataset_for_fullgp_mll(
        calibration_dataset,
        max_time=int(args.fit_max_time),
        max_locations=int(args.fit_max_locations),
    )
    routeb_subset = routeb.routeb_dataset_from_era5(
        subset,
        sigma2=1.0,
        args=argparse.Namespace(kernel_variance=1.0),
    )
    times = np.asarray(routeb_subset.times, dtype=float)
    spatial = np.asarray(routeb_subset.spatial_coords, dtype=float)
    s_count, t_count = routeb_subset.Y.shape
    x_time = np.repeat(times, s_count)[:, None]
    x_space = np.tile(spatial, (t_count, 1))
    phi = np.asarray(routeb_subset.Phi, dtype=float)
    x = np.concatenate([x_time, x_space, phi], axis=1)
    y = vec_f(routeb_subset.Y)
    residual, beta = ridge_residual(y, routeb_subset.Phi, ridge=float(args.beta_ridge))
    info = {
        "num_time": int(t_count),
        "num_locations": int(s_count),
        "num_observations": int(y.size),
        "num_phi": int(routeb_subset.Phi.shape[1]),
        "target_variance": float(np.var(y)),
        "residual_variance": float(np.var(residual)),
        "beta_norm": float(np.linalg.norm(beta)),
    }
    return x, y, info


def initialise_sm_kernel(kernel: gpytorch.kernels.SpectralMixtureKernel, train_x: torch.Tensor, train_y: torch.Tensor, *, dim: int) -> None:
    q = int(kernel.num_mixtures)
    y_var = torch.clamp(train_y.var(), min=torch.as_tensor(1e-4, dtype=train_y.dtype, device=train_y.device))
    weights = torch.full((q,), float(y_var.item()) / max(q, 1), dtype=train_y.dtype, device=train_y.device)
    means = torch.linspace(1e-3, 1.5, q, dtype=train_y.dtype, device=train_y.device).view(q, 1, 1)
    scales = torch.full((q, 1, 1), 0.35, dtype=train_y.dtype, device=train_y.device)
    if dim > 1:
        means = means.repeat(1, 1, dim)
        scales = scales.repeat(1, 1, dim)
    kernel.initialize(mixture_weights=weights, mixture_means=means, mixture_scales=scales)


def dense_exact_gp_nlml(
    model: ProductSpectralMixtureGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    jitter: float,
) -> torch.Tensor:
    kt = model.temporal_kernel(train_x).to_dense()
    ks = model.spatial_kernel(train_x).to_dense()
    kphi = model.feature_kernel(train_x).to_dense()
    cov = kt * ks + kphi
    mean = model.mean_module(train_x)
    resid = train_y - mean
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    noise = torch.clamp(likelihood.noise.reshape(()), min=torch.as_tensor(1e-8, dtype=cov.dtype, device=cov.device))
    diag_scale = torch.clamp(torch.mean(torch.diagonal(cov)).detach().abs(), min=torch.as_tensor(1.0, dtype=cov.dtype, device=cov.device))
    base_cov = cov + noise * eye
    chol = None
    jitter_value = float(jitter) * diag_scale
    for _ in range(8):
        try:
            chol = torch.linalg.cholesky(base_cov + jitter_value * eye)
            break
        except RuntimeError:
            jitter_value = jitter_value * 10.0
    if chol is None:
        cov_min = torch.min(torch.diagonal(base_cov)).detach().cpu().item()
        raise RuntimeError(f"dense GP covariance is not positive definite after jitter escalation; min diag={cov_min:.6g}")
    alpha = torch.cholesky_solve(resid.unsqueeze(-1), chol).squeeze(-1)
    logdet = 2.0 * torch.sum(torch.log(torch.diagonal(chol)))
    n = train_y.numel()
    return 0.5 * (torch.dot(resid, alpha) + logdet + n * math.log(2.0 * math.pi)) / n


def fit_spectral_mixture(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, float]]]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.set_default_dtype(torch.float64)
    x_np, y_np, info = build_training_subset(args)
    train_x = torch.as_tensor(x_np, dtype=torch.float64)
    train_y = torch.as_tensor(y_np, dtype=torch.float64)
    train_y = train_y - train_y.mean()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    initial_noise = max(float(np.var(y_np)) * 0.05, 1e-4)
    likelihood.initialize(noise=initial_noise)
    model = ProductSpectralMixtureGP(train_x, train_y, likelihood, num_mixtures=int(args.num_mixtures))
    initialise_sm_kernel(model.temporal_kernel, train_x[:, :1], train_y, dim=1)
    initialise_sm_kernel(model.spatial_kernel, train_x[:, 1:], train_y, dim=2)

    model.train()
    likelihood.train()
    params_by_id = {id(param): param for param in list(model.parameters()) + list(likelihood.parameters())}
    optimizer = torch.optim.Adam(list(params_by_id.values()), lr=float(args.lr))
    trace: list[dict[str, float]] = []
    eye_jitter = float(args.cholesky_jitter)
    for iteration in range(1, int(args.training_iters) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = dense_exact_gp_nlml(model, likelihood, train_x, train_y, jitter=eye_jitter)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params_by_id.values()), max_norm=float(args.grad_clip))
        optimizer.step()
        if iteration == 1 or iteration % int(args.log_every) == 0 or iteration == int(args.training_iters):
            trace.append(
                {
                    "iteration": float(iteration),
                    "negative_mll": float(loss.detach().cpu().item()),
                    "noise": float(likelihood.noise.detach().cpu().item()),
                }
            )

    temporal_weights = model.temporal_kernel.mixture_weights.detach().cpu().numpy().reshape(-1)
    temporal_means = model.temporal_kernel.mixture_means.detach().cpu().numpy().reshape(int(args.num_mixtures), -1)[:, 0]
    temporal_scales = model.temporal_kernel.mixture_scales.detach().cpu().numpy().reshape(int(args.num_mixtures), -1)[:, 0]
    spatial_weights = model.spatial_kernel.mixture_weights.detach().cpu().numpy().reshape(-1)
    spatial_means = model.spatial_kernel.mixture_means.detach().cpu().numpy().reshape(int(args.num_mixtures), -1)
    spatial_scales = model.spatial_kernel.mixture_scales.detach().cpu().numpy().reshape(int(args.num_mixtures), -1)

    temporal_amp = max(float(temporal_weights.sum()), 1e-8)
    spatial_amp = max(float(spatial_weights.sum()), 1e-8)
    params = {
        "schema": "trainable_product_spectral_mixture_v1",
        "source": "gpytorch SpectralMixtureKernel product fitted on the initial ERA5 calibration task residuals",
        "num_mixtures": int(args.num_mixtures),
        "kernel_variance": float(temporal_amp * spatial_amp),
        "noise_variance": float(likelihood.noise.detach().cpu().item()),
        "routeb_noise": float(math.sqrt(max(float(likelihood.noise.detach().cpu().item()), 1e-12))),
        "model_ell_t": 1.0,
        "temporal_weights": (temporal_weights / temporal_amp).tolist(),
        "temporal_means": np.maximum(temporal_means, 1e-8).tolist(),
        "temporal_scales": np.maximum(temporal_scales, 1e-8).tolist(),
        "spatial_weights": (spatial_weights / spatial_amp).tolist(),
        "spatial_means": np.maximum(spatial_means, 1e-8).tolist(),
        "spatial_scales": np.maximum(spatial_scales, 1e-8).tolist(),
        "training": {
            **info,
            "iterations": int(args.training_iters),
            "learning_rate": float(args.lr),
            "grad_clip": float(args.grad_clip),
            "cholesky_jitter": float(args.cholesky_jitter),
            "beta_ridge": float(args.beta_ridge),
        },
    }
    return params, trace


def write_trace(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_routeb_with_frozen_sm(args: argparse.Namespace, param_path: Path, params: dict[str, Any]) -> None:
    run_dir = Path(args.outdir) / "routeb_medium_trainable_spectral_mixture"
    run_dir.mkdir(parents=True, exist_ok=True)
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
        *args.routeb_methods,
        "--eval-modes",
        "seen_history",
        "--phi-mode",
        "medium_era5",
        "--ohsvgp-heldout-eval",
        "--heldout-split-seeds",
        *[str(v) for v in args.heldout_split_seeds],
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
        f"{float(params['model_ell_t']):.12g}",
        "--routeb-noise",
        f"{float(params['routeb_noise']):.12g}",
        "--kernel-variance",
        f"{float(params['kernel_variance']):.12g}",
    ]
    if args.save_forgetting_block_pairs:
        cmd.append("--save-forgetting-block-pairs")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def summarise_routeb(outdir: Path) -> dict[str, Any]:
    summary_path = outdir / "routeb_medium_trainable_spectral_mixture/era5_routeb_summary.csv"
    if not summary_path.exists():
        return {"summary_path": str(summary_path), "available": False}
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    picked = [
        row
        for row in rows
        if row.get("eval_mode") == "seen_history" and row.get("method") in {"no_transfer", "mean_field", "structured_joint"}
    ]
    comparison_path = outdir / "trainable_sm_medium_seen_history_summary.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        if picked:
            writer = csv.DictWriter(f, fieldnames=list(picked[0].keys()))
            writer.writeheader()
            writer.writerows(picked)
    return {"summary_path": str(summary_path), "comparison_path": str(comparison_path), "available": True, "rows": picked}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--calibration-tasks", nargs="+", default=["task_1"])
    parser.add_argument("--online-tasks", nargs="+", default=["task_2"])
    parser.add_argument("--variable-index", type=int, default=0)
    parser.add_argument("--split", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-mixtures", type=int, default=3)
    parser.add_argument("--fit-max-time", type=int, default=30)
    parser.add_argument("--fit-max-locations", type=int, default=30)
    parser.add_argument("--training-iters", type=int, default=160)
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
    parser.add_argument("--routeb-methods", nargs="+", default=["no_transfer", "mean_field", "structured_joint"])
    parser.add_argument("--save-forgetting-block-pairs", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--force-fit", action="store_true")
    parser.add_argument("--force-routeb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    param_path = args.outdir / "trainable_sm_params.json"
    trace_path = args.outdir / "trainable_sm_training_trace.csv"

    if args.force_fit or not param_path.exists():
        params, trace = fit_spectral_mixture(args)
        param_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
        write_trace(trace_path, trace)
    else:
        params = json.loads(param_path.read_text(encoding="utf-8"))

    run_summary_path = args.outdir / "routeb_medium_trainable_spectral_mixture/era5_routeb_summary.csv"
    if not args.fit_only and (args.force_routeb or not run_summary_path.exists()):
        run_routeb_with_frozen_sm(args, param_path, params)

    summary = {
        "scope": "trainable spectral-mixture medium-ERA5 experiment",
        "params": str(param_path),
        "training_trace": str(trace_path),
        "routeb": summarise_routeb(args.outdir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
