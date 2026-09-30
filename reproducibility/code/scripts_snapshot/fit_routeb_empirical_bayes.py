#!/usr/bin/env python3
"""Empirical-Bayes calibration for the finite analytic HiPPO Route B model.

The latent Gaussian posterior remains closed form. This script only optimizes
the continuous kernel and likelihood hyperparameters on the independent ERA5
calibration task, after which they can be frozen for Task-2 batch/online runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hipposvgp_era5_routeb import normalise_time_dataset
from stvgp_kronecker.data.hipposvgp_era5 import load_hipposvgp_era5
from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    make_analytic_temporal_builder,
    select_spatial_inducing_indices,
    temporal_spec_for_block,
)


DTYPE = torch.float64


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def standardized_coordinates(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    return (coords - coords.mean(axis=0, keepdims=True)) / np.maximum(
        coords.std(axis=0, keepdims=True), 1e-12
    )


def spatial_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengthscale: torch.Tensor,
    kernel_type: str,
) -> torch.Tensor:
    distance = torch.cdist(x1 / lengthscale, x2 / lengthscale)
    if kernel_type == "rbf":
        return torch.exp(-0.5 * distance.square())
    if kernel_type == "matern32":
        scaled = math.sqrt(3.0) * distance
        return (1.0 + scaled) * torch.exp(-scaled)
    raise ValueError(f"Unsupported kernel type: {kernel_type}")


def robust_cholesky(matrix: torch.Tensor, base_jitter: float = 1e-6) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    diag_scale = torch.clamp(torch.diagonal(matrix).mean().detach().abs(), min=1.0)
    eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    jitter = base_jitter * diag_scale
    for _ in range(8):
        chol, info = torch.linalg.cholesky_ex(matrix + jitter * eye)
        if int(info.max().item()) == 0:
            return chol
        jitter = jitter * 10.0
    raise RuntimeError("Cholesky failed after eight jitter escalations")


def finite_model_nlml(
    y: torch.Tensor,
    times: np.ndarray,
    spatial_train: torch.Tensor,
    spatial_inducing: torch.Tensor,
    builder,
    spatial_log_lengthscale: torch.Tensor,
    noise_log_std: torch.Tensor,
    kernel_type: str,
) -> torch.Tensor:
    horizon = temporal_spec_for_block(times, slice(0, len(times)), moving=False)
    kuu_t = builder.compute_kuu_t(horizon)
    kfu_t = builder.compute_kfu_t(times, horizon)
    chol_t = robust_cholesky(kuu_t)
    feature_t = torch.linalg.solve_triangular(chol_t, kfu_t.transpose(0, 1), upper=False).transpose(0, 1)

    ell_s = torch.exp(spatial_log_lengthscale)
    kuu_s = spatial_kernel(spatial_inducing, spatial_inducing, ell_s, kernel_type)
    kfu_s = spatial_kernel(spatial_train, spatial_inducing, ell_s, kernel_type)
    chol_s = robust_cholesky(kuu_s)
    feature_s = torch.linalg.solve_triangular(chol_s, kfu_s.transpose(0, 1), upper=False).transpose(0, 1)

    u_t, s_t, _ = torch.linalg.svd(feature_t, full_matrices=False)
    u_s, s_s, _ = torch.linalg.svd(feature_s, full_matrices=False)
    projected = u_t.transpose(0, 1) @ y @ u_s
    eigenvalues = s_t.square()[:, None] * s_s.square()[None, :]
    noise_var = torch.exp(2.0 * noise_log_std)
    quad = y.square().sum() / noise_var
    quad = quad - torch.sum(
        projected.square() * eigenvalues / (noise_var * (noise_var + eigenvalues))
    )
    num_obs = y.numel()
    logdet = num_obs * torch.log(noise_var) + torch.log1p(eigenvalues / noise_var).sum()
    return 0.5 * (quad + logdet + num_obs * math.log(2.0 * math.pi)) / num_obs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--task", default="task_1")
    parser.add_argument("--variable-index", type=int, default=0)
    parser.add_argument("--mt", type=int, required=True)
    parser.add_argument("--ms", type=int, required=True)
    parser.add_argument("--kernel-type", choices=["rbf", "matern32"], default="matern32")
    parser.add_argument("--spatial-inducing-selection", choices=["linspace", "farthest", "kmeans"], default="kmeans")
    parser.add_argument("--num-calibration-locations", type=int, default=200)
    parser.add_argument("--num-calibration-times", type=int, default=186)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--initial-ell-t", type=float, default=0.05)
    parser.add_argument("--initial-ell-s", type=float, default=0.35)
    parser.add_argument("--initial-kernel-variance", type=float, default=1.0)
    parser.add_argument("--initial-noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dataset_raw = load_hipposvgp_era5(
        args.root, tasks=(args.task,), variable_index=args.variable_index, split="all"
    )
    dataset, time_scale = normalise_time_dataset(dataset_raw)
    num_times = min(args.num_calibration_times, dataset.Y.shape[0])
    num_locations = min(args.num_calibration_locations, dataset.Y.shape[1])
    time_idx = np.linspace(0, dataset.Y.shape[0] - 1, num_times).round().astype(int)
    spatial_idx = np.linspace(0, dataset.Y.shape[1] - 1, num_locations).round().astype(int)
    coords_all = standardized_coordinates(dataset.coords)
    inducing_idx = select_spatial_inducing_indices(
        coords_all, args.ms, method=args.spatial_inducing_selection
    )
    times = np.asarray(dataset.times[time_idx], dtype=float)
    y = torch.as_tensor(dataset.Y[np.ix_(time_idx, spatial_idx)], dtype=DTYPE)
    spatial_train = torch.as_tensor(coords_all[spatial_idx], dtype=DTYPE)
    spatial_inducing = torch.as_tensor(coords_all[inducing_idx], dtype=DTYPE)

    builder = make_analytic_temporal_builder(
        mt=args.mt,
        lengthscale=args.initial_ell_t,
        variance=args.initial_kernel_variance,
        rff_sample_size=args.rff_sample_size,
        seed=args.seed,
        kernel_type=args.kernel_type,
    )
    spatial_log_lengthscale = torch.nn.Parameter(
        torch.log(torch.tensor(args.initial_ell_s, dtype=DTYPE))
    )
    noise_log_std = torch.nn.Parameter(
        torch.log(torch.tensor(args.initial_noise, dtype=DTYPE))
    )
    parameters = [
        builder.log_lengthscale,
        builder.log_variance,
        spatial_log_lengthscale,
        noise_log_std,
    ]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    trace: list[dict[str, float]] = []
    best: dict[str, float] | None = None
    best_state: list[torch.Tensor] | None = None
    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = finite_model_nlml(
            y,
            times,
            spatial_train,
            spatial_inducing,
            builder,
            spatial_log_lengthscale,
            noise_log_std,
            args.kernel_type,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite empirical-Bayes objective at iteration {iteration}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            builder.log_lengthscale.clamp_(math.log(0.005), math.log(3.0))
            builder.log_variance.clamp_(math.log(0.01), math.log(10.0))
            spatial_log_lengthscale.clamp_(math.log(0.02), math.log(5.0))
            noise_log_std.clamp_(math.log(0.01), math.log(1.0))
        row = {
            "iteration": float(iteration),
            "nlml_per_observation": float(loss.detach()),
            "ell_t": float(builder.lengthscale.detach()),
            "ell_s": float(torch.exp(spatial_log_lengthscale).detach()),
            "kernel_variance": float(builder.variance.detach()),
            "noise_std": float(torch.exp(noise_log_std).detach()),
        }
        trace.append(row)
        if best is None or row["nlml_per_observation"] < best["nlml_per_observation"]:
            best = dict(row)
            best_state = [parameter.detach().clone() for parameter in parameters]
        if iteration == 1 or iteration % 10 == 0 or iteration == args.iterations:
            print(json.dumps(row), flush=True)

    assert best is not None and best_state is not None
    with torch.no_grad():
        for parameter, value in zip(parameters, best_state):
            parameter.copy_(value)
    payload = {
        "method": "finite_model_empirical_bayes",
        "calibration_task": args.task,
        "target": "direct_y",
        "kernel_type": args.kernel_type,
        "mt": args.mt,
        "ms": args.ms,
        "num_calibration_times": num_times,
        "num_calibration_locations": num_locations,
        "time_scale": time_scale,
        "spatial_inducing_selection": args.spatial_inducing_selection,
        "spatial_inducing_indices": inducing_idx.tolist(),
        "initial": {
            "ell_t": args.initial_ell_t,
            "ell_s": args.initial_ell_s,
            "kernel_variance": args.initial_kernel_variance,
            "noise_std": args.initial_noise,
        },
        "best": best,
        "args": vars(args),
    }
    write_csv(trace, outdir / "training_trace.csv")
    (outdir / "learned_hyperparameters.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
