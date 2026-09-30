#!/usr/bin/env python3
"""P2 exact/direct/residual decomposition under one fixed Matérn protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hipposvgp_era5_routeb import augment_dataset_phi, fixed_spatial_train_test_split
from scripts.run_iclr_formal_stvgp_baseline import (
    MIN_VAR,
    coverage90,
    ece_gaussian,
    fit_ridge_mean,
    gaussian_nll,
    phi_for_time_space,
    predict_ridge_mean,
)
from scripts.run_iclr_four_step_joint_stvgp_diagnostic import (
    kron_inv_apply_matrix,
    kron_posterior_mean_and_var,
)
from stvgp_kronecker.data.hipposvgp_era5 import load_hipposvgp_era5


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(method: str, seed: int, y: np.ndarray, mean: np.ndarray, var: np.ndarray, runtime: float, **extra: Any) -> dict[str, Any]:
    var = np.maximum(np.asarray(var, dtype=float), MIN_VAR)
    y = np.asarray(y, dtype=float)
    mean = np.asarray(mean, dtype=float)
    return {
        "method": method,
        "heldout_split_seed": seed,
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "nll": gaussian_nll(y, mean, var),
        "coverage90": coverage90(y, mean, var),
        "ece": ece_gaussian(y, mean, var),
        "avg_std": float(np.mean(np.sqrt(var))),
        "runtime_sec": runtime,
        "num_test": int(y.size),
        **extra,
    }


def add_pointwise(
    rows: list[dict[str, Any]],
    *,
    method: str,
    seed: int,
    dataset: Any,
    test_idx: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
) -> None:
    y = np.asarray(y).reshape(dataset.Y.shape[0], test_idx.size)
    mean = np.asarray(mean).reshape(dataset.Y.shape[0], test_idx.size)
    var = np.asarray(var).reshape(dataset.Y.shape[0], test_idx.size)
    for t in range(dataset.Y.shape[0]):
        for j, location in enumerate(test_idx):
            rows.append(
                {
                    "method": method,
                    "heldout_split_seed": seed,
                    "time_index": t,
                    "time": float(dataset.times[t]),
                    "location_index": int(location),
                    "lat": float(dataset.coords[location, 0]),
                    "lon": float(dataset.coords[location, 1]),
                    "y_true": float(y[t, j]),
                    "pred_mean": float(mean[t, j]),
                    "pred_var": float(var[t, j]),
                    "error": float(y[t, j] - mean[t, j]),
                }
            )


def run_seed(dataset: Any, seed: int, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_idx, test_idx = fixed_spatial_train_test_split(
        dataset.Y.shape[1], test_fraction=args.test_fraction, seed=seed
    )
    time_idx = np.arange(dataset.Y.shape[0])
    train_times = dataset.times[time_idx]
    test_times = dataset.times[time_idx]
    y_train = dataset.Y[np.ix_(time_idx, train_idx)]
    y_test = dataset.Y[np.ix_(time_idx, test_idx)]
    coord_mean = dataset.coords.mean(axis=0, keepdims=True)
    coord_scale = np.maximum(dataset.coords.std(axis=0, keepdims=True), 1e-8)
    time_origin = float(dataset.times[0])
    time_scale = max(float(dataset.times[-1] - dataset.times[0]), 1e-12)
    beta_ridge, residual_var = fit_ridge_mean(dataset, time_idx, train_idx, ridge=args.ridge)
    train_mean = predict_ridge_mean(dataset, time_idx, train_idx, beta_ridge)
    test_mean = predict_ridge_mean(dataset, time_idx, test_idx, beta_ridge)
    noise = max(args.noise_scale * residual_var, MIN_VAR)
    common = {
        "train_times": train_times,
        "train_coords": dataset.coords[train_idx],
        "test_times": test_times,
        "test_coords": dataset.coords[test_idx],
        "coord_mean": coord_mean,
        "coord_scale": coord_scale,
        "time_origin": time_origin,
        "time_scale": time_scale,
        "ell_t": args.ell_t,
        "ell_s": args.ell_s,
        "noise": noise,
    }
    rows: list[dict[str, Any]] = []
    pointwise: list[dict[str, Any]] = []

    started = time.perf_counter()
    direct_mean, direct_var, _, _ = kron_posterior_mean_and_var(
        y_residual=y_train, **common
    )
    method = "Original STVGP direct target"
    rows.append(metric_row(method, seed, y_test, direct_mean, direct_var, time.perf_counter() - started, noise_variance=noise, ell_t=args.ell_t, ell_s=args.ell_s))
    add_pointwise(pointwise, method=method, seed=seed, dataset=dataset, test_idx=test_idx, y=y_test, mean=direct_mean, var=direct_var)

    started = time.perf_counter()
    mean_only_var = np.full_like(y_test, residual_var, dtype=float)
    method = "X-lag mean only"
    rows.append(metric_row(method, seed, y_test, test_mean, mean_only_var, time.perf_counter() - started, residual_variance=residual_var))
    add_pointwise(pointwise, method=method, seed=seed, dataset=dataset, test_idx=test_idx, y=y_test, mean=test_mean, var=mean_only_var)

    started = time.perf_counter()
    gp_mean, gp_var, _, cache = kron_posterior_mean_and_var(
        y_residual=y_train - train_mean, **common
    )
    two_stage_mean = test_mean + gp_mean
    method = "STVGP residual + X-lag two-stage"
    rows.append(metric_row(method, seed, y_test, two_stage_mean, gp_var, time.perf_counter() - started, residual_variance=residual_var, noise_variance=noise, ell_t=args.ell_t, ell_s=args.ell_s))
    add_pointwise(pointwise, method=method, seed=seed, dataset=dataset, test_idx=test_idx, y=y_test, mean=two_stage_mean, var=gp_var)

    started = time.perf_counter()
    phi_train = phi_for_time_space(dataset, time_idx, train_idx)
    phi_test = phi_for_time_space(dataset, time_idx, test_idx)
    y_vec = y_train.reshape(-1)
    ut, us, denom, kt_star, ks_star, _ = cache
    ky_inv_y = kron_inv_apply_matrix(
        y_vec[:, None], ut, us, denom, nt=time_idx.size, ns=train_idx.size
    )[:, 0]
    ky_inv_phi = kron_inv_apply_matrix(
        phi_train, ut, us, denom, nt=time_idx.size, ns=train_idx.size
    )
    prior_prec = np.eye(phi_train.shape[1]) / max(args.beta_prior_variance, 1e-12)
    beta_prec = prior_prec + phi_train.T @ ky_inv_phi
    beta_rhs = phi_train.T @ ky_inv_y
    beta_cov = np.linalg.solve(
        beta_prec + 1e-8 * np.eye(beta_prec.shape[0]), np.eye(beta_prec.shape[0])
    )
    beta_mean = beta_cov @ beta_rhs
    residual_vec = y_vec - phi_train @ beta_mean
    ky_inv_resid = kron_inv_apply_matrix(
        residual_vec[:, None], ut, us, denom, nt=time_idx.size, ns=train_idx.size
    )[:, 0]
    alpha = ky_inv_resid.reshape(time_idx.size, train_idx.size)
    joint_gp_mean = kt_star @ alpha @ ks_star.T
    joint_mean = phi_test @ beta_mean + joint_gp_mean.reshape(-1)
    delta = np.empty_like(phi_test)
    for column in range(phi_train.shape[1]):
        inv_col = ky_inv_phi[:, column].reshape(time_idx.size, train_idx.size)
        projected = kt_star @ inv_col @ ks_star.T
        delta[:, column] = phi_test[:, column] - projected.reshape(-1)
    beta_var = np.einsum("ij,jk,ik->i", delta, beta_cov, delta)
    joint_var = gp_var.reshape(-1) + np.maximum(beta_var, 0.0)
    method = "Joint exact STVGP beta-GP"
    rows.append(metric_row(method, seed, y_test.reshape(-1), joint_mean, joint_var, time.perf_counter() - started, noise_variance=noise, beta_prior_variance=args.beta_prior_variance, ell_t=args.ell_t, ell_s=args.ell_s))
    add_pointwise(pointwise, method=method, seed=seed, dataset=dataset, test_idx=test_idx, y=y_test, mean=joint_mean, var=joint_var)
    return rows, pointwise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--xlag-length", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--ell-t", type=float, default=0.1)
    parser.add_argument("--ell-s", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--beta-prior-variance", type=float, default=100.0)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dataset = load_hipposvgp_era5(
        root=args.root, tasks=("task_2",), variable_index=0, split="all"
    )
    dataset = augment_dataset_phi(
        dataset, phi_mode="medium_era5_xlag", xlag_length=args.xlag_length
    )
    rows: list[dict[str, Any]] = []
    pointwise: list[dict[str, Any]] = []
    for seed in args.split_seeds:
        seed_rows, seed_pointwise = run_seed(dataset, seed, args)
        rows.extend(seed_rows)
        pointwise.extend(seed_pointwise)
    write_csv(rows, outdir / "exact_decomposition_metrics.csv")
    write_csv(pointwise, outdir / "exact_decomposition_pointwise.csv")
    payload = {"args": vars(args), "rows": rows}
    (outdir / "exact_decomposition.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
