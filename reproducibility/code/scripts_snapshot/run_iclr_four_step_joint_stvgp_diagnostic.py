#!/usr/bin/env python3
"""Final-block exact STVGP joint-mean diagnostic for the four-step report."""

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
    matern32_kernel,
    phi_for_time_space,
    predict_ridge_mean,
)
from stvgp_kronecker.data.hipposvgp_era5 import load_hipposvgp_era5


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(method: str, y: np.ndarray, mean: np.ndarray, var: np.ndarray, runtime: float, **extra: Any) -> dict[str, Any]:
    var = np.maximum(var, MIN_VAR)
    return {
        "method": method,
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "nll": gaussian_nll(y, mean, var),
        "coverage90": coverage90(y, mean, var),
        "ece": ece_gaussian(y, mean, var),
        "avg_std": float(np.mean(np.sqrt(var))),
        "runtime": float(runtime),
        "num_test": int(y.size),
        **extra,
    }


def kron_inv_apply_matrix(B: np.ndarray, ut: np.ndarray, us: np.ndarray, denom: np.ndarray, *, nt: int, ns: int) -> np.ndarray:
    """Apply (Kt kron Ks + noise I)^-1 to columns of B."""

    out = np.empty_like(B, dtype=float)
    for j in range(B.shape[1]):
        mat = B[:, j].reshape(nt, ns)
        coeff = (ut.T @ mat @ us) / denom
        out[:, j] = (ut @ coeff @ us.T).reshape(-1)
    return out


def kron_posterior_mean_and_var(
    *,
    y_residual: np.ndarray,
    train_times: np.ndarray,
    train_coords: np.ndarray,
    test_times: np.ndarray,
    test_coords: np.ndarray,
    coord_mean: np.ndarray,
    coord_scale: np.ndarray,
    time_origin: float,
    time_scale: float,
    ell_t: float,
    ell_s: float,
    noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    xt = ((train_times - time_origin) / time_scale)[:, None]
    xs = (train_coords - coord_mean) / coord_scale
    xte = ((test_times - time_origin) / time_scale)[:, None]
    xse = (test_coords - coord_mean) / coord_scale
    kt = matern32_kernel(xt, lengthscale=ell_t, variance=1.0)
    ks = matern32_kernel(xs, lengthscale=ell_s, variance=1.0)
    kt = 0.5 * (kt + kt.T) + 1e-7 * np.eye(kt.shape[0])
    ks = 0.5 * (ks + ks.T) + 1e-7 * np.eye(ks.shape[0])
    lt, ut = np.linalg.eigh(kt)
    ls, us = np.linalg.eigh(ks)
    lt = np.maximum(lt, 1e-10)
    ls = np.maximum(ls, 1e-10)
    denom = lt[:, None] * ls[None, :] + noise
    y_tilde = ut.T @ y_residual @ us
    alpha_tilde = y_tilde / denom
    alpha = ut @ alpha_tilde @ us.T
    kt_star = matern32_kernel(xte, xt, lengthscale=ell_t, variance=1.0)
    ks_star = matern32_kernel(xse, xs, lengthscale=ell_s, variance=1.0)
    mean = kt_star @ alpha @ ks_star.T
    qt = kt_star @ ut
    qs = ks_star @ us
    reduction = (qt * qt) @ (1.0 / denom) @ (qs * qs).T
    var = np.maximum(1.0 - reduction + noise, MIN_VAR)
    return mean, var, alpha.reshape(-1), (ut, us, denom, kt_star, ks_star, 1.0 / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/four_step_priority_experiments/step3_joint_stvgp")
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--heldout-test-fraction", type=float, default=0.2)
    parser.add_argument("--xlag-length", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--ell-t", type=float, default=0.1)
    parser.add_argument("--ell-s", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--beta-prior-variance", type=float, default=100.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dataset = load_hipposvgp_era5(root=args.root, tasks=("task_2",), variable_index=0, split="all")
    dataset = augment_dataset_phi(dataset, phi_mode="medium_era5_xlag", xlag_length=args.xlag_length)
    train_idx, test_idx = fixed_spatial_train_test_split(dataset.Y.shape[1], test_fraction=args.heldout_test_fraction, seed=args.split_seed)
    train_times_idx = np.arange(dataset.Y.shape[0])
    test_times_idx = np.arange(dataset.Y.shape[0])
    train_times = dataset.times[train_times_idx]
    test_times = dataset.times[test_times_idx]
    coord_mean = dataset.coords.mean(axis=0, keepdims=True)
    coord_scale = np.maximum(dataset.coords.std(axis=0, keepdims=True), 1e-8)
    time_origin = float(dataset.times[0])
    time_scale = max(float(dataset.times[-1] - dataset.times[0]), 1e-12)
    y_train = dataset.Y[np.ix_(train_times_idx, train_idx)]
    y_test = dataset.Y[np.ix_(test_times_idx, test_idx)]
    rows: list[dict[str, Any]] = []

    started = time.perf_counter()
    beta_ridge, residual_var = fit_ridge_mean(dataset, train_times_idx, train_idx, ridge=args.ridge)
    train_mean = predict_ridge_mean(dataset, train_times_idx, train_idx, beta_ridge)
    test_mean = predict_ridge_mean(dataset, test_times_idx, test_idx, beta_ridge)
    noise = max(args.noise_scale * residual_var, MIN_VAR)
    gp_mean, gp_var, _, cache = kron_posterior_mean_and_var(
        y_residual=y_train - train_mean,
        train_times=train_times,
        train_coords=dataset.coords[train_idx],
        test_times=test_times,
        test_coords=dataset.coords[test_idx],
        coord_mean=coord_mean,
        coord_scale=coord_scale,
        time_origin=time_origin,
        time_scale=time_scale,
        ell_t=args.ell_t,
        ell_s=args.ell_s,
        noise=noise,
    )
    runtime = time.perf_counter() - started
    rows.append(metrics("Exact STVGP residual two-stage ridge mean", y_test, test_mean + gp_mean, gp_var, runtime, residual_var=float(residual_var), noise=float(noise)))

    started = time.perf_counter()
    phi_train = phi_for_time_space(dataset, train_times_idx, train_idx)
    phi_test = phi_for_time_space(dataset, test_times_idx, test_idx)
    y_vec = y_train.reshape(-1)
    ut, us, denom, kt_star, ks_star, denom_inv = cache
    ky_inv_y = kron_inv_apply_matrix(y_vec[:, None], ut, us, denom, nt=len(train_times_idx), ns=len(train_idx))[:, 0]
    ky_inv_phi = kron_inv_apply_matrix(phi_train, ut, us, denom, nt=len(train_times_idx), ns=len(train_idx))
    prior_prec = np.eye(phi_train.shape[1]) / max(args.beta_prior_variance, 1e-12)
    beta_prec = prior_prec + phi_train.T @ ky_inv_phi
    beta_rhs = phi_train.T @ ky_inv_y
    beta_cov = np.linalg.solve(beta_prec + 1e-8 * np.eye(beta_prec.shape[0]), np.eye(beta_prec.shape[0]))
    beta_mean = beta_cov @ beta_rhs
    residual_vec = y_vec - phi_train @ beta_mean
    ky_inv_resid = kron_inv_apply_matrix(residual_vec[:, None], ut, us, denom, nt=len(train_times_idx), ns=len(train_idx))[:, 0]
    alpha = ky_inv_resid.reshape(len(train_times_idx), len(train_idx))
    joint_gp_mean = kt_star @ alpha @ ks_star.T
    joint_mean = phi_test @ beta_mean + joint_gp_mean.reshape(-1)

    # Predictive variance: exact GP conditional variance plus beta uncertainty.
    delta = np.empty_like(phi_test)
    for j in range(phi_train.shape[1]):
        inv_col = ky_inv_phi[:, j].reshape(len(train_times_idx), len(train_idx))
        projected = kt_star @ inv_col @ ks_star.T
        delta[:, j] = phi_test[:, j] - projected.reshape(-1)
    beta_var = np.einsum("ij,jk,ik->i", delta, beta_cov, delta)
    joint_var = gp_var.reshape(-1) + np.maximum(beta_var, 0.0)
    runtime = time.perf_counter() - started
    rows.append(metrics("Joint exact STVGP beta-GP coupling", y_test.reshape(-1), joint_mean, joint_var, runtime, beta_prior_variance=args.beta_prior_variance, noise=float(noise)))

    write_csv(rows, outdir / "joint_stvgp_final_block_summary.csv")
    (outdir / "joint_stvgp_final_block_report.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows, "outdir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
