#!/usr/bin/env python3
"""ICLR formal ERA5 baseline: STVGP-style separable spatio-temporal GP.

The official AaltoML/spatio-temporal-GPs code depends on Bayes-Newton/JAX
versions that are not available in the current Python environment. This script
therefore implements a local, auditable STVGP-style baseline under the same ERA5
held-out protocol:

- separable temporal/spatial Matern-3/2 covariance;
- exact grid algebra via Kronecker eigensystems;
- optional X-lag ridge mean, with the GP fitted to residuals.

It is not a drop-in Bayes-Newton reproduction; it is a comparable baseline
adapter that mirrors the paper's core separable spatio-temporal modelling
assumption while preserving the local ERA5 split and metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hipposvgp_era5_routeb import augment_dataset_phi, fixed_spatial_train_test_split
from stvgp_kronecker.data.hipposvgp_era5 import HippoERA5Dataset, iter_online_blocks, load_hipposvgp_era5


Z90 = 1.6448536269514722
MIN_VAR = 1e-8


def matern32_kernel(x: np.ndarray, y: np.ndarray | None = None, *, lengthscale: float = 1.0, variance: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = x if y is None else np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    diff = x[:, None, :] - y[None, :, :]
    dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
    scaled = np.sqrt(3.0) * dist / max(float(lengthscale), 1e-12)
    return float(variance) * (1.0 + scaled) * np.exp(-scaled)


def gaussian_nll(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    var = np.maximum(np.asarray(var, dtype=float), MIN_VAR)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * var) + (np.asarray(y) - mean) ** 2 / var))


def coverage90(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    half = Z90 * np.sqrt(np.maximum(var, MIN_VAR))
    return float(np.mean((y >= mean - half) & (y <= mean + half)))


def ece_gaussian(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    levels = np.asarray([0.5, 0.8, 0.9, 0.95])
    z_values = np.asarray([0.67448975, 1.28155157, 1.64485363, 1.95996398])
    sigma = np.sqrt(np.maximum(var, MIN_VAR))
    errors = []
    for level, z in zip(levels, z_values):
        half = z * sigma
        empirical = np.mean((y >= mean - half) & (y <= mean + half))
        errors.append(abs(float(empirical) - float(level)))
    return float(np.mean(errors))


def phi_for_time_space(dataset: HippoERA5Dataset, time_indices: np.ndarray, spatial_indices: np.ndarray) -> np.ndarray:
    ns = dataset.coords.shape[0]
    rows: list[int] = []
    for t_idx in np.asarray(time_indices, dtype=int).reshape(-1):
        rows.extend((int(t_idx) * ns + np.asarray(spatial_indices, dtype=int)).tolist())
    return dataset.Phi[np.asarray(rows, dtype=int)]


def fit_ridge_mean(
    dataset: HippoERA5Dataset,
    time_indices: np.ndarray,
    spatial_indices: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, float]:
    phi = phi_for_time_space(dataset, time_indices, spatial_indices)
    y = dataset.Y[np.ix_(time_indices, spatial_indices)].reshape(-1)
    eye = np.eye(phi.shape[1])
    beta = np.linalg.solve(phi.T @ phi + float(ridge) * eye, phi.T @ y)
    residual = y - phi @ beta
    return beta, float(max(np.var(residual), MIN_VAR))


def predict_ridge_mean(dataset: HippoERA5Dataset, time_indices: np.ndarray, spatial_indices: np.ndarray, beta: np.ndarray) -> np.ndarray:
    phi = phi_for_time_space(dataset, time_indices, spatial_indices)
    return (phi @ beta).reshape(len(time_indices), len(spatial_indices))


@dataclass
class SeparablePosterior:
    times_train: np.ndarray
    coords_train: np.ndarray
    y_residual_train: np.ndarray
    ell_t: float
    ell_s: float
    noise: float
    variance: float
    coord_mean: np.ndarray
    coord_scale: np.ndarray
    time_origin: float
    time_scale: float
    eig_t: tuple[np.ndarray, np.ndarray] | None = None
    eig_s: tuple[np.ndarray, np.ndarray] | None = None
    alpha: np.ndarray | None = None

    def _norm_time(self, times: np.ndarray) -> np.ndarray:
        return ((np.asarray(times, dtype=float).reshape(-1) - self.time_origin) / self.time_scale)[:, None]

    def _norm_coords(self, coords: np.ndarray) -> np.ndarray:
        return (np.asarray(coords, dtype=float) - self.coord_mean) / self.coord_scale

    def fit(self) -> None:
        kt = matern32_kernel(self._norm_time(self.times_train), lengthscale=self.ell_t, variance=self.variance)
        ks = matern32_kernel(self._norm_coords(self.coords_train), lengthscale=self.ell_s, variance=1.0)
        kt = 0.5 * (kt + kt.T) + 1e-7 * np.eye(kt.shape[0])
        ks = 0.5 * (ks + ks.T) + 1e-7 * np.eye(ks.shape[0])
        lt, ut = np.linalg.eigh(kt)
        ls, us = np.linalg.eigh(ks)
        lt = np.maximum(lt, 1e-10)
        ls = np.maximum(ls, 1e-10)
        y_tilde = ut.T @ self.y_residual_train @ us
        denom = lt[:, None] * ls[None, :] + self.noise
        alpha_tilde = y_tilde / np.maximum(denom, 1e-12)
        self.eig_t = (lt, ut)
        self.eig_s = (ls, us)
        self.alpha = ut @ alpha_tilde @ us.T

    def predict(self, times: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.eig_t is None or self.eig_s is None or self.alpha is None:
            raise RuntimeError("Posterior must be fitted before prediction")
        lt, ut = self.eig_t
        ls, us = self.eig_s
        kt_star = matern32_kernel(self._norm_time(times), self._norm_time(self.times_train), lengthscale=self.ell_t, variance=self.variance)
        ks_star = matern32_kernel(self._norm_coords(coords), self._norm_coords(self.coords_train), lengthscale=self.ell_s, variance=1.0)
        mean = kt_star @ self.alpha @ ks_star.T
        qt = kt_star @ ut
        qs = ks_star @ us
        denom_inv = 1.0 / np.maximum(lt[:, None] * ls[None, :] + self.noise, 1e-12)
        reduction = (qt * qt) @ denom_inv @ (qs * qs).T
        prior_diag = float(self.variance) * np.ones((len(times), len(coords)))
        var = np.maximum(prior_diag - reduction + self.noise, MIN_VAR)
        return mean, var


def metric_row(
    *,
    method: str,
    split_seed: int,
    block_id: int,
    y: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    runtime: float,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    std = np.sqrt(np.maximum(var, MIN_VAR))
    return {
        "method": method,
        "eval_mode": "seen_history",
        "heldout_split_seed": int(split_seed),
        "block_id": int(block_id),
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "mae": float(np.mean(np.abs(y - mean))),
        "nll": gaussian_nll(y, mean, var),
        "coverage90": coverage90(y, mean, var),
        "ece": ece_gaussian(y, mean, var),
        "avg_var": float(np.mean(var)),
        "avg_std": float(np.mean(std)),
        "avg_width90": float(np.mean(2.0 * Z90 * std)),
        "runtime": float(runtime),
        "runtime_per_block": float(runtime),
        "num_test": int(y.size),
        **diagnostics,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["eval_mode"], int(row["heldout_split_seed"])), []).append(row)
    out_rows = []
    metrics = ["rmse", "mae", "nll", "coverage90", "ece", "avg_var", "avg_std", "avg_width90", "runtime_per_block", "num_test"]
    for (method, mode, seed), group in sorted(groups.items()):
        out: dict[str, Any] = {
            "method": method,
            "eval_mode": mode,
            "heldout_split_seed": seed,
            "num_rows": len(group),
        }
        for metric in metrics:
            vals = np.asarray([float(row[metric]) for row in group], dtype=float)
            out[metric] = float(np.mean(vals))
            out[f"{metric}_se"] = float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
        for key in ["ell_t", "ell_s", "noise", "mean_mode", "ridge", "num_train_space", "num_test_space"]:
            out[key] = group[-1].get(key, "")
        out_rows.append(out)
    return out_rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_split(
    dataset: HippoERA5Dataset,
    *,
    split_seed: int,
    test_fraction: float,
    block_size: int,
    ell_t: float,
    ell_s: float,
    noise_scale: float,
    ridge: float,
    mean_mode: str,
    schedule: str,
) -> list[dict[str, Any]]:
    train_idx, test_idx = fixed_spatial_train_test_split(dataset.Y.shape[1], test_fraction=test_fraction, seed=split_seed)
    blocks = iter_online_blocks(dataset.Y.shape[0], block_size)
    rows: list[dict[str, Any]] = []
    time_all = np.arange(dataset.Y.shape[0])
    coord_mean = dataset.coords.mean(axis=0, keepdims=True)
    coord_scale = np.maximum(dataset.coords.std(axis=0, keepdims=True), 1e-8)
    time_origin = float(dataset.times[0])
    time_scale = max(float(dataset.times[-1] - dataset.times[0]), 1e-12)
    for block_id, block in enumerate(blocks):
        stop = block.stop or dataset.Y.shape[0]
        if schedule == "seen_history_refit":
            train_times = time_all[:stop]
            eval_times = time_all[:stop]
            eval_mode = "seen_history"
        elif schedule == "current_block_refit":
            train_times = time_all[block]
            eval_times = time_all[block]
            eval_mode = "current_block"
        elif schedule == "initial_only":
            train_times = time_all[blocks[0]]
            eval_times = time_all[block]
            eval_mode = "current_block_initial_only"
        else:
            raise ValueError("schedule must be seen_history_refit, current_block_refit, or initial_only")
        started = time.perf_counter()
        if mean_mode == "xlag_ridge":
            beta, residual_var = fit_ridge_mean(dataset, train_times, train_idx, ridge=ridge)
            train_mean = predict_ridge_mean(dataset, train_times, train_idx, beta)
            test_mean = predict_ridge_mean(dataset, eval_times, test_idx, beta)
        elif mean_mode == "xlag_mean_only":
            beta, residual_var = fit_ridge_mean(dataset, train_times, train_idx, ridge=ridge)
            test_mean = predict_ridge_mean(dataset, eval_times, test_idx, beta)
            y_true = dataset.Y[np.ix_(eval_times, test_idx)]
            var = np.full_like(y_true, max(residual_var, MIN_VAR), dtype=float)
            runtime = time.perf_counter() - started
            rows.append(
                metric_row(
                    method=f"stvgp_exact_{mean_mode}_{schedule}",
                    split_seed=split_seed,
                    block_id=block_id,
                    y=y_true,
                    mean=test_mean,
                    var=var,
                    runtime=runtime,
                    diagnostics={
                        "ell_t": float("nan"),
                        "ell_s": float("nan"),
                        "noise": float(residual_var),
                        "noise_scale": float("nan"),
                        "residual_var": float(residual_var),
                        "mean_mode": mean_mode,
                        "schedule": schedule,
                        "ridge": float(ridge),
                        "num_train_space": int(train_idx.size),
                        "num_test_space": int(test_idx.size),
                        "num_train_time": int(train_times.size),
                    },
                )
            )
            rows[-1]["eval_mode"] = eval_mode
            continue
        elif mean_mode == "zero":
            beta = np.zeros(0)
            train_mean = np.zeros((len(train_times), len(train_idx)))
            test_mean = np.zeros((len(eval_times), len(test_idx)))
            residual_var = float(max(np.var(dataset.Y[np.ix_(train_times, train_idx)]), MIN_VAR))
        else:
            raise ValueError("mean_mode must be zero or xlag_ridge")
        y_resid = dataset.Y[np.ix_(train_times, train_idx)] - train_mean
        noise = max(float(noise_scale) * residual_var, MIN_VAR)
        posterior = SeparablePosterior(
            times_train=dataset.times[train_times],
            coords_train=dataset.coords[train_idx],
            y_residual_train=y_resid,
            ell_t=ell_t,
            ell_s=ell_s,
            noise=noise,
            variance=1.0,
            coord_mean=coord_mean,
            coord_scale=coord_scale,
            time_origin=time_origin,
            time_scale=time_scale,
        )
        posterior.fit()
        gp_mean, gp_var = posterior.predict(dataset.times[eval_times], dataset.coords[test_idx])
        mean = test_mean + gp_mean
        y_true = dataset.Y[np.ix_(eval_times, test_idx)]
        runtime = time.perf_counter() - started
        rows.append(
            metric_row(
                method=f"stvgp_exact_{mean_mode}",
                split_seed=split_seed,
                block_id=block_id,
                y=y_true,
                mean=mean,
                var=gp_var,
                runtime=runtime,
                diagnostics={
                    "ell_t": float(ell_t),
                    "ell_s": float(ell_s),
                    "noise": float(noise),
                    "noise_scale": float(noise_scale),
                    "residual_var": float(residual_var),
                    "mean_mode": mean_mode,
                    "schedule": schedule,
                    "ridge": float(ridge),
                    "num_train_space": int(train_idx.size),
                    "num_test_space": int(test_idx.size),
                    "num_train_time": int(train_times.size),
                },
            )
        )
        rows[-1]["eval_mode"] = eval_mode
    return rows


def make_plot(summary: list[dict[str, Any]], outdir: Path) -> None:
    if not summary:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    methods = [row["method"] for row in summary]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
    axes[0].bar(x, [float(row["rmse"]) for row in summary], color="#6B8BA4")
    axes[0].axhline(0.1899366567512086, color="black", linestyle="--", linewidth=1, label="X-lag + RBF")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Held-out seen-history RMSE")
    axes[1].bar(x, [float(row["nll"]) for row in summary], color="#8A6F9E")
    axes[1].axhline(-0.035034236524608, color="black", linestyle="--", linewidth=1, label="X-lag + RBF")
    axes[1].set_ylabel("NLL/NLPD")
    axes[1].set_title("Held-out seen-history NLL")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    fig.savefig(outdir / "stvgp_baseline_vs_xlag_rbf.png", dpi=220)
    fig.savefig(outdir / "stvgp_baseline_vs_xlag_rbf.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/stvgp_era5_baseline")
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--task", default="task_2")
    parser.add_argument("--variable-index", type=int, default=0)
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--heldout-test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--mean-modes", nargs="+", choices=["zero", "xlag_ridge", "xlag_mean_only"], default=["zero", "xlag_ridge"])
    parser.add_argument("--schedules", nargs="+", choices=["seen_history_refit", "current_block_refit", "initial_only"], default=["seen_history_refit"])
    parser.add_argument("--xlag-length", type=int, default=10)
    parser.add_argument("--ell-t", type=float, default=0.1)
    parser.add_argument("--ell-s", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dataset = load_hipposvgp_era5(
        root=args.root,
        tasks=(args.task,),
        variable_index=args.variable_index,
        split=args.split,
    )
    dataset = augment_dataset_phi(dataset, phi_mode="medium_era5_xlag", xlag_length=args.xlag_length)
    all_rows: list[dict[str, Any]] = []
    for seed in args.split_seeds:
        for schedule in args.schedules:
            for mean_mode in args.mean_modes:
                rows = run_split(
                    dataset,
                    split_seed=int(seed),
                    test_fraction=args.heldout_test_fraction,
                    block_size=args.block_size,
                    ell_t=args.ell_t,
                    ell_s=args.ell_s,
                    noise_scale=args.noise_scale,
                    ridge=args.ridge,
                    mean_mode=mean_mode,
                    schedule=schedule,
                )
                all_rows.extend(rows)
    summary = summarize(all_rows)
    write_csv(all_rows, outdir / "stvgp_era5_baseline_metrics.csv")
    write_csv(summary, outdir / "stvgp_era5_baseline_summary.csv")
    make_plot([row for row in summary if int(row["heldout_split_seed"]) == 1], outdir / "figures")
    report = {
        "description": "Local STVGP-style exact separable Matern baseline for ERA5 held-out seen-history.",
        "official_repo": "baselines/external/aaltoml_spatio_temporal_gps",
        "official_smoke": "../official_stvgp_smoke_stdout.log",
        "reference_xlag_rbf": {"test_rmse": 0.1899366567512086, "test_nll": -0.035034236524608},
        "args": vars(args),
        "outputs": {
            "metrics": str(outdir / "stvgp_era5_baseline_metrics.csv"),
            "summary": str(outdir / "stvgp_era5_baseline_summary.csv"),
        },
    }
    (outdir / "stvgp_era5_baseline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
