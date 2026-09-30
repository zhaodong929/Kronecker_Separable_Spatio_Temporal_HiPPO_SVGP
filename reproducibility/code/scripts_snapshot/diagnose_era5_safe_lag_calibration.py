#!/usr/bin/env python3
"""Safe-lag ERA5 calibration diagnostics.

This standalone script intentionally leaves the main ERA5 runner unchanged. It
first creates pointwise held-out predictions using the corrected
``--ohsvgp-heldout-eval`` path, so medium-ERA5 target lags at test locations are
recursively filled from predictive means rather than from ground-truth labels.

The generated diagnostics are:
1. NLL decomposition for base and medium-ERA5 safe-lag predictions.
2. A variance-scaling baseline fitted on an early time prefix and evaluated on
   the later held-out times.
3. A post-hoc lag-assisted mean shrink/noise stress test using the safe-lag
   base and medium predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER_READY = ROOT / "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready"
OUTDIR = PAPER_READY / "safe_lag_calibration_diagnostics"
RUNNER = ROOT / "scripts/run_hipposvgp_era5_routeb.py"
PYTHON = Path(sys.executable)
Z90 = 1.6448536269514722


def require_pyplot():
    import matplotlib.pyplot as plt

    if not hasattr(plt, "subplots"):
        raise RuntimeError("matplotlib.pyplot is available but does not expose subplots()")
    return plt


BASE_RUN_ARGS = [
    "--root",
    "data/era5/processed_timeseries_4",
    "--calibration-tasks",
    "task_1",
    "--online-tasks",
    "task_2",
    "--variable-index",
    "0",
    "--split",
    "all",
    "--block-size",
    "10",
    "--routeb-methods",
    "structured_joint",
    "--eval-modes",
    "seen_history",
    "--ohsvgp-heldout-eval",
    "--heldout-split-seeds",
    "0",
    "--seeds",
    "0",
    "--save-per-location-predictions",
    "--prediction-mode",
    "streaming_sylvester",
    "--prediction-chunk-size",
    "8192",
    "--hyperparam-fit-mode",
    "none",
    "--ell-t-fit-mode",
    "none",
    "--model-ell-t",
    "0.05",
    "--routeb-noise",
    "0.1",
    "--kernel-type",
    "rbf",
    "--kernel-variance",
    "1.0",
    "--mt",
    "8",
    "--ms",
    "64",
    "--temporal-backend",
    "analytic_hippo_rff",
]


def run_prediction(phi_mode: str, force: bool, outdir: Path) -> Path:
    run_dir = outdir / f"{phi_mode}_heldout_split0"
    pred_path = run_dir / "era5_routeb_per_location_predictions.csv"
    if pred_path.exists() and not force:
        return pred_path
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--outdir",
        str(run_dir),
        *BASE_RUN_ARGS,
        "--phi-mode",
        phi_mode,
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return pred_path


def read_prediction_csv(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows.extend(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No prediction rows in {path}")
    rows.sort(key=lambda r: (int(float(r["time_index"])), int(float(r["location_index"]))))
    return {
        "time_index": np.asarray([int(float(r["time_index"])) for r in rows], dtype=int),
        "location_index": np.asarray([int(float(r["location_index"])) for r in rows], dtype=int),
        "y": np.asarray([float(r["y_true"]) for r in rows], dtype=float),
        "mean": np.asarray([float(r["pred_mean"]) for r in rows], dtype=float),
        "var": np.maximum(np.asarray([float(r["pred_var_y"]) for r in rows], dtype=float), 1e-10),
    }


def nll_parts(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> dict[str, np.ndarray]:
    var = np.maximum(var, 1e-10)
    err2 = (y - mean) ** 2
    return {
        "nll": 0.5 * (np.log(2.0 * np.pi * var) + err2 / var),
        "logvar_term": 0.5 * np.log(2.0 * np.pi * var),
        "quad_term": 0.5 * err2 / var,
    }


def metrics(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> dict[str, float]:
    parts = nll_parts(y, mean, var)
    sd = np.sqrt(np.maximum(var, 1e-10))
    return {
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "nll": float(np.mean(parts["nll"])),
        "logvar_term": float(np.mean(parts["logvar_term"])),
        "quad_term": float(np.mean(parts["quad_term"])),
        "avg_var": float(np.mean(var)),
        "avg_width90": float(np.mean(2.0 * Z90 * sd)),
        "coverage90": float(np.mean((y >= mean - Z90 * sd) & (y <= mean + Z90 * sd))),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fit_variance_scale(y: np.ndarray, mean: np.ndarray, var: np.ndarray, train_mask: np.ndarray) -> dict[str, float]:
    alpha_grid = np.concatenate([np.linspace(0.05, 1.0, 20), np.linspace(1.1, 5.0, 40), np.linspace(5.5, 20.0, 30)])
    tau_grid = np.concatenate([[0.0], np.logspace(-8, 0, 33)])
    best = {"alpha": 1.0, "tau2": 0.0, "train_nll": float("inf")}
    yt, mt, vt = y[train_mask], mean[train_mask], var[train_mask]
    for alpha in alpha_grid:
        for tau2 in tau_grid:
            nll = metrics(yt, mt, alpha * vt + tau2)["nll"]
            if nll < best["train_nll"]:
                best = {"alpha": float(alpha), "tau2": float(tau2), "train_nll": float(nll)}
    return best


def align_pair(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    left_keys = [(int(t), int(s)) for t, s in zip(left["time_index"], left["location_index"])]
    right_pos = {(int(t), int(s)): i for i, (t, s) in enumerate(zip(right["time_index"], right["location_index"]))}
    left_take: list[int] = []
    right_take: list[int] = []
    for i, key in enumerate(left_keys):
        j = right_pos.get(key)
        if j is not None:
            left_take.append(i)
            right_take.append(j)
    li = np.asarray(left_take, dtype=int)
    ri = np.asarray(right_take, dtype=int)
    return ({k: v[li] for k, v in left.items()}, {k: v[ri] for k, v in right.items()})


def plot_decomposition(outdir: Path, rows: list[dict[str, Any]]) -> Path:
    plt = require_pyplot()
    labels = [r["mode"] for r in rows]
    logv = np.asarray([float(r["logvar_term"]) for r in rows])
    quad = np.asarray([float(r["quad_term"]) for r in rows])
    rmse = np.asarray([float(r["rmse"]) for r in rows])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), constrained_layout=True)
    axes[0].bar(x, logv, label="0.5 log(2pi var)", color="#4E79A7")
    axes[0].bar(x, quad, bottom=logv, label="0.5 error^2/var", color="#F28E2B")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("NLL contribution")
    axes[0].set_title("Safe-lag NLL decomposition")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, rmse, color="#59A14F")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Mean error")
    path = outdir / "fig_safe_lag_nll_decomposition.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_scaling(outdir: Path, rows: list[dict[str, Any]]) -> Path:
    plt = require_pyplot()
    labels = [r["variant"] for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.0), constrained_layout=True)
    for ax, key, title, color in [
        (axes[0], "nll", "NLL", "#F28E2B"),
        (axes[1], "avg_width90", "Avg 90% width", "#4E79A7"),
        (axes[2], "coverage90", "Coverage90", "#59A14F"),
    ]:
        vals = [float(r[key]) for r in rows]
        ax.bar(x, vals, color=color)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=25, ha="right")
    axes[2].axhline(0.9, color="black", linestyle="--", linewidth=0.8)
    path = outdir / "fig_safe_lag_variance_scaling.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_lag_stress(outdir: Path, rows: list[dict[str, Any]]) -> Path:
    plt = require_pyplot()
    labels = [r["variant"] for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1), constrained_layout=True)
    axes[0].bar(x, [float(r["rmse"]) for r in rows], color="#59A14F")
    axes[0].set_title("RMSE")
    axes[1].bar(x, [float(r["nll"]) for r in rows], color="#F28E2B")
    axes[1].set_title("NLL")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
    path = outdir / "fig_safe_lag_lag_shrink_noise.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    base_path = run_prediction("base", force=args.force, outdir=outdir)
    medium_path = run_prediction("medium_era5", force=args.force, outdir=outdir)

    base = read_prediction_csv(base_path)
    medium = read_prediction_csv(medium_path)
    base, medium = align_pair(base, medium)

    decomposition_rows = [
        {"mode": "base safe-lag heldout", **metrics(base["y"], base["mean"], base["var"])},
        {"mode": "medium-ERA5 safe-lag heldout", **metrics(medium["y"], medium["mean"], medium["var"])},
    ]
    write_csv(outdir / "safe_lag_nll_decomposition.csv", decomposition_rows)
    decomp_fig = plot_decomposition(outdir, decomposition_rows)

    unique_times = np.asarray(sorted(set(int(t) for t in medium["time_index"])), dtype=int)
    train_n_times = max(5, int(round(unique_times.size * float(args.calibration_fraction))))
    train_times = set(int(t) for t in unique_times[:train_n_times])
    train_mask = np.asarray([int(t) in train_times for t in medium["time_index"]], dtype=bool)
    test_mask = ~train_mask
    best = fit_variance_scale(medium["y"], medium["mean"], medium["var"], train_mask)
    scaled_var = best["alpha"] * medium["var"] + best["tau2"]
    oracle = fit_variance_scale(medium["y"], medium["mean"], medium["var"], np.ones_like(train_mask, dtype=bool))
    oracle_var = oracle["alpha"] * medium["var"] + oracle["tau2"]
    scaling_rows = [
        {"variant": "original_late_test", "alpha": 1.0, "tau2": 0.0, "train_nll": "", **metrics(medium["y"][test_mask], medium["mean"][test_mask], medium["var"][test_mask])},
        {"variant": "prefix_scaled_late_test", **best, **metrics(medium["y"][test_mask], medium["mean"][test_mask], scaled_var[test_mask])},
        {"variant": "oracle_scaled_all", **oracle, **metrics(medium["y"], medium["mean"], oracle_var)},
    ]
    write_csv(outdir / "safe_lag_variance_scaling.csv", scaling_rows)
    scaling_fig = plot_scaling(outdir, scaling_rows)

    rng = np.random.default_rng(args.seed)
    component = medium["mean"] - base["mean"]
    lag_rows: list[dict[str, Any]] = []
    for shrink in [1.0, 0.9, 0.75, 0.5, 0.25]:
        mean = base["mean"] + shrink * component
        lag_rows.append({"variant": f"shrink_{shrink:.2f}", "shrink": shrink, "noise_sd": 0.0, **metrics(medium["y"], mean, medium["var"])})
    for noise_sd in [0.01, 0.025, 0.05, 0.10]:
        mean = medium["mean"] + rng.normal(0.0, noise_sd, size=medium["mean"].shape)
        lag_rows.append({"variant": f"noise_{noise_sd:.3f}", "shrink": 1.0, "noise_sd": noise_sd, **metrics(medium["y"], mean, medium["var"] + noise_sd**2)})
    write_csv(outdir / "safe_lag_lag_shrink_noise.csv", lag_rows)
    lag_fig = plot_lag_stress(outdir, lag_rows)

    summary = {
        "scope": "safe-lag held-out diagnostics; medium target lags at test locations are recursively predicted, not read from y_true",
        "base_predictions": str(base_path),
        "medium_predictions": str(medium_path),
        "decomposition_csv": str(outdir / "safe_lag_nll_decomposition.csv"),
        "variance_scaling_csv": str(outdir / "safe_lag_variance_scaling.csv"),
        "lag_stress_csv": str(outdir / "safe_lag_lag_shrink_noise.csv"),
        "figures": [str(decomp_fig), str(scaling_fig), str(lag_fig)],
        "rows": {
            "decomposition": decomposition_rows,
            "variance_scaling": scaling_rows,
            "lag_stress": lag_rows,
        },
    }
    (outdir / "safe_lag_calibration_diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
