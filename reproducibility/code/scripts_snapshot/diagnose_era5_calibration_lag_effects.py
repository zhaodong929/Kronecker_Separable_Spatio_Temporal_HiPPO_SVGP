#!/usr/bin/env python3
"""Standalone ERA5 calibration diagnostics for the paper-ready report.

This script intentionally does not modify the main ERA5 runner. It reads the
existing single-location pointwise prediction files and produces post-hoc
diagnostics for:

1. NLL decomposition into mean-error and log-variance terms.
2. Variance scaling baselines fitted on a calibration prefix.
3. Lag-feature shrink/noise stress tests approximated by perturbing the strong
   lag-assisted mean trajectory while keeping the reported Route B variance.

The third item is a diagnostic, not a replacement for a full Route B rerun:
it asks whether small degradation/noise in the lag-assisted mean would fix the
NLL issue without materially damaging RMSE. This keeps the main experiment
untouched and makes the test fast and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_READY = ROOT / "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready"
DEFAULT_OUTDIR = DEFAULT_PAPER_READY / "calibration_lag_diagnostics"
DEFAULT_INPUTS = {
    "base": DEFAULT_PAPER_READY / "phi_mode_revision/per_location_base/era5_routeb_per_location_predictions.csv",
    "medium-ERA5": DEFAULT_PAPER_READY / "phi_mode_revision/per_location_medium_era5/era5_routeb_per_location_predictions.csv",
    "rich-ERA5": DEFAULT_PAPER_READY / "phi_mode_revision/per_location_rich_era5/era5_routeb_per_location_predictions.csv",
}
Z90 = 1.6448536269514722


def require_pyplot():
    import matplotlib.pyplot as plt

    if not hasattr(plt, "subplots"):
        raise RuntimeError("matplotlib.pyplot is available but does not expose subplots()")
    return plt


def read_prediction_csv(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    rows.sort(key=lambda r: int(float(r["time_index"])))
    return {
        "time_index": np.asarray([int(float(r["time_index"])) for r in rows], dtype=int),
        "actual_time": np.asarray([float(r["actual_time"]) for r in rows], dtype=float),
        "y": np.asarray([float(r["y_true"]) for r in rows], dtype=float),
        "mean": np.asarray([float(r["pred_mean"]) for r in rows], dtype=float),
        "var": np.maximum(np.asarray([float(r["pred_var_y"]) for r in rows], dtype=float), 1e-10),
        "location_index": np.asarray([int(float(r["location_index"])) for r in rows], dtype=int),
    }


def nll_parts(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> dict[str, np.ndarray]:
    var = np.maximum(var, 1e-10)
    err2 = (y - mean) ** 2
    return {
        "nll": 0.5 * (np.log(2.0 * np.pi * var) + err2 / var),
        "logvar_term": 0.5 * np.log(2.0 * np.pi * var),
        "quad_term": 0.5 * err2 / var,
        "rmse_point": np.sqrt(err2),
        "err2": err2,
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


def fit_variance_scale(y: np.ndarray, mean: np.ndarray, var: np.ndarray, train_mask: np.ndarray) -> dict[str, float]:
    """Grid-search alpha and tau2 for var' = alpha * var + tau2."""
    alpha_grid = np.concatenate(
        [
            np.linspace(0.05, 1.0, 20),
            np.linspace(1.1, 5.0, 40),
            np.linspace(5.5, 20.0, 30),
        ]
    )
    tau_grid = np.concatenate([[0.0], np.logspace(-8, 0, 33)])
    best = {"alpha": 1.0, "tau2": 0.0, "train_nll": float("inf")}
    yt, mt, vt = y[train_mask], mean[train_mask], var[train_mask]
    for alpha in alpha_grid:
        for tau2 in tau_grid:
            scaled = alpha * vt + tau2
            nll = metrics(yt, mt, scaled)["nll"]
            if nll < best["train_nll"]:
                best = {"alpha": float(alpha), "tau2": float(tau2), "train_nll": float(nll)}
    return best


def lag_stress_variants(
    y: np.ndarray,
    medium_mean: np.ndarray,
    base_mean: np.ndarray,
    var: np.ndarray,
    *,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Approximate lag-feature shrink/noise tests without changing the main model.

    The lag-assisted component is represented by medium_mean - base_mean. Shrink
    reduces this component; noise injects small state noise into the mean. This
    probes whether NLL can improve by weakening the lag-induced mean confidence.
    """

    out: list[dict[str, object]] = []
    component = medium_mean - base_mean
    for shrink in [1.0, 0.9, 0.75, 0.5, 0.25]:
        m = base_mean + shrink * component
        out.append({"variant": f"shrink_{shrink:.2f}", "mean": m, "var": var, "shrink": shrink, "noise_sd": 0.0})
    # Deterministic one-shot noise stress, averaged over repeated noise draws.
    for noise_sd in [0.01, 0.025, 0.05, 0.10]:
        means = []
        for _ in range(32):
            means.append(medium_mean + rng.normal(0.0, noise_sd, size=medium_mean.shape))
        mean_stack = np.stack(means)
        # Marginalize the injected covariate noise approximately by adding its
        # variance to the predictive variance while retaining the original mean.
        out.append(
            {
                "variant": f"noise_{noise_sd:.3f}",
                "mean": np.mean(mean_stack, axis=0),
                "var": var + noise_sd**2,
                "shrink": 1.0,
                "noise_sd": noise_sd,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_decomposition(outdir: Path, rows: list[dict[str, object]]) -> Path:
    plt = require_pyplot()
    modes = [r["mode"] for r in rows]
    quad = np.asarray([float(r["quad_term"]) for r in rows])
    logv = np.asarray([float(r["logvar_term"]) for r in rows])
    rmse = np.asarray([float(r["rmse"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    x = np.arange(len(modes))
    axes[0].bar(x, logv, label="0.5 log(2pi var)", color="#6A8CAF")
    axes[0].bar(x, quad, bottom=logv, label="0.5 error^2/var", color="#F28E2B")
    axes[0].set_xticks(x, modes, rotation=25, ha="right")
    axes[0].set_title("Pointwise NLL decomposition")
    axes[0].set_ylabel("mean NLL contribution")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, rmse, color="#59A14F")
    axes[1].set_xticks(x, modes, rotation=25, ha="right")
    axes[1].set_title("Mean trajectory error")
    axes[1].set_ylabel("RMSE")
    fig.tight_layout()
    path = outdir / "fig_era5_nll_decomposition_single_location.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_scaling(outdir: Path, rows: list[dict[str, object]]) -> Path:
    plt = require_pyplot()
    labels = [r["variant"] for r in rows]
    nll = np.asarray([float(r["nll"]) for r in rows])
    width = np.asarray([float(r["avg_width90"]) for r in rows])
    cov = np.asarray([float(r["coverage90"]) for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    x = np.arange(len(labels))
    axes[0].bar(x, nll, color="#F28E2B")
    axes[0].set_title("NLL")
    axes[1].bar(x, width, color="#4E79A7")
    axes[1].set_title("Avg 90% width")
    axes[2].bar(x, cov, color="#59A14F")
    axes[2].axhline(0.9, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_title("Coverage90")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
    fig.suptitle("Variance scaling baseline on medium-ERA5, location 99", fontsize=11)
    fig.tight_layout()
    path = outdir / "fig_era5_variance_scaling_single_location.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_lag_stress(outdir: Path, rows: list[dict[str, object]]) -> Path:
    plt = require_pyplot()
    labels = [r["variant"] for r in rows]
    nll = np.asarray([float(r["nll"]) for r in rows])
    rmse = np.asarray([float(r["rmse"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.1))
    x = np.arange(len(labels))
    axes[0].bar(x, rmse, color="#59A14F")
    axes[0].set_title("RMSE")
    axes[1].bar(x, nll, color="#F28E2B")
    axes[1].set_title("NLL")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
    fig.suptitle("Lag-feature shrink/noise stress test, location 99", fontsize=11)
    fig.tight_layout()
    path = outdir / "fig_era5_lag_shrink_noise_single_location.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    data = {mode: read_prediction_csv(path) for mode, path in DEFAULT_INPUTS.items()}

    # Align by time index. The existing per-location files are all location 99.
    common = set(data["medium-ERA5"]["time_index"].tolist())
    for d in data.values():
        common &= set(d["time_index"].tolist())
    common_idx = np.asarray(sorted(common), dtype=int)
    aligned = {}
    for mode, d in data.items():
        order = {int(t): i for i, t in enumerate(d["time_index"])}
        take = np.asarray([order[int(t)] for t in common_idx], dtype=int)
        aligned[mode] = {k: v[take] if isinstance(v, np.ndarray) and v.shape[0] == d["time_index"].shape[0] else v for k, v in d.items()}

    decomposition_rows = []
    for mode, d in aligned.items():
        m = metrics(d["y"], d["mean"], d["var"])
        decomposition_rows.append({"mode": mode, **m})
    write_csv(outdir / "era5_nll_decomposition_single_location.csv", decomposition_rows)
    decomp_fig = plot_decomposition(outdir, decomposition_rows)

    d = aligned["medium-ERA5"]
    n = d["y"].shape[0]
    train_stop = max(10, int(round(n * float(args.calibration_fraction))))
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:train_stop] = True
    test_mask = ~train_mask
    best = fit_variance_scale(d["y"], d["mean"], d["var"], train_mask)
    base_metrics = metrics(d["y"][test_mask], d["mean"][test_mask], d["var"][test_mask])
    scaled_var = best["alpha"] * d["var"] + best["tau2"]
    scaled_metrics = metrics(d["y"][test_mask], d["mean"][test_mask], scaled_var[test_mask])
    oracle = fit_variance_scale(d["y"], d["mean"], d["var"], np.ones(n, dtype=bool))
    oracle_var = oracle["alpha"] * d["var"] + oracle["tau2"]
    oracle_metrics = metrics(d["y"], d["mean"], oracle_var)
    scaling_rows = [
        {"variant": "original_test", "alpha": 1.0, "tau2": 0.0, "train_nll": "", **base_metrics},
        {"variant": "prefix_scaled_test", **best, **scaled_metrics},
        {"variant": "oracle_scaled_all", **oracle, **oracle_metrics},
    ]
    write_csv(outdir / "era5_variance_scaling_single_location.csv", scaling_rows)
    scaling_fig = plot_scaling(outdir, scaling_rows)

    rng = np.random.default_rng(0)
    medium = aligned["medium-ERA5"]
    base = aligned["base"]
    lag_rows = []
    for item in lag_stress_variants(medium["y"], medium["mean"], base["mean"], medium["var"], rng=rng):
        mm = metrics(medium["y"], np.asarray(item["mean"]), np.asarray(item["var"]))
        lag_rows.append({"variant": item["variant"], "shrink": item["shrink"], "noise_sd": item["noise_sd"], **mm})
    write_csv(outdir / "era5_lag_shrink_noise_single_location.csv", lag_rows)
    lag_fig = plot_lag_stress(outdir, lag_rows)

    summary = {
        "scope": "single-location post-hoc diagnostic, location 99; main ERA5 experiments unchanged",
        "n_time": int(n),
        "calibration_prefix_points": int(train_stop),
        "figures": {
            "nll_decomposition": str(decomp_fig),
            "variance_scaling": str(scaling_fig),
            "lag_shrink_noise": str(lag_fig),
        },
        "best_variance_scaling": best,
        "oracle_variance_scaling": oracle,
        "key_metrics": {
            "decomposition": decomposition_rows,
            "variance_scaling": scaling_rows,
            "lag_shrink_noise": lag_rows,
        },
    }
    (outdir / "era5_calibration_lag_diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
