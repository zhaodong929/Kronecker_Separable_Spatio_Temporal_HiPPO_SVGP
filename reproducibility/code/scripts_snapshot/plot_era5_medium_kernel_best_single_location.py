#!/usr/bin/env python3
"""Plot single-location fits for best medium-ERA5 kernel settings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_READY = ROOT / "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready"
FIG_DIR = PAPER_READY / "analytic_hippo_rff_fixed_rerun/figures"
Z90 = 1.6448536269514722


INPUTS = [
    (
        "RBF (best fixed sweep, Mt=32, Ms=256)",
        "#4C78A8",
        PAPER_READY
        / "analytic_hippo_rff_fixed_rerun/medium_kernel_capacity_safe_lag_diagnostic/rbf_medium_safe_lag_Mt32_Ms256_singlepoint/era5_routeb_per_location_predictions.csv",
    ),
    (
        "Matern-3/2 (best fixed sweep, Mt=32, Ms=256)",
        "#F58518",
        PAPER_READY
        / "analytic_hippo_rff_fixed_rerun/medium_kernel_capacity_safe_lag_diagnostic/matern32_medium_safe_lag_Mt32_Ms256_singlepoint/era5_routeb_per_location_predictions.csv",
    ),
    (
        "Trainable spectral mixture (Q=3, Mt=16, Ms=128)",
        "#1B7F4C",
        PAPER_READY
        / "trainable_spectral_mixture_medium/routeb_medium_trainable_spectral_mixture_singlepoint/era5_routeb_per_location_predictions.csv",
    ),
]


def metrics(df: pd.DataFrame) -> dict[str, float]:
    err = df["y_true"].to_numpy() - df["pred_mean"].to_numpy()
    var = np.maximum(df["pred_var_y"].to_numpy(), 1e-10)
    sd = np.sqrt(var)
    nll = 0.5 * (np.log(2.0 * np.pi * var) + err**2 / var)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "nll": float(np.mean(nll)),
        "coverage90": float(np.mean(np.abs(err) <= Z90 * sd)),
        "avg_width90": float(np.mean(2.0 * Z90 * sd)),
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[str, str, pd.DataFrame, dict[str, float]]] = []
    rows: list[dict[str, float | str | int]] = []
    for label, color, path in INPUTS:
        df = pd.read_csv(path).sort_values("time_index")
        m = metrics(df)
        row = {
            "kernel": label,
            "location_index": int(df["location_index"].iloc[0]),
            "latitude": float(df["latitude"].iloc[0]),
            "longitude": float(df["longitude"].iloc[0]),
            "n": int(len(df)),
            **m,
        }
        rows.append(row)
        frames.append((label, color, df, m))

    metrics_path = FIG_DIR / "era5_medium_kernel_best_single_location_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    loc = int(rows[0]["location_index"])
    lat = float(rows[0]["latitude"])
    lon = float(rows[0]["longitude"])
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 7.2), sharex=True, constrained_layout=True)
    for ax, (label, color, df, m) in zip(axes, frames):
        x = df["time_index"].to_numpy()
        y = df["y_true"].to_numpy()
        mean = df["pred_mean"].to_numpy()
        sd = np.sqrt(np.maximum(df["pred_var_y"].to_numpy(), 1e-10))
        ax.plot(x, y, color="black", linewidth=1.15, label="ERA5 target")
        ax.plot(x, mean, color=color, linewidth=1.55, label="prediction mean")
        ax.fill_between(x, mean - Z90 * sd, mean + Z90 * sd, color=color, alpha=0.18, label="90% interval")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("scaled value")
        ax.grid(True, alpha=0.18)
        ax.text(
            0.012,
            0.06,
            f"RMSE={m['rmse']:.3f}; NLL={m['nll']:.3f}; Cov90={m['coverage90']:.3f}; width90={m['avg_width90']:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.85"},
        )
        ax.legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("task-2 time index")
    fig.suptitle(
        f"Medium-ERA5 safe-lag single-location fit at held-out location {loc} (lat={lat:.1f}, lon={lon:.1f})",
        fontsize=12,
    )
    out = FIG_DIR / "era5_medium_kernel_best_single_location_fit.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(out)
    print(metrics_path)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
