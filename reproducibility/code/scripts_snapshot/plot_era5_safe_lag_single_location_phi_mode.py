#!/usr/bin/env python3
"""Plot safe-lag held-out single-location Phi-mode comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_READY = ROOT / "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready"
DEFAULT_OUT = PAPER_READY / "safe_lag_calibration_diagnostics/fig_safe_lag_single_location_phi_mode_comparison.png"
Z90 = 1.6448536269514722

INPUTS = {
    "Base Phi": PAPER_READY / "safe_lag_calibration_diagnostics/base_heldout_split0/era5_routeb_per_location_predictions.csv",
    "medium-ERA5": PAPER_READY / "safe_lag_calibration_diagnostics/medium_era5_heldout_split0/era5_routeb_per_location_predictions.csv",
    "rich-ERA5": PAPER_READY / "safe_lag_calibration_diagnostics/rich_era5_heldout_split0/era5_routeb_per_location_predictions.csv",
}

COLORS = {
    "Base Phi": "#6B7C93",
    "medium-ERA5": "#1F77D4",
    "rich-ERA5": "#1B7F4C",
}


def metrics(df: pd.DataFrame) -> dict[str, float]:
    err = df["y_true"].to_numpy() - df["pred_mean"].to_numpy()
    sd = np.sqrt(np.maximum(df["pred_var_y"].to_numpy(), 1e-10))
    nll = 0.5 * (np.log(2.0 * np.pi * sd**2) + err**2 / (sd**2))
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "nll": float(np.mean(nll)),
        "width90": float(np.mean(2.0 * Z90 * sd)),
    }


def choose_location(frames: dict[str, pd.DataFrame]) -> int:
    medium = frames["medium-ERA5"]
    by_loc = []
    for loc, group in medium.groupby("location_index"):
        m = metrics(group.sort_values("time_index"))
        by_loc.append((int(loc), m["rmse"]))
    by_loc.sort(key=lambda x: x[1])
    return by_loc[len(by_loc) // 2][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--location-index", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    frames = {name: pd.read_csv(path) for name, path in INPUTS.items()}
    common_locs = set(frames["Base Phi"]["location_index"].unique())
    for df in frames.values():
        common_locs &= set(df["location_index"].unique())
    if not common_locs:
        raise RuntimeError("No common held-out locations across safe-lag prediction files")

    loc = int(args.location_index) if args.location_index is not None else choose_location(frames)
    if loc not in common_locs:
        raise ValueError(f"location {loc} is not in the common held-out test set")

    import matplotlib.pyplot as plt

    loc_frames = {name: df[df["location_index"] == loc].sort_values("time_index") for name, df in frames.items()}
    first = next(iter(loc_frames.values())).iloc[0]
    lat = float(first["latitude"])
    lon = float(first["longitude"])

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 6.4), sharex=True, constrained_layout=True)
    for ax, (name, df) in zip(axes, loc_frames.items()):
        color = COLORS[name]
        x = df["actual_time"].to_numpy()
        y = df["y_true"].to_numpy()
        mean = df["pred_mean"].to_numpy()
        sd = np.sqrt(np.maximum(df["pred_var_y"].to_numpy(), 1e-10))
        m = metrics(df)
        ax.plot(x, y, color="black", linewidth=1.15, label="ERA5 target")
        ax.plot(x, mean, color=color, linewidth=1.65, label="prediction mean")
        ax.fill_between(x, mean - Z90 * sd, mean + Z90 * sd, color=color, alpha=0.18, label="90% interval")
        ax.set_title(f"{name}, safe-lag held-out, Mt=8, Ms=64", fontsize=10)
        ax.set_ylabel("scaled value")
        ax.grid(True, alpha=0.18)
        ax.legend(loc="upper right", fontsize=8)
        ax.text(
            0.015,
            0.06,
            f"n={len(df)} points | RMSE={m['rmse']:.3f} | NLL={m['nll']:.3f} | avg 90% width={m['width90']:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )
    axes[-1].set_xlabel("time")
    fig.suptitle(
        f"Safe-lag held-out Phi-mode diagnostic at ERA5 test location {loc} "
        f"(lat={lat:.1f}, lon={lon:.1f})",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
