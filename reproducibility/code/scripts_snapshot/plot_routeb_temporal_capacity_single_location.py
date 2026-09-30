#!/usr/bin/env python3
"""Plot a protocol-aligned held-out location for Route B temporal capacity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Z90 = 1.6448536269514722


def _location_frame(path: Path, location_index: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["location_index"] == location_index].copy()
    return frame.sort_values("time_index").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    seed_dir = report_dir / "phase_d_joint_xlag_controlled" / f"seed{args.seed}"
    capacity_dir = report_dir / "phase_i_routeb_temporal_capacity" / f"seed{args.seed}"

    mt32_path = seed_dir / "routeb_Mt32_Ms128" / "final_pointwise_predictions.csv"
    mt128_path = capacity_dir / "routeb_Mt128_Ms128" / "final_pointwise_predictions.csv"
    official_path = seed_dir / "official_st_svgp_Ms128" / "predictions.npz"
    data_path = seed_dir / f"era5_xlag_seed{args.seed}.npz"

    mt32_all = pd.read_csv(mt32_path)
    mt128_all = pd.read_csv(mt128_path)
    official = np.load(official_path)
    data = np.load(data_path)

    def rmse_by_location(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            frame.assign(squared_error=lambda x: (x["pred_mean"] - x["y_true"]) ** 2)
            .groupby("location_index", as_index=False)
            .agg(rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))))
        )

    location_scores = (
        mt128_all[["location_index", "lat", "lon"]]
        .drop_duplicates("location_index")
        .merge(rmse_by_location(mt32_all).rename(columns={"rmse": "rmse_mt32"}), on="location_index")
        .merge(rmse_by_location(mt128_all).rename(columns={"rmse": "rmse_mt128"}), on="location_index")
    )
    official_indices = np.asarray(data["test_indices"], dtype=int)
    official_rmse = np.sqrt(
        np.mean((official["pred_mean"] - official["y_true"]) ** 2, axis=0)
    )
    official_scores = pd.DataFrame(
        {"location_index": official_indices, "rmse_official": official_rmse}
    )
    location_scores = location_scores.merge(official_scores, on="location_index")
    global_rmse_mt32 = float(
        np.sqrt(np.mean((mt32_all["pred_mean"] - mt32_all["y_true"]) ** 2))
    )
    global_rmse_mt128 = float(
        np.sqrt(np.mean((mt128_all["pred_mean"] - mt128_all["y_true"]) ** 2))
    )
    global_rmse_official = float(
        np.sqrt(np.mean((official["pred_mean"] - official["y_true"]) ** 2))
    )
    global_rmses = np.asarray(
        [global_rmse_mt32, global_rmse_mt128, global_rmse_official]
    )
    local_rmses = location_scores[
        ["rmse_mt32", "rmse_mt128", "rmse_official"]
    ].to_numpy()
    location_scores["representativeness_score"] = np.sqrt(
        np.mean(((local_rmses - global_rmses) / global_rmses) ** 2, axis=1)
    )
    selected = location_scores.sort_values(
        ["representativeness_score", "location_index"]
    ).iloc[0]
    location_index = int(selected["location_index"])

    mt32 = _location_frame(mt32_path, location_index)
    mt128 = _location_frame(mt128_path, location_index)
    matches = np.flatnonzero(data["test_indices"] == location_index)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one official column for location {location_index}, found {len(matches)}"
        )
    official_col = int(matches[0])

    truth = official["y_true"][:, official_col]
    if not np.allclose(truth, mt128["y_true"].to_numpy(), atol=1e-8):
        raise RuntimeError("Official and Route B truth arrays are not aligned")
    if not np.array_equal(
        mt32["time_index"].to_numpy(), mt128["time_index"].to_numpy()
    ):
        raise RuntimeError("Route B time indices are not aligned")

    time_index = mt128["time_index"].to_numpy()
    mean32 = mt32["pred_mean"].to_numpy()
    std32 = np.sqrt(np.maximum(mt32["pred_var"].to_numpy(), 0.0))
    mean128 = mt128["pred_mean"].to_numpy()
    std128 = np.sqrt(np.maximum(mt128["pred_var"].to_numpy(), 0.0))
    mean_official = official["pred_mean"][:, official_col]
    std_official = np.sqrt(
        np.maximum(official["pred_var"][:, official_col], 0.0)
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rmse32 = float(np.sqrt(np.mean((mean32 - truth) ** 2)))
    rmse128 = float(np.sqrt(np.mean((mean128 - truth) ** 2)))
    rmse_official = float(np.sqrt(np.mean((mean_official - truth) ** 2)))
    panels = [
        (r"Route B $M_t=32$", mean32, std32, "#7A5195", rmse32),
        (r"Route B $M_t=128$", mean128, std128, "#0072B2", rmse128),
        ("Official ST-SVGP", mean_official, std_official, "#D55E00", rmse_official),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7.35, 6.25), sharex=True, sharey=True)
    truth_color = "#111111"
    for panel_index, (title, mean, std, color, rmse) in enumerate(panels):
        ax = axes[panel_index]
        ax.fill_between(
            time_index,
            mean - Z90 * std,
            mean + Z90 * std,
            color=color,
            alpha=0.16,
            linewidth=0,
            label="90% interval",
            zorder=1,
        )
        ax.plot(
            time_index,
            truth,
            color=truth_color,
            linewidth=1.4,
            label="Ground truth",
            zorder=4,
        )
        ax.plot(
            time_index,
            mean,
            color=color,
            linewidth=1.55,
            label="Prediction mean",
            zorder=3,
        )
        ax.set_title(
            f"{chr(ord('a') + panel_index)}   {title} (RMSE = {rmse:.4f})",
            color=color,
            loc="left",
            pad=5,
        )
        ax.set_xlim(time_index.min(), time_index.max())
        ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.985,
            0.08,
            "Spatially held out at all times",
            transform=ax.transAxes,
            ha="right",
            fontsize=7.4,
            color="#777777",
        )
        ax.legend(
            loc="upper right",
            ncol=3,
            frameon=False,
            handlelength=2.2,
            columnspacing=1.0,
            borderaxespad=0.2,
        )
        ax.set_ylabel("Scaled target")
    axes[-1].set_xlabel("Time index")
    fig.suptitle(
        "Representative average-performance spatial holdout across all methods "
        f"(seed {args.seed}, location {location_index}; "
        f"{selected['lat']:.2f} N, {abs(selected['lon']):.2f} W)",
        y=0.995,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.93, bottom=0.075, hspace=0.34)

    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_stem = figures_dir / "figure1b_routeb_mt128_single_location_vertical"
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "selection_rule": (
            "seed-0 held-out location minimizing the root-mean-square relative "
            "deviation of all three per-location RMSEs from their method-specific "
            "full-test RMSEs"
        ),
        "seed": args.seed,
        "num_heldout_locations": int(len(location_scores)),
        "location_index": location_index,
        "lat": float(selected["lat"]),
        "lon": float(selected["lon"]),
        "global_rmse_mt32": global_rmse_mt32,
        "global_rmse_mt128": global_rmse_mt128,
        "global_rmse_official": global_rmse_official,
        "selected_representativeness_score": float(
            selected["representativeness_score"]
        ),
        "selected_rmse_mt32": float(selected["rmse_mt32"]),
        "selected_rmse_mt128": float(selected["rmse_mt128"]),
        "selected_rmse_official": float(selected["rmse_official"]),
        "routeb_mt128_location_rmse": rmse128,
        "routeb_mt32_location_rmse": rmse32,
        "official_location_rmse": rmse_official,
    }
    output_stem.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
