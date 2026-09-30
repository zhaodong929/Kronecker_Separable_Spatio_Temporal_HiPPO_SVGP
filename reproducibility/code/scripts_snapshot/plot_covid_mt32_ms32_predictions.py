#!/usr/bin/env python3
"""Plot representative strict-online COVID predictions for the Mt32/Ms32 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
Z90 = 1.6448536269514722


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "results" / "diagnostics" / "covid_optimization" / "mt32_ms32_rff64_1000",
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=ROOT / "data" / "epidemiology" / "protocol" / "covid",
    )
    args = parser.parse_args()

    prediction_path = (
        args.result_root / f"seed{args.seed}" / "cumulative_hippo" / "online" / "predictions.npz"
    )
    protocol_path = args.protocol_root / f"seed{args.seed}" / "protocol.json"
    with np.load(prediction_path) as data:
        y_true = np.asarray(data["y_true"], dtype=np.float64)
        pred_mean = np.asarray(data["pred_mean"], dtype=np.float64)
        pred_var = np.asarray(data["pred_var"], dtype=np.float64)
        test_indices = np.asarray(data["test_indices"], dtype=int)
    if np.any(pred_var <= 0.0) or not np.isfinite(pred_var).all():
        raise FloatingPointError("Expected finite, strictly positive predictive variances")
    metadata = json.loads(protocol_path.read_text(encoding="utf-8"))
    dates = np.asarray(metadata["raw_dates"][-y_true.shape[0] :], dtype="datetime64[D]")
    names = [metadata["location_names"][index] for index in test_indices]

    per_location_rmse = np.sqrt(np.mean((y_true - pred_mean) ** 2, axis=0))
    order = np.argsort(per_location_rmse)
    chosen = (order[0], order[len(order) // 2], order[-1])
    labels = ("Lowest RMSE", "Median RMSE", "Highest RMSE")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "lines.linewidth": 1.8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.5), sharex=True, constrained_layout=True)
    for axis, index, label in zip(axes, chosen, labels):
        std = np.sqrt(pred_var[:, index])
        lower = pred_mean[:, index] - Z90 * std
        upper = pred_mean[:, index] + Z90 * std
        coverage = float(np.mean(np.abs(y_true[:, index] - pred_mean[:, index]) <= Z90 * std))
        axis.fill_between(dates, lower, upper, color="#56B4E9", alpha=0.28, linewidth=0, label="Nominal 90% interval")
        axis.plot(dates, pred_mean[:, index], color="#0072B2", label="Route B HiPPO mean")
        axis.plot(dates, y_true[:, index], color="#202020", linewidth=1.35, label="Observed target")
        axis.set_title(
            f"{label}: {names[index]}  |  RMSE {per_location_rmse[index]:.3f}, Coverage90 {coverage:.2f}",
            loc="left",
        )

    axes[-1].set_xlabel("Weekly observation date")
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=6))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.supylabel("Standardized log admission rate", x=0.01)
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.suptitle("COVID strict-online prediction: causal Route B cumulative HiPPO (Mt=32, Ms=32)", y=1.09, fontsize=12)

    output_dir = args.result_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"covid_mt32_ms32_seed{args.seed}_representative_predictions"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(
        json.dumps(
            {
                "png": str(stem.with_suffix(".png").resolve()),
                "pdf": str(stem.with_suffix(".pdf").resolve()),
                "selected_locations": [names[index] for index in chosen],
                "selected_rmse": [float(per_location_rmse[index]) for index in chosen],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
