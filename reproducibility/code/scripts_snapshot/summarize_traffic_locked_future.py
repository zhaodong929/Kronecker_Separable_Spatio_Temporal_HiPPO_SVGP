#!/usr/bin/env python3
"""Summarise audited PEMS-BAY Protocol-F runs and plot horizon performance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HORIZONS = (3, 6, 12)
MINUTES = {3: 15, 6: 30, 12: 60}
LABELS = {
    "persistence": "Delayed-target persistence",
    "kron_stgp": "Kron-STGP",
    "kronhippo_stgp": "KronHiPPO-STGP",
    "graph_wavenet": "Graph WaveNet",
    "dcrnn": "DCRNN",
}
METRICS = ("rmse_speed", "mae_speed", "crps", "gaussian_nlpd", "ece", "coverage90")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--methods", nargs="+", default=list(LABELS))
    args = parser.parse_args()

    audit = json.loads((args.root / "PROTOCOL_F_ARCHIVE_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise RuntimeError("Protocol-F archive audit has not passed")

    per_seed: list[dict[str, object]] = []
    for method in args.methods:
        for seed in args.seeds:
            path = args.root / "pems_bay" / "forecast" / method / f"seed{seed}" / "result.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            archive = np.load(path.parent / "predictions.npz")
            scale = float(payload["target_standardisation"]["scale"])
            for horizon in HORIZONS:
                metrics = payload["result"]["final"][str(horizon)]
                y = np.asarray(archive[f"y_h{horizon}"], dtype=float)
                mean = np.asarray(archive[f"mean_h{horizon}"], dtype=float)
                per_seed.append(
                    {
                        "method": method,
                        "seed": seed,
                        "horizon_steps": horizon,
                        "horizon_minutes": MINUTES[horizon],
                        "rmse_speed": float(metrics["rmse_speed"]),
                        "mae_speed": float(scale * np.mean(np.abs(mean - y))),
                        **{
                            metric: (float(metrics[metric]) if metric in metrics else None)
                            for metric in METRICS[2:]
                        },
                    }
                )
    write_csv(args.root / "future_per_seed_results.csv", per_seed)

    aggregate: list[dict[str, object]] = []
    for method in args.methods:
        for horizon in HORIZONS:
            selected = [row for row in per_seed if row["method"] == method and row["horizon_steps"] == horizon]
            row: dict[str, object] = {
                "method": method,
                "horizon_steps": horizon,
                "horizon_minutes": MINUTES[horizon],
                "num_seeds": len(selected),
            }
            for metric in METRICS:
                values = np.asarray([item[metric] for item in selected if item[metric] is not None], dtype=float)
                row[f"{metric}_mean"] = float(values.mean()) if values.size else None
                row[f"{metric}_sd"] = float(values.std(ddof=1)) if values.size > 1 else None
            aggregate.append(row)
    write_csv(args.root / "future_aggregate_results.csv", aggregate)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"persistence": "#666666", "kron_stgp": "#D55E00", "kronhippo_stgp": "#0072B2", "graph_wavenet": "#009E73", "dcrnn": "#CC79A7"}
    markers = {"persistence": "s", "kron_stgp": "^", "kronhippo_stgp": "o", "graph_wavenet": "D", "dcrnn": "P"}
    fig, axis = plt.subplots(figsize=(4.8, 3.15))
    for method in args.methods:
        selected = [row for row in aggregate if row["method"] == method]
        axis.errorbar(
            [row["horizon_minutes"] for row in selected],
            [row["rmse_speed_mean"] for row in selected],
            yerr=[row["rmse_speed_sd"] for row in selected],
            color=colors[method],
            marker=markers[method],
            linewidth=1.2,
            capsize=2.5,
            label=LABELS[method],
        )
    axis.set_xticks([15, 30, 60])
    axis.set_xlabel("Forecast horizon (minutes)")
    axis.set_ylabel("RMSE (mph)")
    axis.set_title("Strict Protocol-F forecasting", loc="left")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.root / "fig_pems_future_horizon_rmse.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(args.root / "fig_pems_future_horizon_rmse.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
