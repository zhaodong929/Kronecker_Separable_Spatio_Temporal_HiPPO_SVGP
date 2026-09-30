#!/usr/bin/env python3
"""Recompute and compare the COVID OHSVGP feasibility-pilot artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

try:
    from scripts.run_epidemiology_pilot import predictive_metrics
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_epidemiology_pilot import predictive_metrics


METRICS = (
    "rmse",
    "nll",
    "coverage50",
    "coverage80",
    "coverage90",
    "coverage95",
    "mean_predictive_std",
    "mean_interval_width90",
)


def mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohsvgp-root", type=Path, required=True)
    parser.add_argument("--routeb-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        ohsvgp_dir = args.ohsvgp_root / f"seed{seed}"
        routeb_dir = args.routeb_root / f"seed{seed}" / "cumulative_hippo" / "online"
        ohsvgp_predictions = np.load(ohsvgp_dir / "predictions.npz")
        routeb_predictions = np.load(routeb_dir / "predictions.npz")
        if not np.allclose(ohsvgp_predictions["y_true"], routeb_predictions["y_true"]):
            raise ValueError(f"Seed {seed} uses different held-out labels between OHSVGP and Route B")
        variance = ohsvgp_predictions["pred_var"]
        if not np.isfinite(variance).all() or np.any(variance <= 0.0):
            raise ValueError(f"Seed {seed} has invalid OHSVGP predictive variance")
        ohsvgp_metrics = predictive_metrics(
            ohsvgp_predictions["y_true"], ohsvgp_predictions["pred_mean"], variance
        )
        routeb_metrics = json.loads((routeb_dir / "result.json").read_text(encoding="utf-8"))["overall_current_block"]
        result = json.loads((ohsvgp_dir / "result.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": int(seed),
                "best_validation_iteration": result["best_validation_iteration"],
                "process_total_seconds": result["timing"]["process_total_seconds"],
                "variance_min": float(variance.min()),
                "variance_max": float(variance.max()),
                **{f"ohsvgp_{key}": ohsvgp_metrics[key] for key in METRICS},
                **{f"routeb_{key}": routeb_metrics[key] for key in METRICS},
            }
        )

    summary = {
        method: {metric: mean_sd([float(row[f"{method}_{metric}"]) for row in rows]) for metric in METRICS}
        for method in ("ohsvgp", "routeb")
    }
    paired = {
        metric: mean_sd([float(row[f"ohsvgp_{metric}"]) - float(row[f"routeb_{metric}"]) for row in rows])
        for metric in METRICS
    }
    payload = {"seeds": args.seeds, "per_seed": rows, "summary": summary, "paired_ohsvgp_minus_routeb": paired}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics_per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "aggregate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# COVID OHSVGP Baseline",
        "",
        "This is a 91-week, 52-location feasibility pilot, not a final benchmark.",
        "OHSVGP uses the pinned upstream multidimensional state/update with one M=32 HiPPO state and RFF=64.",
        "Route B uses Mt=32, Ms=32, RFF=64, fixed Q=2 temporal spectral mixture, and full-joint-conditional prediction.",
        "",
        "| Method | RMSE | NLL | Coverage90 | Mean predictive std |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, label in (("ohsvgp", "OHSVGP RBF"), ("routeb", "Route B Q=2 SM")):
        lines.append(
            f"| {label} | {summary[method]['rmse']['mean']:.4f} +/- {summary[method]['rmse']['sample_sd']:.4f} | "
            f"{summary[method]['nll']['mean']:.4f} +/- {summary[method]['nll']['sample_sd']:.4f} | "
            f"{summary[method]['coverage90']['mean']:.4f} +/- {summary[method]['coverage90']['sample_sd']:.4f} | "
            f"{summary[method]['mean_predictive_std']['mean']:.4f} +/- {summary[method]['mean_predictive_std']['sample_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Paired OHSVGP minus Route B: RMSE {paired['rmse']['mean']:.4f}, NLL {paired['nll']['mean']:.4f}, Coverage90 {paired['coverage90']['mean']:.4f}.",
            "All OHSVGP prediction archives were recomputed; all variances were finite and strictly positive; held-out labels match Route B for every seed.",
        ]
    )
    (args.output_dir / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
