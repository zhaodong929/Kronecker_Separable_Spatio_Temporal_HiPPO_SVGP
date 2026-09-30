#!/usr/bin/env python3
"""Recompute and summarize the new-split COVID Route B confirmation run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from scripts.run_epidemiology_pilot import predictive_metrics
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_epidemiology_pilot import predictive_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        run = args.root / f"seed{seed}" / "online"
        protocol_npz = args.protocol_root / f"seed{seed}" / "protocol.npz"
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        with np.load(protocol_npz) as protocol, np.load(run / "predictions.npz") as predictions:
            y_true = np.asarray(predictions["y_true"], dtype=np.float64)
            pred_mean = np.asarray(predictions["pred_mean"], dtype=np.float64)
            pred_var = np.asarray(predictions["pred_var"], dtype=np.float64)
            expected = np.asarray(protocol["stream_y"], dtype=np.float64)[:, protocol["test_indices"]]
            if not np.array_equal(predictions["test_indices"], protocol["test_indices"]):
                raise ValueError(f"seed {seed}: test indices differ")
            if not np.allclose(y_true, expected, rtol=0.0, atol=1e-12):
                raise ValueError(f"seed {seed}: labels differ")
        if not np.isfinite(pred_mean).all() or not np.isfinite(pred_var).all() or (pred_var <= 0.0).any():
            raise ValueError(f"seed {seed}: invalid prediction archive")
        metrics = predictive_metrics(y_true, pred_mean, pred_var)
        reported = result["overall_current_block"]
        for metric in ("rmse", "nll", "coverage90"):
            if abs(float(metrics[metric]) - float(reported[metric])) > 1e-10:
                raise ValueError(f"seed {seed}: {metric} mismatch")
        rows.append(
            {
                "seed": seed,
                **{metric: float(metrics[metric]) for metric in ("rmse", "nll", "coverage90", "mean_predictive_std", "mean_interval_width90")},
                "runtime_seconds": float(result["timing"]["process_total_seconds"]),
                "peak_allocated_mib": float(result["resources"]["peak_cuda_allocated_mib"]),
                "peak_reserved_mib": float(result["resources"]["peak_cuda_reserved_mib"]),
                "task1_rows": int(result["task1_posterior_initialization_rows"]),
                "delayed_rows": int(result["delayed_observation_rows"]),
            }
        )

    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "metrics_per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ("rmse", "nll", "coverage90", "mean_predictive_std", "mean_interval_width90")
    aggregate = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        aggregate[f"{metric}_mean"] = float(values.mean())
        aggregate[f"{metric}_sd"] = float(values.std(ddof=1))
    (args.root / "metrics_aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# COVID Route B new-split confirmation",
        "",
        "New spatial split seeds 5-9; no structure was selected from these runs. Each row uses 52 calibration weeks, 39 delayed strict-online weeks, 42 visible, 4 validation and 10 held-out states, float64, Mt=32, Ms=32, RFF=64, fixed Q=2 spectral-mixture family, full-joint-conditional variance, and visible-state Task-1 posterior initialization.",
        "",
        "| Seed | RMSE | NLL | Coverage90 | Runtime (s) | Peak alloc (MiB) | Peak reserved (MiB) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['rmse']:.6f} | {row['nll']:.6f} | {row['coverage90']:.4f} | "
            f"{row['runtime_seconds']:.3f} | {row['peak_allocated_mib']:.3f} | {row['peak_reserved_mib']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Aggregate: RMSE **{aggregate['rmse_mean']:.4f} +/- {aggregate['rmse_sd']:.4f}**, NLL **{aggregate['nll_mean']:.4f} +/- {aggregate['nll_sd']:.4f}**, Coverage90 **{aggregate['coverage90_mean']:.4f} +/- {aggregate['coverage90_sd']:.4f}**.",
            "",
            "This confirms the retained P0 protocol on new spatial splits only. It is still a COVID feasibility pilot because the archive contains 91 weekly observations and 52 state locations.",
            "",
            "The Task-1 posterior initialization absorbs 2184 visible-state rows per run and delayed online updates absorb 380 held-out rows per run. It is not an all-52-state Task-1 posterior because the current single-Kronecker state representation cannot exactly mix train and held-out spatial projection matrices.",
        ]
    )
    (args.root / "confirmation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.root / "artifact_audit.json").write_text(
        json.dumps({"status": "complete", "seeds": args.seeds, "prediction_metrics_recomputed": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
