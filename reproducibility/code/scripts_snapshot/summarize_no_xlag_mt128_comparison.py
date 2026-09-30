#!/usr/bin/env python3
"""Summarize the paired X-lag removal experiment at Mt=128, Ms=128."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("rmse", "nll", "coverage90")


def official_metrics(path: Path) -> dict[str, float]:
    result = json.loads(path.read_text(encoding="utf-8"))
    return {metric: float(result[metric]) for metric in METRICS}


def routeb_metrics(path: Path) -> dict[str, float]:
    result = json.loads(path.read_text(encoding="utf-8"))["final"]
    return {metric: float(result[metric]) for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    output_dir = base / "phase_j_no_xlag_mt128_controlled"

    conditions = {
        ("Official ST-SVGP", "with_xlag"): (
            "official",
            base / "phase_d_joint_xlag_controlled",
            "official_st_svgp_Ms128/result.json",
        ),
        ("Official ST-SVGP", "no_xlag"): (
            "official",
            base / "phase_h_xlag_temporal_sparse_markov",
            "full_markov_no_xlag_Ms128/result.json",
        ),
        ("Route B Mt=128", "with_xlag"): (
            "routeb",
            base / "phase_i_routeb_temporal_capacity",
            "routeb_Mt128_Ms128/run_metadata.json",
        ),
        ("Route B Mt=128", "no_xlag"): (
            "routeb",
            output_dir,
            "routeb_no_xlag_Mt128_Ms128/run_metadata.json",
        ),
    }

    seed_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    for (method, condition), (kind, directory, suffix) in conditions.items():
        grouped[(method, condition)] = []
        for seed in range(3):
            path = directory / f"seed{seed}" / suffix
            metrics = official_metrics(path) if kind == "official" else routeb_metrics(path)
            grouped[(method, condition)].append(metrics)
            seed_rows.append(
                {"method": method, "condition": condition, "split_seed": seed, **metrics}
            )

    summary_rows: list[dict[str, object]] = []
    for method in ("Route B Mt=128", "Official ST-SVGP"):
        with_xlag = grouped[(method, "with_xlag")]
        no_xlag = grouped[(method, "no_xlag")]
        row: dict[str, object] = {"method": method}
        for metric in METRICS:
            with_values = np.asarray([item[metric] for item in with_xlag])
            no_values = np.asarray([item[metric] for item in no_xlag])
            row[f"with_xlag_{metric}_mean"] = float(with_values.mean())
            row[f"with_xlag_{metric}_sd"] = float(with_values.std(ddof=1))
            row[f"no_xlag_{metric}_mean"] = float(no_values.mean())
            row[f"no_xlag_{metric}_sd"] = float(no_values.std(ddof=1))
            row[f"paired_delta_no_minus_with_{metric}_mean"] = float(
                (no_values - with_values).mean()
            )
        row["rmse_relative_increase_percent"] = 100.0 * float(
            (row["no_xlag_rmse_mean"] - row["with_xlag_rmse_mean"])
            / row["with_xlag_rmse_mean"]
        )
        summary_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_split_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "protocol": {
            "num_times": 186,
            "num_locations": 1000,
            "train_locations": 800,
            "test_locations": 200,
            "split_seeds": [0, 1, 2],
            "mt": 128,
            "ms": 128,
            "target": "original scaled y",
            "kernel": "temporal and separable spatial Matern-3/2",
            "routeb_temporal_representation": "analytic HiPPO-RFF",
            "protocol": "batch/full-history",
        },
        "summary": summary_rows,
        "per_split": seed_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
