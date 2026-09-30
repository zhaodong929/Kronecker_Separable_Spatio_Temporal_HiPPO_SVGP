#!/usr/bin/env python3
"""Standalone lag-feature shrink/noise test for ERA5 Route B predictions.

This diagnostic perturbs the saved medium-ERA5 mean trajectory post hoc to ask
whether weakening the lag-assisted component improves NLL. It does not modify
or rerun the main ERA5 experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_era5_calibration_lag_effects import (
    DEFAULT_INPUTS,
    DEFAULT_OUTDIR,
    lag_stress_variants,
    metrics,
    plot_lag_stress,
    read_prediction_csv,
    write_csv,
)


def align_pair(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    common = sorted(set(left["time_index"].tolist()) & set(right["time_index"].tolist()))
    left_order = {int(t): i for i, t in enumerate(left["time_index"])}
    right_order = {int(t): i for i, t in enumerate(right["time_index"])}
    left_take = np.asarray([left_order[int(t)] for t in common], dtype=int)
    right_take = np.asarray([right_order[int(t)] for t in common], dtype=int)
    left_aligned = {
        k: v[left_take] if isinstance(v, np.ndarray) and v.shape[0] == left["time_index"].shape[0] else v
        for k, v in left.items()
    }
    right_aligned = {
        k: v[right_take] if isinstance(v, np.ndarray) and v.shape[0] == right["time_index"].shape[0] else v
        for k, v in right.items()
    }
    return left_aligned, right_aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    base = read_prediction_csv(DEFAULT_INPUTS["base"])
    medium = read_prediction_csv(DEFAULT_INPUTS["medium-ERA5"])
    base, medium = align_pair(base, medium)

    rng = np.random.default_rng(args.seed)
    rows = []
    for item in lag_stress_variants(medium["y"], medium["mean"], base["mean"], medium["var"], rng=rng):
        item_metrics = metrics(medium["y"], np.asarray(item["mean"]), np.asarray(item["var"]))
        rows.append({"variant": item["variant"], "shrink": item["shrink"], "noise_sd": item["noise_sd"], **item_metrics})

    csv_path = args.outdir / "era5_lag_shrink_noise_single_location.csv"
    write_csv(csv_path, rows)
    fig_path = args.outdir / "fig_era5_lag_shrink_noise_single_location.png"
    try:
        fig_path = plot_lag_stress(args.outdir, rows)
    except Exception as exc:  # pragma: no cover - depends on local plotting stack.
        print(f"Plot skipped: {exc}")

    summary = {
        "scope": "standalone lag-feature shrink/noise test; main ERA5 experiments unchanged",
        "csv": str(csv_path),
        "figure": str(fig_path),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
