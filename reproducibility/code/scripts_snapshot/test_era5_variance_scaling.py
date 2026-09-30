#!/usr/bin/env python3
"""Standalone variance-scaling test for ERA5 Route B predictions.

This diagnostic fits var_scaled = alpha * var + tau^2 on a calibration prefix
of saved medium-ERA5 pointwise predictions. It does not modify or rerun the
main ERA5 experiment.
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
    fit_variance_scale,
    metrics,
    plot_scaling,
    read_prediction_csv,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    d = read_prediction_csv(DEFAULT_INPUTS["medium-ERA5"])
    n = d["y"].shape[0]
    train_stop = max(10, int(round(n * float(args.calibration_fraction))))
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:train_stop] = True
    test_mask = ~train_mask

    best = fit_variance_scale(d["y"], d["mean"], d["var"], train_mask)
    original_metrics = metrics(d["y"][test_mask], d["mean"][test_mask], d["var"][test_mask])
    scaled_var = best["alpha"] * d["var"] + best["tau2"]
    scaled_metrics = metrics(d["y"][test_mask], d["mean"][test_mask], scaled_var[test_mask])

    oracle = fit_variance_scale(d["y"], d["mean"], d["var"], np.ones(n, dtype=bool))
    oracle_var = oracle["alpha"] * d["var"] + oracle["tau2"]
    oracle_metrics = metrics(d["y"], d["mean"], oracle_var)

    rows = [
        {"variant": "original_test", "alpha": 1.0, "tau2": 0.0, "train_nll": "", **original_metrics},
        {"variant": "prefix_scaled_test", **best, **scaled_metrics},
        {"variant": "oracle_scaled_all", **oracle, **oracle_metrics},
    ]

    csv_path = args.outdir / "era5_variance_scaling_single_location.csv"
    write_csv(csv_path, rows)
    fig_path = args.outdir / "fig_era5_variance_scaling_single_location.png"
    try:
        fig_path = plot_scaling(args.outdir, rows)
    except Exception as exc:  # pragma: no cover - depends on local plotting stack.
        print(f"Plot skipped: {exc}")

    summary = {
        "scope": "standalone variance-scaling test; main ERA5 experiments unchanged",
        "calibration_prefix_points": int(train_stop),
        "csv": str(csv_path),
        "figure": str(fig_path),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
