#!/usr/bin/env python3
"""Standalone NLL decomposition test for ERA5 Route B predictions.

This diagnostic only reads saved pointwise predictions and writes a CSV/figure
under the paper-ready diagnostics directory. It does not modify or rerun the
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
    metrics,
    plot_decomposition,
    read_prediction_csv,
    write_csv,
)


def align_by_time(data: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    common = set(next(iter(data.values()))["time_index"].tolist())
    for d in data.values():
        common &= set(d["time_index"].tolist())
    common_idx = np.asarray(sorted(common), dtype=int)
    aligned: dict[str, dict[str, np.ndarray]] = {}
    for mode, d in data.items():
        order = {int(t): i for i, t in enumerate(d["time_index"])}
        take = np.asarray([order[int(t)] for t in common_idx], dtype=int)
        aligned[mode] = {
            k: v[take] if isinstance(v, np.ndarray) and v.shape[0] == d["time_index"].shape[0] else v
            for k, v in d.items()
        }
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    data = {mode: read_prediction_csv(path) for mode, path in DEFAULT_INPUTS.items()}
    aligned = align_by_time(data)

    rows = []
    for mode, d in aligned.items():
        rows.append({"mode": mode, **metrics(d["y"], d["mean"], d["var"])})

    csv_path = args.outdir / "era5_nll_decomposition_single_location.csv"
    write_csv(csv_path, rows)
    fig_path = args.outdir / "fig_era5_nll_decomposition_single_location.png"
    try:
        fig_path = plot_decomposition(args.outdir, rows)
    except Exception as exc:  # pragma: no cover - depends on local plotting stack.
        print(f"Plot skipped: {exc}")

    summary = {
        "scope": "standalone NLL decomposition test; main ERA5 experiments unchanged",
        "csv": str(csv_path),
        "figure": str(fig_path),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
