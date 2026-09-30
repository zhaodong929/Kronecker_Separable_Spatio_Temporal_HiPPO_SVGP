#!/usr/bin/env python3
"""Summarize early, middle and late strict-online metric drift."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.blocks_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 30:
        raise ValueError("Need at least 30 online blocks for a three-window drift diagnostic")
    windows = {"early": rows[:10], "middle": rows[10:-10], "late": rows[-10:]}
    summary: dict[str, dict[str, float]] = {}
    for metric in ("rmse", "nll", "coverage90"):
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary[metric] = {
            **{name: float(np.mean([float(row[metric]) for row in window])) for name, window in windows.items()},
            "linear_slope_per_block": float(np.polyfit(np.arange(values.size), values, 1)[0]),
        }
    decision = "skip_periodic_theta_no_late_rmse_deterioration" if summary["rmse"]["late"] <= summary["rmse"]["early"] else "periodic_theta_warranted"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "decision": decision}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "rmse": summary["rmse"]}, indent=2))


if __name__ == "__main__":
    main()
