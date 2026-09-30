#!/usr/bin/env python3
"""Write fixed paired-spatial split manifests for PEMS-BAY and METR-LA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stvgp_kronecker.data.traffic import load_traffic_dataset, write_spatial_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/traffic/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("results/traffic/protocols"))
    parser.add_argument("--dataset", choices=["all", "pems_bay", "metr_la"], default="all")
    parser.add_argument("--task1-steps", type=int, default=2016, help="One seven-day, five-minute calibration prefix")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    datasets = ["pems_bay", "metr_la"] if args.dataset == "all" else [args.dataset]
    for dataset_name in datasets:
        dataset = load_traffic_dataset(args.data_root, dataset_name, task1_steps=args.task1_steps)
        output = args.output_root / dataset_name
        paths = write_spatial_splits(dataset, output, seeds=args.seeds)
        summary = {
            "dataset": dataset_name,
            "num_time": dataset.num_time,
            "num_sensors": dataset.num_sensors,
            "task1_steps": dataset.task1_steps,
            "stream_steps": dataset.num_time - dataset.task1_steps,
            "missing_values_forward_filled": dataset.missing_values_forward_filled,
            "timestamp_gap_steps": dataset.timestamp_gap_steps,
            "mean_columns": ["1", "sin_tod", "cos_tod", "sin_dow", "cos_dow", "lat", "lon"],
            "split_manifests": [str(path) for path in paths],
        }
        (output / "protocol_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
