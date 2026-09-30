#!/usr/bin/env python3
"""Create a Task-1-only causal pseudo-holdout split for traffic tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stvgp_kronecker.data.traffic import SpatialSplit, load_spatial_split, load_traffic_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pems_bay", "metr_la"], required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/traffic/raw")
    parser.add_argument("--outer-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task1-steps", type=int, default=1512)
    parser.add_argument("--inner-validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if not 0.0 < args.inner_validation_fraction < 1.0:
        raise ValueError("inner-validation-fraction must lie in (0, 1)")
    dataset = load_traffic_dataset(args.data_root, args.dataset, task1_steps=args.task1_steps)
    outer = load_spatial_split(args.outer_split)
    if outer.dataset != dataset.name:
        raise ValueError("Outer split and dataset do not match")
    pseudo_heldout = np.asarray(outer.visible_validation_indices, dtype=int)
    pseudo_visible = np.asarray(outer.visible_calibration_indices, dtype=int)
    count = max(1, min(pseudo_visible.size - 1, int(round(pseudo_visible.size * args.inner_validation_fraction))))
    rng = np.random.default_rng(10_000 + outer.seed)
    inner_validation = np.sort(rng.choice(pseudo_visible, size=count, replace=False))
    inner_calibration = np.setdiff1d(pseudo_visible, inner_validation, assume_unique=True)
    split = SpatialSplit(
        dataset=dataset.name,
        seed=10_000 + outer.seed,
        heldout_indices=tuple(int(value) for value in pseudo_heldout),
        visible_indices=tuple(int(value) for value in pseudo_visible),
        visible_calibration_indices=tuple(int(value) for value in inner_calibration),
        visible_validation_indices=tuple(int(value) for value in inner_validation),
        heldout_sensor_ids=tuple(dataset.sensor_ids[int(value)] for value in pseudo_heldout),
        visible_sensor_ids=tuple(dataset.sensor_ids[int(value)] for value in pseudo_visible),
        sensor_id_sha256=outer.sensor_id_sha256,
        coordinates_sha256=outer.coordinates_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "role": "development_only_causal_pseudo_holdout",
        "outer_split": str(args.outer_split),
        "task1_fit_prefix_steps": args.task1_steps,
        "validation_stream_steps": 2016 - args.task1_steps,
        "pseudo_heldout_source": "outer visible_validation_indices",
        "pseudo_heldout_count": int(pseudo_heldout.size),
        "pseudo_visible_count": int(pseudo_visible.size),
        "inner_calibration_count": int(inner_calibration.size),
        "inner_validation_count": int(inner_validation.size),
    }
    payload = {
        "schema_version": 1,
        "protocol": "paired_spatial_streaming",
        "target": "traffic speed",
        "task1_steps": args.task1_steps,
        "cadence_minutes": 5,
        "development_only": True,
        "split": split.as_dict(),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
