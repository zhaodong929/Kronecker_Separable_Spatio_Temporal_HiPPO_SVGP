#!/usr/bin/env python3
"""Materialise the shared ICLR ERA5 protocol without duplicating raw arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.iclr_era5_full_benchmark_protocol import (
    LONG_ROOT,
    LONG_STREAM_TASKS,
    SHORT_ROOT,
    SHORT_STREAM_TASKS,
    load_benchmark_data,
    protocol_metadata,
    sha256_file,
)


def export_one(
    *,
    root: Path,
    stream_tasks: tuple[str, ...],
    split_seed: int,
    output: Path,
) -> dict:
    data = load_benchmark_data(
        root=root,
        stream_tasks=stream_tasks,
        split_seed=split_seed,
        inducing_sizes=(30, 64, 128),
    )
    payload = {
        "train_indices": data.train_indices,
        "fit_indices": data.fit_indices,
        "validation_indices": data.validation_indices,
        "test_indices": data.test_indices,
        "task1_ridge_beta": data.task1_ridge_beta,
        "batch_ridge_beta": data.batch_ridge_beta,
        "calibration_y": np.asarray(data.calibration.Y, dtype=np.float32),
        "stream_y": np.asarray(data.stream.Y, dtype=np.float32),
        "task1_calibration_mean": np.asarray(data.task1_ridge_mean, dtype=np.float32),
        "task1_stream_mean": np.asarray(data.task1_stream_mean, dtype=np.float32),
        "batch_stream_mean": np.asarray(data.batch_ridge_mean, dtype=np.float32),
        "coordinates": data.coordinates,
        "calibration_times": data.calibration.times,
        "stream_times": data.stream.times,
        "block_start": np.asarray([block.start for block in data.blocks], dtype=int),
        "block_stop": np.asarray([block.stop for block in data.blocks], dtype=int),
    }
    for size, coordinates in data.spatial_inducing.items():
        payload[f"inducing_coords_ms{size}"] = coordinates
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    metadata = protocol_metadata(data)
    metadata["npz"] = str(output.resolve())
    metadata["npz_sha256"] = sha256_file(output)
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    records = []
    for label, root, tasks in (
        ("task1_2", SHORT_ROOT, SHORT_STREAM_TASKS),
        ("task1_10", LONG_ROOT, LONG_STREAM_TASKS),
    ):
        for seed in args.split_seeds:
            records.append(
                {
                    "dataset": label,
                    **export_one(
                        root=root,
                        stream_tasks=tasks,
                        split_seed=seed,
                        output=args.outdir / label / f"seed{seed}" / "protocol.npz",
                    ),
                }
            )
    manifest = {
        "schema_version": 1,
        "purpose": "shared protocol for all ICLR ERA5 batch and streaming wrappers",
        "records": records,
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
