#!/usr/bin/env python3
"""Export a shared ERA5 protocol NPZ to the official ST-VGP wrapper schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


EXPECTED_SPLIT_SIZES = {
    "fit_indices": 720,
    "validation_indices": 80,
    "train_indices": 800,
    "test_indices": 200,
}


def export_protocol(protocol_npz: Path, output: Path) -> None:
    with np.load(protocol_npz) as arrays:
        indices = {
            key: np.asarray(arrays[key], dtype=int)
            for key in EXPECTED_SPLIT_SIZES
        }
        for key, expected_size in EXPECTED_SPLIT_SIZES.items():
            if indices[key].size != expected_size:
                raise ValueError(
                    f"{key} must contain {expected_size} locations, got {indices[key].size}"
                )
        if set(indices["fit_indices"]) | set(indices["validation_indices"]) != set(
            indices["train_indices"]
        ):
            raise ValueError("fit_indices and validation_indices must partition train_indices")
        if set(indices["train_indices"]) & set(indices["test_indices"]):
            raise ValueError("train_indices and test_indices must be disjoint")

        coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
        y = np.asarray(arrays["stream_y"], dtype=np.float64)
        xlag_mean = np.asarray(arrays["batch_stream_mean"], dtype=np.float64)
        extra = {
            key: np.asarray(arrays[key], dtype=np.float64)
            for key in arrays.files
            if key.startswith("inducing_coords_ms")
        }
        for size in (64, 128):
            key = f"inducing_coords_ms{size}"
            if key not in extra:
                raise KeyError(f"{protocol_npz} does not contain {key}")

        fit = indices["fit_indices"]
        validation = indices["validation_indices"]
        train = indices["train_indices"]
        test = indices["test_indices"]
        payload = {
            "times": np.asarray(arrays["stream_times"], dtype=np.float64),
            "fit_coords": coordinates[fit],
            "validation_coords": coordinates[validation],
            "train_coords": coordinates[train],
            "test_coords": coordinates[test],
            "y_fit": y[:, fit],
            "y_validation": y[:, validation],
            "y_train": y[:, train],
            "y_test": y[:, test],
            "xlag_mean_fit": xlag_mean[:, fit],
            "xlag_mean_validation": xlag_mean[:, validation],
            "xlag_mean_train": xlag_mean[:, train],
            "xlag_mean_test": xlag_mean[:, test],
            **indices,
            **extra,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **payload,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_protocol(args.protocol_npz, args.output)


if __name__ == "__main__":
    main()
