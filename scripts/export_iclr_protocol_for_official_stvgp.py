#!/usr/bin/env python3
"""Export a shared ERA5 protocol NPZ to the official ST-VGP wrapper schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arrays = np.load(args.protocol_npz)
    train = np.asarray(arrays["train_indices"], dtype=int)
    test = np.asarray(arrays["test_indices"], dtype=int)
    coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
    y = np.asarray(arrays["stream_y"], dtype=np.float64)
    xlag_mean = np.asarray(arrays["batch_stream_mean"], dtype=np.float64)
    extra = {
        key: np.asarray(arrays[key], dtype=np.float64)
        for key in arrays.files
        if key.startswith("inducing_coords_ms")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        times=np.asarray(arrays["stream_times"], dtype=np.float64),
        train_coords=coordinates[train],
        test_coords=coordinates[test],
        y_train=y[:, train],
        y_test=y[:, test],
        xlag_mean_train=xlag_mean[:, train],
        xlag_mean_test=xlag_mean[:, test],
        train_indices=train,
        test_indices=test,
        **extra,
    )


if __name__ == "__main__":
    main()
