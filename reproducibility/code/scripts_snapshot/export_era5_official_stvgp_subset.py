#!/usr/bin/env python3
"""Export a deterministic ERA5 subset for the legacy official ST-VGP runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hipposvgp_era5_routeb import (
    augment_dataset_phi,
    fixed_spatial_train_test_split,
    normalise_time_dataset,
    normalise_time_dataset_with_scale,
    selected_locations_from_dataset,
)
from stvgp_kronecker.data.hipposvgp_era5 import load_hipposvgp_era5
from stvgp_kronecker.joint_ssgp_kron.synthetic import select_spatial_inducing_indices


def spread_subset(indices: np.ndarray, size: int) -> np.ndarray:
    size = min(int(size), int(indices.size))
    if size == indices.size:
        return indices.copy()
    positions = np.linspace(0, indices.size - 1, size).round().astype(int)
    return indices[positions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--task", default="task_2")
    parser.add_argument("--calibration-task", default="task_1")
    parser.add_argument("--variable-index", type=int, default=0)
    parser.add_argument("--split", default="all")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--num-times", type=int, default=20)
    parser.add_argument("--num-train-space", type=int, default=12)
    parser.add_argument("--num-test-space", type=int, default=4)
    parser.add_argument(
        "--phi-mode",
        choices=["direct_y", "medium_era5_xlag"],
        default="direct_y",
    )
    parser.add_argument("--xlag-length", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--spatial-inducing-sizes", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    calibration = load_hipposvgp_era5(
        root=args.root,
        tasks=(args.calibration_task,),
        variable_index=args.variable_index,
        split=args.split,
    )
    selected_locations = selected_locations_from_dataset(calibration)
    dataset_raw = load_hipposvgp_era5(
        root=args.root,
        tasks=(args.task,),
        variable_index=args.variable_index,
        split=args.split,
        selected_locations=selected_locations,
    )
    _, calibration_scale = normalise_time_dataset(calibration)
    dataset = normalise_time_dataset_with_scale(
        dataset_raw,
        scale=calibration_scale,
        source="calibration_task_span",
    )
    dataset = augment_dataset_phi(
        dataset,
        phi_mode=args.phi_mode,
        xlag_length=args.xlag_length,
    )
    train_idx, test_idx = fixed_spatial_train_test_split(
        dataset.Y.shape[1], test_fraction=args.test_fraction, seed=args.split_seed
    )
    train_idx = spread_subset(train_idx, args.num_train_space)
    test_idx = spread_subset(test_idx, args.num_test_space)
    num_times = min(args.num_times, dataset.Y.shape[0])

    times = np.asarray(dataset.times[:num_times], dtype=np.float64)
    coords = np.asarray(dataset.coords, dtype=np.float64)
    coord_mean = coords.mean(axis=0, keepdims=True)
    coord_scale = np.maximum(coords.std(axis=0, keepdims=True), 1e-12)
    coords = (coords - coord_mean) / coord_scale

    phi_grid = np.asarray(dataset.Phi, dtype=np.float64).reshape(
        dataset.Y.shape[0], dataset.Y.shape[1], -1
    )
    phi_train = phi_grid[:num_times, train_idx].reshape(-1, phi_grid.shape[-1])
    y_train = np.asarray(dataset.Y[:num_times, train_idx], dtype=np.float64)
    if phi_train.shape[1]:
        precision = phi_train.T @ phi_train + args.ridge * np.eye(phi_train.shape[1])
        beta = np.linalg.solve(precision, phi_train.T @ y_train.reshape(-1))
        mean_grid = (phi_grid[:num_times].reshape(-1, phi_grid.shape[-1]) @ beta).reshape(
            num_times, dataset.Y.shape[1]
        )
    else:
        beta = np.empty(0, dtype=np.float64)
        mean_grid = np.zeros((num_times, dataset.Y.shape[1]), dtype=np.float64)

    payload = {
        "times": times,
        "train_coords": coords[train_idx],
        "test_coords": coords[test_idx],
        "y_train": y_train,
        "y_test": np.asarray(dataset.Y[:num_times, test_idx], dtype=np.float64),
        "xlag_mean_train": mean_grid[:, train_idx],
        "xlag_mean_test": mean_grid[:, test_idx],
        "xlag_phi_train": phi_grid[:num_times, train_idx],
        "xlag_phi_test": phi_grid[:num_times, test_idx],
        "xlag_beta": beta,
        "train_indices": train_idx,
        "test_indices": test_idx,
    }
    inducing_global_indices = {}
    for size in sorted(set(args.spatial_inducing_sizes)):
        local_idx = select_spatial_inducing_indices(
            coords[train_idx], min(int(size), train_idx.size), method="kmeans"
        )
        payload[f"inducing_coords_ms{size}"] = coords[train_idx][local_idx]
        payload[f"inducing_global_indices_ms{size}"] = train_idx[local_idx]
        inducing_global_indices[str(size)] = train_idx[local_idx].tolist()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    metadata = {
        "source_root": str(args.root),
        "task": args.task,
        "calibration_task": args.calibration_task,
        "variable_index": args.variable_index,
        "split": args.split,
        "heldout_split_seed": args.split_seed,
        "test_fraction": args.test_fraction,
        "num_times": int(num_times),
        "num_train_space": int(train_idx.size),
        "num_test_space": int(test_idx.size),
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "time_normalization": "P1 calibration_task_span",
        "time_scale": float(calibration_scale),
        "space_normalization": "global coordinate mean/std before subsetting",
        "target_model": (
            "direct y; no X-lag or residual decomposition"
            if args.phi_mode == "direct_y"
            else "original y with shared non-leaking X-lag covariates; ridge beta is initialization/reference only"
        ),
        "phi_mode": args.phi_mode,
        "xlag_length": int(args.xlag_length),
        "num_phi_features": int(phi_grid.shape[-1]),
        "ridge": float(args.ridge),
        "spatial_inducing_selection": "deterministic kmeans over training coordinates only",
        "spatial_inducing_global_indices": inducing_global_indices,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
