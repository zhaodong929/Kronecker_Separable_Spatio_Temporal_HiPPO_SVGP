#!/usr/bin/env python3
"""Export the locked PEMS-BAY Protocol N for existing external GP adapters."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stvgp_kronecker.data.traffic import (
    build_road_context_xlag_features,
    load_spatial_split,
    load_traffic_dataset,
)
from stvgp_kronecker.joint_ssgp_kron.synthetic import select_spatial_inducing_indices


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ridge_offset(phi: np.ndarray, targets: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(phi[:, fit], dtype=np.float64).reshape(-1, phi.shape[-1])
    response = np.asarray(targets[:, fit], dtype=np.float64).reshape(-1)
    beta = np.linalg.solve(
        design.T @ design + 1e-3 * np.eye(design.shape[1]),
        design.T @ response,
    )
    return np.einsum("tsp,p->ts", phi, beta), beta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/traffic/raw"))
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task1-steps", type=int, default=2016)
    parser.add_argument("--spatial-inducing", type=int, nargs="+", default=[32])
    parser.add_argument(
        "--road-distance-csv",
        type=Path,
        default=Path("data/traffic/raw/pems_bay/distances_bay_2017.csv"),
    )
    parser.add_argument("--xlag-length", type=int, default=10)
    parser.add_argument("--graph-diffusion", type=float, default=7.448975327393576)
    parser.add_argument("--include-features", action="store_true",
                        help="Include the same causal features for joint-mean Route B runs")
    args = parser.parse_args()

    split = load_spatial_split(args.split_manifest)
    dataset = load_traffic_dataset(
        args.data_root, "pems_bay", task1_steps=args.task1_steps,
        scaler_fit_indices=split.visible_calibration_indices,
    )
    if split.dataset != dataset.name or split.seed not in (1, 2, 3):
        raise ValueError("Only locked PEMS-BAY seeds 1, 2 and 3 may be exported")
    visible = np.asarray(split.visible_indices, dtype=np.int64)
    heldout = np.asarray(split.heldout_indices, dtype=np.int64)
    fit = np.asarray(split.visible_calibration_indices, dtype=np.int64)
    validation = np.asarray(split.visible_validation_indices, dtype=np.int64)
    dataset, mean_metadata = build_road_context_xlag_features(
        dataset,
        context_indices=fit,
        scaler_fit_indices=fit,
        road_distance_csv=args.road_distance_csv,
        lag_count=args.xlag_length,
        graph_diffusion=args.graph_diffusion,
    )
    task1 = np.asarray(dataset.values_standardised[: args.task1_steps], dtype=np.float32)
    stream = np.asarray(dataset.values_standardised[args.task1_steps :], dtype=np.float32)
    offset, beta = ridge_offset(dataset.phi[: args.task1_steps], task1, fit)
    stream_offset = np.einsum(
        "tsp,p->ts", dataset.phi[args.task1_steps :], beta
    ).astype(np.float32)
    calibration_offset = np.asarray(offset, dtype=np.float32)

    payload: dict[str, np.ndarray] = {
        "calibration_y": task1,
        "stream_y": stream,
        "task1_calibration_mean": calibration_offset,
        "task1_stream_mean": stream_offset,
        "calibration_times": np.asarray(dataset.times_hours[: args.task1_steps], dtype=np.float64),
        "stream_times": np.asarray(dataset.times_hours[args.task1_steps :], dtype=np.float64),
        "coordinates": np.asarray(dataset.coordinates_standardised, dtype=np.float64),
        "train_indices": visible,
        "fit_indices": fit,
        "validation_indices": validation,
        "test_indices": heldout,
        "block_start": np.arange(stream.shape[0], dtype=np.int64),
        "block_stop": np.arange(1, stream.shape[0] + 1, dtype=np.int64),
        "task1_mean_beta": np.asarray(beta, dtype=np.float64),
    }
    if args.include_features:
        payload["calibration_phi"] = dataset.phi[:args.task1_steps].astype(np.float32)
        payload["stream_phi"] = dataset.phi[args.task1_steps:].astype(np.float32)
    for count in sorted(set(int(value) for value in args.spatial_inducing)):
        if not 1 <= count <= visible.size:
            raise ValueError("Spatial inducing count must not exceed visible sensor count")
        local = select_spatial_inducing_indices(
            dataset.coordinates_standardised[visible], count, method="farthest"
        )
        payload[f"inducing_coords_ms{count}"] = np.asarray(
            dataset.coordinates_standardised[visible[local]], dtype=np.float64
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "protocol.npz"
    np.savez_compressed(archive_path, **payload)
    metadata = {
        "schema_version": 1,
        "protocol_id": "pems_bay_protocol_n",
        "dataset": "pems_bay",
        "split_seed": int(split.seed),
        "task1_steps": int(args.task1_steps),
        "online_steps": int(stream.shape[0]),
        "visible_sensors": int(visible.size),
        "heldout_sensors": int(heldout.size),
        "visible_calibration_sensors": int(fit.size),
        "visible_validation_sensors": int(validation.size),
        "delayed_target_steps": 1,
        "target_scale": "Task-1 visible-calibration standardised speed",
        "target_standardisation": {
            "mean": float(dataset.target_mean),
            "scale": float(dataset.target_scale),
            "fit_prefix": "Task-1 only",
            "fit_indices": fit.tolist(),
            "fit_locations": "visible_calibration_only",
        },
        "task1_mean": "locked Road-context L10 ridge; fit on visible-calibration sensors only",
        "task1_mean_metadata": mean_metadata,
        "information_order": [
            "absorb y_H,t-1 exactly once",
            "condition on y_V,t",
            "predict y_H,t",
            "reveal y_H,t after prediction",
        ],
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "protocol_npz_sha256": file_sha256(archive_path),
    }
    (output / "protocol.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
