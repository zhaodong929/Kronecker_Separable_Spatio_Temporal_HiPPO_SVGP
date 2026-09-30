#!/usr/bin/env python3
"""Shared ERA5 protocol for the ICLR batch and streaming benchmark.

All model wrappers import this module so that data selection, spatial splits,
time/coordinate normalisation, X-lag features, and inducing coordinates cannot
silently drift between implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

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
from stvgp_kronecker.data.hipposvgp_era5 import HippoERA5Dataset, load_hipposvgp_era5
from stvgp_kronecker.joint_ssgp_kron.synthetic import select_spatial_inducing_indices


SHORT_ROOT = Path("data/era5/processed_timeseries_4")
LONG_ROOT = Path("data/era5/processed_timeseries_4_task1_10_extension")
CALIBRATION_TASKS = ("task_1",)
SHORT_STREAM_TASKS = ("task_2",)
LONG_STREAM_TASKS = tuple(f"task_{index}" for index in range(2, 11))


@dataclass(frozen=True)
class BenchmarkData:
    calibration: HippoERA5Dataset
    stream: HippoERA5Dataset
    coordinates: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    fit_indices: np.ndarray
    test_indices: np.ndarray
    spatial_inducing: dict[int, np.ndarray]
    task1_ridge_beta: np.ndarray
    task1_ridge_mean: np.ndarray
    task1_stream_mean: np.ndarray
    batch_ridge_beta: np.ndarray
    batch_ridge_mean: np.ndarray
    blocks: tuple[slice, ...]
    block_tasks: tuple[str, ...]
    split_seed: int
    time_scale: float
    xlag_length: int
    root: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardize_coordinates(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    return (coords - coords.mean(axis=0, keepdims=True)) / np.maximum(
        coords.std(axis=0, keepdims=True), 1e-12
    )


def inner_spatial_split(
    train_indices: np.ndarray,
    *,
    split_seed: int,
    validation_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1729 + int(split_seed))
    permutation = rng.permutation(np.asarray(train_indices, dtype=int))
    count = max(1, int(round(validation_fraction * permutation.size)))
    validation = np.sort(permutation[:count])
    fit = np.sort(permutation[count:])
    return fit, validation


def fit_ridge_mean(
    dataset: HippoERA5Dataset,
    spatial_indices: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    phi = np.asarray(dataset.Phi, dtype=np.float64).reshape(
        dataset.Y.shape[0], dataset.Y.shape[1], -1
    )
    design = phi[:, spatial_indices].reshape(-1, phi.shape[-1])
    target = np.asarray(dataset.Y[:, spatial_indices], dtype=np.float64).reshape(-1)
    precision = design.T @ design + float(ridge) * np.eye(design.shape[1])
    beta = np.linalg.solve(precision, design.T @ target)
    mean = np.einsum("tsd,d->ts", phi, beta)
    return beta, mean


def task_aware_blocks(
    *,
    root: Path,
    stream_tasks: Sequence[str],
    stream_length: int,
    block_size: int,
) -> tuple[tuple[slice, ...], tuple[str, ...]]:
    manifest_path = root / "long_streaming_manifest.json"
    if manifest_path.exists() and tuple(stream_tasks) == LONG_STREAM_TASKS:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest["streaming"]["task_aware_blocks"]["blocks"]
        calibration_stop = int(manifest["calibration"]["global_time_stop_exclusive"])
        blocks = tuple(
            slice(
                int(row["global_time_start"]) - calibration_stop,
                int(row["global_time_stop_exclusive"]) - calibration_stop,
            )
            for row in rows
        )
        tasks = tuple(str(row["task"]) for row in rows)
    else:
        blocks = tuple(
            slice(start, min(stream_length, start + int(block_size)))
            for start in range(0, stream_length, int(block_size))
        )
        tasks = tuple(str(stream_tasks[0]) for _ in blocks)
    if not blocks or blocks[0].start != 0 or blocks[-1].stop != stream_length:
        raise ValueError("Streaming blocks do not exactly cover the requested stream")
    return blocks, tasks


def augment_multitask_xlag(
    *,
    root: Path,
    stream: HippoERA5Dataset,
    stream_tasks: Sequence[str],
    selected_locations: Sequence[tuple[float, float]],
    xlag_length: int,
) -> HippoERA5Dataset:
    """Build X-lag features per task while retaining one global time axis.

    The public loader intentionally exposes one location-file list. For a
    concatenated task sequence that list points at the first task only, so the
    multivariate covariate reader cannot discover later task files. Loading and
    augmenting each task here also prevents lagged covariates from crossing a
    synthetic task boundary accidentally.
    """

    phi_chunks: list[np.ndarray] = []
    columns: list[str] | None = None
    offset = 0
    for task in stream_tasks:
        raw = load_hipposvgp_era5(
            root,
            tasks=(task,),
            variable_index=0,
            split="all",
            selected_locations=selected_locations,
        )
        stop = offset + raw.Y.shape[0]
        if stop > stream.Y.shape[0]:
            raise ValueError(f"{task} extends beyond the concatenated stream")
        np.testing.assert_allclose(raw.Y, stream.Y[offset:stop], atol=0.0, rtol=0.0)
        augmented = augment_dataset_phi(
            raw,
            phi_mode="medium_era5_xlag",
            xlag_length=xlag_length,
        )
        task_columns = list((augmented.metadata or {}).get("phi_columns", []))
        if columns is None:
            columns = task_columns
        elif task_columns != columns:
            raise ValueError("X-lag feature columns differ across ERA5 tasks")
        phi_chunks.append(np.asarray(augmented.Phi, dtype=np.float64))
        offset = stop
    if offset != stream.Y.shape[0]:
        raise ValueError("Per-task X-lag chunks do not cover the full stream")
    metadata = dict(stream.metadata or {})
    metadata.update(
        {
            "phi_mode": "medium_era5_xlag",
            "phi_columns": columns or [],
            "xlag_length": int(xlag_length),
            "multitask_xlag_boundary": "reset independently at each 186-hour task",
        }
    )
    return replace(stream, Phi=np.concatenate(phi_chunks, axis=0), metadata=metadata)


def load_benchmark_data(
    *,
    root: str | Path,
    stream_tasks: Sequence[str],
    split_seed: int,
    xlag_length: int = 10,
    ridge: float = 1e-3,
    inducing_sizes: Iterable[int] = (30, 64, 128),
    block_size: int = 10,
    validation_fraction: float = 0.1,
) -> BenchmarkData:
    root_path = Path(root)
    calibration_raw = load_hipposvgp_era5(
        root_path,
        tasks=CALIBRATION_TASKS,
        variable_index=0,
        split="all",
    )
    selected_locations = selected_locations_from_dataset(calibration_raw)
    stream_raw = load_hipposvgp_era5(
        root_path,
        tasks=tuple(stream_tasks),
        variable_index=0,
        split="all",
        selected_locations=selected_locations,
    )
    calibration, time_scale = normalise_time_dataset(calibration_raw)
    stream = normalise_time_dataset_with_scale(
        stream_raw,
        scale=time_scale,
        source="task_1_span",
    )
    calibration = augment_dataset_phi(
        calibration, phi_mode="medium_era5_xlag", xlag_length=xlag_length
    )
    stream = augment_multitask_xlag(
        root=root_path,
        stream=stream,
        stream_tasks=tuple(stream_tasks),
        selected_locations=selected_locations,
        xlag_length=xlag_length,
    )
    coordinates = standardize_coordinates(stream.coords)
    train_indices, test_indices = fixed_spatial_train_test_split(
        stream.Y.shape[1], test_fraction=0.2, seed=split_seed
    )
    fit_indices, validation_indices = inner_spatial_split(
        train_indices,
        split_seed=split_seed,
        validation_fraction=validation_fraction,
    )
    inducing: dict[int, np.ndarray] = {}
    for size in sorted({int(value) for value in inducing_sizes}):
        local = select_spatial_inducing_indices(
            coordinates[train_indices],
            min(size, train_indices.size),
            method="kmeans",
        )
        inducing[size] = coordinates[train_indices][local]
    task1_beta, task1_mean = fit_ridge_mean(
        calibration, train_indices, ridge=ridge
    )
    stream_phi = np.asarray(stream.Phi, dtype=np.float64).reshape(
        stream.Y.shape[0], stream.Y.shape[1], -1
    )
    task1_stream_mean = np.einsum("tsd,d->ts", stream_phi, task1_beta)
    batch_beta, batch_mean = fit_ridge_mean(stream, train_indices, ridge=ridge)
    blocks, block_tasks = task_aware_blocks(
        root=root_path,
        stream_tasks=tuple(stream_tasks),
        stream_length=stream.Y.shape[0],
        block_size=block_size,
    )
    return BenchmarkData(
        calibration=calibration,
        stream=stream,
        coordinates=coordinates,
        train_indices=train_indices,
        validation_indices=validation_indices,
        fit_indices=fit_indices,
        test_indices=test_indices,
        spatial_inducing=inducing,
        task1_ridge_beta=task1_beta,
        task1_ridge_mean=task1_mean,
        task1_stream_mean=task1_stream_mean,
        batch_ridge_beta=batch_beta,
        batch_ridge_mean=batch_mean,
        blocks=blocks,
        block_tasks=block_tasks,
        split_seed=int(split_seed),
        time_scale=float(time_scale),
        xlag_length=int(xlag_length),
        root=root_path,
    )


def flattened_inputs(
    times: np.ndarray,
    coordinates: np.ndarray,
    spatial_indices: np.ndarray,
    time_slice: slice | None = None,
    *,
    spatial_first: bool = False,
) -> np.ndarray:
    time_slice = time_slice or slice(0, len(times))
    selected_times = np.asarray(times[time_slice], dtype=np.float64)
    selected_coords = np.asarray(coordinates[spatial_indices], dtype=np.float64)
    tiled_time = np.repeat(selected_times, selected_coords.shape[0])[:, None]
    tiled_space = np.tile(selected_coords, (selected_times.shape[0], 1))
    return (
        np.column_stack([tiled_space, tiled_time])
        if spatial_first
        else np.column_stack([tiled_time, tiled_space])
    )


def flattened_targets(
    values: np.ndarray,
    spatial_indices: np.ndarray,
    time_slice: slice | None = None,
) -> np.ndarray:
    time_slice = time_slice or slice(0, values.shape[0])
    return np.asarray(values[time_slice][:, spatial_indices], dtype=np.float64).reshape(-1, 1)


def predictive_metrics(
    y_true: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    var = np.maximum(np.asarray(variance, dtype=np.float64).reshape(-1), 1e-10)
    half_width = 1.6448536269514722 * np.sqrt(var)
    return {
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "nll": float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var))),
        "coverage90": float(np.mean((y >= mu - half_width) & (y <= mu + half_width))),
        "mean_predictive_std": float(np.mean(np.sqrt(var))),
    }


def protocol_metadata(data: BenchmarkData) -> dict[str, Any]:
    return {
        "root": str(data.root.resolve()),
        "calibration_tasks": list(data.calibration.tasks),
        "stream_tasks": list(data.stream.tasks),
        "variable_index": 0,
        "num_calibration_times": int(data.calibration.Y.shape[0]),
        "num_stream_times": int(data.stream.Y.shape[0]),
        "num_locations": int(data.stream.Y.shape[1]),
        "num_train_locations": int(data.train_indices.size),
        "num_validation_locations": int(data.validation_indices.size),
        "num_fit_locations": int(data.fit_indices.size),
        "num_test_locations": int(data.test_indices.size),
        "split_seed": data.split_seed,
        "time_normalization": "(t - stream_start) / Task-1 span",
        "time_scale": data.time_scale,
        "coordinate_normalization": "global mean/std over the fixed 1000 locations",
        "xlag": {
            "mode": "medium_era5_xlag",
            "length": data.xlag_length,
            "features": int(data.stream.Phi.shape[1]),
            "ridge": 1e-3,
            "batch_mean_fit": "all stream times at the 800 training locations",
            "strict_online_mean_fit": "Task 1 at the same 800 training locations, then frozen",
        },
        "spatial_split": "seeded 800/200 held-out location split",
        "inducing_selection": "deterministic k-means over training coordinates only",
        "num_blocks": len(data.blocks),
        "block_lengths": [int(block.stop - block.start) for block in data.blocks],
        "task_aware_blocks": len(set(data.block_tasks)) > 1,
    }
