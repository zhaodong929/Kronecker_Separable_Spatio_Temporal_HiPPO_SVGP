"""Loader for processed HiPPO-SVGP ERA5 per-location time series.

The public loader returns arrays in the convention requested for baseline
experiments:

- `times`: shape `[T]`
- `coords`: shape `[S, 2]`, columns `(lat, lon)`
- `Y`: shape `[T, S]`
- `Phi`: shape `[T * S, p]`, ordered by time first, then location

The helper `to_routeb_synthetic_dataset` converts this representation to the
existing Route B `SyntheticDataset`, whose `Y` convention is `[S, T]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np

from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    BlockFactors,
    SyntheticDataset,
    make_block_factors,
)


LAT_LON_RE = re.compile(r"lat_([-0-9.]+)_lon_([-0-9.]+)")
SPLIT_KEYS = {
    "train": ("data_train", "time_train"),
    "val": ("data_val", "time_val"),
    "test": ("data_test", "time_test"),
}


@dataclass(frozen=True)
class HippoERA5Dataset:
    times: np.ndarray
    coords: np.ndarray
    Y: np.ndarray
    Phi: np.ndarray
    tasks: tuple[str, ...]
    variable_index: int
    scaled: bool
    selected_files: tuple[Path, ...]
    Y_unscaled: np.ndarray | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class HippoERA5Block:
    block_id: int
    block_slice: slice
    times: np.ndarray
    coords: np.ndarray
    Y: np.ndarray
    Phi: np.ndarray
    Y_unscaled: np.ndarray | None = None


def parse_lat_lon_from_filename(path: str | Path) -> tuple[float, float]:
    path = Path(path)
    match = LAT_LON_RE.search(path.stem.replace("_scaled", ""))
    if match is None:
        raise ValueError(f"Cannot parse lat/lon from {path.name}")
    return float(match.group(1)), float(match.group(2))


def _real_npz_files(sequence_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.name.endswith(".npz") and ":Zone.Identifier" not in path.name
    )


def _base_location_name(path: Path) -> str:
    return path.name.replace("_scaled.npz", ".npz")


def discover_sequence_files(root: str | Path, tasks: Sequence[str], *, prefer_scaled: bool = True) -> list[Path]:
    root = Path(root)
    files: list[Path] = []
    for task in tasks:
        seq_dir = root / task / "sequences"
        if not seq_dir.exists():
            raise FileNotFoundError(f"Missing sequence directory: {seq_dir}")
        task_files = _real_npz_files(seq_dir)
        if prefer_scaled:
            task_files = [path for path in task_files if path.name.endswith("_scaled.npz")]
        else:
            task_files = [path for path in task_files if not path.name.endswith("_scaled.npz")]
        files.extend(task_files)
    if not files:
        suffix = "*_scaled.npz" if prefer_scaled else "*.npz"
        raise FileNotFoundError(f"No {suffix} files found under {root} for tasks {tasks}")
    return sorted(files)


def _filter_by_bounds(
    files: Iterable[Path],
    *,
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
) -> list[Path]:
    selected: list[Path] = []
    for path in files:
        lat, lon = parse_lat_lon_from_filename(path)
        if lat_bounds is not None:
            lo, hi = sorted(lat_bounds)
            if not lo <= lat <= hi:
                continue
        if lon_bounds is not None:
            lo, hi = sorted(lon_bounds)
            if not lo <= lon <= hi:
                continue
        selected.append(path)
    return selected


def select_location_files(
    files: Sequence[Path],
    *,
    first_n: int | None = None,
    random_n: int | None = None,
    seed: int = 0,
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
) -> list[Path]:
    """Select location files, preserving at most one file per lat/lon per task."""

    selected = _filter_by_bounds(files, lat_bounds=lat_bounds, lon_bounds=lon_bounds)
    selected = sorted(selected, key=lambda path: (path.parent.parent.name, parse_lat_lon_from_filename(path)))
    if random_n is not None:
        rng = np.random.default_rng(seed)
        if random_n < len(selected):
            idx = np.sort(rng.choice(len(selected), size=random_n, replace=False))
            selected = [selected[int(i)] for i in idx]
    if first_n is not None:
        selected = selected[:first_n]
    if not selected:
        raise ValueError("No ERA5 location files remain after spatial selection")
    return selected


def _split_names(split: str) -> list[str]:
    if split == "all":
        return ["train", "val", "test"]
    if split not in SPLIT_KEYS:
        raise ValueError(f"split must be one of {['all', *SPLIT_KEYS]}")
    return [split]


def _read_one_file(path: Path, variable_index: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    times_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    with np.load(path, allow_pickle=True) as data:
        for split_name in _split_names(split):
            data_key, time_key = SPLIT_KEYS[split_name]
            values = np.asarray(data[data_key][variable_index], dtype=float).reshape(-1)
            times = np.asarray(data[time_key], dtype=float).reshape(-1)
            if values.shape[0] != times.shape[0]:
                raise ValueError(f"{path.name}: {data_key} and {time_key} lengths do not match")
            order = np.argsort(times)
            times_chunks.append(times[order])
            value_chunks.append(values[order])
    times_full = np.concatenate(times_chunks)
    values_full = np.concatenate(value_chunks)
    order = np.argsort(times_full)
    return times_full[order], values_full[order]


def _matching_unscaled_file(path: Path) -> Path | None:
    if not path.name.endswith("_scaled.npz"):
        return path
    candidate = path.with_name(path.name.replace("_scaled.npz", ".npz"))
    return candidate if candidate.exists() else None


def _align_files(files: Sequence[Path], variable_index: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    series = []
    common: set[float] | None = None
    for path in files:
        times, values = _read_one_file(path, variable_index, split)
        lookup = {float(t): float(v) for t, v in zip(times, values)}
        series.append(lookup)
        keys = set(lookup.keys())
        common = keys if common is None else common.intersection(keys)
    if not common:
        raise ValueError("No common timestamps across selected ERA5 files")
    times_common = np.asarray(sorted(common), dtype=float)
    Y = np.asarray([[lookup[float(t)] for lookup in series] for t in times_common], dtype=float)
    return times_common, Y


def _concatenate_tasks(
    root: Path,
    tasks: Sequence[str],
    *,
    variable_index: int,
    split: str,
    prefer_scaled: bool,
    selected_locations: Sequence[tuple[float, float]] | None,
    first_n: int | None,
    random_n: int | None,
    seed: int,
    lat_bounds: tuple[float, float] | None,
    lon_bounds: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, list[Path], np.ndarray | None]:
    time_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    unscaled_chunks: list[np.ndarray] = []
    selected_files_all: list[Path] = []
    coords_reference: np.ndarray | None = None
    last_time: float | None = None

    selected_location_set = set(selected_locations) if selected_locations is not None else None

    for task in tasks:
        task_files = discover_sequence_files(root, [task], prefer_scaled=prefer_scaled)
        if selected_location_set is not None:
            task_files = [
                path for path in task_files if parse_lat_lon_from_filename(path) in selected_location_set
            ]
        else:
            task_files = select_location_files(
                task_files,
                first_n=first_n,
                random_n=random_n,
                seed=seed,
                lat_bounds=lat_bounds,
                lon_bounds=lon_bounds,
            )
            selected_location_set = {parse_lat_lon_from_filename(path) for path in task_files}

        task_files = sorted(task_files, key=lambda path: parse_lat_lon_from_filename(path))
        coords = np.asarray([parse_lat_lon_from_filename(path) for path in task_files], dtype=float)
        if coords_reference is None:
            coords_reference = coords
        elif coords.shape != coords_reference.shape or not np.allclose(coords, coords_reference):
            raise ValueError("All requested ERA5 tasks must share the selected spatial locations")

        times, Y = _align_files(task_files, variable_index, split)
        if last_time is not None and times[0] <= last_time:
            times = times + (last_time + 1.0 - times[0])
        last_time = float(times[-1])
        time_chunks.append(times)
        y_chunks.append(Y)
        selected_files_all.extend(task_files)

        unscaled_paths = [_matching_unscaled_file(path) for path in task_files]
        if all(path is not None and path.exists() for path in unscaled_paths):
            _, Y_unscaled = _align_files([path for path in unscaled_paths if path is not None], variable_index, split)
            unscaled_chunks.append(Y_unscaled)

    if coords_reference is None:
        raise ValueError("No coordinates were loaded")
    times_full = np.concatenate(time_chunks)
    Y_full = np.concatenate(y_chunks, axis=0)
    Y_unscaled_full = np.concatenate(unscaled_chunks, axis=0) if len(unscaled_chunks) == len(tasks) else None
    return times_full, Y_full, selected_files_all, Y_unscaled_full


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    scale = np.maximum(values.std(axis=0, keepdims=True), 1e-8)
    return (values - mean) / scale, mean, scale


def build_phi_features(
    times: np.ndarray,
    coords: np.ndarray,
    *,
    time_period: float | None = None,
) -> np.ndarray:
    """Build linear features in time-major order.

    Columns are:
    constant, scaled time, scaled latitude, scaled longitude,
    sin/cos first harmonic, sin/cos second harmonic.
    """

    times = np.asarray(times, dtype=float).reshape(-1)
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape [S, 2] with columns (lat, lon)")
    t_span = max(float(times[-1] - times[0]), 1e-12)
    t_scaled = (times - times[0]) / t_span
    coords_scaled, _, _ = _standardize(coords)
    period = float(time_period) if time_period is not None else 1.0
    phase = 2.0 * np.pi * t_scaled / max(period, 1e-12)

    S = coords.shape[0]
    t_rep = np.repeat(t_scaled, S)
    lat_tile = np.tile(coords_scaled[:, 0], times.shape[0])
    lon_tile = np.tile(coords_scaled[:, 1], times.shape[0])
    phase_rep = np.repeat(phase, S)
    return np.column_stack(
        [
            np.ones(times.shape[0] * S),
            t_rep,
            lat_tile,
            lon_tile,
            np.sin(phase_rep),
            np.cos(phase_rep),
            np.sin(2.0 * phase_rep),
            np.cos(2.0 * phase_rep),
        ]
    )


def load_hipposvgp_era5(
    root: str | Path = "data/era5/processed_timeseries_4",
    *,
    tasks: Sequence[str] = ("task_1",),
    variable_index: int = 0,
    prefer_scaled: bool = True,
    split: str = "all",
    first_n_locations: int | None = None,
    random_n_locations: int | None = None,
    selected_locations: Sequence[tuple[float, float]] | None = None,
    seed: int = 0,
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
    max_time: int | None = None,
    include_unscaled: bool = True,
) -> HippoERA5Dataset:
    """Load processed ERA5 tasks and stack per-location series into `[T, S]`."""

    root_path = Path(root)
    times, Y, selected_files, Y_unscaled = _concatenate_tasks(
        root_path,
        tasks,
        variable_index=variable_index,
        split=split,
        prefer_scaled=prefer_scaled,
        selected_locations=selected_locations,
        first_n=first_n_locations,
        random_n=random_n_locations,
        seed=seed,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
    )
    if max_time is not None:
        times = times[:max_time]
        Y = Y[:max_time]
        if Y_unscaled is not None:
            Y_unscaled = Y_unscaled[:max_time]
    coords = np.asarray([parse_lat_lon_from_filename(path) for path in selected_files[: Y.shape[1]]], dtype=float)
    Phi = build_phi_features(times, coords)
    metadata = {
        "root": str(root_path),
        "split": split,
        "prefer_scaled": prefer_scaled,
        "coords_order": "lat_lon",
        "phi_columns": [
            "1",
            "time_scaled",
            "lat_scaled",
            "lon_scaled",
            "sin_time",
            "cos_time",
            "sin_2time",
            "cos_2time",
        ],
    }
    return HippoERA5Dataset(
        times=np.asarray(times, dtype=float),
        coords=coords,
        Y=np.asarray(Y, dtype=float),
        Phi=Phi,
        tasks=tuple(tasks),
        variable_index=int(variable_index),
        scaled=bool(prefer_scaled),
        selected_files=tuple(selected_files[: Y.shape[1]]),
        Y_unscaled=np.asarray(Y_unscaled, dtype=float) if include_unscaled and Y_unscaled is not None else None,
        metadata=metadata,
    )


def iter_online_blocks(num_time: int, block_size: int) -> list[slice]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return [slice(start, min(num_time, start + block_size)) for start in range(0, num_time, block_size)]


def make_blocks(dataset: HippoERA5Dataset, block_size: int) -> list[HippoERA5Block]:
    blocks = []
    S = dataset.coords.shape[0]
    for block_id, block in enumerate(iter_online_blocks(dataset.Y.shape[0], block_size)):
        row_idx = []
        for t_idx in range(block.start or 0, block.stop or dataset.Y.shape[0]):
            row_idx.extend(range(t_idx * S, (t_idx + 1) * S))
        blocks.append(
            HippoERA5Block(
                block_id=block_id,
                block_slice=block,
                times=dataset.times[block],
                coords=dataset.coords,
                Y=dataset.Y[block],
                Phi=dataset.Phi[np.asarray(row_idx)],
                Y_unscaled=dataset.Y_unscaled[block] if dataset.Y_unscaled is not None else None,
            )
        )
    return blocks


def to_routeb_synthetic_dataset(
    dataset: HippoERA5Dataset,
    *,
    sigma2: float,
    gp_prior_variance: float = 1.0,
    standardize_coords: bool = True,
) -> SyntheticDataset:
    coords = dataset.coords
    if standardize_coords:
        coords, _, _ = _standardize(coords)
    Y_st = dataset.Y.T
    return SyntheticDataset(
        times=dataset.times,
        spatial_coords=coords,
        Y=Y_st,
        F=np.zeros_like(Y_st),
        Phi=dataset.Phi,
        beta_true=np.zeros(dataset.Phi.shape[1]),
        sigma2=float(sigma2),
        gp_prior_variance=float(gp_prior_variance),
    )


def make_routeb_block_factors(
    dataset: HippoERA5Dataset,
    *,
    block: slice,
    z_t: np.ndarray,
    z_t_old: np.ndarray | None,
    lengthscale: float,
    sigma2: float,
    gp_prior_variance: float = 1.0,
) -> BlockFactors:
    routeb_dataset = to_routeb_synthetic_dataset(
        dataset,
        sigma2=sigma2,
        gp_prior_variance=gp_prior_variance,
    )
    return make_block_factors(
        routeb_dataset,
        block=block,
        z_t=z_t,
        z_t_old=z_t_old,
        lengthscale=lengthscale,
    )
