"""Causal, dataset-agnostic adapters for the PEMS-BAY and METR-LA benchmarks.

The module intentionally does not implement the usual traffic-forecasting time
split.  It materialises this project's paired spatial streaming protocol:
fixed spatial hold-outs, a Task-1 calibration prefix, and later strict-online
nowcasting or forecasting.  Arrays follow the repository-wide convention
``[time, sensor]`` and feature rows are time-major.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from stvgp_kronecker.joint_ssgp_kron.synthetic import SyntheticDataset


DATASET_SPECS = {
    "pems_bay": {
        "h5_name": "PEMS-BAY.h5",
        "coordinates_name": "graph_sensor_locations_bay.csv",
        "sensors": 325,
    },
    "metr_la": {
        "h5_name": "metr-la.h5",
        "coordinates_name": "graph_sensor_locations.csv",
        "sensors": 207,
    },
}


def _normalise_sensor_id(value: object) -> str:
    """Represent numeric CSV ids and HDF column labels in one stable form."""

    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if not np.isfinite(numeric):
        return text
    return str(int(numeric)) if numeric.is_integer() else text


def _standardise_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, keepdims=True)
    scale = np.std(values, axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-8)
    return (values - mean) / scale, mean.reshape(-1), scale.reshape(-1)


def _causal_fill_missing(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Forward-fill missing readings without borrowing a future observation.

    A leading missing reading has no causal value available and is rejected
    rather than silently back-filled from the future.  The released DCRNN HDF
    files are complete, but this makes a different raw mirror auditable.
    """

    values = np.asarray(values, dtype=np.float64).copy()
    if values.ndim != 2:
        raise ValueError(f"Traffic readings must have shape [time, sensor], got {values.shape}")
    missing = ~np.isfinite(values)
    count = int(missing.sum())
    if not count:
        return values, 0
    for sensor in range(values.shape[1]):
        column = values[:, sensor]
        if not np.isfinite(column[0]):
            raise ValueError(
                f"Sensor index {sensor} has a leading missing value; causal filling is undefined"
            )
        for time_index in range(1, column.size):
            if not np.isfinite(column[time_index]):
                column[time_index] = column[time_index - 1]
        values[:, sensor] = column
    return values, count


def _read_coordinate_table(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] < 3:
        raise ValueError(f"Coordinate file needs at least three columns: {path}")
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    numeric_rows = numeric.dropna(axis=0, how="any")
    if numeric_rows.empty:
        raise ValueError(f"Coordinate file has no numeric sensor rows: {path}")
    offset = 1 if raw.shape[1] >= 4 else 0
    ids = tuple(_normalise_sensor_id(value) for value in numeric_rows.iloc[:, offset])
    coords = numeric_rows.iloc[:, offset + 1 : offset + 3].to_numpy(dtype=np.float64)
    if len(set(ids)) != len(ids):
        raise ValueError(f"Coordinate file contains duplicate sensor identifiers: {path}")
    if not np.isfinite(coords).all():
        raise ValueError(f"Coordinate file contains non-finite latitude/longitude values: {path}")
    return ids, coords


def _validate_traffic_frame(frame: pd.DataFrame, *, expected_sensors: int) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame from HDF, got {type(frame).__name__}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Traffic HDF index must be a DatetimeIndex")
    if frame.shape[1] != expected_sensors:
        raise ValueError(f"Expected {expected_sensors} sensors, found {frame.shape[1]}")
    if frame.index.has_duplicates:
        raise ValueError("Traffic timestamps must be unique")
    frame = frame.sort_index()
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Traffic timestamps must be chronologically ordered")
    if len(frame.index) > 1:
        timestamps_ns = frame.index.to_numpy(dtype="datetime64[ns]").astype("int64")
        seconds = np.diff(timestamps_ns) / 1_000_000_000.0
        if np.any(seconds <= 0.0) or not np.allclose(seconds / 300.0, np.round(seconds / 300.0)):
            raise ValueError("Traffic timestamps must advance in positive five-minute multiples")
    return frame


def _read_traffic_hdf(path: Path) -> pd.DataFrame:
    """Read both current pandas HDF and the older DCRNN fixed-format HDF files.

    The public files advertise pandas 0.15.2 metadata.  Modern PyTables rejects
    their byte-valued metadata before pandas gets to the numerical arrays, so a
    small HDF5 fallback reconstructs the same DataFrame from its one numeric
    block.  No target value is transformed in this fallback.
    """

    try:
        return pd.read_hdf(path)
    except (ImportError, OSError, TypeError, ValueError):
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "Reading this legacy traffic HDF requires h5py (`pip install h5py`)"
            ) from exc
        with h5py.File(path, "r") as handle:
            candidates = [
                group
                for group in handle.values()
                if hasattr(group, "keys")
                and {"axis0", "axis1", "block0_values"}.issubset(group.keys())
            ]
            if len(candidates) != 1:
                raise ValueError(f"Could not identify one fixed-format DataFrame in {path}")
            group = candidates[0]
            columns_raw = np.asarray(group["axis0"])
            columns = [
                value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else value
                for value in columns_raw
            ]
            timestamps = pd.to_datetime(np.asarray(group["axis1"]), unit="ns")
            values = np.asarray(group["block0_values"], dtype=np.float64)
        return pd.DataFrame(values, index=pd.DatetimeIndex(timestamps), columns=columns)


@dataclass(frozen=True)
class TrafficDataset:
    name: str
    timestamps: pd.DatetimeIndex
    times_hours: np.ndarray
    values: np.ndarray
    values_standardised: np.ndarray
    coordinates_lat_lon: np.ndarray
    coordinates_standardised: np.ndarray
    sensor_ids: tuple[str, ...]
    phi: np.ndarray
    target_mean: float
    target_scale: float
    task1_steps: int
    missing_values_forward_filled: int
    timestamp_gap_steps: int
    source_paths: dict[str, str]

    @property
    def num_time(self) -> int:
        return int(self.values.shape[0])

    @property
    def num_sensors(self) -> int:
        return int(self.values.shape[1])

    def to_routeb_synthetic_dataset(self, *, noise_variance: float, prior_variance: float = 1.0) -> SyntheticDataset:
        return SyntheticDataset(
            times=self.times_hours,
            spatial_coords=self.coordinates_standardised,
            Y=self.values_standardised.T,
            F=np.zeros_like(self.values_standardised.T),
            Phi=self.phi.reshape(-1, self.phi.shape[-1]),
            beta_true=np.zeros(self.phi.shape[-1]),
            sigma2=float(noise_variance),
            gp_prior_variance=float(prior_variance),
        )

    def target_to_speed(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.target_scale + self.target_mean


@dataclass(frozen=True)
class SpatialSplit:
    dataset: str
    seed: int
    heldout_indices: tuple[int, ...]
    visible_indices: tuple[int, ...]
    visible_calibration_indices: tuple[int, ...]
    visible_validation_indices: tuple[int, ...]
    heldout_sensor_ids: tuple[str, ...]
    visible_sensor_ids: tuple[str, ...]
    sensor_id_sha256: str
    coordinates_sha256: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_calendar_spatial_features(
    timestamps: pd.DatetimeIndex,
    coordinates_standardised: np.ndarray,
) -> np.ndarray:
    """Return [1, sin/cos(time-of-day), sin/cos(day-of-week), lat, lon]."""

    timestamps = pd.DatetimeIndex(timestamps)
    coords = np.asarray(coordinates_standardised, dtype=np.float64)
    minutes = (
        timestamps.hour.to_numpy(dtype=float) * 60.0
        + timestamps.minute.to_numpy(dtype=float)
        + timestamps.second.to_numpy(dtype=float) / 60.0
    )
    tod_angle = 2.0 * np.pi * minutes / (24.0 * 60.0)
    dow_angle = 2.0 * np.pi * timestamps.dayofweek.to_numpy(dtype=float) / 7.0
    temporal = np.column_stack(
        (
            np.ones(len(timestamps), dtype=np.float64),
            np.sin(tod_angle),
            np.cos(tod_angle),
            np.sin(dow_angle),
            np.cos(dow_angle),
        )
    )
    return np.concatenate(
        (
            np.broadcast_to(temporal[:, None, :], (len(timestamps), coords.shape[0], temporal.shape[1])),
            np.broadcast_to(coords[None, :, :], (len(timestamps), coords.shape[0], coords.shape[1])),
        ),
        axis=-1,
    ).copy()


def build_road_context_xlag_features(
    dataset: TrafficDataset,
    *,
    context_indices: np.ndarray,
    scaler_fit_indices: np.ndarray,
    road_distance_csv: str | Path,
    lag_count: int = 10,
    graph_diffusion: float = 7.448975327393576,
    feature_mean: np.ndarray | None = None,
    feature_scale: np.ndarray | None = None,
) -> tuple[TrafficDataset, dict[str, object]]:
    """Append causal road-context lags to the deterministic features.

    PEMS-BAY contains only the speed target, so these columns are not independent
    exogenous variables. Each dynamic covariate is instead a graph-weighted
    summary of sensors visible under the active spatial split. For a visible
    query sensor its own value is excluded. Hidden-sensor values therefore never
    enter the feature map, while current visible readings remain legal for the
    contemporaneous nowcasting protocol.
    """

    if lag_count < 1:
        raise ValueError("lag_count must be positive")
    if graph_diffusion <= 0.0:
        raise ValueError("graph_diffusion must be positive")
    context = np.asarray(context_indices, dtype=int)
    fit = np.asarray(scaler_fit_indices, dtype=int)
    if context.ndim != 1 or fit.ndim != 1 or not context.size or not fit.size:
        raise ValueError("context_indices and scaler_fit_indices must be non-empty vectors")
    if np.any(context < 0) or np.any(context >= dataset.num_sensors):
        raise ValueError("context_indices contain an invalid sensor index")
    if np.any(fit < 0) or np.any(fit >= dataset.num_sensors):
        raise ValueError("scaler_fit_indices contain an invalid sensor index")

    from stvgp_kronecker.traffic_spatial_kernels import load_road_laplacian

    laplacian, road_metadata = load_road_laplacian(road_distance_csv, dataset.sensor_ids)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    spectrum = np.exp(-float(graph_diffusion) * np.maximum(eigenvalues, 0.0))
    heat = (eigenvectors * spectrum[None, :]) @ eigenvectors.T
    heat = np.maximum(0.5 * (heat + heat.T), 0.0)
    weights = heat[:, context].copy()
    context_position = {int(sensor): position for position, sensor in enumerate(context)}
    for query, position in context_position.items():
        weights[query, position] = 0.0

    row_sum = weights.sum(axis=1)
    empty = row_sum <= 1e-12
    if np.any(empty):
        coords = dataset.coordinates_standardised
        distances = np.linalg.norm(coords[empty, None, :] - coords[context][None, :, :], axis=-1)
        fallback = 1.0 / np.maximum(distances, 1e-6)
        for row, query in enumerate(np.flatnonzero(empty)):
            position = context_position.get(int(query))
            if position is not None:
                fallback[row, position] = 0.0
        weights[empty] = fallback
        row_sum = weights.sum(axis=1)
    weights /= np.maximum(row_sum[:, None], 1e-12)

    road_context = dataset.values_standardised[:, context] @ weights.T
    dynamic_array = np.empty(
        (dataset.num_time, dataset.num_sensors, 1 + 2 * lag_count), dtype=np.float32
    )
    dynamic_array[:, :, 0] = road_context
    names = ["road_context_t"]
    for lag in range(1, lag_count + 1):
        dynamic_array[:lag, :, lag] = road_context[0]
        dynamic_array[lag:, :, lag] = road_context[:-lag]
        names.append(f"road_context_t-{lag}")
    for lag in range(1, lag_count + 1):
        dynamic_array[:, :, lag_count + lag] = road_context - dynamic_array[:, :, lag]
        names.append(f"road_context_t-road_context_t-{lag}")

    if feature_mean is None or feature_scale is None:
        fit_values = dynamic_array[: dataset.task1_steps, fit, :].reshape(-1, dynamic_array.shape[-1])
        feature_mean = fit_values.mean(axis=0, dtype=np.float64)
        feature_scale = np.maximum(fit_values.std(axis=0, dtype=np.float64), 1e-8)
    else:
        feature_mean = np.asarray(feature_mean, dtype=np.float64)
        feature_scale = np.asarray(feature_scale, dtype=np.float64)
        expected = (dynamic_array.shape[-1],)
        if feature_mean.shape != expected or feature_scale.shape != expected:
            raise ValueError(f"Locked road-context scaler must have shape {expected}")
        if np.any(feature_scale <= 0.0):
            raise ValueError("Locked road-context feature scales must be positive")
    np.subtract(dynamic_array, feature_mean[None, None, :], out=dynamic_array)
    np.divide(dynamic_array, feature_scale[None, None, :], out=dynamic_array)
    augmented_phi = np.concatenate((dataset.phi.astype(np.float32), dynamic_array), axis=-1)
    augmented = replace(dataset, phi=augmented_phi)
    metadata: dict[str, object] = {
        "mode": "road_context_xlag",
        "interpretation": "graph-weighted observed-speed context; not an independent exogenous channel",
        "lag_count": int(lag_count),
        "graph_diffusion": float(graph_diffusion),
        "base_feature_count": int(dataset.phi.shape[-1]),
        "dynamic_feature_count": int(dynamic_array.shape[-1]),
        "total_feature_count": int(augmented.phi.shape[-1]),
        "dynamic_feature_names": names,
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "context_sensor_count": int(context.size),
        "scaler_fit_sensor_count": int(fit.size),
        "uses_current_hidden_target": False,
        "uses_current_visible_targets": True,
        "road_metadata": road_metadata,
    }
    return augmented, metadata


def load_traffic_dataset(
    root: str | Path,
    dataset: str,
    *,
    task1_steps: int = 2016,
    scaler_fit_indices: Iterable[int],
) -> TrafficDataset:
    """Load traffic and fit scaling only on explicitly available Task-1 sensors.

    Select the spatial split before calling this function. Requiring indices
    prevents accidentally fitting preprocessing on the held-out targets.
    """

    dataset = str(dataset).lower().replace("-", "_")
    if dataset not in DATASET_SPECS:
        raise ValueError(f"dataset must be one of {sorted(DATASET_SPECS)}, got {dataset!r}")
    if task1_steps < 2:
        raise ValueError("task1_steps must be at least two five-minute samples")
    spec = DATASET_SPECS[dataset]
    root = Path(root)
    dataset_root = root / dataset
    h5_path = dataset_root / str(spec["h5_name"])
    coordinates_path = dataset_root / str(spec["coordinates_name"])
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing traffic HDF: {h5_path}. Run scripts/download_traffic_benchmarks.py first.")
    if not coordinates_path.exists():
        raise FileNotFoundError(f"Missing sensor coordinates: {coordinates_path}")
    frame = _read_traffic_hdf(h5_path)
    frame = _validate_traffic_frame(frame, expected_sensors=int(spec["sensors"]))
    if task1_steps >= frame.shape[0]:
        raise ValueError(f"task1_steps={task1_steps} leaves no strict-online stream in {h5_path}")
    data_ids = tuple(_normalise_sensor_id(column) for column in frame.columns)
    if len(set(data_ids)) != len(data_ids):
        raise ValueError("Traffic HDF has duplicate sensor columns after normalisation")
    coordinate_ids, coordinate_values = _read_coordinate_table(coordinates_path)
    coordinate_lookup = {sensor: coordinate_values[index] for index, sensor in enumerate(coordinate_ids)}
    missing_coords = [sensor for sensor in data_ids if sensor not in coordinate_lookup]
    if missing_coords:
        raise ValueError(f"Coordinates missing for {len(missing_coords)} HDF sensors; first={missing_coords[:3]}")
    coordinates = np.asarray([coordinate_lookup[sensor] for sensor in data_ids], dtype=np.float64)
    values, missing_count = _causal_fill_missing(frame.to_numpy(dtype=np.float64))
    fit = np.asarray(tuple(scaler_fit_indices))
    if (fit.ndim != 1 or fit.size == 0 or not np.issubdtype(fit.dtype, np.integer)
            or np.any(fit < 0) or np.any(fit >= values.shape[1])
            or np.unique(fit).size != fit.size):
        raise ValueError("scaler_fit_indices must be unique valid integer sensor indices")
    task1_values = values[:task1_steps, fit]
    target_mean = float(task1_values.mean())
    target_scale = float(max(task1_values.std(), 1e-8))
    values_standardised = (values - target_mean) / target_scale
    coords_standardised, _, _ = _standardise_columns(coordinates)
    timestamps_ns = frame.index.to_numpy(dtype="datetime64[ns]").astype("int64")
    times_hours = (timestamps_ns - timestamps_ns[0]).astype(np.float64) / 3_600_000_000_000.0
    cadence_steps = np.diff(timestamps_ns) / (300.0 * 1_000_000_000.0)
    timestamp_gap_steps = int(np.maximum(np.round(cadence_steps).astype(int) - 1, 0).sum())
    phi = build_calendar_spatial_features(frame.index, coords_standardised)
    return TrafficDataset(
        name=dataset,
        timestamps=frame.index,
        times_hours=times_hours,
        values=values,
        values_standardised=values_standardised,
        coordinates_lat_lon=coordinates,
        coordinates_standardised=coords_standardised,
        sensor_ids=data_ids,
        phi=phi,
        target_mean=target_mean,
        target_scale=target_scale,
        task1_steps=int(task1_steps),
        missing_values_forward_filled=missing_count,
        timestamp_gap_steps=timestamp_gap_steps,
        source_paths={"h5": str(h5_path), "coordinates": str(coordinates_path)},
    )


def make_spatial_split(
    dataset: TrafficDataset,
    *,
    seed: int,
    heldout_fraction: float = 0.2,
    visible_validation_fraction: float = 0.1,
) -> SpatialSplit:
    """Create one deterministic paired spatial split without touching target values."""

    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in (0, 1)")
    if not 0.0 < visible_validation_fraction < 1.0:
        raise ValueError("visible_validation_fraction must be in (0, 1)")
    rng = np.random.default_rng(int(seed))
    n = dataset.num_sensors
    heldout_count = int(round(n * heldout_fraction))
    heldout_count = max(1, min(n - 2, heldout_count))
    heldout = np.sort(rng.choice(n, size=heldout_count, replace=False))
    visible = np.setdiff1d(np.arange(n, dtype=int), heldout, assume_unique=True)
    validation_count = int(round(visible.size * visible_validation_fraction))
    validation_count = max(1, min(visible.size - 1, validation_count))
    validation = np.sort(rng.choice(visible, size=validation_count, replace=False))
    calibration = np.setdiff1d(visible, validation, assume_unique=True)
    id_bytes = "\n".join(dataset.sensor_ids).encode("utf-8")
    coord_bytes = np.ascontiguousarray(dataset.coordinates_lat_lon, dtype=np.float64).tobytes()
    return SpatialSplit(
        dataset=dataset.name,
        seed=int(seed),
        heldout_indices=tuple(int(index) for index in heldout),
        visible_indices=tuple(int(index) for index in visible),
        visible_calibration_indices=tuple(int(index) for index in calibration),
        visible_validation_indices=tuple(int(index) for index in validation),
        heldout_sensor_ids=tuple(dataset.sensor_ids[int(index)] for index in heldout),
        visible_sensor_ids=tuple(dataset.sensor_ids[int(index)] for index in visible),
        sensor_id_sha256=sha256(id_bytes).hexdigest(),
        coordinates_sha256=sha256(coord_bytes).hexdigest(),
    )


def write_spatial_splits(
    dataset: TrafficDataset,
    output_dir: str | Path,
    *,
    seeds: Iterable[int] = range(5),
    heldout_fraction: float = 0.2,
    visible_validation_fraction: float = 0.1,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in seeds:
        split = make_spatial_split(
            dataset,
            seed=int(seed),
            heldout_fraction=heldout_fraction,
            visible_validation_fraction=visible_validation_fraction,
        )
        payload = {
            "schema_version": 1,
            "protocol": "paired_spatial_streaming",
            "target": "traffic speed",
            "task1_steps": dataset.task1_steps,
            "cadence_minutes": 5,
            "mean_columns": ["1", "sin_tod", "cos_tod", "sin_dow", "cos_dow", "lat", "lon"],
            "coordinate_order": "latitude_longitude",
            "missing_policy": "causal_forward_fill_only",
            "split": split.as_dict(),
        }
        path = output_dir / f"{dataset.name}_seed{seed}_spatial_split.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def load_spatial_split(path: str | Path) -> SpatialSplit:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    split = dict(payload["split"])
    for key in (
        "heldout_indices",
        "visible_indices",
        "visible_calibration_indices",
        "visible_validation_indices",
    ):
        split[key] = tuple(int(value) for value in split[key])
    for key in ("heldout_sensor_ids", "visible_sensor_ids"):
        split[key] = tuple(str(value) for value in split[key])
    return SpatialSplit(**split)
