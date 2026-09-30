#!/usr/bin/env python3
"""Reconstruct and extend the public HiPPO-SVGP ERA5-Land task sequence.

The public repository contains only ``task_1`` and ``task_2`` although its
preprocessing notebook creates 50 chronological tasks of 186 hours each.
This tool deliberately writes an independent extension root.  It never
modifies the preserved ``processed_timeseries_4`` directory.

The workflow is intentionally staged:

1. ``download`` retrieves a bounded ERA5-Land raw-data slice from CDS.
2. ``validate`` restores chronological values from the preserved Task 1--2
   files and compares them to the downloaded raw fields.
3. ``generate`` is permitted only after the Task 1--2 raw-value validation
   passes.  It copies the original Task 1--2 directories and generates
   deterministic Task 3--10 splits, all scaled with the original Task 1
   scaler.

The original notebook uses unseeded ``np.random.permutation`` calls.  Its
historic split assignments therefore cannot be reconstructed from the public
files.  This tool preserves those assignments for Task 1--2 and uses a fixed,
recorded seed for newly created Task 3--10 assignments.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable, Sequence
import zipfile

import numpy as np


DATASET_NAME = "reanalysis-era5-land"
DEFAULT_AREA = (60.0, -10.0, 49.0, 2.0)  # north, west, south, east
TASK_LENGTH = 186
REFERENCE_TASKS = (1, 2)
EXTENSION_TASKS = tuple(range(3, 11))
DEFAULT_SPLIT_SEED = 20260801
LAT_LON_RE = re.compile(r"lat_([-0-9.]+)_lon_([-0-9.]+)")

# Keep the official downloader's requested order.  The validation stage checks
# the resulting feature ordering against the preserved unscaled task arrays.
ERA5_VARIABLES = (
    "2m_dewpoint_temperature",
    "2m_temperature",
    "skin_temperature",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "skin_reservoir_content",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "forecast_albedo",
    "surface_latent_heat_flux",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_sensible_heat_flux",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "evaporation_from_bare_soil",
    "evaporation_from_open_water_surfaces_excluding_oceans",
    "evaporation_from_the_top_of_canopy",
    "evaporation_from_vegetation_transpiration",
    "potential_evaporation",
    "runoff",
    "snow_evaporation",
    "sub_surface_runoff",
    "surface_runoff",
    "total_evaporation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_precipitation",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
)

# CDS accepts descriptive variable names, while NetCDF responses commonly use
# ERA5 short names.  The original notebook serialised ``Dataset.to_dataframe``
# values, so the validation below is the authority on whether this canonical
# request order reproduces the historic feature order.
NETCDF_VARIABLE_ALIASES = {
    "2m_dewpoint_temperature": ("d2m",),
    "2m_temperature": ("t2m",),
    "skin_temperature": ("skt",),
    "soil_temperature_level_1": ("stl1",),
    "soil_temperature_level_2": ("stl2",),
    "soil_temperature_level_3": ("stl3",),
    "soil_temperature_level_4": ("stl4",),
    "skin_reservoir_content": ("src",),
    "volumetric_soil_water_layer_1": ("swvl1",),
    "volumetric_soil_water_layer_2": ("swvl2",),
    "volumetric_soil_water_layer_3": ("swvl3",),
    "volumetric_soil_water_layer_4": ("swvl4",),
    "forecast_albedo": ("fal",),
    "surface_latent_heat_flux": ("slhf",),
    "surface_net_solar_radiation": ("ssr",),
    "surface_net_thermal_radiation": ("str",),
    "surface_sensible_heat_flux": ("sshf",),
    "surface_solar_radiation_downwards": ("ssrd",),
    "surface_thermal_radiation_downwards": ("strd",),
    "evaporation_from_bare_soil": ("evabs",),
    "evaporation_from_open_water_surfaces_excluding_oceans": ("evaow",),
    "evaporation_from_the_top_of_canopy": ("evatc",),
    "evaporation_from_vegetation_transpiration": ("evavt",),
    "potential_evaporation": ("pev",),
    "runoff": ("ro",),
    "snow_evaporation": ("es",),
    "sub_surface_runoff": ("ssro",),
    "surface_runoff": ("sro",),
    "total_evaporation": ("e",),
    "10m_u_component_of_wind": ("u10",),
    "10m_v_component_of_wind": ("v10",),
    "surface_pressure": ("sp",),
    "total_precipitation": ("tp",),
    "leaf_area_index_high_vegetation": ("lai_hv",),
    "leaf_area_index_low_vegetation": ("lai_lv",),
}
HOURS = tuple(f"{hour:02d}:00" for hour in range(24))


@dataclass(frozen=True)
class ValidationSummary:
    status: str
    reference_root: str
    raw_dir: str
    tasks: list[str]
    locations: int
    features: int
    hours_compared: int
    raw_time_start: str
    raw_time_end: str
    time_preprocessing: str
    locations_with_compacted_rows: int
    max_dropped_rows_before_horizon: int
    max_absolute_error: float
    rmse: float
    num_nonfinite_raw: int
    nonfinite_examples: list[dict[str, Any]]
    num_mismatched_finite: int
    first_mismatch_examples: list[dict[str, Any]]
    per_variable_errors: list[dict[str, Any]]
    tolerances: dict[str, float]
    raw_files: dict[str, str]


def _require(module: str, purpose: str) -> Any:
    try:
        return __import__(module)
    except ImportError as exc:
        raise RuntimeError(
            f"{module} is required for {purpose}. Install the reconstruction "
            "dependencies with: .venv/bin/python -m pip install cdsapi xarray netCDF4"
        ) from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed.replace(minute=0, second=0, microsecond=0)


def _parse_area(value: str) -> tuple[float, float, float, float]:
    parts = tuple(float(item.strip()) for item in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("area must be NORTH,WEST,SOUTH,EAST")
    north, west, south, east = parts
    if north < south or west > east:
        raise argparse.ArgumentTypeError("area must satisfy north >= south and west <= east")
    return parts


def _task_name(task_id: int) -> str:
    return f"task_{task_id}"


def _task_slice(task_id: int) -> slice:
    start = (task_id - 1) * TASK_LENGTH
    return slice(start, start + TASK_LENGTH)


def _base_location_name(path: Path) -> str:
    return path.name.replace("_scaled.npz", ".npz")


def _parse_coordinate(path: Path) -> tuple[float, float]:
    match = LAT_LON_RE.search(path.stem.replace("_scaled", ""))
    if match is None:
        raise ValueError(f"Cannot parse latitude/longitude from {path.name}")
    return float(match.group(1)), float(match.group(2))


def _unscaled_location_files(task_dir: Path) -> list[Path]:
    sequence_dir = task_dir / "sequences"
    files = sorted(
        path
        for path in sequence_dir.glob("*.npz")
        if "_scaled" not in path.name and ":Zone.Identifier" not in path.name
    )
    if not files:
        raise FileNotFoundError(f"No unscaled sequence files under {sequence_dir}")
    return files


def _reference_locations(reference_root: Path) -> list[tuple[float, float]]:
    files = _unscaled_location_files(reference_root / "task_1")
    locations = [_parse_coordinate(path) for path in files]
    if len(locations) != len(set(locations)):
        raise ValueError("Task 1 contains duplicate spatial coordinates")
    return sorted(locations)


def _reference_file_map(task_dir: Path) -> dict[tuple[float, float], Path]:
    mapping = {_parse_coordinate(path): path for path in _unscaled_location_files(task_dir)}
    if len(mapping) != len(_unscaled_location_files(task_dir)):
        raise ValueError(f"Duplicate coordinates under {task_dir}")
    return mapping


def _restore_chronological_file(path: Path, expected_times: np.ndarray) -> np.ndarray:
    """Restore a `[feature, time]` array from the saved random split chunks."""

    restored: np.ndarray | None = None
    filled = np.zeros(expected_times.size, dtype=bool)
    index_by_time = {int(time): index for index, time in enumerate(expected_times)}
    with np.load(path, allow_pickle=False) as source:
        for split in ("train", "val", "test"):
            values = np.asarray(source[f"data_{split}"], dtype=np.float64)
            times = np.asarray(source[f"time_{split}"], dtype=np.int64)
            if values.ndim != 2 or values.shape[1] != times.size:
                raise ValueError(f"Malformed {split} data in {path}")
            if restored is None:
                restored = np.empty((values.shape[0], expected_times.size), dtype=np.float64)
            elif values.shape[0] != restored.shape[0]:
                raise ValueError(f"Feature count changes across splits in {path}")
            for column, time in enumerate(times):
                if int(time) not in index_by_time:
                    raise ValueError(f"Unexpected absolute time {time} in {path}")
                target = index_by_time[int(time)]
                if filled[target]:
                    raise ValueError(f"Duplicate absolute time {time} in {path}")
                restored[:, target] = values[:, column]
                filled[target] = True
    if restored is None or not np.all(filled):
        missing = expected_times[~filled]
        raise ValueError(f"Missing {missing.size} timestamps in {path}")
    return restored


def load_preserved_reference(
    reference_root: Path,
    task_ids: Sequence[int],
    locations: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Return the preserved unscaled sequence as `[feature, time, location]`."""

    chunks: list[np.ndarray] = []
    for task_id in task_ids:
        expected_times = np.arange(_task_slice(task_id).start, _task_slice(task_id).stop, dtype=np.int64)
        path_map = _reference_file_map(reference_root / _task_name(task_id))
        if not set(locations).issubset(path_map):
            raise ValueError(f"{_task_name(task_id)} is missing requested Task 1 locations")
        location_series = [_restore_chronological_file(path_map[location], expected_times) for location in locations]
        chunk = np.stack(location_series, axis=-1)
        chunks.append(chunk)
    result = np.concatenate(chunks, axis=1)
    if result.shape[0] != len(ERA5_VARIABLES):
        raise ValueError(
            f"Expected {len(ERA5_VARIABLES)} variables but preserved data has {result.shape[0]}"
        )
    return result


def _date_chunks(
    start: datetime,
    hours: int,
    days_per_request: int,
) -> Iterable[tuple[int, int, list[str]]]:
    if hours <= 0:
        raise ValueError("hours must be positive")
    if days_per_request <= 0:
        raise ValueError("days_per_request must be positive")
    end = start + timedelta(hours=hours - 1)
    cursor = start.date()
    final_date = end.date()
    while cursor <= final_date:
        year, month = cursor.year, cursor.month
        days = []
        while (
            cursor <= final_date
            and cursor.year == year
            and cursor.month == month
            and len(days) < days_per_request
        ):
            days.append(f"{cursor.day:02d}")
            cursor += timedelta(days=1)
        yield year, month, days


def _download_request(
    year: int,
    month: int,
    days: Sequence[str],
    area: tuple[float, float, float, float],
) -> dict[str, Any]:
    return {
        "variable": list(ERA5_VARIABLES),
        "year": f"{year:04d}",
        "month": f"{month:02d}",
        "day": list(days),
        "time": list(HOURS),
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": list(area),
    }


def download_raw_slice(
    raw_dir: Path,
    *,
    start: datetime,
    hours: int,
    area: tuple[float, float, float, float],
    days_per_request: int,
    max_workers: int,
    force: bool,
) -> list[Path]:
    """Download only calendar days touching the requested hourly slice."""

    cdsapi = _require("cdsapi", "ERA5-Land download")
    raw_dir.mkdir(parents=True, exist_ok=True)
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    requests: list[tuple[Path, dict[str, Any], str]] = []
    targets: list[Path] = []
    for year, month, days in _date_chunks(start, hours, days_per_request):
        target = raw_dir / (
            f"era5_land_{year:04d}{month:02d}{days[0]}_{year:04d}{month:02d}{days[-1]}.nc"
        )
        if target.exists() and not force:
            print(f"Reusing existing raw file: {target}", flush=True)
            targets.append(target)
            continue
        request = _download_request(year, month, days, area)
        label = f"{year:04d}-{month:02d} days {days[0]}--{days[-1]}"
        requests.append((target, request, label))

    def retrieve_one(item: tuple[Path, dict[str, Any], str]) -> Path:
        target, request, label = item
        print(f"Requesting {label}, area={area} -> {target}", flush=True)
        cdsapi.Client().retrieve(DATASET_NAME, request).download(str(target))
        print(f"Completed {label}: {target}", flush=True)
        return target

    if max_workers == 1:
        targets.extend(retrieve_one(item) for item in requests)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(retrieve_one, item): item for item in requests}
            for future in as_completed(futures):
                targets.append(future.result())
    targets = sorted(targets)

    _write_json(
        raw_dir / "download_manifest.json",
        {
            "created_at_utc": _utc_now(),
            "dataset": DATASET_NAME,
            "variables": list(ERA5_VARIABLES),
            "start": start.isoformat(),
            "requested_hours": int(hours),
            "area": list(area),
            "files": [
                {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in targets
            ],
        },
    )
    return targets


def _expanded_netcdf_paths(raw_dir: Path) -> list[Path]:
    """Return real NetCDF members, expanding ZIP responses from CDS as needed."""

    paths: list[Path] = []
    for source in sorted(raw_dir.glob("*.nc")):
        if not zipfile.is_zipfile(source):
            paths.append(source)
            continue

        archive_hash = _sha256(source)
        destination = raw_dir / "expanded" / source.stem
        marker = destination / "archive_manifest.json"
        current = None
        if marker.exists():
            current = json.loads(marker.read_text(encoding="utf-8"))
        if current is None or current.get("archive_sha256") != archive_hash:
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True, exist_ok=False)
            members: list[str] = []
            with zipfile.ZipFile(source) as archive:
                for info in archive.infolist():
                    member = Path(info.filename)
                    if info.is_dir() or member.suffix.lower() != ".nc":
                        continue
                    if member.is_absolute() or ".." in member.parts:
                        raise ValueError(f"Unsafe path in CDS archive {source}: {info.filename}")
                    target = destination / member
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as input_handle, target.open("wb") as output_handle:
                        shutil.copyfileobj(input_handle, output_handle)
                    members.append(str(member))
            if not members:
                raise ValueError(f"CDS ZIP response contains no NetCDF members: {source}")
            _write_json(
                marker,
                {
                    "archive": str(source.resolve()),
                    "archive_sha256": archive_hash,
                    "members": members,
                },
            )
        paths.extend(sorted(destination.rglob("*.nc")))
    if not paths:
        raise FileNotFoundError(f"No NetCDF files or NetCDF-containing CDS archives found in {raw_dir}")
    return paths


def _normalise_raw_dataset(dataset: Any, path: Path) -> Any:
    lat_name = next(
        (name for name in ("latitude", "lat") if name in dataset.coords or name in dataset.dims), None
    )
    lon_name = next(
        (name for name in ("longitude", "lon") if name in dataset.coords or name in dataset.dims), None
    )
    time_name = next((name for name in ("valid_time", "time") if name in dataset.dims), None)
    if lat_name is None or lon_name is None or time_name is None:
        raise ValueError(
            f"Could not identify coordinates in {path}: dims={dict(dataset.sizes)}, "
            f"coords={list(dataset.coords)}"
        )
    rename = {}
    if lat_name != "latitude":
        rename[lat_name] = "latitude"
    if lon_name != "longitude":
        rename[lon_name] = "longitude"
    if time_name != "valid_time":
        rename[time_name] = "valid_time"
    if rename:
        dataset = dataset.rename(rename)

    longitudes = np.asarray(dataset["longitude"].values, dtype=float)
    normalized = (longitudes + 180.0) % 360.0 - 180.0
    dataset = dataset.assign_coords(longitude=(dataset["longitude"].dims, normalized))
    if "longitude" in dataset.dims:
        dataset = dataset.sortby("longitude")
    return dataset


def _select_raw_locations(
    dataset: Any,
    locations: Sequence[tuple[float, float]],
    *,
    path: Path,
) -> Any:
    xr = _require("xarray", "selecting ERA5-Land locations")
    requested_lats = np.asarray([item[0] for item in locations], dtype=float)
    requested_lons = np.asarray([item[1] for item in locations], dtype=float)
    if "latitude" in dataset.dims and "longitude" in dataset.dims:
        selected = dataset.sel(
            {
                "latitude": xr.DataArray(requested_lats, dims="location"),
                "longitude": xr.DataArray(requested_lons, dims="location"),
            },
            method="nearest",
        )
        selected_lats = np.asarray(selected["latitude"].values, dtype=float)
        selected_lons = np.asarray(selected["longitude"].values, dtype=float)
    else:
        if len(locations) != 1:
            raise ValueError(
                f"{path} contains one grid point but extraction requested {len(locations)} locations"
            )
        selected_lats = np.asarray([float(dataset["latitude"].values)], dtype=float)
        selected_lons = np.asarray([float(dataset["longitude"].values)], dtype=float)
        selected = dataset.expand_dims(location=[0])
    if not np.allclose(selected_lats, requested_lats, atol=1e-7, rtol=0.0) or not np.allclose(
        selected_lons, requested_lons, atol=1e-7, rtol=0.0
    ):
        max_lat = float(np.max(np.abs(selected_lats - requested_lats)))
        max_lon = float(np.max(np.abs(selected_lons - requested_lons)))
        raise ValueError(
            f"{path} does not reproduce saved coordinates exactly: "
            f"max latitude diff={max_lat:.3e}, longitude diff={max_lon:.3e}"
        )
    return selected.load()


def _open_raw_dataset(
    raw_dir: Path,
    locations: Sequence[tuple[float, float]],
) -> tuple[Any, str, str, str]:
    xr = _require("xarray", "reading CDS NetCDF files")
    paths = _expanded_netcdf_paths(raw_dir)
    opened = [xr.open_dataset(path) for path in paths]
    try:
        datasets = [
            _select_raw_locations(
                _normalise_raw_dataset(dataset, path),
                locations,
                path=path,
            )
            for dataset, path in zip(opened, paths)
        ]
        chunks_by_variable: dict[str, list[Any]] = {}
        for dataset in datasets:
            for name, array in dataset.data_vars.items():
                chunks_by_variable.setdefault(name, []).append(array)

        data_vars: dict[str, Any] = {}
        for name, chunks in chunks_by_variable.items():
            combined_array = chunks[0] if len(chunks) == 1 else xr.concat(chunks, dim="valid_time")
            values = np.asarray(combined_array["valid_time"].values)
            order = np.argsort(values)
            combined_array = combined_array.isel(valid_time=order)
            sorted_values = np.asarray(combined_array["valid_time"].values)
            _, first_indices = np.unique(sorted_values, return_index=True)
            if first_indices.size != sorted_values.size:
                combined_array = combined_array.isel(valid_time=np.sort(first_indices))
            data_vars[name] = combined_array
        combined = xr.Dataset(data_vars)
        combined.load()
        return combined, "valid_time", "latitude", "longitude"
    finally:
        for dataset in opened:
            dataset.close()


def extract_raw_features(
    raw_dir: Path,
    locations: Sequence[tuple[float, float]],
    *,
    max_hours: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract complete rows using the public notebook's per-location semantics.

    The upstream notebook first calls ``Dataset.to_dataframe().dropna()`` and
    only then groups rows by location.  A missing field therefore removes one
    complete location-time row and shifts that location's later synthetic time
    indices.  We reproduce that behavior exactly.  The returned physical time
    array has shape ``[time, location]`` because locations can consequently
    have different physical timestamps at the same synthetic index.
    """

    xr = _require("xarray", "extracting ERA5-Land features")
    dataset, time_dim, lat_name, lon_name = _open_raw_dataset(raw_dir, locations)
    raw_names: list[str] = []
    unresolved: list[str] = []
    for variable in ERA5_VARIABLES:
        candidates = (variable, *NETCDF_VARIABLE_ALIASES[variable])
        raw_name = next((candidate for candidate in candidates if candidate in dataset.data_vars), None)
        if raw_name is None:
            unresolved.append(variable)
        else:
            raw_names.append(raw_name)
    if unresolved:
        raise ValueError(
            "Raw data is missing requested ERA5-Land variables: "
            f"{unresolved}; available variables are {list(dataset.data_vars)}"
        )

    array = dataset[raw_names].to_array(dim="feature")
    extra_dims = [dim for dim in array.dims if dim not in {"feature", time_dim, "location"}]
    for dim in extra_dims:
        if array.sizes[dim] != 1:
            raise ValueError(f"Unexpected non-singleton raw-data dimension {dim}={array.sizes[dim]}")
        array = array.isel({dim: 0}, drop=True)
    array = array.transpose("feature", time_dim, "location")
    features = np.asarray(array.values, dtype=np.float64)
    times = np.asarray(dataset[time_dim].values)
    features, times = _compact_complete_rows(features, times, max_hours=max_hours)
    if features.shape[0] != len(ERA5_VARIABLES) or features.shape[2] != len(locations):
        raise AssertionError(f"Unexpected extracted shape {features.shape}")
    return features, times


def _compact_complete_rows(
    features: np.ndarray,
    times: np.ndarray,
    *,
    max_hours: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop incomplete rows independently per location, preserving provenance."""

    values = np.asarray(features, dtype=np.float64)
    physical_times = np.asarray(times)
    if values.ndim != 3:
        raise ValueError(f"Expected [feature, time, location], got {values.shape}")
    if physical_times.ndim != 1 or physical_times.size != values.shape[1]:
        raise ValueError("Physical time coordinate must be one-dimensional and match the time axis")
    if max_hours is not None and max_hours <= 0:
        raise ValueError("max_hours must be positive")

    complete_indices = [
        np.flatnonzero(np.all(np.isfinite(values[:, :, location_index]), axis=0))
        for location_index in range(values.shape[2])
    ]
    target_hours = (
        min(indices.size for indices in complete_indices) if max_hours is None else int(max_hours)
    )
    insufficient = [
        (location_index, int(indices.size))
        for location_index, indices in enumerate(complete_indices)
        if indices.size < target_hours
    ]
    if insufficient:
        examples = ", ".join(f"location {index}: {count}" for index, count in insufficient[:10])
        raise ValueError(
            f"{len(insufficient)} locations have fewer than {target_hours} complete rows ({examples})"
        )

    compact = np.empty((values.shape[0], target_hours, values.shape[2]), dtype=np.float64)
    compact_times = np.empty((target_hours, values.shape[2]), dtype=physical_times.dtype)
    for location_index, indices in enumerate(complete_indices):
        selected = indices[:target_hours]
        compact[:, :, location_index] = values[:, selected, location_index]
        compact_times[:, location_index] = physical_times[selected]
    return compact, compact_times


def _physical_time_compaction_summary(physical_times: np.ndarray) -> tuple[int, int]:
    """Count locations affected by complete-row compaction within the horizon."""

    times = np.asarray(physical_times)
    if times.ndim != 2:
        raise ValueError(f"Expected [time, location] physical timestamps, got {times.shape}")
    global_start = np.min(times[0])
    offsets = ((times - global_start) / np.timedelta64(1, "h")).astype(np.int64)
    expected = np.arange(times.shape[0], dtype=np.int64)[:, None]
    affected = int(np.any(offsets != expected, axis=0).sum())
    dropped_before_horizon = offsets[-1] - (times.shape[0] - 1)
    return affected, int(np.max(dropped_before_horizon))


def _raw_file_hashes(raw_dir: Path) -> dict[str, str]:
    return {path.name: _sha256(path) for path in sorted(raw_dir.glob("*.nc"))}


def validate_reference(
    reference_root: Path,
    raw_dir: Path,
    *,
    task_ids: Sequence[int],
    selected_locations: Sequence[tuple[float, float]] | None,
    max_hours: int | None,
    atol: float,
    rtol: float,
    report_path: Path,
) -> ValidationSummary:
    all_locations = _reference_locations(reference_root)
    locations = list(selected_locations) if selected_locations is not None else all_locations
    unknown = sorted(set(locations) - set(all_locations))
    if unknown:
        raise ValueError(f"Requested probe locations are absent from preserved Task 1: {unknown}")
    reference = load_preserved_reference(reference_root, task_ids, locations)
    if max_hours is not None:
        if max_hours <= 0:
            raise ValueError("max_hours must be positive")
        reference = reference[:, :max_hours, :]
    raw, raw_times = extract_raw_features(raw_dir, locations, max_hours=reference.shape[1])
    if raw.shape != reference.shape:
        raise AssertionError(f"Raw shape {raw.shape} does not match preserved shape {reference.shape}")

    finite_mask = np.isfinite(raw)
    nonfinite = int(np.size(raw) - finite_mask.sum())
    error = raw - reference
    finite_errors = error[finite_mask]
    max_abs = float(np.max(np.abs(finite_errors)))
    rmse = float(np.sqrt(np.mean(np.square(finite_errors))))
    nonfinite_examples = []
    for feature_index, time_index, location_index in np.argwhere(~finite_mask)[:20]:
        nonfinite_examples.append(
            {
                "variable_index": int(feature_index),
                "variable": ERA5_VARIABLES[int(feature_index)],
                "time_index_within_validation": int(time_index),
                "raw_time": str(raw_times[int(time_index), int(location_index)]),
                "location_index": int(location_index),
                "latitude": float(locations[int(location_index)][0]),
                "longitude": float(locations[int(location_index)][1]),
                "reference_value": float(reference[feature_index, time_index, location_index]),
            }
        )
    close_mask = np.isclose(raw, reference, atol=atol, rtol=rtol, equal_nan=False)
    mismatch_mask = finite_mask & ~close_mask
    first_mismatch_examples = []
    for feature_index, time_index, location_index in np.argwhere(mismatch_mask)[:20]:
        raw_value = float(raw[feature_index, time_index, location_index])
        reference_value = float(reference[feature_index, time_index, location_index])
        first_mismatch_examples.append(
            {
                "variable_index": int(feature_index),
                "variable": ERA5_VARIABLES[int(feature_index)],
                "time_index_within_validation": int(time_index),
                "raw_time": str(raw_times[int(time_index), int(location_index)]),
                "location_index": int(location_index),
                "latitude": float(locations[int(location_index)][0]),
                "longitude": float(locations[int(location_index)][1]),
                "raw_value": raw_value,
                "reference_value": reference_value,
                "absolute_error": abs(raw_value - reference_value),
            }
        )
    per_variable_errors = []
    for feature_index, variable in enumerate(ERA5_VARIABLES):
        feature_finite = finite_mask[feature_index]
        feature_errors = error[feature_index][feature_finite]
        per_variable_errors.append(
            {
                "variable_index": feature_index,
                "variable": variable,
                "num_nonfinite": int(np.size(feature_finite) - feature_finite.sum()),
                "num_mismatched_finite": int(mismatch_mask[feature_index].sum()),
                "max_absolute_error": float(np.max(np.abs(feature_errors))),
                "rmse": float(np.sqrt(np.mean(np.square(feature_errors)))),
            }
        )
    num_mismatched_finite = int(mismatch_mask.sum())
    passed = bool(nonfinite == 0 and num_mismatched_finite == 0)
    affected_locations, max_dropped_rows = _physical_time_compaction_summary(raw_times)
    summary = ValidationSummary(
        status="passed" if passed else "failed",
        reference_root=str(reference_root.resolve()),
        raw_dir=str(raw_dir.resolve()),
        tasks=[_task_name(task_id) for task_id in task_ids],
        locations=len(locations),
        features=len(ERA5_VARIABLES),
        hours_compared=int(reference.shape[1]),
        raw_time_start=str(np.min(raw_times[0])),
        raw_time_end=str(np.max(raw_times[-1])),
        time_preprocessing=(
            "Per-location complete-case row compaction matching "
            "Dataset.to_dataframe().dropna() in the public notebook"
        ),
        locations_with_compacted_rows=affected_locations,
        max_dropped_rows_before_horizon=max_dropped_rows,
        max_absolute_error=max_abs,
        rmse=rmse,
        num_nonfinite_raw=nonfinite,
        nonfinite_examples=nonfinite_examples,
        num_mismatched_finite=num_mismatched_finite,
        first_mismatch_examples=first_mismatch_examples,
        per_variable_errors=per_variable_errors,
        tolerances={"atol": float(atol), "rtol": float(rtol)},
        raw_files=_raw_file_hashes(raw_dir),
    )
    _write_json(report_path, asdict(summary))
    print(json.dumps(asdict(summary), indent=2), flush=True)
    if not passed:
        raise RuntimeError(
            "Raw-value validation failed. Do not generate later tasks until the CDS request, "
            "time origin, or feature ordering has been reconciled."
        )
    return summary


def _split_indices(task_id: int, seed: int) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(np.random.SeedSequence([seed, task_id]))
    permutation = generator.permutation(TASK_LENGTH)
    train_end = int(TASK_LENGTH * 0.7)
    val_end = int(TASK_LENGTH * 0.8)
    return {
        "train": permutation[:train_end],
        "val": permutation[train_end:val_end],
        "test": permutation[val_end:],
    }


def _copy_reference_tasks(reference_root: Path, output_root: Path) -> None:
    for task_id in REFERENCE_TASKS:
        source = reference_root / _task_name(task_id)
        destination = output_root / _task_name(task_id)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        shutil.copytree(source, destination)
    source_scaler = reference_root / "global_scaler.pkl"
    if not source_scaler.exists():
        raise FileNotFoundError(f"Missing original Task 1 scaler: {source_scaler}")
    shutil.copy2(source_scaler, output_root / "global_scaler.pkl")


def _load_reference_scaler(reference_root: Path) -> Any:
    joblib = _require("joblib", "loading the preserved Task 1 scaler")
    return joblib.load(reference_root / "global_scaler.pkl")


def _save_task(
    output_root: Path,
    task_id: int,
    data: np.ndarray,
    locations: Sequence[tuple[float, float]],
    scaler: Any,
    split_seed: int,
) -> dict[str, Any]:
    if data.shape != (len(ERA5_VARIABLES), TASK_LENGTH, len(locations)):
        raise ValueError(f"Unexpected Task {task_id} array shape {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Task {task_id} contains non-finite values")

    task_dir = output_root / _task_name(task_id)
    sequence_dir = task_dir / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=False)
    indices = _split_indices(task_id, split_seed)
    absolute_times = np.arange(_task_slice(task_id).start, _task_slice(task_id).stop, dtype=np.int64)

    for location_index, (latitude, longitude) in enumerate(locations):
        # The preserved public tasks store both raw and scaled payloads as
        # float32.  Match that representation to avoid doubling extension size
        # and to preserve StandardScaler's original float32 rounding path.
        values = np.asarray(data[:, :, location_index], dtype=np.float32)
        raw_payload: dict[str, np.ndarray] = {}
        scaled_payload: dict[str, np.ndarray] = {}
        for split, relative_idx in indices.items():
            split_values = values[:, relative_idx]
            raw_payload[f"data_{split}"] = split_values
            raw_payload[f"time_{split}"] = absolute_times[relative_idx]
            scaled_payload[f"data_{split}"] = np.asarray(
                scaler.transform(split_values.T).T,
                dtype=np.float32,
            )
            scaled_payload[f"time_{split}"] = absolute_times[relative_idx]
        filename = f"lat_{latitude:.4f}_lon_{longitude:.4f}.npz"
        np.savez(sequence_dir / filename, **raw_payload)
        np.savez(sequence_dir / filename.replace(".npz", "_scaled.npz"), **scaled_payload)

    shutil.copy2(output_root / "global_scaler.pkl", task_dir / "scaler.pkl")
    return {
        "task": _task_name(task_id),
        "absolute_time_start": int(absolute_times[0]),
        "absolute_time_stop_exclusive": int(absolute_times[-1] + 1),
        "seed": int(split_seed),
        "split_sizes": {name: int(values.size) for name, values in indices.items()},
        "split_index_sha256": {
            name: hashlib.sha256(values.tobytes()).hexdigest() for name, values in indices.items()
        },
    }


def _build_streaming_manifest(output_root: Path) -> dict[str, Any]:
    task_aware_blocks: list[dict[str, Any]] = []
    continuous_blocks: list[dict[str, Any]] = []

    task_block_id = 0
    for task_id in range(2, 11):
        task_start = _task_slice(task_id).start
        for offset in range(0, TASK_LENGTH, 10):
            stop = min(offset + 10, TASK_LENGTH)
            task_aware_blocks.append(
                {
                    "block_id": task_block_id,
                    "task": _task_name(task_id),
                    "global_time_start": task_start + offset,
                    "global_time_stop_exclusive": task_start + stop,
                    "hours": stop - offset,
                    "reset_state_at_boundary": False,
                }
            )
            task_block_id += 1

    streaming_start = _task_slice(2).start
    streaming_stop = _task_slice(10).stop
    for block_id, start in enumerate(range(streaming_start, streaming_stop, 10)):
        stop = min(start + 10, streaming_stop)
        continuous_blocks.append(
            {
                "block_id": block_id,
                "global_time_start": start,
                "global_time_stop_exclusive": stop,
                "hours": stop - start,
                "reset_state_at_boundary": False,
            }
        )

    return {
        "created_at_utc": _utc_now(),
        "dataset_root": str(output_root.resolve()),
        "calibration": {
            "task": "task_1",
            "global_time_start": 0,
            "global_time_stop_exclusive": TASK_LENGTH,
            "purpose": "Task-1 Route-B empirical-Bayes calibration before streaming",
        },
        "streaming": {
            "tasks": [_task_name(task_id) for task_id in range(2, 11)],
            "global_time_start": streaming_start,
            "global_time_stop_exclusive": streaming_stop,
            "hours": streaming_stop - streaming_start,
            "state_transfer": "continuous; do not reset at task boundaries",
            "task_aware_blocks": {
                "count": len(task_aware_blocks),
                "description": (
                    "171 task-aware blocks: 19 per 186-hour task. The nineteenth block in each "
                    "task is 6 hours, not 10 hours."
                ),
                "blocks": task_aware_blocks,
            },
            "continuous_10_hour_blocks": {
                "count": len(continuous_blocks),
                "description": (
                    "168 uninterrupted 10-hour blocks after flattening Task 2--10; the final "
                    "block is 4 hours."
                ),
                "blocks": continuous_blocks,
            },
        },
    }


def generate_extension(
    reference_root: Path,
    raw_dir: Path,
    output_root: Path,
    validation_report: Path,
    *,
    split_seed: int,
) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output root already contains files: {output_root}. Refusing to overwrite preserved data."
        )
    if not validation_report.exists():
        raise FileNotFoundError(f"Required validation report is missing: {validation_report}")
    validation = json.loads(validation_report.read_text(encoding="utf-8"))
    required_tasks = {_task_name(task_id) for task_id in REFERENCE_TASKS}
    if validation.get("status") != "passed" or not required_tasks.issubset(validation.get("tasks", [])):
        raise RuntimeError("Task 1--2 raw-value validation has not passed; generation is blocked.")
    if int(validation.get("hours_compared", 0)) < TASK_LENGTH * len(REFERENCE_TASKS):
        raise RuntimeError("Validation did not cover the complete preserved Task 1--2 horizon.")

    locations = _reference_locations(reference_root)
    required_hours = _task_slice(max(EXTENSION_TASKS)).stop
    raw, raw_times = extract_raw_features(raw_dir, locations, max_hours=required_hours)
    if raw.shape[1] != required_hours:
        raise AssertionError("Raw extraction did not provide the required 1,860 hours")

    output_root.mkdir(parents=True, exist_ok=False)
    try:
        _copy_reference_tasks(reference_root, output_root)
        provenance_dir = output_root / "provenance"
        provenance_dir.mkdir(parents=True, exist_ok=False)
        np.save(provenance_dir / "physical_valid_times.npy", raw_times)
        np.save(provenance_dir / "location_coordinates.npy", np.asarray(locations, dtype=np.float64))
        scaler = _load_reference_scaler(reference_root)
        generated_tasks = []
        for task_id in EXTENSION_TASKS:
            task_data = raw[:, _task_slice(task_id), :]
            generated_tasks.append(_save_task(output_root, task_id, task_data, locations, scaler, split_seed))
            print(f"Generated {_task_name(task_id)}", flush=True)

        source_reference_files = [
            reference_root / "global_scaler.pkl",
            *[reference_root / _task_name(task_id) / "scaler.pkl" for task_id in REFERENCE_TASKS],
        ]
        manifest = {
            "created_at_utc": _utc_now(),
            "kind": "paper-inspired deterministic long-horizon ERA5-Land extension",
            "reference_root": str(reference_root.resolve()),
            "output_root": str(output_root.resolve()),
            "raw_dir": str(raw_dir.resolve()),
            "validation_report": str(validation_report.resolve()),
            "source_task_1_task_2": "byte-for-byte copied from preserved reference root",
            "scaling": {
                "protocol": "reuse preserved Task-1 StandardScaler",
                "path": "global_scaler.pkl",
                "storage_dtype": "float32, matching preserved Task 1--2 payloads",
                "reason": "keeps Task 1--2 and new Task 3--10 on the original experimental scale",
            },
            "split_protocol": {
                "task_1_task_2": "preserved historic assignments; upstream random seed was not recorded",
                "task_3_task_10": "independent deterministic SeedSequence([split_seed, task_id])",
                "split_seed": int(split_seed),
                "ratios": {"train": 0.7, "val": 0.1, "test": 0.2},
            },
            "task_length_hours": TASK_LENGTH,
            "num_locations": len(locations),
            "num_variables": len(ERA5_VARIABLES),
            "variables": list(ERA5_VARIABLES),
            "time_axis": {
                "model_time": (
                    "Synthetic global indices 0..1859, matching the public notebook after "
                    "per-location complete-case compaction"
                ),
                "physical_time_shape": list(raw_times.shape),
                "physical_time_path": "provenance/physical_valid_times.npy",
                "location_coordinate_path": "provenance/location_coordinates.npy",
                "warning": (
                    "Physical timestamps can differ by location after a missing raw row; use the "
                    "sidecar when physical-time alignment matters."
                ),
            },
            "raw_time_start": str(np.min(raw_times[0])),
            "raw_time_end": str(np.max(raw_times[required_hours - 1])),
            "raw_files": _raw_file_hashes(raw_dir),
            "reference_file_hashes": {
                str(path.relative_to(reference_root)): _sha256(path) for path in source_reference_files
            },
            "generated_tasks": generated_tasks,
        }
        _write_json(output_root / "dataset_manifest.json", manifest)
        _write_json(output_root / "long_streaming_manifest.json", _build_streaming_manifest(output_root))
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def verify_extension(
    reference_root: Path,
    output_root: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Audit the generated task tree, split payloads, scaling, and manifests."""

    errors: list[str] = []
    expected_locations = _reference_locations(reference_root)
    expected_location_set = set(expected_locations)
    scaler = _load_reference_scaler(reference_root)
    max_scaling_error = 0.0
    generated_max_scaling_error = 0.0
    task_summaries: list[dict[str, Any]] = []

    for task_id in range(1, 11):
        task_max_scaling_error = 0.0
        task_dir = output_root / _task_name(task_id)
        raw_files = _unscaled_location_files(task_dir)
        scaled_files = sorted((task_dir / "sequences").glob("*_scaled.npz"))
        raw_map = {_parse_coordinate(path): path for path in raw_files}
        scaled_map = {_parse_coordinate(path): path for path in scaled_files}
        if set(raw_map) != expected_location_set:
            errors.append(f"{_task_name(task_id)} unscaled coordinate set differs from Task 1")
        if set(scaled_map) != expected_location_set:
            errors.append(f"{_task_name(task_id)} scaled coordinate set differs from Task 1")

        expected_times = np.arange(
            _task_slice(task_id).start,
            _task_slice(task_id).stop,
            dtype=np.int64,
        )
        for location in expected_locations:
            raw_path = raw_map.get(location)
            scaled_path = scaled_map.get(location)
            if raw_path is None or scaled_path is None:
                continue
            observed_times: list[np.ndarray] = []
            with (
                np.load(raw_path, allow_pickle=False) as raw_payload,
                np.load(scaled_path, allow_pickle=False) as scaled_payload,
            ):
                for split in ("train", "val", "test"):
                    data_key = f"data_{split}"
                    time_key = f"time_{split}"
                    raw_values = np.asarray(raw_payload[data_key])
                    scaled_values = np.asarray(scaled_payload[data_key])
                    split_times = np.asarray(raw_payload[time_key], dtype=np.int64)
                    scaled_times = np.asarray(scaled_payload[time_key], dtype=np.int64)
                    if raw_values.shape != (len(ERA5_VARIABLES), split_times.size):
                        errors.append(f"Malformed {data_key} in {raw_path}")
                        continue
                    if scaled_values.shape != raw_values.shape:
                        errors.append(f"Scaled shape mismatch in {scaled_path}")
                        continue
                    if raw_values.dtype != np.float32 or scaled_values.dtype != np.float32:
                        errors.append(f"Payload dtype is not float32 in {raw_path} or {scaled_path}")
                    if not np.array_equal(split_times, scaled_times):
                        errors.append(f"Scaled time mismatch in {scaled_path}")
                    if not np.all(np.isfinite(raw_values)) or not np.all(np.isfinite(scaled_values)):
                        errors.append(f"Non-finite values in {raw_path} or {scaled_path}")
                    expected_scaled = scaler.transform(raw_values.T).T
                    scaling_error = float(np.max(np.abs(expected_scaled - scaled_values)))
                    task_max_scaling_error = max(task_max_scaling_error, scaling_error)
                    max_scaling_error = max(max_scaling_error, scaling_error)
                    if task_id >= min(EXTENSION_TASKS):
                        generated_max_scaling_error = max(
                            generated_max_scaling_error,
                            scaling_error,
                        )
                    observed_times.append(split_times)
            if observed_times:
                combined_times = np.sort(np.concatenate(observed_times))
                if not np.array_equal(combined_times, expected_times):
                    errors.append(f"Incomplete or duplicate synthetic time indices in {raw_path}")

        task_summaries.append(
            {
                "task": _task_name(task_id),
                "unscaled_files": len(raw_files),
                "scaled_files": len(scaled_files),
                "locations": len(raw_map),
                "hours": TASK_LENGTH,
                "max_scaling_error": task_max_scaling_error,
            }
        )

    reference_copy_differences: list[str] = []
    for relative_root in (Path("task_1"), Path("task_2")):
        source_files = {
            path.relative_to(reference_root)
            for path in (reference_root / relative_root).rglob("*")
            if path.is_file()
        }
        copied_files = {
            path.relative_to(output_root)
            for path in (output_root / relative_root).rglob("*")
            if path.is_file()
        }
        for relative_path in sorted(source_files | copied_files):
            source = reference_root / relative_path
            copied = output_root / relative_path
            if not source.exists() or not copied.exists() or _sha256(source) != _sha256(copied):
                reference_copy_differences.append(str(relative_path))
    if reference_copy_differences:
        errors.append("Task 1--2 are not byte-for-byte copies of the preserved reference")

    physical_times = np.load(output_root / "provenance" / "physical_valid_times.npy")
    coordinates = np.load(output_root / "provenance" / "location_coordinates.npy")
    if physical_times.shape != (TASK_LENGTH * 10, len(expected_locations)):
        errors.append(f"Unexpected physical timestamp sidecar shape {physical_times.shape}")
    if coordinates.shape != (len(expected_locations), 2) or not np.allclose(
        coordinates, np.asarray(expected_locations)
    ):
        errors.append("Location-coordinate sidecar does not match preserved Task 1")
    affected_locations, max_dropped_rows = _physical_time_compaction_summary(physical_times)

    streaming_manifest = json.loads(
        (output_root / "long_streaming_manifest.json").read_text(encoding="utf-8")
    )["streaming"]
    task_aware = streaming_manifest["task_aware_blocks"]
    continuous = streaming_manifest["continuous_10_hour_blocks"]
    if task_aware["count"] != 171 or sum(block["hours"] for block in task_aware["blocks"]) != 1674:
        errors.append("Task-aware streaming manifest is inconsistent")
    if continuous["count"] != 168 or sum(block["hours"] for block in continuous["blocks"]) != 1674:
        errors.append("Continuous streaming manifest is inconsistent")

    if generated_max_scaling_error > 1e-7:
        errors.append(
            "Generated scaled payloads differ from this run's Task-1 scaler transform by "
            f"{generated_max_scaling_error:.3e}"
        )
    report = {
        "created_at_utc": _utc_now(),
        "status": "passed" if not errors else "failed",
        "output_root": str(output_root.resolve()),
        "task_summaries": task_summaries,
        "total_unscaled_values": len(ERA5_VARIABLES) * TASK_LENGTH * len(expected_locations) * 10,
        "model_loader_expected_shapes": {
            "times": [TASK_LENGTH * 10],
            "coords": [len(expected_locations), 2],
            "Y": [TASK_LENGTH * 10, len(expected_locations)],
        },
        "max_scaling_error": max_scaling_error,
        "generated_task_max_scaling_error": generated_max_scaling_error,
        "reference_scaling_note": (
            "Task 1--2 are validated by byte identity. Their small transform deviation can reflect "
            "replaying a scikit-learn 1.6.1 scaler under the current scikit-learn runtime."
        ),
        "task_1_task_2_byte_identical": not reference_copy_differences,
        "reference_copy_differences": reference_copy_differences[:20],
        "physical_time_sidecar_shape": list(physical_times.shape),
        "locations_with_compacted_rows": affected_locations,
        "max_dropped_rows_before_horizon": max_dropped_rows,
        "task_aware_streaming_blocks": task_aware["count"],
        "continuous_streaming_blocks": continuous["count"],
        "errors": errors[:100],
    }
    _write_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise RuntimeError(f"Extension verification failed with {len(errors)} errors")
    return report


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("data/era5/processed_timeseries_4"),
        help="Preserved public Task 1--2 root; this directory is never modified.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/era5/raw_task1_10_reconstruction"),
        help="Independent directory for downloaded CDS NetCDF files and manifest.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Retrieve a bounded raw ERA5-Land time slice from CDS.")
    _add_common_paths(download)
    download.add_argument("--start", default="2020-01-01T00:00:00")
    download.add_argument("--hours", type=int, default=TASK_LENGTH * 10)
    download.add_argument("--area", type=_parse_area, default=DEFAULT_AREA)
    download.add_argument(
        "--days-per-request",
        type=int,
        default=5,
        help="Maximum valid calendar days per CDS request; smaller values avoid CDS cost limits.",
    )
    download.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of concurrent independent CDS chunks; use conservatively to respect service load.",
    )
    download.add_argument(
        "--point",
        nargs=2,
        type=float,
        metavar=("LAT", "LON"),
        help="Download a one-grid-cell probe instead of the full UK bounding box.",
    )
    download.add_argument("--force", action="store_true", help="Replace an existing raw monthly file.")

    validate = subparsers.add_parser(
        "validate", help="Compare raw CDS values to preserved unscaled Task 1--2 arrays."
    )
    _add_common_paths(validate)
    validate.add_argument("--tasks", nargs="+", type=int, default=list(REFERENCE_TASKS))
    validate.add_argument(
        "--location",
        nargs=2,
        type=float,
        action="append",
        metavar=("LAT", "LON"),
        help="Restrict validation to one or more preserved locations (repeatable).",
    )
    validate.add_argument(
        "--max-hours",
        type=int,
        help=(
            "Compare only an initial probe prefix. This is useful for checking the CDS time "
            "origin, but cannot unlock Task 3--10 generation."
        ),
    )
    validate.add_argument("--atol", type=float, default=1e-6)
    validate.add_argument("--rtol", type=float, default=1e-6)
    validate.add_argument(
        "--report",
        type=Path,
        default=Path("data/era5/raw_task1_10_reconstruction/task1_task2_validation.json"),
    )

    generate = subparsers.add_parser(
        "generate", help="Create a validated Task 1--10 extension without changing the original root."
    )
    _add_common_paths(generate)
    generate.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/era5/processed_timeseries_4_task1_10_extension"),
    )
    generate.add_argument(
        "--validation-report",
        type=Path,
        default=Path("data/era5/raw_task1_10_reconstruction/task1_task2_validation.json"),
    )
    generate.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)

    verify = subparsers.add_parser("verify", help="Audit a generated Task 1--10 extension.")
    _add_common_paths(verify)
    verify.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/era5/processed_timeseries_4_task1_10_extension"),
    )
    verify.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/era5/processed_timeseries_4_task1_10_extension/verification_report.json"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "download":
        area = tuple(args.area)
        if args.point is not None:
            latitude, longitude = args.point
            area = (latitude, longitude, latitude, longitude)
        download_raw_slice(
            args.raw_dir,
            start=_parse_datetime(args.start),
            hours=args.hours,
            area=area,
            days_per_request=args.days_per_request,
            max_workers=args.max_workers,
            force=args.force,
        )
    elif args.command == "validate":
        validate_reference(
            args.reference_root,
            args.raw_dir,
            task_ids=args.tasks,
            selected_locations=(
                [tuple(location) for location in args.location] if args.location is not None else None
            ),
            max_hours=args.max_hours,
            atol=args.atol,
            rtol=args.rtol,
            report_path=args.report,
        )
    elif args.command == "generate":
        generate_extension(
            args.reference_root,
            args.raw_dir,
            args.output_root,
            args.validation_report,
            split_seed=args.split_seed,
        )
    elif args.command == "verify":
        verify_extension(args.reference_root, args.output_root, args.report)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
