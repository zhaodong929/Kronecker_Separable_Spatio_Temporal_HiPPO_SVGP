#!/usr/bin/env python
"""Run Route B internal methods on the processed HiPPO-SVGP ERA5 subset.

The script is intentionally separate from the baseline runner. It reuses the
same processed ERA5 loader, location-selection arguments, block split, scaled
targets, and metric definitions, then writes Route B results plus combined
baseline-vs-Route-B tables into the baseline output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stvgp_kronecker.data.hipposvgp_era5 import (
    SPLIT_KEYS,
    HippoERA5Dataset,
    _split_names,
    load_hipposvgp_era5,
    to_routeb_synthetic_dataset,
)
from stvgp_kronecker.joint_ssgp_kron.kron_utils import add_jitter, dense_A_from_factors, dense_Du_for_tests, inv_spd, solve_spd, symmetrize, vec_f
from stvgp_kronecker.joint_ssgp_kron.kron_utils import unvec_f
from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    BlockFactors,
    covariance_kernel,
    make_block_factors_analytic_hippo,
    make_block_factors,
    make_spatial_projection,
    temporal_inducing_for_block,
)


Z90 = 1.6448536269514722
METHOD_LABELS = {
    "no_transfer": "no_transfer",
    "mean_field": "mean_field",
    "structured_joint": "structured_joint",
}


def _load_spectral_mixture_params(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"spectral mixture parameter file not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload


def _default_spectral_mixture_params(num_mixtures: int) -> dict[str, list[float]]:
    q = max(int(num_mixtures), 1)
    weights = np.ones(q, dtype=float) / float(q)
    means = np.linspace(0.0, 2.0, q, dtype=float)
    scales = np.geomspace(0.25, 1.0, q).astype(float)
    return {
        "weights": weights.tolist(),
        "means": means.tolist(),
        "scales": scales.tolist(),
    }


def _sm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    params = getattr(args, "spectral_mixture_params", None) or {}
    defaults = _default_spectral_mixture_params(getattr(args, "num_mixtures", 4))
    return {
        "spectral_mixture_weights": params.get("temporal_weights", defaults["weights"]),
        "spectral_mixture_means": params.get("temporal_means", defaults["means"]),
        "spectral_mixture_scales": params.get("temporal_scales", defaults["scales"]),
    }


def _sm_spatial_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    params = getattr(args, "spectral_mixture_params", None) or {}
    defaults = _default_spectral_mixture_params(getattr(args, "num_mixtures", 4))
    return {
        "spectral_mixture_weights": params.get("spatial_weights", defaults["weights"]),
        "spectral_mixture_means": params.get("spatial_means", defaults["means"]),
        "spectral_mixture_scales": params.get("spatial_scales", defaults["scales"]),
    }

ERA5_VARIABLE_NAMES = {
    0: "2m_dewpoint_temperature",
    1: "2m_temperature",
    2: "skin_temperature",
    3: "soil_temperature_level_1",
    4: "soil_temperature_level_2",
    5: "soil_temperature_level_3",
    6: "soil_temperature_level_4",
    7: "skin_reservoir_content",
    8: "volumetric_soil_water_layer_1",
    9: "volumetric_soil_water_layer_2",
    10: "volumetric_soil_water_layer_3",
    11: "volumetric_soil_water_layer_4",
    12: "forecast_albedo",
    13: "surface_latent_heat_flux",
    14: "surface_net_solar_radiation",
    15: "surface_net_thermal_radiation",
    16: "surface_sensible_heat_flux",
    17: "surface_solar_radiation_downwards",
    18: "surface_thermal_radiation_downwards",
    19: "evaporation_from_bare_soil",
    20: "evaporation_from_open_water_surfaces_excluding_oceans",
    21: "evaporation_from_the_top_of_canopy",
    22: "evaporation_from_vegetation_transpiration",
    23: "potential_evaporation",
    24: "runoff",
    25: "snow_evaporation",
    26: "sub_surface_runoff",
    27: "surface_runoff",
    28: "total_evaporation",
    29: "10m_u_component_of_wind",
    30: "10m_v_component_of_wind",
    31: "surface_pressure",
    32: "total_precipitation",
    33: "leaf_area_index_high_vegetation",
    34: "leaf_area_index_low_vegetation",
}

SURFACE_METEOROLOGY_INDICES = (1, 2, 29, 30, 31, 32)
STATIC_LAND_PROXY_INDICES = (33, 34)


def spatial_kernel_lengthscale(args: argparse.Namespace) -> float | np.ndarray:
    if args.kernel_type == "ard_rbf":
        if args.spatial_ard_lengthscales is None:
            return np.asarray([args.spatial_lengthscale, args.spatial_lengthscale], dtype=float)
        vals = np.asarray(args.spatial_ard_lengthscales, dtype=float)
        if vals.shape != (2,):
            raise ValueError("--spatial-ard-lengthscales must contain exactly two values: lat lon")
        return vals
    return float(args.spatial_lengthscale)


def _parse_snapshot_indices(values: list[int] | None, num_time: int) -> set[int]:
    if values is None:
        values = [0, -1]
    out: set[int] = set()
    for value in values:
        idx = int(value)
        if idx < 0:
            idx = num_time + idx
        if 0 <= idx < num_time:
            out.add(idx)
    return out


def gaussian_nll(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    var = np.maximum(np.asarray(var, dtype=float), 1e-10)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * var) + (np.asarray(y) - mean) ** 2 / var))


def coverage90(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    half = Z90 * np.sqrt(np.maximum(var, 1e-10))
    return float(np.mean((y >= mean - half) & (y <= mean + half)))


def ece_gaussian(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    levels = np.asarray([0.5, 0.8, 0.9, 0.95])
    z_values = np.asarray([0.67448975, 1.28155157, 1.64485363, 1.95996398])
    sigma = np.sqrt(np.maximum(var, 1e-10))
    errors = []
    for level, z in zip(levels, z_values):
        half = z * sigma
        empirical = np.mean((y >= mean - half) & (y <= mean + half))
        errors.append(abs(float(empirical) - float(level)))
    return float(np.mean(errors))


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_map_snapshot(
    *,
    outdir: Path,
    coords: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    method: str,
    eval_mode: str,
    task_label: str,
    t_index: int,
    time_value: float,
    variable_label: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    lat = coords[:, 0]
    lon = coords[:, 1]
    err = mean - y
    vmin = float(np.nanmin([np.nanmin(y), np.nanmin(mean)]))
    vmax = float(np.nanmax([np.nanmax(y), np.nanmax(mean)]))
    err_abs = float(max(np.nanmax(np.abs(err)), 1e-8))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    for ax, values, title, cmap, lo, hi in [
        (axes[0], y, "Ground Truth", "coolwarm", vmin, vmax),
        (axes[1], mean, "Prediction", "coolwarm", vmin, vmax),
        (axes[2], err, "Error", "RdBu_r", -err_abs, err_abs),
    ]:
        sc = ax.scatter(lon, lat, c=values, cmap=cmap, vmin=lo, vmax=hi, s=22, alpha=0.9, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.18)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"{variable_label} | {method} | {eval_mode} | {task_label} | t_index={t_index} | time={time_value:.3f}")
    fig.savefig(outdir / f"{method}_{eval_mode}_{task_label}_t{t_index:03d}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    sc = ax.scatter(lon, lat, c=mean, cmap="coolwarm", vmin=vmin, vmax=vmax, s=24, alpha=0.9, edgecolors="none")
    ax.set_title(f"Prediction mean | {method} | t_index={t_index}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.18)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(outdir / f"{method}_{eval_mode}_{task_label}_mean_only_t{t_index:03d}.png", dpi=180)
    plt.close(fig)

    np.savez(
        outdir / f"{method}_{eval_mode}_{task_label}_snapshot_t{t_index:03d}.npz",
        coords=coords,
        y=y,
        mean=mean,
        variance=var,
        error=err,
        t_index=np.asarray(t_index),
        time_value=np.asarray(time_value),
    )


def append_per_location_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    method: str,
    eval_mode: str,
    block_id: int,
    eval_block: slice,
    y: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    dataset_raw: HippoERA5Dataset,
    dataset: HippoERA5Dataset,
    args: argparse.Namespace,
    seed: int,
    sigma2: float,
    ell_t: float,
    future_basis_applied: str,
    location_indices: np.ndarray | None = None,
) -> None:
    """Append pointwise predictions in time-major rows for later diagnostics."""

    eval_start = eval_block.start or 0
    eval_stop = eval_block.stop or dataset.Y.shape[0]
    t_count = eval_stop - eval_start
    if location_indices is None:
        location_indices = np.arange(dataset.coords.shape[0], dtype=int)
    location_indices = np.asarray(location_indices, dtype=int)
    s_count = int(location_indices.size)
    if y.size != s_count * t_count:
        raise ValueError(
            f"Cannot save per-location predictions for {method}/{eval_mode}: "
            f"got y.size={y.size}, expected {s_count * t_count}."
        )
    y_mat = np.asarray(y, dtype=float).reshape((s_count, t_count), order="F").T
    mean_mat = np.asarray(mean, dtype=float).reshape((s_count, t_count), order="F").T
    var_mat = np.maximum(np.asarray(var, dtype=float), 1e-10).reshape((s_count, t_count), order="F").T
    requested_locations = getattr(args, "per_location_indices", None)
    if requested_locations:
        requested = {int(idx) for idx in requested_locations}
        keep = np.asarray([i for i, idx in enumerate(location_indices) if int(idx) in requested], dtype=int)
        if keep.size == 0:
            return
        location_indices = location_indices[keep]
        y_mat = y_mat[:, keep]
        mean_mat = mean_mat[:, keep]
        var_mat = var_mat[:, keep]
    coords = np.asarray(dataset_raw.coords, dtype=float)[location_indices]
    raw_times = np.asarray(dataset_raw.times, dtype=float)
    origin_block = block_id if eval_mode == "future" else ""
    future_block_id = block_id + 1 if eval_mode == "future" else ""
    for local_t, time_index in enumerate(range(eval_start, eval_stop)):
        horizon = local_t + 1 if eval_mode == "future" else ""
        for local_s, location_index in enumerate(location_indices):
            pred_var = float(var_mat[local_t, local_s])
            rows.append(
                {
                    "eval_mode": eval_mode,
                    "method": method,
                    "phi_mode": args.phi_mode,
                    "basis_mode": future_basis_applied if eval_mode == "future" else "observed",
                    "block_id": block_id,
                    "train_block_id": block_id,
                    "eval_start": eval_start,
                    "eval_stop": eval_stop,
                    "origin_block": origin_block,
                    "future_block_id": future_block_id,
                    "horizon": horizon,
                    "time_index": time_index,
                    "actual_time": float(raw_times[time_index]),
                    "location_index": int(location_index),
                    "latitude": float(coords[local_s, 0]),
                    "longitude": float(coords[local_s, 1]) if coords.shape[1] > 1 else float("nan"),
                    "y_true": float(y_mat[local_t, local_s]),
                    "pred_mean": float(mean_mat[local_t, local_s]),
                    "pred_var_y": pred_var,
                    "pred_std_y": float(np.sqrt(pred_var)),
                    "observation_noise_variance": float(sigma2),
                    "seed": int(seed),
                    "block_size": int(args.block_size),
                    "mt": int(args.mt),
                    "ms": int(args.ms),
                    "model_ell_t": float(ell_t),
                    "kernel_variance": float(args.kernel_variance),
                    "tasks": ",".join(dataset.tasks),
                    "calibration_tasks": ",".join(getattr(args, "calibration_tasks", [])),
                    "online_tasks": ",".join(getattr(args, "online_tasks", args.tasks)),
                }
            )


def lag_ar_columns_for_times(Y: np.ndarray, times_idx: range, *, future_context_stop: int | None = None) -> np.ndarray:
    """Return lag AR columns in time-major order without future-label leakage.

    When `future_context_stop` is provided, all requested rows use only
    `Y[future_context_stop-1]` and `Y[future_context_stop-2]` as context. This is
    used for future-block diagnostics so labels inside the future block are not
    read to build features.
    """

    T, S = Y.shape
    cols = []
    for t_idx in times_idx:
        if future_context_stop is None:
            lag1_idx = max(t_idx - 1, 0)
            lag2_idx = max(t_idx - 2, 0)
        else:
            lag1_idx = max(min(future_context_stop - 1, T - 1), 0)
            lag2_idx = max(min(future_context_stop - 2, T - 1), 0)
        lag1 = Y[lag1_idx]
        lag2 = Y[lag2_idx]
        diff = lag1 - lag2
        cols.append(np.column_stack([lag1, lag2, diff]))
    return np.vstack(cols)


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0, keepdims=True)
    scale = np.maximum(values.std(axis=0, keepdims=True), 1e-8)
    return (values - mean) / scale


def _time_space_tiles(dataset: HippoERA5Dataset) -> dict[str, np.ndarray]:
    times = np.asarray(dataset.times, dtype=float).reshape(-1)
    coords = np.asarray(dataset.coords, dtype=float)
    t_span = max(float(times[-1] - times[0]), 1e-12)
    t_scaled = (times - times[0]) / t_span
    t_centered = t_scaled - float(np.mean(t_scaled))
    coords_scaled = _standardize_columns(coords)
    lat = coords_scaled[:, 0]
    lon = coords_scaled[:, 1]
    phase = 2.0 * np.pi * t_scaled

    S = coords.shape[0]
    t_rep = np.repeat(t_centered, S)
    t2_rep = np.repeat(t_centered**2, S)
    lat_tile = np.tile(lat, times.shape[0])
    lon_tile = np.tile(lon, times.shape[0])
    lat2_tile = np.tile(lat**2, times.shape[0])
    lon2_tile = np.tile(lon**2, times.shape[0])
    latlon_tile = np.tile(lat * lon, times.shape[0])
    sin1 = np.repeat(np.sin(phase), S)
    cos1 = np.repeat(np.cos(phase), S)
    sin2 = np.repeat(np.sin(2.0 * phase), S)
    cos2 = np.repeat(np.cos(2.0 * phase), S)
    sin3 = np.repeat(np.sin(3.0 * phase), S)
    cos3 = np.repeat(np.cos(3.0 * phase), S)
    sin4 = np.repeat(np.sin(4.0 * phase), S)
    cos4 = np.repeat(np.cos(4.0 * phase), S)
    return {
        "times": times,
        "coords": coords,
        "coords_scaled": coords_scaled,
        "t_rep": t_rep,
        "t2_rep": t2_rep,
        "lat_tile": lat_tile,
        "lon_tile": lon_tile,
        "lat2_tile": lat2_tile,
        "lon2_tile": lon2_tile,
        "latlon_tile": latlon_tile,
        "sin1": sin1,
        "cos1": cos1,
        "sin2": sin2,
        "cos2": cos2,
        "sin3": sin3,
        "cos3": cos3,
        "sin4": sin4,
        "cos4": cos4,
    }


def minimal_columns(dataset: HippoERA5Dataset) -> tuple[np.ndarray, list[str]]:
    """Minimal climatology Phi without a linear time trend."""

    z = _time_space_tiles(dataset)
    columns = np.column_stack(
        [
            np.ones(dataset.Y.size),
            z["sin1"],
            z["cos1"],
            z["sin2"],
            z["cos2"],
            z["lat_tile"],
            z["lon_tile"],
        ]
    )
    names = [
        "1",
        "sin_doy",
        "cos_doy",
        "sin_2doy",
        "cos_2doy",
        "lat_scaled",
        "lon_scaled",
    ]
    metadata = dict(dataset.metadata or {})
    missing_static = []
    if not metadata.get("elevation_available", False):
        missing_static.append("elevation")
    if not metadata.get("land_sea_mask_available", False):
        missing_static.append("land_sea_mask")
    if missing_static:
        metadata["missing_requested_static_features"] = missing_static
    return columns, names


def rich_v1_columns(dataset: HippoERA5Dataset) -> tuple[np.ndarray, list[str]]:
    """Current rich Phi without explicit seasonal-spatial interaction terms."""

    z = _time_space_tiles(dataset)
    columns = np.column_stack(
        [
            z["t2_rep"],
            z["lat2_tile"],
            z["lon2_tile"],
            z["latlon_tile"],
            z["sin3"],
            z["cos3"],
            z["sin4"],
            z["cos4"],
        ]
    )
    names = [
        "time_centered_sq",
        "lat_scaled_sq",
        "lon_scaled_sq",
        "lat_lon_interaction",
        "sin_3time",
        "cos_3time",
        "sin_4time",
        "cos_4time",
    ]
    return columns, names


def rich_seasonal_spatial_columns(dataset: HippoERA5Dataset) -> tuple[np.ndarray, list[str]]:
    """Non-leaking richer Phi columns from time and coordinates only."""

    z = _time_space_tiles(dataset)
    lat_tile = z["lat_tile"]
    lon_tile = z["lon_tile"]
    sin1 = z["sin1"]
    cos1 = z["cos1"]
    sin2 = z["sin2"]
    cos2 = z["cos2"]
    t_rep = z["t_rep"]

    columns = np.column_stack(
        [
            z["t2_rep"],
            z["lat2_tile"],
            z["lon2_tile"],
            z["latlon_tile"],
            z["sin3"],
            z["cos3"],
            z["sin4"],
            z["cos4"],
            t_rep * lat_tile,
            t_rep * lon_tile,
            sin1 * lat_tile,
            cos1 * lat_tile,
            sin1 * lon_tile,
            cos1 * lon_tile,
            sin2 * lat_tile,
            cos2 * lat_tile,
            sin2 * lon_tile,
            cos2 * lon_tile,
        ]
    )
    names = [
        "time_centered_sq",
        "lat_scaled_sq",
        "lon_scaled_sq",
        "lat_lon_interaction",
        "sin_3time",
        "cos_3time",
        "sin_4time",
        "cos_4time",
        "time_lat_interaction",
        "time_lon_interaction",
        "sin_time_lat",
        "cos_time_lat",
        "sin_time_lon",
        "cos_time_lon",
        "sin_2time_lat",
        "cos_2time_lat",
        "sin_2time_lon",
        "cos_2time_lon",
    ]
    return columns, names


def _read_multivar_one_file(path: Path, variable_indices: Sequence[int], split: str) -> tuple[np.ndarray, np.ndarray]:
    times_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    with np.load(path, allow_pickle=True) as data:
        for split_name in _split_names(split):
            data_key, time_key = SPLIT_KEYS[split_name]
            values = np.asarray(data[data_key][list(variable_indices)], dtype=float)
            times = np.asarray(data[time_key], dtype=float).reshape(-1)
            if values.shape[1] != times.shape[0]:
                raise ValueError(f"{path.name}: {data_key} and {time_key} lengths do not match")
            order = np.argsort(times)
            times_chunks.append(times[order])
            value_chunks.append(values[:, order].T)
    times_full = np.concatenate(times_chunks)
    values_full = np.concatenate(value_chunks, axis=0)
    order = np.argsort(times_full)
    return times_full[order], values_full[order]


def _read_aligned_multivar_for_selected_files(
    dataset: HippoERA5Dataset,
    variable_indices: Sequence[int],
) -> np.ndarray:
    if not variable_indices:
        return np.empty((dataset.Y.shape[0], dataset.Y.shape[1], 0), dtype=float)
    split = str((dataset.metadata or {}).get("split", "all"))
    series = []
    metadata = dict(dataset.metadata or {})
    align_times = np.asarray(dataset.times, dtype=float)
    if "routeb_raw_time_scale" in metadata and "routeb_raw_time_start" in metadata:
        align_times = align_times * float(metadata["routeb_raw_time_scale"]) + float(metadata["routeb_raw_time_start"])
    for path in dataset.selected_files:
        times, values = _read_multivar_one_file(path, variable_indices, split)
        lookup = {float(t): values[i] for i, t in enumerate(times)}
        try:
            aligned = np.asarray([lookup[float(t)] for t in align_times], dtype=float)
        except KeyError as exc:
            raise ValueError(f"{path.name}: missing aligned timestamp {exc}") from exc
        series.append(aligned)
    # [S,T,V] -> [T,S,V]
    return np.stack(series, axis=0).transpose(1, 0, 2)


def _standardize_3d_columns(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    flat = arr.reshape((-1, arr.shape[-1]))
    flat = _standardize_columns(flat)
    return flat.reshape(arr.shape)


def era5_covariate_columns(
    dataset: HippoERA5Dataset,
    variable_indices: Sequence[int],
    *,
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    values = _read_aligned_multivar_for_selected_files(dataset, variable_indices)
    if values.shape[-1] == 0:
        return np.empty((dataset.Y.size, 0), dtype=float), []
    values = _standardize_3d_columns(values)
    names = [f"{prefix}_{ERA5_VARIABLE_NAMES.get(int(idx), f'var_{idx}')}" for idx in variable_indices]
    return values.reshape((dataset.Y.size, values.shape[-1])), names


def era5_xlag_covariate_columns(
    dataset: HippoERA5Dataset,
    variable_indices: Sequence[int],
    *,
    prefix: str,
    lag_length: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """Current, lagged and differenced exogenous ERA5 covariates."""

    values = _read_aligned_multivar_for_selected_files(dataset, variable_indices)
    if values.shape[-1] == 0:
        return np.empty((dataset.Y.size, 0), dtype=float), []
    lag_length = max(int(lag_length), 1)
    lagged_values = []
    differenced_values = []
    time_index = np.arange(values.shape[0])
    for lag in range(1, lag_length + 1):
        lag_idx = np.maximum(time_index - lag, 0)
        lagged = values[lag_idx]
        lagged_values.append(lagged)
        differenced_values.append(values - lagged)
    stacked = np.concatenate([values, *lagged_values, *differenced_values], axis=-1)
    stacked = _standardize_3d_columns(stacked)
    base_names = [ERA5_VARIABLE_NAMES.get(int(idx), f"var_{idx}") for idx in variable_indices]
    names = (
        [f"{prefix}_current_{name}" for name in base_names]
        + [f"{prefix}_lag{lag}_{name}" for lag in range(1, lag_length + 1) for name in base_names]
        + [
            f"{prefix}_diff_current_lag{lag}_{name}"
            for lag in range(1, lag_length + 1)
            for name in base_names
        ]
    )
    return stacked.reshape((dataset.Y.size, stacked.shape[-1])), names


def era5_pca_columns(
    dataset: HippoERA5Dataset,
    variable_indices: Sequence[int],
    *,
    n_components: int,
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    values = _read_aligned_multivar_for_selected_files(dataset, variable_indices)
    if values.shape[-1] == 0 or n_components <= 0:
        return np.empty((dataset.Y.size, 0), dtype=float), []
    flat = _standardize_columns(values.reshape((-1, values.shape[-1])))
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    n_components = min(int(n_components), vt.shape[0])
    scores = flat @ vt[:n_components].T
    scores = _standardize_columns(scores)
    names = [f"{prefix}_pc{i + 1}" for i in range(n_components)]
    return scores, names


def medium_era5_columns(dataset: HippoERA5Dataset) -> tuple[np.ndarray, list[str]]:
    lag_cols = lag_ar_columns_for_times(dataset.Y, range(dataset.Y.shape[0]))
    lag_names = ["lag_y_t_minus_1", "lag_y_t_minus_2", "lag_y_diff_1_2"]
    covariate_indices = [idx for idx in SURFACE_METEOROLOGY_INDICES if idx != dataset.variable_index]
    cov_cols, cov_names = era5_covariate_columns(dataset, covariate_indices, prefix="surface")
    return np.column_stack([lag_cols, cov_cols]), lag_names + cov_names


def medium_era5_xlag_columns(dataset: HippoERA5Dataset, *, lag_length: int = 1) -> tuple[np.ndarray, list[str]]:
    covariate_indices = [idx for idx in SURFACE_METEOROLOGY_INDICES if idx != dataset.variable_index]
    return era5_xlag_covariate_columns(dataset, covariate_indices, prefix="surface_x", lag_length=lag_length)


def rich_era5_columns(dataset: HippoERA5Dataset, *, n_components: int = 6) -> tuple[np.ndarray, list[str]]:
    medium_cols, medium_names = medium_era5_columns(dataset)
    used = {dataset.variable_index, *SURFACE_METEOROLOGY_INDICES}
    pca_indices = [idx for idx in ERA5_VARIABLE_NAMES if idx not in used]
    pca_cols, pca_names = era5_pca_columns(dataset, pca_indices, n_components=n_components, prefix="era5_remaining")
    return np.column_stack([medium_cols, pca_cols]), medium_names + pca_names


def rich_v3_rbf_columns(dataset: HippoERA5Dataset, *, n_centers: int = 8) -> tuple[np.ndarray, list[str]]:
    """Rich v2 plus non-leaking spatial RBF region features."""

    rich_cols, rich_names = rich_seasonal_spatial_columns(dataset)
    z = _time_space_tiles(dataset)
    coords_scaled = np.asarray(z["coords_scaled"], dtype=float)
    if coords_scaled.shape[0] == 0:
        raise ValueError("Cannot build RBF Phi columns without spatial coordinates")
    n_centers = min(int(n_centers), coords_scaled.shape[0])
    center_idx = np.linspace(0, coords_scaled.shape[0] - 1, n_centers, dtype=int)
    centers = coords_scaled[center_idx]
    diff = coords_scaled[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff**2, axis=2)
    nonzero = dist2[dist2 > 0.0]
    bandwidth2 = float(np.median(nonzero)) if nonzero.size else 1.0
    bandwidth2 = max(bandwidth2, 1e-6)
    rbf_space = np.exp(-0.5 * dist2 / bandwidth2)
    rbf_space = _standardize_columns(rbf_space)
    rbf_cols = np.tile(rbf_space, (np.asarray(dataset.times).shape[0], 1))
    names = rich_names + [f"spatial_rbf_{i}" for i in range(n_centers)]
    return np.column_stack([rich_cols, rbf_cols]), names


def augment_dataset_phi(dataset: HippoERA5Dataset, *, phi_mode: str, xlag_length: int = 1) -> HippoERA5Dataset:
    if phi_mode == "direct_y":
        metadata = dict(dataset.metadata or {})
        metadata["phi_mode"] = "direct_y"
        metadata["phi_columns"] = []
        return replace(dataset, Phi=np.zeros((dataset.Y.size, 0), dtype=float), metadata=metadata)
    if phi_mode == "minimal":
        phi, names = minimal_columns(dataset)
        metadata = dict(dataset.metadata or {})
        metadata["phi_mode"] = "minimal"
        metadata["phi_columns"] = names
        metadata.setdefault("missing_requested_static_features", ["elevation", "land_sea_mask"])
        return replace(dataset, Phi=phi, metadata=metadata)
    if phi_mode == "base":
        return dataset
    if phi_mode in {"rich_v1", "rich_v2", "rich_v3", "rich_seasonal_spatial", "engineered", "rich_v3_lag_ar"}:
        base_rich_mode = "rich_v3" if phi_mode == "rich_v3_lag_ar" else phi_mode
        if base_rich_mode == "engineered":
            base_rich_mode = "rich_seasonal_spatial"
        if base_rich_mode == "rich_v1":
            rich_cols, rich_names = rich_v1_columns(dataset)
        elif base_rich_mode == "rich_v3":
            rich_cols, rich_names = rich_v3_rbf_columns(dataset)
        else:
            rich_cols, rich_names = rich_seasonal_spatial_columns(dataset)
        if phi_mode == "rich_v3_lag_ar":
            lag_cols = lag_ar_columns_for_times(dataset.Y, range(dataset.Y.shape[0]))
            rich_cols = np.column_stack([rich_cols, lag_cols])
            rich_names = rich_names + ["lag_y_t_minus_1", "lag_y_t_minus_2", "lag_y_diff_1_2"]
        phi = np.column_stack([dataset.Phi, rich_cols])
        metadata = dict(dataset.metadata or {})
        metadata["phi_mode"] = phi_mode
        metadata["phi_columns"] = list(metadata.get("phi_columns", [])) + rich_names
        return replace(dataset, Phi=phi, metadata=metadata)
    if phi_mode in {"medium_era5", "medium_era5_xlag", "medium_era5_oracle_ylag", "rich_era5"}:
        base_phi, base_names = minimal_columns(dataset)
        if phi_mode in {"medium_era5", "medium_era5_oracle_ylag"}:
            extra_cols, extra_names = medium_era5_columns(dataset)
        elif phi_mode == "medium_era5_xlag":
            extra_cols, extra_names = medium_era5_xlag_columns(dataset, lag_length=xlag_length)
        else:
            extra_cols, extra_names = rich_era5_columns(dataset)
        phi = np.column_stack([base_phi, extra_cols])
        metadata = dict(dataset.metadata or {})
        metadata["phi_mode"] = phi_mode
        metadata["phi_columns"] = base_names + extra_names
        metadata["era5_covariates_in_phi_only"] = True
        if phi_mode == "medium_era5_xlag":
            metadata["xlag_length"] = int(xlag_length)
        metadata.setdefault("missing_requested_static_features", ["elevation", "land_sea_mask"])
        if phi_mode == "rich_era5":
            metadata["rich_era5_pressure_level_status"] = "pressure-level variables unavailable in processed_timeseries_4; PCA uses remaining available ERA5 single-level variables"
        return replace(dataset, Phi=phi, metadata=metadata)
    if phi_mode != "lag_ar":
        raise ValueError(f"Unknown phi_mode: {phi_mode}")
    lag_cols = lag_ar_columns_for_times(dataset.Y, range(dataset.Y.shape[0]))
    phi = np.column_stack([dataset.Phi, lag_cols])
    metadata = dict(dataset.metadata or {})
    metadata["phi_mode"] = "lag_ar"
    metadata["phi_columns"] = list(metadata.get("phi_columns", [])) + ["lag_y_t_minus_1", "lag_y_t_minus_2", "lag_y_diff_1_2"]
    return replace(dataset, Phi=phi, metadata=metadata)


def replace_factors_phi(factors: BlockFactors, phi: np.ndarray) -> BlockFactors:
    return replace(factors, Phi=phi)


def phi_for_eval_block(dataset: HippoERA5Dataset, block: slice, *, phi_mode: str, future_context_stop: int | None = None) -> np.ndarray:
    base_phi = phi_for_slice(dataset, block)
    if phi_mode in {
        "base",
        "minimal",
        "rich_v1",
        "rich_v2",
        "rich_v3",
        "rich_seasonal_spatial",
        "engineered",
        "medium_era5",
        "medium_era5_xlag",
        "medium_era5_oracle_ylag",
        "rich_era5",
    }:
        return base_phi
    if phi_mode == "rich_v3_lag_ar":
        # The dataset has already been augmented with rich-v3 and lag columns.
        # This mode is primarily used by the baseline runner; future no-leakage
        # lag construction should use the explicit lag_ar diagnostic path.
        return base_phi
    start = block.start or 0
    stop = block.stop or dataset.Y.shape[0]
    lag_cols = lag_ar_columns_for_times(dataset.Y, range(start, stop), future_context_stop=future_context_stop)
    return np.column_stack([base_phi, lag_cols])


def _safe_lag_phi_template(dataset: HippoERA5Dataset, block: slice, spatial_indices: np.ndarray) -> np.ndarray:
    """Return Phi rows for a held-out spatial subset in time-major order."""

    spatial_indices = np.asarray(spatial_indices, dtype=int)
    start = block.start or 0
    stop = block.stop or dataset.Y.shape[0]
    s_count_full = dataset.Y.shape[1]
    row_idx: list[int] = []
    for t_idx in range(start, stop):
        row_idx.extend((t_idx * s_count_full + spatial_indices).tolist())
    return np.asarray(dataset.Phi[np.asarray(row_idx, dtype=int)], dtype=float)


def _safe_lag_column_indices(dataset: HippoERA5Dataset) -> tuple[int, int, int] | None:
    names = list((dataset.metadata or {}).get("phi_columns", []))
    required = ["lag_y_t_minus_1", "lag_y_t_minus_2", "lag_y_diff_1_2"]
    try:
        return tuple(names.index(name) for name in required)  # type: ignore[return-value]
    except ValueError:
        return None


def _lag_noise_column_indices(dataset: HippoERA5Dataset) -> list[int]:
    names = list((dataset.metadata or {}).get("phi_columns", []))
    return [i for i, name in enumerate(names) if name in {"lag_y_t_minus_1", "lag_y_t_minus_2", "lag_y_diff_1_2"}]


def maybe_add_lag_training_noise(dataset: HippoERA5Dataset, *, args: argparse.Namespace, seed: int) -> HippoERA5Dataset:
    std = float(getattr(args, "lag_train_noise_std", 0.0) or 0.0)
    if std <= 0.0:
        return dataset
    cols = _lag_noise_column_indices(dataset)
    if not cols:
        return dataset
    rng = np.random.default_rng(int(seed) + int(getattr(args, "lag_train_noise_seed_offset", 10000)))
    phi = np.asarray(dataset.Phi, dtype=float).copy()
    phi[:, cols] += rng.normal(0.0, std, size=(phi.shape[0], len(cols)))
    metadata = dict(dataset.metadata or {})
    metadata["lag_train_noise_std"] = std
    metadata["lag_train_noise_columns"] = [list((dataset.metadata or {}).get("phi_columns", []))[i] for i in cols]
    return replace(dataset, Phi=phi, metadata=metadata)


def beta_prior_cov_for_dataset(dataset: HippoERA5Dataset, *, args: argparse.Namespace) -> np.ndarray:
    dim = int(dataset.Phi.shape[1])
    cov = float(args.beta_prior_variance) * np.eye(dim)
    lag_var = getattr(args, "lag_beta_prior_variance", None)
    if lag_var is None:
        return cov
    lag_var = float(lag_var)
    if lag_var <= 0.0:
        return cov
    for idx in _lag_noise_column_indices(dataset):
        cov[idx, idx] = lag_var
    return cov


def _replace_safe_lag_columns(
    phi: np.ndarray,
    *,
    dataset: HippoERA5Dataset,
    block: slice,
    spatial_indices: np.ndarray,
    y_history: np.ndarray,
) -> np.ndarray:
    """Replace target-lag columns with no-test-label values for held-out points.

    ``y_history`` is updated by the recursive predictor: train-visible history is
    initialized from observed values, while held-out future rows are filled with
    model predictions before later lags are constructed.
    """

    lag_idx = _safe_lag_column_indices(dataset)
    if lag_idx is None:
        return phi
    lag1_col, lag2_col, diff_col = lag_idx
    spatial_indices = np.asarray(spatial_indices, dtype=int)
    start = block.start or 0
    stop = block.stop or dataset.Y.shape[0]
    s_count = spatial_indices.size
    safe_phi = np.asarray(phi, dtype=float).copy()
    for local_t, t_idx in enumerate(range(start, stop)):
        lag1_idx = max(t_idx - 1, 0)
        lag2_idx = max(t_idx - 2, 0)
        rows = slice(local_t * s_count, (local_t + 1) * s_count)
        lag1 = np.nan_to_num(y_history[lag1_idx, spatial_indices], nan=0.0)
        lag2 = np.nan_to_num(y_history[lag2_idx, spatial_indices], nan=0.0)
        safe_phi[rows, lag1_col] = lag1
        safe_phi[rows, lag2_col] = lag2
        safe_phi[rows, diff_col] = lag1 - lag2
    return safe_phi


def _predict_with_safe_heldout_lags(
    model: JointSSGPKronHiPPOSVGP,
    state,
    factors: BlockFactors,
    *,
    dataset: HippoERA5Dataset,
    routeb_dataset,
    eval_block: slice,
    spatial_indices: np.ndarray,
    c_eval: np.ndarray,
    ell_t: float,
    kernel_variance: float,
    kernel_type: str,
    prediction_mode: str,
    chunk_size: int,
    args: argparse.Namespace,
    variance_scale: float = 1.0,
    variance_add: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], BlockFactors]:
    """Recursive prediction for held-out locations without test-label lag leakage."""

    start = eval_block.start or 0
    stop = eval_block.stop or dataset.Y.shape[0]
    spatial_indices = np.asarray(spatial_indices, dtype=int)
    y_history = np.asarray(dataset.Y, dtype=float).copy()
    # Held-out labels must never be available when their lag features are built.
    # We recursively predict from the beginning of the online task up to the end
    # of the requested evaluation block, then return only the requested block.
    # The first online time point has no held-out target history, so missing lag
    # values are cold-started to zero by _replace_safe_lag_columns().
    if stop > 0:
        y_history[:stop, spatial_indices] = np.nan

    all_mean: list[np.ndarray] = []
    phi_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for t_idx in range(0, stop):
        one_block = slice(t_idx, t_idx + 1)
        phi_one = _safe_lag_phi_template(dataset, one_block, spatial_indices)
        phi_one = _replace_safe_lag_columns(
            phi_one,
            dataset=dataset,
            block=one_block,
            spatial_indices=spatial_indices,
            y_history=y_history,
        )
        if factors.temporal_backend == "analytic_hippo_rff":
            if factors.temporal_builder is None or factors.temporal_basis_spec is None:
                raise ValueError("analytic safe-lag rollout requires temporal builder and basis spec")
            kfu = (
                factors.temporal_builder.compute_kfu_t(
                    routeb_dataset.times[one_block],
                    factors.temporal_basis_spec,
                )
                .detach()
                .cpu()
                .numpy()
            )
            t_one = solve_spd(factors.Kt, kfu.T).T
        else:
            kfu = covariance_kernel(
                routeb_dataset.times[one_block],
                factors.inducing_times,
                lengthscale=ell_t,
                variance=kernel_variance,
                kernel_type=kernel_type,
            )
            # Recompute only the temporal interpolation row; the inducing grid and
            # spatial projection are the same as the actual prediction factors.
            t_one = solve_spd(factors.Kt, kfu.T).T
        y_one = routeb_dataset.Y[spatial_indices, one_block]
        gp_mean_one = np.einsum("ij,jk,k->i", c_eval, state.M_u, t_one[0])
        mean_one = phi_one @ state.beta_mean + gp_mean_one
        y_history[t_idx, spatial_indices] = mean_one
        if start <= t_idx < stop:
            all_mean.append(mean_one)
            phi_parts.append(phi_one)
            t_parts.append(t_one)
            y_parts.append(y_one)

    safe_factors = replace(
        factors,
        y_vec=vec_f(np.concatenate(y_parts, axis=1)) if y_parts else factors.y_vec[:0],
        Phi=np.vstack(phi_parts) if phi_parts else factors.Phi[:0],
        Y=np.concatenate(y_parts, axis=1) if y_parts else factors.Y[:, :0],
        T=np.vstack(t_parts) if t_parts else factors.T[:0],
    )
    mean, var, diagnostics = vectorized_predict_with_C(
        model,
        state,
        safe_factors,
        c_eval,
        prediction_mode=prediction_mode,
        chunk_size=chunk_size,
    )
    variance_scale = float(variance_scale)
    variance_add = float(variance_add)
    if variance_scale != 1.0 or variance_add != 0.0:
        var = np.maximum(variance_scale * var + variance_add, 1e-10)
        diagnostics = dict(diagnostics)
        diagnostics["safe_lag_variance_scale"] = variance_scale
        diagnostics["safe_lag_variance_add"] = variance_add
    return mean, var, diagnostics, safe_factors


def predict_with_eval_phi_policy(
    model: JointSSGPKronHiPPOSVGP,
    state,
    factors: BlockFactors,
    *,
    dataset: HippoERA5Dataset,
    routeb_dataset,
    eval_block: slice,
    ell_t: float,
    args: argparse.Namespace,
    c_eval: np.ndarray,
    spatial_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], BlockFactors]:
    if spatial_indices is not None and args.phi_mode in {"medium_era5", "rich_era5"}:
        return _predict_with_safe_heldout_lags(
            model,
            state,
            factors,
            dataset=dataset,
            routeb_dataset=routeb_dataset,
            eval_block=eval_block,
            spatial_indices=np.asarray(spatial_indices, dtype=int),
            c_eval=c_eval,
            ell_t=ell_t,
            kernel_variance=args.kernel_variance,
            kernel_type=args.kernel_type,
            prediction_mode=args.prediction_mode,
            chunk_size=args.prediction_chunk_size,
            args=args,
            variance_scale=float(getattr(args, "safe_lag_variance_scale", 1.0) or 1.0),
            variance_add=float(getattr(args, "safe_lag_variance_add", 0.0) or 0.0),
        )
    mean, var, diagnostics = vectorized_predict_with_C(
        model,
        state,
        factors,
        c_eval,
        prediction_mode=args.prediction_mode,
        chunk_size=args.prediction_chunk_size,
    )
    return mean, var, diagnostics, factors


def transfer_structured_joint_state_for_prediction(
    model: JointSSGPKronHiPPOSVGP,
    state,
    *,
    Kt_pred: np.ndarray,
    K_on_t: np.ndarray,
):
    """Temporarily transfer Route-B old likelihood statistics to a prediction basis.

    This does not add a new likelihood block and does not mutate the true online
    state. It is used only for the horizon-extended future diagnostic.
    """

    beta_dim = int(state.beta_mean.shape[0])
    old_stats = model._routeB_transfer_old_stats(
        state,
        Kt_pred,
        K_on_t,
        no_transfer=False,
        beta_dim=beta_dim,
    )
    beta_prior_precision = np.zeros((0, 0)) if beta_dim == 0 else inv_spd(model.beta_prior_cov, jitter=model.jitter)
    beta_prior_natural = np.zeros(0) if beta_dim == 0 else beta_prior_precision @ model.beta_prior_mean
    A_beta = symmetrize(beta_prior_precision + old_stats["R_beta_beta"])
    h_beta_total = beta_prior_natural + old_stats["h_beta"]
    schur = model.recover_posterior_mean_structured(
        A_beta=A_beta,
        B_beta_u=old_stats["R_beta_u"],
        h_beta=h_beta_total,
        h_u=vec_f(old_stats["H_info"]),
        Kt_new=Kt_pred,
        B_temporal=old_stats["B_temporal"],
    )
    ms = model.C.shape[1]
    mt = Kt_pred.shape[0]
    return state.copy_with(
        beta_mean=schur["m_beta"],
        beta_cov=schur["S_beta_beta"],
        M_u=unvec_f(schur["m_u"], (ms, mt)),
        B_temporal=old_stats["B_temporal"],
        H_info=old_stats["H_info"],
        Kt_current=Kt_pred,
        R_beta_beta=old_stats["R_beta_beta"],
        R_beta_u=old_stats["R_beta_u"],
        h_beta=old_stats["h_beta"],
        beta_prior_precision=beta_prior_precision,
        beta_prior_natural=beta_prior_natural,
        Lambda_beta_given_u=schur["Lambda_beta_given_u"],
        S_beta_beta=schur["S_beta_beta"],
        metadata={**state.metadata, "future_basis_mode": "extended", "temporary_prediction_state": True},
    )


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["method"]),
                str(row["eval_mode"]),
                str(row.get("phi_mode", "base")),
                str(row.get("future_basis_mode", "observed")),
            ),
            [],
        ).append(row)
    out_rows = []
    metrics = [
        "rmse",
        "mae",
        "nll",
        "coverage90",
        "ece",
        "avg_var",
        "avg_std",
        "avg_width90",
        "avg_predictive_variance",
        "avg_interval_width90",
        "runtime",
        "runtime_per_block",
        "num_train",
        "num_test",
        "coverage_sample_count",
        "routeb_sigma2",
        "selected_ell_t",
        "avg_sigma2",
        "avg_nu_star",
        "avg_u_posterior_term",
        "avg_beta_schur_term",
        "R_beta_u_norm",
        "R_beta_beta_norm",
        "R_uu_norm",
        "beta_u_coupling_ratio",
    ]
    for (method, mode, phi_mode, future_basis_mode), group in sorted(groups.items()):
        row: dict[str, Any] = {
            "method": method,
            "eval_mode": mode,
            "phi_mode": phi_mode,
            "future_basis_mode": future_basis_mode,
            "num_rows": len(group),
        }
        for metric in metrics:
            vals = np.asarray([float(item[metric]) for item in group if metric in item and str(item[metric]) != "nan"], dtype=float)
            if vals.size == 0:
                continue
            row[metric] = float(np.mean(vals))
            row[f"{metric}_se"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        out_rows.append(row)
    return out_rows


def forgetting_curve_rows(block_pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    by_train: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for row in block_pair_rows:
        method = str(row["method"])
        seed = int(row["seed"])
        split_seed = int(row.get("heldout_split_seed", seed))
        train_block = int(row["train_block_id"])
        eval_block = int(row["eval_block_id"])
        if train_block == eval_block:
            baseline[(method, seed, split_seed, eval_block)] = row
        by_train.setdefault((method, seed, split_seed, train_block), []).append(row)

    out: list[dict[str, Any]] = []
    for (method, seed, split_seed, train_block), rows in sorted(by_train.items()):
        rmse_deltas = []
        nll_deltas = []
        skipped = 0
        for row in rows:
            eval_block = int(row["eval_block_id"])
            if eval_block >= train_block:
                continue
            base = baseline.get((method, seed, split_seed, eval_block))
            if base is None:
                skipped += 1
                continue
            rmse_deltas.append(float(row["rmse"]) - float(base["rmse"]))
            nll_deltas.append(float(row["nll"]) - float(base["nll"]))
        if rmse_deltas:
            out.append(
                {
                    "method": method,
                    "seed": seed,
                    "heldout_split_seed": split_seed,
                    "online_block_index": train_block,
                    "rmse_forgetting": float(np.mean(rmse_deltas)),
                    "nll_forgetting": float(np.mean(nll_deltas)),
                    "num_old_blocks": len(rmse_deltas),
                    "num_skipped_blocks": skipped,
                }
            )
    return out


def summarize_forgetting_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), int(row["online_block_index"])), []).append(row)
    out = []
    for (method, block_id), group in sorted(groups.items()):
        item: dict[str, Any] = {"method": method, "online_block_index": block_id, "num_rows": len(group)}
        for metric in ["rmse_forgetting", "nll_forgetting"]:
            vals = np.asarray([float(row[metric]) for row in group], dtype=float)
            item[metric] = float(np.mean(vals))
            item[f"{metric}_se"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        out.append(item)
    return out


def summarize_heldout_seen_history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["eval_mode"])), []).append(row)
    out = []
    metrics = [
        "rmse",
        "mae",
        "nll",
        "coverage90",
        "ece",
        "avg_var",
        "avg_std",
        "avg_width90",
        "avg_predictive_variance",
        "runtime_per_block",
        "num_train",
        "num_test",
        "coverage_sample_count",
    ]
    for (method, eval_mode), group in sorted(groups.items()):
        item: dict[str, Any] = {"method": method, "eval_mode": eval_mode, "num_rows": len(group)}
        for metric in metrics:
            vals = np.asarray([float(row[metric]) for row in group if metric in row and str(row[metric]) != "nan"], dtype=float)
            if vals.size == 0:
                continue
            item[metric] = float(np.mean(vals))
            item[f"{metric}_se"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        out.append(item)
    return out


def summarize_heldout_by_independent_run(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        split_seed = int(row.get("heldout_split_seed", row.get("seed", 0)))
        run_groups.setdefault((str(row["method"]), split_seed), []).append(row)

    metrics = ["nll", "rmse", "coverage90", "ece", "runtime_per_block"]
    run_rows: list[dict[str, Any]] = []
    for (method, split_seed), group in sorted(run_groups.items()):
        item: dict[str, Any] = {"method": method, "heldout_split_seed": split_seed, "num_block_pairs": len(group)}
        for metric in metrics:
            vals = np.asarray([float(row[metric]) for row in group if metric in row and str(row[metric]) != "nan"], dtype=float)
            if vals.size:
                item[metric] = float(np.mean(vals))
        run_rows.append(item)

    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in run_rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    out: list[dict[str, Any]] = []
    for method, group in sorted(by_method.items()):
        item = {"method": method, "num_runs": len(group)}
        for metric in metrics:
            vals = np.asarray([float(row[metric]) for row in group if metric in row], dtype=float)
            if vals.size == 0:
                continue
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            item[metric] = float(np.mean(vals))
            item[f"{metric}_std"] = std
            item[f"{metric}_ci95"] = float(1.96 * std / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        out.append(item)
    return out


def summarize_final_forgetting_by_independent_run(forgetting_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_by_run: dict[tuple[str, int], dict[str, Any]] = {}
    for row in forgetting_rows:
        method = str(row["method"])
        split_seed = int(row.get("heldout_split_seed", row.get("seed", 0)))
        key = (method, split_seed)
        if key not in final_by_run or int(row["online_block_index"]) > int(final_by_run[key]["online_block_index"]):
            final_by_run[key] = row

    by_method: dict[str, list[dict[str, Any]]] = {}
    for (method, _), row in final_by_run.items():
        by_method.setdefault(method, []).append(row)
    out: list[dict[str, Any]] = []
    for method, group in sorted(by_method.items()):
        item = {"method": method, "num_runs": len(group)}
        for metric in ["nll_forgetting", "rmse_forgetting"]:
            vals = np.asarray([float(row[metric]) for row in group], dtype=float)
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            item[metric] = float(np.mean(vals))
            item[f"{metric}_std"] = std
            item[f"{metric}_ci95"] = float(1.96 * std / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        out.append(item)
    return out


def horizon_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["method"]),
                str(row.get("future_basis_mode", "observed")),
                int(row["horizon_index"]),
                str(row.get("phi_mode", "base")),
            ),
            [],
        ).append(row)
    out = []
    metrics = ["rmse", "mae", "nll", "coverage90", "ece", "avg_var", "avg_predictive_variance", "avg_nu_star"]
    for (method, future_basis_mode, horizon_index, phi_mode), group in sorted(groups.items()):
        row: dict[str, Any] = {
            "method": method,
            "future_basis_mode": future_basis_mode,
            "horizon_index": horizon_index,
            "phi_mode": phi_mode,
            "num_rows": len(group),
        }
        for metric in metrics:
            vals = np.asarray([float(item[metric]) for item in group if metric in item and str(item[metric]) != "nan"], dtype=float)
            if vals.size == 0:
                continue
            row[metric] = float(np.mean(vals))
            row[f"{metric}_se"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        out.append(row)
    return out


def capacity_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["mt"]), int(row["ms"]), str(row["method"]), str(row["eval_mode"])), []).append(row)
    out = []
    metrics = [
        "rmse",
        "mae",
        "nll",
        "coverage90",
        "ece",
        "avg_var",
        "avg_std",
        "avg_width90",
        "avg_predictive_variance",
        "runtime_per_block",
        "avg_sigma2",
        "avg_nu_star",
        "avg_u_posterior_term",
        "avg_beta_schur_term",
        "beta_u_coupling_ratio",
    ]
    for (mt, ms, method, mode), group in sorted(groups.items()):
        row: dict[str, Any] = {"mt": mt, "ms": ms, "method": method, "eval_mode": mode, "num_rows": len(group)}
        for metric in metrics:
            vals = np.asarray([float(item[metric]) for item in group if metric in item and str(item[metric]) != "nan"], dtype=float)
            if vals.size == 0:
                continue
            row[metric] = float(np.mean(vals))
            row[f"{metric}_se"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        out.append(row)
    return out


def eval_blocks(blocks: list[slice], block_id: int, mode: str) -> list[slice]:
    if mode == "current":
        return [blocks[block_id]]
    if mode == "seen_history":
        return [slice(0, blocks[block_id].stop)]
    if mode == "batch":
        return [slice(0, blocks[-1].stop)] if block_id == len(blocks) - 1 else []
    if mode == "future":
        return [] if block_id + 1 >= len(blocks) else [blocks[block_id + 1]]
    raise ValueError(f"Unknown eval mode: {mode}")


def normalise_time_dataset(dataset: HippoERA5Dataset) -> tuple[HippoERA5Dataset, float]:
    raw = np.asarray(dataset.times, dtype=float)
    scale = max(float(raw[-1] - raw[0]), 1e-12)
    return normalise_time_dataset_with_scale(dataset, scale=scale, source="dataset_span"), scale


def normalise_time_dataset_with_scale(dataset: HippoERA5Dataset, *, scale: float, source: str) -> HippoERA5Dataset:
    raw = np.asarray(dataset.times, dtype=float)
    scale = max(float(scale), 1e-12)
    times = (raw - raw[0]) / scale
    return (
        HippoERA5Dataset(
            times=times,
            coords=dataset.coords,
            Y=dataset.Y,
            Phi=dataset.Phi,
            tasks=dataset.tasks,
            variable_index=dataset.variable_index,
            scaled=dataset.scaled,
            selected_files=dataset.selected_files,
            Y_unscaled=dataset.Y_unscaled,
            metadata={
                **(dataset.metadata or {}),
                "routeb_time_normalization": source,
                "routeb_raw_time_scale": scale,
                "routeb_raw_time_start": float(raw[0]),
            },
        )
    )


def full_task_slice(dataset: HippoERA5Dataset) -> slice:
    return slice(0, dataset.Y.shape[0])


def selected_locations_from_dataset(dataset: HippoERA5Dataset) -> list[tuple[float, float]]:
    return [(float(lat), float(lon)) for lat, lon in np.asarray(dataset.coords, dtype=float)]


def routeb_dataset_from_era5(dataset: HippoERA5Dataset, *, sigma2: float, args: argparse.Namespace):
    return to_routeb_synthetic_dataset(
        dataset,
        sigma2=sigma2,
        gp_prior_variance=args.kernel_variance,
        standardize_coords=True,
    )


def phi_for_slice(dataset: HippoERA5Dataset, block: slice) -> np.ndarray:
    s_count = dataset.coords.shape[0]
    row_idx = []
    for t_idx in range(block.start or 0, block.stop or dataset.Y.shape[0]):
        row_idx.extend(range(t_idx * s_count, (t_idx + 1) * s_count))
    return dataset.Phi[np.asarray(row_idx)]


def fixed_spatial_train_test_split(num_space: int, *, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError("--heldout-test-fraction must be in (0, 1)")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(num_space))
    n_test = max(1, int(round(float(test_fraction) * num_space)))
    n_test = min(n_test, num_space - 1)
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx.astype(int), test_idx.astype(int)


def temporal_backend(args: argparse.Namespace) -> str:
    return str(getattr(args, "temporal_backend", "analytic_hippo_rff"))


def routeb_block_factors(
    dataset,
    *,
    block: slice,
    basis_block: slice,
    old_basis_block: slice | None,
    z_t: np.ndarray,
    z_t_old: np.ndarray | None,
    lengthscale: float,
    kernel_variance: float,
    kernel_type: str,
    args: argparse.Namespace,
) -> BlockFactors:
    """Build temporal Route-B factors from the selected shared backend."""

    if temporal_backend(args) == "analytic_hippo_rff":
        return make_block_factors_analytic_hippo(
            dataset,
            block=block,
            basis_block=basis_block,
            old_basis_block=old_basis_block,
            mt=int(args.mt),
            lengthscale=lengthscale,
            kernel_variance=kernel_variance,
            rff_sample_size=int(getattr(args, "temporal_rff_sample_size", 256)),
            seed=int(getattr(args, "temporal_rff_seed", 0)),
            moving=True,
            kernel_type=kernel_type,
            **_sm_kwargs(args),
        )
    return make_block_factors(
        dataset,
        block=block,
        z_t=z_t,
        z_t_old=z_t_old,
        lengthscale=lengthscale,
        kernel_variance=kernel_variance,
        kernel_type=kernel_type,
        **_sm_kwargs(args),
    )


def make_block_factors_subset(
    dataset,
    *,
    block: slice,
    basis_block: slice | None = None,
    old_basis_block: slice | None = None,
    z_t: np.ndarray,
    z_t_old: np.ndarray | None,
    lengthscale: float,
    kernel_variance: float,
    kernel_type: str,
    spatial_indices: np.ndarray,
    args: argparse.Namespace | None = None,
) -> BlockFactors:
    if args is None:
        base = make_block_factors(
            dataset,
            block=block,
            z_t=z_t,
            z_t_old=z_t_old,
            lengthscale=lengthscale,
            kernel_variance=kernel_variance,
            kernel_type=kernel_type,
        )
    else:
        base = routeb_block_factors(
            dataset,
            block=block,
            basis_block=block if basis_block is None else basis_block,
            old_basis_block=old_basis_block,
            z_t=z_t,
            z_t_old=z_t_old,
            lengthscale=lengthscale,
            kernel_variance=kernel_variance,
            kernel_type=kernel_type,
            args=args,
        )
    spatial_indices = np.asarray(spatial_indices, dtype=int)
    start = block.start or 0
    stop = block.stop or dataset.Y.shape[1]
    ns_full = dataset.Y.shape[0]
    row_idx: list[int] = []
    for t_idx in range(start, stop):
        row_idx.extend((t_idx * ns_full + spatial_indices).tolist())
    y_subset = dataset.Y[spatial_indices, block]
    return BlockFactors(
        y_vec=vec_f(y_subset),
        Phi=dataset.Phi[np.asarray(row_idx, dtype=int)],
        Y=y_subset,
        T=base.T,
        Kt=base.Kt,
        K_on_t=base.K_on_t,
        block_slice=block,
        inducing_times=base.inducing_times,
        temporal_backend=base.temporal_backend,
        temporal_builder=base.temporal_builder,
        temporal_basis_spec=base.temporal_basis_spec,
    )


def estimate_sigma2_from_initial_block(dataset: HippoERA5Dataset, block: slice, ridge: float = 1e-3) -> float:
    """Estimate observation variance from initial-task beta-only residuals.

    This uses only the first online block, before any future labels exist. It is
    shared by all Route B methods for a given seed.
    """

    phi = phi_for_slice(dataset, block)
    y = vec_f(dataset.Y[block].T)
    precision = phi.T @ phi + ridge * np.eye(phi.shape[1])
    coef = np.linalg.solve(precision, phi.T @ y)
    resid = y - phi @ coef
    return float(max(np.var(resid), 1e-4))


def estimate_sigma2_from_calibration_task(dataset: HippoERA5Dataset, ridge: float = 1e-3) -> float:
    """Estimate observation variance from beta-only residuals on the calibration task."""

    return estimate_sigma2_from_initial_block(dataset, full_task_slice(dataset), ridge=ridge)


def full_gp_initial_task_nlml(
    routeb_dataset,
    block: slice,
    ell_t: float,
    sigma2: float,
    spatial_lengthscale: float | np.ndarray,
    kernel_variance: float,
    kernel_type: str = "rbf",
) -> float:
    times = routeb_dataset.times[block]
    spatial = routeb_dataset.spatial_coords
    y = vec_f(routeb_dataset.Y[:, block])
    phi = routeb_dataset.Phi[: y.size]
    beta_prior_cov = 10.0 * np.eye(phi.shape[1])
    kt = covariance_kernel(times, lengthscale=ell_t, variance=kernel_variance, kernel_type=kernel_type)
    ks = covariance_kernel(spatial, lengthscale=spatial_lengthscale, variance=1.0, kernel_type=kernel_type)
    cov = np.kron(kt, ks) + phi @ beta_prior_cov @ phi.T + sigma2 * np.eye(y.size)
    cov = symmetrize(cov)
    jitter = 1e-8 * max(1.0, float(np.mean(np.diag(cov))))
    for _ in range(8):
        try:
            chol = np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return float(0.5 * (y @ alpha + logdet + y.size * np.log(2.0 * np.pi)) / y.size)


def select_ell_t(
    routeb_dataset,
    block: slice,
    args: argparse.Namespace,
    sigma2: float,
) -> tuple[float, float, str]:
    if args.ell_t_fit_mode == "none":
        return float(args.model_ell_t), float("nan"), str(args.model_ell_t)
    grid = [float(x) for x in args.ell_t_grid]
    scores = [
        full_gp_initial_task_nlml(
            routeb_dataset,
            block,
            ell_t=value,
            sigma2=sigma2,
            spatial_lengthscale=spatial_kernel_lengthscale(args),
            kernel_variance=args.kernel_variance,
            kernel_type=args.kernel_type,
        )
        for value in grid
    ]
    best_idx = int(np.argmin(scores))
    return grid[best_idx], float(scores[best_idx]), " ".join(f"{value:g}:{score:.8g}" for value, score in zip(grid, scores))


def select_ell_t_from_calibration_task(
    calibration_dataset: HippoERA5Dataset,
    args: argparse.Namespace,
    sigma2: float,
) -> tuple[float, float, str]:
    calibration_routeb = routeb_dataset_from_era5(calibration_dataset, sigma2=sigma2, args=args)
    return select_ell_t(calibration_routeb, full_task_slice(calibration_dataset), args, sigma2)


def subset_dataset_for_fullgp_mll(
    dataset: HippoERA5Dataset,
    *,
    max_time: int,
    max_locations: int,
) -> HippoERA5Dataset:
    """Small calibration subset for dense full-GP MLL hyperparameter diagnostics."""

    t_count = min(int(max_time), dataset.Y.shape[0])
    s_count = min(int(max_locations), dataset.Y.shape[1])
    loc_idx = np.arange(s_count, dtype=int)
    row_idx: list[int] = []
    full_s = dataset.Y.shape[1]
    for t_idx in range(t_count):
        row_idx.extend((t_idx * full_s + loc_idx).tolist())
    return replace(
        dataset,
        times=np.asarray(dataset.times[:t_count], dtype=float),
        coords=np.asarray(dataset.coords[loc_idx], dtype=float),
        Y=np.asarray(dataset.Y[:t_count][:, loc_idx], dtype=float),
        Phi=np.asarray(dataset.Phi[np.asarray(row_idx, dtype=int)], dtype=float),
        Y_unscaled=None if dataset.Y_unscaled is None else np.asarray(dataset.Y_unscaled[:t_count][:, loc_idx], dtype=float),
        metadata={
            **(dataset.metadata or {}),
            "fullgp_mll_subset_num_time": t_count,
            "fullgp_mll_subset_num_locations": s_count,
        },
    )


def select_hyperparams_from_calibration_fullgp_mll(
    calibration_dataset: HippoERA5Dataset,
    args: argparse.Namespace,
) -> tuple[float, float, float, float, str]:
    """Select ell_t, sigma2 and kernel variance by dense GP NLML on an initial-task subset."""

    subset = subset_dataset_for_fullgp_mll(
        calibration_dataset,
        max_time=args.hyperparam_fit_max_time,
        max_locations=args.hyperparam_fit_max_locations,
    )
    best: tuple[float, float, float, float] | None = None
    scores: list[str] = []
    original_kernel_variance = float(args.kernel_variance)
    try:
        for kernel_variance in [float(x) for x in args.kernel_variance_grid]:
            args.kernel_variance = kernel_variance
            routeb_subset = routeb_dataset_from_era5(subset, sigma2=1.0, args=args)
            for ell_t in [float(x) for x in args.ell_t_grid]:
                for noise in [float(x) for x in args.noise_grid]:
                    sigma2 = float(noise) ** 2
                    score = full_gp_initial_task_nlml(
                        routeb_subset,
                        full_task_slice(subset),
                        ell_t=ell_t,
                        sigma2=sigma2,
                        spatial_lengthscale=spatial_kernel_lengthscale(args),
                        kernel_variance=kernel_variance,
                        kernel_type=args.kernel_type,
                    )
                    scores.append(f"kernel={args.kernel_type},ell={ell_t:g},noise={noise:g},kv={kernel_variance:g}:{score:.8g}")
                    if best is None or score < best[0]:
                        best = (score, ell_t, sigma2, kernel_variance)
    finally:
        args.kernel_variance = original_kernel_variance
    if best is None:
        raise RuntimeError("Full-GP MLL hyperparameter grid was empty.")
    score, ell_t, sigma2, kernel_variance = best
    return float(ell_t), float(sigma2), float(kernel_variance), float(score), " ".join(scores)


def vectorized_predict(
    model: JointSSGPKronHiPPOSVGP,
    state,
    factors,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    return vectorized_predict_with_C(model, state, factors, model.C)


def vectorized_predict_with_C(
    model: JointSSGPKronHiPPOSVGP,
    state,
    factors,
    C_eval: np.ndarray,
    *,
    prediction_mode: str = "streaming_sylvester",
    chunk_size: int = 8192,
    include_conditional_residual_variance: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if prediction_mode == "dense":
        return vectorized_predict_with_C_dense(
            model,
            state,
            factors,
            C_eval,
            include_conditional_residual_variance=include_conditional_residual_variance,
        )
    if prediction_mode == "streaming_sylvester":
        return vectorized_predict_with_C_streaming_sylvester(
            model,
            state,
            factors,
            C_eval,
            chunk_size=chunk_size,
            include_conditional_residual_variance=include_conditional_residual_variance,
        )
    raise ValueError(f"Unknown prediction_mode: {prediction_mode}")


def vectorized_predict_with_C_dense(
    model: JointSSGPKronHiPPOSVGP,
    state,
    factors,
    C_eval: np.ndarray,
    *,
    include_conditional_residual_variance: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    C_eval = np.asarray(C_eval, dtype=float)
    a_mat = dense_A_from_factors(factors.T, C_eval)
    mean = factors.Phi @ state.beta_mean + a_mat @ vec_f(state.M_u)

    kt_inv = inv_spd(state.Kt_current, jitter=model.jitter)
    precision_G = state.G if getattr(state, "G", None) is not None else model.G
    du = dense_Du_for_tests(kt_inv, model.Ks_inv, state.B_temporal, precision_G)
    du_inv = inv_spd(du, jitter=model.jitter)
    v_mat = a_mat @ du_inv
    u_terms = np.sum(a_mat * v_mat, axis=1)

    if state.R_beta_u is not None and state.S_beta_beta is not None:
        adjusted_phi = factors.Phi - v_mat @ state.R_beta_u.T
        beta_terms = np.einsum("ij,jk,ik->i", adjusted_phi, state.S_beta_beta, adjusted_phi)
    else:
        beta_terms = np.einsum("ij,jk,ik->i", factors.Phi, state.beta_cov, factors.Phi)

    t_var = np.sum((factors.T @ state.Kt_current) * factors.T, axis=1)
    s_var = np.sum((C_eval @ model.Ks) * C_eval, axis=1)
    projected_prior = np.repeat(t_var, C_eval.shape[0]) * np.tile(s_var, factors.T.shape[0])
    nu_raw = np.maximum(0.0, model.prior_point_variance - projected_prior)
    nu = nu_raw if include_conditional_residual_variance else np.zeros_like(nu_raw)
    var = np.maximum(model.sigma2 + nu + u_terms + beta_terms, model.jitter)
    diagnostics = {
        "avg_sigma2": float(model.sigma2),
        "avg_nu_star": float(np.mean(nu)),
        "avg_nu_star_raw": float(np.mean(nu_raw)),
        "avg_u_posterior_term": float(np.mean(u_terms)),
        "avg_beta_schur_term": float(np.mean(beta_terms)),
    }
    return mean, var, diagnostics


def _du_sylvester_eigendecomp(
    model: JointSSGPKronHiPPOSVGP,
    state,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute factors for repeated structured ``D_u`` solves.

    This is the prediction-side analogue of the Sylvester solve used in Route B
    posterior recovery. It avoids materializing dense ``D_u`` and ``D_u^{-1}``.
    """

    kt_inv = inv_spd(state.Kt_current, jitter=model.jitter)
    ks_inv = model.Ks_inv
    precision_G = state.G if getattr(state, "G", None) is not None else model.G
    left_vals, P = eigh(
        symmetrize(precision_G),
        add_jitter(ks_inv, model.jitter),
        check_finite=False,
    )
    right_vals, Q = eigh(
        symmetrize(state.B_temporal),
        add_jitter(kt_inv, model.jitter),
        check_finite=False,
    )
    denom = 1.0 + np.outer(left_vals, right_vals)
    denom = np.where(np.abs(denom) < model.jitter, np.sign(denom) * model.jitter + (denom == 0) * model.jitter, denom)
    return P, Q, denom


def vectorized_predict_with_C_streaming_sylvester(
    model: JointSSGPKronHiPPOSVGP,
    state,
    factors,
    C_eval: np.ndarray,
    *,
    chunk_size: int = 8192,
    include_conditional_residual_variance: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Predict without dense ``A``, dense ``D_u`` or explicit ``D_u^{-1}``.

    For a point with projection ``a_* = t_* kron c_*``, the variance formula
    only needs ``D_u^{-1} a_*`` through quadratic forms. The computation below
    uses the same Sylvester diagonalization as posterior recovery and streams
    over prediction points in chunks.
    """

    C_eval = np.asarray(C_eval, dtype=float)
    T_eval = np.asarray(factors.T, dtype=float)
    Phi = np.asarray(factors.Phi, dtype=float)
    n_t, _ = T_eval.shape
    n_s, _ = C_eval.shape
    n = n_t * n_s
    if Phi.shape[0] != n:
        raise ValueError(f"Phi rows ({Phi.shape[0]}) do not match T/C product ({n}).")

    chunk_size = max(1, int(chunk_size))
    P, Q, denom = _du_sylvester_eigendecomp(model, state)
    inv_denom = 1.0 / denom
    c_proj_all = C_eval @ P
    t_proj_all = T_eval @ Q
    s_var_all = np.sum((C_eval @ model.Ks) * C_eval, axis=1)
    t_var_all = np.sum((T_eval @ state.Kt_current) * T_eval, axis=1)

    r_tilde: list[np.ndarray] = []
    if state.R_beta_u is not None and state.S_beta_beta is not None:
        for row in state.R_beta_u:
            r_matrix = unvec_f(row, (C_eval.shape[1], T_eval.shape[1]))
            r_tilde.append(P.T @ r_matrix @ Q)

    mean = np.empty(n, dtype=float)
    var = np.empty(n, dtype=float)
    u_terms_all = np.empty(n, dtype=float)
    beta_terms_all = np.empty(n, dtype=float)
    nu_all = np.empty(n, dtype=float)

    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        flat_idx = np.arange(start, stop, dtype=int)
        time_idx = flat_idx // n_s
        space_idx = flat_idx % n_s
        c_raw = C_eval[space_idx]
        t_raw = T_eval[time_idx]
        c_proj = c_proj_all[space_idx]
        t_proj = t_proj_all[time_idx]

        gp_mean = np.einsum("ij,jk,ik->i", c_raw, state.M_u, t_raw)
        beta_mean = Phi[start:stop] @ state.beta_mean
        mean[start:stop] = beta_mean + gp_mean

        t_proj_sq = t_proj * t_proj
        c_proj_sq = c_proj * c_proj
        u_inner = t_proj_sq @ inv_denom.T
        u_terms = np.sum(c_proj_sq * u_inner, axis=1)
        u_terms_all[start:stop] = u_terms

        if r_tilde:
            cross_cols = []
            for rt in r_tilde:
                transformed = t_proj @ (rt * inv_denom).T
                cross_cols.append(np.sum(c_proj * transformed, axis=1))
            cross = np.column_stack(cross_cols)
            adjusted_phi = Phi[start:stop] - cross
            beta_terms = np.einsum("ij,jk,ik->i", adjusted_phi, state.S_beta_beta, adjusted_phi)
        else:
            beta_terms = np.einsum("ij,jk,ik->i", Phi[start:stop], state.beta_cov, Phi[start:stop])
        beta_terms_all[start:stop] = beta_terms

        projected_prior = t_var_all[time_idx] * s_var_all[space_idx]
        nu_raw = np.maximum(0.0, model.prior_point_variance - projected_prior)
        nu = nu_raw if include_conditional_residual_variance else np.zeros_like(nu_raw)
        nu_all[start:stop] = nu
        var[start:stop] = np.maximum(model.sigma2 + nu + u_terms + beta_terms, model.jitter)

    diagnostics = {
        "avg_sigma2": float(model.sigma2),
        "avg_nu_star": float(np.mean(nu_all)),
        "avg_nu_star_raw": float(
            np.mean(np.maximum(0.0, model.prior_point_variance - np.repeat(t_var_all, n_s) * np.tile(s_var_all, n_t)))
        ),
        "avg_u_posterior_term": float(np.mean(u_terms_all)),
        "avg_beta_schur_term": float(np.mean(beta_terms_all)),
    }
    return mean, var, diagnostics


def state_coupling_diagnostics(model: JointSSGPKronHiPPOSVGP, state) -> dict[str, float]:
    r_beta_u = getattr(state, "R_beta_u", None)
    r_beta_beta = getattr(state, "R_beta_beta", None)
    rbu_norm = float(np.linalg.norm(r_beta_u)) if r_beta_u is not None else 0.0
    rbb_norm = float(np.linalg.norm(r_beta_beta)) if r_beta_beta is not None else 0.0
    ruu_norm = float(np.linalg.norm(state.B_temporal) * np.linalg.norm(model.G))
    denom = np.sqrt(max(rbb_norm * ruu_norm, 1e-30))
    ratio = float(rbu_norm / denom) if rbu_norm > 0.0 and denom > 0.0 else 0.0
    return {
        "R_beta_u_norm": rbu_norm,
        "R_beta_beta_norm": rbb_norm,
        "R_uu_norm": ruu_norm,
        "beta_u_coupling_ratio": ratio,
    }


def metric_row(
    *,
    method: str,
    eval_mode: str,
    block_id: int,
    y: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    runtime: float,
    diagnostics: dict[str, float],
    dataset: HippoERA5Dataset,
    args: argparse.Namespace,
    seed: int,
    sigma2: float,
    ell_t: float,
    ell_score: float,
) -> dict[str, Any]:
    var = np.maximum(var, 1e-10)
    std = np.sqrt(var)
    row = {
        "method": method,
        "eval_mode": eval_mode,
        "block_id": block_id,
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "mae": float(np.mean(np.abs(y - mean))),
        "nll": gaussian_nll(y, mean, var),
        "coverage90": coverage90(y, mean, var),
        "ece": ece_gaussian(y, mean, var),
        "avg_var": float(np.mean(var)),
        "avg_std": float(np.mean(std)),
        "avg_width90": float(np.mean(2.0 * Z90 * std)),
        "avg_predictive_variance": float(np.mean(var)),
        "avg_interval_width90": float(np.mean(2.0 * Z90 * std)),
        "runtime": runtime,
        "runtime_per_block": runtime,
        "num_train": int((block_id + 1) * args.block_size * dataset.coords.shape[0]),
        "num_test": int(y.size),
        "coverage_sample_count": int(y.size),
        "dataset": "era5_processed_timeseries_4",
        "tasks": ",".join(dataset.tasks),
        "calibration_tasks": ",".join(getattr(args, "calibration_tasks", [])),
        "online_tasks": ",".join(getattr(args, "online_tasks", args.tasks)),
        "variable_index": dataset.variable_index,
        "num_time": dataset.Y.shape[0],
        "num_space": dataset.Y.shape[1],
        "block_size": args.block_size,
        "seed": seed,
        "scale": "scaled" if dataset.scaled else "unscaled",
        "routeb_sigma2": float(sigma2),
        "routeb_noise": float(np.sqrt(sigma2)),
        "routeb_sigma2_source": args.sigma2_source,
        "selected_ell_t": float(ell_t),
        "model_ell_t": float(ell_t),
        "selected_ell_t_score": ell_score,
        "ell_t_fit_mode": args.ell_t_fit_mode,
        "ell_t_fit_dataset": getattr(args, "ell_t_fit_dataset", "online_initial_block"),
        "phi_mode": args.phi_mode,
        "future_basis_mode": args.future_basis_mode,
        "mt": args.mt,
        "ms": args.ms,
        "kernel_type": args.kernel_type,
        "spatial_lengthscale": args.spatial_lengthscale,
        "spatial_ard_lengthscale_lat": float(args.spatial_ard_lengthscales[0]) if args.spatial_ard_lengthscales else float("nan"),
        "spatial_ard_lengthscale_lon": float(args.spatial_ard_lengthscales[1]) if args.spatial_ard_lengthscales else float("nan"),
        "kernel_variance": args.kernel_variance,
        "prediction_mode": args.prediction_mode,
        "prediction_chunk_size": args.prediction_chunk_size,
        **diagnostics,
    }
    return row


def run_routeb_method(
    dataset_raw: HippoERA5Dataset,
    *,
    seed: int,
    heldout_split_seed: int,
    method: str,
    args: argparse.Namespace,
    sigma2: float,
    ell_t: float,
    ell_score: float,
    ell_grid_scores: str,
    raw_time_scale: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_base = normalise_time_dataset_with_scale(dataset_raw, scale=raw_time_scale, source="calibration_task_span")
    dataset = augment_dataset_phi(dataset_base, phi_mode=args.phi_mode, xlag_length=args.xlag_length)
    dataset = maybe_add_lag_training_noise(dataset, args=args, seed=seed)
    routeb_dataset = routeb_dataset_from_era5(dataset, sigma2=sigma2, args=args)
    blocks = [slice(start, min(dataset.Y.shape[0], start + args.block_size)) for start in range(0, dataset.Y.shape[0], args.block_size)]
    routeb_dataset = SimpleNamespace(**{**routeb_dataset.__dict__, "sigma2": sigma2})
    snapshot_indices = _parse_snapshot_indices(args.map_snapshot_time_indices, dataset.Y.shape[0])
    saved_snapshots: set[tuple[str, int]] = set()

    _, ks, c_mat = make_spatial_projection(
        routeb_dataset.spatial_coords,
        args.ms,
        lengthscale=spatial_kernel_lengthscale(args),
        kernel_type=args.kernel_type,
        inducing_selection=args.spatial_inducing_selection,
        **_sm_spatial_kwargs(args),
    )
    if args.ohsvgp_heldout_eval:
        train_spatial_idx, test_spatial_idx = fixed_spatial_train_test_split(
            c_mat.shape[0],
            test_fraction=args.heldout_test_fraction,
            seed=int(heldout_split_seed),
        )
        c_model = c_mat[train_spatial_idx]
    else:
        train_spatial_idx = np.arange(c_mat.shape[0], dtype=int)
        test_spatial_idx = np.arange(c_mat.shape[0], dtype=int)
        c_model = c_mat
    model = JointSSGPKronHiPPOSVGP(
        Ks=ks,
        C=c_model,
        sigma2=sigma2,
        beta_prior_mean=np.zeros(routeb_dataset.Phi.shape[1]),
        beta_prior_cov=beta_prior_cov_for_dataset(dataset, args=args),
        prior_point_variance=args.kernel_variance,
    )
    rows: list[dict[str, Any]] = []
    block_pair_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    ohsvgp_pair_rows: list[dict[str, Any]] = []
    per_location_rows: list[dict[str, Any]] = []
    state = None
    old_z = None
    old_block = None
    method_key = METHOD_LABELS[method]
    for block_id, block in enumerate(blocks):
        block_wall_started = time.perf_counter()
        inducing_moving = str(getattr(args, "temporal_inducing_mode", "moving")) == "moving"
        z_t = temporal_inducing_for_block(routeb_dataset.times, block, args.mt, moving=inducing_moving)
        if args.ohsvgp_heldout_eval:
            factors = make_block_factors_subset(
                routeb_dataset,
                block=block,
                basis_block=block,
                old_basis_block=old_block,
                z_t=z_t,
                z_t_old=old_z,
                lengthscale=ell_t,
                kernel_variance=args.kernel_variance,
                kernel_type=args.kernel_type,
                spatial_indices=train_spatial_idx,
                args=args,
            )
        else:
            factors = routeb_block_factors(
                routeb_dataset,
                block=block,
                basis_block=block,
                old_basis_block=old_block,
                z_t=z_t,
                z_t_old=old_z,
                lengthscale=ell_t,
                kernel_variance=args.kernel_variance,
                kernel_type=args.kernel_type,
                args=args,
            )
        start = time.perf_counter()
        kwargs = {
            "y_vec": factors.y_vec,
            "Phi": factors.Phi,
            "T_n": factors.T,
            "Kt_new": factors.Kt,
            "K_on_t": factors.K_on_t,
            "state": state,
        }
        if method == "no_transfer":
            state = model.update_block_no_transfer(**kwargs, inner_iters=args.inner_iters)
        elif method == "mean_field":
            state = model.update_block_mean_field_ssgp_transfer(**kwargs, inner_iters=args.inner_iters)
        elif method == "structured_joint":
            state = model.update_block_structured_joint_ssgp_transfer(**kwargs)
        else:
            raise ValueError(f"Unknown Route B internal method: {method}")
        runtime = time.perf_counter() - start

        for mode in args.eval_modes:
            for eval_block in eval_blocks(blocks, block_id, mode):
                if args.ohsvgp_heldout_eval:
                    eval_factors = make_block_factors_subset(
                        routeb_dataset,
                        block=eval_block,
                        basis_block=block,
                        old_basis_block=None,
                        z_t=z_t,
                        z_t_old=None,
                        lengthscale=ell_t,
                        kernel_variance=args.kernel_variance,
                        kernel_type=args.kernel_type,
                        spatial_indices=test_spatial_idx,
                        args=args,
                    )
                else:
                    eval_factors = routeb_block_factors(
                        routeb_dataset,
                        block=eval_block,
                        basis_block=block,
                        old_basis_block=None,
                        z_t=z_t,
                        z_t_old=None,
                        lengthscale=ell_t,
                        kernel_variance=args.kernel_variance,
                        kernel_type=args.kernel_type,
                        args=args,
                    )
                prediction_state = state
                future_basis_applied = "observed"
                if mode == "future" and args.future_basis_mode == "extended" and method == "structured_joint":
                    horizon_block = slice(0, eval_block.stop or routeb_dataset.Y.shape[1])
                    z_pred = temporal_inducing_for_block(
                        routeb_dataset.times,
                        horizon_block,
                        args.mt,
                        moving=inducing_moving,
                    )
                    if args.ohsvgp_heldout_eval:
                        eval_factors = make_block_factors_subset(
                            routeb_dataset,
                            block=eval_block,
                            basis_block=horizon_block,
                            old_basis_block=block,
                            z_t=z_pred,
                            z_t_old=z_t,
                            lengthscale=ell_t,
                            kernel_variance=args.kernel_variance,
                            kernel_type=args.kernel_type,
                            spatial_indices=test_spatial_idx,
                            args=args,
                        )
                    else:
                        eval_factors = routeb_block_factors(
                            routeb_dataset,
                            block=eval_block,
                            basis_block=horizon_block,
                            old_basis_block=block,
                            z_t=z_pred,
                            z_t_old=z_t,
                            lengthscale=ell_t,
                            kernel_variance=args.kernel_variance,
                            kernel_type=args.kernel_type,
                            args=args,
                        )
                    prediction_state = transfer_structured_joint_state_for_prediction(
                        model,
                        state,
                        Kt_pred=eval_factors.Kt,
                        K_on_t=eval_factors.K_on_t,
                    )
                    future_basis_applied = "extended"
                if args.phi_mode == "lag_ar" and mode == "future":
                    eval_factors = replace_factors_phi(
                        eval_factors,
                        phi_for_eval_block(
                            dataset_base,
                            eval_block,
                            phi_mode=args.phi_mode,
                            future_context_stop=block.stop or dataset_base.Y.shape[0],
                        ),
                    )
                c_eval = c_mat[test_spatial_idx] if args.ohsvgp_heldout_eval else model.C
                prediction_started = time.perf_counter()
                mean, var, diagnostics, eval_factors = predict_with_eval_phi_policy(
                    model,
                    prediction_state,
                    eval_factors,
                    dataset=dataset,
                    routeb_dataset=routeb_dataset,
                    eval_block=eval_block,
                    ell_t=ell_t,
                    args=args,
                    c_eval=c_eval,
                    spatial_indices=test_spatial_idx if args.ohsvgp_heldout_eval else None,
                )
                prediction_runtime = time.perf_counter() - prediction_started
                diagnostics["prediction_runtime"] = prediction_runtime
                diagnostics["block_incremental_runtime"] = time.perf_counter() - block_wall_started
                diagnostics.update(state_coupling_diagnostics(model, prediction_state))
                y = eval_factors.y_vec
                save_pointwise = args.save_per_location_predictions and (mode != "seen_history" or block_id == len(blocks) - 1)
                if save_pointwise:
                    append_per_location_prediction_rows(
                        per_location_rows,
                        method=method_key,
                        eval_mode=mode,
                        block_id=block_id,
                        eval_block=eval_block,
                        y=y,
                        mean=mean,
                        var=var,
                        dataset_raw=dataset_raw,
                        dataset=dataset,
                        args=args,
                        seed=seed,
                        sigma2=sigma2,
                        ell_t=ell_t,
                        future_basis_applied=future_basis_applied,
                        location_indices=test_spatial_idx if args.ohsvgp_heldout_eval else None,
                    )
                if (
                    args.save_map_snapshots
                    and method == args.map_snapshot_method
                    and mode in args.map_snapshot_eval_modes
                    and int(seed) == int(args.map_snapshot_seed)
                ):
                    eval_start = eval_block.start or 0
                    eval_stop = eval_block.stop or dataset.Y.shape[0]
                    target_here = sorted(idx for idx in snapshot_indices if eval_start <= idx < eval_stop)
                    if target_here:
                        s_count = dataset.coords.shape[0]
                        t_count = eval_stop - eval_start
                        y_mat = y.reshape((s_count, t_count), order="F").T
                        mean_mat = mean.reshape((s_count, t_count), order="F").T
                        var_mat = var.reshape((s_count, t_count), order="F").T
                        task_label = "_".join(dataset.tasks)
                        if args.capacity_sweep:
                            task_label = f"{task_label}_Mt{args.mt}_Ms{args.ms}"
                        for t_index in target_here:
                            key = (mode, t_index)
                            if key in saved_snapshots:
                                continue
                            local = t_index - eval_start
                            save_map_snapshot(
                                outdir=Path(args.map_outdir),
                                coords=dataset_base.coords,
                                y=y_mat[local],
                                mean=mean_mat[local],
                                var=var_mat[local],
                                method=method_key,
                                eval_mode=mode,
                                task_label=task_label,
                                t_index=t_index,
                                time_value=float(dataset_raw.times[t_index]),
                                variable_label=args.map_variable_label,
                            )
                            saved_snapshots.add(key)
                row = metric_row(
                    method=method_key,
                    eval_mode=mode,
                    block_id=block_id,
                    y=y,
                    mean=mean,
                    var=var,
                    runtime=runtime,
                    diagnostics=diagnostics,
                    dataset=dataset,
                    args=args,
                    seed=seed,
                    sigma2=sigma2,
                    ell_t=ell_t,
                    ell_score=ell_score,
                )
                row["routeb_raw_time_scale"] = raw_time_scale
                row["ell_t_grid_scores"] = ell_grid_scores
                row["future_basis_applied"] = future_basis_applied
                if args.ohsvgp_heldout_eval:
                    row["num_train"] = int((block_id + 1) * args.block_size * train_spatial_idx.size)
                    row["heldout_split_seed"] = int(heldout_split_seed)
                    row["heldout_split_axis"] = "spatial"
                    row["heldout_test_fraction"] = float(args.heldout_test_fraction)
                    row["num_train_locations"] = int(train_spatial_idx.size)
                    row["num_test_locations"] = int(test_spatial_idx.size)
                rows.append(row)
                if args.save_future_horizon_metrics and mode == "future":
                    eval_start = eval_block.start or 0
                    eval_stop = eval_block.stop or dataset.Y.shape[0]
                    s_count = dataset.coords.shape[0]
                    t_count = eval_stop - eval_start
                    y_mat = y.reshape((s_count, t_count), order="F").T
                    mean_mat = mean.reshape((s_count, t_count), order="F").T
                    var_mat = var.reshape((s_count, t_count), order="F").T
                    for local_h in range(t_count):
                        h_y = y_mat[local_h]
                        h_mean = mean_mat[local_h]
                        h_var = var_mat[local_h]
                        h_row = metric_row(
                            method=method_key,
                            eval_mode="future_horizon",
                            block_id=block_id,
                            y=h_y,
                            mean=h_mean,
                            var=h_var,
                            runtime=runtime,
                            diagnostics=diagnostics,
                            dataset=dataset,
                            args=args,
                            seed=seed,
                            sigma2=sigma2,
                            ell_t=ell_t,
                            ell_score=ell_score,
                        )
                        h_row["future_basis_applied"] = future_basis_applied
                        h_row["future_block_id"] = block_id + 1
                        h_row["horizon_index"] = local_h + 1
                        h_row["absolute_time_index"] = eval_start + local_h
                        h_row["num_space_at_horizon"] = s_count
                        horizon_rows.append(h_row)
        if args.save_forgetting_block_pairs:
            for eval_block_id in range(block_id + 1):
                eval_block = blocks[eval_block_id]
                eval_factors = routeb_block_factors(
                    routeb_dataset,
                    block=eval_block,
                    basis_block=block,
                    old_basis_block=None,
                    z_t=z_t,
                    z_t_old=None,
                    lengthscale=ell_t,
                    kernel_variance=args.kernel_variance,
                    kernel_type=args.kernel_type,
                    args=args,
                )
                c_eval = c_mat if args.ohsvgp_heldout_eval else model.C
                mean, var, diagnostics = vectorized_predict_with_C(
                    model,
                    state,
                    eval_factors,
                    c_eval,
                    prediction_mode=args.prediction_mode,
                    chunk_size=args.prediction_chunk_size,
                )
                diagnostics.update(state_coupling_diagnostics(model, state))
                y = eval_factors.y_vec
                pair_row = metric_row(
                    method=method_key,
                    eval_mode="seen_history_block_pair",
                    block_id=block_id,
                    y=y,
                    mean=mean,
                    var=var,
                    runtime=runtime,
                    diagnostics=diagnostics,
                    dataset=dataset,
                    args=args,
                    seed=seed,
                    sigma2=sigma2,
                    ell_t=ell_t,
                    ell_score=ell_score,
                )
                pair_row["train_block_id"] = block_id
                pair_row["eval_block_id"] = eval_block_id
                pair_row["routeb_raw_time_scale"] = raw_time_scale
                block_pair_rows.append(pair_row)
        if args.ohsvgp_heldout_eval and not getattr(args, "skip_heldout_block_pairs", False):
            for eval_block_id in range(block_id + 1):
                eval_block = blocks[eval_block_id]
                eval_factors = make_block_factors_subset(
                    routeb_dataset,
                    block=eval_block,
                    basis_block=block,
                    old_basis_block=None,
                    z_t=z_t,
                    z_t_old=None,
                    lengthscale=ell_t,
                    kernel_variance=args.kernel_variance,
                    kernel_type=args.kernel_type,
                    spatial_indices=test_spatial_idx,
                    args=args,
                )
                mean, var, diagnostics, eval_factors = predict_with_eval_phi_policy(
                    model,
                    state,
                    eval_factors,
                    dataset=dataset,
                    routeb_dataset=routeb_dataset,
                    eval_block=eval_block,
                    ell_t=ell_t,
                    args=args,
                    c_eval=c_mat[test_spatial_idx],
                    spatial_indices=test_spatial_idx,
                )
                diagnostics.update(state_coupling_diagnostics(model, state))
                y = eval_factors.y_vec
                pair_row = metric_row(
                    method=method_key,
                    eval_mode="ohsvgp_heldout_seen_history_block_pair",
                    block_id=block_id,
                    y=y,
                    mean=mean,
                    var=var,
                    runtime=runtime,
                    diagnostics=diagnostics,
                    dataset=dataset,
                    args=args,
                    seed=seed,
                    sigma2=sigma2,
                    ell_t=ell_t,
                    ell_score=ell_score,
                )
                pair_row["train_block_id"] = block_id
                pair_row["eval_block_id"] = eval_block_id
                pair_row["train_block_1based"] = block_id + 1
                pair_row["eval_block_1based"] = eval_block_id + 1
                pair_row["heldout_split_seed"] = int(heldout_split_seed)
                pair_row["heldout_split_axis"] = "spatial"
                pair_row["heldout_test_fraction"] = float(args.heldout_test_fraction)
                pair_row["num_train_locations"] = int(train_spatial_idx.size)
                pair_row["num_test_locations"] = int(test_spatial_idx.size)
                pair_row["num_train"] = int((block_id + 1) * args.block_size * train_spatial_idx.size)
                pair_row["routeb_raw_time_scale"] = raw_time_scale
                pair_row["ell_t_grid_scores"] = ell_grid_scores
                ohsvgp_pair_rows.append(pair_row)
        old_z = z_t
        old_block = block
    return rows, block_pair_rows, horizon_rows, ohsvgp_pair_rows, per_location_rows


def plot_combined(summary: list[dict[str, Any]], outdir: Path) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["rmse", "nll", "coverage90", "ece", "avg_var", "avg_width90"]
    modes = ["current", "seen_history", "future"]
    methods = sorted({row["method"] for row in summary})
    for metric in metrics:
        fig, axes = plt.subplots(1, len(modes), figsize=(4.6 * len(modes), 3.3), squeeze=False)
        for ax, mode in zip(axes[0], modes):
            vals = []
            labels = []
            for method in methods:
                found = [row for row in summary if row["method"] == method and row["eval_mode"] == mode and metric in row]
                if not found:
                    continue
                labels.append(method)
                vals.append(float(found[0][metric]))
            ax.bar(np.arange(len(vals)), vals)
            ax.set_title(mode)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=40, ha="right")
            ax.set_ylabel(metric)
            if metric == "coverage90":
                ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
        fig.tight_layout()
        fig.savefig(plot_dir / f"era5_combined_{metric}.png", dpi=180)
        plt.close(fig)


def plot_forgetting_curves(summary: list[dict[str, Any]], outdir: Path) -> None:
    if not summary:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = ["no_transfer", "mean_field", "structured_joint"]
    for metric in ["rmse_forgetting", "nll_forgetting"]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for method in methods:
            rows = [row for row in summary if row["method"] == method and metric in row]
            if not rows:
                continue
            rows = sorted(rows, key=lambda row: int(row["online_block_index"]))
            x = np.asarray([int(row["online_block_index"]) for row in rows], dtype=int)
            y = np.asarray([float(row[metric]) for row in rows], dtype=float)
            se = np.asarray([float(row.get(f"{metric}_se", 0.0)) for row in rows], dtype=float)
            ax.plot(x, y, marker="o", linewidth=1.8, label=method)
            if np.any(se > 0):
                ax.fill_between(x, y - se, y + se, alpha=0.16)
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("online block index")
        ax.set_ylabel(metric)
        ax.set_title(f"ERA5 seen-history {metric.replace('_', ' ')}")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"era5_seen_history_{metric}.png", dpi=180)
        plt.close(fig)


def plot_future_horizon_curves(summary: list[dict[str, Any]], outdir: Path) -> None:
    if not summary:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["rmse", "nll", "coverage90", "ece", "avg_predictive_variance", "avg_nu_star"]
    series_keys = sorted({(str(row["method"]), str(row.get("future_basis_mode", "observed"))) for row in summary})
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for method, future_basis_mode in series_keys:
            rows = [
                row
                for row in summary
                if row["method"] == method and row.get("future_basis_mode", "observed") == future_basis_mode and metric in row
            ]
            if not rows:
                continue
            rows = sorted(rows, key=lambda row: int(row["horizon_index"]))
            x = np.asarray([int(row["horizon_index"]) for row in rows], dtype=int)
            y = np.asarray([float(row[metric]) for row in rows], dtype=float)
            se = np.asarray([float(row.get(f"{metric}_se", 0.0)) for row in rows], dtype=float)
            label = f"{method}, {future_basis_mode}"
            ax.plot(x, y, marker="o", linewidth=1.8, label=label)
            if np.any(se > 0):
                ax.fill_between(x, y - se, y + se, alpha=0.14)
        if metric == "coverage90":
            ax.axhline(0.9, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("future horizon inside block")
        ax.set_ylabel(metric)
        ax.set_title(f"ERA5 future horizon diagnostic: {metric}")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"era5_future_horizon_{metric}.png", dpi=180)
        plt.close(fig)


def plot_capacity(summary: list[dict[str, Any]], outdir: Path) -> None:
    if not summary:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in summary if row["method"] == "structured_joint" and row["eval_mode"] == "seen_history"]
    if not rows:
        return
    labels = [f"Mt={row['mt']},Ms={row['ms']}" for row in rows]
    for metric in ["rmse", "nll", "coverage90", "ece", "avg_nu_star", "avg_predictive_variance", "beta_u_coupling_ratio", "runtime_per_block"]:
        vals = [float(row[metric]) for row in rows]
        fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(labels)), 3.4))
        ax.bar(np.arange(len(labels)), vals)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"Route B capacity sweep, seen_history: {metric}")
        if metric == "coverage90":
            ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
        fig.tight_layout()
        fig.savefig(plot_dir / f"era5_routeb_capacity_{metric}.png", dpi=180)
        plt.close(fig)


def format_metric(row: dict[str, Any], metric: str) -> str:
    if metric not in row:
        return "NA"
    se = row.get(f"{metric}_se", 0.0)
    return f"{float(row[metric]):.4f} +/- {float(se):.4f}"


def write_markdown_report(
    *,
    outdir: Path,
    baseline_summary: list[dict[str, Any]],
    routeb_summary: list[dict[str, Any]],
    combined_summary: list[dict[str, Any]],
    args: argparse.Namespace,
    dataset_shape: dict[str, int],
) -> None:
    lines = [
        "# ERA5 Baselines + Route B Comparison",
        "",
        "This report reuses the existing processed HiPPO-SVGP ERA5 baseline outputs and adds Route B internal methods on the same task, location subset, standardization, block split, and metrics.",
        "",
        "## Shared Data Protocol",
        "",
        f"- calibration tasks: `{', '.join(args.calibration_tasks)}`",
        f"- online/evaluation tasks: `{', '.join(args.online_tasks)}`",
        f"- split: `{args.split}`",
        f"- variable_index: `{args.variable_index}`",
        f"- scaled targets: `True`",
        f"- seeds/location subsets: `{args.seeds}`",
        f"- random_n_locations: `{args.random_n_locations}`",
        f"- first_n_locations: `{args.first_n_locations}`",
        f"- max_time: `{args.max_time}`",
        f"- block_size: `{args.block_size}`",
        f"- loaded shape: `T={dataset_shape['T']}, S={dataset_shape['S']}, p={dataset_shape['p']}`",
        "",
        "## Route B Protocol",
        "",
        f"- methods: `{', '.join(args.routeb_methods)}`",
        f"- phi mode: `{args.phi_mode}`",
        f"- future basis mode: `{args.future_basis_mode}`",
        f"- M_t: `{args.mt}`",
        f"- M_s: `{args.ms}`",
        f"- ell_t fit mode: `{args.ell_t_fit_mode}`",
        f"- ell_t fit dataset: `{getattr(args, 'ell_t_fit_dataset', 'calibration_task_full')}`",
        f"- ell_t grid: `{args.ell_t_grid}`",
        f"- sigma2 source: calibration-task beta-ridge residual unless `--routeb-noise` is provided",
        f"- kernel variance: `{args.kernel_variance}`",
        f"- map snapshots: `{args.save_map_snapshots}`",
        f"- map snapshot directory: `{args.map_outdir}`",
        "",
        "## Leakage and Metric Checks",
        "",
        "- current: update on block n, then evaluate block n.",
        "- seen_history: update on block n, then evaluate blocks 1..n.",
        "- future: update on block n, then predict block n+1 only; future labels are used only after prediction for metrics.",
        "- NLL, coverage, ECE, RMSE, and MAE are computed on the same scaled `Y` used by baselines.",
        "- Route B predictive variance includes sigma2, sparse conditional residual variance, u posterior term, and beta/Schur term.",
        "- All Route B internal methods share the same selected ell_t and sigma2 for a given seed.",
        "- Kernel hyperparameters are selected on the calibration task only, then frozen for the online task.",
        "- `lag_ar` Phi mode is a forecasting diagnostic only: it appends lagged observations to Phi and uses no future-block labels when building future-block features.",
        "- `extended` future-basis mode is a forecasting diagnostic only: it uses future time coordinates and the future interval length, transfers old likelihood statistics to that prediction basis, and does not use future labels.",
        "",
        "## Route B Summary",
        "",
        "| method | eval_mode | RMSE | NLL | Coverage90 | ECE | Avg var | Width90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in routeb_summary:
        lines.append(
            f"| {row['method']} | {row['eval_mode']} | {format_metric(row, 'rmse')} | {format_metric(row, 'nll')} | "
            f"{format_metric(row, 'coverage90')} | {format_metric(row, 'ece')} | {format_metric(row, 'avg_var')} | {format_metric(row, 'avg_width90')} |"
        )
    lines.extend(
        [
            "",
            "## Combined Summary",
            "",
            "| method | eval_mode | RMSE | NLL | Coverage90 | ECE | Avg var | Width90 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in combined_summary:
        lines.append(
            f"| {row['method']} | {row['eval_mode']} | {format_metric(row, 'rmse')} | {format_metric(row, 'nll')} | "
            f"{format_metric(row, 'coverage90')} | {format_metric(row, 'ece')} | {format_metric(row, 'avg_var')} | {format_metric(row, 'avg_width90')} |"
        )
    lines.extend(
        [
            "",
            "## Plot Paths",
            "",
            "- `plots/era5_combined_rmse.png`",
            "- `plots/era5_combined_nll.png`",
            "- `plots/era5_combined_coverage90.png`",
            "- `plots/era5_combined_ece.png`",
            "- `plots/era5_combined_avg_var.png`",
            "- `plots/era5_combined_avg_width90.png`",
        ]
    )
    (outdir / "era5_routeb_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_capacity_report(outdir: Path, capacity_summary: list[dict[str, Any]], args: argparse.Namespace) -> None:
    rows = [row for row in capacity_summary if row["method"] == "structured_joint" and row["eval_mode"] == "seen_history"]
    best_rmse = min(rows, key=lambda row: float(row["rmse"])) if rows else None
    best_nll = min(rows, key=lambda row: float(row["nll"])) if rows else None
    lines = [
        "# ERA5 Route B Capacity Sweep",
        "",
        "This sweep does not change the Route B core formulas. It changes only sparse basis capacity.",
        "",
        "## Protocol",
        "",
        f"- calibration task: `{', '.join(args.calibration_tasks)}`",
        f"- online task: `{', '.join(args.online_tasks)}`",
        f"- random location subsets: `{args.seeds}`",
        f"- locations: `{args.random_n_locations}`",
        f"- block_size: `{args.block_size}`",
        f"- M_t grid: `{args.mt_grid}`",
        f"- M_s grid: `{args.ms_grid}`",
        f"- eval modes: `{args.eval_modes}`",
        f"- ell_t fit mode: `{args.ell_t_fit_mode}`",
        f"- ell_t fit dataset: `{getattr(args, 'ell_t_fit_dataset', 'calibration_task_full')}`",
        "",
    ]
    if best_rmse is not None:
        lines.extend(
            [
                "## Best Seen-History Configurations",
                "",
                f"- Best RMSE: `M_t={best_rmse['mt']}, M_s={best_rmse['ms']}` with RMSE `{float(best_rmse['rmse']):.4f}`, NLL `{float(best_rmse['nll']):.4f}`.",
                f"- Best NLL: `M_t={best_nll['mt']}, M_s={best_nll['ms']}` with RMSE `{float(best_nll['rmse']):.4f}`, NLL `{float(best_nll['nll']):.4f}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Structured Joint Seen-History Summary",
            "",
            "| M_t | M_s | RMSE | NLL | Cov90 | ECE | avg_nu_star | avg_sigma2 | u term | beta/Schur | avg var | coupling ratio | runtime/block |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['mt']} | {row['ms']} | {float(row['rmse']):.4f} | {float(row['nll']):.4f} | "
            f"{float(row['coverage90']):.4f} | {float(row['ece']):.4f} | {float(row['avg_nu_star']):.4f} | "
            f"{float(row['avg_sigma2']):.4f} | {float(row['avg_u_posterior_term']):.4f} | "
            f"{float(row['avg_beta_schur_term']):.4f} | {float(row['avg_predictive_variance']):.4f} | "
            f"{float(row['beta_u_coupling_ratio']):.4f} | {float(row['runtime_per_block']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- If RMSE/NLL improve as M_t or M_s increases and avg_nu_star decreases, the gap is likely sparse-basis capacity.",
            "- If avg_nu_star remains large and coverage remains above 0.9, the model is still conservative.",
            "- If beta-u coupling ratio is small, ERA5's real Phi features do not strongly trigger the Route B cross-covariance mechanism.",
            "- Scaling to 400/1000 locations is justified only if increasing M_s gives a clear RMSE/NLL improvement per runtime cost.",
            "",
            "## Plots",
            "",
            "- `plots/era5_routeb_capacity_rmse.png`",
            "- `plots/era5_routeb_capacity_nll.png`",
            "- `plots/era5_routeb_capacity_coverage90.png`",
            "- `plots/era5_routeb_capacity_avg_nu_star.png`",
            "- `plots/era5_routeb_capacity_beta_u_coupling_ratio.png`",
        ]
    )
    (outdir / "era5_routeb_capacity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_forgetting_report(outdir: Path, forgetting_summary: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# ERA5 Route B Seen-History Forgetting Curves",
        "",
        "Forgetting is computed from saved block-pair metrics, not from aggregate summary rows.",
        "",
        "After training through online block `n`, each method is evaluated separately on every old block `B_j` with `j < n`.",
        "For metric `M`, the block-level forgetting contribution is `M_{n,j} - M_{j,j}`.",
        "The reported curve averages this quantity over all `j < n`; lower is better for both RMSE and NLL.",
        "",
        "## Protocol",
        "",
        f"- calibration tasks: `{', '.join(args.calibration_tasks)}`",
        f"- online tasks: `{', '.join(args.online_tasks)}`",
        f"- phi mode: `{args.phi_mode}`",
        f"- block size: `{args.block_size}`",
        f"- methods: `{', '.join(args.routeb_methods)}`",
        "",
        "## Output files",
        "",
        "- `era5_routeb_block_pair_metrics.csv`",
        "- `era5_routeb_forgetting_curve.csv`",
        "- `era5_routeb_forgetting_curve_summary.csv`",
        "- `plots/era5_seen_history_rmse_forgetting.png`",
        "- `plots/era5_seen_history_nll_forgetting.png`",
        "",
        "## Final block values",
        "",
        "| method | online block | RMSE forgetting | NLL forgetting |",
        "|---|---:|---:|---:|",
    ]
    final_by_method: dict[str, dict[str, Any]] = {}
    for row in forgetting_summary:
        method = str(row["method"])
        if method not in final_by_method or int(row["online_block_index"]) > int(final_by_method[method]["online_block_index"]):
            final_by_method[method] = row
    for method, row in sorted(final_by_method.items()):
        lines.append(
            f"| {method} | {int(row['online_block_index'])} | {float(row['rmse_forgetting']):.4f} | {float(row['nll_forgetting']):.4f} |"
        )
    (outdir / "era5_routeb_forgetting_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_ohsvgp_forgetting_curves(summary: list[dict[str, Any]], outdir: Path) -> None:
    if not summary:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = ["no_transfer", "mean_field", "structured_joint"]
    for metric in ["rmse_forgetting", "nll_forgetting"]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for method in methods:
            rows = [row for row in summary if row["method"] == method and metric in row]
            if not rows:
                continue
            rows = sorted(rows, key=lambda row: int(row["online_block_index"]))
            x = np.asarray([int(row["online_block_index"]) + 1 for row in rows], dtype=int)
            y = np.asarray([float(row[metric]) for row in rows], dtype=float)
            se = np.asarray([float(row.get(f"{metric}_se", 0.0)) for row in rows], dtype=float)
            ax.plot(x, y, marker="o", linewidth=1.8, label=method)
            if np.any(se > 0):
                ax.fill_between(x, y - se, y + se, alpha=0.16)
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("trained through online block n")
        ax.set_ylabel(metric)
        ax.set_title(f"OHSVGP-style held-out {metric.replace('_', ' ')}")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"era5_ohsvgp_heldout_{metric}.png", dpi=180)
        plt.close(fig)


def plot_ohsvgp_panel(block_pair_rows: list[dict[str, Any]], outdir: Path, panel_blocks: list[int]) -> None:
    if not block_pair_rows:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = ["no_transfer", "mean_field", "structured_joint"]
    available = {int(row["eval_block_id"]) + 1 for row in block_pair_rows}
    target_blocks = [int(block) for block in panel_blocks if int(block) in available]
    if not target_blocks:
        return
    for metric in ["nll", "rmse"]:
        fig, axes = plt.subplots(1, len(target_blocks), figsize=(4.0 * len(target_blocks), 3.4), squeeze=False)
        for ax, eval_block_1based in zip(axes[0], target_blocks):
            for method in methods:
                rows = [row for row in block_pair_rows if row["method"] == method and int(row["eval_block_id"]) + 1 == eval_block_1based]
                by_train: dict[int, list[float]] = {}
                for row in rows:
                    by_train.setdefault(int(row["train_block_id"]) + 1, []).append(float(row[metric]))
                if not by_train:
                    continue
                x = np.asarray(sorted(by_train), dtype=int)
                y = np.asarray([float(np.mean(by_train[int(value)])) for value in x], dtype=float)
                se = np.asarray(
                    [
                        float(np.std(by_train[int(value)], ddof=1) / np.sqrt(len(by_train[int(value)])))
                        if len(by_train[int(value)]) > 1
                        else 0.0
                        for value in x
                    ],
                    dtype=float,
                )
                ax.plot(x, y, marker="o", linewidth=1.6, label=method)
                if np.any(se > 0):
                    ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.14)
            ax.set_title(f"held-out block j={eval_block_1based}")
            ax.set_xlabel("trained through block n")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.2)
        axes[0, 0].legend()
        fig.suptitle(f"OHSVGP-style held-out {metric.upper()} after later tasks")
        fig.tight_layout()
        fig.savefig(plot_dir / f"era5_ohsvgp_panel_{metric}.png", dpi=180)
        plt.close(fig)


def write_ohsvgp_heldout_report(
    outdir: Path,
    heldout_summary: list[dict[str, Any]],
    forgetting_summary: list[dict[str, Any]],
    independent_summary: list[dict[str, Any]],
    final_forgetting_summary: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# ERA5 OHSVGP-Style Held-Out Seen-History Evaluation",
        "",
        "This evaluation follows the spirit of OHSVGP ERA5 continual-learning evaluation: after learning task/block `n`, the model is evaluated on held-out test data from already seen blocks, including old blocks learned earlier.",
        "It is not the main next-block future forecast diagnostic.",
        "",
        "## Protocol",
        "",
        f"- calibration tasks: `{', '.join(args.calibration_tasks)}`",
        f"- online tasks: `{', '.join(args.online_tasks)}`",
        f"- block size: `{args.block_size}`",
        f"- methods: `{', '.join(args.routeb_methods)}`",
        f"- held-out split axis: `spatial`",
        f"- held-out test fraction: `{args.heldout_test_fraction}`",
        f"- held-out split seeds: `{args.heldout_split_seeds}`",
        f"- panel blocks: `{args.ohsvgp_panel_blocks}`",
        "",
        "For each online block `B_j`, the update uses only `B_j_train`.",
        "Held-out seen-history evaluation after training through block `n` uses `B_1_test, ..., B_n_test`.",
        "The saved block-pair metric is `M_{n,j}`, where `n` is the latest trained block and `j <= n` is the held-out test block being evaluated.",
        "",
        "A fixed spatial train/test split is used across blocks. This avoids test-label leakage while keeping the Route B Kronecker likelihood structure valid; changing the spatial observation operator per block would require changing the core model formula.",
        "",
        "Held-out forgetting is computed as `F_n(M)=average_{j<n}[M_{n,j}-M_{j,j}]` for RMSE and NLL.",
        "",
        "## Main Held-Out Summary",
        "",
        "| method | eval_mode | NLL | RMSE | Cov90 | ECE | runtime/block | num test |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in heldout_summary:
        lines.append(
            f"| {row['method']} | {row['eval_mode']} | {format_metric(row, 'nll')} | {format_metric(row, 'rmse')} | "
            f"{format_metric(row, 'coverage90')} | {format_metric(row, 'ece')} | {format_metric(row, 'runtime_per_block')} | "
            f"{format_metric(row, 'num_test')} |"
        )
    lines.extend(
        [
            "",
            "## Independent-Run Mean, Std, and 95% CI",
            "",
            "The following table first averages block-pair results within each held-out split run, then computes mean/std/95% CI across independent split runs.",
            "",
            "| method | runs | NLL | RMSE | Cov90 | ECE | runtime/block |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in independent_summary:
        lines.append(
            f"| {row['method']} | {int(row['num_runs'])} | "
            f"{float(row['nll']):.4f} +/- {float(row['nll_std']):.4f} (95% CI {float(row['nll_ci95']):.4f}) | "
            f"{float(row['rmse']):.4f} +/- {float(row['rmse_std']):.4f} (95% CI {float(row['rmse_ci95']):.4f}) | "
            f"{float(row['coverage90']):.4f} +/- {float(row['coverage90_std']):.4f} (95% CI {float(row['coverage90_ci95']):.4f}) | "
            f"{float(row['ece']):.4f} +/- {float(row['ece_std']):.4f} (95% CI {float(row['ece_ci95']):.4f}) | "
            f"{float(row['runtime_per_block']):.4f} +/- {float(row['runtime_per_block_std']):.4f} (95% CI {float(row['runtime_per_block_ci95']):.4f}) |"
        )
    lines.extend(
        [
            "",
            "## Final Held-Out Forgetting",
            "",
            "| method | final block n | RMSE forgetting | NLL forgetting |",
            "|---|---:|---:|---:|",
        ]
    )
    final_by_method: dict[str, dict[str, Any]] = {}
    for row in forgetting_summary:
        method = str(row["method"])
        if method not in final_by_method or int(row["online_block_index"]) > int(final_by_method[method]["online_block_index"]):
            final_by_method[method] = row
    for method, row in sorted(final_by_method.items()):
        lines.append(
            f"| {method} | {int(row['online_block_index']) + 1} | {float(row['rmse_forgetting']):.4f} | {float(row['nll_forgetting']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Final Forgetting Across Independent Runs",
            "",
            "| method | runs | final NLL forgetting | final RMSE forgetting |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in final_forgetting_summary:
        lines.append(
            f"| {row['method']} | {int(row['num_runs'])} | "
            f"{float(row['nll_forgetting']):.4f} +/- {float(row['nll_forgetting_std']):.4f} (95% CI {float(row['nll_forgetting_ci95']):.4f}) | "
            f"{float(row['rmse_forgetting']):.4f} +/- {float(row['rmse_forgetting_std']):.4f} (95% CI {float(row['rmse_forgetting_ci95']):.4f}) |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `era5_ohsvgp_heldout_block_pair_metrics.csv`",
            "- `era5_ohsvgp_heldout_summary.csv`",
            "- `era5_ohsvgp_heldout_forgetting_curve.csv`",
            "- `era5_ohsvgp_heldout_forgetting_summary.csv`",
            "- `era5_ohsvgp_heldout_independent_run_summary.csv`",
            "- `era5_ohsvgp_heldout_final_forgetting_independent_run_summary.csv`",
            "- `plots/era5_ohsvgp_panel_nll.png`",
            "- `plots/era5_ohsvgp_panel_rmse.png`",
            "- `plots/era5_ohsvgp_heldout_nll_forgetting.png`",
            "- `plots/era5_ohsvgp_heldout_rmse_forgetting.png`",
        ]
    )
    (outdir / "era5_ohsvgp_heldout_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_future_horizon_report(outdir: Path, horizon_summary: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# ERA5 Future Horizon Diagnostic",
        "",
        "This diagnostic breaks each future block prediction into horizon positions inside the block.",
        "For `block_size=10`, horizon `h=1` is the first unseen time point in `B_{n+1}`, and `h=10` is the last unseen time point in that same block.",
        "",
        "The Route B core formulas are unchanged. The diagnostic only changes how metrics are aggregated after prediction.",
        "",
        "## Protocol",
        "",
        f"- calibration tasks: `{', '.join(args.calibration_tasks)}`",
        f"- online tasks: `{', '.join(args.online_tasks)}`",
        f"- phi mode: `{args.phi_mode}`",
        f"- future basis mode: `{args.future_basis_mode}`",
        f"- block size: `{args.block_size}`",
        f"- methods: `{', '.join(args.routeb_methods)}`",
        "",
        "## Output files",
        "",
        "- `era5_routeb_future_horizon_metrics.csv`",
        "- `era5_routeb_future_horizon_summary.csv`",
        "- `plots/era5_future_horizon_rmse.png`",
        "- `plots/era5_future_horizon_nll.png`",
        "- `plots/era5_future_horizon_coverage90.png`",
        "- `plots/era5_future_horizon_avg_predictive_variance.png`",
        "",
        "## Summary",
        "",
        "| method | future basis | h | RMSE | NLL | Cov90 | ECE | Avg var | avg_nu_star |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in horizon_summary:
        lines.append(
            f"| {row['method']} | {row.get('future_basis_mode', 'observed')} | {int(row['horizon_index'])} | "
            f"{float(row.get('rmse', np.nan)):.4f} | {float(row.get('nll', np.nan)):.4f} | "
            f"{float(row.get('coverage90', np.nan)):.4f} | {float(row.get('ece', np.nan)):.4f} | "
            f"{float(row.get('avg_predictive_variance', np.nan)):.4f} | {float(row.get('avg_nu_star', np.nan)):.4f} |"
        )
    (outdir / "era5_routeb_future_horizon_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_baseline_args(outdir: Path) -> dict[str, Any]:
    report_path = outdir / "era5_baseline_report.json"
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8")).get("args", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/experiments_era5_baselines_tiny_gpytorch")
    parser.add_argument("--reuse-baseline-args", action="store_true", default=False)
    parser.add_argument("--root", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--calibration-tasks", nargs="+", default=None)
    parser.add_argument("--online-tasks", nargs="+", default=None)
    parser.add_argument("--variable-index", type=int, default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default=None)
    parser.add_argument("--first-n-locations", type=int, default=None)
    parser.add_argument("--random-n-locations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-time", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--routeb-methods", nargs="+", choices=["no_transfer", "mean_field", "structured_joint"], default=["no_transfer", "mean_field", "structured_joint"])
    parser.add_argument("--eval-modes", nargs="+", choices=["current", "seen_history", "batch", "future"], default=None)
    parser.add_argument("--mt", type=int, default=8)
    parser.add_argument("--ms", type=int, default=16)
    parser.add_argument("--temporal-backend", choices=["analytic_hippo_rff", "inducing_points"], default="analytic_hippo_rff")
    parser.add_argument("--temporal-inducing-mode", choices=["moving", "global"], default="moving")
    parser.add_argument("--temporal-rff-sample-size", type=int, default=256)
    parser.add_argument("--temporal-rff-seed", type=int, default=0)
    parser.add_argument("--capacity-sweep", action="store_true")
    parser.add_argument("--mt-grid", nargs="+", type=int, default=[8, 12, 16])
    parser.add_argument("--ms-grid", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--model-ell-t", type=float, default=0.25)
    parser.add_argument("--ell-t-fit-mode", choices=["none", "initial_task_fullgp"], default="initial_task_fullgp")
    parser.add_argument("--ell-t-grid", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6])
    parser.add_argument("--routeb-noise", type=float, default=None)
    parser.add_argument("--hyperparam-fit-mode", choices=["none", "initial_task_fullgp_grid"], default="none")
    parser.add_argument("--noise-grid", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    parser.add_argument("--kernel-variance-grid", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument("--hyperparam-fit-max-time", type=int, default=30)
    parser.add_argument("--hyperparam-fit-max-locations", type=int, default=30)
    parser.add_argument("--kernel-variance", type=float, default=1.0)
    parser.add_argument("--kernel-type", choices=["rbf", "matern32", "ard_rbf", "spectral_mixture"], default="rbf")
    parser.add_argument("--num-mixtures", type=int, default=4)
    parser.add_argument("--spectral-mixture-param-path", default=None)
    parser.add_argument("--spatial-lengthscale", type=float, default=0.35)
    parser.add_argument("--spatial-inducing-selection", choices=["linspace", "farthest", "kmeans"], default="linspace")
    parser.add_argument("--spatial-ard-lengthscales", nargs=2, type=float, default=None)
    parser.add_argument("--beta-prior-variance", type=float, default=10.0)
    parser.add_argument("--lag-train-noise-std", type=float, default=0.0)
    parser.add_argument("--xlag-length", type=int, default=1)
    parser.add_argument("--lag-train-noise-seed-offset", type=int, default=10000)
    parser.add_argument("--lag-beta-prior-variance", type=float, default=None)
    parser.add_argument("--safe-lag-variance-scale", type=float, default=1.0)
    parser.add_argument("--safe-lag-variance-add", type=float, default=0.0)
    parser.add_argument("--inner-iters", type=int, default=2)
    parser.add_argument(
        "--phi-mode",
        choices=[
            "base",
            "direct_y",
            "minimal",
            "rich_v1",
            "rich_v2",
            "rich_v3",
            "rich_seasonal_spatial",
            "engineered",
            "lag_ar",
            "rich_v3_lag_ar",
            "medium_era5",
            "medium_era5_xlag",
            "medium_era5_oracle_ylag",
            "rich_era5",
        ],
        default="base",
    )
    parser.add_argument("--future-basis-mode", choices=["observed", "extended"], default="observed")
    parser.add_argument("--prediction-mode", choices=["dense", "streaming_sylvester"], default="streaming_sylvester")
    parser.add_argument("--prediction-chunk-size", type=int, default=8192)
    parser.add_argument("--save-forgetting-block-pairs", action="store_true")
    parser.add_argument("--save-future-horizon-metrics", action="store_true")
    parser.add_argument("--ohsvgp-heldout-eval", action="store_true")
    parser.add_argument(
        "--skip-heldout-block-pairs",
        action="store_true",
        help="Skip the quadratic held-out train/eval block-pair diagnostic when only per-block seen-history metrics are required.",
    )
    parser.add_argument("--heldout-test-fraction", type=float, default=0.2)
    parser.add_argument("--heldout-split-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--ohsvgp-panel-blocks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--save-map-snapshots", action="store_true")
    parser.add_argument("--save-per-location-predictions", action="store_true")
    parser.add_argument("--per-location-predictions-path", default=None)
    parser.add_argument("--per-location-indices", nargs="+", type=int, default=None)
    parser.add_argument("--map-outdir", default="outputs/stvgp_kronecker_maps/routeB_task1_calibration_task2_online/test")
    parser.add_argument("--map-snapshot-method", choices=["no_transfer", "mean_field", "structured_joint"], default="structured_joint")
    parser.add_argument("--map-snapshot-eval-mode", choices=["current", "seen_history", "future"], default="current")
    parser.add_argument("--map-snapshot-eval-modes", nargs="+", choices=["current", "seen_history", "future"], default=None)
    parser.add_argument("--map-snapshot-seed", type=int, default=0)
    parser.add_argument("--map-snapshot-time-indices", nargs="+", type=int, default=None)
    parser.add_argument("--map-variable-label", default="ERA5 scaled variable")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    outdir = Path(args.outdir)
    baseline_args = load_baseline_args(outdir) if args.reuse_baseline_args else {}
    defaults = {
        "root": "data/era5/processed_timeseries_4",
        "variable_index": 0,
        "split": "all",
        "first_n_locations": None,
        "random_n_locations": None,
        "seed": 0,
        "seeds": None,
        "max_time": None,
        "block_size": 10,
        "eval_modes": ["current", "seen_history", "future"],
    }
    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, baseline_args.get(key, default))
    if args.calibration_tasks is None:
        args.calibration_tasks = ["task_1"]
    if args.online_tasks is None:
        args.online_tasks = args.tasks if args.tasks is not None else ["task_2"]
    args.tasks = args.online_tasks
    if args.seeds is None:
        args.seeds = baseline_args.get("seeds") or [args.seed]
    if args.heldout_split_seeds is None:
        args.heldout_split_seeds = args.seeds
    args.ell_t_fit_dataset = "calibration_task_full" if args.ell_t_fit_mode == "initial_task_fullgp" else "manual"
    if args.map_snapshot_eval_modes is None:
        args.map_snapshot_eval_modes = [args.map_snapshot_eval_mode]
    args.spectral_mixture_params = _load_spectral_mixture_params(args.spectral_mixture_param_path)
    if args.spectral_mixture_params is not None and args.kernel_type != "spectral_mixture":
        raise ValueError("--spectral-mixture-param-path requires --kernel-type spectral_mixture")
    return args


def main() -> None:
    args = resolve_args(parse_args())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    routeb_rows: list[dict[str, Any]] = []
    block_pair_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    ohsvgp_pair_rows: list[dict[str, Any]] = []
    per_location_rows: list[dict[str, Any]] = []
    dataset_shape: dict[str, int] | None = None
    capacities = [(args.mt, args.ms)] if not args.capacity_sweep else [(mt, ms) for mt in args.mt_grid for ms in args.ms_grid]
    for mt, ms in capacities:
        args.mt = int(mt)
        args.ms = int(ms)
        for seed in args.seeds:
            calibration_raw = load_hipposvgp_era5(
                args.root,
                tasks=args.calibration_tasks,
                variable_index=args.variable_index,
                prefer_scaled=True,
                split=args.split,
                first_n_locations=args.first_n_locations,
                random_n_locations=args.random_n_locations,
                seed=int(seed),
                max_time=args.max_time,
            )
            selected_locations = selected_locations_from_dataset(calibration_raw)
            online_raw = load_hipposvgp_era5(
                args.root,
                tasks=args.online_tasks,
                variable_index=args.variable_index,
                prefer_scaled=True,
                split=args.split,
                selected_locations=selected_locations,
                seed=int(seed),
                max_time=args.max_time,
            )
            calibration_dataset, calibration_time_scale = normalise_time_dataset(calibration_raw)
            calibration_dataset = augment_dataset_phi(calibration_dataset, phi_mode=args.phi_mode, xlag_length=args.xlag_length)
            if args.hyperparam_fit_mode == "initial_task_fullgp_grid":
                ell_t, sigma2, selected_kernel_variance, ell_score, ell_grid_scores = select_hyperparams_from_calibration_fullgp_mll(
                    calibration_dataset,
                    args,
                )
                args.kernel_variance = selected_kernel_variance
                args.sigma2_source = "calibration_task_fullgp_mll_grid"
            elif args.routeb_noise is None:
                sigma2 = estimate_sigma2_from_calibration_task(calibration_dataset)
                args.sigma2_source = "calibration_task_beta_ridge_residual"
                ell_t, ell_score, ell_grid_scores = select_ell_t_from_calibration_task(calibration_dataset, args, sigma2)
            else:
                sigma2 = float(args.routeb_noise) ** 2
                args.sigma2_source = "manual_routeb_noise"
                ell_t, ell_score, ell_grid_scores = select_ell_t_from_calibration_task(calibration_dataset, args, sigma2)
            dataset_shape = {
                "T": online_raw.Y.shape[0],
                "S": online_raw.Y.shape[1],
                "p_raw": online_raw.Phi.shape[1],
                "p": augment_dataset_phi(online_raw, phi_mode=args.phi_mode, xlag_length=args.xlag_length).Phi.shape[1],
            }
            split_seeds = args.heldout_split_seeds if args.ohsvgp_heldout_eval else [int(seed)]
            for heldout_split_seed in split_seeds:
                for method in args.routeb_methods:
                    method_rows, method_pair_rows, method_horizon_rows, method_ohsvgp_rows, method_per_location_rows = run_routeb_method(
                        online_raw,
                        seed=int(seed),
                        heldout_split_seed=int(heldout_split_seed),
                        method=method,
                        args=args,
                        sigma2=sigma2,
                        ell_t=ell_t,
                        ell_score=ell_score,
                        ell_grid_scores=ell_grid_scores,
                        raw_time_scale=calibration_time_scale,
                    )
                    routeb_rows.extend(method_rows)
                    block_pair_rows.extend(method_pair_rows)
                    horizon_rows.extend(method_horizon_rows)
                    ohsvgp_pair_rows.extend(method_ohsvgp_rows)
                    per_location_rows.extend(method_per_location_rows)

    routeb_summary = summary_rows(routeb_rows)
    stem = "era5_routeb_capacity" if args.capacity_sweep else "era5_routeb"
    write_csv(routeb_rows, outdir / f"{stem}_metrics.csv")
    write_csv(routeb_summary, outdir / f"{stem}_summary.csv")
    if args.save_per_location_predictions:
        per_location_path = Path(args.per_location_predictions_path) if args.per_location_predictions_path else outdir / "era5_routeb_per_location_predictions.csv"
        write_csv(per_location_rows, per_location_path)
    if args.save_forgetting_block_pairs:
        forgetting_rows = forgetting_curve_rows(block_pair_rows)
        forgetting_summary = summarize_forgetting_rows(forgetting_rows)
        write_csv(block_pair_rows, outdir / "era5_routeb_block_pair_metrics.csv")
        write_csv(forgetting_rows, outdir / "era5_routeb_forgetting_curve.csv")
        write_csv(forgetting_summary, outdir / "era5_routeb_forgetting_curve_summary.csv")
        plot_forgetting_curves(forgetting_summary, outdir)
        write_forgetting_report(outdir, forgetting_summary, args)
    if args.save_future_horizon_metrics:
        horizon_summary = horizon_summary_rows(horizon_rows)
        write_csv(horizon_rows, outdir / "era5_routeb_future_horizon_metrics.csv")
        write_csv(horizon_summary, outdir / "era5_routeb_future_horizon_summary.csv")
        plot_future_horizon_curves(horizon_summary, outdir)
        write_future_horizon_report(outdir, horizon_summary, args)
    if args.ohsvgp_heldout_eval and not args.skip_heldout_block_pairs:
        ohsvgp_summary = summarize_heldout_seen_history(ohsvgp_pair_rows)
        ohsvgp_forgetting_rows = forgetting_curve_rows(ohsvgp_pair_rows)
        ohsvgp_forgetting_summary = summarize_forgetting_rows(ohsvgp_forgetting_rows)
        ohsvgp_independent_summary = summarize_heldout_by_independent_run(ohsvgp_pair_rows)
        ohsvgp_final_forgetting_independent_summary = summarize_final_forgetting_by_independent_run(ohsvgp_forgetting_rows)
        write_csv(ohsvgp_pair_rows, outdir / "era5_ohsvgp_heldout_block_pair_metrics.csv")
        write_csv(ohsvgp_summary, outdir / "era5_ohsvgp_heldout_summary.csv")
        write_csv(ohsvgp_forgetting_rows, outdir / "era5_ohsvgp_heldout_forgetting_curve.csv")
        write_csv(ohsvgp_forgetting_summary, outdir / "era5_ohsvgp_heldout_forgetting_summary.csv")
        write_csv(ohsvgp_independent_summary, outdir / "era5_ohsvgp_heldout_independent_run_summary.csv")
        write_csv(ohsvgp_final_forgetting_independent_summary, outdir / "era5_ohsvgp_heldout_final_forgetting_independent_run_summary.csv")
        plot_ohsvgp_forgetting_curves(ohsvgp_forgetting_summary, outdir)
        plot_ohsvgp_panel(ohsvgp_pair_rows, outdir, args.ohsvgp_panel_blocks)
        write_ohsvgp_heldout_report(
            outdir,
            ohsvgp_summary,
            ohsvgp_forgetting_summary,
            ohsvgp_independent_summary,
            ohsvgp_final_forgetting_independent_summary,
            args,
        )
    capacity_summary = capacity_summary_rows(routeb_rows)
    if args.capacity_sweep:
        write_csv(capacity_summary, outdir / "era5_routeb_capacity_by_config_summary.csv")
        plot_capacity(capacity_summary, outdir)
        write_capacity_report(outdir, capacity_summary, args)

    baseline_summary = read_csv_dicts(outdir / "era5_baseline_summary.csv")
    baseline_tasks = set()
    for row in baseline_summary:
        if row.get("tasks"):
            baseline_tasks.add(row["tasks"])
    if baseline_tasks and ",".join(args.online_tasks) not in baseline_tasks:
        baseline_summary = []
    phi_summary = read_csv_dicts(outdir / "era5_phi_baseline_summary.csv")
    if phi_summary:
        phi_tasks = {row["tasks"] for row in phi_summary if row.get("tasks")}
        if not phi_tasks or ",".join(args.online_tasks) in phi_tasks:
            baseline_summary = baseline_summary + phi_summary
    combined_summary: list[dict[str, Any]] = [dict(row) for row in baseline_summary] + [dict(row) for row in routeb_summary]
    write_csv(combined_summary, outdir / "era5_combined_summary.csv")
    plot_combined(combined_summary, outdir)
    report = {
        "args": vars(args),
        "dataset_shape": dataset_shape,
        "routeb_summary": routeb_summary,
        "baseline_summary_reused": baseline_summary,
        "outputs": {
            "routeb_metrics": str(outdir / "era5_routeb_metrics.csv"),
            "routeb_summary": str(outdir / "era5_routeb_summary.csv"),
            "per_location_predictions": str(Path(args.per_location_predictions_path) if args.per_location_predictions_path else outdir / "era5_routeb_per_location_predictions.csv") if args.save_per_location_predictions else "",
            "capacity_metrics": str(outdir / "era5_routeb_capacity_metrics.csv") if args.capacity_sweep else "",
            "capacity_summary": str(outdir / "era5_routeb_capacity_by_config_summary.csv") if args.capacity_sweep else "",
            "combined_summary": str(outdir / "era5_combined_summary.csv"),
            "markdown": str(outdir / "era5_routeb_comparison_report.md"),
            "capacity_markdown": str(outdir / "era5_routeb_capacity_report.md") if args.capacity_sweep else "",
            "block_pair_metrics": str(outdir / "era5_routeb_block_pair_metrics.csv") if args.save_forgetting_block_pairs else "",
            "forgetting_curve": str(outdir / "era5_routeb_forgetting_curve.csv") if args.save_forgetting_block_pairs else "",
            "forgetting_summary": str(outdir / "era5_routeb_forgetting_curve_summary.csv") if args.save_forgetting_block_pairs else "",
            "future_horizon_metrics": str(outdir / "era5_routeb_future_horizon_metrics.csv") if args.save_future_horizon_metrics else "",
            "future_horizon_summary": str(outdir / "era5_routeb_future_horizon_summary.csv") if args.save_future_horizon_metrics else "",
            "ohsvgp_heldout_block_pair_metrics": str(outdir / "era5_ohsvgp_heldout_block_pair_metrics.csv") if args.ohsvgp_heldout_eval else "",
            "ohsvgp_heldout_summary": str(outdir / "era5_ohsvgp_heldout_summary.csv") if args.ohsvgp_heldout_eval else "",
            "ohsvgp_heldout_forgetting_curve": str(outdir / "era5_ohsvgp_heldout_forgetting_curve.csv") if args.ohsvgp_heldout_eval else "",
            "ohsvgp_heldout_forgetting_summary": str(outdir / "era5_ohsvgp_heldout_forgetting_summary.csv") if args.ohsvgp_heldout_eval else "",
            "ohsvgp_heldout_independent_summary": str(outdir / "era5_ohsvgp_heldout_independent_run_summary.csv") if args.ohsvgp_heldout_eval else "",
            "ohsvgp_heldout_final_forgetting_independent_summary": str(outdir / "era5_ohsvgp_heldout_final_forgetting_independent_run_summary.csv") if args.ohsvgp_heldout_eval else "",
            "ohsvgp_heldout_markdown": str(outdir / "era5_ohsvgp_heldout_report.md") if args.ohsvgp_heldout_eval else "",
        },
    }
    (outdir / "era5_routeb_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(
        outdir=outdir,
        baseline_summary=baseline_summary,
        routeb_summary=routeb_summary,
        combined_summary=combined_summary,
        args=args,
        dataset_shape=dataset_shape or {"T": 0, "S": 0, "p": 0},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
