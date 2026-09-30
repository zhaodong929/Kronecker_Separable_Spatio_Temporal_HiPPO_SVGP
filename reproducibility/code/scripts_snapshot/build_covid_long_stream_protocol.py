#!/usr/bin/env python3
"""Materialise an audited, causal 2020--2024 COVID long-stream protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


START_DATE = "2020-08-01"
END_DATE = "2024-04-30"
TARGET_MODES = (
    "raw_count",
    "per_100k",
    "log1p_per_100k",
)


def transform_count_target(
    counts: np.ndarray,
    population: np.ndarray,
    target_mode: str,
) -> tuple[np.ndarray, str]:
    """Return the continuous Gaussian target before Task-1 standardisation."""

    values = np.asarray(counts, dtype=np.float64)
    exposure = np.asarray(population, dtype=np.float64) / 100000.0
    if values.ndim != 2 or exposure.shape != (values.shape[1],):
        raise ValueError("counts must be [time, location] with one population per location")
    if not np.isfinite(values).all() or (values < 0.0).any() or (exposure <= 0.0).any():
        raise ValueError("COVID counts must be finite/non-negative and populations positive")
    if target_mode == "raw_count":
        return values, "weekly confirmed COVID hospital admissions"
    rate = values / exposure[None, :]
    if target_mode == "per_100k":
        return rate, "weekly confirmed COVID hospital admissions per 100,000 population"
    if target_mode == "log1p_per_100k":
        return np.log1p(rate), "log1p weekly confirmed COVID hospital admissions per 100,000 population"
    raise ValueError(f"Unsupported target mode: {target_mode}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def column_by_alias(frame: pd.DataFrame, aliases: tuple[str, ...], label: str) -> str:
    names = {normalise(name): str(name) for name in frame.columns}
    for alias in aliases:
        if alias in names:
            return names[alias]
    raise ValueError(f"CDC source has no {label} column; available columns: {list(frame.columns)}")


def target_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"--target-column {requested!r} is not present")
        return requested
    names = {normalise(name): str(name) for name in frame.columns}
    for alias in (
        "totalcovid19admissions",
        "totalcovidadmissions",
        "covid19admissions",
        "covidadmissions",
        "weeklycovid19admissions",
        "weeklycovidadmissions",
    ):
        if alias in names:
            return names[alias]
    candidates = [
        str(name)
        for name in frame.columns
        if "covid" in normalise(name) and "admission" in normalise(name)
    ]
    raise ValueError(
        "CDC target column is ambiguous. Re-run with --target-column. "
        f"COVID admission candidates: {candidates}"
    )


def census_centroids(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if len(members) != 1:
            raise ValueError(f"Expected one Gazetteer text file, found {members}")
        with archive.open(members[0]) as handle:
            header = handle.readline().decode("utf-8")
            handle.seek(0)
            frame = pd.read_csv(handle, sep="|" if "|" in header else "\t", dtype={"GEOID": str})
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame.rename(columns={"GEOID": "location"})


def current_location_frame(locations_csv: Path) -> pd.DataFrame:
    locations = pd.read_csv(locations_csv, dtype={"location": str})
    locations = locations.loc[locations["location"] != "US"].copy()
    if locations.shape[0] != 52:
        raise ValueError(f"Expected the current 52-location panel, found {locations.shape[0]}")
    locations["location"] = locations["location"].str.zfill(2)
    return locations.sort_values("location").reset_index(drop=True)


def jurisdiction_map(locations: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in locations.itertuples(index=False):
        for key in (row.location, row.abbreviation, row.location_name):
            mapping[normalise(key)] = str(row.location)
    return mapping


def split_locations(num_locations: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(num_locations)
    test = np.sort(permutation[:10])
    train = np.sort(permutation[10:])
    inner = np.random.default_rng(1729 + int(seed)).permutation(train)
    validation = np.sort(inner[:4])
    fit = np.sort(inner[4:])
    return train, fit, validation, test


def standardise_coordinates(coordinates: np.ndarray) -> np.ndarray:
    return (coordinates - coordinates.mean(axis=0, keepdims=True)) / np.maximum(
        coordinates.std(axis=0, keepdims=True), 1e-12
    )


def inducing_coordinates(coordinates: np.ndarray, train: np.ndarray, sizes: tuple[int, ...]) -> dict[int, np.ndarray]:
    candidates = coordinates[train]
    output: dict[int, np.ndarray] = {}
    for requested in sizes:
        size = min(int(requested), candidates.shape[0])
        output[int(requested)] = (
            candidates.copy()
            if size == candidates.shape[0]
            else KMeans(n_clusters=size, random_state=0, n_init=10).fit(candidates).cluster_centers_
        )
    return output


def availability_aware_phi(
    *,
    y: np.ndarray,
    coordinates_raw: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    calibration_steps: int,
    delay_weeks: int,
) -> tuple[np.ndarray, list[str]]:
    """Build state lag features using only labels available at each prediction week."""

    times, locations = y.shape
    index = np.arange(times)
    phase = 2.0 * np.pi * index / 52.1775
    train_mask = np.zeros(locations, dtype=bool)
    train_mask[train] = True
    test_mask = np.zeros(locations, dtype=bool)
    test_mask[test] = True
    lags = np.full((4, times, locations), np.nan, dtype=np.float64)
    for lag_index, lag in enumerate((1, 2, 3, 4)):
        for current in range(lag, times):
            source = current - lag
            values = y[source]
            if current < calibration_steps:
                known = np.ones(locations, dtype=bool)
            else:
                stream_now = current - calibration_steps
                known = train_mask.copy()
                known[~train_mask] = source < calibration_steps or (
                    source - calibration_steps <= stream_now - delay_weeks
                )
            lags[lag_index, current, known] = values[known]
    visible_mean = np.mean(y[:, train], axis=1)
    visible_lag1 = np.full(times, np.nan, dtype=np.float64)
    visible_lag1[1:] = visible_mean[:-1]
    lag_count = np.sum(np.isfinite(lags), axis=0)
    rolling4 = np.nansum(lags, axis=0) / np.maximum(lag_count, 1)
    rolling4[lag_count == 0] = np.nan
    dynamic = np.stack(
        [
            np.broadcast_to(index[:, None] / 51.0, (times, locations)),
            np.broadcast_to(np.sin(phase)[:, None], (times, locations)),
            np.broadcast_to(np.cos(phase)[:, None], (times, locations)),
            *lags,
            rolling4,
            lags[0] - lags[1],
            np.broadcast_to(visible_lag1[:, None], (times, locations)),
            np.broadcast_to(coordinates_raw[None, :, 0], (times, locations)),
            np.broadcast_to(coordinates_raw[None, :, 1], (times, locations)),
        ],
        axis=-1,
    )
    reference = dynamic[:calibration_steps, train].reshape(-1, dynamic.shape[-1])
    mean = np.nanmean(reference, axis=0)
    scale = np.maximum(np.nanstd(reference, axis=0), 1e-12)
    dynamic = np.where(np.isfinite(dynamic), dynamic, mean[None, None, :])
    dynamic = (dynamic - mean[None, None, :]) / scale[None, None, :]
    phi = np.concatenate([np.ones((times, locations, 1)), dynamic], axis=-1)
    return phi, [
        "intercept", "time_trend", "season_sin", "season_cos", "state_lag1",
        "state_lag2", "state_lag3", "state_lag4", "state_rolling4", "state_growth1",
        "visible_mean_lag1", "latitude", "longitude",
    ]


def ridge(phi: np.ndarray, y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    design = phi[:, indices].reshape(-1, phi.shape[-1])
    target = y[:, indices].reshape(-1)
    return np.linalg.solve(design.T @ design + 1e-3 * np.eye(phi.shape[-1]), design.T @ target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--locations-csv", type=Path, required=True)
    parser.add_argument("--census-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-column")
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--calibration-weeks", type=int, default=52)
    parser.add_argument("--reporting-tasks", type=int, default=9)
    parser.add_argument("--delay-weeks", type=int, default=1)
    parser.add_argument("--inducing-sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--target-mode", choices=TARGET_MODES, default="log1p_per_100k")
    args = parser.parse_args()

    if args.delay_weeks < 1:
        raise ValueError("--delay-weeks must be positive")
    source = args.source_csv.resolve()
    locations = current_location_frame(args.locations_csv.resolve())
    raw = pd.read_csv(source, low_memory=False)
    date_column = column_by_alias(raw, ("weekendingdate", "weekenddate", "date"), "weekly date")
    jurisdiction_column = column_by_alias(
        raw,
        ("jurisdiction", "state", "location", "reportingjurisdiction", "geographicaggregation"),
        "jurisdiction",
    )
    value_column = target_column(raw, args.target_column)
    frame = raw[[date_column, jurisdiction_column, value_column]].copy()
    frame.columns = ["date", "jurisdiction", "value"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    mapping = jurisdiction_map(locations)
    frame["location"] = frame["jurisdiction"].map(lambda value: mapping.get(normalise(value)))
    frame = frame.loc[frame["location"].notna()].copy()
    frame = frame.loc[(frame["date"] >= args.start_date) & (frame["date"] <= args.end_date)].copy()
    if frame.empty:
        raise ValueError("CDC source has no rows in the mandatory 2020-08 through 2024-04 window")
    if frame["value"].isna().any() or (frame["value"] < 0.0).any():
        raise ValueError("CDC target values must be finite and non-negative")
    duplicates = frame.duplicated(["date", "location"], keep=False)
    if duplicates.any():
        example = frame.loc[duplicates, ["date", "jurisdiction", "location"]].head(8).to_dict("records")
        raise ValueError(f"CDC source has duplicate location-week rows: {example}")
    location_codes = locations["location"].tolist()
    pivot = frame.pivot(index="date", columns="location", values="value").sort_index()
    pivot = pivot.reindex(columns=location_codes)
    if pivot.isna().any().any():
        missing = pivot.columns[pivot.isna().any()].tolist()
        raise ValueError(f"CDC source has missing mandatory-window observations for {missing}")
    dates = pd.DatetimeIndex(pivot.index)
    expected_dates = pd.date_range(dates.min(), dates.max(), freq="7D")
    if not dates.equals(expected_dates):
        missing = expected_dates.difference(dates).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"CDC weekly dates are discontinuous; missing={missing[:12]}")
    if len(dates) < 190:
        raise ValueError(f"CDC window has {len(dates)} weekly observations; need at least 190")
    if int(args.calibration_weeks) >= len(dates):
        raise ValueError("Calibration window leaves no online observations")

    centroids = census_centroids(args.census_zip.resolve()).set_index("location")
    missing_centroids = sorted(set(location_codes) - set(centroids.index))
    if missing_centroids:
        raise ValueError(f"Census centroids missing locations: {missing_centroids}")
    population = locations.set_index("location").loc[location_codes, "population"].to_numpy(dtype=np.float64)
    coordinates_raw = centroids.loc[location_codes, ["INTPTLAT", "INTPTLONG"]].to_numpy(dtype=np.float64)
    counts = pivot.to_numpy(dtype=np.float64)
    if not np.allclose(counts, np.rint(counts), atol=1e-8, rtol=0.0):
        raise ValueError("Negative-Binomial target requires integer weekly admission counts")
    y_raw, target_label = transform_count_target(counts, population, args.target_mode)
    train, fit, validation, test = split_locations(len(location_codes), args.seed)
    reference = y_raw[: args.calibration_weeks, train]
    target_mean = float(reference.mean())
    target_scale = float(max(reference.std(), 1e-12))
    y = (y_raw - target_mean) / target_scale
    phi, feature_columns = availability_aware_phi(
        y=y,
        coordinates_raw=coordinates_raw,
        train=train,
        test=test,
        calibration_steps=args.calibration_weeks,
        delay_weeks=args.delay_weeks,
    )
    beta = ridge(phi[: args.calibration_weeks], y[: args.calibration_weeks], train)
    stream_y = y[args.calibration_weeks :]
    stream_phi = phi[args.calibration_weeks :]
    chunks = np.array_split(np.arange(stream_y.shape[0]), int(args.reporting_tasks))
    if any(chunk.size == 0 for chunk in chunks):
        raise ValueError("Not enough online weeks for the requested reporting tasks")
    task_ids = np.empty(stream_y.shape[0], dtype=np.int64)
    task_start = []
    task_stop = []
    for offset, chunk in enumerate(chunks, start=2):
        task_ids[chunk] = offset
        task_start.append(int(chunk[0]))
        task_stop.append(int(chunk[-1] + 1))
    coordinates = standardise_coordinates(coordinates_raw)
    payload = {
        "train_indices": train,
        "fit_indices": fit,
        "validation_indices": validation,
        "test_indices": test,
        # The formal Route B protocol is float64 end-to-end.  Keep the on-disk
        # arrays at that precision so a downstream runner cannot silently lose it.
        "calibration_y": y[: args.calibration_weeks].astype(np.float64),
        "stream_y": stream_y.astype(np.float64),
        "calibration_counts": counts[: args.calibration_weeks].astype(np.int64),
        "stream_counts": counts[args.calibration_weeks :].astype(np.int64),
        "population": population.astype(np.float64),
        "population_per_100k": (population / 100000.0).astype(np.float64),
        "calibration_phi": phi[: args.calibration_weeks].astype(np.float64),
        "stream_phi": stream_phi.astype(np.float64),
        "task1_ridge_beta": beta,
        "task1_calibration_mean": np.einsum("tsp,p->ts", phi[: args.calibration_weeks], beta).astype(np.float64),
        "task1_stream_mean": np.einsum("tsp,p->ts", stream_phi, beta).astype(np.float64),
        "coordinates": coordinates,
        "calibration_times": np.arange(args.calibration_weeks, dtype=np.float64) / 51.0,
        "stream_times": np.arange(stream_y.shape[0], dtype=np.float64) / 51.0,
        "calibration_week_dates": dates[: args.calibration_weeks].strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        "stream_week_dates": dates[args.calibration_weeks :].strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        "block_start": np.arange(stream_y.shape[0], dtype=np.int64),
        "block_stop": np.arange(1, stream_y.shape[0] + 1, dtype=np.int64),
        "reporting_task_id": task_ids,
        "reporting_task_start": np.asarray(task_start, dtype=np.int64),
        "reporting_task_stop": np.asarray(task_stop, dtype=np.int64),
    }
    for size, values in inducing_coordinates(coordinates, train, tuple(args.inducing_sizes)).items():
        payload[f"inducing_coords_ms{size}"] = values
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    metadata = {
        "schema_version": 3,
        "dataset": "cdc_covid_nhsn_mandatory_long_stream",
        "npz": str(output),
        "source_csv": str(source),
        "source_sha256": sha256_file(source),
        "source_columns": {"date": date_column, "jurisdiction": jurisdiction_column, "target": value_column},
        "target": target_label,
        "target_mode": args.target_mode,
        "mandatory_window": {"start": args.start_date, "end": args.end_date},
        "raw_dates": [date.date().isoformat() for date in dates],
        "num_total_times": int(len(dates)),
        "num_calibration_times": int(args.calibration_weeks),
        "num_stream_times": int(stream_y.shape[0]),
        "num_blocks": int(stream_y.shape[0]),
        "num_locations": int(len(location_codes)),
        "num_train_locations": int(train.size),
        "num_fit_locations": int(fit.size),
        "num_validation_locations": int(validation.size),
        "num_test_locations": int(test.size),
        "location_codes": location_codes,
        "location_names": locations["location_name"].astype(str).tolist(),
        "split_seed": int(args.seed),
        "reporting_tasks": [
            {"task": task, "stream_start": start, "stream_stop": stop, "weeks": stop - start}
            for task, start, stop in zip(range(2, 2 + len(task_start)), task_start, task_stop)
        ],
        "xlag": {
            "mode": "availability-aware state lag4 and visible aggregate lag",
            "delay_weeks": int(args.delay_weeks),
            "features": len(feature_columns),
            "columns": feature_columns,
            "test_target_lags_used": True,
            "test_target_lag_information": "Only labels whose reporting delay has elapsed at that prediction week.",
        },
        "target_standardization": {
            "mean": target_mean,
            "scale": target_scale,
            "fit_scope": "Task-1 visible locations only",
        },
        "storage_dtype": "float64",
        "inducing_sizes_requested": [int(size) for size in args.inducing_sizes],
        "npz_sha256": sha256_file(output),
    }
    audit = {
        "status": "complete",
        "weekly_observations": int(len(dates)),
        "location_count": int(len(location_codes)),
        "date_start": dates.min().date().isoformat(),
        "date_end": dates.max().date().isoformat(),
        "no_missing_location_week_cells": True,
        "no_duplicate_location_week_cells": True,
        "mandatory_window_only": True,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "metadata": metadata, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
