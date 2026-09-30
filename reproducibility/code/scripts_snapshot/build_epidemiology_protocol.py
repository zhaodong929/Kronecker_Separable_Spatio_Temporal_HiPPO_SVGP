#!/usr/bin/env python3
"""Build rectangular, causal Route B protocols from audited disease panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spatial_split(num_locations: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(num_locations)
    test_count = max(1, int(round(0.2 * num_locations)))
    test = np.sort(permutation[:test_count])
    train = np.sort(permutation[test_count:])
    inner_rng = np.random.default_rng(1729 + int(seed))
    inner = inner_rng.permutation(train)
    validation_count = max(1, int(round(0.1 * train.size)))
    validation = np.sort(inner[:validation_count])
    fit = np.sort(inner[validation_count:])
    return train, fit, validation, test


def standardize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    return (coordinates - coordinates.mean(axis=0, keepdims=True)) / np.maximum(
        coordinates.std(axis=0, keepdims=True), 1e-12
    )


def _visible_lag_features(y: np.ndarray, train_indices: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    visible_mean = np.mean(y[:, train_indices], axis=1)
    output = np.zeros((y.shape[0], len(lags)), dtype=np.float64)
    for column, lag in enumerate(lags):
        output[lag:, column] = visible_mean[:-lag]
    return output


def build_causal_features(
    *,
    y: np.ndarray,
    coordinates: np.ndarray,
    population: np.ndarray,
    train_indices: np.ndarray,
    calibration_stop: int,
    period: float,
    feature_mode: str = "base",
) -> tuple[np.ndarray, list[str]]:
    """Features may use static data and lagged visible-location aggregates only."""

    num_times, num_locations = y.shape
    time = np.arange(num_times, dtype=np.float64)
    phase = 2.0 * np.pi * time / float(period)
    lag_features = _visible_lag_features(
        y, train_indices, (1, 2, 4) if feature_mode == "base" else (1, 2, 4, 8)
    )
    dynamic_columns = [
        "time_trend",
        "season_sin_1",
        "season_cos_1",
        "season_sin_2",
        "season_cos_2",
        "visible_mean_lag_1",
        "visible_mean_lag_2",
        "visible_mean_lag_4",
    ]
    dynamic_parts = [
        time / max(num_times - 1, 1),
        np.sin(phase),
        np.cos(phase),
        np.sin(2.0 * phase),
        np.cos(2.0 * phase),
        lag_features[:, :3],
    ]
    if feature_mode == "lag_dynamics":
        lag_1, lag_2, lag_4, lag_8 = lag_features.T
        dynamic_parts.extend(
            [
                lag_8,
                lag_1 - lag_2,
                lag_1 - lag_4,
                np.mean(lag_features[:, :3], axis=1),
            ]
        )
        dynamic_columns.extend(
            [
                "visible_mean_lag_8",
                "visible_mean_change_1",
                "visible_mean_change_3",
                "visible_mean_rolling_4",
            ]
        )
    elif feature_mode != "base":
        raise ValueError(f"Unsupported causal feature mode: {feature_mode}")
    dynamic = np.column_stack(dynamic_parts)
    static = np.column_stack(
        [coordinates[:, 0], coordinates[:, 1], np.log1p(np.asarray(population, dtype=np.float64))]
    )
    phi = np.empty((num_times, num_locations, 1 + dynamic.shape[1] + static.shape[1]), dtype=np.float64)
    phi[..., 0] = 1.0
    phi[..., 1 : 1 + dynamic.shape[1]] = dynamic[:, None, :]
    phi[..., 1 + dynamic.shape[1] :] = static[None, :, :]

    # Scale using only calibration-time visible locations. The intercept remains one.
    reference = phi[:calibration_stop, train_indices, 1:].reshape(-1, phi.shape[-1] - 1)
    mean = reference.mean(axis=0)
    scale = np.maximum(reference.std(axis=0), 1e-12)
    phi[..., 1:] = (phi[..., 1:] - mean) / scale
    columns = ["intercept", *dynamic_columns, "latitude", "longitude", "log_population"]
    return phi, columns


def _ridge(phi: np.ndarray, y: np.ndarray, spatial_indices: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    design = phi[:, spatial_indices].reshape(-1, phi.shape[-1])
    target = y[:, spatial_indices].reshape(-1)
    precision = design.T @ design + ridge * np.eye(design.shape[1])
    return np.linalg.solve(precision, design.T @ target)


def _inducing_coordinates(
    coordinates: np.ndarray, train_indices: np.ndarray, sizes: tuple[int, ...]
) -> dict[int, np.ndarray]:
    candidates = coordinates[train_indices]
    outputs = {}
    for requested in sizes:
        size = min(int(requested), candidates.shape[0])
        if size == candidates.shape[0]:
            outputs[requested] = candidates.copy()
        else:
            outputs[requested] = KMeans(
                n_clusters=size, random_state=0, n_init=10
            ).fit(candidates).cluster_centers_
    return outputs


def _census_state_centroids(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if len(names) != 1:
            raise ValueError(f"Expected one Gazetteer text file, found {names}")
        with archive.open(names[0]) as handle:
            header = handle.readline().decode("utf-8")
            separator = "|" if "|" in header else "\t"
            handle.seek(0)
            frame = pd.read_csv(handle, sep=separator, dtype={"GEOID": str})
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame.rename(columns={"GEOID": "location"})


def load_covid_panel(raw_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    target = pd.read_csv(raw_root / "covid_hospital_admissions.csv", dtype={"location": str})
    locations = pd.read_csv(raw_root / "locations.csv", dtype={"location": str})
    centroids = _census_state_centroids(raw_root / "2025_Gaz_state_national.zip")
    target = target[target["location"] != "US"].copy()
    target["target_end_date"] = pd.to_datetime(target["target_end_date"])
    pivot = target.pivot(index="target_end_date", columns="location", values="value").sort_index()
    common_locations = sorted(
        set(pivot.columns) & set(locations["location"]) & set(centroids["location"])
    )
    pivot = pivot[common_locations].dropna(axis=0, how="any")
    location_rows = locations.set_index("location").loc[common_locations]
    centroid_rows = centroids.set_index("location").loc[common_locations]
    population = location_rows["population"].to_numpy(dtype=np.float64)
    rates = pivot.to_numpy(dtype=np.float64) / population[None, :] * 100000.0
    y = np.log1p(rates)
    coordinates = centroid_rows[["INTPTLAT", "INTPTLONG"]].to_numpy(dtype=np.float64)
    dates = pivot.index.to_numpy(dtype="datetime64[D]")
    metadata = {
        "target": "log1p weekly confirmed hospital admissions per 100,000 population",
        "location_codes": common_locations,
        "location_names": location_rows["location_name"].astype(str).tolist(),
        "raw_dates": [date.date().isoformat() for date in pivot.index],
        "data_vintage": "current retrospective CDC Forecast Hub snapshot",
    }
    return y, coordinates, population, dates, metadata


def load_dengue_panel(raw_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    rows = [json.loads(line) for line in (raw_root / "infodengue.jsonl").read_text(encoding="utf-8").splitlines() if line]
    frame = pd.DataFrame(rows)
    frame["data_iniSE"] = pd.to_datetime(frame["data_iniSE"])
    frame["municipio_geocodigo"] = frame["municipio_geocodigo"].astype(str)
    coordinates_frame = pd.read_csv(
        raw_root / "municipality_coordinates.csv", dtype={"codigo_ibge": str}
    ).set_index("codigo_ibge")
    pivot = frame.pivot(index="data_iniSE", columns="municipio_geocodigo", values="casos").sort_index()
    coverage = pivot.notna().mean(axis=0)
    common_locations = sorted(
        set(coverage[coverage == 1.0].index) & set(coordinates_frame.index)
    )
    if len(common_locations) < 100:
        raise ValueError(
            f"Only {len(common_locations)} municipalities have complete records and coordinates"
        )
    pivot = pivot[common_locations]
    latest = (
        frame.sort_values("data_iniSE")
        .drop_duplicates("municipio_geocodigo", keep="last")
        .set_index("municipio_geocodigo")
        .loc[common_locations]
    )
    population = pd.to_numeric(latest["pop"], errors="raise").to_numpy(dtype=np.float64)
    rates = pivot.to_numpy(dtype=np.float64) / population[None, :] * 100000.0
    y = np.log1p(rates)
    coordinate_rows = coordinates_frame.loc[common_locations]
    coordinates = coordinate_rows[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    dates = pivot.index.to_numpy(dtype="datetime64[D]")
    metadata = {
        "target": "log1p weekly notified dengue cases per 100,000 population",
        "location_codes": common_locations,
        "location_names": latest["municipio_nome"].astype(str).tolist(),
        "raw_dates": [date.date().isoformat() for date in pivot.index],
        "data_vintage": "fixed retrospective Infodengue snapshot; notified cases can be revised",
    }
    return y, coordinates, population, dates, metadata


def build_protocol(
    *,
    dataset: str,
    raw_root: Path,
    output: Path,
    seed: int,
    calibration_steps: int,
    block_size: int,
    inducing_sizes: tuple[int, ...],
    feature_mode: str = "base",
) -> dict:
    if dataset == "covid":
        y_raw, coordinates_raw, population, dates, source_metadata = load_covid_panel(raw_root)
        period = 52.1775
    elif dataset == "dengue":
        y_raw, coordinates_raw, population, dates, source_metadata = load_dengue_panel(raw_root)
        period = 52.1775
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if calibration_steps < 8 or calibration_steps >= y_raw.shape[0]:
        raise ValueError("Calibration steps must leave a non-empty stream and contain at least 8 observations")
    train, fit, validation, test = spatial_split(y_raw.shape[1], seed)
    target_reference = y_raw[:calibration_steps, train]
    target_mean = float(target_reference.mean())
    target_scale = float(max(target_reference.std(), 1e-12))
    y = (y_raw - target_mean) / target_scale
    coordinates = standardize_coordinates(coordinates_raw)
    phi, feature_columns = build_causal_features(
        y=y,
        coordinates=coordinates_raw,
        population=population,
        train_indices=train,
        calibration_stop=calibration_steps,
        period=period,
        feature_mode=feature_mode,
    )
    calibration_y = y[:calibration_steps]
    stream_y = y[calibration_steps:]
    calibration_phi = phi[:calibration_steps]
    stream_phi = phi[calibration_steps:]
    beta = _ridge(calibration_phi, calibration_y, train)
    batch_beta = _ridge(stream_phi, stream_y, train)
    calibration_mean = np.einsum("tsp,p->ts", calibration_phi, beta)
    stream_mean = np.einsum("tsp,p->ts", stream_phi, beta)
    batch_mean = np.einsum("tsp,p->ts", stream_phi, batch_beta)
    blocks = [
        (start, min(stream_y.shape[0], start + block_size))
        for start in range(0, stream_y.shape[0], block_size)
    ]
    inducing = _inducing_coordinates(coordinates, train, inducing_sizes)
    calibration_span = max(calibration_steps - 1, 1)
    payload = {
        "train_indices": train,
        "fit_indices": fit,
        "validation_indices": validation,
        "test_indices": test,
        "task1_ridge_beta": beta,
        "batch_ridge_beta": batch_beta,
        "calibration_y": calibration_y.astype(np.float32),
        "stream_y": stream_y.astype(np.float32),
        "calibration_phi": calibration_phi.astype(np.float32),
        "stream_phi": stream_phi.astype(np.float32),
        "task1_calibration_mean": calibration_mean.astype(np.float32),
        "task1_stream_mean": stream_mean.astype(np.float32),
        "batch_stream_mean": batch_mean.astype(np.float32),
        "coordinates": coordinates,
        "calibration_times": np.arange(calibration_steps, dtype=np.float64) / calibration_span,
        "stream_times": np.arange(stream_y.shape[0], dtype=np.float64) / calibration_span,
        "block_start": np.asarray([start for start, _ in blocks], dtype=int),
        "block_stop": np.asarray([stop for _, stop in blocks], dtype=int),
    }
    for size, values in inducing.items():
        payload[f"inducing_coords_ms{size}"] = values
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    metadata = {
        "schema_version": 1,
        "dataset": dataset,
        "root": str(raw_root.resolve()),
        "npz": str(output.resolve()),
        "num_calibration_times": calibration_steps,
        "num_stream_times": int(stream_y.shape[0]),
        "num_locations": int(y.shape[1]),
        "num_train_locations": int(train.size),
        "num_fit_locations": int(fit.size),
        "num_validation_locations": int(validation.size),
        "num_test_locations": int(test.size),
        "split_seed": seed,
        "feature_source": "protocol_npz",
        "xlag": {
            "mode": "causal epidemiology covariates",
            "feature_mode": feature_mode,
            "length": 8 if feature_mode == "lag_dynamics" else 4,
            "features": len(feature_columns),
            "columns": feature_columns,
            "test_target_lags_used": False,
            "visible_aggregate_lags_use_train_locations_only": True,
        },
        "target_standardization": {
            "mean": target_mean,
            "scale": target_scale,
            "fit_scope": "calibration times at visible train locations only",
        },
        "num_blocks": len(blocks),
        "block_lengths": [stop - start for start, stop in blocks],
        "inducing_sizes_requested": list(inducing_sizes),
        **source_metadata,
    }
    metadata["npz_sha256"] = sha256_file(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["covid", "dengue"], required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--calibration-steps",
        type=int,
        default=52,
        help="Use at least one full annual cycle for weekly seasonal features.",
    )
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--inducing-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--feature-mode", choices=["base", "lag_dynamics"], default="base")
    args = parser.parse_args()
    metadata = build_protocol(
        dataset=args.dataset,
        raw_root=args.raw_root,
        output=args.output,
        seed=args.seed,
        calibration_steps=args.calibration_steps,
        block_size=args.block_size,
        inducing_sizes=tuple(args.inducing_sizes),
        feature_mode=args.feature_mode,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
