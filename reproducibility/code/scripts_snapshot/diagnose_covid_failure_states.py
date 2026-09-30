#!/usr/bin/env python3
"""Diagnose per-state errors for a COVID Route B held-out-state run."""

from __future__ import annotations

import argparse
import json
import zipfile
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scripts.run_epidemiology_pilot import predictive_metrics
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_epidemiology_pilot import predictive_metrics


BOUNDARY_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2024/shp/"
    "cb_2024_us_state_500k.zip"
)


def _read_gazetteer(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".txt")]
        if len(members) != 1:
            raise ValueError(f"Expected one Gazetteer text file in {path}, found {members}")
        return pd.read_csv(BytesIO(archive.read(members[0])), sep="|", dtype={"GEOID": str})


def _read_adjacency(path: Path, codes: list[str]) -> tuple[dict[str, set[str]], str]:
    """Derive polygon adjacency from a Census cartographic-boundary shapefile."""
    try:
        import shapefile
        from shapely.geometry import shape as shapely_shape
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Boundary diagnostics require pyshp and shapely") from exc

    with zipfile.ZipFile(path) as archive:
        files = {Path(name).suffix.lower(): archive.read(name) for name in archive.namelist()}
    reader = shapefile.Reader(
        shp=BytesIO(files[".shp"]),
        shx=BytesIO(files[".shx"]),
        dbf=BytesIO(files[".dbf"]),
    )
    fields = [field[0] for field in reader.fields[1:]]
    geometries: dict[str, object] = {}
    for record, shp in zip(reader.records(), reader.shapes()):
        values = dict(zip(fields, record))
        code = str(values.get("STATEFP", values.get("GEOID", ""))).zfill(2)
        if code in codes:
            geometries[code] = shapely_shape(shp.__geo_interface__)

    missing = sorted(set(codes) - set(geometries))
    if missing:
        raise ValueError(f"Boundary source lacks protocol locations: {missing}")

    neighbors = {code: set() for code in codes}
    for left, left_code in enumerate(codes):
        left_geometry = geometries[left_code]
        for right_code in codes[left + 1 :]:
            right_geometry = geometries[right_code]
            left_min_x, left_min_y, left_max_x, left_max_y = left_geometry.bounds
            right_min_x, right_min_y, right_max_x, right_max_y = right_geometry.bounds
            if (
                left_max_x < right_min_x
                or right_max_x < left_min_x
                or left_max_y < right_min_y
                or right_max_y < left_min_y
            ):
                continue
            # A point contact is not a shared state border. Length is measured
            # in the source CRS (degrees), so this is only a topology test.
            shared = left_geometry.boundary.intersection(right_geometry.boundary)
            if float(shared.length) > 1e-6:
                neighbors[left_code].add(right_code)
                neighbors[right_code].add(left_code)
    return neighbors, BOUNDARY_URL


def _nearest_centroid_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2.0) ** 2
    distance = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    np.fill_diagonal(distance, np.inf)
    return np.min(distance, axis=1)


def _lag_correlation(series: np.ndarray) -> float:
    if series.size < 3 or np.std(series[:-1]) < 1e-12 or np.std(series[1:]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(series[1:], series[:-1])[0, 1])


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if valid.sum() < 3:
        return float("nan")
    x = left[valid].rank(method="average").to_numpy(dtype=float)
    y = right[valid].rank(method="average").to_numpy(dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--routeb-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--locations-csv", type=Path, required=True)
    parser.add_argument("--gazetteer-zip", type=Path, required=True)
    parser.add_argument("--state-boundaries", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    codes = [str(code).zfill(2) for code in metadata["location_codes"]]
    names = list(metadata["location_names"])
    if len(codes) != len(names):
        raise ValueError("Protocol location codes and names have different lengths")

    with np.load(args.protocol_npz) as arrays:
        calibration_y = np.asarray(arrays["calibration_y"], dtype=np.float64)
        stream_y = np.asarray(arrays["stream_y"], dtype=np.float64)
        coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
        test_indices = np.asarray(arrays["test_indices"], dtype=int)
    with np.load(args.routeb_predictions) as predictions:
        y_true = np.asarray(predictions["y_true"], dtype=np.float64)
        pred_mean = np.asarray(predictions["pred_mean"], dtype=np.float64)
        pred_var = np.asarray(predictions["pred_var"], dtype=np.float64)
        saved_test_indices = np.asarray(predictions["test_indices"], dtype=int)
    expected = stream_y[:, test_indices]
    if not np.array_equal(saved_test_indices, test_indices):
        raise ValueError("Prediction test_indices do not match the protocol")
    if not np.allclose(y_true, expected, rtol=0.0, atol=1e-12):
        raise ValueError("Prediction y_true does not match protocol held-out labels")
    if pred_mean.shape != y_true.shape or pred_var.shape != y_true.shape:
        raise ValueError("Prediction mean/variance shape mismatch")
    if not np.isfinite(pred_mean).all() or not np.isfinite(pred_var).all() or (pred_var <= 0).any():
        raise ValueError("Prediction mean/variance contains non-finite values or non-positive variance")

    location_frame = pd.read_csv(args.locations_csv, dtype={"location": str})
    location_frame["location"] = location_frame["location"].str.zfill(2)
    location_frame = location_frame.set_index("location")
    gazetteer = _read_gazetteer(args.gazetteer_zip).set_index("GEOID")
    missing = [code for code in codes if code not in location_frame.index or code not in gazetteer.index]
    if missing:
        raise ValueError(f"Missing population or centroid metadata for {missing}")

    lat = gazetteer.loc[codes, "INTPTLAT"].to_numpy(dtype=float)
    lon = gazetteer.loc[codes, "INTPTLONG"].to_numpy(dtype=float)
    populations = location_frame.loc[codes, "population"].to_numpy(dtype=float)
    nearest_km = _nearest_centroid_km(lat, lon)
    neighbors, boundary_source = _read_adjacency(args.state_boundaries, codes)
    neighbor_count = {code: len(neighbors[code]) for code in codes}

    full_y = np.concatenate([calibration_y, stream_y], axis=0)
    rows: list[dict[str, object]] = []
    test_set = set(test_indices.tolist())
    total_sse = float(np.sum((y_true - pred_mean) ** 2))
    for column, location_index in enumerate(test_indices):
        truth = y_true[:, column]
        mean = pred_mean[:, column]
        variance = pred_var[:, column]
        metrics = predictive_metrics(truth, mean, variance)
        squared_error = float(np.sum((truth - mean) ** 2))
        calibration = calibration_y[:, location_index]
        full_series = full_y[:, location_index]
        rows.append(
            {
                "rank_by_rmse": 0,
                "location_index": int(location_index),
                "location_code": codes[location_index],
                "state": names[location_index],
                **metrics,
                "squared_error": squared_error,
                "sse_share": squared_error / total_sse if total_sse else float("nan"),
                "population": populations[location_index],
                "task1_std": float(np.std(calibration)),
                "task1_range": float(np.max(calibration) - np.min(calibration)),
                "full_series_std": float(np.std(full_series)),
                "lag1_correlation": _lag_correlation(full_series),
                "nearest_centroid_km": nearest_km[location_index],
                "neighbor_count": neighbor_count[codes[location_index]],
            }
        )

    rows.sort(key=lambda row: float(row["rmse"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_by_rmse"] = rank

    characteristics: list[dict[str, object]] = []
    heldout_rmse = {int(row["location_index"]): float(row["rmse"]) for row in rows}
    for index, (code, state) in enumerate(zip(codes, names)):
        characteristics.append(
            {
                "location_index": index,
                "location_code": code,
                "state": state,
                "heldout": index in test_set,
                "population": populations[index],
                "latitude": lat[index],
                "longitude": lon[index],
                "nearest_centroid_km": nearest_km[index],
                "neighbor_count": neighbor_count[code],
                "task1_std": float(np.std(calibration_y[:, index])),
                "task1_range": float(np.max(calibration_y[:, index]) - np.min(calibration_y[:, index])),
                "full_series_std": float(np.std(full_y[:, index])),
                "lag1_correlation": _lag_correlation(full_y[:, index]),
                "heldout_rmse": heldout_rmse.get(index, float("nan")),
            }
        )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "heldout_per_state.csv", index=False)
    pd.DataFrame(characteristics).to_csv(output / "all_states_characteristics.csv", index=False)
    pd.DataFrame(
        [{"location_code": code, "state": names[index], "neighbor_count": neighbor_count[code]} for index, code in enumerate(codes)]
    ).to_csv(output / "state_adjacency.csv", index=False)
    pd.DataFrame(
        [
            {"left_code": left, "right_code": right}
            for left in codes
            for right in sorted(neighbors[left])
            if left < right
        ]
    ).to_csv(output / "state_adjacency_edges.csv", index=False)

    plot_rows = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(plot_rows["state"], plot_rows["rmse"], color="#b23a48")
    axis.axhline(float(np.sqrt(np.mean((y_true - pred_mean) ** 2))), color="#1f4e79", linestyle="--", label="overall RMSE")
    axis.set_ylabel("Stream RMSE on transformed target")
    axis.set_title("COVID Route B D: held-out-state RMSE")
    axis.tick_params(axis="x", rotation=55)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "heldout_rmse_by_state.png", dpi=180)
    plt.close(fig)

    overall_rmse = float(np.sqrt(np.mean((y_true - pred_mean) ** 2)))
    sse_sorted = plot_rows["squared_error"].to_numpy(dtype=float)
    sse_cumulative = np.cumsum(sse_sorted) / total_sse if total_sse else np.full(len(rows), np.nan)
    median_row = plot_rows["rmse"].median()
    high = plot_rows[plot_rows["rmse"] >= median_row]
    low = plot_rows[plot_rows["rmse"] < median_row]
    factor_columns = [
        "population", "task1_std", "task1_range", "full_series_std",
        "lag1_correlation", "nearest_centroid_km", "neighbor_count",
    ]
    correlations = {column: _rank_correlation(plot_rows["rmse"], plot_rows[column]) for column in factor_columns}

    report = [
        "# COVID Route B delayed-history failure-state diagnostic",
        "",
        "## Scope and interpretation",
        "",
        "This is a seed-0 diagnostic of the current best D run: delayed posterior observations plus the causal state-history mean. The COVID archive is a 91-week, 52-location feasibility pilot, not a final multi-seed benchmark.",
        "",
        f"The archive was checked against the protocol: {len(test_indices)} held-out trajectories and {stream_y.shape[0]} causal stream weeks match exactly. Metrics use the model's transformed target (`{metadata.get('target', 'protocol target')}`), not raw admission counts.",
        "",
        "## Overall result",
        "",
        f"Overall held-out RMSE: **{overall_rmse:.4f}**. Total squared error: **{total_sse:.4f}**.",
        f"The worst state contributes {sse_cumulative[0] * 100:.1f}% of total squared error; the worst three contribute {sse_cumulative[min(2, len(rows) - 1)] * 100:.1f}%; the worst five contribute {sse_cumulative[min(4, len(rows) - 1)] * 100:.1f}%.",
        "",
        "A high concentration would mean that a few states dominate the aggregate score. It would point toward state-specific heterogeneity rather than a uniform failure of the whole predictor.",
        "",
        "## Held-out states",
        "",
        "| Rank | State | RMSE | NLL | Coverage90 | Mean std | SSE share | Population | Task-1 std | Full std | Lag-1 corr. | Neighbors | Nearest centroid (km) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['rank_by_rmse']} | {row['state']} | {row['rmse']:.4f} | {row['nll']:.4f} | {row['coverage90']:.4f} | {row['mean_predictive_std']:.4f} | {row['sse_share'] * 100:.1f}% | {row['population']:,.0f} | {row['task1_std']:.4f} | {row['full_series_std']:.4f} | {row['lag1_correlation']:.3f} | {row['neighbor_count']} | {row['nearest_centroid_km']:.0f} |"
        )
    report.extend(
        [
            "",
            "## What the small diagnostic can and cannot show",
            "",
            "The following are Spearman correlations across only the 10 held-out states. They are descriptive associations, not causal tests; one unusual state can move a correlation substantially.",
            "",
            "| Factor | Spearman correlation with per-state RMSE |",
            "|---|---:|",
        ]
    )
    for column, correlation in correlations.items():
        report.append(f"| {column} | {correlation:.3f} |" if np.isfinite(correlation) else f"| {column} | unavailable |")
    report.extend(
        [
            "",
            f"Among the high-error half (RMSE >= median {median_row:.4f}), median population was {high['population'].median():,.0f}, median Task-1 standard deviation was {high['task1_std'].median():.4f}, median full-series standard deviation was {high['full_series_std'].median():.4f}, median lag-1 correlation was {high['lag1_correlation'].median():.3f}, and median neighbor count was {high['neighbor_count'].median():.1f}.",
            f"The corresponding low-error half had median population {low['population'].median():,.0f}, Task-1 standard deviation {low['task1_std'].median():.4f}, full-series standard deviation {low['full_series_std'].median():.4f}, lag-1 correlation {low['lag1_correlation'].median():.3f}, and neighbor count {low['neighbor_count'].median():.1f}.",
            "",
            "Population is included as a scale/context variable, not as a feature newly given to the predictor. Geographic isolation is measured as the nearest distance between Census internal-point centroids. Neighbor counts are derived from Census 2024 cartographic-boundary polygon contacts; point-only contacts are excluded.",
            f"Boundary source: `{boundary_source}`. Gazetteer source: `{args.gazetteer_zip}`.",
            "",
            "## Decision",
            "",
            "Do not change the model based on this table alone. If the SSE share is concentrated in a few states, the next targeted experiment should model state heterogeneity or graph connectivity and report per-state results. If errors are broadly similar across states, a global model/protocol change is more appropriate. The present table is diagnostic evidence only and does not establish either explanation causally.",
        ]
    )
    (output / "failure_state_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output.resolve()), "overall_rmse": overall_rmse, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
