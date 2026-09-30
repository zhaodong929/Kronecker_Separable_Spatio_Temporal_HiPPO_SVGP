#!/usr/bin/env python3
"""Create the paper-ready CDC COVID long-stream dataset overview figure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
import numpy as np
import requests
import shapefile


ROOT = Path(__file__).resolve().parents[1]
CENSUS_BOUNDARY_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"
DEFAULT_OUTPUT = Path(
    "/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/"
    "kronecker+s2vgp/ICLR Formal experiment/covid_delayed_history_baseline_paper_materials/"
    "long_stream_2020_2024_mandatory/dataset_overview"
)
MAP_WINDOWS = (
    ("Winter 2020-21", "2020-11-01", "2021-03-31"),
    ("Omicron", "2021-12-01", "2022-02-28"),
    ("Winter 2023-24", "2023-11-01", "2024-02-29"),
)
INSET_EXTENTS = {
    "02": (-172.0, -130.0, 51.0, 72.5),
    "15": (-161.0, -154.0, 18.5, 22.7),
    "72": (-68.6, -65.0, 17.5, 18.7),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_census_boundaries(boundary_root: Path) -> tuple[Path, dict[str, object]]:
    """Download and unpack a stable Census state-boundary snapshot once."""

    zip_path = boundary_root / "cb_2023_us_state_20m.zip"
    extract_root = boundary_root / "cb_2023_us_state_20m"
    shp_files = sorted(extract_root.glob("*.shp"))
    metadata: dict[str, object]
    if not zip_path.is_file():
        boundary_root.mkdir(parents=True, exist_ok=True)
        response = requests.get(CENSUS_BOUNDARY_URL, timeout=90)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
        metadata = {
            "url": CENSUS_BOUNDARY_URL,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "http_status": response.status_code,
            "content_length_header": response.headers.get("content-length"),
            "etag": response.headers.get("etag"),
            "last_modified": response.headers.get("last-modified"),
        }
    else:
        metadata = {"url": CENSUS_BOUNDARY_URL, "retrieved_at_utc": "preexisting snapshot"}
    if not shp_files:
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_root)
        shp_files = sorted(extract_root.glob("*.shp"))
    if len(shp_files) != 1:
        raise RuntimeError(f"Expected one Census state shapefile under {extract_root}, found {shp_files}")
    metadata["zip_path"] = str(zip_path.resolve())
    metadata["zip_sha256"] = sha256(zip_path)
    metadata["shapefile_path"] = str(shp_files[0].resolve())
    return shp_files[0], metadata


def load_state_polygons(shapefile_path: Path, codes: set[str]) -> dict[str, list[np.ndarray]]:
    reader = shapefile.Reader(str(shapefile_path))
    fields = [field[0] for field in reader.fields[1:]]
    state_index = fields.index("STATEFP")
    polygons: dict[str, list[np.ndarray]] = {}
    for record, shape in zip(reader.records(), reader.shapes()):
        code = str(record[state_index]).zfill(2)
        if code not in codes:
            continue
        points = np.asarray(shape.points, dtype=np.float64)
        starts = list(shape.parts) + [points.shape[0]]
        polygons[code] = [points[start:stop] for start, stop in zip(starts[:-1], starts[1:])]
    missing = sorted(codes - set(polygons))
    if missing:
        raise ValueError(f"Census state boundary snapshot is missing protocol FIPS codes: {missing}")
    return polygons


def trajectory_layout(values: np.ndarray, names: np.ndarray) -> tuple[np.ndarray, list[int], str]:
    """Order states by trajectory and return separators for six coarse groups."""

    normalized = (values - values.mean(axis=0, keepdims=True)) / np.maximum(values.std(axis=0, keepdims=True), 1e-10)
    try:
        from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
        from scipy.spatial.distance import pdist

        linkage_matrix = linkage(pdist(normalized.T, metric="correlation"), method="average")
        order = leaves_list(linkage_matrix)
        clusters = fcluster(linkage_matrix, t=6, criterion="maxclust")[order]
        separators = (np.flatnonzero(clusters[1:] != clusters[:-1]) + 1).tolist()
        return order, separators, "average-linkage clustering into six trajectory groups"
    except Exception:
        return np.argsort(names, kind="stable"), [], "alphabetic fallback because SciPy clustering is unavailable"


def choose_reference_weeks(dates: np.ndarray, values: np.ndarray) -> list[dict[str, object]]:
    national_mean = values.mean(axis=1)
    selected: list[dict[str, object]] = []
    for label, start, stop in MAP_WINDOWS:
        mask = (dates >= np.datetime64(start)) & (dates <= np.datetime64(stop))
        if not np.any(mask):
            raise ValueError(f"No weekly endpoint is available for reference window {label}")
        candidates = np.flatnonzero(mask)
        index = int(candidates[np.argmax(national_mean[candidates])])
        selected.append({"label": label, "index": index, "date": str(dates[index]), "mean_log_rate": float(national_mean[index])})
    return selected


def draw_region(
    axis: plt.Axes,
    polygons: dict[str, list[np.ndarray]],
    values: dict[str, float],
    codes: tuple[str, ...],
    *,
    extent: tuple[float, float, float, float],
    cmap: plt.Colormap,
    norm: Normalize,
    linewidth: float,
) -> None:
    for code in codes:
        for points in polygons[code]:
            axis.add_patch(
                Polygon(
                    points,
                    closed=True,
                    facecolor=cmap(norm(values[code])),
                    edgecolor="#F9F9F7",
                    linewidth=linewidth,
                )
            )
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("equal")
    axis.axis("off")


def draw_state_map(
    axis: plt.Axes,
    polygons: dict[str, list[np.ndarray]],
    values: dict[str, float],
    codes: np.ndarray,
    *,
    cmap: plt.Colormap,
    norm: Normalize,
    title: str,
) -> None:
    contiguous = tuple(code for code in codes if code not in INSET_EXTENTS)
    draw_region(
        axis,
        polygons,
        values,
        contiguous,
        extent=(-125.2, -65.2, 24.0, 50.7),
        cmap=cmap,
        norm=norm,
        linewidth=0.30,
    )
    axis.set_title(title, fontsize=9, fontweight="bold", pad=2)
    inset_specs = (("02", "AK", (0.01, 0.01, 0.27, 0.27)), ("15", "HI", (0.28, 0.01, 0.17, 0.17)), ("72", "PR", (0.80, 0.01, 0.18, 0.17)))
    for code, label, bounds in inset_specs:
        inset = axis.inset_axes(bounds)
        draw_region(inset, polygons, values, (code,), extent=INSET_EXTENTS[code], cmap=cmap, norm=norm, linewidth=0.25)
        inset.text(0.03, 0.03, label, transform=inset.transAxes, fontsize=5.5, fontweight="bold", color="#242424")


def generate_figure(
    *,
    protocol_npz: Path,
    protocol_json: Path,
    boundary_root: Path,
    output: Path,
) -> dict[str, object]:
    metadata = json.loads(protocol_json.read_text(encoding="utf-8"))
    with np.load(protocol_npz) as data:
        standardized_target = np.vstack(
            [
                np.asarray(data["calibration_y"], dtype=np.float64),
                np.asarray(data["stream_y"], dtype=np.float64),
            ]
        )
    dates = np.asarray(metadata["raw_dates"], dtype="datetime64[D]")
    codes = np.asarray(metadata["location_codes"], dtype=str)
    names = np.asarray(metadata["location_names"], dtype=str)
    if standardized_target.shape != (dates.size, codes.size):
        raise ValueError(
            f"Protocol target shape {standardized_target.shape} does not match {dates.size} dates and {codes.size} locations"
        )
    standardization = metadata["target_standardization"]
    target = (
        standardized_target * float(standardization["scale"])
        + float(standardization["mean"])
    )
    if not np.isfinite(target).all() or np.any(target < 0.0):
        raise FloatingPointError("Dataset overview target must be finite and non-negative")

    shapefile_path, boundary_metadata = ensure_census_boundaries(boundary_root)
    polygons = load_state_polygons(shapefile_path, set(codes.tolist()))
    order, separators, ordering_method = trajectory_layout(target, names)
    selected = choose_reference_weeks(dates, target)
    color_upper = float(np.quantile(target, 0.99))
    norm = Normalize(vmin=0.0, vmax=color_upper, clip=True)
    cmap = plt.get_cmap("viridis")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titleweight": "bold",
            "figure.dpi": 180,
            "savefig.dpi": 320,
        }
    )
    figure = plt.figure(figsize=(10.9, 6.3), constrained_layout=False)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.57, 0.92), left=0.085, right=0.94, top=0.85, bottom=0.15, hspace=0.25, wspace=0.06)
    heatmap_axis = figure.add_subplot(grid[0, :])
    image = heatmap_axis.imshow(target[:, order].T, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    heatmap_axis.axvspan(-0.5, 51.5, color="#FFFFFF", alpha=0.08, linewidth=0)
    heatmap_axis.axvline(51.5, color="#202020", linewidth=0.85)
    heatmap_axis.annotate("", xy=(-0.5, 1.10), xytext=(51.5, 1.10), xycoords=("data", "axes fraction"), arrowprops={"arrowstyle": "<->", "color": "#3C3C3C", "lw": 0.65})
    heatmap_axis.annotate("", xy=(51.5, 1.10), xytext=(dates.size - 0.5, 1.10), xycoords=("data", "axes fraction"), arrowprops={"arrowstyle": "<->", "color": "#3C3C3C", "lw": 0.65})
    heatmap_axis.text(25.5, 1.125, "Task 1: 52-week initialization", transform=heatmap_axis.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.1, color="#303030")
    heatmap_axis.text(123.0, 1.125, "Strict online evaluation: 143 weekly updates", transform=heatmap_axis.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.1, color="#303030")
    for ordinal, entry in enumerate(selected, start=1):
        heatmap_axis.axvline(float(entry["index"]), color="#FFFFFF", linewidth=0.7, linestyle="--", alpha=0.9)
        heatmap_axis.text(float(entry["index"]), 1.025, f"({['i', 'ii', 'iii'][ordinal - 1]}) {entry['label']}", transform=heatmap_axis.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.7, color="#262626")
    for separator in separators:
        heatmap_axis.axhline(separator - 0.5, color="#FFFFFF", linewidth=0.65, alpha=0.80)
    year_starts = [index for index, date in enumerate(dates) if str(date)[5:] == "01-01"]
    ticks = [0, *year_starts, dates.size - 1]
    labels = [str(dates[index])[:4] for index in ticks]
    heatmap_axis.set_xticks(ticks, labels)
    heatmap_axis.set_xlim(-0.5, dates.size - 0.5)
    # Keep the overview readable at paper width: label only the two
    # jurisdictions shown in the trajectory panels below.
    display_names = {"Connecticut", "Nevada"}
    display_positions = [i for i, name in enumerate(names[order]) if name in display_names]
    display_labels = [names[order][i] for i in display_positions]
    heatmap_axis.set_yticks(display_positions, display_labels, fontsize=7.0)
    heatmap_axis.tick_params(axis="y", length=0, pad=1.2)
    heatmap_axis.tick_params(axis="x", length=3, pad=2)
    heatmap_axis.set_ylabel("Jurisdiction (selected labels)")
    figure.text(0.085, 0.955, "(a) State-level COVID-19 hospitalization dynamics across the 195-week study period", fontsize=10.2, fontweight="bold", ha="left")

    for column, entry in enumerate(selected):
        axis = figure.add_subplot(grid[1, column])
        week_values = {code: float(target[int(entry["index"]), position]) for position, code in enumerate(codes)}
        draw_state_map(
            axis,
            polygons,
            week_values,
            codes,
            cmap=cmap,
            norm=norm,
            title=f"({['i', 'ii', 'iii'][column]}) {entry['label']}\n{entry['date']}",
        )
        if column == 0:
            axis.text(-0.14, 1.04, "(b)", transform=axis.transAxes, fontsize=10, fontweight="bold", va="bottom")
    figure.text(0.51, 0.108, "Solid line: strict-online start. Selected peak weeks reveal changing geographic burden; Alaska (AK), Hawaii (HI), and Puerto Rico (PR) are shown as insets.", ha="center", fontsize=6.8)
    colorbar_axis = figure.add_axes((0.365, 0.043, 0.30, 0.014))
    colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Shared scale: log(1 + weekly admissions per 100k)", fontsize=6.9, labelpad=2)
    colorbar.ax.tick_params(labelsize=6.2, length=2)

    output.mkdir(parents=True, exist_ok=True)
    target_path = output / "fig_covid_dataset_overview"
    figure.savefig(target_path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(target_path.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(figure)
    audit = {
        "status": "complete",
        "figure": "COVID Dataset Overview / Fig. 1(a-b)",
        "protocol_npz": str(protocol_npz.resolve()),
        "protocol_npz_sha256": sha256(protocol_npz),
        "protocol_json": str(protocol_json.resolve()),
        "source_csv": metadata["source_csv"],
        "source_csv_sha256": metadata["source_sha256"],
        "target": metadata["target"],
        "weekly_endpoints": int(dates.size),
        "date_range": [str(dates[0]), str(dates[-1])],
        "locations": int(codes.size),
        "state_order": names[order].tolist(),
        "state_ordering": ordering_method,
        "trajectory_cluster_separators": separators,
        "selected_weeks": selected,
        "color_scale": {"vmin": 0.0, "vmax": color_upper, "vmax_rule": "99th percentile of all 52 x 195 target values; larger values are color-clipped"},
        "census_boundary_snapshot": boundary_metadata,
        "map_insets": {"02": "Alaska", "15": "Hawaii", "72": "Puerto Rico"},
        "outputs": [str(target_path.with_suffix(".pdf")), str(target_path.with_suffix(".png"))],
    }
    target_path.with_suffix(".config.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed5/protocol.npz"))
    parser.add_argument("--protocol-json", type=Path, default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed5/protocol.json"))
    parser.add_argument("--boundary-root", type=Path, default=Path("data/epidemiology/raw/covid_long_2020_2024/census_boundaries"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = generate_figure(
        protocol_npz=(ROOT / args.protocol_npz).resolve(),
        protocol_json=(ROOT / args.protocol_json).resolve(),
        boundary_root=(ROOT / args.boundary_root).resolve(),
        output=args.output_dir.resolve(),
    )
    print(json.dumps({"status": "complete", "output": audit["outputs"], "selected_weeks": audit["selected_weeks"]}, indent=2))


if __name__ == "__main__":
    main()
