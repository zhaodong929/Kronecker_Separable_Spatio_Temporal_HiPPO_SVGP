#!/usr/bin/env python3
"""Quantify the association between geography and COVID trajectory similarity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
EARTH_RADIUS_KM = 6371.0088
EXCLUDED_NONCONTIGUOUS_CODES = {"02", "15", "72"}  # Alaska, Hawaii, Puerto Rico
DEFAULT_OUTPUT = Path(
    "/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/"
    "kronecker+s2vgp/ICLR Formal experiment/covid_delayed_history_baseline_paper_materials/"
    "long_stream_2020_2024_mandatory/dataset_overview/geographic_proximity_diagnostic"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_centroids(path: Path, location_codes: np.ndarray) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if len(members) != 1:
            raise ValueError(f"Expected one Gazetteer text file in {path}, found {members}")
        with archive.open(members[0]) as handle:
            header = handle.readline().decode("utf-8")
            handle.seek(0)
            frame = pd.read_csv(handle, sep="|" if "|" in header else "\t", dtype={"GEOID": str})
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(columns={"GEOID": "location"}).set_index("location")
    missing = sorted(set(location_codes.tolist()) - set(frame.index))
    if missing:
        raise ValueError(f"Census Gazetteer is missing protocol locations: {missing}")
    return frame.loc[location_codes, ["INTPTLAT", "INTPTLONG"]].to_numpy(dtype=np.float64)


def standardise_coordinates(coordinates: np.ndarray) -> np.ndarray:
    return (coordinates - coordinates.mean(axis=0, keepdims=True)) / np.maximum(
        coordinates.std(axis=0, keepdims=True), 1e-12
    )


def haversine_distances_km(coordinates: np.ndarray) -> np.ndarray:
    latitude = np.deg2rad(coordinates[:, 0])
    longitude = np.deg2rad(coordinates[:, 1])
    delta_latitude = latitude[:, None] - latitude[None, :]
    delta_longitude = longitude[:, None] - longitude[None, :]
    half_chord_squared = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(latitude[:, None]) * np.cos(latitude[None, :]) * np.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(half_chord_squared, 0.0, 1.0)))


def pairwise_similarity(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("Trajectory matrix must have at least three time points")
    if not np.isfinite(values).all() or np.any(np.std(values, axis=0) < 1e-12):
        raise ValueError("Trajectory matrix must be finite with non-constant jurisdiction trajectories")
    return np.corrcoef(values.T)


def spearman(distance: np.ndarray, similarity: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(distance), rankdata(similarity))[0, 1])


def label_permutation_p_value(
    distance_matrix: np.ndarray,
    similarity_matrix: np.ndarray,
    upper_triangle: tuple[np.ndarray, np.ndarray],
    *,
    seed: int,
    permutations: int,
) -> tuple[float, np.ndarray]:
    """Use whole-jurisdiction relabeling, preserving all pair dependence."""

    rows, columns = upper_triangle
    distance = distance_matrix[rows, columns]
    observed = spearman(distance, similarity_matrix[rows, columns])
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        permutation = rng.permutation(similarity_matrix.shape[0])
        null[index] = spearman(distance, similarity_matrix[permutation[rows], permutation[columns]])
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (permutations + 1))
    return p_value, null


def association_summary(
    distance_matrix: np.ndarray,
    similarity_matrix: np.ndarray,
    indices: np.ndarray,
    *,
    seed: int,
    permutations: int,
) -> dict[str, float | int]:
    sub_distance = distance_matrix[np.ix_(indices, indices)]
    sub_similarity = similarity_matrix[np.ix_(indices, indices)]
    upper_triangle = np.triu_indices(indices.size, k=1)
    distance = sub_distance[upper_triangle]
    similarity = sub_similarity[upper_triangle]
    p_value, null = label_permutation_p_value(
        sub_distance,
        sub_similarity,
        upper_triangle,
        seed=seed,
        permutations=permutations,
    )
    return {
        "jurisdictions": int(indices.size),
        "pairs": int(distance.size),
        "spearman_rho": spearman(distance, similarity),
        "label_permutation_two_sided_p": p_value,
        "permutations": int(permutations),
        "null_rho_2_5_percent": float(np.quantile(null, 0.025)),
        "null_rho_97_5_percent": float(np.quantile(null, 0.975)),
    }


def equal_count_bins(distance: np.ndarray, similarity: np.ndarray, count: int) -> pd.DataFrame:
    order = np.argsort(distance, kind="stable")
    parts = np.array_split(order, count)
    return pd.DataFrame(
        [
            {
                "bin": number,
                "pairs": int(part.size),
                "distance_km_min": float(distance[part].min()),
                "distance_km_max": float(distance[part].max()),
                "distance_km_median": float(np.median(distance[part])),
                "similarity_median": float(np.median(similarity[part])),
            }
            for number, part in enumerate(parts, start=1)
        ]
    )


def write_summary(path: Path, full: dict[str, float | int], task1: dict[str, float | int], contiguous: dict[str, float | int]) -> str:
    effect = abs(float(full["spearman_rho"]))
    if effect < 0.20:
        conclusion = (
            "Geographic proximity is only weakly associated with similarity in state-level hospitalization "
            "trajectories. Jurisdiction pairs at comparable geographic distances span a broad range of temporal "
            "correlations, so geographic structure alone does not capture the heterogeneous COVID dynamics."
        )
    elif effect < 0.40:
        conclusion = (
            "Geographic proximity is moderately associated with similarity in state-level hospitalization trajectories, "
            "but the broad within-distance spread shows that geographic structure alone is insufficient to capture the "
            "heterogeneous COVID dynamics."
        )
    else:
        conclusion = (
            "Geographic proximity is strongly associated with similarity in state-level hospitalization trajectories, "
            "although residual variation at comparable distances means that geography is not a complete explanation."
        )
    task1_difference = abs(float(full["spearman_rho"]) - float(task1["spearman_rho"]))
    temporal_statement = (
        "The full-period and Task-1 associations are similar."
        if task1_difference < 0.05
        else "The full-period and Task-1 associations differ, indicating that the distance-similarity relationship changes across the stream."
    )
    contiguous_difference = abs(float(full["spearman_rho"]) - float(contiguous["spearman_rho"]))
    sensitivity_statement = (
        "Excluding Alaska, Hawaii, and Puerto Rico does not materially change the descriptive association."
        if contiguous_difference < 0.05
        else "Excluding Alaska, Hawaii, and Puerto Rico materially changes the descriptive association, so long-distance jurisdictions influence the aggregate pattern."
    )
    text = "\n".join(
        [
            "# Geographic Distance vs. COVID Trajectory Similarity",
            "",
            conclusion,
            "",
            f"Across all 1,326 jurisdiction pairs, Spearman rho was {float(full['spearman_rho']):.3f} over the complete 195-week period and {float(task1['spearman_rho']):.3f} in the 52-week Task-1 period.",
            f"The contiguous-US sensitivity analysis (excluding Alaska, Hawaii, and Puerto Rico; {int(contiguous['pairs'])} pairs) gave rho = {float(contiguous['spearman_rho']):.3f}.",
            temporal_statement,
            sensitivity_statement,
            "",
            "The reported permutation p-values use whole-jurisdiction label permutations, not an IID regression over the 1,326 overlapping pairs. They are included as a dependence-aware diagnostic; the effect sizes are the primary evidence.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return text


def generate_figure(
    *,
    protocol_npz: Path,
    protocol_json: Path,
    census_zip: Path,
    output_dir: Path,
    bins: int,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    metadata = json.loads(protocol_json.read_text(encoding="utf-8"))
    codes = np.asarray(metadata["location_codes"], dtype=str)
    names = np.asarray(metadata["location_names"], dtype=str)
    with np.load(protocol_npz) as arrays:
        target_standardized = np.vstack(
            [np.asarray(arrays["calibration_y"], dtype=np.float64), np.asarray(arrays["stream_y"], dtype=np.float64)]
        )
        routeb_coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
    dates = np.asarray(metadata["raw_dates"], dtype="datetime64[D]")
    if target_standardized.shape != (dates.size, codes.size) or codes.size != 52:
        raise ValueError("Diagnostic must use exactly the audited 52-location, 195-week COVID protocol")
    if int(metadata["num_calibration_times"]) != 52 or dates.size != 195:
        raise ValueError("Diagnostic expects the audited 52-week Task-1 and 195-week long-stream protocol")
    target_spec = metadata["target_standardization"]
    target = target_standardized * float(target_spec["scale"]) + float(target_spec["mean"])
    raw_coordinates = load_centroids(census_zip, codes)
    if not np.allclose(routeb_coordinates, standardise_coordinates(raw_coordinates), rtol=0.0, atol=1e-12):
        raise ValueError("Census centroids do not reproduce the Route B geographic-kernel coordinates")

    distance_matrix = haversine_distances_km(raw_coordinates)
    full_similarity = pairwise_similarity(target)
    task1_similarity = pairwise_similarity(target[:52])
    all_indices = np.arange(codes.size)
    contiguous_indices = np.flatnonzero(~np.isin(codes, sorted(EXCLUDED_NONCONTIGUOUS_CODES)))
    full_stats = association_summary(distance_matrix, full_similarity, all_indices, seed=seed, permutations=permutations)
    task1_stats = association_summary(distance_matrix, task1_similarity, all_indices, seed=seed + 1, permutations=permutations)
    contiguous_stats = association_summary(distance_matrix, full_similarity, contiguous_indices, seed=seed + 2, permutations=permutations)

    upper_triangle = np.triu_indices(codes.size, k=1)
    distance = distance_matrix[upper_triangle]
    similarity = full_similarity[upper_triangle]
    pairs = pd.DataFrame(
        {
            "left_code": codes[upper_triangle[0]],
            "left_name": names[upper_triangle[0]],
            "right_code": codes[upper_triangle[1]],
            "right_name": names[upper_triangle[1]],
            "geographic_distance_km": distance,
            "trajectory_similarity_195_weeks": similarity,
            "trajectory_similarity_task1_52_weeks": task1_similarity[upper_triangle],
        }
    )
    bin_frame = equal_count_bins(distance, similarity, bins)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titleweight": "bold",
            "figure.dpi": 180,
            "savefig.dpi": 360,
        }
    )
    figure, axis = plt.subplots(figsize=(5.25, 3.45))
    axis.scatter(distance, similarity, s=4.0, c="#9B9B9B", alpha=0.18, linewidths=0, rasterized=True)
    axis.plot(bin_frame["distance_km_median"], bin_frame["similarity_median"], color="#1E5A7A", linewidth=1.35, zorder=3)
    axis.scatter(bin_frame["distance_km_median"], bin_frame["similarity_median"], s=20, c="#1E5A7A", edgecolors="white", linewidths=0.45, zorder=4)
    annotation = (
        f"Spearman $\\rho$ (195 weeks) = {float(full_stats['spearman_rho']):.3f}\n"
        f"Spearman $\\rho$ (Task 1) = {float(task1_stats['spearman_rho']):.3f}"
    )
    axis.text(
        0.98,
        0.96,
        annotation,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.7,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#C8C8C8", "linewidth": 0.45, "alpha": 0.93},
    )
    axis.text(0.02, 0.04, f"Grey: {distance.size:,} pairs; blue: {bins} equal-count bin medians", transform=axis.transAxes, fontsize=6.6, color="#4A4A4A")
    axis.set_title("Geographic proximity only partially explains temporal similarity", fontsize=9.2, pad=7)
    axis.set_xlabel("Geographic distance (km)")
    axis.set_ylabel("COVID trajectory similarity\n(Pearson correlation of weekly log rates)")
    axis.set_ylim(-0.55, 1.03)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=3, color="#555555")
    figure.tight_layout(pad=0.65)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fig_geographic_proximity_temporal_similarity"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(figure)
    pairs.to_csv(output_dir / "geographic_similarity_pairs.csv", index=False)
    bin_frame.to_csv(output_dir / "geographic_similarity_binned_medians.csv", index=False)
    summary = write_summary(output_dir / "geographic_proximity_summary.md", full_stats, task1_stats, contiguous_stats)
    audit = {
        "status": "complete",
        "figure": "Geographic proximity only partially explains temporal similarity",
        "protocol_npz": str(protocol_npz.resolve()),
        "protocol_npz_sha256": sha256(protocol_npz),
        "protocol_json": str(protocol_json.resolve()),
        "source_csv": metadata["source_csv"],
        "source_csv_sha256": metadata["source_sha256"],
        "target": metadata["target"],
        "dates": [str(dates[0]), str(dates[-1])],
        "weekly_endpoints": int(dates.size),
        "task1_weeks": 52,
        "location_codes": codes.tolist(),
        "coordinate_source": {"census_gazetteer_zip": str(census_zip.resolve()), "sha256": sha256(census_zip)},
        "routeb_coordinate_convention": "Census Gazetteer INTPTLAT/INTPTLONG, column-standardised before the product Matérn-3/2 geographic kernel",
        "distance_convention": f"great-circle Haversine distance from the same source centroids, Earth radius {EARTH_RADIUS_KM} km",
        "trajectory_similarity": "Pearson correlation of complete transformed weekly trajectories z=log1p(admissions per 100k)",
        "full_195_week_association": full_stats,
        "task1_52_week_association": task1_stats,
        "contiguous_sensitivity_full_195_week_association": contiguous_stats,
        "excluded_contiguous_sensitivity_codes": sorted(EXCLUDED_NONCONTIGUOUS_CODES),
        "binning": {"method": "equal pair count after sorting by geographic distance", "bins": int(bins)},
        "permutation_design": "whole-jurisdiction trajectory label permutations; no IID pairwise regression inference",
        "summary_path": str((output_dir / "geographic_proximity_summary.md").resolve()),
        "outputs": [str(stem.with_suffix(".pdf")), str(stem.with_suffix(".png"))],
    }
    stem.with_suffix(".config.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return {"audit": audit, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed5/protocol.npz"))
    parser.add_argument("--protocol-json", type=Path, default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed5/protocol.json"))
    parser.add_argument("--census-zip", type=Path, default=Path("data/epidemiology/raw/covid/2025_Gaz_state_national.zip"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bins", type=int, default=9)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be at least two")
    if args.permutations < 100:
        raise ValueError("--permutations must be at least 100")
    result = generate_figure(
        protocol_npz=(ROOT / args.protocol_npz).resolve(),
        protocol_json=(ROOT / args.protocol_json).resolve(),
        census_zip=(ROOT / args.census_zip).resolve(),
        output_dir=args.output_dir.resolve(),
        bins=args.bins,
        permutations=args.permutations,
        seed=args.seed,
    )
    print(json.dumps({"status": "complete", "outputs": result["audit"]["outputs"], "full": result["audit"]["full_195_week_association"], "task1": result["audit"]["task1_52_week_association"]}, indent=2))


if __name__ == "__main__":
    main()
