#!/usr/bin/env python3
"""Render strict-online spatial nowcasting examples for cumulative HiPPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_covid_dataset_overview import (
    INSET_EXTENTS,
    ensure_census_boundaries,
    load_state_polygons,
    sha256,
)


DEFAULT_OUTPUT = Path(
    "/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/"
    "kronecker+s2vgp/ICLR Formal experiment/covid_delayed_history_baseline_paper_materials/"
    "long_stream_2020_2024_mandatory/online_nowcasting_examples"
)
FIXED_EXAMPLES = (
    ("Omicron peak", "2022-01-15", "fixed epidemic-regime peak"),
    ("Late winter wave", "2024-01-06", "fixed late-stream epidemic-regime peak"),
)
INSET_SPECS = (
    ("02", "AK", (0.01, 0.01, 0.27, 0.27)),
    ("15", "HI", (0.28, 0.01, 0.17, 0.17)),
    ("72", "PR", (0.80, 0.01, 0.18, 0.17)),
)
CONTIGUOUS_EXTENT = (-125.2, -65.2, 24.0, 50.7)
OBSERVED_COLOR = "#E7E7E7"


def draw_region(
    axis: plt.Axes,
    polygons: dict[str, list[np.ndarray]],
    values: dict[str, float],
    codes: tuple[str, ...],
    *,
    extent: tuple[float, float, float, float],
    cmap: plt.Colormap,
    norm: Normalize,
    observed_codes: set[str],
) -> None:
    for code in codes:
        for points in polygons[code]:
            observed = code in observed_codes
            axis.add_patch(
                Polygon(
                    points,
                    closed=True,
                    facecolor=OBSERVED_COLOR if observed else cmap(norm(values[code])),
                    edgecolor="#F9F9F7" if observed else "#303030",
                    linewidth=0.28 if observed else 0.52,
                )
            )
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("equal")
    axis.axis("off")


def draw_map(
    axis: plt.Axes,
    polygons: dict[str, list[np.ndarray]],
    values: dict[str, float],
    codes: np.ndarray,
    *,
    cmap: plt.Colormap,
    norm: Normalize,
    observed_codes: set[str],
) -> None:
    contiguous = tuple(code for code in codes if code not in INSET_EXTENTS)
    draw_region(
        axis,
        polygons,
        values,
        contiguous,
        extent=CONTIGUOUS_EXTENT,
        cmap=cmap,
        norm=norm,
        observed_codes=observed_codes,
    )
    for code, label, bounds in INSET_SPECS:
        inset = axis.inset_axes(bounds)
        draw_region(
            inset,
            polygons,
            values,
            (code,),
            extent=INSET_EXTENTS[code],
            cmap=cmap,
            norm=norm,
            observed_codes=observed_codes,
        )
        inset.text(0.03, 0.03, label, transform=inset.transAxes, fontsize=5.4, fontweight="bold", color="#242424")


def read_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        required = ("y_true", "pred_mean", "pred_var", "test_indices", "mt", "ms", "variance_mode")
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"Prediction archive lacks {missing}: {path}")
        output = {key: np.asarray(archive[key]) for key in required}
    for key in ("y_true", "pred_mean", "pred_var"):
        output[key] = np.asarray(output[key], dtype=np.float64)
    output["test_indices"] = np.asarray(output["test_indices"], dtype=np.int64)
    if output["y_true"].shape != output["pred_mean"].shape or output["y_true"].shape != output["pred_var"].shape:
        raise ValueError("Prediction target, mean, and variance shapes differ")
    if not np.isfinite(output["y_true"]).all() or not np.isfinite(output["pred_mean"]).all():
        raise ValueError("Prediction archive has non-finite targets or means")
    if not np.isfinite(output["pred_var"]).all() or np.any(output["pred_var"] <= 0.0):
        raise ValueError("Prediction archive has non-positive or non-finite variance")
    if output["test_indices"].size != 10 or np.unique(output["test_indices"]).size != 10:
        raise ValueError("Formal COVID map must score exactly ten unique hidden jurisdictions")
    if int(output["mt"].item()) != 32 or int(output["ms"].item()) != 32:
        raise ValueError("Map is restricted to the formal Mt=32, Ms=32 cumulative HiPPO configuration")
    if str(output["variance_mode"].item()) != "full_joint_conditional":
        raise ValueError("Map requires full-joint-conditional predictive variance")
    return output


def map_metrics(target: np.ndarray, prediction: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    error = target - prediction
    standard_deviation = np.sqrt(variance)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "coverage90": float(np.mean(np.abs(error) <= 1.6448536269514722 * standard_deviation)),
    }


def selected_examples(
    stream_dates: np.ndarray,
) -> list[dict[str, object]]:
    """Select pre-specified epidemiological regimes, never by prediction error."""

    selected: list[dict[str, object]] = []
    for label, date, selection_rule in FIXED_EXAMPLES:
        match = np.flatnonzero(stream_dates == date)
        if match.size != 1:
            raise ValueError(f"Strict-online stream has no unique selected date {date}")
        position = int(match[0])
        selected.append({"label": label, "position": position, "selection_rule": selection_rule})
    return selected


def generate_figure(
    *,
    protocol_npz: Path,
    protocol_json: Path,
    prediction_npz: Path,
    boundary_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    metadata = json.loads(protocol_json.read_text(encoding="utf-8"))
    codes = np.asarray(metadata["location_codes"], dtype=str)
    names = np.asarray(metadata["location_names"], dtype=str)
    with np.load(protocol_npz) as protocol:
        calibration_y = np.asarray(protocol["calibration_y"], dtype=np.float64)
        stream_y = np.asarray(protocol["stream_y"], dtype=np.float64)
        stream_dates = np.asarray(protocol["stream_week_dates"], dtype=str)
    prediction = read_predictions(prediction_npz)
    if codes.size != 52 or calibration_y.shape != (52, 52) or stream_y.shape != (143, 52):
        raise ValueError("Expected the audited 52-location, 52+143-week long COVID protocol")
    expected_truth = stream_y[:, prediction["test_indices"]]
    if not np.allclose(prediction["y_true"], expected_truth, rtol=0.0, atol=1e-12):
        raise ValueError("Prediction archive targets do not match the protocol's hidden-state stream labels")
    standardization = metadata["target_standardization"]
    target_mean = float(standardization["mean"])
    target_scale = float(standardization["scale"])
    restore = lambda values: values * target_scale + target_mean
    all_target = restore(np.vstack([calibration_y, stream_y]))
    stream_target = restore(stream_y)
    predicted_target = restore(prediction["pred_mean"])
    prediction_variance = prediction["pred_var"] * target_scale**2
    if np.any(all_target < 0.0):
        raise ValueError("Restored log1p targets must be non-negative")
    examples = selected_examples(stream_dates)

    shapefile_path, boundary_metadata = ensure_census_boundaries(boundary_root)
    polygons = load_state_polygons(shapefile_path, set(codes.tolist()))
    color_upper = float(np.quantile(all_target, 0.99))
    target_norm = Normalize(vmin=0.0, vmax=color_upper, clip=True)
    target_cmap = plt.get_cmap("viridis")
    hidden_codes = set(codes[prediction["test_indices"]].tolist())
    visible_codes = set(codes.tolist()) - hidden_codes

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.4,
            "axes.titleweight": "bold",
            "figure.dpi": 180,
            "savefig.dpi": 320,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.1, 7.15), constrained_layout=False)
    figure.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.105, hspace=0.18, wspace=0.04)
    for axis, title in zip(
        axes[0],
        ("Held-out ground truth", "Cumulative HiPPO prediction"),
    ):
        axis.set_title(title, fontsize=9.1, pad=13)
    selected_rows = []
    for row, example in enumerate(examples):
        position = int(example["position"])
        truth_values = {code: float(stream_target[position, index]) for index, code in enumerate(codes)}
        prediction_values = truth_values.copy()
        for column, location_index in enumerate(prediction["test_indices"]):
            prediction_values[codes[location_index]] = float(predicted_target[position, column])
        draw_map(axes[row, 0], polygons, truth_values, codes, cmap=target_cmap, norm=target_norm, observed_codes=visible_codes)
        draw_map(axes[row, 1], polygons, prediction_values, codes, cmap=target_cmap, norm=target_norm, observed_codes=visible_codes)
        date = stream_dates[position]
        horizon = position + 1
        scores = map_metrics(
            stream_target[position, prediction["test_indices"]],
            predicted_target[position],
            prediction_variance[position],
        )
        selected_rows.append(
            {
                "label": str(example["label"]),
                "selection_rule": str(example["selection_rule"]),
                "date": str(date),
                "online_week": int(horizon),
                "hidden_states": [str(name) for name in names[prediction["test_indices"]]],
                **scores,
            }
        )
        axes[row, 0].set_title(
            f"Held-out ground truth\n{example['label']}: {date} (online week {horizon})",
            fontsize=8.9,
            pad=2.5,
        )
        axes[row, 1].set_title(
            f"Cumulative HiPPO prediction\n{example['label']}: {date} (online week {horizon})\nRMSE = {scores['rmse']:.2f}",
            fontsize=8.9,
            pad=2.5,
        )

    figure.text(0.06, 0.925, "Strict-online spatial nowcasting with cumulative HiPPO", fontsize=11.1, fontweight="bold", ha="left")
    figure.text(0.06, 0.900, "Formal split seed 5. Every panel colors the same 10 held-out jurisdictions; the 42 jurisdictions observed at week t are muted. Dates were selected as fixed epidemic-regime peaks, not by prediction error.", fontsize=6.95, ha="left", color="#303030")
    figure.text(0.06, 0.064, "Muted grey: observed at week t, not a prediction target", fontsize=6.65, color="#404040")
    target_colorbar_axis = figure.add_axes((0.53, 0.052, 0.27, 0.014))
    target_colorbar = figure.colorbar(ScalarMappable(norm=target_norm, cmap=target_cmap), cax=target_colorbar_axis, orientation="horizontal")
    target_colorbar.set_label("Target: log(1 + weekly admissions per 100k)", fontsize=6.45, labelpad=2)
    target_colorbar.ax.tick_params(labelsize=5.8, length=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fig_online_nowcasting_maps_cumulative_hippo"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(figure)
    summary = "\n".join(
        [
            "# Cumulative HiPPO Strict-Online Map Examples",
            "",
            "Each panel colors only the ten spatially held-out jurisdictions from formal split seed 5. The other 42 jurisdictions were observed at that week and are deliberately muted rather than displayed as model predictions.",
            "",
            *[
                f"- {row['label']}, {row['date']} (online week {row['online_week']}; {row['selection_rule']}): hidden-state RMSE {row['rmse']:.3f}; Coverage90 {row['coverage90']:.2f}."
                for row in selected_rows
            ],
            "",
        ]
    )
    (output_dir / "online_nowcasting_maps_summary.md").write_text(summary, encoding="utf-8")
    audit = {
        "status": "complete",
        "figure": "Strict-online spatial nowcasting: held-out truth vs. cumulative HiPPO prediction",
        "protocol_npz": str(protocol_npz.resolve()),
        "protocol_npz_sha256": sha256(protocol_npz),
        "protocol_json": str(protocol_json.resolve()),
        "prediction_npz": str(prediction_npz.resolve()),
        "prediction_npz_sha256": sha256(prediction_npz),
        "source_csv": metadata["source_csv"],
        "source_csv_sha256": metadata["source_sha256"],
        "target": metadata["target"],
        "seed": 5,
        "method": "Route B cumulative HiPPO",
        "mt": 32,
        "ms": 32,
        "variance_mode": "full_joint_conditional",
        "delayed_observation_protocol": "previous hidden labels -> current visible update -> current hidden prediction",
        "hidden_location_codes": sorted(hidden_codes),
        "hidden_location_names": [str(name) for name in names[prediction["test_indices"]]],
        "selected_examples": selected_rows,
        "target_color_scale": {"vmin": 0.0, "vmax": color_upper, "rule": "99th percentile of all 52 x 195 observed transformed targets"},
        "census_boundary_snapshot": boundary_metadata,
        "panel_rule": "Only hidden-state targets or predictive means are colored; all 42 current-week visible jurisdictions are muted in every panel.",
        "outputs": [str(stem.with_suffix(".pdf")), str(stem.with_suffix(".png"))],
    }
    stem.with_suffix(".config.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed5/protocol.npz"))
    parser.add_argument("--protocol-json", type=Path, default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed5/protocol.json"))
    parser.add_argument("--prediction-npz", type=Path, default=Path("results/diagnostics/covid_long_stream_2020_2024_mandatory/seed5/routeb_cumulative/online/predictions.npz"))
    parser.add_argument("--boundary-root", type=Path, default=Path("data/epidemiology/raw/covid_long_2020_2024/census_boundaries"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = generate_figure(
        protocol_npz=(ROOT / args.protocol_npz).resolve(),
        protocol_json=(ROOT / args.protocol_json).resolve(),
        prediction_npz=(ROOT / args.prediction_npz).resolve(),
        boundary_root=(ROOT / args.boundary_root).resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({"status": "complete", "outputs": audit["outputs"], "selected_examples": audit["selected_examples"]}, indent=2))


if __name__ == "__main__":
    main()
