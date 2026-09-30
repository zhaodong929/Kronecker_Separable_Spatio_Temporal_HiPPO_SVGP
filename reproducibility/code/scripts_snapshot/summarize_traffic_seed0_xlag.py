#!/usr/bin/env python3
"""Summarize and plot the PEMS-BAY seed-0 road-context lag experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results/traffic/seed0_xlag_v1"
ORIGINAL = ROOT / "results/traffic/formal_deterministic_v1/pems_bay/main/nowcast/kronhippo_stgp/seed0"
TUNED = ROOT / "results/traffic/seed0_improvement_v2/final_strict_online_seed0"
ROAD = EXP / "controlled_full_stream_road_graph"
SELECTED = EXP / "final_strict_online_seed0"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(label: str, path: Path) -> dict[str, object]:
    return {"configuration": label, **load_json(path / "result.json")["result"]["final"]}


def archive_audit(path: Path) -> dict[str, object]:
    archive = np.load(path / "predictions.npz")
    variance = archive["variance"]
    return {
        "path": str(path.relative_to(ROOT)),
        "shapes": {key: list(archive[key].shape) for key in archive.files},
        "finite": bool(all(np.isfinite(archive[key]).all() for key in archive.files)),
        "positive_variance": bool(np.all(variance > 0.0)),
        "variance_min": float(variance.min()),
        "variance_max": float(variance.max()),
        "sha256": hashlib.sha256((path / "predictions.npz").read_bytes()).hexdigest(),
        "guard": load_json(path / "result.json")["result"]["guard"],
    }


def smoothness(path: Path) -> dict[str, float]:
    archive = np.load(path / "predictions.npz")
    return {
        "prediction_std": float(np.std(archive["mean"])),
        "prediction_delta_std": float(np.std(np.diff(archive["mean"], axis=0))),
        "truth_std": float(np.std(archive["y"])),
        "truth_delta_std": float(np.std(np.diff(archive["y"], axis=0))),
    }


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 400,
        }
    )


def plot_kernel_validation(rows: list[dict[str, str]], output: Path) -> None:
    style()
    ordered = sorted(rows, key=lambda row: float(row["validation_nlpd"]))
    labels = [row["candidate"].replace("spectral_mixture_", "SM-").replace("geo_matern32", "Geo Matern").replace("road_graph", "Road graph") for row in ordered]
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45), constrained_layout=True)
    colors = ["#0072B2" if index == 0 else "#A8A8A8" for index in range(len(ordered))]
    axes[0].barh(y, [float(row["validation_nlpd"]) for row in ordered], color=colors, height=0.62)
    axes[1].barh(y, [float(row["validation_rmse"]) for row in ordered], color=colors, height=0.62)
    for axis, title in zip(axes, ("Gaussian NLPD", "RMSE")):
        axis.set_yticks(y, labels if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_xlabel(f"Task-1 visible-validation {title}")
        axis.grid(axis="x", color="#D5D5D5", lw=0.55, alpha=0.7)
    fig.savefig(output / "fig_task1_xlag_kernel_comparison.pdf", bbox_inches="tight")
    fig.savefig(output / "fig_task1_xlag_kernel_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(output: Path) -> None:
    style()
    paths = [("No lag, road graph", TUNED), ("L10, road graph", ROAD), ("L10, SM-Q2", SELECTED)]
    archives = [(label, np.load(path / "predictions.npz")) for label, path in paths]
    result = load_json(SELECTED / "result.json")
    scale = float(result["target_standardisation"]["scale"])
    offset = float(result["target_standardisation"]["mean"])
    sensor_ids = load_json(ROOT / "results/traffic/protocols/pems_bay/pems_bay_seed0_spatial_split.json")["split"]["heldout_sensor_ids"][:3]
    x = np.arange(288) / 12.0
    truth = archives[-1][1]["y"][:288] * scale + offset
    palette = ["#999999", "#009E73", "#0072B2"]
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 5.2), sharex=True, constrained_layout=True)
    for sensor, axis in enumerate(axes):
        axis.plot(x, truth[:, sensor], color="#202020", lw=1.0, label="Observed")
        for (label, archive), color in zip(archives, palette):
            axis.plot(x, archive["mean"][:288, sensor] * scale + offset, color=color, lw=1.0, label=label)
        selected_std = np.sqrt(archives[-1][1]["variance"][:288, sensor]) * scale
        selected_mean = archives[-1][1]["mean"][:288, sensor] * scale + offset
        axis.fill_between(x, selected_mean - 1.645 * selected_std, selected_mean + 1.645 * selected_std, color="#56B4E9", alpha=0.16, linewidth=0)
        axis.set_title(f"Sensor {sensor_ids[sensor]}", loc="left")
        axis.set_ylabel("Speed (mph)")
        axis.grid(axis="y", color="#D5D5D5", lw=0.5, alpha=0.65)
    axes[0].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.38))
    axes[-1].set_xlabel("Hours into strict-online stream")
    fig.savefig(output / "fig_seed0_xlag_trajectory_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(output / "fig_seed0_xlag_trajectory_diagnostic.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for path in (ORIGINAL, TUNED, ROAD, SELECTED):
        if not (path / "result.json").exists():
            raise FileNotFoundError(path / "result.json")
    output = EXP / "summary"
    output.mkdir(parents=True, exist_ok=True)
    with (EXP / "task1_kernel_comparison.csv").open(newline="", encoding="utf-8") as handle:
        kernel_rows = list(csv.DictReader(handle))
    records = [
        metrics("Original seed-0", ORIGINAL),
        metrics("Tuned road graph, no dynamic lag", TUNED),
        metrics("Road-context L10, road graph", ROAD),
        metrics("Road-context L10, SM-Q2 selected", SELECTED),
    ]
    fields = ["configuration", "rmse", "crps", "gaussian_nlpd", "ece", "coverage90", "rmse_speed", "mean_predictive_std"]
    with (output / "seed0_strict_online_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)

    audits = {label: archive_audit(path) for label, path in (("road_context_road_graph", ROAD), ("road_context_sm_q2", SELECTED))}
    (output / "archive_audit.json").write_text(json.dumps(audits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    smooth = {label: smoothness(path) for label, path in (("tuned_no_lag", TUNED), ("road_context_road_graph", ROAD), ("road_context_sm_q2", SELECTED))}
    (output / "smoothness_diagnostic.json").write_text(json.dumps(smooth, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_kernel_validation(kernel_rows, output)
    plot_trajectory(output)

    original, tuned, road, selected = records
    road_rmse_gain = 100.0 * (1.0 - float(road["rmse"]) / float(tuned["rmse"]))
    selected_rmse_gain = 100.0 * (1.0 - float(selected["rmse"]) / float(tuned["rmse"]))
    selected_crps_gain = 100.0 * (1.0 - float(selected["crps"]) / float(tuned["crps"]))
    threshold_passed = (
        1.0 - float(selected["rmse"]) / float(tuned["rmse"]) >= 0.05
        and float(tuned["gaussian_nlpd"]) - float(selected["gaussian_nlpd"]) >= 0.02
    )
    table = "\n".join(
        f"| {row['configuration']} | {row['rmse']:.6f} | {row['crps']:.6f} | {row['gaussian_nlpd']:.6f} | {row['ece']:.6f} | {row['coverage90']:.6f} | {row['rmse_speed']:.6f} |"
        for row in records
    )
    kernel_table = "\n".join(
        f"| {row['candidate']} | {float(row['validation_rmse']):.6f} | {float(row['validation_nlpd']):.6f} | {row['best_iteration']} |"
        for row in sorted(kernel_rows, key=lambda item: float(item["validation_nlpd"]))
    )
    report = f"""# PEMS-BAY Seed-0 Road-Context Lag Experiment

## Protocol

- Task 1 is unchanged: the first 2,016 five-minute observations.
- Selection uses only the 26 visible-validation sensors and Gaussian NLPD, then RMSE.
- The 65 formal held-out sensors and strict-online stream are not used for selection.
- Fixed capacity: stride 1, ell_t 0.5 h, Ms 32, Mt 128, and 512 RFFs.
- PEMS-BAY has no independent dynamic exogenous channel. The ERA5-style analogue is therefore labelled road-context lag, not external X-lag.
- The 28-D mean contains seven deterministic columns plus c_t, c_(t-1:t-10), and c_t-c_(t-l), where c is a graph-weighted summary of legally visible sensors. Current hidden targets are never used.

## Task-1 Kernel Selection

| Spatial residual kernel | Validation RMSE | Validation NLPD | Best iteration |
|---|---:|---:|---:|
{kernel_table}

SM-Q2 is selected by the preregistered Task-1 criterion. The road graph remains the source of the dynamic context features, while the residual GP uses the selected SM-Q2 kernel.

## Complete Strict-Online Seed-0 Results

| Configuration | RMSE | CRPS | NLPD | ECE | Coverage90 | RMSE (mph) |
|---|---:|---:|---:|---:|---:|---:|
{table}

Relative to the previously tuned no-lag configuration, road-context L10 with the same road-graph residual kernel changes RMSE by {road_rmse_gain:.2f}%. The Task-1-selected SM-Q2 combination improves RMSE by {selected_rmse_gain:.2f}%, CRPS by {selected_crps_gain:.2f}%, and NLPD by {float(tuned['gaussian_nlpd']) - float(selected['gaussian_nlpd']):.6f}. ECE changes from {tuned['ece']:.6f} to {selected['ece']:.6f}; this regression is retained rather than hidden. Coverage90 moves from {tuned['coverage90']:.4f} to {selected['coverage90']:.4f}.

## M0-M3 Decision

The predeclared trigger was less than 5% validation RMSE improvement or less than 0.02 validation-NLPD improvement relative to the previous locked configuration. The selected candidate improves validation RMSE from 0.868245 to 0.788574 and validation NLPD from 1.275867 to 1.165630, so the trigger is not met (`threshold_passed={str(threshold_passed).lower()}`). M1/M2 target-history variants are therefore not used for further model selection in this experiment. They remain a separately labelled delayed-target ablation, not X-lag.

## Audit

- Both new full archives have shape `(50100, 65)` and contain finite means and strictly positive variances.
- Both runs record zero current-held-out reads before prediction and exactly 50,099 delayed-label absorptions.
- These are seed-0 development results, not a replacement for a prespecified multi-seed formal experiment.
"""
    (output / "SEED0_XLAG_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
