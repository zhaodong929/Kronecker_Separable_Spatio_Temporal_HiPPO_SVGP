#!/usr/bin/env python3
"""Summarize the locked PEMS-BAY seed-0 development and final evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "results/traffic/seed0_improvement_v2"
OLD = ROOT / "results/traffic/formal_deterministic_v1/pems_bay/main/nowcast/kronhippo_stgp/seed0"
NEW = DEV / "final_strict_online_seed0"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_record(label: str, path: Path) -> dict[str, object]:
    result = load_json(path / "result.json")
    return {"configuration": label, **result["result"]["final"]}


def smoothness(path: Path) -> dict[str, float]:
    archive = np.load(path / "predictions.npz")
    y = archive["y"]
    mean = archive["mean"]
    return {
        "truth_standard_deviation": float(np.std(y)),
        "prediction_standard_deviation": float(np.std(mean)),
        "truth_delta_standard_deviation": float(np.std(np.diff(y, axis=0))),
        "prediction_delta_standard_deviation": float(np.std(np.diff(mean, axis=0))),
        "truth_mean_absolute_delta": float(np.mean(np.abs(np.diff(y, axis=0)))),
        "prediction_mean_absolute_delta": float(np.mean(np.abs(np.diff(mean, axis=0)))),
    }


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 400,
        }
    )


def plot_validation(rows: list[dict[str, str]], output: Path) -> None:
    style()
    blue, orange, green = "#0072B2", "#E69F00", "#009E73"
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.45), constrained_layout=True)

    stage_a = [row for row in rows if row["stage"] == "A_stride_ell_t"]
    for stride, marker in ((1, "o"), (3, "s")):
        subset = sorted((row for row in stage_a if int(row["stride"]) == stride), key=lambda row: float(row["ell_t"]))
        axes[0].plot([float(row["ell_t"]) for row in subset], [float(row["validation_nlpd"]) for row in subset], marker=marker, lw=1.4, ms=4, label=f"stride {stride}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\ell_t$ (hours)")
    axes[0].set_ylabel("Task-1 validation NLPD")
    axes[0].set_title("(a) Temporal scale")
    axes[0].legend(frameon=False)

    specifications = [
        ("B_ms", "ms", r"$M_s$", "(b) Spatial capacity"),
        ("C_mt", "mt", r"$M_t$", "(c) Temporal capacity"),
        ("D_rff", "rff", "RFF", "(d) Spectral features"),
    ]
    for axis, (stage_name, key, xlabel, title) in zip(axes[1:4], specifications):
        subset = sorted((row for row in rows if row["stage"] == stage_name), key=lambda row: int(row[key]))
        axis.plot([int(row[key]) for row in subset], [float(row["validation_nlpd"]) for row in subset], color=blue, marker="o", lw=1.5, ms=4)
        axis.set_xlabel(xlabel)
        axis.set_title(title)

    subset = [row for row in rows if row["stage"] == "E_spatial_kernel"]
    order = ["geo_matern32", "spectral_mixture", "road_graph"]
    labels = ["Geo\nMatern", "Spectral\nmixture", "Road\ngraph"]
    values = [float(next(row["validation_nlpd"] for row in subset if row["spatial_kernel"] == name)) for name in order]
    axes[4].bar(labels, values, color=[blue, orange, green], width=0.68)
    axes[4].set_title("(e) Spatial kernel")
    axes[4].set_ylim(min(values) - 0.025, max(values) + 0.018)
    for axis in axes:
        axis.grid(axis="y", color="#D0D0D0", lw=0.55, alpha=0.65)
    fig.savefig(output / "fig_task1_validation_sweeps.pdf", bbox_inches="tight")
    fig.savefig(output / "fig_task1_validation_sweeps.png", bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(output: Path) -> None:
    style()
    old_archive = np.load(OLD / "predictions.npz")
    new_archive = np.load(NEW / "predictions.npz")
    manifest = load_json(ROOT / "results/traffic/protocols/pems_bay/pems_bay_seed0_spatial_split.json")
    sensor_ids = manifest["split"]["heldout_sensor_ids"][:3]
    window = slice(0, 288)
    x = np.arange(288) / 12.0
    result = load_json(NEW / "result.json")
    scale = float(result["target_standardisation"]["scale"])
    offset = float(result["target_standardisation"]["mean"])
    truth = new_archive["y"][window] * scale + offset
    old_mean = old_archive["mean"][window] * scale + offset
    new_mean = new_archive["mean"][window] * scale + offset
    new_std = np.sqrt(new_archive["variance"][window]) * scale

    fig, axes = plt.subplots(3, 1, figsize=(7.1, 5.4), sharex=True, constrained_layout=True)
    for index, axis in enumerate(axes):
        axis.fill_between(x, new_mean[:, index] - 1.645 * new_std[:, index], new_mean[:, index] + 1.645 * new_std[:, index], color="#56B4E9", alpha=0.18, linewidth=0)
        axis.plot(x, truth[:, index], color="#202020", lw=1.05, label="Observed")
        axis.plot(x, old_mean[:, index], color="#999999", lw=1.0, label="Previous configuration")
        axis.plot(x, new_mean[:, index], color="#0072B2", lw=1.15, label="Locked Task-1 configuration")
        axis.set_ylabel("Speed (mph)")
        axis.set_title(f"Sensor {sensor_ids[index]}", loc="left")
        axis.grid(axis="y", color="#D0D0D0", lw=0.5, alpha=0.6)
    axes[-1].set_xlabel("Hours into strict-online stream")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#56B4E9", alpha=0.18, edgecolor="none"))
    labels.append("Locked 90% interval")
    axes[0].legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.38))
    fig.savefig(output / "fig_seed0_locked_trajectory_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(output / "fig_seed0_locked_trajectory_diagnostic.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not (NEW / "result.json").exists():
        raise FileNotFoundError("The locked full-stream result is not complete")
    output = DEV / "summary"
    output.mkdir(parents=True, exist_ok=True)
    with (DEV / "candidate_scores.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = [metric_record("Previous seed-0 configuration", OLD), metric_record("Locked Task-1 configuration", NEW)]
    metric_names = ["rmse", "crps", "gaussian_nlpd", "ece", "coverage90", "rmse_speed", "mean_predictive_std"]
    with (output / "seed0_metric_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["configuration", *metric_names])
        writer.writeheader()
        writer.writerows({key: record[key] for key in writer.fieldnames} for record in records)

    diagnostics = {"previous": smoothness(OLD), "locked": smoothness(NEW)}
    (output / "smoothness_diagnostic.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_validation(rows, output)
    plot_trajectories(output)

    old, new = records
    selection = load_json(DEV / "selection.json")["selected_candidate"]
    guard = load_json(NEW / "result.json")["result"]["guard"]
    old_smooth = diagnostics["previous"]
    new_smooth = diagnostics["locked"]
    delta_increase = new_smooth["prediction_delta_standard_deviation"] / old_smooth["prediction_delta_standard_deviation"] - 1.0
    old_delta_ratio = old_smooth["prediction_delta_standard_deviation"] / old_smooth["truth_delta_standard_deviation"]
    new_delta_ratio = new_smooth["prediction_delta_standard_deviation"] / new_smooth["truth_delta_standard_deviation"]
    report = f"""# PEMS-BAY KronHiPPO-STGP Seed-0 Improvement

## Protocol

- Task 1 remains the first 2,016 five-minute observations.
- Configuration selection uses only the 26 visible-validation sensors over all Task-1 times; the 65 formal held-out sensors and the strict-online stream are excluded from selection.
- Search is staged rather than a factorial grid: stride/temporal scale, spatial capacity, temporal capacity, RFF count, then spatial kernel.
- Mean Gaussian NLPD is the selection metric and RMSE is the tie-breaker.
- After selection, all hyperparameters, inducing locations, graph kernel and RFF realization are frozen. Only the bounded posterior state is updated online.

## Locked configuration

| Field | Value |
|---|---:|
| Calibration stride | {selection['stride']} |
| Temporal lengthscale | {selection['ell_t']} h |
| Spatial inducing count | {selection['ms']} |
| Temporal capacity | {selection['mt']} |
| RFF count | {selection['rff']} |
| Spatial kernel | {selection['spatial_kernel']} |
| Graph diffusion | {selection['learned_theta']['graph_diffusion']:.6f} |
| Task-1 validation NLPD | {selection['validation_nlpd']:.6f} |
| Task-1 validation RMSE | {selection['validation_rmse']:.6f} |

## Staged Task-1 findings

| Stage | Tested values | Selected value | Task-1 conclusion |
|---|---|---|---|
| Calibration stride and temporal scale | stride = 1, 3; ell_t = 0.083, 0.25, 0.5, 1, 2 h | stride 1; ell_t = 0.5 h | The two strides are numerically tied at the selected temporal scale; 2 h is too smooth for validation. |
| Spatial capacity | Ms = 32, 64, 128, 260 | Ms = 32 | Extra spatial inducing locations do not improve visible-validation NLPD. |
| Temporal capacity | Mt = 32, 64, 128 | Mt = 128 | The main capacity gain is temporal: NLPD falls from 1.445899 to 1.331652. |
| RFF count | 256, 512 | 512 | The gain is small but positive: NLPD falls from 1.331652 to 1.328097. |
| Spatial kernel | geographic Matern-3/2, spectral mixture, road graph | road graph | The road graph is best: NLPD 1.275867 versus 1.328097 for geographic Matern. |

## Full strict-online seed-0 evaluation

| Configuration | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 | RMSE (mph) |
|---|---:|---:|---:|---:|---:|---:|
| Previous | {old['rmse']:.6f} | {old['crps']:.6f} | {old['gaussian_nlpd']:.6f} | {old['ece']:.6f} | {old['coverage90']:.6f} | {old['rmse_speed']:.6f} |
| Locked Task-1 | {new['rmse']:.6f} | {new['crps']:.6f} | {new['gaussian_nlpd']:.6f} | {new['ece']:.6f} | {new['coverage90']:.6f} | {new['rmse_speed']:.6f} |

The strict-online result is an evaluation of the frozen Task-1 choice, not an input to model selection. Relative RMSE change is {(new['rmse'] / old['rmse'] - 1.0) * 100.0:.2f}% and relative CRPS change is {(new['crps'] / old['crps'] - 1.0) * 100.0:.2f}% (negative means improvement).

## Smoothness diagnostic

| Quantity | Previous | Locked Task-1 | Truth |
|---|---:|---:|---:|
| Overall standard deviation | {old_smooth['prediction_standard_deviation']:.6f} | {new_smooth['prediction_standard_deviation']:.6f} | {new_smooth['truth_standard_deviation']:.6f} |
| First-difference standard deviation | {old_smooth['prediction_delta_standard_deviation']:.6f} | {new_smooth['prediction_delta_standard_deviation']:.6f} | {new_smooth['truth_delta_standard_deviation']:.6f} |
| Mean absolute first difference | {old_smooth['prediction_mean_absolute_delta']:.6f} | {new_smooth['prediction_mean_absolute_delta']:.6f} | {new_smooth['truth_mean_absolute_delta']:.6f} |

The locked prediction is less over-smoothed: its first-difference standard deviation increases by {delta_increase * 100.0:.1f}%, from {old_delta_ratio:.3f} to {new_delta_ratio:.3f} of the observed first-difference scale. It remains substantially smoother than the raw traffic series, so this is a partial correction rather than a complete resolution.

## Causal and numerical audit

| Counter | Observed |
|---|---:|
| Current hidden reads before prediction | {guard['current_hidden_reads_before_prediction']} |
| Current hidden reveals | {guard['current_hidden_reveals']} |
| Current visible reads | {guard['current_visible_reads']} |
| Unique delayed hidden absorptions | {guard['unique_delayed_hidden_absorptions']} |

All prediction arrays are checked separately for finite values and positive variances in the accompanying audit command. This is a single development seed and must not replace a multi-seed formal result without rerunning the locked configuration on the remaining prespecified splits.

## Road graph provenance

The road-distance CSV is the PEMS-BAY sensor graph released with the official DCRNN repository. Its path and SHA-256 digest are recorded in `LOCK.json` and `data/traffic/raw/download_manifest.json`.
"""
    (output / "SEED0_IMPROVEMENT_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
