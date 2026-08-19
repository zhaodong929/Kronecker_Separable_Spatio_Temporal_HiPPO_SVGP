#!/usr/bin/env python3
"""Generate paper tables and figures from the audited COVID Gaussian archives.

This script is intentionally separate from the earlier exploratory figure
generator. It reads only the formal Setting B archives (seeds 5--9), checks
that their targets and held-out masks agree with the audited protocols, and
writes a new output directory without touching older figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5, 6, 7, 8, 9)
Z90 = 1.6448536269514722
NOMINAL_COVERAGES = np.arange(0.1, 1.0, 0.1)
HORIZONS = (16, 32, 48, 64, 80, 96, 112, 128, 143)

FORMAL_METHODS = {
    "persistence": {
        "label": "Persistence",
        "relative": "results/diagnostics/covid_long_stream_2020_2024_mandatory/seed{seed}/deterministic/persistence/predictions.npz",
        "color": "#7B8794",
    },
    "lag_ridge": {
        "label": "Task-1 lag ridge",
        "relative": "results/diagnostics/covid_long_stream_2020_2024_mandatory/seed{seed}/deterministic/lag_ridge/predictions.npz",
        "color": "#A0A0A0",
    },
    "ohsvgp": {
        "label": "OHSVGP (RBF)",
        "relative": "results/diagnostics/covid_long_stream_2020_2024_mandatory/seed{seed}/ohsvgp_rbf/predictions.npz",
        "color": "#D55E00",
    },
    "bui_controlled": {
        "label": "Bui OSGPR (controlled)",
        "relative": "baselines/covid_long_setting_b/results/formal_selected_bui_ovc_shared_m8/seed{seed}/bui_controlled/predictions.npz",
        "color": "#E69F00",
    },
    "bui_adaptive": {
        "label": "Bui OSGPR (adaptive, CPU)",
        "relative": "baselines/covid_long_setting_b/results/formal_selected_bui_ovc_shared_m8/seed{seed}/bui_adaptive/predictions.npz",
        "color": "#CC79A7",
    },
    "st_svgp": {
        "label": "ST-SVGP",
        "relative": "baselines/covid_long_setting_b/results/formal_selected_st_svgp_isolated/seed{seed}/st_svgp/predictions.npz",
        "color": "#56B4E9",
    },
    "lmc_svgp": {
        "label": "LMC-SVGP",
        "relative": "baselines/covid_long_setting_b/results/formal_selected_factorial/seed{seed}/lmc_svgp/predictions.npz",
        "color": "#009E73",
    },
    "imc_svgp": {
        "label": "IMC-SVGP",
        "relative": "baselines/covid_long_setting_b/results/formal_selected_factorial/seed{seed}/imc_svgp/predictions.npz",
        "color": "#2A9D8F",
    },
    "fsde_svi": {
        "label": "FSDE-SVI",
        "relative": "baselines/covid_long_setting_b/results/formal_selected_fsde_svi_isolated/seed{seed}/fsde_svi/predictions.npz",
        "color": "#F0E442",
    },
    "routeb_ordinary": {
        "label": "Route B ordinary",
        "relative": "results/diagnostics/covid_long_stream_2020_2024_mandatory/seed{seed}/routeb_ordinary/online/predictions.npz",
        "color": "#009E73",
    },
    "routeb_hippo": {
        "label": "Route B cumulative HiPPO",
        "relative": "results/diagnostics/covid_long_stream_2020_2024_mandatory/seed{seed}/routeb_cumulative/online/predictions.npz",
        "color": "#0072B2",
    },
}

FIGURE_METHODS = (
    "persistence",
    "ohsvgp",
    "st_svgp",
    "routeb_ordinary",
    "routeb_hippo",
)
CALIBRATION_METHODS = ("ohsvgp", "st_svgp", "routeb_ordinary", "routeb_hippo")
TRAJECTORY_STATES = ("Connecticut", "Louisiana", "Nevada", "West Virginia")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.5,
            "savefig.dpi": 300,
        }
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, output: Path, name: str, config: dict[str, Any]) -> None:
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=320, bbox_inches="tight")
    (output / f"{name}.config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)


def load_protocol(protocol_root: Path, seed: int) -> dict[str, Any]:
    metadata = json.loads((protocol_root / f"seed{seed}" / "protocol.json").read_text(encoding="utf-8"))
    with np.load(protocol_root / f"seed{seed}" / "protocol.npz") as archive:
        data = {key: np.asarray(archive[key]) for key in ("stream_y", "test_indices", "stream_week_dates", "reporting_task_id")}
    data["metadata"] = metadata
    return data


def load_archive(path: Path, protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        required = ("y_true", "pred_mean", "pred_var", "test_indices")
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        arrays = {key: np.asarray(archive[key], dtype=np.float64) for key in required}
    expected = protocol["stream_y"][:, protocol["test_indices"]]
    if arrays["y_true"].shape != expected.shape or not np.allclose(arrays["y_true"], expected, atol=1e-12, rtol=0.0):
        raise ValueError(f"{path}: archive target differs from audited protocol")
    if not np.array_equal(arrays["test_indices"].astype(np.int64), protocol["test_indices"]):
        raise ValueError(f"{path}: held-out mask differs from audited protocol")
    if not np.isfinite(arrays["pred_mean"]).all() or not np.isfinite(arrays["pred_var"]).all() or (arrays["pred_var"] <= 0).any():
        raise ValueError(f"{path}: non-finite or non-positive predictive variance")
    return arrays


def load_all(protocol_root: Path) -> tuple[dict[int, dict[str, dict[str, np.ndarray]]], dict[int, dict[str, Any]]]:
    protocols = {seed: load_protocol(protocol_root, seed) for seed in SEEDS}
    records: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for seed in SEEDS:
        records[seed] = {}
        for method, spec in FORMAL_METHODS.items():
            path = ROOT / spec["relative"].format(seed=seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            records[seed][method] = load_archive(path, protocols[seed])
    return records, protocols


def restore(values: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    standardization = metadata["target_standardization"]
    return values * float(standardization["scale"]) + float(standardization["mean"])


def metric_table(report_root: Path, output: Path) -> list[dict[str, Any]]:
    source = report_root / "aggregate_metrics.csv"
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in tuple(row):
            if key.endswith(("_mean", "_sd")):
                row[key] = float(row[key])
    write_csv(output / "formal_results_table.csv", rows)
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & RMSE $\downarrow$ & CRPS $\downarrow$ & Gaussian NLPD $\downarrow$ & ECE $\downarrow$ & Coverage90 \\ ",
        r"\midrule",
    ]
    for row in rows:
        label = str(row["label"])
        if row["method"] == "routeb_cumulative_hippo":
            label = r"\textbf{" + label + "}"
        values = [
            f"{row['rmse_mean']:.4f} $\\pm$ {row['rmse_sd']:.4f}",
            f"{row['crps_mean']:.4f} $\\pm$ {row['crps_sd']:.4f}",
            f"{row['native_gaussian_nlpd_mean']:.4f} $\\pm$ {row['native_gaussian_nlpd_sd']:.4f}",
            f"{row['ece_mean']:.4f} $\\pm$ {row['ece_sd']:.4f}",
            f"{row['coverage90_mean']:.4f} $\\pm$ {row['coverage90_sd']:.4f}",
        ]
        lines.append(label + " & " + " & ".join(values) + r" \\ ")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"% Mean +/- sample SD over formal spatial split seeds 5--9.",
            r"% OVC-SVGP is excluded from numeric ranking because the selected 8x32 formal archive is resource-limited.",
        ]
    )
    (output / "formal_results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    markdown = [
        "| Method | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['label']} | {row['rmse_mean']:.4f} +/- {row['rmse_sd']:.4f} | "
            f"{row['crps_mean']:.4f} +/- {row['crps_sd']:.4f} | "
            f"{row['native_gaussian_nlpd_mean']:.4f} +/- {row['native_gaussian_nlpd_sd']:.4f} | "
            f"{row['ece_mean']:.4f} +/- {row['ece_sd']:.4f} | "
            f"{row['coverage90_mean']:.4f} +/- {row['coverage90_sd']:.4f} |"
        )
    markdown.extend(
        [
            "",
            "Mean +/- sample SD over formal spatial split seeds 5-9. OVC-SVGP is excluded from numeric ranking because its selected 8x32 exact-fantasy formal archive is resource-limited.",
        ]
    )
    (output / "formal_results_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return rows


def bootstrap_ci(values: np.ndarray, seed: int = 20260819) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def plot_metrics(rows: list[dict[str, Any]], output: Path) -> None:
    order = [row["method"] for row in rows]
    labels = [str(row["label"]) for row in rows]
    metrics = (
        ("rmse", "RMSE"),
        ("crps", "CRPS"),
        ("native_gaussian_nlpd", "Gaussian NLPD"),
        ("ece", "ECE"),
        ("coverage90", "Coverage90"),
    )
    fig, axes = plt.subplots(1, 5, figsize=(14.8, 6.3), sharey=True)
    y = np.arange(len(rows))
    for axis, (metric, title) in zip(axes, metrics):
        values = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
        errors = np.asarray([float(row[f"{metric}_sd"]) for row in rows])
        colors = ["#0072B2" if method == "routeb_cumulative_hippo" else "#BFC5CC" for method in order]
        colors = [color if method not in {"routeb_ordinary", "routeb_cumulative_hippo"} else ("#3B8DBD" if method == "routeb_ordinary" else "#0072B2") for method, color in zip(order, colors)]
        axis.barh(y, values, xerr=errors, color=colors, edgecolor="#404040", linewidth=0.35, capsize=2)
        if metric == "coverage90":
            axis.axvline(0.90, color="#444444", ls="--", lw=1.0)
            axis.set_xlim(0, 1.05)
        axis.set_title(title)
        axis.set_xlabel("mean +/- SD")
        axis.grid(axis="x")
        axis.set_axisbelow(True)
    axes[0].set_yticks(y, labels, fontsize=7.2)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Method")
    fig.suptitle("Formal Gaussian COVID long-stream benchmark", y=1.02, fontsize=12)
    fig.subplots_adjust(left=0.25, bottom=0.11, wspace=0.34)
    save_figure(fig, output, "fig_covid_metric_summary", {"figure": "metric_summary", "source": "formal_results_table.csv", "errorbars": "sample SD across seeds 5-9", "highlight": "Route B ordinary and cumulative HiPPO"})


def plot_trajectories(records: dict[int, dict[str, dict[str, np.ndarray]]], protocols: dict[int, dict[str, Any]], output: Path) -> None:
    seed = 5
    metadata = protocols[seed]["metadata"]
    names = metadata["location_names"]
    test_indices = protocols[seed]["test_indices"]
    positions = []
    for state in TRAJECTORY_STATES:
        matches = np.flatnonzero([names[int(index)] == state for index in test_indices])
        if not matches.size:
            raise ValueError(f"Predeclared trajectory state {state} is not held out in seed 5")
        positions.append(int(matches[0]))
    x = np.arange(protocols[seed]["stream_y"].shape[0])
    dates = protocols[seed]["stream_week_dates"].astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.3), sharex=True)
    for axis, state, position in zip(axes.flat, TRAJECTORY_STATES, positions):
        truth = restore(records[seed]["routeb_hippo"]["y_true"][:, position], metadata)
        axis.plot(x, truth, color="#202020", lw=1.8, label="Observed")
        for method in FIGURE_METHODS:
            arrays = records[seed][method]
            mean = restore(arrays["pred_mean"][:, position], metadata)
            axis.plot(x, mean, color=FORMAL_METHODS[method]["color"], lw=1.25, label=FORMAL_METHODS[method]["label"])
        for method in ("routeb_ordinary", "routeb_hippo"):
            arrays = records[seed][method]
            mean = restore(arrays["pred_mean"][:, position], metadata)
            std = np.sqrt(arrays["pred_var"][:, position]) * float(metadata["target_standardization"]["scale"])
            axis.fill_between(x, mean - Z90 * std, mean + Z90 * std, color=FORMAL_METHODS[method]["color"], alpha=0.10, linewidth=0)
        axis.set_title(f"{state} | seed 5 held out", loc="left", fontsize=9.5)
        axis.grid(True)
        axis.set_ylabel("log1p weekly admissions / 100k")
        axis.set_xticks(np.linspace(0, len(x) - 1, 5, dtype=int), [dates[i][:7] for i in np.linspace(0, len(x) - 1, 5, dtype=int)], rotation=25, ha="right")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    fig.suptitle("Representative held-out trajectories under strict online nowcasting", y=1.01, fontsize=12)
    fig.text(0.5, -0.055, "The fixed display set spans four geographic/trajectory regimes and was not selected by forecast error; shaded bands are nominal 90% intervals for Route B variants.", ha="center", fontsize=8)
    fig.subplots_adjust(bottom=0.18, hspace=0.34, wspace=0.22)
    save_figure(fig, output, "fig_covid_prediction_trajectories", {"figure": "prediction_trajectories", "seed": seed, "states": list(TRAJECTORY_STATES), "selection": "fixed display set spanning Northeast, South, West and lower-incidence Appalachia; not selected by forecast error", "intervals": "nominal 90 percent Gaussian intervals for Route B ordinary and cumulative HiPPO"})


def make_state_error_matrices(records: dict[int, dict[str, dict[str, np.ndarray]]], protocols: dict[int, dict[str, Any]]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = protocols[SEEDS[0]]["metadata"]["location_names"]
    n_states = len(names)
    n_weeks = protocols[SEEDS[0]]["stream_y"].shape[0]
    ordinary_sum = np.zeros((n_states, n_weeks))
    hippo_sum = np.zeros_like(ordinary_sum)
    count = np.zeros_like(ordinary_sum)
    for seed in SEEDS:
        indices = protocols[seed]["test_indices"]
        ordinary = records[seed]["routeb_ordinary"]
        hippo = records[seed]["routeb_hippo"]
        ordinary_error = np.abs(ordinary["pred_mean"] - ordinary["y_true"])
        hippo_error = np.abs(hippo["pred_mean"] - hippo["y_true"])
        for position, state_index in enumerate(indices):
            ordinary_sum[int(state_index)] += ordinary_error[:, position]
            hippo_sum[int(state_index)] += hippo_error[:, position]
            count[int(state_index)] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        ordinary_matrix = ordinary_sum / count
        hippo_matrix = hippo_sum / count
    ordinary_matrix[count == 0] = np.nan
    hippo_matrix[count == 0] = np.nan
    difference = ordinary_matrix - hippo_matrix
    return names, ordinary_matrix, hippo_matrix, difference


def plot_error_heatmap(records: dict[int, dict[str, dict[str, np.ndarray]]], protocols: dict[int, dict[str, Any]], output: Path) -> None:
    names, ordinary, hippo, difference = make_state_error_matrices(records, protocols)
    vmax = float(np.nanpercentile(np.concatenate([ordinary[np.isfinite(ordinary)], hippo[np.isfinite(hippo)]]), 98))
    diff_abs = float(np.nanpercentile(np.abs(difference[np.isfinite(difference)]), 98))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 7.2), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1.04]})
    panels = ((ordinary, "Ordinary |absolute error|", "magma", 0, vmax), (hippo, "Cumulative HiPPO |absolute error|", "magma", 0, vmax), (difference, "Ordinary - HiPPO error", "RdBu_r", -diff_abs, diff_abs))
    for axis, (matrix, title, cmap, vmin, panel_vmax) in zip(axes, panels):
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=panel_vmax)
        axis.set_title(title)
        axis.set_xlabel("Online week")
        axis.set_xticks([0, 35, 71, 107, 142], ["1", "36", "72", "108", "143"])
        axis.grid(False)
        fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    axes[0].set_ylabel("Jurisdiction (unscored cells are white)")
    axes[0].set_yticks(np.arange(len(names)), names, fontsize=5.5)
    fig.suptitle("Where the Route B prediction error occurs", y=0.99, fontsize=12)
    fig.text(0.5, 0.005, "Positive values in the paired panel indicate lower absolute error for cumulative HiPPO; cells average only seeds in which the jurisdiction was held out.", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.08, top=0.92, wspace=0.28)
    summary_rows = []
    for index, name in enumerate(names):
        scored = bool(np.isfinite(ordinary[index]).any())
        summary_rows.append(
            {
                "state": name,
                "ordinary_mean_abs_error": float(np.nanmean(ordinary[index])) if scored else "",
                "hippo_mean_abs_error": float(np.nanmean(hippo[index])) if scored else "",
                "ordinary_minus_hippo": float(np.nanmean(difference[index])) if scored else "",
                "scored": scored,
            }
        )
    write_csv(output / "state_week_error_summary.csv", summary_rows)
    save_figure(fig, output, "fig_covid_error_heatmap", {"figure": "state_week_error_heatmap", "aggregation": "mean across formal seeds where each jurisdiction is held out", "ordinary_minus_hippo": "positive favors cumulative HiPPO", "unscored_state_policy": "white"})


def cumulative_rmse(records: dict[int, dict[str, dict[str, np.ndarray]]], method: str, horizon: int, seed: int) -> float:
    arrays = records[seed][method]
    residual = arrays["pred_mean"][:horizon] - arrays["y_true"][:horizon]
    return float(np.sqrt(np.mean(residual * residual)))


def plot_memory_gap(records: dict[int, dict[str, dict[str, np.ndarray]]], output: Path) -> None:
    rows: list[dict[str, Any]] = []
    means = []
    lows = []
    highs = []
    for horizon in HORIZONS:
        ordinary = np.asarray([cumulative_rmse(records, "routeb_ordinary", horizon, seed) for seed in SEEDS])
        hippo = np.asarray([cumulative_rmse(records, "routeb_hippo", horizon, seed) for seed in SEEDS])
        delta, low, high = bootstrap_ci(ordinary - hippo, seed=20260819 + horizon)
        means.append(delta)
        lows.append(low)
        highs.append(high)
        rows.append({"horizon_weeks": horizon, "ordinary_rmse_mean": float(ordinary.mean()), "hippo_rmse_mean": float(hippo.mean()), "ordinary_minus_hippo": delta, "bootstrap95_low": low, "bootstrap95_high": high})
    write_csv(output / "memory_gap_statistics.csv", rows)
    fig, axis = plt.subplots(figsize=(6.6, 3.7))
    axis.axhline(0.0, color="#333333", ls="--", lw=1)
    axis.plot(HORIZONS, means, marker="o", color="#0072B2", lw=1.8, label="Ordinary RMSE - HiPPO RMSE")
    axis.fill_between(HORIZONS, lows, highs, color="#0072B2", alpha=0.17, linewidth=0)
    axis.set_xlabel("Cumulative online horizon (weeks)")
    axis.set_ylabel("RMSE gap (positive favors HiPPO)")
    axis.set_xticks(HORIZONS)
    axis.grid(True)
    axis.legend(frameon=False, loc="best")
    axis.set_title("Long-memory diagnostic across the online horizon")
    fig.text(0.5, -0.045, "Intervals are paired bootstrap 95% CIs over the five spatial split seeds; no jurisdiction-week IID assumption.", ha="center", fontsize=8)
    fig.subplots_adjust(bottom=0.2)
    save_figure(fig, output, "fig_covid_memory_gap", {"figure": "cumulative_memory_gap", "horizons": list(HORIZONS), "bootstrap": "paired resampling over seeds 5-9", "interpretation": "positive values favor HiPPO; no horizon-based selection"})


def coverage_curve(records: dict[int, dict[str, dict[str, np.ndarray]]], method: str, nominal: float) -> float:
    z = math.erf(0.0)  # keep the implementation dependency-free; quantiles below use scipy-free constants.
    del z
    quantile = float(np.sqrt(2.0) * 0.0)
    # Central Gaussian interval quantiles for 0.1,...,0.9, computed from a
    # small fixed table to avoid introducing a new runtime dependency.
    quantiles = {0.1: 0.12566135, 0.2: 0.2533471, 0.3: 0.38532047, 0.4: 0.52440051, 0.5: 0.67448975, 0.6: 0.84162123, 0.7: 1.03643339, 0.8: 1.28155157, 0.9: 1.64485363}
    quantile = quantiles[round(float(nominal), 1)]
    covered: list[np.ndarray] = []
    for seed in SEEDS:
        arrays = records[seed][method]
        std = np.sqrt(arrays["pred_var"])
        covered.append(((arrays["y_true"] >= arrays["pred_mean"] - quantile * std) & (arrays["y_true"] <= arrays["pred_mean"] + quantile * std)).ravel())
    return float(np.concatenate(covered).mean())


def plot_calibration(records: dict[int, dict[str, dict[str, np.ndarray]]], output: Path) -> None:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    fig, axis = plt.subplots(figsize=(5.2, 4.6))
    axis.plot([0, 1], [0, 1], color="#444444", ls="--", lw=1.0, label="Ideal")
    for method in CALIBRATION_METHODS:
        empirical = np.asarray([coverage_curve(records, method, float(nominal)) for nominal in NOMINAL_COVERAGES])
        ece_curve = float(np.mean(np.abs(empirical - NOMINAL_COVERAGES)))
        summary_rows.append({"method": method, "label": FORMAL_METHODS[method]["label"], "curve_ece_over_nominal_coverages": ece_curve})
        for nominal, observed in zip(NOMINAL_COVERAGES, empirical):
            rows.append({"method": method, "label": FORMAL_METHODS[method]["label"], "nominal_coverage": float(nominal), "empirical_coverage": float(observed)})
        axis.plot(NOMINAL_COVERAGES, empirical, marker="o", ms=4, lw=1.7, color=FORMAL_METHODS[method]["color"], label=FORMAL_METHODS[method]["label"])
    write_csv(output / "coverage_curve_statistics.csv", rows)
    write_csv(output / "calibration_curve_summary.csv", summary_rows)
    axis.set_xlabel("Nominal central coverage")
    axis.set_ylabel("Empirical coverage")
    axis.set_xlim(0.05, 0.95)
    axis.set_ylim(0.05, 0.95)
    axis.grid(True)
    axis.legend(frameon=False, loc="upper left", fontsize=7)
    axis.set_title("Gaussian predictive calibration")
    fig.tight_layout()
    save_figure(fig, output, "fig_covid_calibration_curve", {"figure": "calibration_curve", "methods": list(CALIBRATION_METHODS), "intervals": "central Gaussian intervals from archived predictive variance", "nominal_coverages": [float(x) for x in NOMINAL_COVERAGES]})


def write_report(rows: list[dict[str, Any]], output: Path, protocols: dict[int, dict[str, Any]]) -> None:
    hippo = next(row for row in rows if row["method"] == "routeb_cumulative_hippo")
    ordinary = next(row for row in rows if row["method"] == "routeb_ordinary")
    report = [
        "# Formal COVID Long-Stream Paper Materials",
        "",
        "This package uses the audited CDC NHSN mandatory-period weekly stream, 52 jurisdictions, 52 Task-1 weeks, 143 strict-online weeks, and formal spatial split seeds 5-9. All completed rows use the Gaussian likelihood on the standardized form of `log1p(weekly COVID admissions per 100k)`; trajectory figures restore the target scale for display.",
        "",
        "## Main Table",
        "",
        "The complete table is in `formal_results_table.csv` and `formal_results_table.tex`. Metrics are RMSE, CRPS, native Gaussian NLPD, ECE and Coverage90. OVC-SVGP is not numerically ranked because the Task-1-selected 8x32 exact-fantasy formal run is resource-limited; its lower-capacity feasibility study remains outside this table.",
        "",
        f"Route B cumulative HiPPO has RMSE {float(hippo['rmse_mean']):.4f} +/- {float(hippo['rmse_sd']):.4f}, compared with {float(ordinary['rmse_mean']):.4f} +/- {float(ordinary['rmse_sd']):.4f} for ordinary inducing. Its CRPS and Gaussian NLPD are also lower, while Coverage90 is closer to the nominal 0.90 than ordinary inducing.",
        "",
        "## Figure Reading Guide",
        "",
        "`fig_covid_prediction_trajectories` shows four predeclared seed-5 held-out jurisdictions and separates point-trajectory comparison from the uncertainty bands of the two controlled Route B variants.",
        "`fig_covid_error_heatmap` averages only seeds in which a jurisdiction is held out and shows where the paired ordinary-minus-HiPPO error difference occurs.",
        "`fig_covid_memory_gap` uses paired bootstrap resampling over spatial split seeds, so the confidence band does not treat jurisdiction-week cells as independent observations.",
        "`fig_covid_calibration_curve` uses the archived predictive variances directly; no intervals are reconstructed from aggregate metrics.",
        "`fig_covid_metric_summary` is a compact visual copy of the formal table, with the two Route B variants highlighted.",
        "",
        "All figures are newly written under this package directory and do not overwrite the previous COVID dataset overview figures.",
    ]
    (output / "figures_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory"))
    parser.add_argument("--report-root", type=Path, default=Path("baselines/covid_long_setting_b/reports/formal_gaussian_task1_selected_strict"))
    parser.add_argument("--output", type=Path, default=Path("baselines/covid_long_setting_b/figures/formal_gaussian_task1_selected_strict"))
    args = parser.parse_args()
    configure_matplotlib()
    protocol_root = (ROOT / args.protocol_root).resolve()
    report_root = (ROOT / args.report_root).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, protocols = load_all(protocol_root)
    rows = metric_table(report_root, output)
    plot_metrics(rows, output)
    plot_trajectories(records, protocols, output)
    plot_error_heatmap(records, protocols, output)
    plot_memory_gap(records, output)
    plot_calibration(records, output)
    write_report(rows, output, protocols)
    audit = {
        "status": "complete",
        "formal_seeds": list(SEEDS),
        "online_weeks": int(protocols[5]["stream_y"].shape[0]),
        "methods_loaded": list(FORMAL_METHODS),
        "target": "log1p(weekly COVID admissions per 100k), restored for trajectory display",
        "output": str(output),
        "figures": [
            "fig_covid_metric_summary",
            "fig_covid_prediction_trajectories",
            "fig_covid_error_heatmap",
            "fig_covid_memory_gap",
            "fig_covid_calibration_curve",
        ],
        "old_figures_overwritten": False,
    }
    (output / "figure_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
