#!/usr/bin/env python3
"""Generate the Stage 2+ ERA5 benchmark report and publication diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "xlag_mean_only": "X-lag mean only",
    "routeb_residual_analytic_hippo_rff": "Route B HiPPO residual",
    "routeb_residual_inducing_points": "Route B ordinary residual",
    "routeb_joint_analytic_hippo_rff": "Structured-joint Route B HiPPO",
    "routeb_joint_inducing_points": "Structured-joint Route B ordinary",
    "xlag_task1_fixed": "Task-1 frozen X-lag mean",
    "xlag_recursive_rls": "Recursive X-lag / RLS",
    "routeb_analytic_hippo_rff": "Route B cumulative HiPPO",
    "routeb_inducing_points": "Route B ordinary inducing",
}

COLORS = {
    "xlag_mean_only": "#7f7f7f",
    "xlag_task1_fixed": "#7f7f7f",
    "xlag_recursive_rls": "#bcbd22",
    "routeb_residual_analytic_hippo_rff": "#0072b2",
    "routeb_joint_analytic_hippo_rff": "#0072b2",
    "routeb_analytic_hippo_rff": "#0072b2",
    "routeb_residual_inducing_points": "#d55e00",
    "routeb_joint_inducing_points": "#d55e00",
    "routeb_inducing_points": "#d55e00",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(payload: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = payload
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            return current
    return None


def method_label(method: str) -> str:
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    return (
        method.replace("official_", "Official ")
        .replace("gpflow_svgp", "GPflow SVGP")
        .replace("bui_osgpr", "Bui OSGPR")
        .replace("maddox_streaming_sgpr", "Maddox StreamingSGPR")
        .replace("ohsvgp", "OHSVGP")
        .replace("markovflow_sparse_variational", "Markovflow sparse variational")
        .replace("markovflow_sparse_cvi", "Markovflow sparse CVI")
        .replace("st_svgp", "ST-SVGP")
        .replace("mf_ST-SVGP", "MF-ST-SVGP")
        .replace("_", " ")
    )


def method_color(method: str, index: int = 0) -> str:
    if method in COLORS:
        return COLORS[method]
    palette = ["#009e73", "#cc79a7", "#56b4e9", "#e69f00", "#332288", "#44aa99"]
    return palette[index % len(palette)]


def collect_runs(benchmark: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = benchmark / "runs"
    if not root.is_dir():
        return pd.DataFrame()
    for method_dir in sorted(root.glob("*/*/*")):
        if not method_dir.is_dir():
            continue
        relative = method_dir.relative_to(root).parts
        if len(relative) != 3:
            continue
        scope, branch, method = relative
        for seed_dir in sorted(method_dir.glob("seed*")):
            try:
                seed = int(seed_dir.name.removeprefix("seed"))
            except ValueError:
                continue
            result_path = seed_dir / "result.json"
            status_path = seed_dir / "status.json"
            status = read_json(status_path) if status_path.is_file() else {}
            payload = read_json(result_path) if result_path.is_file() else {}
            rows.append(
                {
                    "scope": scope,
                    "branch": branch,
                    "method": method,
                    "label": method_label(method),
                    "seed": seed,
                    "status": status.get("status", "complete" if payload else "missing"),
                    "device_class": status.get("device_class"),
                    "legacy": bool(status.get("legacy", False)),
                    "rmse": nested(payload, "overall_current_block.rmse", "final.rmse", "rmse"),
                    "nll": nested(payload, "overall_current_block.nll", "final.nll", "nll"),
                    "coverage90": nested(
                        payload,
                        "overall_current_block.coverage90",
                        "final.coverage90",
                        "coverage90",
                    ),
                    "mean_predictive_std": nested(
                        payload,
                        "overall_current_block.mean_predictive_std",
                        "final.mean_predictive_std",
                        "mean_predictive_std",
                    ),
                    "result_path": str(result_path),
                    "run_dir": str(seed_dir),
                }
            )
    return pd.DataFrame(rows)


def aggregate_runs(frame: pd.DataFrame, expected_seeds: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    completed = frame[(frame.status == "complete") & frame.rmse.notna()].copy()
    rows: list[dict[str, Any]] = []
    for (scope, branch, method, label), group in completed.groupby(
        ["scope", "branch", "method", "label"], sort=False
    ):
        row: dict[str, Any] = {
            "scope": scope,
            "branch": branch,
            "method": method,
            "label": label,
            "completed_seeds": int(group.seed.nunique()),
            "expected_seeds": int(expected_seeds),
            "device_class": ", ".join(sorted(set(group.device_class.dropna().astype(str)))),
        }
        for metric in ("rmse", "nll", "coverage90", "mean_predictive_std"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(values.mean()) if values.size else np.nan
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def format_mean_sd(mean: Any, sd: Any, digits: int = 4) -> str:
    if mean is None or not np.isfinite(float(mean)):
        return "--"
    return f"{float(mean):.{digits}f} +/- {float(sd):.{digits}f}"


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metric_bars(summary: pd.DataFrame, scope: str, output: Path) -> Path | None:
    if summary.empty or not {"scope", "branch"}.issubset(summary.columns):
        return None
    data = summary[(summary.scope == scope) & (summary.branch == "batch")].copy()
    if data.empty:
        return None
    data = data.sort_values("rmse_mean", ascending=True)
    height = max(4.2, 0.48 * len(data) + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, height), sharey=True)
    positions = np.arange(len(data))
    colors = [method_color(method, index) for index, method in enumerate(data.method)]
    for axis, metric, title in zip(axes, ("rmse", "nll"), ("RMSE (lower is better)", "NLL / NLPD (lower is better)")):
        axis.barh(
            positions,
            data[f"{metric}_mean"],
            xerr=data[f"{metric}_sd"],
            color=colors,
            alpha=0.9,
            capsize=3,
        )
        axis.set_title(title)
        axis.grid(axis="x", color="#d9d9d9", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[0].set_yticks(positions, data.label)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle(f"Controlled shared-X-lag batch comparison: {scope}", fontsize=13)
    fig.tight_layout()
    path = output / f"batch_metrics_{scope}"
    save_figure(fig, path)
    return path.with_suffix(".png")


def plot_paired_differences(frame: pd.DataFrame, scope: str, output: Path) -> Path | None:
    if frame.empty or not {"scope", "branch", "status"}.issubset(frame.columns):
        return None
    data = frame[(frame.scope == scope) & (frame.branch == "batch") & (frame.status == "complete")].copy()
    reference = "routeb_joint_analytic_hippo_rff"
    if reference not in set(data.method):
        return None
    pivot = data.pivot_table(index="seed", columns="method", values="rmse", aggfunc="first")
    candidates = [method for method in pivot.columns if method != reference]
    paired = []
    labels = []
    methods = []
    for method in candidates:
        values = (pivot[method] - pivot[reference]).dropna()
        if values.empty:
            continue
        paired.append(values)
        labels.append(method_label(method))
        methods.append(method)
    if not paired:
        return None
    fig, ax = plt.subplots(figsize=(max(8.5, 1.15 * len(paired)), 5.2))
    rng = np.random.default_rng(20260803)
    for index, (values, method) in enumerate(zip(paired, methods)):
        jitter = rng.uniform(-0.07, 0.07, size=len(values))
        ax.scatter(
            index + jitter,
            values.to_numpy(float),
            s=40,
            color=method_color(method, index),
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.plot([index - 0.18, index + 0.18], [values.mean(), values.mean()], color="black", linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
    ax.set_ylabel("Paired RMSE difference vs structured-joint Route B HiPPO")
    ax.set_title(f"Paired spatial-split comparison: {scope}")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    fig.tight_layout()
    path = output / f"paired_seed_rmse_{scope}"
    save_figure(fig, path)
    return path.with_suffix(".png")


def read_blocks(run_dir: str) -> pd.DataFrame | None:
    path = Path(run_dir) / "blocks.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    required = {"block_id", "rmse", "nll"}
    return frame if required.issubset(frame.columns) else None


def plot_online_curves(frame: pd.DataFrame, scope: str, output: Path) -> Path | None:
    if frame.empty or not {"scope", "branch", "status"}.issubset(frame.columns):
        return None
    data = frame[(frame.scope == scope) & (frame.branch == "online") & (frame.status == "complete")]
    series: dict[str, list[pd.DataFrame]] = {}
    for row in data.itertuples(index=False):
        blocks = read_blocks(row.run_dir)
        if blocks is not None:
            series.setdefault(row.method, []).append(blocks)
    if not series:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True)
    for index, (method, runs) in enumerate(series.items()):
        common = min(len(run) for run in runs)
        block_ids = runs[0].block_id.to_numpy()[:common]
        for axis, metric in zip(axes, ("rmse", "nll")):
            values = np.stack([pd.to_numeric(run[metric], errors="coerce").to_numpy()[:common] for run in runs])
            mean = np.nanmean(values, axis=0)
            sd = np.nanstd(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros(common)
            color = method_color(method, index)
            axis.plot(block_ids, mean, label=method_label(method), color=color, linewidth=1.8)
            axis.fill_between(block_ids, mean - sd, mean + sd, color=color, alpha=0.12, linewidth=0)
    axes[0].set_ylabel("Block RMSE")
    axes[1].set_ylabel("Block NLL / NLPD")
    axes[1].set_xlabel("Streaming block")
    for axis in axes:
        axis.grid(color="#dddddd", linewidth=0.7)
    axes[0].legend(ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"Strict-online performance across the stream: {scope}", fontsize=13)
    fig.tight_layout()
    path = output / f"online_block_curves_{scope}"
    save_figure(fig, path)
    return path.with_suffix(".png")


def prediction_path(benchmark: Path, scope: str, branch: str, method: str, seed: int) -> Path:
    return benchmark / "runs" / scope / branch / method / f"seed{seed}" / "predictions.npz"


def plot_representative_trajectories(
    benchmark: Path, frame: pd.DataFrame, scope: str, output: Path
) -> tuple[Path | None, dict[str, Any] | None]:
    if frame.empty or not {"scope", "branch", "seed", "status"}.issubset(frame.columns):
        return None, None
    primary = "routeb_analytic_hippo_rff"
    available = frame[
        (frame.scope == scope)
        & (frame.branch == "online")
        & (frame.seed == 0)
        & (frame.status == "complete")
    ]
    if primary not in set(available.method):
        return None, None
    primary_path = prediction_path(benchmark, scope, "online", primary, 0)
    if not primary_path.is_file():
        return None, None
    with np.load(primary_path) as arrays:
        y = np.asarray(arrays["y_true"], dtype=float)
        mean = np.asarray(arrays["pred_mean"], dtype=float)
        test_indices = np.asarray(arrays.get("test_indices", np.arange(y.shape[1])), dtype=int)
        times = np.asarray(arrays.get("times", np.arange(y.shape[0])), dtype=float)
    per_location = np.sqrt(np.mean((y - mean) ** 2, axis=0))
    quantiles = {"success": 0.1, "median": 0.5, "failure": 0.9}
    chosen: dict[str, int] = {}
    for label, quantile in quantiles.items():
        target = float(np.quantile(per_location, quantile))
        chosen[label] = int(np.argmin(np.abs(per_location - target)))
    preferred = [primary, "routeb_inducing_points"]
    preferred.extend(
        method for method in available.method if method.startswith("official_ohsvgp")
    )
    methods = [method for method in preferred if method in set(available.method)]
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in methods:
        path = prediction_path(benchmark, scope, "online", method, 0)
        if not path.is_file():
            continue
        with np.load(path) as arrays:
            predictions[method] = (
                np.asarray(arrays["pred_mean"], dtype=float),
                np.asarray(arrays["pred_var"], dtype=float),
            )
    if not predictions:
        return None, None
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.2), sharex=True)
    for row, (category, local_index) in enumerate(chosen.items()):
        axis = axes[row]
        axis.plot(times, y[:, local_index], color="black", linewidth=1.35, label="Ground truth")
        for index, (method, (pred_mean, pred_var)) in enumerate(predictions.items()):
            color = method_color(method, index)
            axis.plot(times, pred_mean[:, local_index], color=color, linewidth=1.25, label=method_label(method))
            if method == primary:
                half = 1.6448536269514722 * np.sqrt(np.maximum(pred_var[:, local_index], 1e-10))
                axis.fill_between(
                    times,
                    pred_mean[:, local_index] - half,
                    pred_mean[:, local_index] + half,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                    label="Route B HiPPO 90% interval",
                )
        axis.set_ylabel(category.capitalize())
        axis.grid(color="#e0e0e0", linewidth=0.6)
        axis.set_title(
            f"Global location {test_indices[local_index]} | primary RMSE={per_location[local_index]:.4f}",
            fontsize=10,
            loc="left",
        )
    axes[0].legend(ncol=2, fontsize=8, frameon=False)
    axes[-1].set_xlabel("Normalized stream time")
    fig.suptitle(
        "Predefined representative locations (10th, 50th and 90th RMSE percentiles)",
        fontsize=13,
    )
    fig.tight_layout()
    path = output / f"representative_trajectories_{scope}"
    save_figure(fig, path)
    selection = {
        "scope": scope,
        "seed": 0,
        "selection_method": primary,
        "rule": "closest location to the 10th, 50th and 90th percentiles of per-location RMSE",
        "locations": {
            category: {
                "local_test_index": local,
                "global_location_index": int(test_indices[local]),
                "primary_rmse": float(per_location[local]),
            }
            for category, local in chosen.items()
        },
    }
    return path.with_suffix(".png"), selection


def plot_spatial_protocol(benchmark: Path, output: Path) -> Path | None:
    protocol = benchmark / "protocol" / "task1_2" / "seed0" / "protocol.npz"
    if not protocol.is_file():
        return None
    with np.load(protocol) as arrays:
        coords = np.asarray(arrays["coordinates"], dtype=float)
        train = np.asarray(arrays["train_indices"], dtype=int)
        test = np.asarray(arrays["test_indices"], dtype=int)
        inducing = np.asarray(arrays["inducing_coords_ms128"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(coords[train, 0], coords[train, 1], s=10, color="#8da0cb", alpha=0.45, label="Training locations")
    ax.scatter(coords[test, 0], coords[test, 1], s=16, color="#e41a1c", alpha=0.8, label="Held-out locations")
    ax.scatter(inducing[:, 0], inducing[:, 1], s=42, marker="x", linewidth=1.3, color="black", label="Spatial inducing points (Ms=128)")
    ax.set_xlabel("Standardized longitude")
    ax.set_ylabel("Standardized latitude")
    ax.set_title("Seed-0 spatial split and inducing-point placement")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color="#e0e0e0", linewidth=0.6)
    fig.tight_layout()
    path = output / "spatial_protocol_seed0"
    save_figure(fig, path)
    return path.with_suffix(".png")


def plot_final_snapshot(benchmark: Path, scope: str, output: Path) -> Path | None:
    method = "routeb_analytic_hippo_rff"
    prediction = prediction_path(benchmark, scope, "online", method, 0)
    protocol = benchmark / "protocol" / scope / "seed0" / "protocol.npz"
    if not prediction.is_file() or not protocol.is_file():
        return None
    with np.load(prediction) as pred, np.load(protocol) as arrays:
        truth = np.asarray(pred["y_true"], dtype=float)[-1]
        mean = np.asarray(pred["pred_mean"], dtype=float)[-1]
        test = np.asarray(pred.get("test_indices", arrays["test_indices"]), dtype=int)
        coords = np.asarray(arrays["coordinates"], dtype=float)[test]
    error = mean - truth
    value_limit = max(float(np.max(np.abs(truth))), float(np.max(np.abs(mean))), 1e-9)
    error_limit = max(float(np.max(np.abs(error))), 1e-9)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for axis, values, title, limit in (
        (axes[0], truth, "Ground truth", value_limit),
        (axes[1], mean, "Route B HiPPO prediction", value_limit),
        (axes[2], error, "Prediction error", error_limit),
    ):
        artist = axis.scatter(coords[:, 0], coords[:, 1], c=values, s=25, cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(title)
        axis.set_xlabel("Longitude (standardized)")
        axis.set_aspect("equal", adjustable="box")
        fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("Latitude (standardized)")
    fig.suptitle(f"Final-time held-out spatial field: {scope}", fontsize=13)
    path = output / f"final_spatial_snapshot_{scope}"
    save_figure(fig, path)
    return path.with_suffix(".png")


def table_rows(summary: pd.DataFrame, scope: str, branch: str) -> list[list[str]]:
    if summary.empty or not {"scope", "branch"}.issubset(summary.columns):
        return []
    data = summary[(summary.scope == scope) & (summary.branch == branch)].copy()
    if data.empty:
        return []
    data = data.sort_values("rmse_mean")
    return [
        [
            row.label,
            f"{int(row.completed_seeds)}/{int(row.expected_seeds)}",
            format_mean_sd(row.rmse_mean, row.rmse_sd),
            format_mean_sd(row.nll_mean, row.nll_sd),
            format_mean_sd(row.coverage90_mean, row.coverage90_sd),
            row.device_class or "not recorded",
        ]
        for row in data.itertuples(index=False)
    ]


def efficiency_rows(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    if "legacy" in frame:
        frame = frame[~frame.legacy.fillna(False).astype(bool)]
    rows = []
    for row in frame.sort_values(["scope", "branch", "method"]).itertuples(index=False):
        def value(name: str, digits: int = 3) -> str:
            current = getattr(row, name, np.nan)
            return "--" if pd.isna(current) else f"{float(current):.{digits}f}"

        rows.append(
            [
                row.scope,
                row.branch,
                method_label(row.method),
                str(getattr(row, "device", "--")),
                value("end_to_end_with_calibration_seconds_mean", 2),
                value("mean_iteration_seconds_mean", 4),
                value("mean_block_update_seconds_mean", 4),
                value("prediction_seconds_mean", 3),
                value("peak_rss_mib_mean", 1),
                value("nvidia_smi_peak_total_used_mib_mean", 1),
                value("persistent_state_mib_mean", 2),
                value("estimated_gflops_mean", 2),
            ]
        )
    return rows


def compatibility_rows(frame: pd.DataFrame) -> list[list[str]]:
    if frame.empty:
        return []
    core = frame[frame.legacy | frame.device_class.fillna("").str.contains("legacy")]
    return [
        [row.scope, row.branch, method_label(row.method), str(row.seed), row.status, row.device_class or "legacy"]
        for row in core.sort_values(["scope", "method", "seed"]).itertuples(index=False)
    ]


def best_method_text(summary: pd.DataFrame, scope: str, branch: str) -> str:
    if summary.empty or not {"scope", "branch", "rmse_mean"}.issubset(summary.columns):
        return "No completed run is available yet."
    data = summary[(summary.scope == scope) & (summary.branch == branch)].dropna(subset=["rmse_mean"])
    if data.empty:
        return "No completed run is available yet."
    best = data.loc[data.rmse_mean.idxmin()]
    completeness = f"{int(best.completed_seeds)}/{int(best.expected_seeds)} seeds"
    return (
        f"The lowest observed mean RMSE is {best.rmse_mean:.4f} from "
        f"{best.label} ({completeness}). This is descriptive until all planned "
        "paired seeds complete."
    )


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_table(headers: list[str], rows: list[list[str]]) -> str:
    columns = "l" + "r" * (len(headers) - 1)
    body = [" & ".join(latex_escape(value) for value in headers) + r" \\", r"\midrule"]
    body.extend(" & ".join(latex_escape(str(value)) for value in row) + r" \\" for row in rows)
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering\scriptsize",
            r"\resizebox{\textwidth}{!}{%",
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            *body,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )


def write_latex_report(
    output: Path,
    summary: pd.DataFrame,
    figures: list[Path],
    efficiency: list[list[str]],
    audit: dict[str, Any],
) -> Path:
    sections = []
    for scope in ("task1_2", "task1_10"):
        batch = table_rows(summary, scope, "batch")
        online = table_rows(summary, scope, "online")
        sections.extend(
            [
                rf"\subsection{{{latex_escape(scope)}: controlled shared-X-lag batch}}",
                latex_table(["Method", "Seeds", "RMSE", "NLL", "Coverage90", "Device class"], batch)
                if batch
                else "No completed runs.",
                rf"\subsection{{{latex_escape(scope)}: strict online streaming}}",
                latex_table(["Method", "Seeds", "RMSE", "NLL", "Coverage90", "Device class"], online)
                if online
                else "No completed runs.",
            ]
        )
    if efficiency:
        sections.extend(
            [
                r"\section{Efficiency}",
                latex_table(
                    ["Scope", "Mode", "Method", "Device", "End-to-end s", "Iter s", "Block s", "Predict s", "RSS MiB", "GPU MiB", "State MiB", "GFLOPs"],
                    efficiency,
                ),
            ]
        )
    for figure in figures:
        sections.extend(
            [
                r"\begin{figure}[htbp]",
                r"\centering",
                rf"\includegraphics[width=0.98\textwidth]{{{latex_escape(figure.name)}}}",
                r"\end{figure}",
            ]
        )
    status = audit.get("verification_status", "NOT RUN")
    tex = "\n".join(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[a4paper,margin=1.7cm]{geometry}",
            r"\usepackage{booktabs,graphicx,xcolor,hyperref}",
            r"\title{ERA5 Stage 2+ Benchmark: Controlled Batch, Strict Online, and Efficiency}",
            r"\author{AutoDL RTX 4090 reproducibility run}",
            r"\date{Generated from measured artifacts}",
            r"\begin{document}",
            r"\maketitle",
            r"\section{Protocol}",
            "Task 1 is used for causal hyperparameter calibration. Task 2, or Tasks 2--10, form the evaluation stream. All methods share the same spatial splits, X-lag construction, targets, and metrics. Legacy CPU compatibility runs are not used for same-GPU speed claims.",
            rf"Reproducibility audit status: \textbf{{{latex_escape(str(status))}}}.",
            r"\section{Predictive results}",
            *sections,
            r"\section{Interpretation boundary}",
            "Only paired seeds produced by the same protocol may support method comparisons. FLOP values are compared only within a common instrumentation scope, and legacy TensorFlow/JAX CPU compatibility timings are reported separately from RTX 4090 timings.",
            r"\end{document}",
        ]
    )
    path = output / "report.tex"
    path.write_text(tex, encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    frame = collect_runs(args.benchmark_root)
    summary = aggregate_runs(frame, len(config["split_seeds"]))
    frame.to_csv(args.output_dir / "results_per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "results_summary.csv", index=False)

    audit_path = args.benchmark_root / "audit.json"
    audit = read_json(audit_path) if audit_path.is_file() else {"verification_status": "NOT RUN", "issues": []}
    figures: list[Path] = []
    selections: dict[str, Any] = {}
    spatial = plot_spatial_protocol(args.benchmark_root, args.output_dir)
    if spatial:
        figures.append(spatial)
    for scope in config["scopes"]:
        for candidate in (
            plot_metric_bars(summary, scope, args.output_dir),
            plot_paired_differences(frame, scope, args.output_dir),
            plot_online_curves(frame, scope, args.output_dir),
            plot_final_snapshot(args.benchmark_root, scope, args.output_dir),
        ):
            if candidate:
                figures.append(candidate)
        trajectory, selection = plot_representative_trajectories(
            args.benchmark_root, frame, scope, args.output_dir
        )
        if trajectory:
            figures.append(trajectory)
        if selection:
            selections[scope] = selection
    (args.output_dir / "representative_location_selection.json").write_text(
        json.dumps(selections, indent=2) + "\n", encoding="utf-8"
    )

    efficiency_path = args.benchmark_root / "efficiency" / "efficiency_summary.csv"
    efficiency = efficiency_rows(efficiency_path)
    compatibility = compatibility_rows(frame)
    sections = [
        "# ERA5 Stage 2+ Benchmark Report",
        "",
        "This report is generated only from measured run artifacts. Stage 1 direct-target experiments are intentionally excluded.",
        "",
        "## Experimental protocol",
        "",
        "- **Task 1:** Route-B empirical-Bayes calibration for the strict-online methods.",
        "- **Short horizon:** Task 2, 186 hourly steps, 19 task-aware blocks.",
        "- **Long horizon:** Tasks 2--10, 1,674 hourly steps, 171 task-aware blocks.",
        "- **Spatial evaluation:** 1,000 fixed locations with 800 training and 200 held-out locations.",
        f"- **Paired split seeds:** {', '.join(map(str, config['split_seeds']))}.",
        "- **Shared-X-lag batch:** every residual model receives the same ridge mean and fits the same residual; structured-joint Route B is shown separately because beta and the GP posterior are coupled.",
        "- **Strict online:** only Task-1 calibration and new-block observations are allowed; history replay is audited to be zero.",
        "- **Efficiency boundary:** modern methods are measured on the same RTX 4090 host. Bayes-Newton ST-SVGP and Markovflow v0.0.13 are legacy CPU compatibility runs and are never folded into same-GPU speedup claims.",
    ]
    for scope in config["scopes"]:
        sections.extend(
            [
                "",
                f"## Stage 2: Controlled shared-X-lag batch ({scope})",
                "",
                markdown_table(
                    ["Method", "Seeds", "RMSE", "NLL / NLPD", "Coverage90", "Device class"],
                    table_rows(summary, scope, "batch"),
                )
                if table_rows(summary, scope, "batch")
                else "No completed batch runs are available yet.",
                "",
                best_method_text(summary, scope, "batch"),
                "",
                f"## Stage 3: Strict online streaming ({scope})",
                "",
                markdown_table(
                    ["Method", "Seeds", "RMSE", "NLL / NLPD", "Coverage90", "Device class"],
                    table_rows(summary, scope, "online"),
                )
                if table_rows(summary, scope, "online")
                else "No completed online runs are available yet.",
                "",
                best_method_text(summary, scope, "online"),
            ]
        )
    sections.extend(
        [
            "",
            "## Stage 4: Efficiency",
            "",
            markdown_table(
                ["Scope", "Mode", "Method", "Device", "End-to-end s", "Iteration s", "Block update s", "Prediction s", "Peak RSS MiB", "Peak GPU MiB", "State MiB", "GFLOPs"],
                efficiency,
            )
            if efficiency
            else "Efficiency aggregation has not completed yet.",
            "",
            "The end-to-end online time charges each method for its Task-1 calibration where applicable. FLOP numbers are method/backend measurements or documented analytic estimates; values with different `flops_scope` definitions must not be treated as exact head-to-head hardware instruction counts.",
            "",
            "## Legacy compatibility attempts",
            "",
            markdown_table(["Scope", "Mode", "Method", "Seed", "Status", "Device class"], compatibility)
            if compatibility
            else "No legacy compatibility run was requested.",
            "",
            "An OOM, timeout, dependency failure, or GPU incompatibility is reported as an execution outcome, not as a predictive score. The official legacy stacks preserve their pinned paper-era dependencies; porting them to a modern GPU backend would be a new implementation rather than an official-code timing.",
            "",
            "## Stage 5: Visualization and reproducibility audit",
            "",
            f"Audit status: **{audit.get('verification_status', 'NOT RUN')}**. Completed modern runs: {audit.get('complete_modern_runs', 0)}/{audit.get('expected_modern_runs', 0)}.",
            "",
            f"Open audit issues: {len(audit.get('issues', []))}.",
            "",
        ]
    )
    for figure in figures:
        sections.extend([f"![{figure.stem}]({figure.name})", ""])
    sections.extend(
        [
            "## Interpretation rules",
            "",
            "1. Accuracy claims require paired spatial-split seeds and the same target/X-lag protocol.",
            "2. Streaming claims require zero history replay and causal Task-1-only hyperparameter calibration.",
            "3. Same-GPU speed claims exclude legacy CPU compatibility paths and separately report warm-up, steady-state, prediction, memory, state size, and calibration time.",
            "4. A five-seed mean and standard deviation are descriptive; any confidence interval or paired test should be computed from the saved per-seed table rather than inferred from rounded values.",
            "5. Representative locations are selected by a predefined RMSE-quantile rule saved in `representative_location_selection.json`, not by visual preference.",
        ]
    )
    report_md = args.output_dir / "report.md"
    report_md.write_text("\n".join(sections) + "\n", encoding="utf-8")

    tex_path = write_latex_report(args.output_dir, summary, figures, efficiency, audit)
    latex_status: dict[str, Any] = {"attempted": False, "pdf": None}
    if shutil.which("latexmk"):
        latex_status["attempted"] = True
        completed = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=args.output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
            check=False,
        )
        (args.output_dir / "latex_build.log").write_text(completed.stdout, encoding="utf-8")
        pdf = args.output_dir / "report.pdf"
        latex_status.update({"returncode": completed.returncode, "pdf": str(pdf) if pdf.is_file() else None})

    inputs = sorted((args.benchmark_root / "runs").glob("*/*/*/seed*/result.json"))
    generated = sorted(
        path for path in args.output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "benchmark_root": str(args.benchmark_root.resolve()),
        "config": {"path": str(args.config.resolve()), "sha256": sha256(args.config)},
        "audit_status": audit.get("verification_status", "NOT RUN"),
        "latex": latex_status,
        "source_results": [{"path": str(path), "sha256": sha256(path)} for path in inputs],
        "generated_artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in generated
        ],
    }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(report_md),
                "pdf": latex_status.get("pdf"),
                "figures": len(figures),
                "runs": len(frame),
                "audit_status": audit.get("verification_status", "NOT RUN"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
