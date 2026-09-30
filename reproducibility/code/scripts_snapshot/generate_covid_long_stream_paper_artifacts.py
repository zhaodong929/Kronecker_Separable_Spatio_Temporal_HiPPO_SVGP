#!/usr/bin/env python3
"""Recompute audited COVID long-stream statistics and paper figures from archives."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.run_epidemiology_pilot import predictive_metrics
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_epidemiology_pilot import predictive_metrics


ROOT = Path(__file__).resolve().parents[1]
FORMAL_SEEDS = (5, 6, 7, 8, 9)
METRICS = ("rmse", "nll", "coverage90")
METHODS = (
    ("persistence", "Persistence", "deterministic/persistence", "overall_current_block"),
    ("lag_ridge", "Task-1 lag ridge", "deterministic/lag_ridge", "overall_current_block"),
    ("ohsvgp_rbf", "Official OHSVGP (RBF)", "ohsvgp_rbf", "overall_current_block"),
    ("bui_osgpr_controlled", "Bui OSGPR (controlled)", "bui_osgpr_controlled", "final"),
    ("bui_osgpr_adaptive", "Bui OSGPR (adaptive, CPU)", "bui_osgpr_adaptive", "final"),
    ("routeb_ordinary", "Route B ordinary inducing", "routeb_ordinary/online", "overall_current_block"),
    ("routeb_cumulative", "Route B cumulative HiPPO", "routeb_cumulative/online", "overall_current_block"),
)
COLORS = {
    "persistence": "#7B8794",
    "lag_ridge": "#999999",
    "ohsvgp_rbf": "#D55E00",
    "bui_osgpr_controlled": "#E69F00",
    "bui_osgpr_adaptive": "#CC79A7",
    "routeb_ordinary": "#009E73",
    "routeb_cumulative": "#0072B2",
}
HORIZONS_REQUESTED = (16, 32, 48, 64, 80, 96, 112, 128, 144)
TRAJECTORY_STATES = ("North Dakota", "Hawaii", "Oklahoma", "Louisiana", "California")


def metrics_from_arrays(y_true: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    if y_true.shape != mean.shape or y_true.shape != variance.shape:
        raise ValueError(f"Prediction shapes differ: target={y_true.shape}, mean={mean.shape}, variance={variance.shape}")
    if not np.isfinite(y_true).all() or not np.isfinite(mean).all():
        raise ValueError("Predictions contain a non-finite target or mean")
    if not np.isfinite(variance).all() or (variance <= 0.0).any():
        raise ValueError("Predictions contain a non-positive or non-finite variance")
    return {name: float(value) for name, value in predictive_metrics(y_true, mean, variance).items()}


def read_prediction_archive(path: Path) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    with np.load(path) as archive:
        required = ("y_true", "pred_mean", "pred_var", "test_indices")
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        arrays = {key: np.asarray(archive[key]) for key in required}
    arrays["y_true"] = np.asarray(arrays["y_true"], dtype=np.float64)
    arrays["pred_mean"] = np.asarray(arrays["pred_mean"], dtype=np.float64)
    arrays["pred_var"] = np.asarray(arrays["pred_var"], dtype=np.float64)
    arrays["test_indices"] = np.asarray(arrays["test_indices"], dtype=np.int64)
    return metrics_from_arrays(arrays["y_true"], arrays["pred_mean"], arrays["pred_var"]), arrays


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def paired_bootstrap(values: list[float], *, seed: int = 0) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, array.size, size=(20_000, array.size))].mean(axis=1)
    return float(array.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, target: Path, config: dict[str, Any]) -> None:
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=240, bbox_inches="tight")
    target.with_suffix(".config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)


def run_path(root: Path, seed: int, relative: str) -> Path:
    return root / f"seed{seed}" / relative


def reported_metrics(payload: dict[str, Any], key: str) -> dict[str, float]:
    if key not in payload:
        raise ValueError(f"result.json has no {key!r} metrics")
    return {metric: float(payload[key][metric]) for metric in METRICS}


def timing_and_resources(payload: dict[str, Any]) -> dict[str, Any]:
    timing = payload.get("timing", {})
    resources = payload.get("resources", {})
    return {
        "update_seconds": float(timing.get("mean_steady_state_block_update_seconds", timing.get("mean_block_update_seconds", 0.0))),
        "prediction_seconds": float(timing.get("mean_block_prediction_seconds", 0.0)),
        "compile_seconds": float(timing.get("compile_seconds", timing.get("jit_compile_seconds", 0.0))),
        "persistent_state_mib": float(resources.get("persistent_state_mib", np.nan)),
        "peak_allocated_mib": float(resources.get("peak_cuda_allocated_mib", np.nan)),
        "peak_reserved_mib": float(resources.get("peak_cuda_reserved_mib", np.nan)),
        "cpu_rss_mib": float(resources.get("cpu_rss_mib", np.nan)),
        "device": str(resources.get("device", resources.get("solver_device", "not_instrumented"))),
    }


def audit_run(
    *,
    root: Path,
    seed: int,
    method: str,
    relative: str,
    result_key: str,
    expected_weeks: int,
    delay_weeks: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path = run_path(root, seed, relative)
    result_path = path / "result.json"
    prediction_path = path / "predictions.npz"
    if not result_path.is_file() or not prediction_path.is_file():
        status_path = path / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        return None, {
            "seed": seed,
            "method": method,
            "path": str(path),
            "status": str(status.get("status", "missing_artifact")),
            "reason": str(status.get("reason", status.get("returncode", "result.json or predictions.npz is missing"))),
        }
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    recomputed, arrays = read_prediction_archive(prediction_path)
    if arrays["y_true"].shape[0] != expected_weeks:
        raise ValueError(f"{method} seed {seed}: partial archive has {arrays['y_true'].shape[0]} weeks, expected {expected_weeks}")
    reported = reported_metrics(payload, result_key)
    for metric in METRICS:
        if not np.isclose(recomputed[metric], reported[metric], atol=1e-8, rtol=0.0):
            raise ValueError(f"{method} seed {seed}: recomputed {metric} differs from result.json")
    if int(payload.get("split_seed", -1)) != seed:
        raise ValueError(f"{method} seed {seed}: split seed mismatch")
    if method not in {"persistence", "lag_ridge"}:
        if not bool(payload.get("delayed_observations")):
            raise ValueError(f"{method} seed {seed}: delayed-observation flag is absent")
        expected_delayed_rows = (expected_weeks - delay_weeks) * arrays["test_indices"].size
        if int(payload.get("delayed_observation_rows", -1)) != expected_delayed_rows:
            raise ValueError(f"{method} seed {seed}: hidden labels were not absorbed exactly once")
    return {
        "payload": payload,
        "arrays": arrays,
        "metrics": recomputed,
        "timing": timing_and_resources(payload),
        "path": path,
    }, None


def collect_runs(root: Path, protocol_root: Path, seeds: tuple[int, ...]) -> tuple[dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    records: dict[str, dict[int, dict[str, Any]]] = {method: {} for method, _, _, _ in METHODS}
    failures: list[dict[str, Any]] = []
    protocols: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        metadata_path = protocol_root / f"seed{seed}" / "protocol.json"
        archive_path = protocol_root / f"seed{seed}" / "protocol.npz"
        if not metadata_path.is_file() or not archive_path.is_file():
            raise FileNotFoundError(f"Missing protocol for seed {seed}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(archive_path) as archive:
            protocols[seed] = {
                "metadata": metadata,
                "stream_y": np.asarray(archive["stream_y"], dtype=np.float64),
                "test_indices": np.asarray(archive["test_indices"], dtype=np.int64),
                "task_ids": np.asarray(archive["reporting_task_id"], dtype=np.int64),
                "stream_dates": np.asarray(archive["stream_week_dates"]).astype(str),
            }
        expected_weeks = protocols[seed]["stream_y"].shape[0]
        delay_weeks = int(metadata["xlag"]["delay_weeks"])
        for method, _, relative, result_key in METHODS:
            record, failure = audit_run(
                root=root,
                seed=seed,
                method=method,
                relative=relative,
                result_key=result_key,
                expected_weeks=expected_weeks,
                delay_weeks=delay_weeks,
            )
            if failure is not None:
                failures.append(failure)
                continue
            assert record is not None
            arrays = record["arrays"]
            if not np.array_equal(arrays["test_indices"], protocols[seed]["test_indices"]):
                raise ValueError(f"{method} seed {seed}: held-out locations differ from protocol")
            if not np.allclose(arrays["y_true"], protocols[seed]["stream_y"][:, arrays["test_indices"]], atol=1e-12, rtol=0.0):
                raise ValueError(f"{method} seed {seed}: prediction targets differ from protocol")
            records[method][seed] = record
        available = [record for method in records.values() if seed in method for record in [method[seed]]]
        if available:
            reference = available[0]["arrays"]
            for record in available[1:]:
                arrays = record["arrays"]
                if not np.array_equal(arrays["test_indices"], reference["test_indices"]):
                    raise ValueError(f"seed {seed}: method archives use different held-out locations")
                if not np.allclose(arrays["y_true"], reference["y_true"], atol=1e-12, rtol=0.0):
                    raise ValueError(f"seed {seed}: method archives use different targets")
    return records, failures, protocols


def aggregate(records: dict[str, dict[int, dict[str, Any]]], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, label, _, _ in METHODS:
        available = [seed for seed in seeds if seed in records[method]]
        if not available:
            continue
        row: dict[str, Any] = {"method": method, "label": label, "n_splits": len(available), "seeds": ",".join(map(str, available))}
        for metric in METRICS:
            mean, sd = mean_sd([records[method][seed]["metrics"][metric] for seed in available])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        for key in ("update_seconds", "prediction_seconds", "compile_seconds", "persistent_state_mib", "peak_allocated_mib", "peak_reserved_mib", "cpu_rss_mib"):
            values = [float(records[method][seed]["timing"][key]) for seed in available]
            finite = [value for value in values if np.isfinite(value)]
            row[f"{key}_mean"] = float(np.mean(finite)) if finite else float("nan")
        row["device"] = ";".join(sorted({records[method][seed]["timing"]["device"] for seed in available}))
        rows.append(row)
    return rows


def per_seed_rows(records: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, _, _, _ in METHODS:
        for seed, record in sorted(records[method].items()):
            rows.append({"method": method, "seed": seed, **record["metrics"], **record["timing"], "artifact": str(record["path"])})
    return rows


def task_and_horizon_rows(
    records: dict[str, dict[int, dict[str, Any]]], protocols: dict[int, dict[str, Any]], seeds: tuple[int, ...]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for method, _, _, _ in METHODS:
        for seed in seeds:
            if seed not in records[method]:
                continue
            arrays = records[method][seed]["arrays"]
            protocol = protocols[seed]
            task_ids = protocol["task_ids"]
            terminal = int(task_ids.size)
            effective_horizons = [h for h in HORIZONS_REQUESTED if h <= terminal]
            if terminal not in effective_horizons:
                effective_horizons.append(terminal)
            for task in np.unique(task_ids):
                mask = task_ids == task
                task_rows.append({"method": method, "seed": seed, "task": int(task), "weeks": int(mask.sum()), **metrics_from_arrays(arrays["y_true"][mask], arrays["pred_mean"][mask], arrays["pred_var"][mask])})
            for horizon in effective_horizons:
                horizon_rows.append({"method": method, "seed": seed, "horizon_weeks": int(horizon), **metrics_from_arrays(arrays["y_true"][:horizon], arrays["pred_mean"][:horizon], arrays["pred_var"][:horizon])})
            names = protocol["metadata"]["location_names"]
            for position, location_index in enumerate(arrays["test_indices"]):
                state_rows.append({
                    "method": method,
                    "seed": seed,
                    "location_index": int(location_index),
                    "location_code": protocol["metadata"]["location_codes"][int(location_index)],
                    "location_name": names[int(location_index)],
                    **metrics_from_arrays(arrays["y_true"][:, position], arrays["pred_mean"][:, position], arrays["pred_var"][:, position]),
                })
    return task_rows, horizon_rows, state_rows


def paired_rows(
    source_rows: list[dict[str, Any]], *, group_key: str, metric: str = "rmse"
) -> list[dict[str, Any]]:
    table: dict[int, dict[str, dict[int, float]]] = {}
    for row in source_rows:
        group = int(row[group_key])
        table.setdefault(group, {}).setdefault(str(row["method"]), {})[int(row["seed"])] = float(row[metric])
    rows: list[dict[str, Any]] = []
    for group, methods in sorted(table.items()):
        ordinary = methods.get("routeb_ordinary", {})
        cumulative = methods.get("routeb_cumulative", {})
        common = sorted(set(ordinary) & set(cumulative))
        if not common:
            continue
        differences = [ordinary[seed] - cumulative[seed] for seed in common]
        mean, low, high = paired_bootstrap(differences)
        rows.append({group_key: group, "metric": metric, "n_paired_splits": len(common), "seeds": ",".join(map(str, common)), "ordinary_minus_hippo_mean": mean, "bootstrap95_low": low, "bootstrap95_high": high})
    return rows


def plot_main_metrics(summary: list[dict[str, Any]], output: Path, title_suffix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), sharey=True)
    y = np.arange(len(summary))[::-1]
    labels = [str(row["label"]) for row in summary]
    for axis, metric, title, reference in zip(axes, METRICS, ("RMSE", "NLL", "Coverage90"), (None, None, 0.90)):
        for ypos, row in zip(y, summary):
            axis.errorbar(float(row[f"{metric}_mean"]), ypos, xerr=float(row[f"{metric}_sd"]), fmt="o", color=COLORS[str(row["method"])], capsize=3, markersize=6)
        if reference is not None:
            axis.axvline(reference, color="#333333", lw=1, ls="--")
        axis.set_title(title)
        axis.grid(axis="x", color="#D9D9D9", lw=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.set_xlabel("mean +/- sample SD")
    axes[0].set_yticks(y, labels)
    fig.suptitle(f"CDC COVID long-stream strict-online nowcasting ({title_suffix})", y=1.03, fontsize=12)
    save_figure(fig, output / "fig_long_main_metrics", {"figure": "main_metrics", "seeds": title_suffix, "metrics": list(METRICS), "colors": COLORS})


def curve_by_method(records: dict[str, dict[int, dict[str, Any]]], method: str, seeds: tuple[int, ...], metric: str) -> tuple[np.ndarray, np.ndarray] | None:
    values = []
    for seed in seeds:
        if seed not in records[method]:
            continue
        arrays = records[method][seed]["arrays"]
        values.append([metrics_from_arrays(arrays["y_true"][week : week + 1], arrays["pred_mean"][week : week + 1], arrays["pred_var"][week : week + 1])[metric] for week in range(arrays["y_true"].shape[0])])
    if not values:
        return None
    matrix = np.asarray(values, dtype=np.float64)
    return matrix.mean(axis=0), matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(matrix.shape[1])


def plot_online_curves(records: dict[str, dict[int, dict[str, Any]]], seeds: tuple[int, ...], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), sharex=True)
    for axis, metric, title, reference in zip(axes, METRICS, ("RMSE", "NLL", "Coverage90"), (None, None, 0.90)):
        for method, label, _, _ in METHODS:
            curve = curve_by_method(records, method, seeds, metric)
            if curve is None:
                continue
            mean, sd = curve
            weeks = np.arange(1, mean.size + 1)
            axis.plot(weeks, mean, lw=1.45, color=COLORS[method], label=label)
            axis.fill_between(weeks, mean - sd, mean + sd, color=COLORS[method], alpha=0.11, linewidth=0)
        if reference is not None:
            axis.axhline(reference, color="#333333", lw=1, ls="--")
        axis.set_title(title)
        axis.set_xlabel("Online week")
        axis.grid(color="#D9D9D9", lw=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean across spatial splits")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.22), fontsize=7.5)
    fig.suptitle("Weekly strict-online accuracy and calibration", y=1.03, fontsize=12)
    save_figure(fig, output / "fig_long_online_curves", {"figure": "online_curves", "aggregation": "mean +/- sample SD across available paired spatial splits", "metrics": list(METRICS), "colors": COLORS})


def plot_taskwise(task_rows: list[dict[str, Any]], output: Path) -> None:
    selected = ("persistence", "lag_ridge", "ohsvgp_rbf", "routeb_ordinary", "routeb_cumulative")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5), sharex=True)
    for axis, metric, title, reference in zip(axes, METRICS, ("RMSE", "NLL", "Coverage90"), (None, None, 0.90)):
        for method in selected:
            rows = [row for row in task_rows if row["method"] == method]
            if not rows:
                continue
            tasks = sorted({int(row["task"]) for row in rows})
            means = [np.mean([float(row[metric]) for row in rows if int(row["task"]) == task]) for task in tasks]
            axis.plot(tasks, means, marker="o", ms=3.5, lw=1.5, color=COLORS[method], label=next(label for key, label, _, _ in METHODS if key == method))
        if reference is not None:
            axis.axhline(reference, color="#333333", lw=1, ls="--")
        axis.set_title(title)
        axis.set_xlabel("Reporting task")
        axis.set_xticks(np.arange(2, 11))
        axis.grid(color="#D9D9D9", lw=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean across available splits")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.21), fontsize=8)
    fig.suptitle("Task 2-10 reporting aggregates do not reset the online state", y=1.03, fontsize=12)
    save_figure(fig, output / "fig_long_taskwise_metrics", {"figure": "taskwise_metrics", "tasks": list(range(2, 11)), "metrics": list(METRICS), "colors": COLORS})


def plot_memory_gap(horizon_rows: list[dict[str, Any]], paired: list[dict[str, Any]], output: Path) -> None:
    methods = ("routeb_ordinary", "routeb_cumulative")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5))
    for method in methods:
        rows = [row for row in horizon_rows if row["method"] == method]
        horizons = sorted({int(row["horizon_weeks"]) for row in rows})
        means = [np.mean([float(row["rmse"]) for row in rows if int(row["horizon_weeks"]) == horizon]) for horizon in horizons]
        axes[0].plot(horizons, means, marker="o", ms=3.5, lw=1.7, color=COLORS[method], label=next(label for key, label, _, _ in METHODS if key == method))
    axes[0].set_xlabel("Cumulative online horizon (weeks)")
    axes[0].set_ylabel("Cumulative RMSE")
    axes[0].grid(color="#D9D9D9", lw=0.7)
    axes[0].set_axisbelow(True)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    horizons = [int(row["horizon_weeks"]) for row in paired]
    delta = np.asarray([float(row["ordinary_minus_hippo_mean"]) for row in paired])
    low = np.asarray([float(row["bootstrap95_low"]) for row in paired])
    high = np.asarray([float(row["bootstrap95_high"]) for row in paired])
    axes[1].axhline(0.0, color="#333333", lw=1, ls="--")
    axes[1].plot(horizons, delta, marker="o", ms=3.5, lw=1.7, color=COLORS["routeb_cumulative"])
    axes[1].fill_between(horizons, low, high, color=COLORS["routeb_cumulative"], alpha=0.18)
    axes[1].set_xlabel("Cumulative online horizon (weeks)")
    axes[1].set_ylabel("Ordinary RMSE - HiPPO RMSE")
    axes[1].grid(color="#D9D9D9", lw=0.7)
    axes[1].set_axisbelow(True)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Long-horizon paired memory comparison; positive gap favors HiPPO", y=1.03, fontsize=12)
    save_figure(fig, output / "fig_long_memory_gap", {"figure": "memory_gap", "definition": "ordinary cumulative RMSE minus cumulative HiPPO RMSE", "horizons": horizons, "confidence_interval": "paired nonparametric bootstrap over spatial split seeds"})


def plot_task_gap(task_paired: list[dict[str, Any]], output: Path) -> None:
    tasks = [int(row["task"]) for row in task_paired]
    means = np.asarray([float(row["ordinary_minus_hippo_mean"]) for row in task_paired])
    low = np.asarray([float(row["bootstrap95_low"]) for row in task_paired])
    high = np.asarray([float(row["bootstrap95_high"]) for row in task_paired])
    fig, axis = plt.subplots(figsize=(8.2, 3.5))
    axis.axhline(0.0, color="#333333", lw=1, ls="--")
    axis.errorbar(tasks, means, yerr=np.vstack([means - low, high - means]), fmt="o-", color=COLORS["routeb_cumulative"], capsize=3)
    axis.set_xticks(tasks)
    axis.set_xlabel("Reporting task")
    axis.set_ylabel("Ordinary RMSE - HiPPO RMSE")
    axis.grid(color="#D9D9D9", lw=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("Paired task-wise comparison; positive values favor HiPPO")
    save_figure(fig, output / "fig_long_task_gap", {"figure": "task_gap", "definition": "ordinary task RMSE minus HiPPO task RMSE", "tasks": tasks, "confidence_interval": "paired nonparametric bootstrap over spatial split seeds"})


def plot_state_heatmap(state_rows: list[dict[str, Any]], output: Path) -> None:
    selected = ("routeb_ordinary", "routeb_cumulative")
    states = sorted({str(row["location_name"]) for row in state_rows})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 12.0), sharey=True)
    for axis, metric, cmap in ((axes[0], "rmse", "viridis"), (axes[1], "coverage90", "cividis")):
        matrix = np.asarray(
            [
                [
                    np.mean([float(row[metric]) for row in state_rows if row["method"] == method and str(row["location_name"]) == state])
                    if any(row["method"] == method and str(row["location_name"]) == state for row in state_rows)
                    else np.nan
                    for method in selected
                ]
                for state in states
            ],
            dtype=np.float64,
        )
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap)
        axis.set_title("RMSE" if metric == "rmse" else "Coverage90")
        axis.set_xticks([0, 1], ["Ordinary", "Cumulative HiPPO"], rotation=25, ha="right")
        fig.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    axes[0].set_yticks(np.arange(len(states)), states, fontsize=6.5)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.22, wspace=0.34)
    fig.suptitle("Per-state held-out performance aggregated where a state is scored", y=0.995, fontsize=12)
    save_figure(fig, output / "fig_long_state_heatmap", {"figure": "state_heatmap", "methods": list(selected), "metrics": ["rmse", "coverage90"], "state_aggregation": "mean over formal seeds in which the state is held out; blank cells indicate no scored split"})


def plot_trajectory(records: dict[str, dict[int, dict[str, Any]]], protocols: dict[int, dict[str, Any]], seeds: tuple[int, ...], output: Path) -> dict[str, Any]:
    panels: list[tuple[str, int, int]] = []
    for state in TRAJECTORY_STATES:
        for seed in seeds:
            if seed not in records["routeb_cumulative"]:
                continue
            metadata = protocols[seed]["metadata"]
            indices = records["routeb_cumulative"][seed]["arrays"]["test_indices"]
            names = metadata["location_names"]
            matches = np.flatnonzero(np.asarray([names[int(index)] == state for index in indices]))
            if matches.size:
                panels.append((state, seed, int(matches[0])))
                break
    if not panels:
        raise ValueError("None of the requested diagnostic states is held out in the available seeds")
    fig, axes = plt.subplots(len(panels), 1, figsize=(11.2, 2.45 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    selection = []
    for axis, (state, seed, position) in zip(axes, panels):
        reference = records["routeb_cumulative"][seed]["arrays"]
        standardization = protocols[seed]["metadata"]["target_standardization"]
        scale = float(standardization["scale"])
        mean = float(standardization["mean"])
        dates = protocols[seed]["stream_dates"]
        x = np.arange(reference["y_true"].shape[0])
        restore = lambda value: value * scale + mean
        axis.plot(x, restore(reference["y_true"][:, position]), color="#202020", lw=1.8, label="Observed")
        for method in ("persistence", "routeb_ordinary", "routeb_cumulative"):
            if seed not in records[method]:
                continue
            arrays = records[method][seed]["arrays"]
            axis.plot(x, restore(arrays["pred_mean"][:, position]), color=COLORS[method], lw=1.25, label=next(label for key, label, _, _ in METHODS if key == method))
        std = np.sqrt(reference["pred_var"][:, position])
        axis.fill_between(x, restore(reference["pred_mean"][:, position] - 1.6448536269514722 * std), restore(reference["pred_mean"][:, position] + 1.6448536269514722 * std), color=COLORS["routeb_cumulative"], alpha=0.16, label="HiPPO 90% interval")
        axis.set_ylabel("log1p / 100k")
        axis.set_title(f"{state} (seed {seed}, held out)", loc="left", fontsize=10)
        axis.grid(color="#D9D9D9", lw=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        ticks = np.linspace(0, len(x) - 1, num=min(6, len(x)), dtype=int)
        axis.set_xticks(ticks, [dates[tick] for tick in ticks], rotation=25, ha="right")
        selection.append({"state": state, "seed": seed, "heldout_position": position})
    axes[0].legend(loc="upper right", ncol=3, fontsize=7.5, frameon=False)
    fig.suptitle("Held-out state trajectories under the delayed-observation protocol", y=1.0, fontsize=12)
    save_figure(fig, output / "fig_long_trajectory", {"figure": "trajectory", "states_requested": list(TRAJECTORY_STATES), "panels": selection, "interval": "Route B cumulative HiPPO nominal 90% interval"})
    return {"panels": selection}


def load_efficiency_audit(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise ValueError(f"Efficiency audit is not complete: {path}")
    if int(audit.get("warmups", -1)) != 10 or int(audit.get("steady_state_repeats", -1)) < 30:
        raise ValueError(f"Efficiency audit does not meet the formal timing protocol: {path}")
    required = {"routeb_ordinary", "routeb_cumulative"}
    if not required.issubset(audit.get("methods", {})):
        raise ValueError(f"Efficiency audit is missing Route B methods: {path}")
    audit["path"] = str(path)
    return audit


def plot_efficiency(summary: list[dict[str, Any]], output: Path, audit: dict[str, Any] | None) -> None:
    if audit is None:
        learned = [row for row in summary if row["method"] not in {"persistence", "lag_ridge"}]
        latency = [1000.0 * (float(row["update_seconds_mean"]) + float(row["prediction_seconds_mean"])) for row in learned]
        state = [float(row["persistent_state_mib_mean"]) for row in learned]
        title = "Reported runner timing (not the isolated efficiency audit)"
        latency_scope = "reported update plus prediction from the benchmark runner; not profiler time"
        use_log_scale = True
    else:
        methods = audit["methods"]
        learned = [
            {
                "method": method,
                "label": next(label for key, label, _, _ in METHODS if key == method),
            }
            for method in ("routeb_ordinary", "routeb_cumulative")
        ]
        latency = [1000.0 * float(methods[row["method"]]["steady_update_prediction_seconds_mean"]) for row in learned]
        state = [float(methods[row["method"]]["persistent_state_mib"]) for row in learned]
        title = "Isolated GPU efficiency audit (Route B)"
        latency_scope = "10 warm-ups plus at least 30 synchronized steady-state repeats; profiler excluded"
        use_log_scale = False
    if not learned:
        return
    labels = [str(row["label"]).replace(" ", "\n") for row in learned]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
    x = np.arange(len(learned))
    colors = [COLORS[str(row["method"])] for row in learned]
    axes[0].bar(x, latency, color=colors, edgecolor="#333333", linewidth=0.4)
    if use_log_scale:
        axes[0].set_yscale("log")
    else:
        axes[0].set_ylim(bottom=0.0)
    axes[0].set_xticks(x, labels, fontsize=7)
    axes[0].set_ylabel("update + prediction (ms/week" + (", log)" if use_log_scale else ")"))
    axes[0].set_title("Measured runtime scope")
    axes[1].bar(x, np.nan_to_num(state, nan=0.0), color=colors, edgecolor="#333333", linewidth=0.4)
    axes[1].set_xticks(x, labels, fontsize=7)
    axes[1].set_ylabel("persistent state (MiB)")
    axes[1].set_title("Reported persistent state")
    for index, value in enumerate(state):
        if not np.isfinite(value):
            axes[1].text(index, 0.0, "not\ninstrumented", ha="center", va="bottom", fontsize=7, rotation=90)
    for axis in axes:
        axis.grid(axis="y", color="#D9D9D9", lw=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, y=1.03, fontsize=12)
    save_figure(fig, output / "fig_long_efficiency", {"figure": "efficiency", "latency": latency_scope, "audit_path": None if audit is None else audit["path"], "bui_scope": "CPU-only, excluded from GPU runtime ranking", "colors": COLORS})


def delay_stress_rows(results_root: Path, delay_root: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_target: np.ndarray | None = None
    for delay, root in ((1, results_root), (2, delay_root / "d2"), (4, delay_root / "d4"), (8, delay_root / "d8")):
        for method, label, relative, result_key in METHODS:
            if method not in {"routeb_ordinary", "routeb_cumulative"}:
                continue
            run = run_path(root, seed, relative)
            result_path = run / "result.json"
            prediction_path = run / "predictions.npz"
            if not result_path.is_file() or not prediction_path.is_file():
                return []
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if int(payload.get("delayed_observation_blocks", -1)) != delay:
                raise ValueError(f"delay stress {method} d={delay}: runner delay does not match the protocol")
            metrics, arrays = read_prediction_archive(prediction_path)
            reported = reported_metrics(payload, result_key)
            for metric in METRICS:
                if not np.isclose(metrics[metric], reported[metric], atol=1e-8, rtol=0.0):
                    raise ValueError(f"delay stress {method} d={delay}: archive metrics differ from result.json")
            if reference_target is None:
                reference_target = arrays["y_true"]
            elif not np.allclose(arrays["y_true"], reference_target, atol=1e-12, rtol=0.0):
                raise ValueError("delay stress runs must score the identical target stream")
            rows.append({"delay_weeks": delay, "seed": seed, "method": method, "label": label, **metrics})
    return rows


def plot_delay_stress(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharex=True)
    for axis, metric, title, reference in zip(axes, METRICS, ("RMSE", "NLL", "Coverage90"), (None, None, 0.90)):
        for method in ("routeb_ordinary", "routeb_cumulative"):
            selected = sorted((row for row in rows if row["method"] == method), key=lambda row: int(row["delay_weeks"]))
            axis.plot([row["delay_weeks"] for row in selected], [row[metric] for row in selected], marker="o", lw=1.7, color=COLORS[method], label=next(label for key, label, _, _ in METHODS if key == method))
        if reference is not None:
            axis.axhline(reference, color="#333333", lw=1, ls="--")
        axis.set_title(title)
        axis.set_xlabel("Hidden-label reporting delay (weeks)")
        axis.set_xticks([1, 2, 4, 8])
        axis.grid(color="#D9D9D9", lw=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.19), fontsize=8)
    fig.suptitle("Controlled delayed-observation memory stress test (seed 0)", y=1.03, fontsize=12)
    save_figure(fig, output / "fig_long_delay_stress", {"figure": "delay_stress", "delays": [1, 2, 4, 8], "seed": 0, "availability": "Each protocol recomputes lag4 features using only labels whose reporting delay has elapsed.", "colors": COLORS})


def write_latex(summary: list[dict[str, Any]], path: Path) -> None:
    lines = ["\\begin{tabular}{lcccc}", "\\toprule", "Method & Splits & RMSE $\\downarrow$ & NLL $\\downarrow$ & Coverage90 \\\\", "\\midrule"]
    for row in summary:
        label = str(row["label"])
        if row["method"] == "routeb_cumulative":
            label = "\\textbf{" + label + "}"
        lines.append(f"{label} & {row['n_splits']} & {row['rmse_mean']:.4f} $\\pm$ {row['rmse_sd']:.4f} & {row['nll_mean']:.4f} $\\pm$ {row['nll_sd']:.4f} & {row['coverage90_mean']:.4f} $\\pm$ {row['coverage90_sd']:.4f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "% Mean +/- sample SD over the listed held-out spatial split seeds.", "% Adaptive Bui is CPU-only and is excluded from GPU runtime ranking."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/diagnostics/covid_long_stream_2020_2024_mandatory"))
    parser.add_argument("--protocol-root", type=Path, default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    parser.add_argument("--output", type=Path, default=Path("/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/kronecker+s2vgp/ICLR Formal experiment/covid_delayed_history_baseline_paper_materials/long_stream_2020_2024_mandatory"))
    parser.add_argument("--delay-results-root", type=Path)
    parser.add_argument("--delay-seed", type=int, default=0)
    parser.add_argument("--efficiency-audit", type=Path)
    parser.add_argument("--allow-incomplete-methods", action="store_true")
    args = parser.parse_args()

    results_root = (ROOT / args.results_root).resolve()
    protocol_root = (ROOT / args.protocol_root).resolve()
    output = args.output.resolve()
    seeds = tuple(args.seeds)
    output.mkdir(parents=True, exist_ok=True)
    efficiency_audit = load_efficiency_audit(
        None if args.efficiency_audit is None else (ROOT / args.efficiency_audit).resolve()
    )
    records, failures, protocols = collect_runs(results_root, protocol_root, seeds)
    required = ("routeb_ordinary", "routeb_cumulative")
    missing_core = [method for method in required if set(records[method]) != set(seeds)]
    if missing_core:
        raise ValueError(f"Cannot produce a paired long-memory figure without complete ordinary/HiPPO runs: {missing_core}")
    if failures and not args.allow_incomplete_methods:
        failed = "; ".join(f"{row['method']} seed {row['seed']}: {row['status']}" for row in failures)
        raise ValueError(f"Incomplete method set. Re-run with --allow-incomplete-methods to export an explicit failure table: {failed}")

    summary = aggregate(records, seeds)
    task_rows, horizon_rows, state_rows = task_and_horizon_rows(records, protocols, seeds)
    paired_horizon = paired_rows(horizon_rows, group_key="horizon_weeks")
    paired_task = paired_rows(task_rows, group_key="task")
    if not paired_horizon or not paired_task:
        raise ValueError("Paired ordinary/HiPPO task and horizon statistics are required")
    first_horizon = min(paired_horizon, key=lambda row: int(row["horizon_weeks"]))
    terminal_horizon_row = max(paired_horizon, key=lambda row: int(row["horizon_weeks"]))
    mechanism_result = (
        "expanded"
        if float(terminal_horizon_row["ordinary_minus_hippo_mean"])
        > float(first_horizon["ordinary_minus_hippo_mean"])
        else "not_expanded"
    )
    terminal_ci_crosses_zero = (
        float(terminal_horizon_row["bootstrap95_low"])
        <= 0.0
        <= float(terminal_horizon_row["bootstrap95_high"])
    )
    write_csv(output / "metrics_per_seed.csv", per_seed_rows(records))
    write_csv(output / "metrics_aggregate.csv", summary)
    write_csv(output / "metrics_taskwise.csv", task_rows)
    write_csv(output / "metrics_horizon.csv", horizon_rows)
    write_csv(output / "metrics_per_state.csv", state_rows)
    write_csv(output / "paired_horizon_contrasts.csv", paired_horizon)
    write_csv(output / "paired_task_contrasts.csv", paired_task)
    write_csv(output / "failure_table.csv", failures)
    write_latex(summary, output / "table_covid_long_stream.tex")

    title_suffix = "formal seeds 5-9" if seeds == FORMAL_SEEDS else "development seeds " + ",".join(map(str, seeds))
    plot_main_metrics(summary, output, title_suffix)
    plot_online_curves(records, seeds, output)
    plot_taskwise(task_rows, output)
    plot_memory_gap(horizon_rows, paired_horizon, output)
    plot_task_gap(paired_task, output)
    plot_state_heatmap(state_rows, output)
    trajectory = plot_trajectory(records, protocols, seeds, output)
    plot_efficiency(summary, output, efficiency_audit)
    delay_rows = []
    if args.delay_results_root is not None:
        delay_rows = delay_stress_rows(results_root, (ROOT / args.delay_results_root).resolve(), args.delay_seed)
        if delay_rows:
            write_csv(output / "delay_stress_metrics.csv", delay_rows)
            plot_delay_stress(delay_rows, output)

    terminal_horizon = int(protocols[seeds[0]]["stream_y"].shape[0])
    report = [
        "# COVID Long-Stream Mandatory-Period Benchmark",
        "",
        "## Protocol",
        "",
        f"This report uses {title_suffix}. CDC NHSN weekly hospital-respiratory data are restricted to the mandatory-reporting window 2020-08-01 through 2024-04-30. The audited extract contains 195 complete weekly endpoints (2020-08-08 to 2024-04-27): 52 Task-1 calibration weeks and {terminal_horizon} one-week strict-online updates. Task 2-9 contain 16 reporting weeks each; Task 10 contains 15. Reporting tasks never reset the posterior, inducing state, HiPPO basis, or frozen Task-1 parameters.",
        "",
        "At each online week, labels hidden in the prior week are absorbed once, then current visible labels update the posterior, and current hidden labels are predicted. Route B uses Mt=32, Ms=32, float64, the geographic kernel, availability-aware lag4 mean, Task-1 posterior initialization, and full-joint-conditional predictive variance.",
        "",
        "## Result Scope",
        "",
        "All numeric metrics were recomputed from `predictions.npz` and checked against each runner's JSON summary. Positive ordinary-minus-HiPPO paired gaps favor cumulative HiPPO. The requested 144-week horizon is unavailable because the audited stream has 143, not 144, online weeks; the terminal 143-week value is reported instead.",
        "",
        "The Route B efficiency figure uses an isolated timing audit with 10 warm-ups and at least 30 synchronized steady-state repeats." if efficiency_audit is not None else "No isolated Route B efficiency audit was supplied; the efficiency figure is explicitly limited to runner timing.",
        "",
        "## Main Table",
        "",
        "| Method | Splits | RMSE | NLL | Coverage90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(f"| {row['label']} | {row['n_splits']} | {row['rmse_mean']:.4f} +/- {row['rmse_sd']:.4f} | {row['nll_mean']:.4f} +/- {row['nll_sd']:.4f} | {row['coverage90_mean']:.4f} +/- {row['coverage90_sd']:.4f} |")
    report.extend([
        "",
        "## Long-Horizon Finding",
        "",
        (
            f"The paired ordinary-minus-HiPPO RMSE gap increases from {float(first_horizon['ordinary_minus_hippo_mean']):+.4f} at h={int(first_horizon['horizon_weeks'])} to {float(terminal_horizon_row['ordinary_minus_hippo_mean']):+.4f} at h={int(terminal_horizon_row['horizon_weeks'])}; this is evidence that the advantage expands with horizon."
            if mechanism_result == "expanded"
            else f"The paired ordinary-minus-HiPPO RMSE gap falls from {float(first_horizon['ordinary_minus_hippo_mean']):+.4f} at h={int(first_horizon['horizon_weeks'])} to {float(terminal_horizon_row['ordinary_minus_hippo_mean']):+.4f} at h={int(terminal_horizon_row['horizon_weeks'])}, with terminal paired-bootstrap 95% CI [{float(terminal_horizon_row['bootstrap95_low']):+.4f}, {float(terminal_horizon_row['bootstrap95_high']):+.4f}]. This does not support an advantage that expands with horizon and is recorded as a negative mechanism result."
        ),
        "",
        "## Boundary",
        "",
        "The adaptive Bui row is a CPU-only official-implementation result and is excluded from GPU runtime ranking. Missing/failed methods are listed in `failure_table.csv`, never replaced by another method. The delayed-observation stress test and daily appendix are separate secondary results and must not be mixed into this weekly long-memory table.",
        "",
        f"Trajectory panels: {json.dumps(trajectory['panels'])}.",
    ])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    audit = {
        "status": "complete",
        "results_root": str(results_root),
        "protocol_root": str(protocol_root),
        "output": str(output),
        "seeds": list(seeds),
        "formal_seed_set": seeds == FORMAL_SEEDS,
        "prediction_metrics_recomputed": True,
        "source_weekly_observations": int(protocols[seeds[0]]["metadata"]["num_total_times"]),
        "online_weeks": terminal_horizon,
        "effective_horizons": sorted({int(row["horizon_weeks"]) for row in horizon_rows}),
        "long_horizon_mechanism": {
            "result": mechanism_result,
            "first_horizon": first_horizon,
            "terminal_horizon": terminal_horizon_row,
            "terminal_ci_crosses_zero": terminal_ci_crosses_zero,
        },
        "failures": failures,
        "trajectory": trajectory,
        "delay_stress_generated": bool(delay_rows),
        "efficiency_audit": None if efficiency_audit is None else {
            "path": efficiency_audit["path"],
            "warmups": efficiency_audit["warmups"],
            "steady_state_repeats": efficiency_audit["steady_state_repeats"],
            "nsight_compute": efficiency_audit["nsight_compute"],
            "gpu": efficiency_audit.get("gpu"),
        },
    }
    (output / "artifact_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
