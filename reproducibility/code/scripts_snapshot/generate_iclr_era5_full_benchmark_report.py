#!/usr/bin/env python3
"""Aggregate the complete ERA5 benchmark and write an ICLR-style report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = REPO / (
    "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/"
    "ICLR Formal experiment/iclr_era5_full_benchmark"
)
OFFICIAL_SHORT = REPO / (
    "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/"
    "ICLR Formal experiment/post_meeting_priority_2026-07-14/"
    "p0b_official_full_era5"
)
OFFICIAL_XLAG = REPO / (
    "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/"
    "ICLR Formal experiment/unified_stvgp_routeb_comparison/"
    "phase_d_joint_xlag_controlled"
)
FLOP_ROOT = REPO / (
    "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/"
    "ICLR Formal experiment/unified_stvgp_routeb_comparison/"
    "phase_m_routeb_empirical_bayes"
)


COLORS = {
    "routeb_analytic_hippo_rff": "#C23B3B",
    "routeb_inducing_points": "#1676B8",
    "bui_osgpr_prior_m128": "#6F4C9B",
    "maddox_streaming_sgpr_m128": "#E07B39",
    "official_ohsvgp_m32_rff128": "#2B8C6B",
    "xlag_mean_recursive_rls": "#696969",
    "gpflow_svgp_direct_m512": "#7A5195",
    "gpflow_svgp_shared_residual_m512": "#7A5195",
}

ONLINE_LABELS = {
    "routeb_analytic_hippo_rff": "Route B cumulative HiPPO",
    "routeb_inducing_points": "Route B global inducing",
    "bui_osgpr_prior_m128": "Official Bui OSGPR",
    "maddox_streaming_sgpr_m128": "Official Maddox StreamingSGPR",
    "official_ohsvgp_m32_rff128": "Official OHSVGP",
    "xlag_mean_task1_fixed": "Task-1 frozen X-lag mean",
    "xlag_mean_recursive_rls": "Recursive X-lag RLS",
    "xlag_mean_batch_fixed": "Batch X-lag mean oracle",
}

BATCH_LABELS = {
    "gpflow_svgp_direct_m512": "Official GPflow SVGP",
    "gpflow_svgp_shared_residual_m512": "Official GPflow SVGP residual",
    "routeb_direct_analytic_hippo_rff": "Route B HiPPO direct",
    "routeb_direct_inducing_points": "Route B inducing direct",
    "routeb_shared_residual_analytic_hippo_rff": "Route B HiPPO residual",
    "routeb_shared_residual_inducing_points": "Route B inducing residual",
    "routeb_joint_xlag_analytic_hippo_rff": "Structured-joint Route B HiPPO",
    "routeb_joint_xlag_inducing_points": "Structured-joint Route B inducing",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(payload: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        value: Any = payload
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return None


def sample_sd(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(array.std(ddof=1)) if array.size > 1 else 0.0


def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "rmse",
        "nll",
        "coverage90",
        "final_rmse",
        "runtime_seconds",
        "prediction_seconds",
        "peak_rss_mib",
        "persistent_state_mib",
    ]
    for key, group in frame.groupby(keys, dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key_tuple))
        row["seeds"] = int(group["seed"].nunique()) if "seed" in group else 0
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            row[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
            row[f"{metric}_sd"] = sample_sd(values) if values.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def collect_runs(benchmark: Path, scope: str, branch: str, methods: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in methods:
        for seed in range(3):
            path = benchmark / "runs" / scope / branch / method / f"seed{seed}" / "result.json"
            if not path.is_file():
                continue
            payload = read_json(path)
            final_rmse = nested(payload, "final_block.rmse", "final.rmse", "rmse")
            if branch == "online":
                blocks_path = path.parent / "blocks.csv"
                if blocks_path.is_file():
                    with blocks_path.open(newline="", encoding="utf-8") as handle:
                        block_rows = list(csv.DictReader(handle))
                    if block_rows and block_rows[-1].get("rmse") not in (None, ""):
                        final_rmse = float(block_rows[-1]["rmse"])
            rows.append(
                {
                    "scope": scope,
                    "branch": branch,
                    "method": method,
                    "label": (BATCH_LABELS if branch == "batch" else ONLINE_LABELS).get(method, method),
                    "seed": seed,
                    "rmse": nested(payload, "overall_current_block.rmse", "final.rmse", "rmse"),
                    "nll": nested(payload, "overall_current_block.nll", "final.nll", "nll"),
                    "coverage90": nested(
                        payload, "overall_current_block.coverage90", "final.coverage90", "coverage90"
                    ),
                    "final_rmse": final_rmse,
                    "runtime_seconds": nested(
                        payload,
                        "timing.process_total_seconds",
                        "timing.end_to_end_training_seconds",
                        "train_seconds",
                    ),
                    "prediction_seconds": nested(
                        payload,
                        "timing.stream_prediction_seconds",
                        "timing.prediction_seconds",
                        "final.prediction_seconds",
                    ),
                    "peak_rss_mib": nested(payload, "resources.peak_rss_mib"),
                    "persistent_state_mib": nested(
                        payload,
                        "resources.persistent_state_mib",
                        "resources.persistent_model_state_mib",
                    ),
                    "history_replay_buffer_bytes": nested(payload, "resources.history_replay_buffer_bytes"),
                    "estimated_flops": nested(
                        payload,
                        "resources.estimated_streaming_flops",
                        "resources.estimated_training_flops",
                    ),
                    "result_path": str(path.relative_to(REPO)),
                }
            )
    return pd.DataFrame(rows)


def parse_peak_rss(path: Path) -> float:
    if not path.is_file():
        return float("nan")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Maximum resident set size (kbytes):" in line:
            return float(line.rsplit(":", 1)[1].strip()) / 1024.0
    return float("nan")


def official_short_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in range(3):
        for directory, label, ms in [
            ("st_svgp_Ms30", "Official ST-SVGP", 30),
            ("st_svgp_Ms64", "Official ST-SVGP", 64),
            ("st_svgp_Ms128", "Official ST-SVGP", 128),
            ("mf_st_svgp_Ms30", "Official MF-ST-SVGP", 30),
        ]:
            path = OFFICIAL_SHORT / f"seed{seed}" / directory / "result.json"
            payload = read_json(path)
            rows.append(
                {
                    "scope": "task1_2",
                    "method": directory,
                    "label": label,
                    "seed": seed,
                    "temporal": "Full Markov",
                    "spatial": f"M_s={ms}",
                    "rmse": payload["rmse"],
                    "nll": payload["nll"],
                    "coverage90": payload["coverage90"],
                    "final_rmse": payload["rmse"],
                    "runtime_seconds": payload["train_seconds"],
                    "prediction_seconds": float("nan"),
                    "peak_rss_mib": parse_peak_rss(path.parent / "resource_usage.txt"),
                    "persistent_state_mib": float("nan"),
                    "status": "complete",
                }
            )
    return pd.DataFrame(rows)


def batch_table_data(benchmark: Path, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    batch_methods = list(BATCH_LABELS)
    frame = collect_runs(benchmark, scope, "batch", batch_methods)
    direct = frame[frame.method.isin(
        ["gpflow_svgp_direct_m512", "routeb_direct_analytic_hippo_rff", "routeb_direct_inducing_points"]
    )].copy()
    xlag = frame[frame.method.isin(
        [
            "gpflow_svgp_shared_residual_m512",
            "routeb_shared_residual_analytic_hippo_rff",
            "routeb_shared_residual_inducing_points",
            "routeb_joint_xlag_analytic_hippo_rff",
            "routeb_joint_xlag_inducing_points",
        ]
    )].copy()
    mean = collect_runs(benchmark, scope, "online", ["xlag_mean_batch_fixed"])
    if not mean.empty:
        mean["label"] = "Shared X-lag ridge mean only"
        mean["method"] = "xlag_mean_batch_fixed"
        xlag = pd.concat([mean, xlag], ignore_index=True, sort=False)
    return direct, xlag


def online_table_data(benchmark: Path, scope: str) -> pd.DataFrame:
    methods = [
        "xlag_mean_task1_fixed",
        "xlag_mean_recursive_rls",
        "bui_osgpr_prior_m128",
        "maddox_streaming_sgpr_m128",
        "official_ohsvgp_m32_rff128",
        "routeb_inducing_points",
        "routeb_analytic_hippo_rff",
    ]
    return collect_runs(benchmark, scope, "online", methods)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def direct_status_tables(
    benchmark: Path, short_direct: pd.DataFrame, long_direct: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    short_official = official_short_rows()
    short = pd.concat([short_official, short_direct], ignore_index=True, sort=False)
    short["temporal"] = short["temporal"].fillna(
        short.method.map(
            {
                "gpflow_svgp_direct_m512": "M_t=8 inducing",
                "routeb_direct_analytic_hippo_rff": "M_t=128 HiPPO",
                "routeb_direct_inducing_points": "M_t=128 inducing",
            }
        )
    )
    short["spatial"] = short["spatial"].fillna(
        short.method.map(
            {
                "gpflow_svgp_direct_m512": "M_s=64",
                "routeb_direct_analytic_hippo_rff": "M_s=128",
                "routeb_direct_inducing_points": "M_s=128",
            }
        )
    )
    short["status"] = short.get("status", pd.Series(index=short.index, dtype=object)).fillna("complete")
    short_status = pd.DataFrame(
        [
            {
                "scope": "task1_2",
                "method": "st_vgp_full",
                "label": "Official ST-VGP",
                "temporal": "Full Markov",
                "spatial": "Full 800-state",
                "status": "resource exhausted (9.7 GiB allocation)",
            },
            {
                "scope": "task1_2",
                "method": "markovflow_sparse_variational_mt32_ms128",
                "label": "Official Markovflow sparse variational",
                "temporal": "M_t=32",
                "spatial": "M_s=128",
                "status": "banded-Cholesky tolerance failure",
            },
            {
                "scope": "task1_2",
                "method": "markovflow_sparse_cvi_mt32_ms128",
                "label": "Official Markovflow sparse CVI",
                "temporal": "M_t=32",
                "spatial": "M_s=128",
                "status": "no first iteration within 900 s",
            },
        ]
    )
    short = pd.concat([short, short_status], ignore_index=True, sort=False)

    long = long_direct.copy()
    long["temporal"] = long.method.map(
        {
            "gpflow_svgp_direct_m512": "M_t=8 inducing",
            "routeb_direct_analytic_hippo_rff": "M_t=128 HiPPO",
            "routeb_direct_inducing_points": "M_t=128 inducing",
        }
    )
    long["spatial"] = long.method.map(
        {
            "gpflow_svgp_direct_m512": "M_s=64",
            "routeb_direct_analytic_hippo_rff": "M_s=128",
            "routeb_direct_inducing_points": "M_s=128",
        }
    )
    long["status"] = "complete"
    long_status = pd.DataFrame(
        [
            {
                "scope": "task1_10",
                "method": "st_svgp_ms30",
                "label": "Official ST-SVGP",
                "temporal": "Full Markov",
                "spatial": "M_s=30",
                "status": "OOM before iteration 1 (20.27 GiB allocation)",
            },
            {
                "scope": "task1_10",
                "method": "st_svgp_larger",
                "label": "Official ST-SVGP / MF-ST-SVGP / ST-VGP",
                "temporal": "Full Markov",
                "spatial": "M_s>=30/full",
                "status": "not run after lower-budget OOM",
            },
            {
                "scope": "task1_10",
                "method": "markovflow_long",
                "label": "Official Markovflow ST-SVGP",
                "temporal": "M_t=32/128",
                "spatial": "M_s=128",
                "status": "not run after short lower-capacity failure",
            },
        ]
    )
    long = pd.concat([long, long_status], ignore_index=True, sort=False)
    return short, long


def plot_direct(short_summary: pd.DataFrame, long_summary: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.0), constrained_layout=True)
    for row_index, (frame, title) in enumerate(
        [(short_summary, "Task 2: 186 hours"), (long_summary, "Tasks 2-10: 1674 hours")]
    ):
        frame = frame.loc[frame.status.eq("complete")].copy()
        labels = []
        for _, row in frame.iterrows():
            if row.method.startswith("st_svgp_Ms"):
                labels.append(f"Off. ST-SVGP {row.spatial}")
            elif row.method.startswith("mf_st_svgp"):
                labels.append(f"Off. MF-ST-SVGP {row.spatial}")
            elif row.method.startswith("gpflow"):
                labels.append("Off. GPflow SVGP 8x64")
            elif row.method == "routeb_direct_analytic_hippo_rff":
                labels.append("RB HiPPO 128x128")
            elif row.method == "routeb_direct_inducing_points":
                labels.append("RB inducing 128x128")
            else:
                labels.append(str(row.label))
        x = np.arange(len(frame))
        colors = [
            "#2B8C6B" if "ST-SVGP" in label else
            "#7A5195" if "GPflow" in label else
            "#C23B3B" if "HiPPO" in label else "#1676B8"
            for label in frame.label
        ]
        for col, metric, ylabel in [(0, "rmse", "RMSE"), (1, "nll", "NLL")]:
            ax = axes[row_index, col]
            ax.bar(
                x,
                frame[f"{metric}_mean"],
                yerr=frame[f"{metric}_sd"],
                color=colors,
                edgecolor="white",
                linewidth=0.5,
                capsize=3,
            )
            ax.set_xticks(x, labels, rotation=24, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(title, loc="left", fontweight="bold")
            ax.grid(axis="y", color="#D9DEE2", linewidth=0.6)
            ax.set_axisbelow(True)
    axes[0, 0].text(-0.14, 1.06, "a", transform=axes[0, 0].transAxes, fontweight="bold")
    axes[0, 1].text(-0.14, 1.06, "b", transform=axes[0, 1].transAxes, fontweight="bold")
    axes[1, 0].text(-0.14, 1.06, "c", transform=axes[1, 0].transAxes, fontweight="bold")
    axes[1, 1].text(-0.14, 1.06, "d", transform=axes[1, 1].transAxes, fontweight="bold")
    save_figure(fig, figures / "figure1_direct_target")


def paired_differences(
    short_batch: pd.DataFrame,
    long_batch: pd.DataFrame,
    short_online: pd.DataFrame,
    long_online: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(scope: str, frame: pd.DataFrame, left: str, right: str, comparison: str) -> None:
        lhs = frame.loc[frame.method.eq(left)].set_index("seed")
        rhs = frame.loc[frame.method.eq(right)].set_index("seed")
        for seed in sorted(set(lhs.index) & set(rhs.index)):
            for metric in ("rmse", "nll"):
                rows.append(
                    {
                        "scope": scope,
                        "comparison": comparison,
                        "seed": int(seed),
                        "metric": metric,
                        "difference": float(lhs.loc[seed, metric] - rhs.loc[seed, metric]),
                    }
                )

    for scope, batch, online in [
        ("task1_2", short_batch, short_online),
        ("task1_10", long_batch, long_online),
    ]:
        add(
            scope,
            batch,
            "routeb_direct_analytic_hippo_rff",
            "routeb_direct_inducing_points",
            "Batch direct: HiPPO - inducing",
        )
        add(
            scope,
            batch,
            "routeb_shared_residual_analytic_hippo_rff",
            "routeb_shared_residual_inducing_points",
            "Batch residual: HiPPO - inducing",
        )
        add(
            scope,
            batch,
            "routeb_joint_xlag_inducing_points",
            "routeb_shared_residual_inducing_points",
            "Batch: joint - two-stage",
        )
        add(
            scope,
            online,
            "routeb_analytic_hippo_rff",
            "routeb_inducing_points",
            "Strict online: HiPPO - inducing",
        )
    return pd.DataFrame(rows)


def plot_paired(frame: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.0), constrained_layout=True)
    panels = [
        ("task1_2", "rmse", "Task 2 paired RMSE difference"),
        ("task1_2", "nll", "Task 2 paired NLL difference"),
        ("task1_10", "rmse", "Tasks 2-10 paired RMSE difference"),
        ("task1_10", "nll", "Tasks 2-10 paired NLL difference"),
    ]
    for letter, ax, (scope, metric, title) in zip("abcd", axes.flat, panels):
        selected = frame.loc[(frame.scope == scope) & (frame.metric == metric)]
        comparisons = list(dict.fromkeys(selected.comparison))
        for index, comparison in enumerate(comparisons):
            values = selected.loc[selected.comparison == comparison, "difference"].to_numpy()
            offsets = np.linspace(-0.08, 0.08, len(values))
            ax.scatter(np.full(len(values), index) + offsets, values, s=28, color="#245B78", zorder=3)
            ax.plot([index - 0.18, index + 0.18], [values.mean(), values.mean()], color="#C23B3B", lw=2)
        ax.axhline(0.0, color="#333333", lw=0.9, ls="--")
        ax.set_xticks(
            np.arange(len(comparisons)),
            [
                label.replace("Batch ", "").replace("Strict online", "Online")
                for label in comparisons
            ],
            rotation=22,
            ha="right",
        )
        ax.set_ylabel("Left minus right")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#D9DEE2", linewidth=0.6)
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontweight="bold")
    save_figure(fig, figures / "figure2_paired_seed_differences")


def collect_block_curves(benchmark: Path, scope: str, methods: Iterable[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for method in methods:
        for seed in range(3):
            path = benchmark / "runs" / scope / "online" / method / f"seed{seed}" / "blocks.csv"
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            frame["method"] = method
            frame["seed"] = seed
            rows.append(frame)
    return pd.concat(rows, ignore_index=True, sort=False)


def plot_block_curves(benchmark: Path, figures: Path) -> None:
    methods = [
        "routeb_analytic_hippo_rff",
        "routeb_inducing_points",
        "xlag_mean_recursive_rls",
        "bui_osgpr_prior_m128",
        "official_ohsvgp_m32_rff128",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.1), constrained_layout=True)
    for row_index, (scope, expected_blocks, title) in enumerate(
        [("task1_2", 19, "Task 2: 19 blocks"), ("task1_10", 171, "Tasks 2-10: 171 blocks")]
    ):
        frame = collect_block_curves(benchmark, scope, methods)
        for col, metric, ylabel in [(0, "rmse", "Block RMSE"), (1, "nll", "Block NLL")]:
            ax = axes[row_index, col]
            for method in methods:
                selected = frame.loc[frame.method.eq(method)]
                grouped = selected.groupby("block_id")[metric].agg(["mean", "std"]).reindex(range(expected_blocks))
                x = np.arange(1, expected_blocks + 1)
                mean = grouped["mean"].to_numpy()
                sd = grouped["std"].fillna(0.0).to_numpy()
                color = COLORS[method]
                ax.plot(x, mean, color=color, lw=1.3, label=ONLINE_LABELS[method])
                ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.10, linewidth=0)
            if scope == "task1_10":
                for boundary in range(19, expected_blocks, 19):
                    ax.axvline(boundary + 0.5, color="#D5D9DC", lw=0.5)
            ax.set_xlim(1, expected_blocks)
            ax.set_xlabel("Streaming block")
            ax.set_ylabel(ylabel)
            ax.set_title(title, loc="left", fontweight="bold")
            ax.grid(axis="y", color="#D9DEE2", linewidth=0.6)
    axes[0, 0].legend(ncol=2, fontsize=7, loc="upper right")
    for letter, ax in zip("abcd", axes.flat):
        ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontweight="bold")
    save_figure(fig, figures / "figure3_online_block_curves")


def npz_metrics(y: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> tuple[float, float]:
    var = np.maximum(np.asarray(variance, dtype=float), 1e-10)
    error = np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)
    rmse = float(np.sqrt(np.mean(error**2)))
    nll = float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + error**2 / var)))
    return rmse, nll


def taskwise_metrics(benchmark: Path) -> pd.DataFrame:
    methods = [
        "routeb_analytic_hippo_rff",
        "routeb_inducing_points",
        "xlag_mean_recursive_rls",
        "bui_osgpr_prior_m128",
        "official_ohsvgp_m32_rff128",
        "maddox_streaming_sgpr_m128",
    ]
    rows: list[dict[str, Any]] = []
    for method in methods:
        for seed in range(3):
            path = benchmark / "runs/task1_10/online" / method / f"seed{seed}/predictions.npz"
            with np.load(path) as data:
                y = np.asarray(data["y_true"]).reshape(1674, 200)
                mean = np.asarray(data["pred_mean"]).reshape(1674, 200)
                variance = np.asarray(data["pred_var"]).reshape(1674, 200)
            for task_offset in range(9):
                block = slice(task_offset * 186, (task_offset + 1) * 186)
                rmse, nll = npz_metrics(y[block], mean[block], variance[block])
                rows.append(
                    {
                        "method": method,
                        "label": ONLINE_LABELS[method],
                        "seed": seed,
                        "task": task_offset + 2,
                        "rmse": rmse,
                        "nll": nll,
                    }
                )
    return pd.DataFrame(rows)


def plot_taskwise(frame: pd.DataFrame, figures: Path) -> None:
    main_methods = [
        "routeb_analytic_hippo_rff",
        "routeb_inducing_points",
        "xlag_mean_recursive_rls",
        "bui_osgpr_prior_m128",
        "official_ohsvgp_m32_rff128",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4), constrained_layout=True)
    for method in main_methods:
        selected = frame.loc[frame.method.eq(method)]
        grouped = selected.groupby("task").rmse.agg(["mean", "std"])
        x = grouped.index.to_numpy()
        mean = grouped["mean"].to_numpy()
        sd = grouped["std"].to_numpy()
        color = COLORS[method]
        axes[0].plot(x, mean, marker="o", ms=3.2, color=color, lw=1.4, label=ONLINE_LABELS[method])
        axes[0].fill_between(x, mean - sd, mean + sd, color=color, alpha=0.10, linewidth=0)
    axes[0].set_xlabel("ERA5 task")
    axes[0].set_ylabel("Task-wide RMSE")
    axes[0].set_title("Stable-scale methods", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color="#D9DEE2", linewidth=0.6)
    axes[0].legend(ncol=2, fontsize=7)

    for method in main_methods + ["maddox_streaming_sgpr_m128"]:
        selected = frame.loc[frame.method.eq(method)]
        grouped = selected.groupby("task").rmse.agg(["mean", "std"])
        axes[1].plot(
            grouped.index,
            grouped["mean"],
            marker="o",
            ms=3.0,
            color=COLORS.get(method, "#E07B39"),
            lw=1.2,
            label=ONLINE_LABELS[method],
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("ERA5 task")
    axes[1].set_ylabel("Task-wide RMSE (log scale)")
    axes[1].set_title("Full range including late-stream divergence", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#D9DEE2", linewidth=0.6, which="both")
    axes[0].text(-0.12, 1.06, "a", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.12, 1.06, "b", transform=axes[1].transAxes, fontweight="bold")
    save_figure(fig, figures / "figure4_long_task_curves")


def representative_trajectories(benchmark: Path, figures: Path, tables: Path) -> pd.DataFrame:
    methods = [
        "routeb_analytic_hippo_rff",
        "routeb_inducing_points",
        "official_ohsvgp_m32_rff128",
    ]
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        path = benchmark / "runs/task1_10/online" / method / "seed0/predictions.npz"
        with np.load(path) as data:
            loaded[method] = {key: np.asarray(data[key]) for key in data.files}
    reference = loaded["routeb_analytic_hippo_rff"]
    y = reference["y_true"].reshape(1674, 200)
    pred = reference["pred_mean"].reshape(1674, 200)
    location_rmse = np.sqrt(np.mean((y - pred) ** 2, axis=0))
    order = np.argsort(location_rmse)
    quantiles = [0.10, 0.50, 0.90]
    names = ["Success (10th percentile)", "Median (50th percentile)", "Failure (90th percentile)"]
    chosen = [int(order[int(round(q * (len(order) - 1)))]) for q in quantiles]
    global_indices = reference.get("test_indices", np.arange(200)).reshape(-1)
    selection = pd.DataFrame(
        {
            "case": names,
            "quantile": quantiles,
            "test_column": chosen,
            "global_location_index": [int(global_indices[index]) for index in chosen],
            "routeb_hippo_location_rmse": [float(location_rmse[index]) for index in chosen],
        }
    )
    write_csv(selection, tables / "representative_location_selection.csv")

    fig, axes = plt.subplots(3, 3, figsize=(11.0, 7.1), sharex=True, constrained_layout=True)
    method_titles = {
        "routeb_analytic_hippo_rff": "Route B cumulative HiPPO",
        "routeb_inducing_points": "Route B global inducing",
        "official_ohsvgp_m32_rff128": "Official OHSVGP",
    }
    method_colors = {
        "routeb_analytic_hippo_rff": "#C23B3B",
        "routeb_inducing_points": "#1676B8",
        "official_ohsvgp_m32_rff128": "#2B8C6B",
    }
    for row, (name, column) in enumerate(zip(names, chosen)):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            payload = loaded[method]
            truth = payload["y_true"].reshape(1674, 200)[:, column]
            mean = payload["pred_mean"].reshape(1674, 200)[:, column]
            variance = np.maximum(payload["pred_var"].reshape(1674, 200)[:, column], 1e-10)
            x = np.arange(1674)
            half = 1.6448536269514722 * np.sqrt(variance)
            ax.fill_between(x, mean - half, mean + half, color=method_colors[method], alpha=0.16, linewidth=0)
            ax.plot(x, mean, color=method_colors[method], lw=0.8, label="Prediction")
            ax.plot(x, truth, color="black", lw=0.75, alpha=0.90, label="Ground truth")
            rmse = float(np.sqrt(np.mean((truth - mean) ** 2)))
            if row == 0:
                ax.set_title(method_titles[method], fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{name}\nScaled target")
            if row == 2:
                ax.set_xlabel("Streaming hour")
            ax.text(0.98, 0.05, f"RMSE {rmse:.3f}", transform=ax.transAxes, ha="right", fontsize=7)
            ax.grid(axis="y", color="#E0E3E5", linewidth=0.45)
    axes[0, 0].legend(ncol=2, fontsize=7, loc="upper right")
    save_figure(fig, figures / "figure5_representative_trajectories")
    return selection


def efficiency_tables(
    benchmark: Path,
    short_batch: pd.DataFrame,
    long_online: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    batch_rows: list[dict[str, Any]] = []
    official_path = OFFICIAL_XLAG / "seed0/official_st_svgp_Ms128/result.json"
    official = read_json(official_path)
    official_peak = parse_peak_rss(official_path.parent / "resource_usage.txt")
    hlo = read_json(FLOP_ROOT / "official_stsvgp_hlo_cost_seed0.json")
    route_flops = {row["method"].split()[-1]: row for row in read_json(FLOP_ROOT / "routeb_flop_profile.json")}
    batch_rows.append(
        {
            "method": "Official ST-SVGP + learned X-lag",
            "training_seconds": official["train_seconds"],
            "mean_iteration_seconds": official["train_seconds"] / official["iterations"],
            "iterations": official["iterations"],
            "time_to_best_validation_seconds": float("nan"),
            "prediction_seconds": float("nan"),
            "peak_rss_mib": official_peak,
            "persistent_state_mib": float("nan"),
            "gflops_per_step": hlo["xla_cost_analysis"]["flops"] / 1e9,
            "flops_scope": "XLA cost analysis of one jitted BN/Adam train op",
        }
    )
    for method, representation in [
        ("routeb_joint_xlag_analytic_hippo_rff", "analytic_hippo_rff"),
        ("routeb_joint_xlag_inducing_points", "inducing_points"),
    ]:
        path = benchmark / "runs/task1_2/batch" / method / "seed0/result.json"
        payload = read_json(path)
        batch_rows.append(
            {
                "method": BATCH_LABELS[method],
                "training_seconds": payload["timing"]["training_seconds"],
                "mean_iteration_seconds": payload["timing"]["mean_iteration_seconds"],
                "iterations": payload["args"]["iterations"],
                "time_to_best_validation_seconds": payload["time_to_best_validation_seconds"],
                "prediction_seconds": payload["timing"]["prediction_seconds"],
                "peak_rss_mib": payload["resources"]["peak_rss_mib"],
                "persistent_state_mib": payload["resources"]["persistent_model_state_mib"],
                "gflops_per_step": route_flops[representation]["gflops"],
                "flops_scope": "PyTorch supported-ATen forward/backward count",
            }
        )
    batch = pd.DataFrame(batch_rows)

    online_rows: list[dict[str, Any]] = []
    for method in [
        "xlag_mean_recursive_rls",
        "bui_osgpr_prior_m128",
        "maddox_streaming_sgpr_m128",
        "official_ohsvgp_m32_rff128",
        "routeb_inducing_points",
        "routeb_analytic_hippo_rff",
    ]:
        path = benchmark / "runs/task1_10/online" / method / "seed0/result.json"
        payload = read_json(path)
        timing = payload.get("timing", {})
        resources = payload.get("resources", {})
        online_rows.append(
            {
                "method": ONLINE_LABELS[method],
                "process_total_seconds": timing.get("process_total_seconds"),
                "mean_block_update_seconds": timing.get("mean_block_update_seconds"),
                "mean_block_prediction_seconds": timing.get("mean_block_prediction_seconds"),
                "peak_rss_mib": resources.get("peak_rss_mib"),
                "persistent_state_mib": resources.get("persistent_state_mib"),
                "history_replay_mib": (resources.get("history_replay_buffer_bytes", 0) or 0) / 1024.0**2,
                "estimated_total_gflops": (
                    resources.get("estimated_streaming_flops") / 1e9
                    if resources.get("estimated_streaming_flops") is not None
                    else float("nan")
                ),
                "rmse": nested(payload, "overall_current_block.rmse", "final.rmse"),
                "nll": nested(payload, "overall_current_block.nll", "final.nll"),
            }
        )
    return batch, pd.DataFrame(online_rows)


def latex_escape(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def pm(mean: Any, sd: Any, digits: int = 4) -> str:
    if mean is None or pd.isna(mean):
        return "--"
    if sd is None or pd.isna(sd):
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} $\\pm$ {float(sd):.{digits}f}"


def fnum(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def table_rows(frame: pd.DataFrame, include_status: bool = False, online: bool = False) -> str:
    rows: list[str] = []
    for _, row in frame.iterrows():
        status = row.get("status", "complete")
        if status != "complete":
            if include_status:
                rows.append(
                    f"{latex_escape(row.label)} & {latex_escape(row.get('temporal', '--'))} & "
                    f"{latex_escape(row.get('spatial', '--'))} & "
                    f"\\multicolumn{{3}}{{l}}{{{latex_escape(status)}}} \\\\"
                )
            continue
        if online:
            rows.append(
                f"{latex_escape(row.label)} & {pm(row.rmse_mean, row.rmse_sd)} & "
                f"{pm(row.nll_mean, row.nll_sd)} & {pm(row.coverage90_mean, row.coverage90_sd, 3)} & "
                f"{pm(row.final_rmse_mean, row.final_rmse_sd)} \\\\"
            )
        elif include_status:
            rows.append(
                f"{latex_escape(row.label)} & {latex_escape(row.get('temporal', '--'))} & "
                f"{latex_escape(row.get('spatial', '--'))} & {pm(row.rmse_mean, row.rmse_sd)} & "
                f"{pm(row.nll_mean, row.nll_sd)} & {pm(row.coverage90_mean, row.coverage90_sd, 3)} \\\\"
            )
        else:
            rows.append(
                f"{latex_escape(row.label)} & {pm(row.rmse_mean, row.rmse_sd)} & "
                f"{pm(row.nll_mean, row.nll_sd)} & {pm(row.coverage90_mean, row.coverage90_sd, 3)} \\\\"
            )
    return "\n".join(rows)


def efficiency_batch_rows(frame: pd.DataFrame) -> str:
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            f"{latex_escape(row.method)} & {fnum(row.training_seconds, 1)} & "
            f"{fnum(row.mean_iteration_seconds, 3)} & {int(row.iterations)} & "
            f"{fnum(row.time_to_best_validation_seconds, 1)} & {fnum(row.prediction_seconds, 2)} & "
            f"{fnum(row.peak_rss_mib / 1024.0, 2)} & {fnum(row.persistent_state_mib, 2)} & "
            f"{fnum(row.gflops_per_step, 1)} \\\\"
        )
    return "\n".join(rows)


def efficiency_online_rows(frame: pd.DataFrame) -> str:
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            f"{latex_escape(row.method)} & {fnum(row.process_total_seconds, 1)} & "
            f"{fnum(row.mean_block_update_seconds, 3)} & {fnum(row.mean_block_prediction_seconds, 3)} & "
            f"{fnum(row.peak_rss_mib / 1024.0, 2)} & {fnum(row.persistent_state_mib, 3)} & "
            f"{fnum(row.history_replay_mib, 1)} & {fnum(row.estimated_total_gflops, 1)} \\\\"
        )
    return "\n".join(rows)


def write_report(
    output: Path,
    short_direct: pd.DataFrame,
    long_direct: pd.DataFrame,
    short_xlag: pd.DataFrame,
    long_xlag: pd.DataFrame,
    short_online: pd.DataFrame,
    long_online: pd.DataFrame,
    efficiency_batch: pd.DataFrame,
    efficiency_online: pd.DataFrame,
) -> Path:
    def metric(frame: pd.DataFrame, method: str, name: str = "rmse_mean") -> float:
        return float(frame.loc[frame.method.eq(method), name].iloc[0])

    long_hippo = metric(long_online, "routeb_analytic_hippo_rff")
    long_ordinary = metric(long_online, "routeb_inducing_points")
    short_hippo = metric(short_online, "routeb_analytic_hippo_rff")
    short_ordinary = metric(short_online, "routeb_inducing_points")
    long_joint = metric(long_xlag, "routeb_joint_xlag_inducing_points")
    long_two_stage = metric(long_xlag, "routeb_shared_residual_inducing_points")
    short_joint = metric(short_xlag, "routeb_joint_xlag_inducing_points")
    short_two_stage = metric(short_xlag, "routeb_shared_residual_inducing_points")
    official_speed = float(efficiency_batch.iloc[0].training_seconds)
    hippo_speed = float(efficiency_batch.iloc[1].training_seconds)
    ordinary_speed = float(efficiency_batch.iloc[2].training_seconds)
    official_flops = float(efficiency_batch.iloc[0].gflops_per_step)
    hippo_flops = float(efficiency_batch.iloc[1].gflops_per_step)

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[a4paper,margin=17mm]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,tabularx,array,graphicx,float,microtype,xcolor,hyperref,fancyhdr}}
\definecolor{{accent}}{{HTML}}{{245B78}}
\definecolor{{soft}}{{HTML}}{{EDF3F6}}
\definecolor{{warn}}{{HTML}}{{A14B35}}
\hypersetup{{colorlinks=true,linkcolor=accent,urlcolor=accent}}
\pagestyle{{fancy}}\fancyhf{{}}\lhead{{ERA5 protocol-separated benchmark}}\rhead{{\thepage}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{4.5pt}}
\newcolumntype{{Y}}{{>{{\raggedright\arraybackslash}}X}}
\title{{A Protocol-Separated ERA5 Benchmark of Structured-Joint Route B}}
\author{{Batch accuracy, strict streaming, and implementation-level efficiency}}
\date{{2 August 2026}}
\begin{{document}}
\maketitle

\begin{{abstract}}
We evaluate structured-joint Route B on a spatially held-out ERA5-Land benchmark while separating three questions that are often conflated: literature-faithful direct-target modelling, controlled residual modelling with shared X-lag covariates, and strict blockwise streaming without history replay. Task 1 provides independent calibration; Task 2 contains 186 hourly states and Tasks 2--10 contain 1674 streaming hours. Every completed accuracy row uses the same 1000 locations, paired 800/200 spatial splits with seeds 0--2, and metrics computed on locations excluded at every time. On the short stream, fixed global temporal inducing points are more accurate than cumulative HiPPO (RMSE {short_ordinary:.4f} versus {short_hippo:.4f}). On the nine-task stream the ordering reverses: cumulative-changing HiPPO reaches {long_hippo:.4f}, improving on the fixed global inducing baseline ({long_ordinary:.4f}) by {long_ordinary-long_hippo:.4f}. In batch residual modelling, joint X-lag--GP coupling improves ordinary Route B from {short_two_stage:.4f} to {short_joint:.4f} on Task 2 and from {long_two_stage:.4f} to {long_joint:.4f} on Tasks 2--10. Official full-Markov ST-SVGP completes the short benchmark but its lowest long-stream budget requests 20.27 GiB before the first iteration. We report failed official configurations as failures rather than imputing accuracy.
\end{{abstract}}

\section{{Questions and common protocol}}
The benchmark is organized around four tables. \textbf{{Table 1}} tests raw spatio-temporal capacity by modelling scaled target $y$ directly. \textbf{{Table 2}} gives every completed method the same fixed X-lag ridge mean, or explicitly labels method-specific joint mean inference. \textbf{{Table 3}} is strict online: Task-1 calibration is frozen, each Task-2(+) label is consumed once, and no past observation is replayed. \textbf{{Table 4}} reports observed runtime, per-step cost, memory and state size. Results from these layers are not merged into one ranking.

The dataset uses ERA5-Land variable 0 at 1000 fixed UK locations. For each of three paired split seeds, 800 locations are available for model fitting and validation and 200 locations are held out for all times; the 800 are split into 720 fit and 80 validation locations for empirical-Bayes checkpointing. Task 1 contains 186 hours and is used for calibration. The short stream is Task 2 (186 hours, 19 blocks); the long stream concatenates Tasks 2--10 (1674 hours, 171 task-aware blocks). Each block contains 10 hours except the six-hour task endings. X-lag uses 133 non-target meteorological features with lag length 10. Route B uses $M_t=M_s=128$, deterministic training-only k-means spatial inducing points, fixed temporal inducing coordinates when applicable, and finite DTC prediction without conditional residual variance.

All reported standard deviations are sample SD over the three paired spatial splits. Three seeds support direction and robustness checks but not asymptotic significance claims.

\section{{Table 1: literature-faithful batch direct target $y$}}
\subsection{{Task 2: 186-hour benchmark}}
Official ST-VGP/ST-SVGP rows use the Aalto Bayes--Newton/Objax implementation, direct $y$, Mat\'ern-$3/2$ temporal and spatial kernels, and full temporal Markov states. ST-SVGP is sparse only in space. GPflow SVGP follows the official repository training path with an $8\times64$ Cartesian inducing grid. Route B learns its own kernel/noise parameters by marginal likelihood with fixed $128\times128$ finite representation.

\begin{{table}}[H]\centering\scriptsize
\caption{{Literature-faithful direct-target results on Task 2. Completed rows report mean $\pm$ SD over three paired splits. Failed rows have no imputed metrics.}}
\begin{{tabularx}}{{\textwidth}}{{Yllrrr}}\toprule
Method & Temporal & Spatial & RMSE & NLL & Cov$_{{90}}$ \\ \midrule
{table_rows(short_direct, include_status=True)}
\bottomrule\end{{tabularx}}
\end{{table}}

Full ST-VGP exhausted the available accelerator memory. ST-SVGP improves from $M_s=30$ to 128, showing that spatial residual capacity matters. These rows are literature-faithful references, not matched-$M_t$ comparisons: full-Markov ST-SVGP has no temporal inducing count. Markovflow's target-capacity official path did not yield a valid accuracy row: sparse variational inference failed its banded-Cholesky reconstruction check and sparse CVI did not complete a first iteration within 900 seconds.

\subsection{{Tasks 2--10: 1674-hour benchmark}}
\begin{{table}}[H]\centering\scriptsize
\caption{{Direct-target batch results on Tasks 2--10. The official ST-SVGP $M_s=30$ probe failed before iteration 1 after requesting 20.27 GiB; larger full-Markov configurations were therefore not run.}}
\begin{{tabularx}}{{\textwidth}}{{Yllrrr}}\toprule
Method & Temporal & Spatial & RMSE & NLL & Cov$_{{90}}$ \\ \midrule
{table_rows(long_direct, include_status=True)}
\bottomrule\end{{tabularx}}
\end{{table}}

\begin{{figure}}[H]\centering
\includegraphics[width=0.98\linewidth]{{figures/figure1_direct_target.pdf}}
\caption{{Direct-target batch performance. Error bars are one sample SD over paired split seeds. Panels (a,b) show Task 2; panels (c,d) show Tasks 2--10. Failed official configurations are intentionally absent from bars and remain explicit in the tables.}}
\end{{figure}}

Ordinary temporal inducing points are stronger than HiPPO for batch/full-history prediction at the same $M_t=M_s=128$. This is expected under a fixed known horizon: ordinary inducing coordinates directly cover the complete interval, whereas the analytic HiPPO-RFF basis is optimized for compressed history representation rather than finite-horizon interpolation.

\section{{Table 2: controlled batch with shared X-lag}}
The two-stage rows first fit the same ridge mean to all 800 training locations and then fit the same residual target. The structured-joint rows instead optimize the finite-model marginal likelihood and retain the full Gaussian coupling between the 133 mean coefficients and GP variables. This distinction is part of the proposed architecture and is labeled rather than hidden.

\begin{{table}}[H]\centering\scriptsize
\caption{{Task 2 shared-X-lag batch comparison. Mean $\pm$ SD over three paired splits.}}
\begin{{tabularx}}{{\textwidth}}{{Yrrr}}\toprule
Method & RMSE & NLL & Cov$_{{90}}$ \\ \midrule
{table_rows(short_xlag)}
\bottomrule\end{{tabularx}}
\end{{table}}

\begin{{table}}[H]\centering\scriptsize
\caption{{Tasks 2--10 shared-X-lag batch comparison. The long structured-joint HiPPO run was not repeated after the representation sweep established ordinary inducing points as the stronger batch backend.}}
\begin{{tabularx}}{{\textwidth}}{{Yrrr}}\toprule
Method & RMSE & NLL & Cov$_{{90}}$ \\ \midrule
{table_rows(long_xlag)}
\bottomrule\end{{tabularx}}
\end{{table}}

On Task 2, joint coupling improves ordinary inducing Route B from RMSE {short_two_stage:.4f} to {short_joint:.4f}. On Tasks 2--10 it improves {long_two_stage:.4f} to {long_joint:.4f}. The gain therefore persists beyond the original 186-hour study. The long joint model is nevertheless expensive because validation and prediction materialize large $1674\times1000$ feature workloads; its best checkpoint remains iteration 100, so this row should not be presented as a converged global optimum.

\begin{{figure}}[H]\centering
\includegraphics[width=0.80\linewidth]{{figures/figure2_paired_seed_differences.pdf}}
\caption{{Paired seed differences. Points are split seeds and red segments are means. Positive HiPPO-minus-inducing values favour ordinary inducing points; negative joint-minus-two-stage values favour joint coupling. The online comparison reverses sign on the long stream.}}
\end{{figure}}

\section{{Table 3: strict online streaming}}
Strict-online methods receive Task-1 calibration and then consume each new block once. The main Route B protocol is Task-1 Route-B empirical Bayes $\rightarrow$ freeze $\theta$ $\rightarrow$ Task-2(+) streaming. Cumulative-changing HiPPO rebases the temporal representation on $[t_0,t_k]$ and transfers sufficient statistics. The ordinary baseline uses 128 fixed global linspace coordinates over the known evaluation horizon; the coordinates are not optimized. OSGPR, Maddox StreamingSGPR and OHSVGP are official implementations with thin data wrappers. ST-SVGP all-seen refitting is excluded because it replays history and is not strict online.

\begin{{table}}[H]\centering\scriptsize
\caption{{Strict online results on Task 2. Stream-wide metrics score each block at arrival; final-block RMSE is shown separately.}}
\begin{{tabularx}}{{\textwidth}}{{Yrrrr}}\toprule
Method & Stream RMSE & Stream NLL & Cov$_{{90}}$ & Final RMSE \\ \midrule
{table_rows(short_online, online=True)}
\bottomrule\end{{tabularx}}
\end{{table}}

\begin{{table}}[H]\centering\scriptsize
\caption{{Strict online results on Tasks 2--10, mean $\pm$ SD over three paired splits.}}
\begin{{tabularx}}{{\textwidth}}{{Yrrrr}}\toprule
Method & Stream RMSE & Stream NLL & Cov$_{{90}}$ & Final RMSE \\ \midrule
{table_rows(long_online, online=True)}
\bottomrule\end{{tabularx}}
\end{{table}}

The short stream favours fixed ordinary inducing points ({short_ordinary:.4f} versus {short_hippo:.4f}). Across 1674 hours, cumulative HiPPO improves to {long_hippo:.4f} while the fixed-grid model degrades to {long_ordinary:.4f}. The crossover is the central temporal-representation result: the advantage is not visible in batch or short known-horizon interpolation, but emerges when a fixed budget must cover a stream nine times longer. The ordinary baseline is also given the complete future endpoint when its global linspace grid is created, so it is a favourable oracle-like control rather than an unknown-horizon method.

\begin{{figure}}[H]\centering
\includegraphics[width=0.99\linewidth]{{figures/figure3_online_block_curves.pdf}}
\caption{{Blockwise strict-online RMSE/NLL. Curves are seed means and ribbons are one SD. Vertical separators in the lower panels mark task boundaries. Maddox StreamingSGPR is omitted here because late-stream divergence would obscure the stable-scale methods; it remains in Table 3 and Figure 4b.}}
\end{{figure}}

\begin{{figure}}[H]\centering
\includegraphics[width=0.98\linewidth]{{figures/figure4_long_task_curves.pdf}}
\caption{{Task-wise long-stream RMSE. Panel (a) resolves stable-scale methods. Panel (b) uses a logarithmic scale and includes Maddox StreamingSGPR, exposing late-stream divergence rather than hiding it through averaging.}}
\end{{figure}}

\begin{{figure}}[H]\centering
\includegraphics[width=0.99\linewidth]{{figures/figure5_representative_trajectories.pdf}}
\caption{{Pre-declared representative held-out trajectories for split seed 0. Locations are selected at the 10th, 50th and 90th percentiles of cumulative-HiPPO per-location RMSE before plotting. Columns show the same locations for Route B HiPPO, Route B fixed global inducing points and official OHSVGP. Ground truth is black; coloured curves and bands are predictive means and 90\% intervals.}}
\end{{figure}}

Route B's long-stream predictive intervals under-cover: its mean predictive standard deviation remains close to the Task-1 frozen noise scale while the stream changes across tasks. The strong RMSE does not remove this calibration limitation. OHSVGP has much wider intervals and better nominal coverage but substantially worse RMSE. Accuracy and uncertainty calibration should therefore be reported jointly.

\section{{Table 4: implementation-level efficiency}}
All timings were measured on CPU under WSL2 on an AMD Ryzen 9 7845HX (12 cores, 24 threads) with 15 GiB RAM and 4 GiB swap. They include each wrapper's real data loading and validation policy, so they are end-to-end implementation measurements rather than backend-neutral complexity proofs.

\begin{{table}}[H]\centering\tiny
\caption{{Task-2 batch efficiency, seed 0. Official ST-SVGP uses the learned-X-lag official run at $M_s=128$; both Route B rows learn their own five hyperparameters at $M_t=M_s=128$. FLOPs/step use profiler-specific counts and are interpreted only within the stated scope.}}
\resizebox{{\textwidth}}{{!}}{{\begin{{tabular}}{{lrrrrrrrr}}\toprule
Method & Train s & s/iter & Iter & Time-best s & Predict s & Peak GiB & State MiB & GFLOPs/step \\ \midrule
{efficiency_batch_rows(efficiency_batch)}
\bottomrule\end{{tabular}}}}
\end{{table}}

The observed official-to-Route-B training-time ratio is {official_speed/hippo_speed:.1f}$\times$ for HiPPO and {official_speed/ordinary_speed:.1f}$\times$ for ordinary inducing points. Per optimization step, official ST-SVGP's XLA cost analysis reports {official_flops:.1f} GFLOPs, versus {hippo_flops:.1f} supported-ATen GFLOPs for Route B, a nominal {official_flops/hippo_flops:.1f}$\times$ ratio. These FLOP counters have different operator coverage, and official ST-SVGP uses legacy JAX/Objax while Route B uses PyTorch plus NumPy/SciPy. The wall-clock and memory ratios are reproducible implementation results; a pure algorithmic claim requires a common backend.

\begin{{table}}[H]\centering\tiny
\caption{{Tasks 2--10 strict-online efficiency, seed 0. Persistent state excludes source data and code. Replay is zero for every strict method. A dash means that the wrapper did not expose a compatible FLOP count.}}
\resizebox{{\textwidth}}{{!}}{{\begin{{tabular}}{{lrrrrrrr}}\toprule
Method & Total s & Update/block s & Predict/block s & Peak GiB & State MiB & Replay MiB & Est. GFLOPs \\ \midrule
{efficiency_online_rows(efficiency_online)}
\bottomrule\end{{tabular}}}}
\end{{table}}

HiPPO and ordinary Route B have similar total long-stream runtime, but HiPPO is markedly more accurate. Official online methods retain much smaller states, especially OHSVGP, but lose substantial accuracy. Route B therefore currently occupies an accuracy-heavy point on the memory--accuracy frontier rather than dominating every resource dimension.

\section{{Main conclusions}}
\textbf{{1. Joint coupling is supported in batch.}} The improvement over the shared two-stage residual model persists over both 186 and 1674 hours. This is the cleanest evidence for the structured $\beta$--GP posterior contribution.

\textbf{{2. HiPPO's value is protocol dependent.}} Ordinary temporal inducing points are the correct batch and short-horizon choice. Cumulative-changing HiPPO becomes preferable when the same temporal budget must cover the longer block stream. It should not be advertised as a universal kernel approximation improvement.

\textbf{{3. Streaming transfer is not the dominant long-stream accuracy bottleneck for the cumulative model.}} Route B retains strong RMSE without replay over 171 updates. Remaining issues are uncertainty calibration, persistent state size, and formal separation of basis-transfer error from Task-1-to-Task-10 hyperparameter shift.

\textbf{{4. Official comparisons require two tables, not one.}} Full-Markov ST-SVGP is a literature-faithful batch reference and cannot be assigned an artificial $M_t$. OSGPR, StreamingSGPR and OHSVGP are the strict-online baselines. The report does not present all-seen ST-SVGP refits as online.

\section{{Limitations and next decision gates}}
The official full-Markov long-stream OOM is hardware-specific and is not evidence that ST-SVGP is intrinsically inaccurate. Markovflow accuracy is unreported because target-capacity official runs did not complete reliably; smoke-test accuracy is deliberately excluded. GPflow uses a smaller feasible $8\times64$ inducing grid than Route B's $128\times128$ structured representation. Runtime comparisons span JAX, TensorFlow/GPflow, PyTorch and NumPy/SciPy. Three spatial splits do not support broad significance claims.

The next experiments should: (i) calibrate online predictive variance without leaking future blocks; (ii) compare cumulative HiPPO against moving or adaptively inserted ordinary inducing points under the same unknown-horizon constraint; (iii) reduce the $\beta$--$u$ cross-state footprint; and (iv) rerun the most informative efficiency pair on a common JAX or PyTorch backend. Expanding to more target variables and five spatial splits should follow only after these method-level questions are settled.

\section*{{Reproducibility and audit boundary}}
All 15 required strict-online runs pass shape, finiteness, block-count and positive-variance audits for each scope. Task 1--10 contains 15/15 complete online artifacts. The report generator reads original per-seed files directly, writes all aggregate CSVs next to the report, and records SHA-256 hashes in the artifact manifest. Failed official configurations have status JSON/log evidence and no fabricated metric values.

\end{{document}}
"""
    path = output / "iclr_era5_full_benchmark_report.tex"
    path.write_text(tex, encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_manifest(benchmark: Path, output: Path) -> Path:
    tex_path = output / "iclr_era5_full_benchmark_report.tex"
    generated = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "benchmark_root": str(benchmark),
        "report_tex": str(tex_path),
        "short_online_audit": str(benchmark / "audit_task1_2.json"),
        "long_online_audit": str(benchmark / "audit_task1_10.json"),
        "official_long_status": str(
            benchmark / "probes/official_st_svgp_task1_10_ms30_seed0/status.json"
        ),
        "generated_artifacts": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in generated
        ],
    }
    manifest_path = output / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--refresh-manifest-only",
        action="store_true",
        help="Re-hash compiled report artifacts without regenerating tables or figures.",
    )
    args = parser.parse_args()
    benchmark = args.benchmark_root.resolve()
    output = (args.output_dir or benchmark / "report").resolve()
    if args.refresh_manifest_only:
        if not output.is_dir():
            raise FileNotFoundError(f"Report directory does not exist: {output}")
        manifest_path = write_artifact_manifest(benchmark, output)
        print(json.dumps({"manifest": str(manifest_path)}, indent=2))
        return

    figures = output / "figures"
    tables = output / "tables"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    short_direct_raw, short_xlag_raw = batch_table_data(benchmark, "task1_2")
    long_direct_raw, long_xlag_raw = batch_table_data(benchmark, "task1_10")
    short_online_raw = online_table_data(benchmark, "task1_2")
    long_online_raw = online_table_data(benchmark, "task1_10")

    short_direct_status, long_direct_status = direct_status_tables(
        benchmark, short_direct_raw, long_direct_raw
    )
    short_direct = aggregate(short_direct_status, ["method", "label", "temporal", "spatial", "status"])
    long_direct = aggregate(long_direct_status, ["method", "label", "temporal", "spatial", "status"])
    short_xlag = aggregate(short_xlag_raw, ["method", "label"])
    long_xlag = aggregate(long_xlag_raw, ["method", "label"])
    short_online = aggregate(short_online_raw, ["method", "label"])
    long_online = aggregate(long_online_raw, ["method", "label"])

    write_csv(short_direct_status, tables / "table1_task1_2_direct_per_seed_and_status.csv")
    write_csv(short_direct, tables / "table1_task1_2_direct_summary.csv")
    write_csv(long_direct_status, tables / "table1_task1_10_direct_per_seed_and_status.csv")
    write_csv(long_direct, tables / "table1_task1_10_direct_summary.csv")
    write_csv(short_xlag_raw, tables / "table2_task1_2_xlag_per_seed.csv")
    write_csv(short_xlag, tables / "table2_task1_2_xlag_summary.csv")
    write_csv(long_xlag_raw, tables / "table2_task1_10_xlag_per_seed.csv")
    write_csv(long_xlag, tables / "table2_task1_10_xlag_summary.csv")
    write_csv(short_online_raw, tables / "table3_task1_2_online_per_seed.csv")
    write_csv(short_online, tables / "table3_task1_2_online_summary.csv")
    write_csv(long_online_raw, tables / "table3_task1_10_online_per_seed.csv")
    write_csv(long_online, tables / "table3_task1_10_online_summary.csv")

    paired = paired_differences(
        pd.concat([short_direct_raw, short_xlag_raw], ignore_index=True, sort=False),
        pd.concat([long_direct_raw, long_xlag_raw], ignore_index=True, sort=False),
        short_online_raw,
        long_online_raw,
    )
    write_csv(paired, tables / "paired_seed_differences.csv")

    taskwise = taskwise_metrics(benchmark)
    write_csv(taskwise, tables / "task1_10_online_taskwise_per_seed.csv")
    write_csv(
        aggregate(taskwise.assign(coverage90=np.nan, final_rmse=np.nan), ["method", "label", "task"]),
        tables / "task1_10_online_taskwise_summary.csv",
    )

    plot_direct(short_direct, long_direct, figures)
    plot_paired(paired, figures)
    plot_block_curves(benchmark, figures)
    plot_taskwise(taskwise, figures)
    representative_trajectories(benchmark, figures, tables)

    efficiency_batch, efficiency_online = efficiency_tables(
        benchmark,
        pd.concat([short_direct_raw, short_xlag_raw], ignore_index=True, sort=False),
        long_online_raw,
    )
    write_csv(efficiency_batch, tables / "table4_batch_efficiency_seed0.csv")
    write_csv(efficiency_online, tables / "table4_online_efficiency_seed0.csv")

    tex_path = write_report(
        output,
        short_direct,
        long_direct,
        short_xlag,
        long_xlag,
        short_online,
        long_online,
        efficiency_batch,
        efficiency_online,
    )

    commands = """# ERA5 full benchmark reproducibility entry points

# Shared protocol exports
.venv/bin/python scripts/export_iclr_era5_full_benchmark_protocol.py

# Batch matrices
bash scripts/run_iclr_era5_long_routeb_batch_matrix.sh
bash scripts/run_iclr_era5_long_gpflow_batch_matrix.sh
bash scripts/run_iclr_era5_joint_batch_matrix.sh

# Mean and strict-online baselines
bash scripts/run_iclr_era5_xlag_mean_matrix.sh
.venv/bin/python scripts/audit_iclr_era5_full_benchmark_results.py --help

# Aggregation and report
.venv/bin/python scripts/generate_iclr_era5_full_benchmark_report.py

# After compiling the LaTeX report, refresh hashes for the final PDF and log files
.venv/bin/python scripts/generate_iclr_era5_full_benchmark_report.py --refresh-manifest-only
"""
    (output / "reproduction_commands.txt").write_text(commands, encoding="utf-8")

    write_artifact_manifest(benchmark, output)
    print(json.dumps({"output": str(output), "tex": str(tex_path)}, indent=2))


if __name__ == "__main__":
    main()
