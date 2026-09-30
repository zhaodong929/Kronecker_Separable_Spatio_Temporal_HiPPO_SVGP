#!/usr/bin/env python3
"""Aggregate P0-P3 outputs, paired statistics, and standardized P4 figures."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hipposvgp_era5_routeb import fixed_spatial_train_test_split
from stvgp_kronecker.data.hipposvgp_era5 import load_hipposvgp_era5


PALETTE = {
    "blue": "#3977A8",
    "red": "#B24A4A",
    "green": "#4A8062",
    "gold": "#B58A34",
    "charcoal": "#333333",
    "grey": "#8A8F93",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def parse_peak_rss(path: Path) -> float:
    if not path.exists():
        return float("nan")
    match = re.search(r"Maximum resident set size \(kbytes\):\s*([0-9]+)", path.read_text(errors="ignore"))
    return float(match.group(1)) / 1024.0 if match else float("nan")


def capacity_from_path(path: Path) -> tuple[int, int]:
    match = re.search(r"Mt(\d+)_Ms(\d+)", str(path))
    if match is None:
        raise ValueError(f"Capacity not found in {path}")
    return int(match.group(1)), int(match.group(2))


def normalize_local_p1(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["peak_rss_mb"] = parse_peak_rss(path.parent / "resource_usage.txt")
    return frame


def normalize_online_p1(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.loc[raw["eval_mode"] == "seen_history"].copy()
    frame = pd.DataFrame(
        {
            "architecture": "structured_joint_hippo_stvgp",
            "protocol": "online",
            "temporal_representation": "analytic_hippo_rff",
            "heldout_split_seed": raw["heldout_split_seed"].astype(int),
            "block_size": raw["block_size"].astype(int),
            "block_id": raw["block_id"].astype(int),
            "num_seen_time": (raw["num_test"] / raw["num_test_locations"]).round().astype(int),
            "mt": raw["mt"].astype(int),
            "effective_mt": raw["mt"].astype(int),
            "ms": raw["ms"].astype(int),
            "rmse": raw["rmse"],
            "nll": raw["nll"],
            "coverage90": raw["coverage90"],
            "ece": raw["ece"],
            "avg_std": raw["avg_std"],
            "update_runtime_sec": raw["runtime_per_block"],
            "prediction_runtime_sec": raw.get("prediction_runtime", np.nan),
            "block_incremental_runtime_sec": raw.get("block_incremental_runtime", raw["runtime_per_block"]),
            "num_test": raw["num_test"].astype(int),
        }
    )
    max_block = frame["block_id"].max()
    frame["is_final_block"] = frame["block_id"] == max_block
    frame["peak_rss_mb"] = parse_peak_rss(path.parent / "resource_usage.txt")
    return frame


def aggregate_mean_sd(frame: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    records = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        item = dict(zip(group_cols, keys))
        item["num_seeds"] = int(group["heldout_split_seed"].nunique()) if "heldout_split_seed" in group else len(group)
        for metric in metrics:
            values = group[metric].astype(float).to_numpy()
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        records.append(item)
    return pd.DataFrame(records)


def paired_rows(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    metrics = ["rmse", "nll", "coverage90", "block_incremental_runtime_sec", "peak_rss_mb"]
    final = frame.loc[frame["is_final_block"]].copy()
    for architecture in sorted(final["architecture"].unique()):
        subset = final.loc[final["architecture"] == architecture]
        for (mt, ms, block), group in subset.groupby(["mt", "ms", "block_size"]):
            records.extend(pair_group(group, "protocol", "online", "batch", metrics, f"online_minus_batch:{architecture}", mt, ms, block))
    for protocol in sorted(final["protocol"].unique()):
        subset = final.loc[final["protocol"] == protocol]
        for (mt, ms, block), group in subset.groupby(["mt", "ms", "block_size"]):
            records.extend(
                pair_group(
                    group,
                    "architecture",
                    "structured_joint_hippo_stvgp",
                    "matched_sparse_stvgp",
                    metrics,
                    f"structured_minus_matched:{protocol}",
                    mt,
                    ms,
                    block,
                )
            )
    return pd.DataFrame(records)


def pair_group(
    group: pd.DataFrame,
    column: str,
    left: str,
    right: str,
    metrics: list[str],
    comparison: str,
    mt: int,
    ms: int,
    block: int,
) -> list[dict[str, Any]]:
    records = []
    for metric in metrics:
        pivot = group.pivot_table(index="heldout_split_seed", columns=column, values=metric, aggfunc="first")
        if left not in pivot or right not in pivot:
            continue
        paired = pivot[[left, right]].dropna()
        if paired.empty:
            continue
        difference = paired[left].to_numpy() - paired[right].to_numpy()
        t_result = stats.ttest_rel(paired[left], paired[right]) if difference.size >= 2 else None
        sem = stats.sem(difference) if difference.size >= 2 else float("nan")
        half_ci = stats.t.ppf(0.975, difference.size - 1) * sem if difference.size >= 2 else float("nan")
        records.append(
            {
                "comparison": comparison,
                "mt": mt,
                "ms": ms,
                "block_size": block,
                "metric": metric,
                "num_pairs": int(difference.size),
                "mean_paired_difference": float(np.mean(difference)),
                "sd_paired_difference": float(np.std(difference, ddof=1)) if difference.size > 1 else 0.0,
                "ci95_low": float(np.mean(difference) - half_ci) if np.isfinite(half_ci) else float("nan"),
                "ci95_high": float(np.mean(difference) + half_ci) if np.isfinite(half_ci) else float("nan"),
                "paired_t_pvalue": float(t_result.pvalue) if t_result is not None else float("nan"),
            }
        )
    return records


def collect_p1(root: Path) -> pd.DataFrame:
    frames = [normalize_local_p1(path) for path in root.glob("Mt*_Ms*/block*/seed*/*/block_metrics.csv")]
    frames.extend(normalize_online_p1(path) for path in root.glob("Mt*_Ms*/block*/seed*/structured_joint_online/era5_routeb_metrics.csv"))
    if not frames:
        raise FileNotFoundError(f"No P1 metrics found under {root}")
    return pd.concat(frames, ignore_index=True)


def collect_p2(root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(root / "exact_decomposition_metrics.csv")]
    for seed_dir in sorted(root.glob("seed*")):
        matched_path = seed_dir / "matched_sparse_residual_batch" / "block_metrics.csv"
        if matched_path.exists():
            row = pd.read_csv(matched_path).iloc[-1].to_dict()
            frames[0] = pd.concat(
                [
                    frames[0],
                    pd.DataFrame(
                        [
                            {
                                "method": "Matched sparse residual (Mt=32, Ms=128)",
                                "heldout_split_seed": int(row["heldout_split_seed"]),
                                "rmse": row["rmse"],
                                "nll": row["nll"],
                                "coverage90": row["coverage90"],
                                "ece": row["ece"],
                                "avg_std": row["avg_std"],
                                "runtime_sec": row["block_incremental_runtime_sec"],
                                "num_test": row["num_test"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        online_path = seed_dir / "structured_joint_online" / "era5_routeb_metrics.csv"
        if online_path.exists():
            online = pd.read_csv(online_path)
            row = online.loc[online["eval_mode"] == "seen_history"].sort_values("block_id").iloc[-1]
            frames[0] = pd.concat(
                [
                    frames[0],
                    pd.DataFrame(
                        [
                            {
                                "method": "Online structured-joint HiPPO-STVGP",
                                "heldout_split_seed": int(row["heldout_split_seed"]),
                                "rmse": row["rmse"],
                                "nll": row["nll"],
                                "coverage90": row["coverage90"],
                                "ece": row["ece"],
                                "avg_std": row["avg_std"],
                                "runtime_sec": row.get("block_incremental_runtime", row["runtime_per_block"]),
                                "num_test": row["num_test"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    return frames[0]


def collect_p3(root: Path) -> pd.DataFrame:
    records = []
    for path in root.glob("Mt*_Ms*/seed*/*/run_metadata.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = dict(payload["final"])
        row["peak_rss_mb"] = parse_peak_rss(path.parent / "resource_usage.txt")
        records.append(row)
    return pd.DataFrame(records)


def plot_spatial_layout(outdir: Path, p1_root: Path) -> None:
    dataset = load_hipposvgp_era5("data/era5/processed_timeseries_4", tasks=("task_2",), variable_index=0, split="all")
    train, test = fixed_spatial_train_test_split(dataset.Y.shape[1], test_fraction=0.2, seed=0)
    metadata_path = p1_root / "Mt32_Ms128/block10/seed0/structured_joint_batch/run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    z_std = np.asarray(metadata["spatial_inducing_locations"], dtype=float)
    mean = dataset.coords.mean(axis=0)
    scale = np.maximum(dataset.coords.std(axis=0), 1e-8)
    z = z_std * scale + mean
    fig, ax = plt.subplots(figsize=(5.3, 4.0))
    ax.scatter(dataset.coords[train, 1], dataset.coords[train, 0], s=8, c=PALETTE["grey"], alpha=0.45, label="Observed locations")
    ax.scatter(dataset.coords[test, 1], dataset.coords[test, 0], s=13, c=PALETTE["red"], alpha=0.75, label="Held-out locations")
    ax.scatter(z[:, 1], z[:, 0], s=30, marker="x", linewidths=1.1, c=PALETTE["blue"], label="Spatial inducing points")
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Spatial split and inducing representation (seed 0)")
    ax.legend(loc="best", fontsize=7)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(outdir / "fig_spatial_split_inducing.pdf")
    fig.savefig(outdir / "fig_spatial_split_inducing.png")
    plt.close(fig)


def load_primary_pointwise(p1_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    online_path = p1_root / "Mt32_Ms128/block10/seed0/structured_joint_online/era5_routeb_per_location_predictions.csv"
    batch_path = p1_root / "Mt32_Ms128/block10/seed0/structured_joint_batch/final_pointwise_predictions.csv"
    online = pd.read_csv(online_path).rename(
        columns={"latitude": "lat", "longitude": "lon", "pred_var_y": "pred_var", "pred_std_y": "pred_std"}
    )
    batch = pd.read_csv(batch_path)
    online["pred_std"] = np.sqrt(np.maximum(online["pred_var"], 0.0))
    batch["pred_std"] = np.sqrt(np.maximum(batch["pred_var"], 0.0))
    return online, batch


def plot_final_spatial_maps(outdir: Path, p1_root: Path) -> None:
    online, _ = load_primary_pointwise(p1_root)
    final = online.loc[online["time_index"] == online["time_index"].max()].copy()
    lo = float(min(final["y_true"].min(), final["pred_mean"].min()))
    hi = float(max(final["y_true"].max(), final["pred_mean"].max()))
    err = final["pred_mean"] - final["y_true"]
    err_lim = float(max(abs(err.min()), abs(err.max())))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), constrained_layout=True)
    scatter0 = axes[0].scatter(final["lon"], final["lat"], c=final["y_true"], s=20, cmap="viridis", vmin=lo, vmax=hi)
    axes[1].scatter(final["lon"], final["lat"], c=final["pred_mean"], s=20, cmap="viridis", vmin=lo, vmax=hi)
    scatter2 = axes[2].scatter(final["lon"], final["lat"], c=err, s=20, cmap="RdBu_r", vmin=-err_lim, vmax=err_lim)
    axes[0].set_title("Truth")
    axes[1].set_title("Online prediction")
    axes[2].set_title("Prediction - truth")
    for ax in axes:
        ax.set(xlabel="Longitude", ylabel="Latitude")
        ax.set_aspect("equal", adjustable="box")
    fig.colorbar(scatter0, ax=axes[:2], shrink=0.78, label="Scaled target")
    fig.colorbar(scatter2, ax=axes[2], shrink=0.78, label="Error")
    fig.savefig(outdir / "fig_final_time_spatial_maps.pdf")
    fig.savefig(outdir / "fig_final_time_spatial_maps.png")
    plt.close(fig)


def plot_location_cases(outdir: Path, p1_root: Path) -> pd.DataFrame:
    online, batch = load_primary_pointwise(p1_root)
    scores = online.groupby("location_index").apply(
        lambda group: float(np.sqrt(np.mean((group["y_true"] - group["pred_mean"]) ** 2))),
        include_groups=False,
    ).sort_values()
    quantiles = {"Success (10th percentile)": 0.10, "Median (50th percentile)": 0.50, "Failure (90th percentile)": 0.90}
    selected = []
    for label, quantile in quantiles.items():
        position = int(round(quantile * (len(scores) - 1)))
        selected.append({"case": label, "location_index": int(scores.index[position]), "online_location_rmse": float(scores.iloc[position]), "quantile": quantile})
    selected_frame = pd.DataFrame(selected)
    selected_frame.to_csv(outdir / "location_case_selection.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(10.2, 7.2), sharex=True, constrained_layout=True)
    for row_index, item in selected_frame.iterrows():
        location = int(item["location_index"])
        for column, (name, frame, color) in enumerate(
            [("Online structured joint", online, PALETTE["blue"]), ("Batch structured joint", batch, PALETTE["green"])]
        ):
            group = frame.loc[frame["location_index"] == location].sort_values("time_index")
            ax = axes[row_index, column]
            x = group["time_index"].to_numpy()
            truth = group["y_true"].to_numpy()
            mean = group["pred_mean"].to_numpy()
            std = group["pred_std"].to_numpy()
            ax.plot(x, truth, color=PALETTE["charcoal"], lw=0.9, label="Ground truth")
            ax.plot(x, mean, color=color, lw=1.0, label="Predictive mean")
            ax.fill_between(x, mean - 1.6448536 * std, mean + 1.6448536 * std, color=color, alpha=0.18, linewidth=0, label="90% interval")
            ax.set_title(f"{item['case']} | location {location} | {name}")
            ax.set_ylabel("Scaled target")
            if row_index == 2:
                ax.set_xlabel("Time index")
            if row_index == 0:
                ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.savefig(outdir / "fig_success_median_failure_timeseries.pdf")
    fig.savefig(outdir / "fig_success_median_failure_timeseries.png")
    plt.close(fig)
    return selected_frame


def plot_online_batch_differences(outdir: Path, p1: pd.DataFrame) -> None:
    data = p1.loc[
        (p1["architecture"] == "structured_joint_hippo_stvgp") & (p1["block_size"] == 10)
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), constrained_layout=True)
    for (mt, ms), group in data.groupby(["mt", "ms"]):
        for metric, ax in [("rmse", axes[0]), ("nll", axes[1])]:
            pivot = group.pivot_table(index=["heldout_split_seed", "block_id"], columns="protocol", values=metric, aggfunc="first").dropna()
            delta = (pivot["online"] - pivot["batch"]).rename("delta").reset_index()
            summary = delta.groupby("block_id")["delta"].agg(["mean", "std"]).reset_index()
            x = summary["block_id"].to_numpy() + 1
            y = summary["mean"].to_numpy()
            sd = summary["std"].fillna(0.0).to_numpy()
            label = f"Mt={mt}, Ms={ms}"
            ax.plot(x, y, marker="o", ms=2.5, lw=1.0, label=label)
            ax.fill_between(x, y - sd, y + sd, alpha=0.14)
    axes[0].set(title="Online - batch RMSE", xlabel="Block", ylabel="Paired difference")
    axes[1].set(title="Online - batch NLL", xlabel="Block", ylabel="Paired difference")
    for ax in axes:
        ax.axhline(0.0, color=PALETTE["charcoal"], lw=0.7, ls="--")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(outdir / "fig_online_batch_block_differences.pdf")
    fig.savefig(outdir / "fig_online_batch_block_differences.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    aggregate_dir = root / "aggregated"
    figure_dir = root / "figures"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    p1 = collect_p1(root / "p1_fair_matrix")
    p1.to_csv(aggregate_dir / "p1_all_block_metrics.csv", index=False)
    final_summary = aggregate_mean_sd(
        p1.loc[p1["is_final_block"]],
        ["architecture", "protocol", "mt", "ms", "block_size"],
        ["rmse", "nll", "coverage90", "ece", "avg_std", "block_incremental_runtime_sec", "peak_rss_mb"],
    )
    final_summary.to_csv(aggregate_dir / "p1_final_block_summary.csv", index=False)
    per_seed_block_average = p1.groupby(
        ["architecture", "protocol", "mt", "ms", "block_size", "heldout_split_seed"], as_index=False
    )[["rmse", "nll", "coverage90", "block_incremental_runtime_sec"]].mean()
    block_average_summary = aggregate_mean_sd(
        per_seed_block_average,
        ["architecture", "protocol", "mt", "ms", "block_size"],
        ["rmse", "nll", "coverage90", "block_incremental_runtime_sec"],
    )
    block_average_summary.to_csv(aggregate_dir / "p1_block_average_summary.csv", index=False)
    paired_rows(p1).to_csv(aggregate_dir / "p1_paired_seed_comparisons.csv", index=False)

    p2 = collect_p2(root / "p2_residual_decomposition")
    p2.to_csv(aggregate_dir / "p2_all_seed_metrics.csv", index=False)
    aggregate_mean_sd(
        p2,
        ["method"],
        ["rmse", "nll", "coverage90", "ece", "avg_std", "runtime_sec"],
    ).to_csv(aggregate_dir / "p2_summary.csv", index=False)

    p3 = collect_p3(root / "p3_hippo_ablation")
    p3.to_csv(aggregate_dir / "p3_all_seed_metrics.csv", index=False)
    aggregate_mean_sd(
        p3,
        ["temporal_representation", "mt", "ms"],
        ["rmse", "nll", "coverage90", "ece", "avg_std", "block_incremental_runtime_sec", "peak_rss_mb"],
    ).to_csv(aggregate_dir / "p3_summary.csv", index=False)

    plot_spatial_layout(figure_dir, root / "p1_fair_matrix")
    plot_final_spatial_maps(figure_dir, root / "p1_fair_matrix")
    selected = plot_location_cases(figure_dir, root / "p1_fair_matrix")
    plot_online_batch_differences(figure_dir, p1)
    manifest = {
        "p1_rows": int(len(p1)),
        "p2_rows": int(len(p2)),
        "p3_rows": int(len(p3)),
        "location_selection": selected.to_dict(orient="records"),
    }
    (aggregate_dir / "aggregation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
