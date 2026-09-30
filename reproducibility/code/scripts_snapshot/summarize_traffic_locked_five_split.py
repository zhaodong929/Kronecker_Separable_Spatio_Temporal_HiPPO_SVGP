#!/usr/bin/env python3
"""Audit and summarize the frozen PEMS-BAY five-split experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/traffic/formal_locked_sm_q2_road_context_v1"


def source(method: str, seed: int) -> Path:
    if method == "persistence":
        return ROOT / f"results/traffic/formal_deterministic_v1/pems_bay/main/nowcast/persistence/seed{seed}"
    if seed == 0:
        return {
            "kronhippo_stgp": ROOT / "results/traffic/seed0_xlag_v1/final_strict_online_seed0",
            "kron_stgp": ROOT / "results/traffic/seed0_frozen_baselines_v1/kron_stgp",
            "joint_fixed_global": ROOT / "results/traffic/seed0_frozen_baselines_v1/joint_fixed_global",
            "decoupled_fixed_global": ROOT / "results/traffic/seed0_frozen_baselines_v1/decoupled_fixed_global",
            "ignnk": ROOT / "results/traffic/seed0_external_baselines_v1/ignnk",
        }[method]
    return OUTPUT / "pems_bay" / "nowcast" / method / f"seed{seed}"


def load_routeb(method: str, seed: int) -> dict[str, object]:
    path = source(method, seed)
    payload = json.loads((path / "result.json").read_text(encoding="utf-8"))
    archive = np.load(path / "predictions.npz")
    final = payload["result"]["final"]
    guard = payload["result"]["guard"]
    expected = (50_100, 65)
    passed = (
        np.asarray(archive["y"]).shape == expected
        and np.asarray(archive["mean"]).shape == expected
        and np.asarray(archive["variance"]).shape == expected
        and np.all(np.isfinite(archive["y"]))
        and np.all(np.isfinite(archive["mean"]))
        and np.all(np.isfinite(archive["variance"]))
        and np.all(np.asarray(archive["variance"]) > 0.0)
        and guard["current_hidden_reads_before_prediction"] == 0
        and guard["current_hidden_reveals"] == expected[0]
        and guard["unique_delayed_hidden_absorptions"] == expected[0] - 1
    )
    return {
        "method": method,
        "seed": seed,
        "rmse": final["rmse"],
        "rmse_mph": final["rmse_speed"],
        "crps": final["crps"],
        "gaussian_nlpd": final["gaussian_nlpd"],
        "ece": final["ece"],
        "coverage90": final["coverage90"],
        "audit": "passed" if passed else "failed",
        "result_path": str(path.relative_to(ROOT)),
    }


def load_ignnk(seed: int) -> dict[str, object]:
    path = source("ignnk", seed)
    payload = json.loads((path / "result.json").read_text(encoding="utf-8"))
    archive = np.load(path / "predictions.npz")
    expected = (50_100, 65)
    passed = (
        np.asarray(archive["y"]).shape == expected
        and np.asarray(archive["mean"]).shape == expected
        and np.all(np.isfinite(archive["mean"]))
        and payload["causal_audit"]["current_hidden_reads_before_prediction"] == 0
        and payload["causal_audit"]["heldout_history_latest_available_lag"] == 1
    )
    return {
        "method": "ignnk",
        "seed": seed,
        "rmse": payload["metrics"]["rmse"],
        "rmse_mph": payload["metrics"]["rmse_speed"],
        "mae": payload["metrics"]["mae"],
        "mae_mph": payload["metrics"]["mae_speed"],
        "audit": "passed" if passed else "failed",
        "result_path": str(path.relative_to(ROOT)),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for method in dict.fromkeys(str(row["method"]) for row in rows):
        subset = [row for row in rows if row["method"] == method]
        item: dict[str, object] = {"method": method, "n": len(subset)}
        for metric in ("rmse", "rmse_mph", "crps", "gaussian_nlpd", "ece", "coverage90", "mae", "mae_mph"):
            values = [float(row[metric]) for row in subset if metric in row]
            if values:
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    seed_tag = "_".join(str(seed) for seed in seeds)

    labels = {
        "persistence": "Delayed-target last-value persistence",
        "kron_stgp": "Kron-STGP (point temporal)",
        "kronhippo_stgp": "KronHiPPO-STGP (joint + changing)",
        "joint_fixed_global": "Joint + fixed-global",
        "decoupled_fixed_global": "Decoupled + fixed-global",
        "ignnk": "IGNNK",
    }
    routeb_methods = (
        "persistence",
        "kron_stgp",
        "kronhippo_stgp",
        "joint_fixed_global",
        "decoupled_fixed_global",
    )
    rows = [load_routeb(method, seed) for method in routeb_methods for seed in seeds]
    rows.extend(load_ignnk(seed) for seed in seeds)
    if any(row["audit"] != "passed" for row in rows):
        raise SystemExit("At least one archive failed the common audit")
    summary = aggregate(rows)
    write_csv(rows, OUTPUT / f"per_seed_results_{seed_tag}.csv")
    write_csv(summary, OUTPUT / f"aggregate_results_{seed_tag}.csv")

    lines = [
        f"# PEMS-BAY Frozen Results: Seeds {', '.join(str(seed) for seed in seeds)}",
        "",
        "Task-1 visible-validation selected SM-Q2 + Road-context L10; ell_t=0.5 h, Ms=32, Mt=128 and RFF=512. Continuous hyperparameters and the mean-feature scaler are fit within each split's Task-1 calibration sensors and then frozen for the online stream.",
        "",
        "Protocol N allows every method to use legally revealed held-out labels through t-1; no method reads y_H,t before prediction. Persistence is therefore a fair delayed-target control, not a no-hidden-history baseline.",
        "",
        "| Method | RMSE | RMSE (mph) | CRPS | Gaussian NLPD | ECE | Coverage90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def cell(metric: str) -> str:
            if f"{metric}_mean" not in row:
                return "N/A"
            return f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_sd']:.4f}"
        lines.append(
            f"| {labels[str(row['method'])]} | {cell('rmse')} | {cell('rmse_mph')} | "
            f"{cell('crps')} | {cell('gaussian_nlpd')} | {cell('ece')} | {cell('coverage90')} |"
        )
    lines.extend(
        [
            "",
            "The changing versus fixed-global comparison changes temporal coordinates as well as the transfer path. It supports an association with the changing-coordinate HiPPO representation, not a standalone causal estimate of transfer approximation benefit.",
            "",
            "Decoupled + changing is excluded from the competitive table and retained as a stability diagnostic with a predeclared per-step RMSE > 5 stopping rule.",
        ]
    )
    (OUTPUT / f"SEEDS_{seed_tag}_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plotted = [row for row in rows if row["method"] != "persistence"]
    order = ["kronhippo_stgp", "ignnk", "kron_stgp", "joint_fixed_global", "decoupled_fixed_global"]
    fig, axis = plt.subplots(figsize=(7.2, 3.4))
    for position, method in enumerate(order):
        values = [float(row["rmse"]) for row in plotted if row["method"] == method]
        axis.scatter(np.full(len(values), position), values, s=22, alpha=0.75)
        axis.plot(position, np.mean(values), marker="_", markersize=18, color="black")
    axis.set_xticks(range(len(order)), [labels[name].replace(" (joint + changing)", "").replace(" (point temporal)", "") for name in order], rotation=18, ha="right")
    axis.set_ylabel("Standardised RMSE")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"fig_seeds_{seed_tag}_rmse.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"fig_seeds_{seed_tag}_rmse.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
