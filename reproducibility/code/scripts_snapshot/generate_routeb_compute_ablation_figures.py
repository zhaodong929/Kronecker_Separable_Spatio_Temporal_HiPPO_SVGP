#!/usr/bin/env python3
"""Create figures and audit summaries for the Route B compute appendix."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison"
OUT = BASE / "phase_n_routeb_compute_and_model_ablation"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_solver() -> None:
    rows = read_csv(OUT / "structured_dense_solver_benchmark.csv")
    sizes = np.asarray([int(row["Ms"]) for row in rows])
    dense = np.asarray([float(row["dense_time_s"]) for row in rows])
    structured = np.asarray([float(row["sylvester_time_s"]) for row in rows])
    speedup = dense / structured
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    axes[0].plot(sizes, dense, "o-", color="#C43C4E", linewidth=2, label="Explicit dense")
    axes[0].plot(sizes, structured, "o-", color="#267365", linewidth=2, label="Kronecker/Sylvester")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("$M_t=M_s$")
    axes[0].set_ylabel("Solve time (s, log scale)")
    axes[0].legend(frameon=False)
    axes[1].plot(sizes, speedup, "o-", color="#2A6F97", linewidth=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("$M_t=M_s$")
    axes[1].set_ylabel("Dense / structured speedup")
    for axis in axes:
        axis.grid(color="#D9E1E5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "structured_dense_solver_scaling.pdf", bbox_inches="tight")
    fig.savefig(OUT / "structured_dense_solver_scaling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stages() -> None:
    rows = json.loads((OUT / "routeb_stage_timing.json").read_text(encoding="utf-8"))
    labels = ["HiPPO\njoint", "Ordinary\njoint", "Ordinary\nzero mean"]
    stages = [
        ("factor", "Factor construction", "#4C78A8"),
        ("structured_objective", "Structured objective", "#F2A541"),
        ("backward", "Backward", "#C43C4E"),
        ("adam", "Adam", "#267365"),
    ]
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    bottom = np.zeros(len(rows))
    x = np.arange(len(rows))
    for key, name, color in stages:
        values = np.asarray([float(row[f"{key}_seconds_mean"]) for row in rows])
        axis.bar(x, values, bottom=bottom, color=color, label=name)
        bottom += values
    axis.set_xticks(x, labels)
    axis.set_ylabel("Seconds per optimization step")
    axis.grid(axis="y", color="#D9E1E5", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "routeb_stage_timing.pdf", bbox_inches="tight")
    fig.savefig(OUT / "routeb_stage_timing.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def official_partial_trajectory() -> list[dict[str, float]]:
    log_path = OUT / "accuracy_trajectory/official_seed0/stdout.log"
    text = log_path.read_text(encoding="utf-8")
    rows = [
        {
            "iteration": int(match.group(1)),
            "rmse": float(match.group(2)),
            "nll": float(match.group(3)),
            "source": "instrumented_partial",
        }
        for match in re.finditer(
            r"trajectory iter (\d+): rmse ([^ ]+) nll ([^\n]+)", text
        )
    ]
    historical = json.loads(
        (
            BASE
            / "phase_d_joint_xlag_controlled/seed0/official_st_svgp_Ms128/result.json"
        ).read_text(encoding="utf-8")
    )
    rows.append(
        {
            "iteration": 100,
            "rmse": float(historical["rmse"]),
            "nll": float(historical["nll"]),
            "source": "historical_completed_run",
        }
    )
    return rows


def plot_official_trajectory(rows: list[dict[str, float]]) -> None:
    partial = [row for row in rows if row["source"] == "instrumented_partial"]
    historical = next(row for row in rows if row["source"] == "historical_completed_run")
    iteration = np.asarray([row["iteration"] for row in partial])
    rmse = np.asarray([row["rmse"] for row in partial])
    nll = np.asarray([row["nll"] for row in partial])
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    axes[0].plot(iteration, rmse, "o-", color="#2A6F97", linewidth=2, label="Instrumented run")
    axes[0].scatter([historical["iteration"]], [historical["rmse"]], marker="*", s=130, color="#1F1F1F", label="Historical completed run", zorder=4)
    axes[0].axhline(0.09, color="#7A7A7A", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Test RMSE")
    axes[1].plot(iteration, nll, "o-", color="#C43C4E", linewidth=2, label="Instrumented run")
    axes[1].scatter([historical["iteration"]], [historical["nll"]], marker="*", s=130, color="#1F1F1F", label="Historical completed run", zorder=4)
    axes[1].axhline(-0.90, color="#7A7A7A", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Test NLL")
    for axis in axes:
        axis.set_xlabel("Official optimizer iteration")
        axis.grid(color="#D9E1E5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "official_partial_accuracy_trajectory.pdf", bbox_inches="tight")
    fig.savefig(OUT / "official_partial_accuracy_trajectory.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_solver()
    plot_stages()
    trajectory = official_partial_trajectory()
    plot_official_trajectory(trajectory)
    dense_128_bytes = (128 * 128) ** 2 * 8
    payload = {
        "hardware": {
            "cpu": "AMD Ryzen 9 7845HX, 12 cores / 24 threads",
            "wsl_memory_gib": 15.0,
            "routeb_benchmark_threads": 8,
            "solver_benchmark_threads": 1,
            "gpu": "not used; installed CUDA driver incompatible with current PyTorch build",
        },
        "official_partial_trajectory": trajectory,
        "common_quality_rule": {"rmse_at_most": 0.09, "nll_at_most": -0.90},
        "official_time_to_common_quality": None,
        "official_time_status": (
            "Unavailable. Historical run did not record checkpoint timing; instrumented rerun "
            "was OOM-killed after iteration 50 at approximately 15.4 GiB RSS."
        ),
        "dense_128_precision_gib": dense_128_bytes / (1024.0**3),
        "dense_128_cubic_proxy": float((128 * 128) ** 3),
        "structured_128_proxy": float(4 * 128**3),
    }
    (OUT / "compute_audit_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
