#!/usr/bin/env python3
"""Aggregate Route B compute and mean/residual ablations."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hipposvgp_era5_routeb import coverage90, gaussian_nll
from scripts.run_routeb_batch_empirical_bayes import load_controlled_grid


BASE = ROOT / "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison"
OUT = BASE / "phase_n_routeb_compute_and_model_ablation"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def linear_only(seed: int, ridge: float = 1e-3) -> dict[str, Any]:
    started = time.perf_counter()
    data = load_controlled_grid(
        root="data/era5/processed_timeseries_4",
        task="task_2",
        controlled_npz=BASE / f"phase_d_joint_xlag_controlled/seed{seed}/era5_xlag_seed{seed}.npz",
        ms=128,
        xlag_length=10,
    )
    x_train = data.phi[:, data.train_indices].reshape(-1, data.phi.shape[-1])
    y_train = data.y[:, data.train_indices].reshape(-1)
    precision = x_train.T @ x_train + ridge * np.eye(x_train.shape[1])
    beta = np.linalg.solve(precision, x_train.T @ y_train)
    residual = y_train - x_train @ beta
    degrees = max(y_train.size - x_train.shape[1], 1)
    noise_variance = float(residual @ residual / degrees)
    beta_covariance = noise_variance * np.linalg.solve(
        precision, np.eye(precision.shape[0])
    )
    x_test = data.phi[:, data.test_indices].reshape(-1, data.phi.shape[-1])
    y_test = data.y[:, data.test_indices].reshape(-1)
    mean = x_test @ beta
    variance = noise_variance + np.einsum(
        "ij,jk,ik->i", x_test, beta_covariance, x_test
    )
    return {
        "method": "X-lag linear only",
        "seed": seed,
        "rmse": float(np.sqrt(np.mean((y_test - mean) ** 2))),
        "nll": gaussian_nll(y_test, mean, variance),
        "coverage90": coverage90(y_test, mean, variance),
        "runtime_seconds": time.perf_counter() - started,
        "noise_std": float(np.sqrt(noise_variance)),
        "ridge": ridge,
    }


def routeb_row(method: str, path: Path, seed: int) -> dict[str, Any]:
    result = read_json(path)
    final = result["final"]
    return {
        "method": method,
        "seed": seed,
        "rmse": final["rmse"],
        "nll": final["nll"],
        "coverage90": final["coverage90"],
        "runtime_seconds": result["timing"]["process_total_seconds"],
        "noise_std": result["learned_theta"]["noise_std"],
        "ell_t": result["learned_theta"]["ell_t"],
        "ell_s_1": result["learned_theta"]["ell_s"][0],
        "ell_s_2": result["learned_theta"]["ell_s"][1],
        "kernel_variance": result["learned_theta"]["kernel_variance"],
        "best_iteration": result["best_iteration"],
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        summary: dict[str, Any] = {"method": method, "n": len(selected)}
        for metric in ["rmse", "nll", "coverage90", "runtime_seconds"]:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def plot(summary: list[dict[str, Any]]) -> None:
    labels = [row["method"] for row in summary]
    display = ["Linear only", "Pure GP", "Two-stage", "Joint Route B"]
    colors = ["#7A7A7A", "#4C78A8", "#F2A541", "#C43C4E"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    x = np.arange(len(labels))
    for axis, metric, ylabel in zip(axes, ["rmse", "nll"], ["Test RMSE", "Test NLL"]):
        axis.bar(
            x,
            [row[f"{metric}_mean"] for row in summary],
            yerr=[row[f"{metric}_sd"] for row in summary],
            color=colors,
            capsize=3,
        )
        axis.set_xticks(x, display, rotation=15, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9E1E5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "routeb_mean_residual_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "routeb_mean_residual_ablation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in range(3):
        rows.append(linear_only(seed))
        rows.append(
            routeb_row(
                "Pure GP (zero mean)", OUT / f"zero/seed{seed}/result.json", seed
            )
        )
        rows.append(
            routeb_row(
                "X-lag + two-stage residual GP",
                OUT / f"residual_xlag/seed{seed}/result.json",
                seed,
            )
        )
        rows.append(
            routeb_row(
                "Joint X-lag + residual Route B",
                BASE
                / f"phase_m_routeb_empirical_bayes/task2_empirical_bayes/inducing_points/seed{seed}/result.json",
                seed,
            )
        )
    summaries = summarize(rows)
    write_csv(rows, OUT / "mean_residual_ablation_per_seed.csv")
    write_csv(summaries, OUT / "mean_residual_ablation_summary.csv")
    plot(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
