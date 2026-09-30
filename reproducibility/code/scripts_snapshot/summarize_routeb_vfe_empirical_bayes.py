#!/usr/bin/env python3
"""Compare finite/DTC and VFE-corrected Task-2 empirical-Bayes Route B."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison"
OLD = BASE / "phase_m_routeb_empirical_bayes/task2_empirical_bayes"
OUT = BASE / "phase_o_routeb_vfe_empirical_bayes"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configurations = [
        ("finite_dtc_100", OLD),
        ("vfe_100", OUT / "task2_vfe"),
        ("vfe_250", OUT / "task2_vfe_250"),
    ]
    for objective, root in configurations:
        for representation in ["analytic_hippo_rff", "inducing_points"]:
            for seed in range(3):
                payload = read_json(root / representation / f"seed{seed}/result.json")
                final = payload["final"]
                theta = payload["learned_theta"]
                rows.append(
                    {
                        "objective": objective,
                        "representation": representation,
                        "seed": seed,
                        "rmse": final["rmse"],
                        "nll": final["nll"],
                        "coverage90": final["coverage90"],
                        "ece": final["ece"],
                        "mean_predictive_std": final["mean_predictive_std"],
                        "best_iteration": payload["best_iteration"],
                        "best_validation_nll": payload["best_validation_nll"],
                        "process_seconds": payload["timing"]["process_total_seconds"],
                        "mean_iteration_seconds": payload["timing"]["mean_iteration_seconds"],
                        "time_to_best_seconds": payload["time_to_best_validation_seconds"],
                        "ell_t": theta["ell_t"],
                        "ell_s_1": theta["ell_s"][0],
                        "ell_s_2": theta["ell_s"][1],
                        "kernel_variance": theta["kernel_variance"],
                        "noise_std": theta["noise_std"],
                    }
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = [
        "rmse",
        "nll",
        "coverage90",
        "ece",
        "mean_predictive_std",
        "best_iteration",
        "process_seconds",
        "mean_iteration_seconds",
        "time_to_best_seconds",
        "ell_t",
        "ell_s_1",
        "ell_s_2",
        "kernel_variance",
        "noise_std",
    ]
    for objective in ["finite_dtc_100", "vfe_100", "vfe_250"]:
        for representation in ["analytic_hippo_rff", "inducing_points"]:
            selected = [
                row
                for row in rows
                if row["objective"] == objective
                and row["representation"] == representation
            ]
            result: dict[str, Any] = {
                "objective": objective,
                "representation": representation,
                "n": len(selected),
            }
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in selected])
                result[f"{metric}_mean"] = float(values.mean())
                result[f"{metric}_sd"] = float(values.std(ddof=1))
            output.append(result)
    return output


def paired_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for vfe_objective in ["vfe_100", "vfe_250"]:
        for representation in ["analytic_hippo_rff", "inducing_points"]:
            for seed in range(3):
                finite = next(
                    row for row in rows
                    if row["objective"] == "finite_dtc_100"
                    and row["representation"] == representation
                    and row["seed"] == seed
                )
                vfe = next(
                    row for row in rows
                    if row["objective"] == vfe_objective
                    and row["representation"] == representation
                    and row["seed"] == seed
                )
                output.append(
                    {
                        "vfe_objective": vfe_objective,
                        "representation": representation,
                        "seed": seed,
                        "rmse_vfe_minus_finite": vfe["rmse"] - finite["rmse"],
                        "nll_vfe_minus_finite": vfe["nll"] - finite["nll"],
                        "coverage90_vfe_minus_finite": vfe["coverage90"] - finite["coverage90"],
                        "noise_std_vfe_minus_finite": vfe["noise_std"] - finite["noise_std"],
                    }
                )
    return output


def plot_comparison(summary: list[dict[str, Any]]) -> None:
    labels = ["HiPPO-RFF", "Ordinary inducing"]
    representations = ["analytic_hippo_rff", "inducing_points"]
    colors = {
        "finite_dtc_100": "#697A86",
        "vfe_100": "#E07A2D",
        "vfe_250": "#9C2F1B",
    }
    names = {
        "finite_dtc_100": "Finite/DTC, 100 steps",
        "vfe_100": "VFE, 100 steps",
        "vfe_250": "VFE, 250 steps",
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.45))
    x = np.arange(2)
    width = 0.24
    for offset, objective in [
        (-width, "finite_dtc_100"),
        (0.0, "vfe_100"),
        (width, "vfe_250"),
    ]:
        selected = [
            next(
                row for row in summary
                if row["objective"] == objective and row["representation"] == representation
            )
            for representation in representations
        ]
        for axis, metric, ylabel in zip(
            axes,
            ["rmse", "nll", "coverage90"],
            ["Test RMSE", "Test NLL", "90% coverage"],
        ):
            axis.bar(
                x + offset,
                [row[f"{metric}_mean"] for row in selected],
                width,
                yerr=[row[f"{metric}_sd"] for row in selected],
                color=colors[objective],
                label=names[objective],
                capsize=3,
            )
            axis.set_ylabel(ylabel)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=10, ha="right")
        axis.grid(axis="y", color="#D9E1E5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "vfe_finite_empirical_bayes_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT / "vfe_finite_empirical_bayes_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_vfe_trajectories() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    colors = {"analytic_hippo_rff": "#2A6F97", "inducing_points": "#C44900"}
    names = {"analytic_hippo_rff": "HiPPO-RFF", "inducing_points": "Ordinary inducing"}
    metrics = [
        ("train_nlml_per_observation", "Negative VFE bound / obs."),
        ("vfe_trace_correction_per_observation", "VFE correction / obs."),
        ("noise_std", "Noise standard deviation"),
    ]
    for representation in ["analytic_hippo_rff", "inducing_points"]:
        path = OUT / f"task2_vfe_250/{representation}/seed0/training_trace.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        iterations = np.asarray([int(row["iteration"]) for row in rows])
        for axis, (metric, label) in zip(axes, metrics):
            values = np.asarray([float(row[metric]) for row in rows])
            axis.plot(
                iterations,
                values,
                color=colors[representation],
                linewidth=1.8,
                label=names[representation],
            )
            axis.set_ylabel(label)
            axis.set_xlabel("Adam iteration")
            axis.grid(color="#D9E1E5", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "vfe_training_trajectories.pdf", bbox_inches="tight")
    fig.savefig(OUT / "vfe_training_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = collect()
    summary = summarize(rows)
    paired = paired_differences(rows)
    write_csv(rows, OUT / "vfe_finite_per_seed.csv")
    write_csv(summary, OUT / "vfe_finite_summary.csv")
    write_csv(paired, OUT / "vfe_finite_paired_differences.csv")
    plot_comparison(summary)
    plot_vfe_trajectories()
    print(json.dumps({"summary": summary, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
