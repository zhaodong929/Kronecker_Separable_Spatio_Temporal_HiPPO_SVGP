#!/usr/bin/env python3
"""Aggregate Route B empirical-Bayes accuracy and compute measurements."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison"
OUT = BASE / "phase_m_routeb_empirical_bayes"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resource_values(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    elapsed_match = re.search(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", text)
    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if elapsed_match is None or rss_match is None:
        return float("nan"), float("nan")
    parts = [float(item) for item in elapsed_match.group(1).split(":")]
    if len(parts) == 2:
        elapsed = 60.0 * parts[0] + parts[1]
    else:
        elapsed = 3600.0 * parts[0] + 60.0 * parts[1] + parts[2]
    return elapsed, float(rss_match.group(1)) / (1024.0**2)


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def shared_official_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for representation in ["analytic_hippo_rff", "inducing_points"]:
        for seed in range(3):
            run = BASE / f"phase_l_temporal_representation_sweep/Ms128/Mt128/{representation}/seed{seed}"
            metadata = read_json(run / "run_metadata.json")
            external_wall, peak_gib = resource_values(run / "resource_usage.txt")
            final = metadata["final"]
            rows.append(
                {
                    "strategy": "shared_official_theta",
                    "representation": representation,
                    "seed": seed,
                    "rmse": final["rmse"],
                    "nll": final["nll"],
                    "coverage90": final["coverage90"],
                    "process_wall_seconds": external_wall,
                    "posterior_setup_seconds": final["update_runtime_sec"],
                    "prediction_seconds": final["prediction_runtime_sec"],
                    "peak_rss_gib": peak_gib,
                    "time_to_best_validation_seconds": float("nan"),
                    "mean_iteration_seconds": float("nan"),
                    "persistent_state_mib": float("nan"),
                    "parameter_source": "official ST-SVGP fitted on Task 2",
                }
            )
    return rows


def empirical_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol in ["task1_calibration_then_freeze", "task2_empirical_bayes"]:
        for representation in ["analytic_hippo_rff", "inducing_points"]:
            for seed in range(3):
                run = OUT / protocol / representation / f"seed{seed}"
                result = read_json(run / "result.json")
                external_wall, peak_gib = resource_values(run / "resource_usage.txt")
                final = result["final"]
                theta = result["learned_theta"]
                rows.append(
                    {
                        "strategy": protocol,
                        "representation": representation,
                        "seed": seed,
                        "rmse": final["rmse"],
                        "nll": final["nll"],
                        "coverage90": final["coverage90"],
                        "process_wall_seconds": external_wall,
                        "posterior_setup_seconds": final["posterior_update_seconds"],
                        "prediction_seconds": final["prediction_seconds"],
                        "peak_rss_gib": peak_gib,
                        "time_to_best_validation_seconds": result["time_to_best_validation_seconds"],
                        "mean_iteration_seconds": result["timing"]["mean_iteration_seconds"],
                        "persistent_state_mib": result["resources"]["persistent_model_state_mib"],
                        "training_checkpoint_mib": result["resources"]["serialized_training_checkpoint_mib"],
                        "best_iteration": result["best_iteration"],
                        "ell_t": theta["ell_t"],
                        "ell_s_lat": theta["ell_s"][0],
                        "ell_s_lon": theta["ell_s"][1],
                        "kernel_variance": theta["kernel_variance"],
                        "noise_std": theta["noise_std"],
                        "parameter_source": "Route B Task 1" if protocol.startswith("task1") else "Route B Task 2",
                    }
                )
    return rows


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy in ["shared_official_theta", "task1_calibration_then_freeze", "task2_empirical_bayes"]:
        for representation in ["analytic_hippo_rff", "inducing_points"]:
            selected = [row for row in rows if row["strategy"] == strategy and row["representation"] == representation]
            row: dict[str, Any] = {
                "strategy": strategy,
                "representation": representation,
                "n": len(selected),
            }
            for metric in [
                "rmse",
                "nll",
                "coverage90",
                "process_wall_seconds",
                "posterior_setup_seconds",
                "prediction_seconds",
                "peak_rss_gib",
                "time_to_best_validation_seconds",
                "mean_iteration_seconds",
                "persistent_state_mib",
            ]:
                values = [float(item[metric]) for item in selected if np.isfinite(float(item[metric]))]
                if values:
                    row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(values)
                else:
                    row[f"{metric}_mean"] = float("nan")
                    row[f"{metric}_sd"] = float("nan")
            output.append(row)
    return output


def official_end_to_end_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in range(3):
        run = BASE / f"phase_d_joint_xlag_controlled/seed{seed}/official_st_svgp_Ms128"
        result = read_json(run / "result.json")
        external_wall, peak_gib = resource_values(run / "resource_usage.txt")
        rows.append(
            {
                "method": "official_st_svgp_xlag",
                "seed": seed,
                "rmse": result["rmse"],
                "nll": result["nll"],
                "process_wall_seconds": external_wall,
                "training_seconds": result["train_seconds"],
                "mean_iteration_seconds": result["train_seconds"] / result["iterations"],
                "time_to_best_validation_seconds": float("nan"),
                "prediction_seconds": float("nan"),
                "peak_rss_gib": peak_gib,
                "persistent_state_mib": float("nan"),
                "note": "Historical run did not checkpoint validation timing, isolated inference, or posterior-state bytes",
            }
        )
    return rows


def compute_table_rows(
    parameter_rows: list[dict[str, Any]], official_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    routeb_flops = {
        row["method"]: float(row["flops"])
        for row in read_json(OUT / "routeb_flop_profile.json")
    }
    official_flops = float(
        read_json(OUT / "official_stsvgp_hlo_cost_seed0.json")["xla_cost_analysis"]["flops"]
    )
    posterior_only: list[dict[str, Any]] = []
    for representation in ["analytic_hippo_rff", "inducing_points"]:
        selected = [
            row
            for row in parameter_rows
            if row["strategy"] == "shared_official_theta"
            and row["representation"] == representation
        ]
        posterior_only.append(
            {
                "method": "Route B HiPPO-RFF" if representation == "analytic_hippo_rff" else "Route B ordinary inducing",
                "parameter_condition": "fixed shared official theta",
                "posterior_setup_seconds_mean": mean_sd([row["posterior_setup_seconds"] for row in selected])[0],
                "posterior_setup_seconds_sd": mean_sd([row["posterior_setup_seconds"] for row in selected])[1],
                "prediction_seconds_mean": mean_sd([row["prediction_seconds"] for row in selected])[0],
                "prediction_seconds_sd": mean_sd([row["prediction_seconds"] for row in selected])[1],
                "process_wall_seconds_mean": mean_sd([row["process_wall_seconds"] for row in selected])[0],
                "process_wall_seconds_sd": mean_sd([row["process_wall_seconds"] for row in selected])[1],
                "peak_rss_gib_mean": mean_sd([row["peak_rss_gib"] for row in selected])[0],
                "peak_rss_gib_sd": mean_sd([row["peak_rss_gib"] for row in selected])[1],
                "note": "No hyperparameter-learning time included",
            }
        )
    posterior_only.insert(
        0,
        {
            "method": "Official ST-SVGP",
            "parameter_condition": "trained official posterior",
            "posterior_setup_seconds_mean": float("nan"),
            "posterior_setup_seconds_sd": float("nan"),
            "prediction_seconds_mean": float("nan"),
            "prediction_seconds_sd": float("nan"),
            "process_wall_seconds_mean": float("nan"),
            "process_wall_seconds_sd": float("nan"),
            "peak_rss_gib_mean": mean_sd([row["peak_rss_gib"] for row in official_rows])[0],
            "peak_rss_gib_sd": mean_sd([row["peak_rss_gib"] for row in official_rows])[1],
            "note": "Isolated inference unavailable: historical runs did not save a reloadable variational posterior",
        },
    )

    end_to_end: list[dict[str, Any]] = []
    end_to_end.append(
        {
            "method": "Official ST-SVGP + X-lag",
            "fit_task": "Task 2",
            "rmse_mean": mean_sd([row["rmse"] for row in official_rows])[0],
            "rmse_sd": mean_sd([row["rmse"] for row in official_rows])[1],
            "nll_mean": mean_sd([row["nll"] for row in official_rows])[0],
            "nll_sd": mean_sd([row["nll"] for row in official_rows])[1],
            "total_wall_seconds_mean": mean_sd([row["process_wall_seconds"] for row in official_rows])[0],
            "total_wall_seconds_sd": mean_sd([row["process_wall_seconds"] for row in official_rows])[1],
            "iteration_seconds_mean": mean_sd([row["mean_iteration_seconds"] for row in official_rows])[0],
            "iteration_seconds_sd": mean_sd([row["mean_iteration_seconds"] for row in official_rows])[1],
            "time_to_best_seconds_mean": float("nan"),
            "time_to_best_seconds_sd": float("nan"),
            "prediction_seconds_mean": float("nan"),
            "prediction_seconds_sd": float("nan"),
            "peak_rss_gib_mean": mean_sd([row["peak_rss_gib"] for row in official_rows])[0],
            "peak_rss_gib_sd": mean_sd([row["peak_rss_gib"] for row in official_rows])[1],
            "persistent_state_mib_mean": float("nan"),
            "persistent_state_mib_sd": float("nan"),
            "flops_per_iteration": official_flops,
            "flop_scope": "XLA HLO: one Bayes-Newton, energy-gradient, and Adam train_op",
            "note": "100 official iterations; unavailable fields were not recorded",
        }
    )
    for strategy, fit_task in [
        ("task1_calibration_then_freeze", "Task 1, then freeze"),
        ("task2_empirical_bayes", "Task 2"),
    ]:
        for representation in ["analytic_hippo_rff", "inducing_points"]:
            selected = [
                row
                for row in parameter_rows
                if row["strategy"] == strategy and row["representation"] == representation
            ]
            end_to_end.append(
                {
                    "method": "Route B HiPPO-RFF" if representation == "analytic_hippo_rff" else "Route B ordinary inducing",
                    "fit_task": fit_task,
                    "rmse_mean": mean_sd([row["rmse"] for row in selected])[0],
                    "rmse_sd": mean_sd([row["rmse"] for row in selected])[1],
                    "nll_mean": mean_sd([row["nll"] for row in selected])[0],
                    "nll_sd": mean_sd([row["nll"] for row in selected])[1],
                    "total_wall_seconds_mean": mean_sd([row["process_wall_seconds"] for row in selected])[0],
                    "total_wall_seconds_sd": mean_sd([row["process_wall_seconds"] for row in selected])[1],
                    "iteration_seconds_mean": mean_sd([row["mean_iteration_seconds"] for row in selected])[0],
                    "iteration_seconds_sd": mean_sd([row["mean_iteration_seconds"] for row in selected])[1],
                    "time_to_best_seconds_mean": mean_sd([row["time_to_best_validation_seconds"] for row in selected])[0],
                    "time_to_best_seconds_sd": mean_sd([row["time_to_best_validation_seconds"] for row in selected])[1],
                    "prediction_seconds_mean": mean_sd([row["prediction_seconds"] for row in selected])[0],
                    "prediction_seconds_sd": mean_sd([row["prediction_seconds"] for row in selected])[1],
                    "peak_rss_gib_mean": mean_sd([row["peak_rss_gib"] for row in selected])[0],
                    "peak_rss_gib_sd": mean_sd([row["peak_rss_gib"] for row in selected])[1],
                    "persistent_state_mib_mean": mean_sd([row["persistent_state_mib"] for row in selected])[0],
                    "persistent_state_mib_sd": mean_sd([row["persistent_state_mib"] for row in selected])[1],
                    "flops_per_iteration": routeb_flops[
                        "Route B analytic_hippo_rff"
                        if representation == "analytic_hippo_rff"
                        else "Route B inducing_points"
                    ],
                    "flop_scope": "PyTorch supported ATen ops: one exact-E-step forward and backward",
                    "note": "100 Adam iterations with exact Gaussian posterior at every step",
                }
            )
    return posterior_only, end_to_end


def plot_strategy(summary: list[dict[str, Any]]) -> None:
    labels = ["Shared official $\\theta$", "Task-1 freeze", "Task-2 EB"]
    strategy_keys = ["shared_official_theta", "task1_calibration_then_freeze", "task2_empirical_bayes"]
    colors = {"analytic_hippo_rff": "#2A6F97", "inducing_points": "#D1495B"}
    names = {"analytic_hippo_rff": "Analytic HiPPO-RFF", "inducing_points": "Ordinary inducing"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    x = np.arange(3)
    width = 0.34
    for offset, representation in [(-width / 2, "analytic_hippo_rff"), (width / 2, "inducing_points")]:
        selected = [next(row for row in summary if row["strategy"] == key and row["representation"] == representation) for key in strategy_keys]
        for axis, metric in zip(axes, ["rmse", "nll"]):
            axis.bar(
                x + offset,
                [row[f"{metric}_mean"] for row in selected],
                width,
                yerr=[row[f"{metric}_sd"] for row in selected],
                color=colors[representation],
                label=names[representation],
                capsize=3,
            )
    axes[0].set_ylabel("Test RMSE")
    axes[1].set_ylabel("Test NLL")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=12, ha="right")
        axis.grid(axis="y", color="#D9E1E5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "empirical_bayes_strategy_performance.pdf", bbox_inches="tight")
    fig.savefig(OUT / "empirical_bayes_strategy_performance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6))
    metrics = [
        ("validation_nll", "Validation NLL"),
        ("ell_t", "$\\ell_t$"),
        ("ell_s_0", "$\\ell_{s,1}$"),
        ("ell_s_1", "$\\ell_{s,2}$"),
        ("kernel_variance", "$\\sigma_f^2$"),
        ("noise_std", "$\\sigma_y$"),
    ]
    colors = {"analytic_hippo_rff": "#2A6F97", "inducing_points": "#D1495B"}
    names = {"analytic_hippo_rff": "HiPPO-RFF", "inducing_points": "Ordinary inducing"}
    for representation in ["analytic_hippo_rff", "inducing_points"]:
        path = OUT / f"task2_empirical_bayes/{representation}/seed0/training_trace.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        iterations = np.asarray([int(row["iteration"]) for row in rows])
        for axis, (metric, label) in zip(axes.flat, metrics):
            if metric == "ell_s_0":
                values = [json.loads(row["ell_s"].replace("'", '"'))[0] for row in rows]
            elif metric == "ell_s_1":
                values = [json.loads(row["ell_s"].replace("'", '"'))[1] for row in rows]
            else:
                values = [float(row[metric]) if row.get(metric, "") else np.nan for row in rows]
            values_array = np.asarray(values, dtype=float)
            finite = np.isfinite(values_array)
            axis.plot(
                iterations[finite],
                values_array[finite],
                color=colors[representation],
                linewidth=1.8,
                label=names[representation],
            )
            axis.set_ylabel(label)
            axis.grid(color="#D9E1E5", linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Adam iteration")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "empirical_bayes_training_trajectories.pdf", bbox_inches="tight")
    fig.savefig(OUT / "empirical_bayes_training_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed = shared_official_rows() + empirical_rows()
    summary = summary_rows(per_seed)
    official = official_end_to_end_rows()
    posterior_only, end_to_end = compute_table_rows(per_seed, official)
    write_csv(per_seed, OUT / "parameter_strategy_per_seed.csv")
    write_csv(summary, OUT / "parameter_strategy_summary.csv")
    write_csv(official, OUT / "official_end_to_end_per_seed.csv")
    write_csv(posterior_only, OUT / "posterior_only_compute_summary.csv")
    write_csv(end_to_end, OUT / "end_to_end_compute_summary.csv")
    plot_strategy(summary)
    plot_trajectories()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
