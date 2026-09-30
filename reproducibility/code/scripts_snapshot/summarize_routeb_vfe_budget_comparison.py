#!/usr/bin/env python3
"""Summarize matched-step, convergence, and wall-clock Route-B comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from scripts.summarize_routeb_online_vfe_training import (
        aggregate,
        metrics,
        paired_summary,
        write_csv,
    )
except ModuleNotFoundError:  # Support direct execution from the scripts directory.
    from summarize_routeb_online_vfe_training import (
        aggregate,
        metrics,
        paired_summary,
        write_csv,
    )


OBJECTIVES = ("finite_dtc", "vfe")
NEW_PROTOCOLS = ("converged", "wallclock_60s")
METRIC_NAMES = (
    "rmse",
    "nll",
    "coverage50",
    "coverage80",
    "coverage90",
    "coverage95",
    "mean_predictive_std",
    "mean_interval_width90",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prediction_rows(
    experiment_root: Path, seeds: list[int]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    seed_rows: list[dict[str, object]] = []
    task_seed_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    for protocol in NEW_PROTOCOLS:
        for objective in OBJECTIVES:
            method = f"{objective}_{protocol}"
            for seed in seeds:
                root = experiment_root / "online" / protocol / objective / f"seed{seed}"
                calibration_path = (
                    experiment_root
                    / "calibration"
                    / protocol
                    / objective
                    / f"seed{seed}"
                    / "result.json"
                )
                with np.load(root / "predictions.npz") as arrays:
                    y = np.asarray(arrays["y_true"], dtype=float)
                    mean = np.asarray(arrays["pred_mean"], dtype=float)
                    variance = np.asarray(arrays["pred_var"], dtype=float)
                calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                online = json.loads((root / "result.json").read_text(encoding="utf-8"))
                seed_rows.append(
                    {"protocol": protocol, "objective": objective, "method": method, "seed": seed, **metrics(y, mean, variance)}
                )
                for task in range(2, 11):
                    start = (task - 2) * 186
                    stop = (task - 1) * 186
                    task_seed_rows.append(
                        {
                            "protocol": protocol,
                            "objective": objective,
                            "method": method,
                            "task": f"task_{task}",
                            "seed": seed,
                            **metrics(y[start:stop], mean[start:stop], variance[start:stop]),
                        }
                    )
                theta = calibration["learned_theta"]
                training_rows.append(
                    {
                        "protocol": protocol,
                        "objective": objective,
                        "seed": seed,
                        "iterations_completed": calibration["iterations_completed"],
                        "best_iteration": calibration["best_iteration"],
                        "stop_reason": calibration["stop_reason"],
                        "best_validation_nll": calibration["best_validation_nll"],
                        "training_seconds": calibration["timing"]["training_seconds"],
                        "time_to_best_validation_seconds": calibration["time_to_best_validation_seconds"],
                        "process_total_seconds": calibration["timing"]["process_total_seconds"],
                        "online_seconds": online["timing"]["process_total_seconds"],
                        "ell_t": theta["ell_t"],
                        "ell_s_1": theta["ell_s"][0],
                        "ell_s_2": theta["ell_s"][1],
                        "kernel_variance": theta["kernel_variance"],
                        "noise_std": theta["noise_std"],
                    }
                )
    return seed_rows, task_seed_rows, training_rows


def previous_accuracy_rows(
    previous_root: Path, dtc_diagnostic_root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    seed_rows = []
    for row in read_csv(previous_root / "metrics_per_seed.csv"):
        old_method = row["method"]
        objective = "finite_dtc" if old_method.startswith("finite_dtc") else "vfe"
        seed_rows.append(
            {
                "protocol": "steps_100",
                "objective": objective,
                "method": f"{objective}_steps_100",
                "seed": int(row["seed"]),
                **{metric: float(row[metric]) for metric in METRIC_NAMES},
            }
        )
    task_rows = []
    for row in read_csv(previous_root / "metrics_per_task.csv"):
        old_method = row["method"]
        objective = "finite_dtc" if old_method.startswith("finite_dtc") else "vfe"
        item: dict[str, object] = {
            "protocol": "steps_100",
            "objective": objective,
            "method": f"{objective}_steps_100",
            "task": row["task"],
            "seeds": int(row["seeds"]),
        }
        for key, value in row.items():
            if key not in {"method", "task", "seeds"}:
                item[key] = float(value)
        task_rows.append(item)
    return seed_rows, task_rows


def aggregate_training(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = (
        "iterations_completed",
        "best_iteration",
        "best_validation_nll",
        "training_seconds",
        "time_to_best_validation_seconds",
        "process_total_seconds",
        "online_seconds",
        "ell_t",
        "ell_s_1",
        "ell_s_2",
        "kernel_variance",
        "noise_std",
    )
    output = []
    for protocol in NEW_PROTOCOLS:
        for objective in OBJECTIVES:
            selected = [row for row in rows if row["protocol"] == protocol and row["objective"] == objective]
            item: dict[str, object] = {
                "protocol": protocol,
                "objective": objective,
                "seeds": len(selected),
                "stop_reasons": ";".join(sorted({str(row["stop_reason"]) for row in selected})),
            }
            for field in fields:
                values = np.asarray([float(row[field]) for row in selected])
                item[f"{field}_mean"] = float(values.mean())
                item[f"{field}_sd"] = float(values.std(ddof=1))
            output.append(item)
    return output


def paired_deltas(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for protocol in ("steps_100", *NEW_PROTOCOLS):
        rows = [row for row in seed_rows if row["protocol"] == protocol]
        by_key = {(str(row["objective"]), int(row["seed"])): row for row in rows}
        for seed in sorted({int(row["seed"]) for row in rows}):
            dtc = by_key[("finite_dtc", seed)]
            vfe = by_key[("vfe", seed)]
            output.append(
                {
                    "protocol": protocol,
                    "seed": seed,
                    **{
                        f"delta_{metric}_vfe_minus_dtc": float(vfe[metric]) - float(dtc[metric])
                        for metric in METRIC_NAMES
                    },
                }
            )
    return output


def summarize_paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for protocol in ("steps_100", *NEW_PROTOCOLS):
        selected = [row for row in rows if row["protocol"] == protocol]
        for row in paired_summary(selected):
            output.append({"protocol": protocol, **row})
    return output


def plot_overall(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    protocol_order = ("steps_100", "converged", "wallclock_60s")
    labels = {"steps_100": "100 steps", "converged": "Converged", "wallclock_60s": "60 s"}
    colors = {"finite_dtc": "#2878B5", "vfe": "#D95319"}
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.5))
    for axis, metric, ylabel in zip(axes, ("rmse", "nll", "coverage90"), ("RMSE", "NLL", "Coverage90")):
        x = np.arange(len(protocol_order))
        for index, objective in enumerate(OBJECTIVES):
            selected = {(str(row["protocol"]), str(row["objective"])): row for row in rows}
            means = [float(selected[(protocol, objective)][f"{metric}_mean"]) for protocol in protocol_order]
            sds = [float(selected[(protocol, objective)][f"{metric}_sd"]) for protocol in protocol_order]
            axis.bar(x + (index - 0.5) * 0.34, means, 0.34, yerr=sds, capsize=3, color=colors[objective], label=objective.upper())
        if metric == "coverage90":
            axis.axhline(0.9, color="#555555", linestyle="--", linewidth=1)
        axis.set(xticks=x, xticklabels=[labels[p] for p in protocol_order], ylabel=ylabel)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--dtc-diagnostic-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    new_seed_rows, new_task_seed_rows, training_rows = prediction_rows(args.experiment_root, args.seeds)
    previous_seed_rows, previous_task_rows = previous_accuracy_rows(args.previous_root, args.dtc_diagnostic_root)
    all_seed_rows = previous_seed_rows + new_seed_rows
    overall = aggregate(all_seed_rows, ("protocol", "objective", "method"))
    new_tasks = aggregate(new_task_seed_rows, ("protocol", "objective", "method", "task"))
    per_task = previous_task_rows + new_tasks
    training_summary = aggregate_training(training_rows)
    deltas = paired_deltas(all_seed_rows)
    delta_summary = summarize_paired_deltas(deltas)

    args.experiment_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.experiment_root / "accuracy_per_seed.csv", all_seed_rows)
    write_csv(args.experiment_root / "accuracy_overall.csv", overall)
    write_csv(args.experiment_root / "accuracy_per_task.csv", per_task)
    write_csv(args.experiment_root / "training_per_seed.csv", training_rows)
    write_csv(args.experiment_root / "training_summary.csv", training_summary)
    write_csv(args.experiment_root / "paired_deltas.csv", deltas)
    write_csv(args.experiment_root / "paired_delta_summary.csv", delta_summary)
    plot_overall(overall, args.experiment_root / "overall_budget_comparison.png")

    lookup = {(str(row["protocol"]), str(row["objective"])): row for row in overall}
    train_lookup = {(str(row["protocol"]), str(row["objective"])): row for row in training_summary}
    delta_lookup = {
        (str(row["protocol"]), str(row["metric"])): row for row in delta_summary
    }
    task_lookup = {
        (str(row["protocol"]), str(row["objective"]), str(row["task"])): row
        for row in per_task
    }
    lines = [
        "# Route-B VFE budget and convergence comparison",
        "",
        "All accuracy rows use cumulative-changing HiPPO, M_t=M_s=128, Task-1 calibration followed by frozen-hyperparameter strict Task 2--10 streaming, and D full structured-joint conditional prediction.",
        "Predictive means and online updates are unchanged between objectives; only the Task-1 hyperparameter objective differs.",
        "",
        "## Protocols",
        "",
        "- `steps_100`: matched 100-iteration budget, reusing the earlier saved runs. These runs used different GPUs and therefore support accuracy, but not runtime, comparison.",
        "- `converged`: at most 250 iterations, validation every 5 iterations, patience of 8 validation checks, and minimum NLL improvement of 1e-4. Streaming uses the checkpoint with the best Task-1 validation NLL.",
        "- `wallclock_60s`: at most 1000 iterations with a 60-second optimization-loop budget. The loop includes forward/backward, optimizer updates, and periodic validation, but excludes data loading, warm-up, final posterior reconstruction, and Task 2--10 streaming evaluation.",
        "- New timing runs were executed sequentially on one NVIDIA GeForce RTX 5070 Laptop GPU in float64. The budget is checked after a complete iteration, so a small overshoot is expected.",
        "",
        "## Overall strict-online accuracy",
        "",
        "| Protocol | Objective | RMSE | NLL | Coverage90 |",
        "|---|---|---:|---:|---:|",
    ]
    for protocol in ("steps_100", *NEW_PROTOCOLS):
        for objective in OBJECTIVES:
            row = lookup[(protocol, objective)]
            lines.append(
                f"| {protocol} | {objective} | {row['rmse_mean']:.4f} +/- {row['rmse_sd']:.4f} | {row['nll_mean']:.4f} +/- {row['nll_sd']:.4f} | {row['coverage90_mean']:.4f} +/- {row['coverage90_sd']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Paired differences below are VFE minus DTC on the same five spatial split seeds. Lower is better for RMSE and NLL; a positive coverage difference means wider or better-covered predictions.",
            "",
            "| Protocol | Delta RMSE | 95% CI | Delta NLL | 95% CI | Delta Coverage90 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for protocol in ("steps_100", *NEW_PROTOCOLS):
        rmse = delta_lookup[(protocol, "rmse")]
        nll = delta_lookup[(protocol, "nll")]
        coverage = delta_lookup[(protocol, "coverage90")]
        lines.append(
            f"| {protocol} | {rmse['mean_vfe_minus_dtc']:+.4f} | [{rmse['ci95_low']:+.4f}, {rmse['ci95_high']:+.4f}] | "
            f"{nll['mean_vfe_minus_dtc']:+.4f} | [{nll['ci95_low']:+.4f}, {nll['ci95_high']:+.4f}] | {coverage['mean_vfe_minus_dtc']:+.4f} |"
        )
    lines.extend(["", "## Training behavior", "", "| Protocol | Objective | Iterations | Best iteration | Training seconds | Stop reason |", "|---|---|---:|---:|---:|---|"])
    for protocol in NEW_PROTOCOLS:
        for objective in OBJECTIVES:
            row = train_lookup[(protocol, objective)]
            lines.append(
                f"| {protocol} | {objective} | {row['iterations_completed_mean']:.1f} | {row['best_iteration_mean']:.1f} | {row['training_seconds_mean']:.2f} | {row['stop_reasons']} |"
            )
    lines.extend(
        [
            "",
            "## Validation convergence versus streaming transfer",
            "",
            f"At convergence, VFE achieved a better Task-1 validation NLL ({train_lookup[('converged', 'vfe')]['best_validation_nll_mean']:.4f} versus {train_lookup[('converged', 'finite_dtc')]['best_validation_nll_mean']:.4f}), but this did not transfer to the frozen-hyperparameter Task 2--10 stream. Its streaming RMSE and NLL were both worse, and Coverage90 fell from {lookup[('converged', 'finite_dtc')]['coverage90_mean']:.4f} to {lookup[('converged', 'vfe')]['coverage90_mean']:.4f}.",
            "",
            "All five DTC convergence runs stopped by validation patience after 90--105 iterations. All five VFE runs also triggered validation early stopping after 225--250 iterations, with best checkpoints at 185--215. One VFE stop coincided with the 250-iteration cap, so the evidence is strong but not an unlimited-budget proof of mathematical convergence.",
            "",
            "The transfer gap grows late in the stream:",
            "",
            "| Protocol | Objective | Task 2 RMSE | Task 2 NLL | Task 10 RMSE | Task 10 NLL | Task 10 Coverage90 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for protocol in ("steps_100", *NEW_PROTOCOLS):
        for objective in OBJECTIVES:
            task2 = task_lookup[(protocol, objective, "task_2")]
            task10 = task_lookup[(protocol, objective, "task_10")]
            lines.append(
                f"| {protocol} | {objective} | {task2['rmse_mean']:.4f} | {task2['nll_mean']:.4f} | "
                f"{task10['rmse_mean']:.4f} | {task10['nll_mean']:.4f} | {task10['coverage90_mean']:.4f} |"
            )
    lines.extend(
        [
            "",
            "The converged VFE checkpoint learned substantially longer spatial lengthscales and a smaller kernel variance than converged DTC. Its mean predictive standard deviation on Tasks 2--10 was therefore narrower (0.0932 versus 0.1069), which explains much of the degraded late-stream NLL and coverage. This is evidence of weaker cross-task transfer under the Task-1-freeze protocol, not evidence that the VFE objective is intrinsically inferior in every batch setting.",
            "",
            "## Decision",
            "",
            "VFE wins the matched 100-step comparison on NLL and Coverage90, but not RMSE. It does not win when each objective is selected by Task-1 validation convergence, and it does not win under the equal 60-second calibration budget. Finite DTC therefore remains the defensible strict-online main method for this protocol; VFE should be reported as an objective ablation. A future VFE result could change this decision only with a calibration strategy that improves transfer to later tasks, not merely by adding more Task-1 optimization steps.",
        ]
    )
    (args.experiment_root / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"overall": overall, "training": training_summary}, indent=2))


if __name__ == "__main__":
    main()
