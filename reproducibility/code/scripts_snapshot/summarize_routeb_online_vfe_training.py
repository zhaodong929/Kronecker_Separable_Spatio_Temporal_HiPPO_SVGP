#!/usr/bin/env python3
"""Compare Task-1 VFE calibration + D prediction with finite-DTC + D."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(y: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    variance = np.asarray(variance, dtype=float).reshape(-1)
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise FloatingPointError("Predictive variance must be finite and positive")
    std = np.sqrt(variance)
    result = {
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "nll": float(
            np.mean(
                0.5
                * (np.log(2.0 * np.pi * variance) + (y - mean) ** 2 / variance)
            )
        ),
        "mean_predictive_std": float(np.mean(std)),
    }
    for level in COVERAGE_LEVELS:
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        result[f"coverage{int(level * 100)}"] = float(
            np.mean(np.abs(y - mean) <= z * std)
        )
    z90 = NormalDist().inv_cdf(0.95)
    result["mean_interval_width90"] = float(np.mean(2.0 * z90 * std))
    return result


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def aggregate(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    metric_names = (
        "rmse",
        "nll",
        "coverage50",
        "coverage80",
        "coverage90",
        "coverage95",
        "mean_predictive_std",
        "mean_interval_width90",
    )
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group, members in groups.items():
        item = dict(zip(keys, group))
        item["seeds"] = len(members)
        for metric in metric_names:
            mean, sd = mean_sd([float(member[metric]) for member in members])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_sd"] = sd
        output.append(item)
    return output


def paired_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for key in rows[0]:
        if not key.startswith("delta_"):
            continue
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half_width = 2.7764451051977987 * sd / math.sqrt(values.size)
        output.append(
            {
                "metric": key.removeprefix("delta_").removesuffix("_vfe_minus_dtc"),
                "mean_vfe_minus_dtc": mean,
                "sd": sd,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            }
        )
    return output


def plot_per_task(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "finite_dtc_training_d_prediction": "Finite DTC + D",
        "vfe_training_d_prediction": "VFE + D",
    }
    colors = {
        "finite_dtc_training_d_prediction": "#2878B5",
        "vfe_training_d_prediction": "#D95319",
    }
    metrics_to_plot = (
        ("rmse", "RMSE"),
        ("nll", "NLL"),
        ("coverage90", "Coverage90"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5))
    for axis, (metric, ylabel) in zip(axes, metrics_to_plot):
        for method in labels:
            selected = [row for row in rows if row["method"] == method]
            selected.sort(key=lambda row: int(str(row["task"]).split("_")[-1]))
            x = [int(str(row["task"]).split("_")[-1]) for row in selected]
            y = [float(row[f"{metric}_mean"]) for row in selected]
            sd = [float(row[f"{metric}_sd"]) for row in selected]
            axis.plot(x, y, marker="o", linewidth=1.8, color=colors[method], label=labels[method])
            axis.fill_between(
                x,
                np.asarray(y) - np.asarray(sd),
                np.asarray(y) + np.asarray(sd),
                color=colors[method],
                alpha=0.13,
                linewidth=0,
            )
        if metric == "coverage90":
            axis.axhline(0.9, color="#555555", linestyle="--", linewidth=1.0)
        axis.set(xlabel="ERA5 task", ylabel=ylabel, xticks=range(2, 11))
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dtc-diagnostic-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    vfe_seed_rows: list[dict[str, object]] = []
    vfe_task_seed_rows: list[dict[str, object]] = []
    theta_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        prediction_path = args.experiment_root / "online" / f"seed{seed}" / "predictions.npz"
        online_result_path = args.experiment_root / "online" / f"seed{seed}" / "result.json"
        calibration_path = args.experiment_root / "calibration" / f"seed{seed}" / "result.json"
        with np.load(prediction_path) as arrays:
            y = np.asarray(arrays["y_true"], dtype=float)
            mean = np.asarray(arrays["pred_mean"], dtype=float)
            variance = np.asarray(arrays["pred_var"], dtype=float)
        online_result = json.loads(online_result_path.read_text(encoding="utf-8"))
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        vfe_seed_rows.append(
            {"method": "vfe_training_d_prediction", "seed": seed, **metrics(y, mean, variance)}
        )
        for task in range(2, 11):
            start = (task - 2) * 186
            stop = (task - 1) * 186
            vfe_task_seed_rows.append(
                {
                    "method": "vfe_training_d_prediction",
                    "task": f"task_{task}",
                    "seed": seed,
                    **metrics(y[start:stop], mean[start:stop], variance[start:stop]),
                }
            )
        theta = calibration["learned_theta"]
        theta_rows.append(
            {
                "method": "vfe_training_d_prediction",
                "seed": seed,
                "best_iteration": calibration["best_iteration"],
                "best_validation_nll": calibration["best_validation_nll"],
                "calibration_seconds": calibration["timing"]["process_total_seconds"],
                "online_seconds": online_result["timing"]["process_total_seconds"],
                "ell_t": theta["ell_t"],
                "ell_s_1": theta["ell_s"][0],
                "ell_s_2": theta["ell_s"][1],
                "kernel_variance": theta["kernel_variance"],
                "noise_std": theta["noise_std"],
            }
        )
        dtc_calibration_path = (
            args.benchmark_root
            / "calibration"
            / "routeb_joint_analytic_hippo_rff"
            / f"seed{seed}"
            / "result.json"
        )
        dtc_calibration = json.loads(
            dtc_calibration_path.read_text(encoding="utf-8")
        )
        dtc_theta = dtc_calibration["learned_theta"]
        theta_rows.append(
            {
                "method": "finite_dtc_training_d_prediction",
                "seed": seed,
                "best_iteration": dtc_calibration["best_iteration"],
                "best_validation_nll": dtc_calibration["best_validation_nll"],
                "calibration_seconds": dtc_calibration["timing"]["process_total_seconds"],
                "online_seconds": "",
                "ell_t": dtc_theta["ell_t"],
                "ell_s_1": dtc_theta["ell_s"][0],
                "ell_s_2": dtc_theta["ell_s"][1],
                "kernel_variance": dtc_theta["kernel_variance"],
                "noise_std": dtc_theta["noise_std"],
            }
        )

    dtc_seed_rows = []
    for row in read_csv(args.dtc_diagnostic_root / "metrics_per_seed.csv"):
        if row["mode"] == "full_joint_conditional":
            dtc_seed_rows.append(
                {
                    "method": "finite_dtc_training_d_prediction",
                    "seed": int(row["seed"]),
                    **{
                        key: float(row[key])
                        for key in (
                            "rmse",
                            "nll",
                            "coverage50",
                            "coverage80",
                            "coverage90",
                            "coverage95",
                            "mean_predictive_std",
                            "mean_interval_width90",
                        )
                    },
                }
            )
    dtc_by_seed = {int(row["seed"]): row for row in dtc_seed_rows}
    paired_rows = []
    for vfe in vfe_seed_rows:
        dtc = dtc_by_seed[int(vfe["seed"])]
        paired_rows.append(
            {
                "seed": vfe["seed"],
                **{
                    f"delta_{metric}_vfe_minus_dtc": float(vfe[metric]) - float(dtc[metric])
                    for metric in (
                        "rmse",
                        "nll",
                        "coverage50",
                        "coverage80",
                        "coverage90",
                        "coverage95",
                        "mean_predictive_std",
                        "mean_interval_width90",
                    )
                },
            }
        )

    overall = aggregate(dtc_seed_rows + vfe_seed_rows, ("method",))
    vfe_per_task = aggregate(vfe_task_seed_rows, ("method", "task"))
    dtc_per_task = []
    for row in read_csv(args.dtc_diagnostic_root / "metrics_per_task.csv"):
        if row["mode"] != "full_joint_conditional":
            continue
        item: dict[str, object] = {
            "method": "finite_dtc_training_d_prediction",
            "task": row["task"],
            "seeds": int(row["seeds"]),
        }
        for key, value in row.items():
            if key not in {"task", "mode", "seeds"}:
                item[key] = float(value)
        dtc_per_task.append(item)
    per_task = dtc_per_task + vfe_per_task
    per_task.sort(key=lambda row: (int(str(row["task"]).split("_")[-1]), str(row["method"])))

    write_csv(args.experiment_root / "metrics_per_seed.csv", dtc_seed_rows + vfe_seed_rows)
    write_csv(args.experiment_root / "metrics_overall.csv", overall)
    write_csv(args.experiment_root / "metrics_per_task.csv", per_task)
    write_csv(args.experiment_root / "paired_deltas.csv", paired_rows)
    paired = paired_summary(paired_rows)
    write_csv(args.experiment_root / "paired_delta_summary.csv", paired)
    write_csv(args.experiment_root / "learned_hyperparameters.csv", theta_rows)
    plot_per_task(per_task, args.experiment_root / "per_task_comparison.png")

    by_method = {str(row["method"]): row for row in overall}
    dtc = by_method["finite_dtc_training_d_prediction"]
    vfe = by_method["vfe_training_d_prediction"]
    paired_by_metric = {str(row["metric"]): row for row in paired}
    task_rows = {
        (str(row["method"]), str(row["task"])): row for row in per_task
    }
    dtc_task10 = task_rows[("finite_dtc_training_d_prediction", "task_10")]
    vfe_task10 = task_rows[("vfe_training_d_prediction", "task_10")]
    rmse_delta = paired_by_metric["rmse"]
    nll_delta = paired_by_metric["nll"]
    coverage_delta = paired_by_metric["coverage90"]
    lines = [
        "# Strict-online Route-B: VFE training + D prediction",
        "",
        "Task 1 is used for Route-B empirical-Bayes calibration. The learned hyperparameters are then frozen for causal Task 2--10 streaming. Both rows use cumulative-changing HiPPO, identical splits and supports, and D full structured-joint conditional prediction.",
        "",
        "| Training objective | RMSE | NLL | Cov50 | Cov80 | Cov90 | Cov95 | Mean std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| finite DTC | {dtc['rmse_mean']:.4f} +/- {dtc['rmse_sd']:.4f} | {dtc['nll_mean']:.4f} +/- {dtc['nll_sd']:.4f} | {dtc['coverage50_mean']:.4f} | {dtc['coverage80_mean']:.4f} | {dtc['coverage90_mean']:.4f} +/- {dtc['coverage90_sd']:.4f} | {dtc['coverage95_mean']:.4f} | {dtc['mean_predictive_std_mean']:.4f} |",
        f"| VFE | {vfe['rmse_mean']:.4f} +/- {vfe['rmse_sd']:.4f} | {vfe['nll_mean']:.4f} +/- {vfe['nll_sd']:.4f} | {vfe['coverage50_mean']:.4f} | {vfe['coverage80_mean']:.4f} | {vfe['coverage90_mean']:.4f} +/- {vfe['coverage90_sd']:.4f} | {vfe['coverage95_mean']:.4f} | {vfe['mean_predictive_std_mean']:.4f} |",
        "",
        "## Paired differences",
        "",
        f"- RMSE (VFE - DTC): `{rmse_delta['mean_vfe_minus_dtc']:.6f}`; 95% CI `[{rmse_delta['ci95_low']:.6f}, {rmse_delta['ci95_high']:.6f}]`.",
        f"- NLL (VFE - DTC): `{nll_delta['mean_vfe_minus_dtc']:.6f}`; 95% CI `[{nll_delta['ci95_low']:.6f}, {nll_delta['ci95_high']:.6f}]`.",
        f"- Coverage90 (VFE - DTC): `{coverage_delta['mean_vfe_minus_dtc']:.6f}`; 95% CI `[{coverage_delta['ci95_low']:.6f}, {coverage_delta['ci95_high']:.6f}]`.",
        "",
        "VFE improves probabilistic scoring and moves aggregate Coverage90 closer to 0.90, but incurs a small RMSE cost. The RMSE increase occurred for every seed. The NLL improvement also occurred for every seed, although its five-seed confidence interval narrowly includes zero.",
        "",
        "## Long-horizon behavior",
        "",
        f"Task 10 remains under-dispersed: Coverage90 is `{dtc_task10['coverage90_mean']:.4f}` for finite DTC + D and `{vfe_task10['coverage90_mean']:.4f}` for VFE + D. Task-10 RMSE changes from `{dtc_task10['rmse_mean']:.4f}` to `{vfe_task10['rmse_mean']:.4f}`. VFE therefore does not solve late-stream drift from frozen Task-1 hyperparameters, changing-basis transfer, or finite-state capacity.",
        "",
        "## Scope and caveats",
        "",
        "The VFE row is training/prediction-consistent and both rows use exactly the same D predictive variance. No Task 2--10 labels update hyperparameters. The finite-DTC control reuses the existing saved Task-1 calibration, whose validation checkpoints used DTC variance; VFE checkpoints use full conditional variance. All VFE runs selected iteration 100, so this is a matched 100-step budget comparison rather than a convergence claim. Runtime values are not compared because the saved DTC calibration ran on an RTX 4090, whereas VFE was run locally on an RTX 5070 Laptop GPU.",
    ]
    (args.experiment_root / "diagnostic_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"overall": overall, "paired": paired_rows}, indent=2))


if __name__ == "__main__":
    main()
