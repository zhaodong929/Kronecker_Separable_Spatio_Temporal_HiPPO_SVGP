#!/usr/bin/env python3
"""Summarize the causal Route-B online parameter-adaptation ablation."""

from __future__ import annotations

import argparse
import csv
import json
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

from baselines.covid_long_setting_b.protocol import COVIDSettingBProtocol
from scripts.compute_covid_long_target_likelihood_crps import normal_crps
from scripts.compute_covid_long_target_likelihood_ece import (
    ECE_COVERAGE_LEVELS,
    empirical_normal_intervals,
    interval_calibration,
)

SEEDS = (5, 6, 7, 8, 9)
REPRESENTATIONS = ("ordinary", "cumulative")
METRICS = ("rmse", "crps", "gaussian_nlpd", "ece", "coverage90")
LABELS = {
    "ordinary": "Kron-STGP",
    "cumulative": "KronHiPPO-STGP",
}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def archive_path(root: Path, seed: int, representation: str, adaptive: bool) -> Path:
    if adaptive:
        return root / f"seed{seed}" / representation / "predictions.npz"
    return root / f"seed{seed}" / f"routeb_{representation}" / "online" / "predictions.npz"


def metric_arrays(
    arrays: dict[str, np.ndarray], standardization: dict[str, Any], *, ece_seed: int
) -> dict[str, float]:
    scale = float(standardization["scale"])
    offset = float(standardization["mean"])
    truth = np.asarray(arrays["y_true"], dtype=np.float64) * scale + offset
    mean = np.asarray(arrays["pred_mean"], dtype=np.float64) * scale + offset
    variance = np.asarray(arrays["pred_var"], dtype=np.float64) * scale**2
    if truth.shape != (143, 10) or mean.shape != truth.shape or variance.shape != truth.shape:
        raise ValueError(f"Unexpected archive shape: {truth.shape}, {mean.shape}, {variance.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance <= 0.0):
        raise ValueError("Prediction archive contains non-finite means or non-positive variances")
    lower, upper = empirical_normal_intervals(
        mean, variance, samples=100, seed=ece_seed
    )
    ece, _, _ = interval_calibration(truth, lower, upper, ECE_COVERAGE_LEVELS)
    error = truth - mean
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "crps": float(np.mean(normal_crps(truth, mean, variance))),
        "gaussian_nlpd": float(
            np.mean(0.5 * (np.log(2.0 * np.pi * variance) + error**2 / variance))
        ),
        "ece": float(ece),
        "coverage90": float(
            np.mean(np.abs(error) <= 1.6448536269514722 * np.sqrt(variance))
        ),
    }


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in ("y_true", "pred_mean", "pred_var")}


def cumulative_rmse(arrays: dict[str, np.ndarray], standardization: dict[str, Any]) -> np.ndarray:
    scale = float(standardization["scale"])
    offset = float(standardization["mean"])
    truth = np.asarray(arrays["y_true"], dtype=np.float64) * scale + offset
    mean = np.asarray(arrays["pred_mean"], dtype=np.float64) * scale + offset
    return np.sqrt(np.mean((truth[:, None, :] - mean[:, None, :]) ** 2, axis=2).cumsum(axis=0) / np.arange(1, truth.shape[0] + 1)[:, None]).mean(axis=1)


def load_result(path: Path) -> dict[str, Any]:
    result_path = path.parent / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    return json.loads(result_path.read_text(encoding="utf-8"))


def plot_metric_deltas(rows: list[dict[str, Any]], output: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.2), squeeze=False)
    colors = {"ordinary": "#2f5d8c", "cumulative": "#a33f2b"}
    for row_index, representation in enumerate(REPRESENTATIONS):
        for col_index, metric in enumerate(METRICS):
            axis = axes[row_index, col_index]
            values = np.asarray(
                [
                    float(row[f"delta_{metric}"])
                    for row in rows
                    if row["representation"] == representation
                ],
                dtype=np.float64,
            )
            axis.axhline(0.0, color="#555555", linewidth=0.8)
            axis.scatter(np.arange(len(values)), values, color=colors[representation], s=28, zorder=3)
            axis.axhline(values.mean(), color=colors[representation], linewidth=1.4)
            axis.set_title(metric.replace("_", " ").upper(), fontsize=9)
            axis.set_xticks(np.arange(len(SEEDS)), [str(seed) for seed in SEEDS])
            axis.set_xlabel("seed")
            if col_index == 0:
                axis.set_ylabel(f"{LABELS[representation]}\nadaptive - fixed")
            axis.text(0.98, 0.92, f"mean={values.mean():+.4f}", transform=axis.transAxes, ha="right", va="top", fontsize=8)
    fig.suptitle("Causal online parameter adaptation: paired formal-seed differences", y=1.01)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_deltas(cumulative: dict[str, list[dict[str, Any]]], output: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    colors = {"ordinary": "#2f5d8c", "cumulative": "#a33f2b"}
    for axis, representation in zip(axes, REPRESENTATIONS):
        groups = [row for row in cumulative[representation]]
        for row in groups:
            axis.plot(row["horizon"], row["delta_rmse"], color="#999999", alpha=0.55, linewidth=0.8)
        horizon = np.asarray(groups[0]["horizon"], dtype=int)
        values = np.asarray([row["delta_rmse"] for row in groups], dtype=np.float64)
        axis.plot(horizon, values.mean(axis=0), color=colors[representation], linewidth=2.0, label="mean")
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.set_title(LABELS[representation])
        axis.set_xlabel("online horizon (weeks)")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[0].set_ylabel("adaptive - fixed cumulative RMSE")
    fig.suptitle("Does online parameter adaptation improve point accuracy over the stream?", y=1.03)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-root", type=Path, default=ROOT / "results/diagnostics/covid_long_stream_2020_2024_mandatory")
    parser.add_argument("--adaptive-root", type=Path, default=ROOT / "results/diagnostics/covid_long_routeb_online_adaptive_stable")
    parser.add_argument("--adaptive-cumulative-root", type=Path, default=ROOT / "results/diagnostics/covid_long_routeb_online_adaptive_stable_cumulative")
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "data/epidemiology/protocol/covid_long_2020_2024_mandatory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "baselines/covid_long_setting_b/reports/routeb_online_parameter_adaptation_20260821")
    args = parser.parse_args()
    fixed_root = args.fixed_root.resolve()
    adaptive_root = args.adaptive_root.resolve()
    adaptive_cumulative_root = args.adaptive_cumulative_root.resolve()
    protocol_root = args.protocol_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    cumulative: dict[str, list[dict[str, Any]]] = {representation: [] for representation in REPRESENTATIONS}
    theta_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    development_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        protocol = COVIDSettingBProtocol(protocol_root / f"seed{seed}" / "protocol.npz")
        standardization = protocol.metadata["target_standardization"]
        for representation in REPRESENTATIONS:
            fixed_path = archive_path(fixed_root, seed, representation, adaptive=False)
            adaptive_source_root = adaptive_cumulative_root if seed == 0 and representation == "cumulative" else adaptive_root
            adaptive_path = archive_path(adaptive_source_root, seed, representation, adaptive=True)
            fixed_arrays = load_arrays(fixed_path)
            adaptive_arrays = load_arrays(adaptive_path)
            fixed_metrics = metric_arrays(fixed_arrays, standardization, ece_seed=1_000_000 + seed)
            adaptive_metrics = metric_arrays(adaptive_arrays, standardization, ece_seed=1_000_000 + seed)
            per_seed.extend(
                [
                    {"representation": representation, "variant": "fixed_task1_eb", "label": f"{LABELS[representation]} fixed Task-1 EB", "seed": seed, **fixed_metrics},
                    {"representation": representation, "variant": "online_adaptive", "label": f"{LABELS[representation]} online adaptive", "seed": seed, **adaptive_metrics},
                ]
            )
            paired_row: dict[str, Any] = {"representation": representation, "seed": seed}
            for metric in METRICS:
                paired_row[f"fixed_{metric}"] = fixed_metrics[metric]
                paired_row[f"adaptive_{metric}"] = adaptive_metrics[metric]
                paired_row[f"delta_{metric}"] = adaptive_metrics[metric] - fixed_metrics[metric]
            paired.append(paired_row)
            fixed_curve = cumulative_rmse(fixed_arrays, standardization)
            adaptive_curve = cumulative_rmse(adaptive_arrays, standardization)
            cumulative[representation].append(
                {"seed": seed, "horizon": list(range(1, 144)), "delta_rmse": (adaptive_curve - fixed_curve).tolist()}
            )

            result = load_result(adaptive_path)
            audit = result["audit"]
            audit_rows.append(
                {
                    "representation": representation,
                    "seed": seed,
                    "status": result.get("status"),
                    "online_steps_completed": audit.get("online_steps_completed"),
                    "delayed_hidden_rows": audit.get("delayed_hidden_rows"),
                    "expected_delayed_hidden_rows": audit.get("expected_delayed_hidden_rows"),
                    "current_visible_rows": audit.get("current_visible_rows"),
                    "expected_current_visible_rows": audit.get("expected_current_visible_rows"),
                    "current_hidden_reads": audit.get("current_hidden_reads"),
                    "hidden_predictions": audit.get("hidden_predictions"),
                    "passed": audit.get("passed"),
                    "rejected_updates": sum(int(row.get("adaptation_rejected", 0)) for row in result.get("theta_updates", [])),
                }
            )
            for update in result.get("theta_updates", []):
                theta = update["theta_after"]
                theta_rows.append(
                    {
                        "representation": representation,
                        "seed": seed,
                        "stream_week": update["stream_week"],
                        "ell_t": theta["ell_t"],
                        "ell_s_0": theta["ell_s"][0],
                        "ell_s_1": theta["ell_s"][1],
                        "kernel_variance": theta["kernel_variance"],
                        "noise_std": theta["noise_std"],
                        "seconds": update["seconds"],
                    }
                )

    development_seed = 0
    protocol = COVIDSettingBProtocol(protocol_root / f"seed{development_seed}" / "protocol.npz")
    standardization = protocol.metadata["target_standardization"]
    for representation in REPRESENTATIONS:
        fixed_metrics = metric_arrays(
            load_arrays(archive_path(fixed_root, development_seed, representation, adaptive=False)),
            standardization,
            ece_seed=1_000_000 + development_seed,
        )
        adaptive_metrics = metric_arrays(
            load_arrays(
                archive_path(
                    adaptive_cumulative_root if representation == "cumulative" else adaptive_root,
                    development_seed,
                    representation,
                    adaptive=True,
                )
            ),
            standardization,
            ece_seed=1_000_000 + development_seed,
        )
        development_rows.extend(
            [
                {"representation": representation, "variant": "fixed_task1_eb", "seed": development_seed, **fixed_metrics},
                {"representation": representation, "variant": "online_adaptive", "seed": development_seed, **adaptive_metrics},
            ]
        )

    aggregate: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for variant in ("fixed_task1_eb", "online_adaptive"):
            group = [row for row in per_seed if row["representation"] == representation and row["variant"] == variant]
            row: dict[str, Any] = {
                "representation": representation,
                "variant": variant,
                "label": f"{LABELS[representation]} {'online adaptive' if variant == 'online_adaptive' else 'fixed Task-1 EB'}",
                "seeds": len(group),
            }
            for metric in METRICS:
                row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd([float(item[metric]) for item in group])
            aggregate.append(row)

    delta_summary: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        group = [row for row in paired if row["representation"] == representation]
        row = {"representation": representation, "label": LABELS[representation], "seeds": len(group)}
        for metric in METRICS:
            values = [float(item[f"delta_{metric}"]) for item in group]
            row[f"delta_{metric}_mean"], row[f"delta_{metric}_sd"] = mean_sd(values)
        delta_summary.append(row)

    write_csv(output / "per_seed_metrics.csv", per_seed, ["representation", "variant", "label", "seed", *METRICS])
    write_csv(output / "aggregate_metrics.csv", aggregate, ["representation", "variant", "label", "seeds", *[f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "sd")]])
    write_csv(output / "paired_deltas.csv", paired, ["representation", "seed", *[f"{prefix}_{metric}" for metric in METRICS for prefix in ("fixed", "adaptive", "delta")]])
    write_csv(output / "paired_delta_summary.csv", delta_summary, ["representation", "label", "seeds", *[f"delta_{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "sd")]])
    write_csv(output / "development_seed0_metrics.csv", development_rows, ["representation", "variant", "seed", *METRICS])
    write_csv(output / "causal_audit.csv", audit_rows, list(audit_rows[0]))
    write_csv(output / "theta_trajectories.csv", theta_rows, ["representation", "seed", "stream_week", "ell_t", "ell_s_0", "ell_s_1", "kernel_variance", "noise_std", "seconds"])
    (output / "cumulative_rmse_deltas.json").write_text(json.dumps(cumulative, indent=2) + "\n", encoding="utf-8")

    plot_metric_deltas(paired, output / "fig_routeb_online_adaptation_delta")
    plot_cumulative_deltas(cumulative, output / "fig_routeb_online_adaptation_cumulative_rmse")

    report_lines = [
        "# Route-B Online Parameter Adaptation",
        "",
        "This is a standalone ablation. The fixed Route-B archives and the current paper table are unchanged.",
        "The adaptive version starts from Task-1 empirical-Bayes parameters and updates ell_t, ell_s, kernel variance and noise every four online weeks using five proximal Adam steps. The trust region is max log-deviation 0.10 and the beta-Schur rejection bound is 10.0.",
        "",
        "## Formal comparison",
        "",
        "Metrics are recomputed from the prediction arrays on restored Z = log1p(weekly admissions per 100,000). Formal seeds are 5-9. ECE uses the existing K=10, S=100 empirical-normal-interval protocol with the same ECE random seed for each fixed/adaptive pair.",
        "",
        "| Representation | Variant | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        fmt = lambda metric: f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_sd']:.4f}"
        report_lines.append(f"| {row['label']} | {row['variant']} | {fmt('rmse')} | {fmt('crps')} | {fmt('gaussian_nlpd')} | {fmt('ece')} | {fmt('coverage90')} |")
    report_lines.extend(
        [
            "",
            "## Adaptive minus fixed",
            "",
            "Negative values improve the metric for RMSE, CRPS, Gaussian NLPD and ECE; for Coverage90, values are descriptive and should be read against the nominal 0.90 target.",
            "",
            "| Representation | Delta RMSE | Delta CRPS | Delta Gaussian NLPD | Delta ECE | Delta Coverage90 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in delta_summary:
        report_lines.append(
            f"| {row['label']} | {row['delta_rmse_mean']:+.4f} +/- {row['delta_rmse_sd']:.4f} | {row['delta_crps_mean']:+.4f} +/- {row['delta_crps_sd']:.4f} | {row['delta_gaussian_nlpd_mean']:+.4f} +/- {row['delta_gaussian_nlpd_sd']:.4f} | {row['delta_ece_mean']:+.4f} +/- {row['delta_ece_sd']:.4f} | {row['delta_coverage90_mean']:+.4f} +/- {row['delta_coverage90_sd']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Seed-0 development check",
            "",
            "Seed 0 was used only as development evidence; it is not included in the formal aggregate.",
            "",
            "| Representation | Variant | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in development_rows:
        report_lines.append(f"| {LABELS[row['representation']]} | {row['variant']} | {row['rmse']:.4f} | {row['crps']:.4f} | {row['gaussian_nlpd']:.4f} | {row['ece']:.4f} | {row['coverage90']:.4f} |")
    report_lines.extend(
        [
            "",
            "## Audit",
            "",
            "All ten formal adaptive archives must have shape (143, 10), finite means, positive variances, zero current-hidden reads, and exactly one absorption per delayed hidden label. See `causal_audit.csv` for the per-seed evidence.",
            "",
            "The main interpretation is intentionally limited: online adaptation is a probabilistic-calibration and uncertainty ablation. A negative CRPS/NLPD/ECE delta does not by itself establish a point-forecast improvement when the RMSE delta is near zero or positive.",
            "",
            "Generated figures: `fig_routeb_online_adaptation_delta.pdf` and `fig_routeb_online_adaptation_cumulative_rmse.pdf` (PNG counterparts are included).",
        ]
    )
    (output / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "status": "complete",
        "formal_seeds": list(SEEDS),
        "development_seed": 0,
        "representations": list(REPRESENTATIONS),
        "fixed_root": str(fixed_root),
        "adaptive_root": str(adaptive_root),
        "adaptive_cumulative_root": str(adaptive_cumulative_root),
        "protocol_root": str(protocol_root),
        "metric_system": ["RMSE", "CRPS", "Gaussian NLPD", "ECE", "Coverage90"],
        "adaptive_configuration": {
            "adapt_every": 4,
            "adapt_steps": 5,
            "learning_rate": 0.005,
            "proximal_lambda": 0.1,
            "max_log_deviation": 0.1,
            "max_beta_schur_term": 10.0,
            "mt": 32,
            "ms": 32,
        },
        "all_formal_audits_passed": all(bool(row["passed"]) for row in audit_rows),
        "rejected_updates_total": int(sum(int(row["rejected_updates"]) for row in audit_rows)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "all_formal_audits_passed": manifest["all_formal_audits_passed"]}, indent=2))


if __name__ == "__main__":
    main()
