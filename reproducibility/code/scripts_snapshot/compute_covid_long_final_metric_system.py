#!/usr/bin/env python3
"""Produce the likelihood-aware final metric tables for COVID long-stream runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_covid_long_target_likelihood_crps import normal_crps
from scripts.compute_covid_long_target_likelihood_ece import (
    ECE_COVERAGE_LEVELS,
    empirical_normal_intervals,
    interval_calibration,
)

FORMAL_SEEDS = (5, 6, 7, 8, 9)


def gaussian_metrics_on_common_scale(
    arrays: dict[str, np.ndarray],
    standardization: dict[str, float],
    *,
    ece_seed: int,
) -> dict[str, float]:
    """Score a Gaussian prediction archive after restoring log1p(per-100k)."""

    scale = float(standardization["scale"])
    offset = float(standardization["mean"])
    truth = np.asarray(arrays["y_true"], dtype=np.float64) * scale + offset
    mean = np.asarray(arrays["pred_mean"], dtype=np.float64) * scale + offset
    variance = np.asarray(arrays["pred_var"], dtype=np.float64) * scale**2
    lower, upper = empirical_normal_intervals(mean, variance, samples=100, seed=ece_seed)
    ece, _, _ = interval_calibration(truth, lower, upper, ECE_COVERAGE_LEVELS)
    error = truth - mean
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "crps": float(np.mean(normal_crps(truth, mean, variance))),
        "ece": ece,
        "coverage90": float(np.mean((truth >= mean - 1.6448536269514722 * np.sqrt(variance)) & (truth <= mean + 1.6448536269514722 * np.sqrt(variance)))),
        "native_gaussian_nlpd": float(np.mean(0.5 * (np.log(2.0 * np.pi * variance) + error**2 / variance))),
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & RMSE $\\downarrow$ & CRPS $\\downarrow$ & ECE $\\downarrow$ & Coverage90 & Gaussian NLPD $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        label = str(row["label"])
        if row["method"] == "routeb_cumulative":
            label = "\\textbf{" + label + "}"
        format_metric = lambda name: f"{row[f'{name}_mean']:.4f} $\\pm$ {row[f'{name}_sd']:.4f}"
        lines.append(
            f"{label} & {format_metric('rmse')} & {format_metric('crps')} & {format_metric('ece')} & "
            f"{format_metric('coverage90')} & {format_metric('native_gaussian_nlpd')} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "% All entries use the Gaussian predictive family on restored log1p admissions per 100,000.",
            "% Mean +/- sample SD over the formal spatial splits. ECE uses K=10 and S=100.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def likelihood_aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for method in ("Gaussian log1p(per-100k) Route B", "Negative-Binomial Route B"):
        group = [row for row in rows if row["method"] == method]
        entry: dict[str, object] = {"method": method, "seeds": len(group)}
        for metric in ("rmse", "crps", "ece", "coverage90"):
            mean, sd = mean_sd([float(row[metric]) for row in group])
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_sd"] = sd
        summary.append(entry)
    return summary


def main() -> None:
    from scripts.generate_covid_long_stream_paper_artifacts import METHODS, collect_runs

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/diagnostics/covid_long_stream_2020_2024_mandatory"))
    parser.add_argument("--protocol-root", type=Path, default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    parser.add_argument("--nb-results-root", type=Path, default=Path("results/diagnostics/covid_long_target_likelihood_ablation/negative_binomial_crps_s2048"))
    parser.add_argument("--nb-crps-dir", type=Path, default=Path("results/diagnostics/covid_long_target_likelihood_ablation/crps_formal_seeds5_9"))
    parser.add_argument("--nb-ece-dir", type=Path, default=Path("results/diagnostics/covid_long_target_likelihood_ablation/ece_formal_seeds5_9"))
    args = parser.parse_args()

    results_root = (ROOT / args.results_root).resolve()
    protocol_root = (ROOT / args.protocol_root).resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(args.seeds)
    records, failures, protocols = collect_runs(results_root, protocol_root, seeds)
    if failures:
        detail = "; ".join(f"{row['method']} seed {row['seed']}: {row['status']}" for row in failures)
        raise ValueError(f"Final metric system requires complete baseline artifacts: {detail}")

    per_seed_rows: list[dict[str, object]] = []
    per_state_rows: list[dict[str, object]] = []
    for method_index, (method, label, _, _) in enumerate(METHODS):
        for seed in seeds:
            record = records[method][seed]
            protocol = protocols[seed]
            metrics = gaussian_metrics_on_common_scale(
                record["arrays"],
                protocol["metadata"]["target_standardization"],
                ece_seed=1_000_000 + seed if method == "routeb_cumulative" else 10_000_000 + 1000 * method_index + seed,
            )
            per_seed_rows.append({"method": method, "label": label, "seed": seed, **metrics})
            scale = float(protocol["metadata"]["target_standardization"]["scale"])
            offset = float(protocol["metadata"]["target_standardization"]["mean"])
            arrays = record["arrays"]
            names = protocol["metadata"]["location_names"]
            for position, index in enumerate(arrays["test_indices"]):
                state_arrays = {
                    "y_true": arrays["y_true"][:, position : position + 1],
                    "pred_mean": arrays["pred_mean"][:, position : position + 1],
                    "pred_var": arrays["pred_var"][:, position : position + 1],
                }
                per_state_rows.append(
                    {
                        "method": method,
                        "label": label,
                        "seed": seed,
                        "location_index": int(index),
                        "location_name": str(names[int(index)]),
                        **gaussian_metrics_on_common_scale(
                            state_arrays,
                            {"scale": scale, "mean": offset},
                            ece_seed=20_000_000 + 100_000 * method_index + 1000 * seed + position,
                        ),
                    }
                )

    aggregate_rows: list[dict[str, object]] = []
    for method, label, _, _ in METHODS:
        group = [row for row in per_seed_rows if row["method"] == method]
        entry: dict[str, object] = {"method": method, "label": label, "seeds": len(group)}
        for metric in ("rmse", "crps", "ece", "coverage90", "native_gaussian_nlpd"):
            mean, sd = mean_sd([float(row[metric]) for row in group])
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_sd"] = sd
        aggregate_rows.append(entry)

    write_csv(per_seed_rows, output / "gaussian_family_metrics_per_seed.csv")
    write_csv(per_state_rows, output / "gaussian_family_metrics_per_state.csv")
    write_csv(aggregate_rows, output / "gaussian_family_metrics_aggregate.csv")
    write_latex(aggregate_rows, output / "table_covid_long_gaussian_family_metrics.tex")

    nb_results_root = (ROOT / args.nb_results_root).resolve()
    nb_crps_rows = read_csv((ROOT / args.nb_crps_dir).resolve() / "per_seed_crps.csv")
    nb_ece_rows = read_csv((ROOT / args.nb_ece_dir).resolve() / "per_seed_ece.csv")
    crps_by_seed = {
        int(row["seed"]): float(row["crps"])
        for row in nb_crps_rows
        if row["method"] == "Negative-Binomial Route B"
    }
    ece_by_seed = {
        int(row["seed"]): float(row["ece"])
        for row in nb_ece_rows
        if row["method"] == "Negative-Binomial Route B"
    }
    gaussian_by_seed = {
        int(row["seed"]): row
        for row in per_seed_rows
        if row["method"] == "routeb_cumulative"
    }
    if set(gaussian_by_seed) != set(seeds) or set(crps_by_seed) != set(seeds) or set(ece_by_seed) != set(seeds):
        raise ValueError("Cross-likelihood metric artifacts do not cover the requested formal seed set")
    cross_rows: list[dict[str, object]] = []
    native_nb_rows: list[dict[str, object]] = []
    for seed in seeds:
        gaussian = gaussian_by_seed[seed]
        cross_rows.append(
            {
                "seed": seed,
                "method": "Gaussian log1p(per-100k) Route B",
                "rmse": float(gaussian["rmse"]),
                "crps": float(gaussian["crps"]),
                "ece": float(gaussian["ece"]),
                "coverage90": float(gaussian["coverage90"]),
            }
        )
        payload = json.loads((nb_results_root / f"seed{seed}" / "result.json").read_text(encoding="utf-8"))
        nb_metrics = payload["overall_current_block"]
        cross_rows.append(
            {
                "seed": seed,
                "method": "Negative-Binomial Route B",
                "rmse": float(nb_metrics["rmse"]),
                "crps": crps_by_seed[seed],
                "ece": ece_by_seed[seed],
                "coverage90": float(nb_metrics["coverage90"]),
            }
        )
        native_nb_rows.append(
            {
                "seed": seed,
                "method": "Negative-Binomial Route B",
                "native_count_nlpd": float(nb_metrics["negative_binomial_count_nll"]),
            }
        )
    cross_summary = likelihood_aggregate(cross_rows)
    native_nb_mean, native_nb_sd = mean_sd([float(row["native_count_nlpd"]) for row in native_nb_rows])
    write_csv(cross_rows, output / "cross_likelihood_metrics_per_seed.csv")
    write_csv(cross_summary, output / "cross_likelihood_metrics_aggregate.csv")
    write_csv(native_nb_rows, output / "negative_binomial_native_count_nlpd_per_seed.csv")
    audit = {
        "status": "complete",
        "metric_system": ["RMSE", "CRPS", "ECE", "Coverage90", "native NLPD"],
        "evaluation_scale": "Z = log1p(weekly admissions per 100,000)",
        "seeds": list(seeds),
        "methods": [method for method, _, _, _ in METHODS],
        "crps": "exact univariate Normal CRPS",
        "ece": "K=10 central coverage levels {0.05, 0.15, ..., 0.95}; S=100 empirical Normal draws per point",
        "native_nlpd_scope": "Gaussian predictive family only; no Gaussian-vs-NB NLPD ranking is produced here",
        "cross_likelihood_scope": "Gaussian-versus-NB comparison uses only RMSE, CRPS, ECE, and Coverage90",
        "negative_binomial_native_count_nlpd": {
            "mean": native_nb_mean,
            "sample_sd": native_nb_sd,
            "source": str(nb_results_root),
        },
        "results_root": str(results_root),
        "protocol_root": str(protocol_root),
    }
    (output / "metric_system_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    report = [
        "# COVID Long-Stream Final Metric System",
        "",
        "## Scope",
        "",
        "This table compares the seven existing Gaussian predictive archives on the common physical target Z = log1p(weekly admissions per 100,000), restored before scoring. RMSE measures mean error; CRPS is the proper distributional score; ECE uses the K=10, S=100 empirical-interval definition; Coverage90 is the nominal 90% central-interval coverage; and NLPD is the native Gaussian density score.",
        "",
        "NLPD is deliberately restricted to this common Gaussian observation family. Negative-Binomial results must be compared to these models with RMSE, CRPS, ECE, and Coverage90 only; their native count NLPD belongs in a separate NB-family table once comparable NB baselines exist.",
        "",
        "| Method | Splits | RMSE | CRPS | ECE | Coverage90 | Gaussian NLPD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        report.append(
            f"| {row['label']} | {row['seeds']} | {row['rmse_mean']:.4f} +/- {row['rmse_sd']:.4f} | "
            f"{row['crps_mean']:.4f} +/- {row['crps_sd']:.4f} | {row['ece_mean']:.4f} +/- {row['ece_sd']:.4f} | "
            f"{row['coverage90_mean']:.4f} +/- {row['coverage90_sd']:.4f} | "
            f"{row['native_gaussian_nlpd_mean']:.4f} +/- {row['native_gaussian_nlpd_sd']:.4f} |"
        )
    report.extend(
        [
            "",
            "## Negative-Binomial Boundary",
            "",
            "The separately audited NB Route B score is native count NLPD, computed by Gauss-Hermite integration of the NB posterior predictive law. It is not numerically ranked against the Gaussian NLPD column because the underlying probability measures differ.",
            "",
            "| Method | Splits | RMSE | CRPS | ECE | Coverage90 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cross_summary:
        report.append(
            f"| {row['method']} | {row['seeds']} | {row['rmse_mean']:.4f} +/- {row['rmse_sd']:.4f} | "
            f"{row['crps_mean']:.4f} +/- {row['crps_sd']:.4f} | {row['ece_mean']:.4f} +/- {row['ece_sd']:.4f} | "
            f"{row['coverage90_mean']:.4f} +/- {row['coverage90_sd']:.4f} |"
        )
    report.extend(
        [
            "",
            f"Negative-Binomial Route B native count NLPD: {native_nb_mean:.4f} +/- {native_nb_sd:.4f}. This is a single-method native score, not a cross-family ranking.",
            "",
            "The Gaussian CRPS is exact; NB CRPS uses 2,048 saved transformed posterior-predictive samples. NB ECE uses the specified 100-sample empirical-interval estimator.",
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "methods": len(aggregate_rows)}, indent=2))


if __name__ == "__main__":
    main()
