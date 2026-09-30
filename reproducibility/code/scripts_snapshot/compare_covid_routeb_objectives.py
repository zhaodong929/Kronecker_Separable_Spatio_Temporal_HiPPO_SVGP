#!/usr/bin/env python3
"""Compare finite-DTC and VFE Route B archives on the common COVID metric system."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_covid_long_final_metric_system import gaussian_metrics_on_common_scale


SEEDS = (5, 6, 7, 8, 9)
METRICS = ("rmse", "crps", "ece", "coverage90", "native_gaussian_nlpd")


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def score_archive(archive: Path, protocol: Path, seed: int) -> dict[str, float]:
    metadata = json.loads(protocol.read_text(encoding="utf-8"))
    with np.load(archive) as arrays:
        required = {"y_true", "pred_mean", "pred_var"}
        missing = required.difference(arrays.files)
        if missing:
            raise ValueError(f"{archive} is missing {sorted(missing)}")
        if arrays["y_true"].shape != (143, 10):
            raise ValueError(f"Unexpected archive shape for {archive}: {arrays['y_true'].shape}")
        if not all(np.isfinite(arrays[key]).all() for key in required):
            raise ValueError(f"Non-finite prediction archive: {archive}")
        if np.any(arrays["pred_var"] <= 0):
            raise ValueError(f"Non-positive predictive variance: {archive}")
        return gaussian_metrics_on_common_scale(
            {key: np.asarray(arrays[key]) for key in required},
            metadata["target_standardization"],
            ece_seed=1_000_000 + seed,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finite-root",
        type=Path,
        default=Path("results/diagnostics/covid_long_stream_2020_2024_mandatory"),
    )
    parser.add_argument(
        "--vfe-root",
        type=Path,
        default=Path("results/diagnostics/covid_long_stream_2020_2024_mandatory_vfe"),
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/diagnostics/covid_routeb_finite_dtc_vs_vfe_20260824"),
    )
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict[str, object]] = []
    for seed in SEEDS:
        protocol = args.protocol_root / f"seed{seed}" / "protocol.json"
        finite_archive = args.finite_root / f"seed{seed}" / "routeb_cumulative" / "online" / "predictions.npz"
        vfe_archive = args.vfe_root / f"seed{seed}" / "routeb_cumulative" / "online" / "predictions.npz"
        finite = score_archive(finite_archive, protocol, seed)
        vfe = score_archive(vfe_archive, protocol, seed)
        for objective, scores in (("finite_dtc", finite), ("vfe", vfe)):
            per_seed.append({"objective": objective, "seed": seed, **scores})

    summary: list[dict[str, object]] = []
    for objective in ("finite_dtc", "vfe"):
        rows = [row for row in per_seed if row["objective"] == objective]
        row: dict[str, object] = {"objective": objective, "seeds": len(rows)}
        for metric in METRICS:
            mean, sd = aggregate([float(item[metric]) for item in rows])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        summary.append(row)

    deltas: list[dict[str, object]] = []
    for seed in SEEDS:
        finite = next(row for row in per_seed if row["objective"] == "finite_dtc" and row["seed"] == seed)
        vfe = next(row for row in per_seed if row["objective"] == "vfe" and row["seed"] == seed)
        deltas.append({"seed": seed, **{metric: float(vfe[metric]) - float(finite[metric]) for metric in METRICS}})
    delta_summary: dict[str, object] = {"comparison": "vfe_minus_finite_dtc", "seeds": len(SEEDS)}
    for metric in METRICS:
        mean, sd = aggregate([float(row[metric]) for row in deltas])
        delta_summary[f"{metric}_mean"] = mean
        delta_summary[f"{metric}_sd"] = sd

    write_csv(per_seed, output / "per_seed_metrics.csv")
    write_csv(summary, output / "aggregate_metrics.csv")
    write_csv(deltas, output / "per_seed_deltas_vfe_minus_finite_dtc.csv")
    write_csv([delta_summary], output / "aggregate_deltas_vfe_minus_finite_dtc.csv")

    lines = [
        "# COVID Route B objective comparison",
        "",
        "Same cumulative HiPPO representation, Mt=32, Ms=32, strict-online protocol, and formal seeds 5-9.",
        "Metrics are computed on restored Z = log1p(weekly admissions per 100,000) using the common Gaussian evaluator.",
        "",
        "| Objective | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['objective']} | {row['rmse_mean']:.4f} +/- {row['rmse_sd']:.4f} | "
            f"{row['crps_mean']:.4f} +/- {row['crps_sd']:.4f} | "
            f"{row['native_gaussian_nlpd_mean']:.4f} +/- {row['native_gaussian_nlpd_sd']:.4f} | "
            f"{row['ece_mean']:.4f} +/- {row['ece_sd']:.4f} | "
            f"{row['coverage90_mean']:.4f} +/- {row['coverage90_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Positive RMSE/CRPS/NLPD/ECE deltas mean VFE is worse; positive Coverage90 delta means VFE coverage is higher.",
            "",
            "| Delta (VFE - finite-DTC) | Mean | SD |",
            "|---|---:|---:|",
        ]
    )
    for metric in METRICS:
        lines.append(f"| {metric} | {delta_summary[f'{metric}_mean']:+.6f} | {delta_summary[f'{metric}_sd']:.6f} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "seeds": list(SEEDS)}, indent=2))


if __name__ == "__main__":
    main()
