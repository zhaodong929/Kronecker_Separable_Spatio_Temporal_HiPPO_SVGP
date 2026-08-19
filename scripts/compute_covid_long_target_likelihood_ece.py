#!/usr/bin/env python3
"""Compute empirical-interval ECE for Gaussian and NB COVID Route B outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ECE_COVERAGE_LEVELS = np.round(np.arange(0.05, 1.0, 0.1), 2)
PREDICTIVE_SAMPLES = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empirical_normal_intervals(
    mean: np.ndarray,
    variance: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return empirical central intervals from the Gaussian predictive law."""

    if samples != PREDICTIVE_SAMPLES:
        raise ValueError(f"This ECE protocol requires S={PREDICTIVE_SAMPLES}")
    mean_array = np.asarray(mean, dtype=np.float64)
    std_array = np.sqrt(np.maximum(np.asarray(variance, dtype=np.float64), 1e-12))
    draws = mean_array[None, :, :] + std_array[None, :, :] * np.random.default_rng(seed).standard_normal(
        (samples, *mean_array.shape)
    )
    lower = np.quantile(draws, (1.0 - ECE_COVERAGE_LEVELS) / 2.0, axis=0)
    upper = np.quantile(draws, (1.0 + ECE_COVERAGE_LEVELS) / 2.0, axis=0)
    return lower, upper


def interval_calibration(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    coverage_levels: np.ndarray = ECE_COVERAGE_LEVELS,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Calculate the screenshot ECE and each central-interval coverage."""

    target = np.asarray(truth, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    levels = np.asarray(coverage_levels, dtype=np.float64)
    if lower_array.shape != upper_array.shape or lower_array.shape[0] != levels.size:
        raise ValueError("Interval arrays must have shape (K, time, held_out_location)")
    if lower_array.shape[1:] != target.shape:
        raise ValueError("Interval and truth shapes do not agree")
    if not np.isfinite(target).all() or not np.isfinite(lower_array).all() or not np.isfinite(upper_array).all():
        raise FloatingPointError("ECE requires finite predictions and targets")
    if np.any(lower_array > upper_array):
        raise ValueError("Every lower interval endpoint must not exceed its upper endpoint")
    empirical_coverage = np.mean(
        (target[None, :, :] >= lower_array) & (target[None, :, :] <= upper_array),
        axis=(1, 2),
    )
    calibration_gap = empirical_coverage - levels
    return float(np.mean(np.abs(calibration_gap))), empirical_coverage, calibration_gap


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap(values: np.ndarray, *, seed: int = 0, repeats: int = 20000) -> tuple[float, float, float]:
    paired = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    resamples = paired[rng.integers(0, paired.size, size=(repeats, paired.size))].mean(axis=1)
    return float(paired.mean()), float(np.quantile(resamples, 0.025)), float(np.quantile(resamples, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--gaussian-root", type=Path, required=True)
    parser.add_argument("--negative-binomial-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--predictive-samples", type=int, default=PREDICTIVE_SAMPLES)
    args = parser.parse_args()
    if args.predictive_samples != PREDICTIVE_SAMPLES:
        raise ValueError(f"The supplied formula fixes S={PREDICTIVE_SAMPLES}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_seed_rows: list[dict[str, object]] = []
    per_level_rows: list[dict[str, object]] = []
    audit: dict[str, object] = {
        "status": "complete",
        "formula": "ECE = mean_k | empirical central-interval coverage(c_k) - c_k |",
        "coverage_levels": ECE_COVERAGE_LEVELS.tolist(),
        "predictive_samples_per_point": PREDICTIVE_SAMPLES,
        "evaluation_scale": "log1p weekly admissions per 100,000",
        "seeds": args.seeds,
        "artifacts": [],
    }

    for seed in args.seeds:
        protocol = args.protocol_root / f"seed{seed}" / "protocol.npz"
        protocol_json = protocol.with_suffix(".json")
        gaussian = args.gaussian_root / f"seed{seed}" / "routeb_cumulative" / "online" / "predictions.npz"
        nb = args.negative_binomial_root / f"seed{seed}" / "predictions.npz"
        for path in (protocol, protocol_json, gaussian, nb):
            if not path.is_file():
                raise FileNotFoundError(path)

        metadata = json.loads(protocol_json.read_text(encoding="utf-8"))
        with np.load(protocol) as protocol_data:
            test = np.asarray(protocol_data["test_indices"], dtype=int)
            exposure = np.asarray(protocol_data["population_per_100k"], dtype=np.float64)[test]
            truth = np.log1p(np.asarray(protocol_data["stream_counts"], dtype=np.float64)[:, test] / exposure[None, :])
        with np.load(gaussian) as gaussian_data:
            mean = (
                np.asarray(gaussian_data["pred_mean"], dtype=np.float64)
                * float(metadata["target_standardization"]["scale"])
                + float(metadata["target_standardization"]["mean"])
            )
            variance = np.asarray(gaussian_data["pred_var"], dtype=np.float64) * float(
                metadata["target_standardization"]["scale"]
            ) ** 2
        gaussian_lower, gaussian_upper = empirical_normal_intervals(
            mean,
            variance,
            samples=args.predictive_samples,
            seed=1_000_000 + seed,
        )
        np.savez_compressed(
            output / f"gaussian_empirical_intervals_seed{seed}.npz",
            y_true=truth,
            ece_coverage_levels=ECE_COVERAGE_LEVELS,
            pred_interval_lower=gaussian_lower,
            pred_interval_upper=gaussian_upper,
            predictive_sample_count=np.asarray(PREDICTIVE_SAMPLES, dtype=np.int64),
        )

        with np.load(nb) as nb_data:
            nb_truth = np.asarray(nb_data["y_true"], dtype=np.float64)
            nb_levels = np.asarray(nb_data["ece_coverage_levels"], dtype=np.float64)
            nb_lower = np.asarray(nb_data["pred_interval_lower"], dtype=np.float64)
            nb_upper = np.asarray(nb_data["pred_interval_upper"], dtype=np.float64)
            nb_samples = int(np.asarray(nb_data["predictive_sample_count"]).item())
        np.testing.assert_allclose(nb_truth, truth, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(nb_levels, ECE_COVERAGE_LEVELS, atol=0.0, rtol=0.0)
        if nb_samples != PREDICTIVE_SAMPLES:
            raise ValueError(f"NB seed {seed} was generated with S={nb_samples}, expected S={PREDICTIVE_SAMPLES}")

        for method, lower, upper in (
            ("Gaussian log1p(per-100k) Route B", gaussian_lower, gaussian_upper),
            ("Negative-Binomial Route B", nb_lower, nb_upper),
        ):
            ece, coverage, gap = interval_calibration(truth, lower, upper)
            per_seed_rows.append({"seed": seed, "method": method, "ece": ece})
            for level, observed, difference in zip(ECE_COVERAGE_LEVELS, coverage, gap):
                per_level_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "nominal_coverage": float(level),
                        "empirical_coverage": float(observed),
                        "signed_calibration_gap": float(difference),
                        "absolute_calibration_gap": float(abs(difference)),
                    }
                )
        audit["artifacts"].append(
            {
                "seed": seed,
                "protocol_sha256": sha256(protocol),
                "gaussian_prediction_sha256": sha256(gaussian),
                "nb_prediction_sha256": sha256(nb),
            }
        )

    aggregate_rows: list[dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in per_seed_rows}):
        values = np.asarray([row["ece"] for row in per_seed_rows if row["method"] == method], dtype=np.float64)
        aggregate_rows.append(
            {
                "method": method,
                "seeds": values.size,
                "ece_mean": float(values.mean()),
                "ece_sd": float(values.std(ddof=1)),
            }
        )
    aggregate_level_rows: list[dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in per_level_rows}):
        for level in ECE_COVERAGE_LEVELS:
            group = [
                row
                for row in per_level_rows
                if row["method"] == method and float(row["nominal_coverage"]) == float(level)
            ]
            observed = np.asarray([row["empirical_coverage"] for row in group], dtype=np.float64)
            absolute_gap = np.asarray([row["absolute_calibration_gap"] for row in group], dtype=np.float64)
            aggregate_level_rows.append(
                {
                    "method": method,
                    "nominal_coverage": float(level),
                    "seeds": observed.size,
                    "empirical_coverage_mean": float(observed.mean()),
                    "empirical_coverage_sd": float(observed.std(ddof=1)),
                    "absolute_calibration_gap_mean": float(absolute_gap.mean()),
                    "absolute_calibration_gap_sd": float(absolute_gap.std(ddof=1)),
                }
            )
    gaussian_ece = {int(row["seed"]): float(row["ece"]) for row in per_seed_rows if row["method"] == "Gaussian log1p(per-100k) Route B"}
    nb_ece = {int(row["seed"]): float(row["ece"]) for row in per_seed_rows if row["method"] == "Negative-Binomial Route B"}
    difference = np.asarray([nb_ece[seed] - gaussian_ece[seed] for seed in args.seeds], dtype=np.float64)
    mean_difference, ci_lower, ci_upper = paired_bootstrap(difference, seed=4242)
    contrast = {
        "comparison": "Negative-Binomial minus Gaussian",
        "seeds": difference.size,
        "paired_ece_difference": mean_difference,
        "bootstrap95_lower": ci_lower,
        "bootstrap95_upper": ci_upper,
        "interpretation": "negative favours Negative-Binomial (lower ECE)",
    }

    write_csv(per_seed_rows, output / "per_seed_ece.csv")
    write_csv(per_level_rows, output / "per_level_coverage.csv")
    write_csv(aggregate_level_rows, output / "aggregate_per_level_coverage.csv")
    write_csv(aggregate_rows, output / "aggregate_ece.csv")
    write_csv([contrast], output / "paired_ece_contrast.csv")
    (output / "ece_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# COVID Long-Stream Empirical-Interval ECE",
        "",
        "ECE uses the supplied empirical-quantile definition with K=10 central coverage levels c in {0.05, 0.15, ..., 0.95} and S=100 predictive samples per test point.",
        "All values are computed on the common log1p(weekly admissions per 100,000) scale over the same formal seeds 5-9.",
        "",
        "| Method | Seeds | ECE |",
        "|---|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(f"| {row['method']} | {row['seeds']} | {row['ece_mean']:.4f} +/- {row['ece_sd']:.4f} |")
    lines.extend(
        [
            "",
            "| Comparison | Paired ECE difference | Paired bootstrap 95% CI |",
            "|---|---:|---:|",
            f"| Negative-Binomial minus Gaussian | {mean_difference:.4f} | [{ci_lower:.4f}, {ci_upper:.4f}] |",
            "",
            "A lower ECE is better. The per-level empirical coverages and all interval endpoints needed for independent recomputation are saved alongside this report.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "aggregate": aggregate_rows, "contrast": contrast}, indent=2))


if __name__ == "__main__":
    main()
