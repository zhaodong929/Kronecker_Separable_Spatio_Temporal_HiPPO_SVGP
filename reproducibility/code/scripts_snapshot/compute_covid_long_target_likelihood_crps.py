#!/usr/bin/env python3
"""Compare Gaussian and NB COVID forecasts with common-scale CRPS.

CRPS is evaluated on Z = log1p(weekly admissions per 100,000).  Gaussian
Route B uses the exact Normal CRPS; the Negative-Binomial Route B score uses
its saved posterior-predictive samples after the same transformation.  This is
deliberately not a cross-likelihood NLPD comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normal_crps(truth: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """Return the exact CRPS of independent univariate Normal forecasts."""

    y = np.asarray(truth, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    variance_array = np.asarray(variance, dtype=np.float64)
    if y.shape != mu.shape or y.shape != variance_array.shape:
        raise ValueError("Normal CRPS inputs must have the same shape")
    if not np.isfinite(y).all() or not np.isfinite(mu).all() or not np.isfinite(variance_array).all():
        raise FloatingPointError("Normal CRPS requires finite targets and predictions")

    score = np.abs(y - mu)
    positive = variance_array > 1e-14
    if not np.any(positive):
        return score
    sigma = np.sqrt(variance_array[positive])
    z = (y[positive] - mu[positive]) / sigma
    cdf = 0.5 * (1.0 + np.vectorize(math.erf, otypes=[np.float64])(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    score[positive] = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    return np.maximum(score, 0.0)


def ensemble_crps(truth: np.ndarray, predictive_samples: np.ndarray) -> np.ndarray:
    """Estimate CRPS from samples without materialising pairwise differences."""

    y = np.asarray(truth, dtype=np.float64)
    draws = np.asarray(predictive_samples, dtype=np.float64)
    if draws.ndim != y.ndim + 1 or draws.shape[1:] != y.shape:
        raise ValueError("Predictive samples must have shape (samples, *truth.shape)")
    if draws.shape[0] < 2:
        raise ValueError("At least two predictive samples are required for ensemble CRPS")
    if not np.isfinite(y).all() or not np.isfinite(draws).all():
        raise FloatingPointError("Ensemble CRPS requires finite targets and predictive samples")

    sample_count = draws.shape[0]
    first_term = np.mean(np.abs(draws - y[None, ...]), axis=0)
    sorted_draws = np.sort(draws, axis=0)
    ranks = np.arange(1, sample_count + 1, dtype=np.float64).reshape((sample_count,) + (1,) * y.ndim)
    second_term = np.sum((2.0 * ranks - sample_count - 1.0) * sorted_draws, axis=0) / sample_count**2
    return first_term - second_term


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
    parser.add_argument("--predictive-samples", type=int, default=2048)
    args = parser.parse_args()
    if args.predictive_samples < 32:
        raise ValueError("The CRPS comparison requires at least 32 NB predictive samples")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_seed_rows: list[dict[str, object]] = []
    per_state_rows: list[dict[str, object]] = []
    audit: dict[str, object] = {
        "status": "complete",
        "metric": "CRPS",
        "evaluation_scale": "Z = log1p(weekly admissions per 100,000)",
        "gaussian_score": "exact univariate Normal CRPS",
        "negative_binomial_score": "ensemble CRPS from saved transformed posterior-predictive samples",
        "not_a_cross_likelihood_nlpd": True,
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
            names = np.asarray(metadata["location_names"], dtype=str)[test]
            exposure = np.asarray(protocol_data["population_per_100k"], dtype=np.float64)[test]
            truth = np.log1p(np.asarray(protocol_data["stream_counts"], dtype=np.float64)[:, test] / exposure[None, :])
        with np.load(gaussian) as gaussian_data:
            gaussian_mean = (
                np.asarray(gaussian_data["pred_mean"], dtype=np.float64)
                * float(metadata["target_standardization"]["scale"])
                + float(metadata["target_standardization"]["mean"])
            )
            gaussian_variance = np.asarray(gaussian_data["pred_var"], dtype=np.float64) * float(
                metadata["target_standardization"]["scale"]
            ) ** 2
        gaussian_score = normal_crps(truth, gaussian_mean, gaussian_variance)

        with np.load(nb) as nb_data:
            if "common_predictive_samples" not in nb_data.files:
                raise KeyError(
                    f"{nb} has no common_predictive_samples. Re-run the NB evaluator with "
                    "--save-common-predictive-samples."
                )
            nb_truth = np.asarray(nb_data["y_true"], dtype=np.float64)
            samples = np.asarray(nb_data["common_predictive_samples"], dtype=np.float64)
            declared_samples = int(np.asarray(nb_data["predictive_sample_count"]).item())
        np.testing.assert_allclose(nb_truth, truth, atol=1e-12, rtol=0.0)
        if declared_samples != args.predictive_samples or samples.shape[0] != args.predictive_samples:
            raise ValueError(
                f"NB seed {seed} has {samples.shape[0]} saved samples (declared {declared_samples}); "
                f"expected {args.predictive_samples}"
            )
        nb_score = ensemble_crps(truth, samples)

        for method, score in (("Gaussian log1p(per-100k) Route B", gaussian_score), ("Negative-Binomial Route B", nb_score)):
            per_seed_rows.append({"seed": seed, "method": method, "crps": float(np.mean(score))})
            for location, location_score in zip(names, np.mean(score, axis=0)):
                per_state_rows.append({"seed": seed, "method": method, "location_name": str(location), "crps": float(location_score)})
        audit["artifacts"].append(
            {
                "seed": seed,
                "protocol_sha256": sha256(protocol),
                "gaussian_prediction_sha256": sha256(gaussian),
                "negative_binomial_prediction_sha256": sha256(nb),
                "negative_binomial_samples_shape": list(samples.shape),
            }
        )

    aggregate_rows: list[dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in per_seed_rows}):
        values = np.asarray([row["crps"] for row in per_seed_rows if row["method"] == method], dtype=np.float64)
        aggregate_rows.append({"method": method, "seeds": values.size, "crps_mean": float(values.mean()), "crps_sd": float(values.std(ddof=1))})
    gaussian_crps = {int(row["seed"]): float(row["crps"]) for row in per_seed_rows if row["method"] == "Gaussian log1p(per-100k) Route B"}
    nb_crps = {int(row["seed"]): float(row["crps"]) for row in per_seed_rows if row["method"] == "Negative-Binomial Route B"}
    difference = np.asarray([nb_crps[seed] - gaussian_crps[seed] for seed in args.seeds], dtype=np.float64)
    mean_difference, ci_lower, ci_upper = paired_bootstrap(difference, seed=20260814)
    contrast = {
        "comparison": "Negative-Binomial minus Gaussian",
        "seeds": difference.size,
        "paired_crps_difference": mean_difference,
        "bootstrap95_lower": ci_lower,
        "bootstrap95_upper": ci_upper,
        "interpretation": "negative favours Negative-Binomial (lower CRPS)",
    }

    write_csv(per_seed_rows, output / "per_seed_crps.csv")
    write_csv(per_state_rows, output / "per_state_crps.csv")
    write_csv(aggregate_rows, output / "aggregate_crps.csv")
    write_csv([contrast], output / "paired_crps_contrast.csv")
    (output / "crps_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# COVID Long-Stream Common-Scale CRPS",
        "",
        "CRPS is evaluated on Z = log1p(weekly admissions per 100,000) over the same formal spatial splits. Gaussian Route B uses the exact Normal CRPS. Negative-Binomial Route B uses S=%d transformed posterior-predictive samples." % args.predictive_samples,
        "",
        "This is the cross-likelihood probability score. It is not a common NLPD: Gaussian density and Negative-Binomial count probability remain defined with different base measures.",
        "",
        "| Method | Seeds | CRPS |",
        "|---|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(f"| {row['method']} | {row['seeds']} | {row['crps_mean']:.4f} +/- {row['crps_sd']:.4f} |")
    lines.extend(
        [
            "",
            "| Comparison | Paired CRPS difference | Paired bootstrap 95% CI |",
            "|---|---:|---:|",
            f"| Negative-Binomial minus Gaussian | {mean_difference:.4f} | [{ci_lower:.4f}, {ci_upper:.4f}] |",
            "",
            "Lower CRPS is better. The archive hashes, per-split scores, and per-state scores are retained for independent recomputation.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "aggregate": aggregate_rows, "contrast": contrast}, indent=2))


if __name__ == "__main__":
    main()
