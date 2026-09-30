#!/usr/bin/env python3
"""Compare long-stream COVID target choices on one physical log-rate scale."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_covid_long_target_likelihood_crps import ensemble_crps


MODES = ("raw_count", "per_100k", "log1p_per_100k")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def common_gaussian_prediction(
    *,
    mode: str,
    native_mean: np.ndarray,
    native_variance: np.ndarray,
    scale: float,
    offset: float,
    exposure: np.ndarray,
    seed: int,
    samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map a native Gaussian target to log1p(per-100k) by fixed MC samples."""

    mean = np.asarray(native_mean, dtype=np.float64) * scale + offset
    std = np.sqrt(np.maximum(np.asarray(native_variance, dtype=np.float64), 1e-12)) * scale
    rng = np.random.default_rng(seed)
    draws = mean[None, :, :] + std[None, :, :] * rng.standard_normal((samples, *mean.shape))
    if mode == "raw_count":
        count_draws = np.maximum(draws, 0.0)
        log_rate_draws = np.log1p(count_draws / exposure[None, None, :])
    elif mode == "per_100k":
        log_rate_draws = np.log1p(np.maximum(draws, 0.0))
    elif mode == "log1p_per_100k":
        log_rate_draws = draws
    else:
        raise ValueError(f"Unsupported mode {mode}")
    return (
        log_rate_draws.mean(axis=0),
        np.maximum(log_rate_draws.var(axis=0), 1e-12),
        np.quantile(log_rate_draws, 0.05, axis=0),
        np.quantile(log_rate_draws, 0.95, axis=0),
        log_rate_draws,
    )


def common_metrics(
    y: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    predictive_samples: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(y, dtype=np.float64).reshape(-1)
    predicted = np.asarray(mean, dtype=np.float64).reshape(-1)
    var = np.maximum(np.asarray(variance, dtype=np.float64).reshape(-1), 1e-12)
    return {
        "rmse_log1p_per_100k": float(np.sqrt(np.mean((truth - predicted) ** 2))),
        "crps_log1p_per_100k": float(np.mean(ensemble_crps(y, predictive_samples))),
        "gaussian_moment_nll_log1p_per_100k": float(
            np.mean(0.5 * (np.log(2.0 * np.pi * var) + (truth - predicted) ** 2 / var))
        ),
        "coverage90_log1p_per_100k": float(
            np.mean((truth >= np.asarray(lower).reshape(-1)) & (truth <= np.asarray(upper).reshape(-1)))
        ),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        entry: dict[str, object] = {"method": method, "seeds": len(group)}
        for metric in (
            "rmse_log1p_per_100k",
            "crps_log1p_per_100k",
            "gaussian_moment_nll_log1p_per_100k",
            "coverage90_log1p_per_100k",
            "negative_binomial_count_nll",
        ):
            values = np.asarray([row[metric] for row in group if metric in row], dtype=np.float64)
            if values.size:
                entry[f"{metric}_mean"] = float(values.mean())
                entry[f"{metric}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary.append(entry)
    return summary


def paired_bootstrap(values: np.ndarray, *, seed: int = 0, repeats: int = 20000) -> tuple[float, float, float]:
    paired = np.asarray(values, dtype=np.float64)
    if paired.size == 0:
        raise ValueError("Paired bootstrap needs at least one value")
    rng = np.random.default_rng(seed)
    samples = paired[rng.integers(0, paired.size, size=(repeats, paired.size))].mean(axis=1)
    return float(paired.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--gaussian-root", type=Path, required=True)
    parser.add_argument("--negative-binomial-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--samples", type=int, default=2048)
    args = parser.parse_args()
    if args.samples < 256:
        raise ValueError("At least 256 common-scale samples are required")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    audit: dict[str, object] = {"status": "complete", "seeds": args.seeds, "models": []}
    for seed in args.seeds:
        reference_truth = None
        reference_split = None
        for mode in MODES:
            protocol = args.protocol_root / mode / f"seed{seed}" / "protocol.npz"
            protocol_json = protocol.with_suffix(".json")
            prediction = args.gaussian_root / f"gaussian_{mode}" / f"seed{seed}" / "routeb_cumulative" / "online" / "predictions.npz"
            if not protocol.is_file() or not protocol_json.is_file() or not prediction.is_file():
                raise FileNotFoundError(f"Missing Gaussian target artifact for seed={seed}, mode={mode}")
            metadata = json.loads(protocol_json.read_text(encoding="utf-8"))
            with np.load(protocol) as data, np.load(prediction) as archive:
                test = np.asarray(data["test_indices"], dtype=int)
                exposure = np.asarray(data["population_per_100k"], dtype=np.float64)[test]
                truth = np.log1p(np.asarray(data["stream_counts"], dtype=np.float64)[:, test] / exposure[None, :])
                common = common_gaussian_prediction(
                    mode=mode,
                    native_mean=np.asarray(archive["pred_mean"], dtype=np.float64),
                    native_variance=np.asarray(archive["pred_var"], dtype=np.float64),
                    scale=float(metadata["target_standardization"]["scale"]),
                    offset=float(metadata["target_standardization"]["mean"]),
                    exposure=exposure,
                    seed=1000 * seed + MODES.index(mode),
                    samples=args.samples,
                )
                if reference_truth is None:
                    reference_truth = truth
                    reference_split = test
                else:
                    np.testing.assert_allclose(truth, reference_truth, atol=1e-12, rtol=0.0)
                    np.testing.assert_array_equal(test, reference_split)
                np.savez_compressed(
                    output / f"common_gaussian_{mode}_seed{seed}.npz",
                    y_true=truth,
                    pred_mean=common[0],
                    pred_variance=common[1],
                    pred_lower90=common[2],
                    pred_upper90=common[3],
                )
                rows.append({"seed": seed, "method": f"Gaussian {mode}", **common_metrics(truth, *common)})
                audit["models"].append(
                    {"seed": seed, "method": f"Gaussian {mode}", "protocol_sha256": sha256(protocol), "prediction_sha256": sha256(prediction)}
                )

        nb_dir = args.negative_binomial_root / f"seed{seed}"
        nb_prediction = nb_dir / "predictions.npz"
        nb_result = nb_dir / "result.json"
        if not nb_prediction.is_file() or not nb_result.is_file():
            raise FileNotFoundError(f"Missing Negative-Binomial artifact for seed={seed}")
        with np.load(nb_prediction) as archive:
            truth = np.asarray(archive["y_true"], dtype=np.float64)
            np.testing.assert_allclose(truth, reference_truth, atol=1e-12, rtol=0.0)
            if "common_predictive_samples" not in archive.files:
                raise KeyError(
                    f"{nb_prediction} has no common_predictive_samples. Re-run NB with "
                    "--save-common-predictive-samples before producing this final report."
                )
            nb_samples = np.asarray(archive["common_predictive_samples"], dtype=np.float64)
            if nb_samples.shape[0] != args.samples:
                raise ValueError(f"NB seed {seed} has {nb_samples.shape[0]} samples, expected {args.samples}")
            nb_metrics = common_metrics(
                truth,
                np.asarray(archive["pred_mean"], dtype=np.float64),
                np.asarray(archive["pred_variance"], dtype=np.float64),
                np.asarray(archive["pred_lower90"], dtype=np.float64),
                np.asarray(archive["pred_upper90"], dtype=np.float64),
                nb_samples,
            )
            nb_metrics["negative_binomial_count_nll"] = float(
                np.mean(np.asarray(archive["negative_binomial_count_nll"], dtype=np.float64))
            )
        rows.append({"seed": seed, "method": "Negative-Binomial Route B", **nb_metrics})
        audit["models"].append(
            {"seed": seed, "method": "Negative-Binomial Route B", "prediction_sha256": sha256(nb_prediction), "result_sha256": sha256(nb_result)}
        )

    summary = summarise(rows)
    contrasts: list[dict[str, object]] = []
    reference = {int(row["seed"]): row for row in rows if row["method"] == "Gaussian log1p_per_100k"}
    for method in ("Gaussian raw_count", "Gaussian per_100k", "Negative-Binomial Route B"):
        paired_rows = sorted((row for row in rows if row["method"] == method), key=lambda row: int(row["seed"]))
        if not paired_rows:
            continue
        for metric in (
            "rmse_log1p_per_100k",
            "crps_log1p_per_100k",
            "coverage90_log1p_per_100k",
        ):
            differences = np.asarray(
                [float(row[metric]) - float(reference[int(row["seed"])][metric]) for row in paired_rows],
                dtype=np.float64,
            )
            mean, lower, upper = paired_bootstrap(differences, seed=100 + len(contrasts))
            contrasts.append(
                {
                    "method_minus_gaussian_log1p": method,
                    "metric": metric,
                    "seeds": differences.size,
                    "paired_mean_difference": mean,
                    "bootstrap95_lower": lower,
                    "bootstrap95_upper": upper,
                }
            )
    write_csv(rows, output / "per_seed_common_scale_metrics.csv")
    write_csv(summary, output / "aggregate_common_scale_metrics.csv")
    write_csv(contrasts, output / "paired_contrasts_vs_gaussian_log1p.csv")
    audit["all_protocol_targets_match"] = True
    audit["common_scale"] = "log1p weekly admissions per 100,000"
    audit["gaussian_transform"] = "fixed Monte-Carlo pushforward; raw/rate draws are clipped at zero"
    audit["scoring_definition"] = {
        "common_gaussian_moment_score": {
            "formula": "Gaussian NLL evaluated from predictive-sample mean and variance on log1p(per-100k)",
            "status": "diagnostic_only",
            "not_a_common_nlpd": "For Negative-Binomial Route B this fits a moment-matched Gaussian after sampling the NB predictive distribution.",
        },
        "gaussian_log1p_nlpd": "A continuous Gaussian density score on the modelled log1p(per-100k) target; reported only within Gaussian target variants.",
        "negative_binomial_count_nlpd": "A discrete count log score numerically integrated over the latent Gaussian predictive distribution by Gauss-Hermite quadrature.",
        "cross_likelihood_primary_metrics": ["RMSE on log1p(per-100k)", "CRPS on log1p(per-100k)", "empirical-interval ECE", "Coverage90"],
    }
    (output / "artifact_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    definition_audit = [
        "# NLPD Definition Audit",
        "",
        "## Decision",
        "The common-scale Gaussian-moment score is diagnostic only. It must not be called common-scale NLPD and must not rank Gaussian and Negative-Binomial likelihoods in the main result.",
        "",
        "## What Is Computed",
        "For each method, prediction samples are expressed as z = log1p(admissions per 100,000). The common score fits a Gaussian using the sample mean m_hat and variance v_hat, then evaluates 0.5 * [log(2*pi*v_hat) + (z-m_hat)^2/v_hat].",
        "For the NB method this is a moment-matched Gaussian score, not -log p_NB(z | D). The NB predictive law is discrete in the observed count, even though its samples can be transformed monotonically to z.",
        "",
        "## Proper Native Scores",
        "Gaussian log1p(per-100k) uses a continuous density score on its modelled target. Negative-Binomial Route B uses negative_binomial_count_nll: a discrete count log score with the latent Gaussian integrated numerically by Gauss-Hermite quadrature. These scores have different base measures and are reported separately.",
        "",
        "## Reporting Rule",
        "Use common-scale RMSE, CRPS, empirical-interval ECE, and Coverage90 for the primary Gaussian-versus-NB comparison. Keep Gaussian-moment NLL as a transformed-scale diagnostic and keep native likelihood NLPD columns separate. A directly comparable proper log score would require an explicitly declared common count-mass/coarsening rule for the Gaussian target.",
    ]
    (output / "nlpd_definition_audit.md").write_text("\n".join(definition_audit) + "\n", encoding="utf-8")

    lines = [
        "# COVID Long-Stream Target and Likelihood Ablation",
        "",
        "All methods use the same CDC mandatory-period weeks, split, delayed-observation order, cumulative HiPPO capacity, and Task-1-to-online protocol.",
        "Gaussian target variants are compared after deterministic Monte-Carlo pushforward to log1p admissions per 100,000.",
        "CRPS is the common cross-likelihood proper score on log1p admissions per 100,000. The common Gaussian-moment score is retained only in the machine-readable diagnostic CSV; it is not reported or ranked as common-scale NLPD. The NB count NLPD is reported separately.",
        "",
        "| Method | Seeds | RMSE (common) | CRPS (common) | Coverage90 (common) | Native NB count NLPD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def value(name: str) -> str:
            mean = row.get(f"{name}_mean")
            if mean is None:
                return "-"
            return f"{mean:.4f} +/- {row[f'{name}_sd']:.4f}"
        lines.append(
            f"| {row['method']} | {row['seeds']} | {value('rmse_log1p_per_100k')} | "
            f"{value('crps_log1p_per_100k')} | {value('coverage90_log1p_per_100k')} | "
            f"{value('negative_binomial_count_nll')} |"
        )
    lines.extend(
        [
            "",
            "Positive paired RMSE and CRPS differences mean worse common-scale forecasts. Positive Coverage differences mean wider or better-covering intervals. Gaussian-moment-score differences are retained only as diagnostics, not a cross-likelihood proper-score ranking.",
            "",
            "| Method minus Gaussian log1p(per-100k) | Metric | Mean difference | Paired bootstrap 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in contrasts:
        lines.append(
            f"| {row['method_minus_gaussian_log1p']} | {row['metric']} | {row['paired_mean_difference']:.4f} | "
            f"[{row['bootstrap95_lower']:.4f}, {row['bootstrap95_upper']:.4f}] |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
