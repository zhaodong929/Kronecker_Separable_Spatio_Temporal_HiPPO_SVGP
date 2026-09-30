#!/usr/bin/env python3
"""Paired Route-B prediction ablation for the conditional residual variance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_routeb_batch_empirical_bayes import evaluate, load_controlled_grid
from stvgp_kronecker.routeb_empirical_bayes import BatchRouteBEmpiricalBayes


SOURCE_RUNS = {
    "finite_dtc": "phase_m_routeb_empirical_bayes/task2_empirical_bayes",
    "vfe": "phase_o_routeb_vfe_empirical_bayes/task2_vfe",
}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    args = parser.parse_args()

    base = args.base.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for objective_type, source_relative in SOURCE_RUNS.items():
        for representation in ("analytic_hippo_rff", "inducing_points"):
            for split_seed in (0, 1, 2):
                source = base / source_relative / representation / f"seed{split_seed}" / "result.json"
                payload = json.loads(source.read_text(encoding="utf-8"))
                run_args = payload["args"]
                controlled_npz = (
                    base
                    / "phase_d_joint_xlag_controlled"
                    / f"seed{split_seed}"
                    / f"era5_xlag_seed{split_seed}.npz"
                )
                data = load_controlled_grid(
                    root=args.root,
                    task="task_2",
                    controlled_npz=controlled_npz,
                    ms=int(payload["ms"]),
                    xlag_length=int(run_args.get("xlag_length", 10)),
                )
                theta = payload["learned_theta"]
                torch.manual_seed(int(payload.get("model_seed", run_args.get("model_seed", 0))))
                model = BatchRouteBEmpiricalBayes(
                    times=data.times,
                    spatial_inducing=data.spatial_inducing,
                    mt=int(payload["mt"]),
                    representation=representation,
                    initial_ell_t=float(theta["ell_t"]),
                    initial_ell_s=tuple(float(value) for value in theta["ell_s"]),
                    initial_kernel_variance=float(theta["kernel_variance"]),
                    initial_noise_std=float(theta["noise_std"]),
                    rff_sample_size=int(run_args.get("rff_sample_size", 256)),
                    seed=int(payload.get("model_seed", run_args.get("model_seed", 0))),
                    objective_type=objective_type,
                )

                paired_means: dict[bool, np.ndarray] = {}
                paired_rmse: dict[bool, float] = {}
                for include_residual in (True, False):
                    metrics, pointwise, _ = evaluate(
                        empirical_model=model,
                        data=data,
                        posterior_indices=data.train_indices,
                        evaluation_indices=data.test_indices,
                        representation=representation,
                        beta_prior_variance=float(run_args.get("beta_prior_variance", 1000.0)),
                        prediction_chunk_size=int(run_args.get("prediction_chunk_size", 8192)),
                        include_conditional_residual_variance=include_residual,
                    )
                    paired_means[include_residual] = np.asarray(
                        [item["pred_mean"] for item in pointwise], dtype=float
                    )
                    paired_rmse[include_residual] = float(metrics["rmse"])
                    rows.append(
                        {
                            "objective": objective_type,
                            "temporal_representation": representation,
                            "split_seed": split_seed,
                            "prediction_variance": (
                                "conditional_residual_included"
                                if include_residual
                                else "strict_finite_projected"
                            ),
                            "rmse": metrics["rmse"],
                            "nll": metrics["nll"],
                            "coverage90": metrics["coverage90"],
                            "ece": metrics["ece"],
                            "mean_predictive_std": metrics["mean_predictive_std"],
                            "avg_sigma2": metrics["diagnostic_avg_sigma2"],
                            "avg_nu_star_applied": metrics["diagnostic_avg_nu_star"],
                            "avg_nu_star_raw": metrics["diagnostic_avg_nu_star_raw"],
                            "avg_u_posterior_term": metrics["diagnostic_avg_u_posterior_term"],
                            "avg_beta_schur_term": metrics["diagnostic_avg_beta_schur_term"],
                        }
                    )
                np.testing.assert_allclose(paired_means[True], paired_means[False], rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(paired_rmse[True], paired_rmse[False], rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(
                    paired_rmse[True], float(payload["final"]["rmse"]), rtol=1e-10, atol=1e-12
                )

    summary: list[dict[str, Any]] = []
    metric_names = ("rmse", "nll", "coverage90", "ece", "mean_predictive_std", "avg_nu_star_raw")
    for objective in SOURCE_RUNS:
        for representation in ("analytic_hippo_rff", "inducing_points"):
            for variance_mode in ("conditional_residual_included", "strict_finite_projected"):
                group = [
                    row
                    for row in rows
                    if row["objective"] == objective
                    and row["temporal_representation"] == representation
                    and row["prediction_variance"] == variance_mode
                ]
                item: dict[str, Any] = {
                    "objective": objective,
                    "temporal_representation": representation,
                    "prediction_variance": variance_mode,
                    "n_seeds": len(group),
                }
                for metric in metric_names:
                    values = [float(row[metric]) for row in group]
                    item[f"{metric}_mean"] = float(np.mean(values))
                    item[f"{metric}_sd"] = sample_sd(values)
                summary.append(item)

    paired: list[dict[str, Any]] = []
    for objective in SOURCE_RUNS:
        for representation in ("analytic_hippo_rff", "inducing_points"):
            for split_seed in (0, 1, 2):
                match = [
                    row
                    for row in rows
                    if row["objective"] == objective
                    and row["temporal_representation"] == representation
                    and row["split_seed"] == split_seed
                ]
                by_mode = {row["prediction_variance"]: row for row in match}
                included = by_mode["conditional_residual_included"]
                strict = by_mode["strict_finite_projected"]
                paired.append(
                    {
                        "objective": objective,
                        "temporal_representation": representation,
                        "split_seed": split_seed,
                        "strict_minus_included_nll": float(strict["nll"] - included["nll"]),
                        "strict_minus_included_coverage90": float(
                            strict["coverage90"] - included["coverage90"]
                        ),
                        "strict_minus_included_ece": float(strict["ece"] - included["ece"]),
                    }
                )

    write_csv(rows, outdir / "per_seed.csv")
    write_csv(summary, outdir / "summary.csv")
    write_csv(paired, outdir / "paired_differences.csv")
    result = {
        "protocol": {
            "comparison": "same learned theta and same posterior; only k**-q** is toggled",
            "task": "ERA5 task_2 variable 0",
            "spatial_splits": [0, 1, 2],
            "mt": 128,
            "ms": 128,
        },
        "summary": summary,
        "paired_differences": paired,
    }
    (outdir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in summary:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
