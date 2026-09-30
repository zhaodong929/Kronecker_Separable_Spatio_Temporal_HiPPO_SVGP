#!/usr/bin/env python3
"""Re-evaluate learned Task-2 batch-EB Route B with strict DTC variance."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_routeb_batch_empirical_bayes import (
    evaluate,
    load_controlled_grid,
    write_csv,
)
from stvgp_kronecker.joint_ssgp_kron.synthetic import temporal_spec_for_block
from stvgp_kronecker.routeb_empirical_bayes import BatchRouteBEmpiricalBayes


def source_path(base: Path, split_seed: int) -> Path:
    return (
        base
        / "phase_m_routeb_empirical_bayes"
        / "task2_empirical_bayes"
        / "analytic_hippo_rff"
        / f"seed{split_seed}"
        / "result.json"
    )


def sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def run_seed(
    *, base: Path, outdir: Path, data_root: str, split_seed: int
) -> dict[str, Any]:
    source = json.loads(source_path(base, split_seed).read_text(encoding="utf-8"))
    args = source["args"]
    controlled = (
        base
        / "phase_d_joint_xlag_controlled"
        / f"seed{split_seed}"
        / f"era5_xlag_seed{split_seed}.npz"
    )
    data = load_controlled_grid(
        root=data_root,
        task="task_2",
        controlled_npz=controlled,
        ms=int(args["ms"]),
        xlag_length=int(args.get("xlag_length", 10)),
    )
    horizon = temporal_spec_for_block(
        data.times, slice(0, data.times.size), moving=True
    )
    theta = source["learned_theta"]
    model_seed = int(args.get("model_seed", 0))
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    model = BatchRouteBEmpiricalBayes(
        times=data.times,
        spatial_inducing=data.spatial_inducing,
        mt=int(args["mt"]),
        representation="analytic_hippo_rff",
        initial_ell_t=float(theta["ell_t"]),
        initial_ell_s=tuple(theta["ell_s"]),
        initial_kernel_variance=float(theta["kernel_variance"]),
        initial_noise_std=float(theta["noise_std"]),
        rff_sample_size=int(args.get("rff_sample_size", 256)),
        seed=model_seed,
        objective_type="finite_dtc",
        temporal_horizon=horizon,
    )
    model.set_theta(theta)
    started = time.perf_counter()
    metrics, _, persistent_state_bytes = evaluate(
        empirical_model=model,
        data=data,
        posterior_indices=data.train_indices,
        evaluation_indices=data.test_indices,
        representation="analytic_hippo_rff",
        beta_prior_variance=float(args.get("beta_prior_variance", 1000.0)),
        prediction_chunk_size=int(args.get("prediction_chunk_size", 8192)),
        include_conditional_residual_variance=False,
        collect_pointwise=False,
    )
    payload = {
        "method": "task2_batch_empirical_bayes_routeb_strict_dtc_reevaluation",
        "split_seed": split_seed,
        "learned_theta": theta,
        "metrics": metrics,
        "evaluation_seconds": time.perf_counter() - started,
        "persistent_state_bytes": persistent_state_bytes,
        "source_result": str(source_path(base, split_seed)),
        "source_training_seconds": source["timing"]["training_total_seconds"],
        "protocol": {
            "training": "existing 100-step full Task-2 empirical-Bayes fit",
            "objective": "finite/DTC marginal likelihood",
            "prediction_variance": "strict projected variance; no conditional residual",
            "target": "joint X-lag mean plus structured GP residual",
            "note": (
                "Only the predictive variance was re-evaluated. Learned theta and "
                "posterior data are unchanged; RMSE should match the source result."
            ),
        },
    }
    seed_dir = outdir / f"seed{split_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "result.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def aggregate(outdir: Path, split_seeds: list[int]) -> dict[str, Any]:
    results = [
        json.loads((outdir / f"seed{seed}" / "result.json").read_text(encoding="utf-8"))
        for seed in split_seeds
    ]
    rows = []
    for result in results:
        row = {
            "split_seed": result["split_seed"],
            "rmse": result["metrics"]["rmse"],
            "nll": result["metrics"]["nll"],
            "coverage90": result["metrics"]["coverage90"],
            "ece": result["metrics"]["ece"],
            "evaluation_seconds": result["evaluation_seconds"],
            "source_training_seconds": result["source_training_seconds"],
            "persistent_state_mib": result["persistent_state_bytes"] / 1024.0**2,
        }
        rows.append(row)
    write_csv(rows, outdir / "per_seed.csv")
    summary: dict[str, Any] = {
        "experiment": "Task-2 batch empirical-Bayes strict-DTC re-evaluation",
        "n_seeds": len(rows),
    }
    for metric in (
        "rmse",
        "nll",
        "coverage90",
        "ece",
        "evaluation_seconds",
        "source_training_seconds",
        "persistent_state_mib",
    ):
        values = [float(row[metric]) for row in rows]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_sd"] = sample_sd(values)
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    base = args.base.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    for seed in args.split_seeds:
        result_path = outdir / f"seed{seed}" / "result.json"
        if args.skip_existing and result_path.exists():
            continue
        run_seed(base=base, outdir=outdir, data_root=args.root, split_seed=seed)
    aggregate(outdir, args.split_seeds)


if __name__ == "__main__":
    main()
