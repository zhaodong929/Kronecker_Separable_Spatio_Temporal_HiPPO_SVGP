#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stvgp_kronecker.joint_ssgp_kron.ssgp_transfer import compute_Lt
from stvgp_kronecker.joint_ssgp_kron.synthetic import covariance_kernel, temporal_inducing_for_block


BACKENDS = ("analytic_hippo_rff", "inducing_points", "inducing_points_global")
CAPACITIES = ((8, 64), (32, 128))
SEEDS = (0, 1, 2)


def sample_sd(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def transfer_spectrum_diagnostic() -> pd.DataFrame:
    times = np.arange(186, dtype=float) / 185.0
    blocks = [slice(start, min(186, start + 10)) for start in range(0, 186, 10)]
    rows = []
    for mt, ms in CAPACITIES:
        for mode, moving in (("moving", True), ("global", False)):
            inducing = [temporal_inducing_for_block(times, block, mt, moving=moving) for block in blocks]
            product = np.eye(mt)
            step_min_singular_values = []
            step_effective_ranks = []
            for old, new in zip(inducing[:-1], inducing[1:]):
                k_new = covariance_kernel(new, lengthscale=0.05, variance=1.0, kernel_type="rbf")
                k_new = k_new + 1e-6 * np.eye(mt)
                k_old_new = covariance_kernel(old, new, lengthscale=0.05, variance=1.0, kernel_type="rbf")
                transfer = compute_Lt(k_old_new, k_new)
                singular_values = np.linalg.svd(transfer, compute_uv=False)
                step_min_singular_values.append(float(singular_values[-1]))
                step_effective_ranks.append(int(np.sum(singular_values > 1e-3)))
                product = product @ transfer
            product_singular_values = np.linalg.svd(product, compute_uv=False)
            rows.append(
                {
                    "mt": mt,
                    "ms": ms,
                    "inducing_mode": mode,
                    "num_transfers": len(blocks) - 1,
                    "median_step_min_singular_value": float(np.median(step_min_singular_values)),
                    "median_step_effective_rank": float(np.median(step_effective_ranks)),
                    "cumulative_max_singular_value": float(product_singular_values[0]),
                    "cumulative_min_singular_value": float(product_singular_values[-1]),
                    "cumulative_effective_rank": int(np.sum(product_singular_values > 1e-3)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()

    per_seed_rows: list[dict[str, float | int | str]] = []
    block_rows: list[pd.DataFrame] = []
    for mt, ms in CAPACITIES:
        for seed in SEEDS:
            for backend in BACKENDS:
                run = base / f"Mt{mt}_Ms{ms}" / "block10" / f"seed{seed}" / backend
                metrics_path = run / "era5_routeb_metrics.csv"
                if not metrics_path.exists():
                    raise FileNotFoundError(metrics_path)
                metrics = pd.read_csv(metrics_path).sort_values("block_id")
                if len(metrics) != 19:
                    raise ValueError(f"Expected 19 blocks in {metrics_path}, found {len(metrics)}")
                metrics.insert(0, "backend", backend)
                metrics.insert(0, "split_seed", seed)
                metrics.insert(0, "ms_config", ms)
                metrics.insert(0, "mt_config", mt)
                block_rows.append(metrics)

                final = metrics.iloc[-1]
                per_seed_rows.append(
                    {
                        "mt": mt,
                        "ms": ms,
                        "split_seed": seed,
                        "backend": backend,
                        "num_blocks": len(metrics),
                        "final_rmse": float(final.rmse),
                        "final_nll": float(final.nll),
                        "final_coverage90": float(final.coverage90),
                        "final_avg_std": float(final.avg_std),
                        "block_mean_rmse": float(metrics.rmse.mean()),
                        "block_mean_nll": float(metrics.nll.mean()),
                        "block_mean_coverage90": float(metrics.coverage90.mean()),
                        "block_mean_runtime": float(metrics.runtime_per_block.mean()),
                    }
                )

    per_seed = pd.DataFrame(per_seed_rows)
    blocks = pd.concat(block_rows, ignore_index=True)
    metric_cols = [
        "final_rmse",
        "final_nll",
        "final_coverage90",
        "final_avg_std",
        "block_mean_rmse",
        "block_mean_nll",
        "block_mean_coverage90",
        "block_mean_runtime",
    ]
    summary_rows = []
    for (mt, ms, backend), group in per_seed.groupby(["mt", "ms", "backend"], sort=True):
        row: dict[str, float | int | str] = {
            "mt": int(mt),
            "ms": int(ms),
            "backend": str(backend),
            "n_seeds": len(group),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_sd"] = sample_sd(group[col])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    wide = per_seed.pivot(index=["mt", "ms", "split_seed"], columns="backend", values=metric_cols)
    paired_rows = []
    for index, row in wide.iterrows():
        mt, ms, seed = index
        out: dict[str, float | int] = {"mt": mt, "ms": ms, "split_seed": seed}
        for backend in ("inducing_points", "inducing_points_global"):
            for metric in metric_cols:
                out[f"{backend}_minus_hippo_{metric}"] = float(
                    row[(metric, backend)] - row[(metric, "analytic_hippo_rff")]
                )
        paired_rows.append(out)
    paired = pd.DataFrame(paired_rows)

    trajectory = (
        blocks.groupby(["mt_config", "ms_config", "backend", "block_id"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", sample_sd),
            nll_mean=("nll", "mean"),
            nll_sd=("nll", sample_sd),
            coverage90_mean=("coverage90", "mean"),
            coverage90_sd=("coverage90", sample_sd),
            runtime_mean=("runtime_per_block", "mean"),
        )
        .rename(columns={"mt_config": "mt", "ms_config": "ms"})
    )

    per_seed.to_csv(base / "per_seed_summary.csv", index=False)
    summary.to_csv(base / "summary.csv", index=False)
    paired.to_csv(base / "paired_inducing_minus_hippo.csv", index=False)
    trajectory.to_csv(base / "block_trajectories.csv", index=False)
    transfer_diagnostic = transfer_spectrum_diagnostic()
    transfer_diagnostic.to_csv(base / "transfer_spectrum_diagnostic.csv", index=False)
    payload = {
        "protocol": {
            "task": "ERA5 task_2 variable 0",
            "spatial_holdout": "800 train / 200 test locations",
            "split_seeds": list(SEEDS),
            "block_size": 10,
            "num_blocks": 19,
            "update": "genuine block-to-block posterior transfer",
            "mean": "medium_era5_xlag, L=10",
            "kernel": "RBF, ell_t=0.05, ell_s=0.35, variance=1.0, noise=0.1",
            "only_changed_factor": "temporal backend",
        },
        "summary": summary.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
        "transfer_spectrum_diagnostic": transfer_diagnostic.to_dict(orient="records"),
    }
    (base / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
