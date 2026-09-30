#!/usr/bin/env python3
"""Summarize the controlled non-HiPPO temporal representation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    base = outdir.parent
    rows = []
    for seed in range(3):
        inducing_metadata = json.loads(
            (outdir / f"seed{seed}" / "routeb_inducing_Mt128_Ms128" / "run_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        hippo_metadata = json.loads(
            (
                base
                / "phase_i_routeb_temporal_capacity"
                / f"seed{seed}"
                / "routeb_Mt128_Ms128"
                / "run_metadata.json"
            ).read_text(encoding="utf-8")
        )
        official = json.loads(
            (
                base
                / "phase_d_joint_xlag_controlled"
                / f"seed{seed}"
                / "official_st_svgp_Ms128"
                / "result.json"
            ).read_text(encoding="utf-8")
        )
        inducing = inducing_metadata["final"]
        hippo = hippo_metadata["final"]
        rows.append(
            {
                "seed": seed,
                **{
                    f"inducing_{key}": float(inducing[key])
                    for key in ("rmse", "nll", "coverage90", "avg_std")
                },
                **{
                    f"hippo_{key}": float(hippo[key])
                    for key in ("rmse", "nll", "coverage90", "avg_std")
                },
                **{
                    f"official_{key}": float(official[key])
                    for key in ("rmse", "nll", "coverage90")
                },
            }
        )
    summary = {}
    for method in ("inducing", "hippo", "official"):
        summary[method] = {
            key: {
                "mean": float(np.mean([row[f"{method}_{key}"] for row in rows])),
                "sd_sample": float(
                    np.std([row[f"{method}_{key}"] for row in rows], ddof=1)
                ),
            }
            for key in ("rmse", "nll", "coverage90")
        }
    paired_differences = {
        comparison: {
            key: {
                "mean": float(np.mean(values)),
                "sd_sample": float(np.std(values, ddof=1)),
                "per_seed": [float(value) for value in values],
            }
            for key in ("rmse", "nll", "coverage90")
            for values in [
                np.asarray(
                    [row[f"{left}_{key}"] - row[f"{right}_{key}"] for row in rows]
                )
            ]
        }
        for comparison, left, right in (
            ("inducing_minus_hippo", "inducing", "hippo"),
            ("inducing_minus_official", "inducing", "official"),
            ("hippo_minus_official", "hippo", "official"),
        )
    }
    payload = {
        "protocol": {
            "temporal_representation": "ordinary temporal inducing points",
            "mt": 128,
            "ms": 128,
            "phi_mode": "medium_era5_xlag",
            "protocol": "batch/full-history",
            "split_seeds": [0, 1, 2],
        },
        "rows": rows,
        "summary": summary,
        "paired_differences": paired_differences,
    }
    (outdir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
