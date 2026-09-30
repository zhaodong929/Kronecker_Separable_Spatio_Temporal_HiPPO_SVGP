#!/usr/bin/env python3
"""Summarize the unified batch temporal-representation sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPRESENTATIONS = ("analytic_hippo_rff", "inducing_points", "full_temporal_kernel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    sweep = base / "phase_l_temporal_representation_sweep"
    rows = []
    for ms in (64, 128):
        mt_values = (8, 16, 32, 64, 128)
        for mt in mt_values:
            for representation in REPRESENTATIONS[:2]:
                for seed in range(3):
                    path = sweep / f"Ms{ms}" / f"Mt{mt}" / representation / f"seed{seed}" / "run_metadata.json"
                    if not path.exists():
                        continue
                    final = json.loads(path.read_text(encoding="utf-8"))["final"]
                    rows.append(
                        {
                            "ms": ms,
                            "mt": mt,
                            "representation": representation,
                            "seed": seed,
                            **{key: float(final[key]) for key in ("rmse", "nll", "coverage90", "avg_std")},
                        }
                    )
        for representation in ("full_temporal_kernel",):
            for seed in range(3):
                path = sweep / f"Ms{ms}" / "Mt186" / representation / f"seed{seed}" / "run_metadata.json"
                if not path.exists():
                    continue
                final = json.loads(path.read_text(encoding="utf-8"))["final"]
                rows.append(
                    {
                        "ms": ms,
                        "mt": 186,
                        "representation": representation,
                        "seed": seed,
                        **{key: float(final[key]) for key in ("rmse", "nll", "coverage90", "avg_std")},
                    }
                )

    summary = []
    for ms in (64, 128):
        configs = sorted({(row["mt"], row["representation"]) for row in rows if row["ms"] == ms})
        for mt, representation in configs:
            subset = [row for row in rows if row["ms"] == ms and row["mt"] == mt and row["representation"] == representation]
            summary.append(
                {
                    "ms": ms,
                    "mt": mt,
                    "representation": representation,
                    "n_seeds": len(subset),
                    **{
                        f"{key}_mean": float(np.mean([row[key] for row in subset]))
                        for key in ("rmse", "nll", "coverage90", "avg_std")
                    },
                    **{
                        f"{key}_sd": float(np.std([row[key] for row in subset], ddof=1))
                        for key in ("rmse", "nll", "coverage90", "avg_std")
                    },
                }
            )

    paired = []
    for ms in (64, 128):
        for mt in (8, 16, 32, 64, 128):
            for seed in range(3):
                left = next((row for row in rows if row["ms"] == ms and row["mt"] == mt and row["seed"] == seed and row["representation"] == "inducing_points"), None)
                right = next((row for row in rows if row["ms"] == ms and row["mt"] == mt and row["seed"] == seed and row["representation"] == "analytic_hippo_rff"), None)
                if left is None or right is None:
                    continue
                paired.append(
                    {
                        "ms": ms,
                        "mt": mt,
                        "seed": seed,
                        "inducing_minus_hippo_rmse": left["rmse"] - right["rmse"],
                        "inducing_minus_hippo_nll": left["nll"] - right["nll"],
                        "inducing_minus_hippo_coverage90": left["coverage90"] - right["coverage90"],
                    }
                )

    sweep.mkdir(parents=True, exist_ok=True)
    for name, values in (("per_seed.csv", rows), ("summary.csv", summary), ("paired_inducing_minus_hippo.csv", paired)):
        if not values:
            continue
        with (sweep / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    payload = {
        "protocol": {
            "target": "original scaled y with learned X-lag covariates",
            "time_points": 186,
            "spatial_points": 1000,
            "train_test_space": "800/200",
            "split_seeds": [0, 1, 2],
            "kernel": "temporal and separable spatial Matern-3/2",
            "spatial_inducing_coordinates": "fixed split-specific coordinates",
            "update_protocol": "batch/full-history",
            "only_changed_factor": "temporal representation and nominal Mt",
        },
        "summary": summary,
        "paired": paired,
    }
    (sweep / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
