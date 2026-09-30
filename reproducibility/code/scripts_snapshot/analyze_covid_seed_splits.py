#!/usr/bin/env python3
"""Diagnose COVID spatial splits and Route B seed instability."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    location_rows = []
    theta_rows = []
    for seed in args.seeds:
        protocol_dir = args.protocol_root / f"seed{seed}"
        metadata = json.loads((protocol_dir / "protocol.json").read_text(encoding="utf-8"))
        with np.load(protocol_dir / "protocol.npz") as arrays:
            train = np.asarray(arrays["train_indices"], dtype=int)
            test = np.asarray(arrays["test_indices"], dtype=int)
            coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
            stream_y = np.asarray(arrays["stream_y"], dtype=np.float64)
        distances = np.linalg.norm(
            coordinates[test, None, :] - coordinates[None, train, :], axis=2
        )
        nearest = distances.min(axis=1)
        names = metadata["location_names"]
        for local, location_index in enumerate(test):
            values = stream_y[:, location_index]
            location_rows.append(
                {
                    "seed": seed,
                    "location_index": int(location_index),
                    "location_name": names[location_index],
                    "nearest_visible_distance": float(nearest[local]),
                    "stream_mean": float(values.mean()),
                    "stream_std": float(values.std()),
                    "stream_min": float(values.min()),
                    "stream_max": float(values.max()),
                }
            )
        pilot = json.loads(
            (args.pilot_root / f"seed{seed}" / "pilot_summary.json").read_text(encoding="utf-8")
        )
        routeb = {row["method"]: row for row in pilot["rows"] if row["method"].startswith("routeb_")}
        seed_rows.append(
            {
                "seed": seed,
                "test_locations": "; ".join(names[index] for index in test),
                "mean_nearest_visible_distance": float(nearest.mean()),
                "max_nearest_visible_distance": float(nearest.max()),
                "test_stream_std": float(stream_y[:, test].std()),
                "global_rmse": routeb["routeb_global_inducing"]["rmse"],
                "hippo_rmse": routeb["routeb_cumulative_hippo"]["rmse"],
            }
        )
        for method, label in (("global", "global_inducing"), ("hippo", "cumulative_hippo")):
            result = json.loads(
                (args.pilot_root / f"seed{seed}" / label / "calibration" / "result.json").read_text(encoding="utf-8")
            )
            theta = result["learned_theta"]
            theta_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "best_iteration": result["best_iteration"],
                    "best_validation_nll": result["best_validation_nll"],
                    "ell_t": theta["ell_t"],
                    "ell_s_0": theta["ell_s"][0],
                    "ell_s_1": theta["ell_s"][1],
                    "kernel_variance": theta["kernel_variance"],
                    "noise_std": theta["noise_std"],
                }
            )

    for name, rows in (("seed_summary.csv", seed_rows), ("heldout_locations.csv", location_rows), ("learned_theta.csv", theta_rows)):
        with (args.output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    seed4 = next(row for row in seed_rows if row["seed"] == 4)
    report = [
        "# COVID seed split diagnostic",
        "",
        "| Seed | Mean nearest visible distance | Max nearest distance | Global RMSE | HiPPO RMSE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        report.append(
            f"| {row['seed']} | {row['mean_nearest_visible_distance']:.4f} | {row['max_nearest_visible_distance']:.4f} "
            f"| {row['global_rmse']:.4f} | {row['hippo_rmse']:.4f} |"
        )
    report.extend(["", "## Seed 4 held-out locations", "", seed4["test_locations"], ""])
    (args.output / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
