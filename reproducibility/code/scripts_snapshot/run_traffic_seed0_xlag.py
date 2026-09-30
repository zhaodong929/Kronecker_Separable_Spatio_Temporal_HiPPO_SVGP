#!/usr/bin/env python3
"""Task-1-only PEMS-BAY road-context lag and spatial-kernel comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ROAD = ROOT / "data/traffic/raw/pems_bay/distances_bay_2017.csv"
SPLIT = ROOT / "results/traffic/protocols/pems_bay/pems_bay_seed0_spatial_split.json"


def candidate_name(kernel: str, mixtures: int) -> str:
    return f"{kernel}_q{mixtures}" if kernel == "spectral_mixture" else kernel


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/traffic/seed0_xlag_v1"))
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--validation-every", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-online", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    candidates = [
        ("road_graph", 2),
        ("geo_matern32", 2),
        ("spectral_mixture", 2),
        ("spectral_mixture", 4),
    ]
    rows: list[dict[str, object]] = []
    for kernel, mixtures in candidates:
        name = candidate_name(kernel, mixtures)
        candidate_output = output / "task1_candidates" / name
        theta_path = candidate_output / "theta.json"
        if args.force or not theta_path.exists():
            command = [
                sys.executable,
                str(ROOT / "scripts/calibrate_traffic_task1.py"),
                "--dataset", "pems_bay",
                "--split-manifest", str(SPLIT),
                "--output", str(candidate_output),
                "--task1-steps", "2016",
                "--calibration-stride", "1",
                "--iterations", str(args.iterations),
                "--validation-every", str(args.validation_every),
                "--fixed-temporal-lengthscale", "0.5",
                "--ms", "32",
                "--mt", "128",
                "--rff", "512",
                "--mean-feature-mode", "road_context_xlag",
                "--xlag-length", "10",
                "--context-graph-diffusion", "7.448975327393576",
                "--spatial-kernel", kernel,
                "--spatial-mixtures", str(mixtures),
                "--road-distance-csv", str(ROAD),
                "--graph-diffusion", "7.448975327393576",
                "--device", args.device,
                "--seed", "0",
            ]
            run(command)
        payload = json.loads(theta_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "candidate": name,
                "spatial_kernel": kernel,
                "spatial_mixtures": mixtures if kernel == "spectral_mixture" else "",
                "calibration_status": payload["calibration_status"],
                "best_iteration": payload["best_iteration"],
                "validation_nlpd": payload["best_validation_gaussian_nlpd"],
                "validation_rmse": payload["best_validation_rmse"],
                "wall_clock_seconds": payload["wall_clock_seconds"],
                "theta_path": str(theta_path.relative_to(ROOT)),
            }
        )

    with (output / "task1_kernel_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    converged = [row for row in rows if row["calibration_status"] == "converged"]
    selected = min(converged or rows, key=lambda row: (float(row["validation_nlpd"]), float(row["validation_rmse"])))
    selection = {
        "schema_version": 1,
        "dataset": "pems_bay",
        "seed": 0,
        "task1_steps": 2016,
        "selection_boundary": "Task-1 visible-validation sensors only",
        "selection_metric": "Gaussian NLPD, then RMSE",
        "formal_stream_used_for_selection": False,
        "fixed_configuration": {
            "calibration_stride": 1,
            "temporal_lengthscale_hours": 0.5,
            "ms": 32,
            "mt": 128,
            "rff": 512,
            "mean_feature_mode": "road_context_xlag",
            "xlag_length": 10,
            "context_graph_diffusion": 7.448975327393576,
        },
        "scientific_label": "road-context lag analogue; PEMS-BAY has no independent dynamic exogenous channel",
        "selected": selected,
        "candidates": rows,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.skip_online:
        final_output = output / "final_strict_online_seed0"
        if args.force or not (final_output / "result.json").exists():
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_traffic_routeb.py"),
                    "--dataset", "pems_bay",
                    "--split-manifest", str(SPLIT),
                    "--output", str(final_output),
                    "--protocol", "nowcast",
                    "--method", "hippo",
                    "--mechanism-mode", "changing",
                    "--task1-steps", "2016",
                    "--stream-stride", "1",
                    "--mt", "128",
                    "--ms", "32",
                    "--rff", "512",
                    "--theta-json", str(ROOT / str(selected["theta_path"])),
                    "--road-distance-csv", str(ROAD),
                    "--temporal-evaluator", "scipy_frozen",
                    "--device", args.device,
                    "--dtype", "float64",
                ]
            )
        selection["strict_online_result"] = str((final_output / "result.json").relative_to(ROOT))
        (output / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
