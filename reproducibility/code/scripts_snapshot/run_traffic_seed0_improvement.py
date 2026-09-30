#!/usr/bin/env python3
"""Sequential Task-1-only capacity search for the PEMS-BAY seed-0 HiPPO model."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def candidate_name(config: dict[str, object]) -> str:
    parts = []
    for key in ("stride", "ell_t", "ms", "mt", "rff", "spatial_kernel"):
        value = str(config[key]).replace(".", "p")
        parts.append(f"{key}-{value}")
    return "__".join(parts)


def run_candidate(
    *,
    config: dict[str, object],
    stage: str,
    output_root: Path,
    iterations: int,
    validation_every: int,
    force: bool,
) -> dict[str, object]:
    output = output_root / "candidates" / stage / candidate_name(config)
    theta_path = output / "theta.json"
    if force or not theta_path.exists():
        command = [
            sys.executable,
            str(ROOT / "scripts/calibrate_traffic_task1.py"),
            "--dataset", "pems_bay",
            "--split-manifest", str(ROOT / "results/traffic/protocols/pems_bay/pems_bay_seed0_spatial_split.json"),
            "--output", str(output),
            "--task1-steps", "2016",
            "--calibration-stride", str(config["stride"]),
            "--iterations", str(iterations),
            "--validation-every", str(validation_every),
            "--mt", str(config["mt"]),
            "--ms", str(config["ms"]),
            "--rff", str(config["rff"]),
            "--fixed-temporal-lengthscale", str(config["ell_t"]),
            "--spatial-kernel", str(config["spatial_kernel"]),
            "--device", "cpu",
            "--seed", "0",
        ]
        if config["spatial_kernel"] == "road_graph":
            command.extend([
                "--road-distance-csv",
                str(ROOT / "data/traffic/raw/pems_bay/distances_bay_2017.csv"),
                "--graph-diffusion", str(config.get("graph_diffusion", 1.0)),
            ])
        elif config["spatial_kernel"] == "spectral_mixture":
            command.extend(["--spatial-mixtures", str(config.get("spatial_mixtures", 2))])
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "8")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    payload = json.loads(theta_path.read_text(encoding="utf-8"))
    return {
        "stage": stage,
        **config,
        "calibration_status": payload["calibration_status"],
        "validation_nlpd": payload["best_validation_gaussian_nlpd"],
        "validation_rmse": payload["best_validation_rmse"],
        "best_iteration": payload["best_iteration"],
        "wall_clock_seconds": payload["wall_clock_seconds"],
        "theta_path": str(theta_path.resolve().relative_to(ROOT)),
        "learned_theta": payload["learned_theta"],
    }


def select(rows: list[dict[str, object]]) -> dict[str, object]:
    converged = [row for row in rows if row["calibration_status"] == "converged"]
    pool = converged or rows
    return min(pool, key=lambda row: (float(row["validation_nlpd"]), float(row["validation_rmse"])))


def write_records(rows: list[dict[str, object]], output_root: Path) -> None:
    serialised = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "learned_theta"}
        flat.update({f"theta_{key}": value for key, value in row["learned_theta"].items()})
        serialised.append(flat)
    fields = sorted({key for row in serialised for key in row})
    with (output_root / "candidate_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(serialised)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/traffic/seed0_improvement_v1"))
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--validation-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--through-stage", choices=["A", "B", "C", "D", "E"], default="E")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    base = {"stride": 3, "ell_t": 0.25, "ms": 32, "mt": 32, "rff": 256, "spatial_kernel": "geo_matern32"}
    all_rows: list[dict[str, object]] = []
    stage_rows: dict[str, list[dict[str, object]]] = {}

    configs_a = [
        {**base, "stride": stride, "ell_t": ell_t}
        for stride in (1, 3)
        for ell_t in (0.083, 0.25, 0.5, 1.0, 2.0)
    ]
    stage_rows["A"] = [run_candidate(config=c, stage="A_stride_ell_t", output_root=args.output, iterations=args.iterations, validation_every=args.validation_every, force=args.force) for c in configs_a]
    all_rows.extend(stage_rows["A"])
    selected = select(stage_rows["A"])
    if args.through_stage == "A":
        final_stage = "A"
    else:
        configs_b = [{**base, "stride": selected["stride"], "ell_t": selected["ell_t"], "ms": ms} for ms in (32, 64, 128, 260)]
        stage_rows["B"] = [run_candidate(config=c, stage="B_ms", output_root=args.output, iterations=args.iterations, validation_every=args.validation_every, force=args.force) for c in configs_b]
        all_rows.extend(stage_rows["B"])
        selected = select(stage_rows["B"])
        final_stage = "B"
    if args.through_stage in {"C", "D", "E"}:
        configs_c = [{**base, "stride": selected["stride"], "ell_t": selected["ell_t"], "ms": selected["ms"], "mt": mt} for mt in (32, 64, 128)]
        stage_rows["C"] = [run_candidate(config=c, stage="C_mt", output_root=args.output, iterations=args.iterations, validation_every=args.validation_every, force=args.force) for c in configs_c]
        all_rows.extend(stage_rows["C"])
        selected = select(stage_rows["C"])
        final_stage = "C"
    if args.through_stage in {"D", "E"}:
        configs_d = [{**base, "stride": selected["stride"], "ell_t": selected["ell_t"], "ms": selected["ms"], "mt": selected["mt"], "rff": rff} for rff in (256, 512)]
        stage_rows["D"] = [run_candidate(config=c, stage="D_rff", output_root=args.output, iterations=args.iterations, validation_every=args.validation_every, force=args.force) for c in configs_d]
        all_rows.extend(stage_rows["D"])
        selected = select(stage_rows["D"])
        final_stage = "D"
    if args.through_stage == "E":
        configs_e = [
            {**base, "stride": selected["stride"], "ell_t": selected["ell_t"], "ms": selected["ms"], "mt": selected["mt"], "rff": selected["rff"], "spatial_kernel": "geo_matern32"},
            {**base, "stride": selected["stride"], "ell_t": selected["ell_t"], "ms": selected["ms"], "mt": selected["mt"], "rff": selected["rff"], "spatial_kernel": "road_graph", "graph_diffusion": 1.0},
            {**base, "stride": selected["stride"], "ell_t": selected["ell_t"], "ms": selected["ms"], "mt": selected["mt"], "rff": selected["rff"], "spatial_kernel": "spectral_mixture", "spatial_mixtures": 2},
        ]
        stage_rows["E"] = [run_candidate(config=c, stage="E_spatial_kernel", output_root=args.output, iterations=args.iterations, validation_every=args.validation_every, force=args.force) for c in configs_e]
        all_rows.extend(stage_rows["E"])
        selected = select(stage_rows["E"])
        final_stage = "E"

    write_records(all_rows, args.output)
    selection = {
        "schema_version": 1,
        "selection_boundary": "Task-1 visible-validation sensors only; full 2016-step validation timeline",
        "selection_metric": "Gaussian NLPD, then RMSE",
        "formal_stream_used_for_selection": False,
        "completed_through_stage": final_stage,
        "selected": selected,
        "stage_winners": {stage: select(rows) for stage, rows in stage_rows.items()},
    }
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_records(all_rows, args.output)
    decision = {
        "schema_version": 1,
        "selection_metric": "full Task-1 visible-validation Gaussian NLPD, then RMSE",
        "formal_stream_used_for_selection": False,
        "completed_through_stage": final_stage,
        "selected_candidate": selected,
        "stage_selections": {stage: select(rows) for stage, rows in stage_rows.items()},
    }
    (args.output / "selection.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
