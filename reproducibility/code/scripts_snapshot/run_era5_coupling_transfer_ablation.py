#!/usr/bin/env python3
"""Run and audit the single-seed ERA5 coupling x transfer ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_iclr_era5_routeb_strict_online.py"


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cell_name(coupling: str, transfer: bool) -> str:
    return f"{coupling}_{'transfer' if transfer else 'no_transfer'}"


def run_cell(args: argparse.Namespace, scope: str, coupling: str, transfer: bool) -> dict:
    cell = cell_name(coupling, transfer)
    out = args.output / scope / "online" / cell / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    protocol_dir = args.protocol_root / ("task1_2" if scope == "short" else "task1_10") / f"seed{args.seed}"
    data_root = args.data_root / ("processed_timeseries_4" if scope == "short" else "processed_timeseries_4_task1_10_extension")
    command = [
        sys.executable,
        str(RUNNER),
        "--protocol-npz", str(protocol_dir / "protocol.npz"),
        "--protocol-json", str(protocol_dir / "protocol.json"),
        "--data-root", str(data_root),
        "--theta-json", str(args.theta_json),
        "--representation", "analytic_hippo_rff",
        "--mt", "128", "--ms", "128", "--rff-sample-size", "256",
        "--temporal-kernel", "spectral_mixture",
        "--spectral-mixture-json", str(args.sm_json),
        "--seed", str(args.seed),
        "--coupling", coupling,
        "--solver-backend", "torch", "--device", "cuda", "--dtype", "float64",
        "--temporal-factor-device", "solver",
        "--output", str(out / "result.json"),
        "--blockwise-output", str(out / "blocks.csv"),
        "--predictions-output", str(out / "predictions.npz"),
    ]
    if transfer:
        command.append("--transfer")
    else:
        command.append("--no-transfer")
    (out / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    with (out / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        (out / "status.json").write_text(json.dumps({"status": "failed", "returncode": completed.returncode}, indent=2) + "\n", encoding="utf-8")
        return {"scope": scope, "coupling": coupling, "transfer": transfer, "status": "failed", "returncode": completed.returncode, "path": str(out)}

    result_path = out / "result.json"
    prediction_path = out / "predictions.npz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(prediction_path) as prediction:
        truth = np.asarray(prediction["y_true"])
        mean = np.asarray(prediction["pred_mean"])
        variance = np.asarray(prediction["pred_var"])
    checks = {
        "shape": list(truth.shape),
        "shape_ok": truth.shape == mean.shape == variance.shape,
        "finite_truth": bool(np.isfinite(truth).all()),
        "finite_mean": bool(np.isfinite(mean).all()),
        "finite_variance": bool(np.isfinite(variance).all()),
        "positive_variance": bool((variance > 0.0).all()),
        "current_hidden_label_reads": int(result.get("current_hidden_label_reads", -1)),
    }
    passed = all(checks[key] for key in ("shape_ok", "finite_truth", "finite_mean", "finite_variance", "positive_variance")) and checks["current_hidden_label_reads"] == 0
    status = {"status": "passed" if passed else "failed_audit", "checks": checks}
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    row = {
        "scope": scope,
        "coupling": coupling,
        "transfer": str(transfer).lower(),
        "cell": cell,
        "seed": args.seed,
        "status": status["status"],
        "path": str(out),
        "rmse": result["overall_current_block"]["rmse"],
        "crps": result["overall_current_block"]["crps"],
        "gaussian_nlpd": result["overall_current_block"]["nll"],
        "ece": result["overall_current_block"]["ece"],
        "coverage90": result["overall_current_block"]["coverage90"],
        "num_blocks": result["num_blocks"],
        "num_stream_times": result["num_stream_times"],
        "peak_cuda_allocated_mib": result["resources"].get("peak_cuda_allocated_mib", ""),
        "shape": str(checks["shape"]),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/era5_coupling_transfer_ablation_20260828_seed0")
    parser.add_argument("--protocol-root", type=Path, default=Path("/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/kronecker+s2vgp/ICLR Formal experiment/iclr_era5_stage2plus/protocol"))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/era5")
    parser.add_argument("--theta-json", type=Path, default=ROOT / "results/era5_stage2plus_sm_baseline_rerun_20260828_formal/calibration/routeb_kronhippo_stgp_sm/seed0/result.json")
    parser.add_argument("--sm-json", type=Path, default=ROOT / "configs/era5_sm_q3.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [
        (scope, coupling, transfer)
        for scope in ("short", "long")
        for coupling in ("mean_field", "structured_joint")
        for transfer in (False, True)
    ]
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        rows = list(executor.map(lambda job: run_cell(args, *job), jobs))
    write_csv(rows, args.output / "coupling_transfer_ablation_summary.csv")
    audit = {
        "status": "complete" if all(row.get("status") == "passed" for row in rows) else "incomplete",
        "cells_expected": 8,
        "cells_recorded": len(rows),
        "seed": args.seed,
        "protocol": "ERA5 Stage2plus strict online; same KronHiPPO-STGP VFE/SM configuration",
        "configuration": {"representation": "analytic_hippo_rff", "training_objective": "vfe", "temporal_kernel": "spectral_mixture", "mt": 128, "ms": 128, "rff": 256},
        "rows": rows,
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
