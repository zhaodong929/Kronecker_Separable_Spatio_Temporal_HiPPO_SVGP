#!/usr/bin/env python3
"""Run the four existing-GP PEMS-BAY Protocol-N baselines on seeds 1--3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ohsvgp", "maddox_streaming_sgpr", "bui_osgpr", "st_svgp")


def complete(path: Path, expected_steps: int) -> bool:
    archive = path / "predictions.npz"
    status_files = (path / "result.json", path / "status.json")
    if not archive.is_file() or not any(item.is_file() for item in status_files):
        return False
    try:
        import numpy as np

        with np.load(archive, allow_pickle=False) as data:
            return np.asarray(data["pred_mean"]).shape == (expected_steps, 65)
    except Exception:
        return False


def command(
    method: str,
    seed: int,
    *,
    env_root: Path,
    protocol_root: Path,
    output_root: Path,
    max_stream_steps: int,
    smoke: bool,
) -> list[str]:
    protocol = protocol_root / f"seed{seed}" / "protocol.npz"
    metadata = protocol.with_suffix(".json")
    theta = ROOT / f"results/traffic/formal_locked_sm_q2_road_context_v1/task1_theta/seed{seed}/theta.json"
    output = output_root / "pems_bay" / "nowcast" / method / f"seed{seed}"
    max_steps = ["--max-blocks", str(max_stream_steps)] if max_stream_steps else []
    if method == "ohsvgp":
        return [
            str(env_root / "routeb/bin/python"),
            "scripts/run_traffic_ohsvgp.py",
            "--protocol-npz", str(protocol),
            "--protocol-json", str(metadata),
            "--theta-json", str(theta),
            "--output-dir", str(output),
            "--inducing-size", "64",
            "--rff-sample-size", "256",
            "--microbatch-size", "325",
            "--subsampling-lag", "10",
            "--task1-block-steps", "12",
            *(["--max-task1-blocks", "2"] if smoke else []),
            "--prediction-chunk-size", "325",
            "--update-steps", "1",
            "--seed", str(seed),
            "--device", "cuda",
            "--dtype", "float64",
            *max_steps,
        ]
    if method == "maddox_streaming_sgpr":
        return [
            str(env_root / "maddox/bin/python"),
            "scripts/run_official_maddox_streaming_sgpr_era5.py",
            "--protocol-npz", str(protocol),
            "--theta-json", str(theta),
            "--output", str(output / "result.json"),
            "--blockwise-output", str(output / "blocks.csv"),
            "--predictions-output", str(output / "predictions.npz"),
            "--mt", "4", "--ms", "32",
            "--resample-ratio", "0.2",
            "--jitter", "0.001",
            "--task1-warm-start",
            "--task1-block-steps", "12",
            "--delayed-observations",
            "--seed", str(seed),
            "--device", "cuda",
            "--dtype", "float64",
            *max_steps,
        ]
    if method == "bui_osgpr":
        calibration_blocks = "2" if smoke else "0"
        return [
            str(env_root / "gpflow/bin/python"),
            "scripts/run_official_bui_osgpr_era5.py",
            "--protocol-npz", str(protocol),
            "--theta-json", str(theta),
            "--output", str(output / "result.json"),
            "--blockwise-output", str(output / "blocks.csv"),
            "--predictions-output", str(output / "predictions.npz"),
            "--mt", "4", "--ms", "32",
            "--task1-posterior-warm-start",
            "--delayed-observations",
            "--max-calibration-blocks", calibration_blocks,
            "--max-stream-blocks", str(max_stream_steps),
            "--seed", str(seed),
            "--device", "cuda",
            "--dtype", "float64",
        ]
    iterations = 10 if smoke else 2500
    return [
        str(env_root / "stvgp_legacy/bin/python"),
        "scripts/run_traffic_st_svgp.py",
        "--protocol-npz", str(protocol),
        "--protocol-json", str(metadata),
        "--output-dir", str(output),
        "--spatial-inducing", "32",
        "--task1-iterations", str(iterations),
        "--task1-check-interval", str(min(250, iterations)),
        "--task1-min-steps", str(iterations if smoke else 2500),
        "--online-inference-steps", "1",
        "--history-window", "12",
        "--max-weeks", str(max_stream_steps),
        "--seed", str(seed),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--env-root", type=Path, default=Path.home() / "stvgp_envs")
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=ROOT / "results/traffic/protocol_n_external_gp",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/traffic/formal_external_gp_a100_v1",
    )
    parser.add_argument("--max-stream-steps", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(seed not in (1, 2, 3) for seed in args.seeds):
        raise ValueError("Formal external-GP runs are restricted to seeds 1, 2 and 3")
    expected_steps = args.max_stream_steps or 50100
    records: list[dict[str, object]] = []
    for method in args.methods:
        for seed in args.seeds:
            destination = args.output / "pems_bay" / "nowcast" / method / f"seed{seed}"
            argv = command(
                method,
                seed,
                env_root=args.env_root,
                protocol_root=args.protocol_root,
                output_root=args.output,
                max_stream_steps=args.max_stream_steps,
                smoke=args.smoke,
            )
            row: dict[str, object] = {"method": method, "seed": seed, "command": argv}
            if complete(destination, expected_steps):
                row["status"] = "skipped_complete"
            elif args.dry_run:
                row["status"] = "planned"
            else:
                destination.mkdir(parents=True, exist_ok=True)
                log = destination / "run.log"
                with log.open("a", encoding="utf-8") as handle:
                    completed = subprocess.run(argv, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                row.update({"status": "complete" if completed.returncode == 0 else "failed", "returncode": completed.returncode, "log": str(log)})
            records.append(row)
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "RUN_STATUS.json").write_text(
                json.dumps(
                    {"updated_utc": datetime.now(timezone.utc).isoformat(), "records": records},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if row["status"] == "failed":
                raise SystemExit(f"{method} seed {seed} failed; see {log}")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
