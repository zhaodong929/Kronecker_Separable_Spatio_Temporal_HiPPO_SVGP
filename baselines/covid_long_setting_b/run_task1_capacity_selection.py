#!/usr/bin/env python3
"""Select eligible COVID Setting B baseline capacities using Task 1 only.

Each candidate is fitted only on the 38 Task-1 fit jurisdictions of seed 0 and
scored only on the four Task-1 validation jurisdictions.  The resulting JSON is
a selection record, not a formal test result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = Path(__file__).resolve().with_name("capacity_policy.json")
METHODS = ("st_svgp", "ovc_svgp", "bui_controlled", "bui_adaptive", "lmc_svgp", "imc_svgp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-npz",
        type=Path,
        default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed0/protocol.npz"),
    )
    parser.add_argument(
        "--theta-json",
        type=Path,
        default=Path(
            "results/diagnostics/covid_long_stream_2020_2024_mandatory/"
            "seed0/routeb_cumulative/calibration/result.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baselines/covid_long_setting_b/results/task1_capacity_selection"),
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--st-python", type=Path, default=Path("baselines/.venvs/st_svgp/bin/python"))
    parser.add_argument("--ovc-python", type=Path, default=Path("baselines/.venvs/ovc_svgp/bin/python"))
    parser.add_argument("--bui-python", type=Path, default=Path(".venv_osgpr/bin/python"))
    parser.add_argument(
        "--factorial-python",
        type=Path,
        default=Path("baselines/.venvs/factorial_sde_gpflow38/bin/python"),
    )
    parser.add_argument("--st-iterations", type=int, default=300)
    parser.add_argument("--ovc-iterations", type=int, default=300)
    parser.add_argument("--ovc-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--bui-adaptive-steps", type=int, default=25)
    parser.add_argument("--factorial-task1-iterations", type=int, default=50)
    parser.add_argument("--factorial-online-steps", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def command_for(
    method: str,
    candidate: dict[str, int],
    args: argparse.Namespace,
    output: Path,
) -> list[str]:
    protocol = resolve(args.protocol_npz)
    if method == "st_svgp":
        return [
            str(resolve(args.st_python)),
            "baselines/covid_long_setting_b/adapters/run_st_svgp.py",
            "--protocol-npz", str(protocol), "--output-dir", str(output), "--seed", "0",
            "--spatial-inducing", str(candidate["spatial_inducing"]),
            "--task1-iterations", str(args.st_iterations), "--task1-validation-only",
        ]
    if method == "ovc_svgp":
        return [
            str(resolve(args.ovc_python)),
            "baselines/covid_long_setting_b/adapters/run_ovc_svgp.py",
            "--protocol-npz", str(protocol), "--output-dir", str(output), "--seed", "0",
            "--temporal-inducing", str(candidate["temporal_inducing"]),
            "--spatial-inducing", str(candidate["spatial_inducing"]),
            "--task1-iterations", str(args.ovc_iterations), "--dtype", args.ovc_dtype,
            "--task1-validation-only",
        ]
    if method in ("lmc_svgp", "imc_svgp"):
        return [
            str(resolve(args.factorial_python)),
            "baselines/covid_long_setting_b/adapters/run_factorial_lmc_imc.py",
            "--protocol-npz", str(protocol), "--output-dir", str(output), "--seed", "0",
            "--method", method.split("_", 1)[0],
            "--temporal-inducing", str(candidate["temporal_inducing"]),
            "--latent-rank", str(candidate["latent_rank"]),
            "--task1-iterations", str(args.factorial_task1_iterations),
            "--online-inference-steps", str(args.factorial_online_steps),
            "--task1-validation-only",
        ]
    command = [
        str(resolve(args.bui_python)), "scripts/run_official_bui_osgpr_era5.py",
        "--protocol-npz", str(protocol), "--output", str(output / "task1_validation.json"),
        "--blockwise-output", str(output / "unused_blocks.csv"), "--mt", str(candidate["temporal_inducing"]),
        "--ms", str(candidate["spatial_inducing"]), "--seed", "0", "--device", "cpu",
        "--dtype", "float64", "--task1-validation-only",
    ]
    if method == "bui_controlled":
        return command + ["--theta-json", str(resolve(args.theta_json))]
    return command + [
        "--adaptive", "--adaptive-calibration-steps", str(args.bui_adaptive_steps),
        "--adaptive-online-steps", "0",
    ]


def candidates(method: str, policy: dict[str, Any]) -> list[dict[str, int]]:
    if method == "st_svgp":
        return policy["families"]["st_svgp"]["candidates"]
    if method in ("lmc_svgp", "imc_svgp"):
        return policy["families"]["lmc_imc_svgp_and_fsde_svi"]["candidates"]
    return policy["families"]["bui_osgpr_and_ovc_svgp"]["candidates"]


def run(command: list[str], output: Path, dry_run: bool) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {"status": "dry_run", "command": command}
    with (output / "run.log").open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        return {"status": "failed", "returncode": completed.returncode, "command": command}
    validation_name = (
        "task1_chronological_validation.json"
        if any("run_factorial_lmc_imc.py" in part for part in command)
        else "task1_validation.json"
    )
    record = json.loads((output / validation_name).read_text(encoding="utf-8"))
    return {"status": record["status"], "command": command, "record": record}


def rank(record: dict[str, Any]) -> tuple[float, float, int]:
    metrics = record["metrics"]
    capacity = record["capacity"]
    if "joint_inducing" in capacity:
        state = int(capacity["joint_inducing"])
    elif "inducing_points" in capacity:
        state = int(capacity["inducing_points"])
    elif "spatial_inducing" in capacity:
        state = int(capacity["spatial_inducing"])
    else:
        state = int(capacity["temporal_inducing"] * capacity["latent_rank"])
    return float(metrics["gaussian_nlpd"]), float(metrics["rmse"]), state


def main() -> None:
    args = parse_args()
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    output_dir = resolve(args.output_dir)
    runs: list[dict[str, Any]] = []
    for method in args.methods:
        for candidate in candidates(method, policy):
            tag = "_".join(f"{key}{value}" for key, value in candidate.items())
            output = output_dir / method / tag
            result = run(command_for(method, candidate, args, output), output, args.dry_run)
            runs.append({"method": method, "candidate": candidate, "output": str(output), **result})

    selected: dict[str, Any] = {}
    for method in args.methods:
        completed = [
            row
            for row in runs
            if row["method"] == method
            and row["status"] in ("task1_validation_complete", "task1_chronological_validation_complete")
        ]
        if completed:
            winner = min(completed, key=lambda row: rank(row["record"]))
            selected[method] = {
                "candidate": winner["candidate"],
                "metrics": winner["record"]["metrics"],
                "output": winner["output"],
            }
        else:
            selected[method] = {"status": "no_completed_task1_validation_candidate"}

    lmc_imc = [row for row in runs if row["method"] in ("lmc_svgp", "imc_svgp") and row["status"] == "task1_chronological_validation_complete"]
    shared_candidates = []
    for candidate in candidates("lmc_svgp", policy):
        group = [row for row in lmc_imc if row["candidate"] == candidate]
        if len(group) == 2:
            shared_candidates.append({
                "candidate": candidate,
                "mean_gaussian_nlpd": sum(float(row["record"]["metrics"]["gaussian_nlpd"]) for row in group) / 2.0,
                "mean_rmse": sum(float(row["record"]["metrics"]["rmse"]) for row in group) / 2.0,
            })
    if shared_candidates:
        selected["lmc_imc_shared"] = min(
            shared_candidates,
            key=lambda row: (row["mean_gaussian_nlpd"], row["mean_rmse"], row["candidate"]["temporal_inducing"] * row["candidate"]["latent_rank"]),
        )

    payload = {
        "status": "complete" if not args.dry_run else "dry_run",
        "protocol": policy["selection_protocol"],
        "selection_rule": policy["common_rules"],
        "runs": runs,
        "selected": selected,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task1_capacity_selection.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
