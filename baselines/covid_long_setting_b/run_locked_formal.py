#!/usr/bin/env python3
"""Execute only configurations frozen in ``BASELINE_FAIRNESS_PROTOCOL.json``."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.covid_long_setting_b.formalization import (
    canonical_json_sha256,
    sha256_file,
    validate_prediction_archive,
    verify_snapshot,
)


METHODS = ("ohsvgp_rbf", "ovc_svgp", "st_svgp", "lmc_svgp", "imc_svgp", "fsde_svi")
CONVERGED = {"converged_elbo_plateau", "converged_objective_plateau"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fairness-protocol",
        type=Path,
        default=Path("baselines/covid_long_setting_b/BASELINE_FAIRNESS_PROTOCOL.json"),
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory"),
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--factorial-batch-size", type=int, default=16)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_lock(lock: dict[str, Any]) -> None:
    recorded = lock.get("lock_sha256")
    candidate = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if recorded != canonical_json_sha256(candidate):
        raise ValueError("Fairness lock integrity check failed")
    if lock.get("status") != "locked_before_formal_seeds":
        raise ValueError("Formal runs require a pre-formal locked fairness protocol")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    if current_commit != lock.get("source_commit"):
        raise ValueError("Current source commit differs from the fairness lock")
    for relative, expected_hash in lock.get("locked_code_sha256", {}).items():
        actual_hash = sha256_file(ROOT / relative)
        if actual_hash != expected_hash:
            raise ValueError(f"Locked implementation changed: {relative}")
    if lock.get("formal_seeds") != [5, 6, 7, 8, 9]:
        raise ValueError("The repaired formal runner is restricted to seeds 5--9")
    if not lock.get("methods") or not set(lock["methods"]).issubset(METHODS):
        raise ValueError("Fairness lock has no valid repaired baseline selection")
    hardware = lock.get("hardware", {})
    if hardware.get("status") != "passed":
        raise ValueError("Recorded hardware gate is not a passing RTX 4090 / 120 GiB result")
    required_gates = {
        "ohsvgp_rbf": "ohsvgp_official_reproduction",
        "ovc_svgp": "ovc_memory_assessment",
    }
    for method, path_key in required_gates.items():
        if method not in lock["methods"]:
            continue
        gate = read_json(Path(lock["gate_evidence"][path_key]))
        if gate.get("status") != "passed":
            raise ValueError(f"Gate no longer passes: {path_key}")
    frozen = read_json(Path(lock["gate_evidence"]["frozen_pre_repair_archives"]))
    mismatches = verify_snapshot(frozen.get("archives", []))
    if mismatches:
        raise ValueError(f"Frozen preliminary archive changed: {mismatches[:3]}")


def validate_locked_environments(lock: dict[str, Any], methods: list[str]) -> None:
    """Reject an environment whose installed package set drifted after locking."""

    names = {lock["methods"][method]["environment"] for method in methods}
    for name in names:
        record = lock["environment_lock"]["environments"][name]
        completed = subprocess.run(
            [str(record["python"]), "-m", "pip", "freeze"], text=True, capture_output=True, check=True
        )
        actual = hashlib.sha256(completed.stdout.strip().encode("utf-8")).hexdigest()
        if actual != record["pip_freeze_sha256"]:
            raise ValueError(f"Environment package lock drifted: {name}")


def validate_live_hardware(lock: dict[str, Any]) -> None:
    """Check that the node executing the locked commands is the intended node."""

    required = lock["hardware"]["requirements"]
    ram_gib = float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1024**3
    if ram_gib < float(required["minimum_host_ram_gib"]):
        raise ValueError("Current node has less host RAM than the locked requirement")
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    devices = [row.split(",") for row in completed.stdout.splitlines() if "," in row]
    for name, memory in devices:
        try:
            if "RTX 4090" in name and float(memory.strip()) >= float(required["minimum_gpu_memory_mib"]):
                return
        except ValueError:
            continue
    raise ValueError("Current node does not expose the locked RTX 4090 capacity")


def python_for(lock: dict[str, Any], method: str) -> str:
    environment_name = lock["methods"][method]["environment"]
    try:
        return str(lock["environment_lock"]["environments"][environment_name]["python"])
    except KeyError as error:
        raise ValueError(f"No locked Python interpreter for {method}") from error


def command_for(lock: dict[str, Any], method: str, protocol: Path, output: Path, seed: int, batch_size: int) -> list[str]:
    config = lock["methods"][method]["configuration"]
    common = ["--protocol-npz", str(protocol), "--protocol-json", str(protocol.with_suffix(".json")), "--output-dir", str(output), "--seed", str(seed)]
    python = python_for(lock, method)
    plateau = [
        "--task1-iterations", "50000", "--task1-check-interval", "250", "--task1-min-steps", "2500",
        "--task1-plateau-checks", "10", "--task1-plateau-relative-improvement", "0.001",
    ]
    if method in ("lmc_svgp", "imc_svgp"):
        return [
            python, "baselines/covid_long_setting_b/adapters/run_factorial_lmc_imc.py", *common,
            "--method", method.split("_", 1)[0],
            "--device", "gpu",
            "--temporal-inducing", str(config["temporal_inducing"]),
            "--latent-rank", str(config["latent_rank"]),
            "--online-inference-steps", str(config["online_inference_steps"]),
            "--batch-size", str(batch_size), *plateau,
        ]
    if method == "fsde_svi":
        return [
            python, "baselines/covid_long_setting_b/adapters/run_factorial_fsde_svi.py", *common,
            "--device", "gpu",
            "--temporal-inducing", str(config["temporal_inducing"]),
            "--latent-rank", str(config["latent_rank"]),
            "--online-inference-steps", str(config["online_inference_steps"]),
            "--batch-size", str(batch_size), *plateau,
        ]
    if method == "st_svgp":
        return [
            python, "baselines/covid_long_setting_b/adapters/run_st_svgp.py", *common,
            "--spatial-inducing", str(config["spatial_inducing"]), "--online-inference-steps", "5", *plateau,
        ]
    if method == "ovc_svgp":
        return [
            python, "baselines/covid_long_setting_b/adapters/run_ovc_svgp.py", *common,
            "--temporal-inducing", str(config["temporal_inducing"]),
            "--spatial-inducing", str(config["spatial_inducing"]), "--dtype", "float64", *plateau,
        ]
    return [
        python, "scripts/run_covid_ohsvgp_own_theta.py", *common, "--kernel", "rbf",
        "--inducing-size", str(config["inducing_size"]), "--rff-sample-size", str(config["rff_sample_size"]),
        "--calibration-iterations", "50000", "--task1-check-interval", "250", "--task1-min-steps", "2500",
        "--task1-plateau-checks", "10", "--task1-plateau-relative-improvement", "0.001",
        "--delayed-observations", "--device", "cuda", "--dtype", "float64",
    ]


def result_path(output: Path) -> Path:
    for name in ("result.json", "status.json"):
        path = output / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No adapter result/status file in {output}")


def verify_completed_output(output: Path) -> dict[str, Any]:
    archive = output / "predictions.npz"
    validate_prediction_archive(archive)
    result = read_json(result_path(output))
    convergence = result.get("task1_convergence", {})
    if convergence.get("status") not in CONVERGED:
        raise ValueError(f"Task-1 did not reach its objective plateau in {output}")
    audit = result.get("audit", {})
    if audit.get("passed") is not True or audit.get("current_hidden_labels_read") != 0:
        raise ValueError(f"Causal archive audit failed in {output}")
    return {"archive_sha256": sha256_file(archive), "result": str(result_path(output)), "audit": audit}


def main() -> None:
    args = parse_args()
    lock = read_json(absolute(args.fairness_protocol))
    validate_lock(lock)
    methods = list(lock["methods"]) if args.methods is None else list(args.methods)
    unknown = set(methods) - set(lock["methods"])
    if unknown:
        raise ValueError(f"Requested methods are not admitted by the fairness lock: {sorted(unknown)}")
    validate_locked_environments(lock, methods)
    validate_live_hardware(lock)
    protocol_root = absolute(args.protocol_root)
    output_root = Path(lock["formal_result_root"])
    manifest_path = output_root / "formal_run_manifest.json"
    if output_root.exists() and not manifest_path.exists():
        raise FileExistsError(f"Refusing to use non-empty untracked formal root: {output_root}")
    if manifest_path.is_file():
        previous = read_json(manifest_path)
        if previous.get("fairness_lock_sha256") != lock["lock_sha256"]:
            raise ValueError("Existing formal root belongs to a different fairness lock")
        if not args.resume:
            raise FileExistsError("Formal root already has a manifest; pass --resume only to continue interrupted runs")
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for method in methods:
        for seed in lock["formal_seeds"]:
            protocol = protocol_root / f"seed{seed}" / "protocol.npz"
            if not protocol.is_file() or not protocol.with_suffix(".json").is_file():
                raise FileNotFoundError(f"Missing audited formal protocol for seed {seed}")
            output = output_root / f"seed{seed}" / method
            command = command_for(lock, method, protocol, output, seed, args.factorial_batch_size)
            if (output / "predictions.npz").is_file():
                if not args.resume:
                    raise FileExistsError(f"Refusing to overwrite formal archive: {output / 'predictions.npz'}")
                records.append({"method": method, "seed": seed, "status": "reused_verified", **verify_completed_output(output)})
                continue
            record: dict[str, Any] = {"method": method, "seed": seed, "resource_class": lock["methods"][method]["resource_class"], "command": command}
            if not args.execute:
                record["status"] = "planned"
                records.append(record)
                continue
            if output.exists():
                raise FileExistsError(f"Refusing to reuse partial output without a valid archive: {output}")
            output.mkdir(parents=True)
            with (output / "run.log").open("w", encoding="utf-8") as handle:
                completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            if completed.returncode:
                record.update({"status": "failed", "returncode": completed.returncode, "log": str(output / "run.log")})
                records.append(record)
                break
            record.update({"status": "complete", **verify_completed_output(output)})
            records.append(record)
        if records and records[-1]["status"] == "failed":
            break
    status = "planned" if not args.execute else ("complete" if len(records) == len(methods) * 5 and all(row["status"] in ("complete", "reused_verified") for row in records) else "incomplete_or_failed")
    manifest = {
        "status": status,
        "fairness_lock": str(absolute(args.fairness_protocol)),
        "fairness_lock_sha256": lock["lock_sha256"],
        "serial_execution": True,
        "methods": methods,
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "records": len(records), "output": str(output_root)}, indent=2))
    if args.execute and status != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
