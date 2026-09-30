#!/usr/bin/env python3
"""Reproducibility and leakage audit for the AutoDL Stage 2+ benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(payload: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = payload
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            return current
    return None


def expected_methods(config: dict[str, Any], branch: str) -> list[str]:
    if branch == "batch":
        gpflow = config["gpflow_svgp"]
        return [
            "xlag_mean_only",
            "routeb_residual_analytic_hippo_rff",
            "routeb_residual_inducing_points",
            "routeb_joint_analytic_hippo_rff",
            "routeb_joint_inducing_points",
            f"gpflow_svgp_residual_mt{gpflow['mt']}_ms{gpflow['ms']}",
        ]
    online = config["strict_online"]
    return [
        "xlag_task1_fixed",
        "xlag_recursive_rls",
        "routeb_analytic_hippo_rff",
        "routeb_inducing_points",
        f"bui_osgpr_mt{online['bui_mt']}_ms{online['bui_ms']}",
        f"maddox_streaming_sgpr_mt{online['maddox_mt']}_ms{online['maddox_ms']}",
        f"official_ohsvgp_m{online['ohsvgp_inducing_size']}_rff{online['ohsvgp_rff_sample_size']}",
    ]


def metric_values(y: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    variance = np.maximum(np.asarray(variance, dtype=np.float64).reshape(-1), 1e-10)
    half = 1.6448536269514722 * np.sqrt(variance)
    return {
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "nll": float(
            np.mean(
                0.5
                * (
                    np.log(2.0 * np.pi * variance)
                    + (y - mean) ** 2 / variance
                )
            )
        ),
        "coverage90": float(np.mean((y >= mean - half) & (y <= mean + half))),
    }


def audit_prediction(
    result_path: Path,
    *,
    expected_times: int,
    expected_space: int,
    branch: str,
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    details: dict[str, Any] = {}
    prediction_path = result_path.parent / "predictions.npz"
    if not prediction_path.is_file():
        return ["missing predictions.npz"], details
    try:
        with np.load(prediction_path) as arrays:
            required = ("y_true", "pred_mean", "pred_var")
            for key in required:
                if key not in arrays:
                    issues.append(f"predictions.npz missing {key}")
            if issues:
                return issues, details
            values = {key: np.asarray(arrays[key]) for key in required}
            target_shape = (expected_times, expected_space)
            for key, value in values.items():
                if value.size != expected_times * expected_space:
                    issues.append(
                        f"{key} size {value.size} != {expected_times * expected_space}"
                    )
                if not np.all(np.isfinite(value)):
                    issues.append(f"{key} contains non-finite values")
            if np.any(values["pred_var"] <= 0):
                issues.append("pred_var contains non-positive values")
            details["prediction_shapes"] = {
                key: list(value.shape) for key, value in values.items()
            }
            if not issues:
                recomputed = metric_values(
                    values["y_true"], values["pred_mean"], values["pred_var"]
                )
                details["recomputed"] = recomputed
                payload = read_json(result_path)
                for metric, value in recomputed.items():
                    reported = nested(
                        payload,
                        f"overall_current_block.{metric}",
                        f"final.{metric}",
                        metric,
                    )
                    if reported is None:
                        issues.append(f"result.json missing {metric}")
                    elif abs(float(reported) - value) > 2e-6:
                        issues.append(
                            f"reported {metric} differs from predictions by "
                            f"{abs(float(reported) - value):.3e}"
                        )
                details["expected_shape"] = list(target_shape)
    except (OSError, ValueError) as exc:
        issues.append(f"cannot read predictions.npz: {exc}")
    return issues, details


def audit_run(
    result_path: Path,
    *,
    scope: str,
    branch: str,
    method: str,
    seed: int,
    expected_times: int,
    expected_blocks: int,
) -> dict[str, Any]:
    issues: list[str] = []
    status_path = result_path.parent / "status.json"
    if not result_path.is_file():
        issues.append("missing result.json")
    if not status_path.is_file():
        issues.append("missing status.json")
        status = {}
    else:
        status = read_json(status_path)
        if status.get("status") != "complete":
            issues.append(f"orchestrator status is {status.get('status')}")
    details: dict[str, Any] = {}
    if result_path.is_file():
        payload = read_json(result_path)
        details["device"] = nested(payload, "resources.device")
        details["dtype"] = nested(payload, "resources.dtype")
        details["git_commit"] = nested(payload, "environment.git.commit")
        if payload.get("split_seed", payload.get("seed")) != seed:
            issues.append("seed metadata does not match directory")
        if branch == "online":
            replay = nested(payload, "resources.history_replay_buffer_bytes")
            if replay not in (0, 0.0):
                issues.append(f"strict-online replay buffer is {replay}")
            if int(payload.get("num_blocks", -1)) != expected_blocks:
                issues.append(
                    f"num_blocks={payload.get('num_blocks')} != {expected_blocks}"
                )
            blocks_path = result_path.parent / "blocks.csv"
            if not blocks_path.is_file():
                issues.append("missing blocks.csv")
            else:
                with blocks_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                if len(rows) != expected_blocks:
                    issues.append(f"blocks.csv has {len(rows)} rows")
                if rows:
                    starts = [int(row["block_start"]) for row in rows]
                    stops = [int(row["block_stop"]) for row in rows]
                    if starts[0] != 0 or stops[-1] != expected_times:
                        issues.append("blocks do not cover the full stream")
                    if any(left != right for left, right in zip(stops[:-1], starts[1:])):
                        issues.append("blocks are not contiguous")
        prediction_issues, prediction_details = audit_prediction(
            result_path,
            expected_times=expected_times,
            expected_space=200,
            branch=branch,
        )
        issues.extend(prediction_issues)
        details.update(prediction_details)
    return {
        "scope": scope,
        "branch": branch,
        "method": method,
        "seed": seed,
        "status": "complete" if not issues else "incomplete",
        "issues": issues,
        "result": str(result_path),
        "details": details,
    }


def audit_protocols(
    benchmark: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    issues: list[str] = []
    expected_lengths = {"task1_2": 186, "task1_10": 1674}
    for scope in config["scopes"]:
        for seed in config["split_seeds"]:
            directory = benchmark / "protocol" / scope / f"seed{seed}"
            npz_path = directory / "protocol.npz"
            json_path = directory / "protocol.json"
            row_issues: list[str] = []
            if not npz_path.is_file() or not json_path.is_file():
                row_issues.append("missing protocol artifact")
            else:
                metadata = read_json(json_path)
                actual_hash = sha256(npz_path)
                if metadata.get("npz_sha256") != actual_hash:
                    row_issues.append("protocol SHA-256 mismatch")
                with np.load(npz_path) as arrays:
                    checks = {
                        "stream_times": expected_lengths[scope],
                        "train_indices": 800,
                        "fit_indices": 720,
                        "validation_indices": 80,
                        "test_indices": 200,
                    }
                    for key, size in checks.items():
                        if key not in arrays or arrays[key].size != size:
                            row_issues.append(f"{key} size mismatch")
                    train = set(np.asarray(arrays["train_indices"], dtype=int).tolist())
                    test = set(np.asarray(arrays["test_indices"], dtype=int).tolist())
                    if train & test:
                        row_issues.append("train/test spatial split overlaps")
            rows.append(
                {
                    "scope": scope,
                    "seed": seed,
                    "status": "complete" if not row_issues else "incomplete",
                    "issues": row_issues,
                }
            )
            issues.extend(f"protocol {scope}/seed{seed}: {issue}" for issue in row_issues)
    return rows, issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    protocol_rows, issues = audit_protocols(args.benchmark_root, config)
    expected_lengths = {"task1_2": 186, "task1_10": 1674}
    expected_blocks = {"task1_2": 19, "task1_10": 171}
    runs = []
    for scope in config["scopes"]:
        for branch in ("batch", "online"):
            for method in expected_methods(config, branch):
                for seed in config["split_seeds"]:
                    path = (
                        args.benchmark_root
                        / "runs"
                        / scope
                        / branch
                        / method
                        / f"seed{seed}"
                        / "result.json"
                    )
                    row = audit_run(
                        path,
                        scope=scope,
                        branch=branch,
                        method=method,
                        seed=int(seed),
                        expected_times=expected_lengths[scope],
                        expected_blocks=expected_blocks[scope],
                    )
                    runs.append(row)
                    issues.extend(
                        f"{scope}/{branch}/{method}/seed{seed}: {issue}"
                        for issue in row["issues"]
                    )

    calibration = []
    for representation in ("analytic_hippo_rff", "inducing_points"):
        for seed in config["split_seeds"]:
            path = (
                args.benchmark_root
                / "calibration"
                / f"routeb_joint_{representation}"
                / f"seed{seed}"
                / "result.json"
            )
            row_issues = []
            if not path.is_file():
                row_issues.append("missing Task-1 calibration result")
            else:
                payload = read_json(path)
                if payload.get("data_part") != "calibration":
                    row_issues.append("calibration result does not identify Task 1")
                if "learned_theta" not in payload:
                    row_issues.append("calibration result lacks learned_theta")
            calibration.append(
                {
                    "representation": representation,
                    "seed": seed,
                    "status": "complete" if not row_issues else "incomplete",
                    "issues": row_issues,
                }
            )
            issues.extend(
                f"calibration/{representation}/seed{seed}: {issue}"
                for issue in row_issues
            )

    commits = sorted(
        {
            row["details"].get("git_commit")
            for row in runs
            if row["details"].get("git_commit")
        }
    )
    if len(commits) > 1:
        issues.append(f"runs were produced from multiple repository commits: {commits}")
    expected_modern = {
        (row["scope"], row["branch"], row["method"], int(row["seed"]))
        for row in runs
    }
    compatibility_runs = []
    runs_root = args.benchmark_root / "runs"
    if runs_root.is_dir():
        for status_path in sorted(runs_root.glob("*/*/*/seed*/status.json")):
            relative = status_path.relative_to(runs_root).parts
            if len(relative) != 5:
                continue
            scope, branch, method, seed_part, _ = relative
            try:
                seed = int(seed_part.removeprefix("seed"))
            except ValueError:
                continue
            if (scope, branch, method, seed) in expected_modern:
                continue
            status = read_json(status_path)
            compatibility_runs.append(
                {
                    "scope": scope,
                    "branch": branch,
                    "method": method,
                    "seed": seed,
                    "status": status.get("status", "unknown"),
                    "device_class": status.get("device_class"),
                    "wall_seconds": status.get("wall_seconds"),
                    "log": status.get("log"),
                }
            )
    payload = {
        "schema_version": 1,
        "verification_status": "VERIFIED" if not issues else "INCOMPLETE",
        "benchmark_root": str(args.benchmark_root.resolve()),
        "expected_modern_runs": len(runs),
        "complete_modern_runs": sum(row["status"] == "complete" for row in runs),
        "protocols": protocol_rows,
        "calibration": calibration,
        "runs": runs,
        "repository_commits": commits,
        "compatibility_runs": compatibility_runs,
        "issues": issues,
        "audit_boundaries": [
            "Legacy ST-SVGP and Markovflow failures are classified by status files but are not required for the modern-run completion verdict.",
            "The audit verifies artifacts, metrics, splits, block coverage and zero replay; it does not certify semantic equivalence between different GP approximations.",
            "GPU FLOP estimates are not treated as exact hardware instruction counts.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verification_status": payload["verification_status"],
                "complete_modern_runs": payload["complete_modern_runs"],
                "expected_modern_runs": payload["expected_modern_runs"],
                "issues": len(issues),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
