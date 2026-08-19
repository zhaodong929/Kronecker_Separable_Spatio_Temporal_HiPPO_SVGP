#!/usr/bin/env python3
"""Run or plan seed-0 blocked development for repaired COVID baselines.

This runner never reads or writes a formal seed-5--9 archive.  It evaluates
one method/configuration on all three frozen chronological folds, restores the
fold-specific original target scale for scoring, and admits a configuration to
selection only after its convergence, causal and numerical gates pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.covid_long_setting_b.development import sha256_file
from scripts.compute_covid_long_final_metric_system import gaussian_metrics_on_common_scale


METHODS = ("lmc", "imc", "fsde", "ohsvgp", "ovc", "st_svgp")
FACTORIAL_GRID = (
    {"temporal_inducing": 16, "latent_rank": 4},
    {"temporal_inducing": 32, "latent_rank": 4},
    {"temporal_inducing": 50, "latent_rank": 4},
    {"temporal_inducing": 32, "latent_rank": 8},
    {"temporal_inducing": 32, "latent_rank": 16},
)
OHSVGP_GRID = tuple(
    {"inducing_size": inducing, "rff_sample_size": rff}
    for inducing in (32, 64)
    for rff in (64, 128, 256)
)
OVC_GRID = tuple({"temporal_inducing": inducing, "spatial_inducing": 32} for inducing in (4, 8, 12))
ST_GRID = tuple({"spatial_inducing": inducing} for inducing in (16, 32, 52))
ONLINE_STEPS = (5, 25, 100)
CONVERGED_STATUSES = {"converged_elbo_plateau", "converged_objective_plateau"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=Path("baselines/covid_long_setting_b/results/convergence_repair_v1/development_protocols/development_manifest.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("baselines/covid_long_setting_b/results/convergence_repair_v1/blocked_development"),
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--phase", choices=("capacity", "online_steps"), default="capacity")
    parser.add_argument("--execute", action="store_true", help="Run commands. The default writes only an execution plan.")
    parser.add_argument("--resume", action="store_true", help="Reuse a completed candidate-fold archive without overwriting it.")
    parser.add_argument("--factorial-python", type=Path, default=Path("baselines/.venvs/factorial_sde_gpflow/bin/python"))
    parser.add_argument("--fsde-python", type=Path, default=Path("baselines/.venvs/factorial_sde_fsde39/bin/python"))
    parser.add_argument("--ovc-python", type=Path, default=Path("baselines/.venvs/ovc_svgp/bin/python"))
    parser.add_argument("--st-python", type=Path, default=Path("baselines/.venvs/st_svgp/bin/python"))
    parser.add_argument("--ohsvgp-python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--task1-max-steps", type=int, default=50000)
    parser.add_argument("--task1-check-interval", type=int, default=250)
    parser.add_argument("--task1-min-steps", type=int, default=2500)
    parser.add_argument("--task1-plateau-checks", type=int, default=10)
    parser.add_argument("--task1-plateau-relative-improvement", type=float, default=1e-3)
    parser.add_argument("--factorial-batch-size", type=int, default=16)
    parser.add_argument("--factorial-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--st-online-steps", type=int, default=5)
    parser.add_argument("--ovc-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--ohsvgp-device", default="cuda")
    parser.add_argument("--ohsvgp-calibration-iterations", type=int, default=50000)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_folds(manifest_path: Path) -> list[Path]:
    payload = json.loads(absolute(manifest_path).read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("selection_metric") != "gaussian_nlpd_log1p_per_100k":
        raise ValueError("The supplied manifest is not the frozen blocked development protocol")
    folds = [Path(item) for item in payload["folds"]]
    if len(folds) != 3 or not all(path.is_file() and path.with_suffix(".json").is_file() for path in folds):
        raise ValueError("Blocked development requires exactly three complete fold protocols")
    return folds


def candidates(method: str, phase: str, selection_root: Path) -> tuple[dict[str, int], ...]:
    if phase == "online_steps":
        if method not in ("lmc", "imc", "fsde"):
            return ()
        record = json.loads((selection_root / "capacity_selection.json").read_text(encoding="utf-8"))
        shared = record.get("selected", {}).get("factorial_lmc_imc_fsde_shared", {})
        if method in shared.get("excluded_methods", []):
            return ()
        chosen = shared.get("candidate")
        if not isinstance(chosen, dict):
            return ()
        return tuple({**chosen, "online_inference_steps": steps} for steps in ONLINE_STEPS)
    if method in ("lmc", "imc", "fsde"):
        return FACTORIAL_GRID
    if method == "ohsvgp":
        return OHSVGP_GRID
    if method == "ovc":
        return OVC_GRID
    return ST_GRID


def tag(candidate: dict[str, int]) -> str:
    return "_".join(f"{key}{value}" for key, value in candidate.items())


def common_task1_args(args: argparse.Namespace) -> list[str]:
    return [
        "--task1-iterations", str(args.task1_max_steps),
        "--task1-check-interval", str(args.task1_check_interval),
        "--task1-min-steps", str(args.task1_min_steps),
        "--task1-plateau-checks", str(args.task1_plateau_checks),
        "--task1-plateau-relative-improvement", str(args.task1_plateau_relative_improvement),
    ]


def command_for(
    method: str,
    candidate: dict[str, int],
    fold: Path,
    output: Path,
    args: argparse.Namespace,
) -> list[str]:
    protocol_json = fold.with_suffix(".json")
    common = ["--protocol-npz", str(fold), "--protocol-json", str(protocol_json), "--output-dir", str(output), "--seed", "0"]
    if method in ("lmc", "imc"):
        return [
            str(absolute(args.factorial_python)),
            "baselines/covid_long_setting_b/adapters/run_factorial_lmc_imc.py",
            *common,
            "--method", method,
            "--device", args.factorial_device,
            "--temporal-inducing", str(candidate["temporal_inducing"]),
            "--latent-rank", str(candidate["latent_rank"]),
            "--online-inference-steps", str(candidate.get("online_inference_steps", 25)),
            "--batch-size", str(args.factorial_batch_size),
            *common_task1_args(args),
        ]
    if method == "fsde":
        return [
            str(absolute(args.fsde_python)),
            "baselines/covid_long_setting_b/adapters/run_factorial_fsde_svi.py",
            *common,
            "--temporal-inducing", str(candidate["temporal_inducing"]),
            "--latent-rank", str(candidate["latent_rank"]),
            "--online-inference-steps", str(candidate.get("online_inference_steps", 25)),
            "--batch-size", str(args.factorial_batch_size),
            *common_task1_args(args),
        ]
    if method == "ovc":
        return [
            str(absolute(args.ovc_python)),
            "baselines/covid_long_setting_b/adapters/run_ovc_svgp.py",
            *common,
            "--temporal-inducing", str(candidate["temporal_inducing"]),
            "--spatial-inducing", str(candidate["spatial_inducing"]),
            "--dtype", args.ovc_dtype,
            *common_task1_args(args),
        ]
    if method == "st_svgp":
        return [
            str(absolute(args.st_python)),
            "baselines/covid_long_setting_b/adapters/run_st_svgp.py",
            *common,
            "--spatial-inducing", str(candidate["spatial_inducing"]),
            "--online-inference-steps", str(args.st_online_steps),
            *common_task1_args(args),
        ]
    return [
        str(absolute(args.ohsvgp_python)),
        "scripts/run_covid_ohsvgp_own_theta.py",
        *common,
        "--kernel", "rbf",
        "--inducing-size", str(candidate["inducing_size"]),
        "--rff-sample-size", str(candidate["rff_sample_size"]),
        "--calibration-iterations", str(args.ohsvgp_calibration_iterations),
        "--task1-check-interval", str(args.task1_check_interval),
        "--task1-min-steps", str(args.task1_min_steps),
        "--task1-plateau-checks", str(args.task1_plateau_checks),
        "--task1-plateau-relative-improvement", str(args.task1_plateau_relative_improvement),
        "--delayed-observations",
        "--device", args.ohsvgp_device,
        "--dtype", "float64",
    ]


def result_json(output: Path) -> Path | None:
    for name in ("result.json", "status.json"):
        path = output / name
        if path.is_file():
            return path
    return None


def read_result(output: Path) -> dict[str, Any]:
    path = result_json(output)
    return {} if path is None else json.loads(path.read_text(encoding="utf-8"))


def convergence_status(record: dict[str, Any]) -> str | None:
    payload = record.get("task1_convergence")
    if isinstance(payload, dict):
        return payload.get("status")
    return None


def adapter_state_gate(method: str, result: dict[str, Any], expected_steps: int) -> tuple[bool, str]:
    """Require the repaired Factorial adapters to prove stateful NGD updates."""

    if method not in ("lmc", "imc", "fsde"):
        return True, "not_applicable"
    convergence = result.get("task1_convergence", {})
    if convergence.get("natural_gradient") is not True:
        return False, "natural_gradient_not_recorded"
    updates = result.get("online_posterior_updates", [])
    if not isinstance(updates, list) or len(updates) != expected_steps:
        return False, "online_posterior_handoff_trace_missing"
    if method == "lmc":
        symmetry = convergence.get("lmc_symmetry_audit", {})
        if not symmetry.get("lengthscale_parameter_ids_distinct") or not symmetry.get("mixing_matrix_trainable"):
            return False, "lmc_parameter_identity_or_trainability_audit_failed"
    if method in ("lmc", "imc"):
        updated = any(
            float(row.get("q_mu_max_abs_update", 0.0)) > 0.0
            or float(row.get("q_sqrt_max_abs_update", 0.0)) > 0.0
            for row in updates[1:]
        )
    else:
        updated = any(
            float(row.get("variational_max_abs_update", 0.0)) > 0.0
            and row.get("frozen_model_parameters_retained") is True
            for row in updates[1:]
        )
    return (True, "passed") if updated else (False, "posterior_state_did_not_change_after_arrived_labels")


def score_fold(method: str, output: Path, fold: Path) -> dict[str, Any]:
    archive_path = output / "predictions.npz"
    result = read_result(output)
    if not archive_path.is_file():
        return {"status": "archive_missing", "result": result}
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name], dtype=np.float64) for name in ("y_true", "pred_mean", "pred_var")}
    metadata = json.loads(fold.with_suffix(".json").read_text(encoding="utf-8"))
    expected_shape = (int(metadata["num_stream_times"]), int(metadata["num_test_locations"]))
    if any(value.shape != expected_shape for value in arrays.values()):
        return {"status": "invalid_shape", "expected_shape": expected_shape, "shapes": {key: value.shape for key, value in arrays.items()}, "result": result}
    if not all(np.isfinite(value).all() for value in arrays.values()) or np.any(arrays["pred_var"] <= 0.0):
        return {"status": "nonfinite_or_nonpositive_variance", "result": result}
    audit = result.get("audit", {}) if isinstance(result.get("audit", {}), dict) else {}
    expected_delayed = max(0, expected_shape[0] - 1) * expected_shape[1]
    causal_ok = (
        audit.get("current_hidden_labels_read") == 0
        and audit.get("delayed_hidden_labels") == expected_delayed
        and audit.get("online_steps_completed") == expected_shape[0]
    )
    convergence = convergence_status(result)
    state_gate_passed, state_gate_reason = adapter_state_gate(method, result, expected_shape[0])
    metrics = gaussian_metrics_on_common_scale(arrays, metadata["target_standardization"], ece_seed=70_000 + int(metadata["fold"]["id"]))
    return {
        "status": "scored" if causal_ok and state_gate_passed else ("causal_audit_failed" if not causal_ok else "adapter_state_gate_failed"),
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "metrics": metrics,
        "task1_convergence_status": convergence,
        "causal_audit": audit,
        "causal_gate_passed": causal_ok,
        "adapter_state_gate_passed": state_gate_passed,
        "adapter_state_gate_reason": state_gate_reason,
        "result": result,
    }


def capacity_size(method: str, candidate: dict[str, int]) -> int:
    if method in ("lmc", "imc", "fsde"):
        return int(candidate["temporal_inducing"] * candidate["latent_rank"])
    if method == "ohsvgp":
        return int(candidate["inducing_size"] * candidate["rff_sample_size"])
    if method == "ovc":
        return int(candidate["temporal_inducing"] * candidate["spatial_inducing"])
    return int(candidate["spatial_inducing"])


def summarize_candidate(method: str, candidate: dict[str, int], folds: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [entry for entry in folds if entry.get("status") == "scored"]
    convergence = [entry.get("task1_convergence_status") for entry in scores]
    converged = len(scores) == 3 and all(item in CONVERGED_STATUSES for item in convergence)
    if len(scores) != 3:
        status = "validation_pending_or_failed"
    elif not converged:
        status = "convergence_pending"
    else:
        status = "eligible_for_selection"
    summary: dict[str, Any] = {
        "method": method,
        "candidate": candidate,
        "capacity_size": capacity_size(method, candidate),
        "status": status,
        "folds": folds,
    }
    if scores:
        for metric in ("rmse", "crps", "native_gaussian_nlpd", "ece", "coverage90"):
            summary[f"mean_{metric}"] = float(np.mean([entry["metrics"][metric] for entry in scores]))
    return summary


def practical_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = sorted((row for row in rows if row["status"] == "eligible_for_selection"), key=lambda item: item["capacity_size"])
    if len(eligible) < 2:
        return {"status": "insufficient_eligible_upper_budget_comparisons"}
    lower, upper = eligible[-2:]
    rmse_relative_change = abs(upper["mean_rmse"] - lower["mean_rmse"]) / max(abs(lower["mean_rmse"]), 1e-12)
    nlpd_change = abs(upper["mean_native_gaussian_nlpd"] - lower["mean_native_gaussian_nlpd"])
    return {
        "status": "practical_stability_passed" if rmse_relative_change <= 0.01 and nlpd_change <= 0.01 else "capacity_unresolved",
        "engineering_criterion": "upper-budget RMSE change <= 1% and Gaussian NLPD change <= 0.01; not a theoretical capacity guarantee",
        "lower_candidate": lower["candidate"],
        "upper_candidate": upper["candidate"],
        "rmse_relative_change": rmse_relative_change,
        "gaussian_nlpd_absolute_change": nlpd_change,
    }


def lmc_imc_cross_audits(phase_root: Path) -> dict[str, dict[str, Any]]:
    """Detect an empirical LMC collapse before its score is eligible."""

    audits: dict[str, dict[str, Any]] = {}
    for candidate in FACTORIAL_GRID:
        candidate_tag = tag(candidate)
        folds: list[dict[str, Any]] = []
        for fold_id in (1, 2, 3):
            lmc_path = phase_root / "lmc" / candidate_tag / f"fold_{fold_id}" / "predictions.npz"
            imc_path = phase_root / "imc" / candidate_tag / f"fold_{fold_id}" / "predictions.npz"
            if not lmc_path.is_file() or not imc_path.is_file():
                folds.append({"fold": fold_id, "status": "archive_missing"})
                continue
            with np.load(lmc_path, allow_pickle=False) as lmc, np.load(imc_path, allow_pickle=False) as imc:
                mean_difference = float(np.max(np.abs(np.asarray(lmc["pred_mean"]) - np.asarray(imc["pred_mean"]))))
                variance_difference = float(np.max(np.abs(np.asarray(lmc["pred_var"]) - np.asarray(imc["pred_var"]))))
            folds.append(
                {
                    "fold": fold_id,
                    "status": "compared",
                    "max_abs_mean_difference": mean_difference,
                    "max_abs_variance_difference": variance_difference,
                }
            )
        compared = [item for item in folds if item["status"] == "compared"]
        collapsed = len(compared) == 3 and all(
            item["max_abs_mean_difference"] < 1e-5 and item["max_abs_variance_difference"] < 1e-6
            for item in compared
        )
        audits[candidate_tag] = {
            "candidate": candidate,
            "folds": folds,
            "status": "empirical_collapse" if collapsed else ("distinct_or_incomplete" if len(compared) == 3 else "pending"),
            "thresholds": {"mean": 1e-5, "variance": 1e-6},
        }
    return audits


def select_shared_factorial(rows: list[dict[str, Any]], lmc_audits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Require a common M,Q pair for LMC, IMC and FSDE before formal work."""

    lmc_survives = any(
        row["method"] == "lmc" and row["status"] == "eligible_for_selection" for row in rows
    )
    required_methods = ("lmc", "imc", "fsde") if lmc_survives else ("imc", "fsde")
    options: list[dict[str, Any]] = []
    for candidate in FACTORIAL_GRID:
        members = [
            row
            for row in rows
            if row["method"] in required_methods
            and row["candidate"] == candidate
            and row["status"] == "eligible_for_selection"
        ]
        if len(members) != len(required_methods):
            continue
        options.append(
            {
                "candidate": candidate,
                "methods": [row["method"] for row in members],
                "mean_gaussian_nlpd": float(np.mean([row["mean_native_gaussian_nlpd"] for row in members])),
                "mean_rmse": float(np.mean([row["mean_rmse"] for row in members])),
                "capacity_size": capacity_size("lmc", candidate),
            }
        )
    if not options:
        return {"status": "no_common_factorial_capacity_passed_all_gates"}
    winner = min(
        options,
        key=lambda item: (item["mean_gaussian_nlpd"], item["mean_rmse"], item["capacity_size"]),
    )
    return {
        "status": "selected",
        **winner,
        "candidates": options,
        "excluded_methods": [] if lmc_survives else ["lmc"],
        "lmc_imc_cross_audit": lmc_audits,
    }


def main() -> None:
    args = parse_args()
    output_root = absolute(args.output_root)
    folds = load_folds(args.development_manifest)
    phase_root = output_root / args.phase
    phase_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    execution_plan: list[dict[str, Any]] = []
    for method in args.methods:
        method_candidates = candidates(method, args.phase, output_root / "capacity")
        for candidate in method_candidates:
            fold_rows: list[dict[str, Any]] = []
            for fold in folds:
                fold_id = json.loads(fold.with_suffix(".json").read_text(encoding="utf-8"))["fold"]["id"]
                output = phase_root / method / tag(candidate) / f"fold_{fold_id}"
                command = command_for(method, candidate, fold, output, args)
                execution_plan.append({"method": method, "candidate": candidate, "fold": str(fold), "output": str(output), "command": command})
                if args.execute and not (args.resume and (output / "predictions.npz").is_file()):
                    if (output / "predictions.npz").exists():
                        raise FileExistsError(f"Refusing to overwrite a development archive: {output / 'predictions.npz'}")
                    output.mkdir(parents=True, exist_ok=True)
                    with (output / "run.log").open("w", encoding="utf-8") as handle:
                        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                    if completed.returncode:
                        fold_rows.append({"status": "adapter_failed", "returncode": completed.returncode, "log": str(output / "run.log")})
                        continue
                fold_rows.append(score_fold(method, output, fold))
            all_rows.append(summarize_candidate(method, candidate, fold_rows))
    cross_audits: dict[str, dict[str, Any]] = {}
    if args.phase == "capacity" and {"lmc", "imc"}.issubset(args.methods):
        cross_audits = lmc_imc_cross_audits(phase_root)
        for row in all_rows:
            if row["method"] != "lmc":
                continue
            audit = cross_audits[tag(row["candidate"])]
            row["lmc_imc_cross_audit"] = audit
            if audit["status"] == "empirical_collapse":
                row["status"] = "empirical_collapse_with_imc"
    selected: dict[str, Any] = {}
    for method in args.methods:
        eligible = [row for row in all_rows if row["method"] == method and row["status"] == "eligible_for_selection"]
        if eligible:
            winner = min(eligible, key=lambda item: (item["mean_native_gaussian_nlpd"], item["mean_rmse"], item["capacity_size"]))
            selected[method] = {"candidate": winner["candidate"], "mean_gaussian_nlpd": winner["mean_native_gaussian_nlpd"], "mean_rmse": winner["mean_rmse"]}
        else:
            selected[method] = {"status": "no_configuration_passed_all_development_gates"}
    if args.phase == "capacity" and {"lmc", "imc", "fsde"}.issubset(args.methods):
        selected["factorial_lmc_imc_fsde_shared"] = select_shared_factorial(all_rows, cross_audits)
    payload = {
        "schema_version": 1,
        "purpose": "seed-0 blocked development only; formal seed-5--9 archives are immutable and excluded",
        "phase": args.phase,
        "selection_metric": "mean Gaussian NLPD on restored log1p(per-100k) scale; mean RMSE tie-breaker",
        "diagnostic_metrics": ["CRPS", "ECE", "Coverage90"],
        "convergence_gate": {
            "check_interval": args.task1_check_interval,
            "minimum_steps": args.task1_min_steps,
            "moving_median_checks": args.task1_plateau_checks,
            "relative_improvement_threshold": args.task1_plateau_relative_improvement,
            "max_steps": args.task1_max_steps,
        },
        "results": all_rows,
        "lmc_imc_cross_audits": cross_audits,
        "selected": selected,
        "practical_capacity_stability": {
            method: practical_stability([row for row in all_rows if row["method"] == method])
            for method in args.methods
        },
    }
    (phase_root / "execution_plan.json").write_text(json.dumps(execution_plan, indent=2) + "\n", encoding="utf-8")
    (phase_root / "capacity_selection.json" if args.phase == "capacity" else phase_root / "online_step_selection.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phase": args.phase, "execute": args.execute, "candidate_records": len(all_rows), "output": str(phase_root)}, indent=2))


if __name__ == "__main__":
    main()
