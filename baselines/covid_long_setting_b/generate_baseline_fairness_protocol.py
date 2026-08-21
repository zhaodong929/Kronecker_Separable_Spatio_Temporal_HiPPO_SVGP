#!/usr/bin/env python3
"""Lock repaired baseline configurations after seed-0 development gates pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.covid_long_setting_b.formalization import canonical_json_sha256, sha256_file, verify_snapshot


CONVERGED = {"converged_elbo_plateau", "converged_objective_plateau"}
METHOD_IDS = ("ohsvgp_rbf", "ovc_svgp", "st_svgp", "lmc_svgp", "imc_svgp", "fsde_svi")
GPU_METHOD_IDS = {"ohsvgp_rbf", "ovc_svgp", "st_svgp", "lmc_svgp", "imc_svgp", "fsde_svi"}
ENVIRONMENT_BY_METHOD = {
    "ohsvgp_rbf": "ohsvgp",
    "ovc_svgp": "ovc",
    "st_svgp": "st_svgp",
    "lmc_svgp": "factorial_gpflow",
    "imc_svgp": "factorial_gpflow",
    "fsde_svi": "factorial_fsde",
}
LOCKED_CODE_FILES = (
    "baselines/covid_long_setting_b/protocol.py",
    "baselines/covid_long_setting_b/archive.py",
    "baselines/covid_long_setting_b/adapters/run_factorial_lmc_imc.py",
    "baselines/covid_long_setting_b/adapters/run_factorial_fsde_svi.py",
    "baselines/covid_long_setting_b/adapters/run_ovc_svgp.py",
    "baselines/covid_long_setting_b/adapters/run_st_svgp.py",
    "scripts/run_covid_ohsvgp_own_theta.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-root",
        type=Path,
        help="Legacy shared development root; cannot be combined with per-method roots.",
    )
    for name in ("factorial", "ohsvgp", "ovc", "st-svgp"):
        parser.add_argument(f"--{name}-development-root", type=Path)
    parser.add_argument(
        "--ohsvgp-gate",
        type=Path,
        default=Path("baselines/covid_long_setting_b/reproduction/convergence_repair_v1/ohsvgp/gate_status.json"),
    )
    parser.add_argument("--ovc-memory-assessment", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--hardware-fingerprint", type=Path, required=True)
    parser.add_argument(
        "--frozen-archives",
        type=Path,
        default=Path("baselines/covid_long_setting_b/reproduction/convergence_repair_v1/frozen_pre_repair_archives.json"),
    )
    parser.add_argument("--catalog", type=Path, default=Path("baselines/covid_long_setting_b/catalog.json"))
    parser.add_argument(
        "--formal-result-root",
        type=Path,
        default=Path("baselines/covid_long_setting_b/results/formal_repaired_4090_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("baselines/covid_long_setting_b/BASELINE_FAIRNESS_PROTOCOL.json"),
    )
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> dict[str, Any]:
    return json.loads(absolute(path).read_text(encoding="utf-8"))


def load_capacity_or_pending(path: Path) -> dict[str, Any]:
    """Keep methods without a completed development root out of the lock."""

    if path.is_file():
        return load(path)
    return {
        "status": "validation_pending_or_failed",
        "purpose": "No completed development capacity record was available before the formal lock.",
        "selected": {},
    }


def maybe_selected(record: dict[str, Any], method: str) -> dict[str, Any] | None:
    item = record.get("selected", {}).get(method)
    if not isinstance(item, dict) or not isinstance(item.get("candidate"), dict):
        return None
    return dict(item["candidate"])


def source_commit(catalog: dict[str, Any], method_id: str) -> str:
    methods = {str(item["id"]): item for item in catalog["methods"]}
    return str(methods[method_id].get("source_commit", "local"))


def development_roots(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve separate seed-0 development roots without mixing their artifacts."""

    names = ("factorial", "ohsvgp", "ovc", "st_svgp")
    supplied = {
        "factorial": args.factorial_development_root,
        "ohsvgp": args.ohsvgp_development_root,
        "ovc": args.ovc_development_root,
        "st_svgp": args.st_svgp_development_root,
    }
    if args.development_root is not None:
        if any(value is not None for value in supplied.values()):
            raise ValueError("Use either --development-root or all four per-method development roots")
        root = absolute(args.development_root)
        return {name: root for name in names}
    missing = [name for name in names if supplied[name] is None]
    if missing:
        raise ValueError(f"Missing per-method development roots: {missing}")
    return {name: absolute(supplied[name]) for name in names}


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def locked_code_hashes() -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in LOCKED_CODE_FILES}


def build_selected_configs(
    capacities: Mapping[str, dict[str, Any]],
    online_steps: dict[str, Any],
    *,
    ohsvgp_gate_passed: bool,
    ovc_memory_passed: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    # Retain the public helper's legacy shared-record input for its focused tests.
    if "factorial" not in capacities:
        capacities = {name: dict(capacities) for name in ("factorial", "ohsvgp", "ovc", "st_svgp")}
    configs: dict[str, dict[str, Any]] = {}
    exclusions: dict[str, str] = {}

    ohsvgp = maybe_selected(capacities["ohsvgp"], "ohsvgp")
    if ohsvgp is None:
        exclusions["ohsvgp_rbf"] = "no_configuration_passed_all_development_gates"
    elif not ohsvgp_gate_passed:
        exclusions["ohsvgp_rbf"] = "official_ohsvgp_reproduction_gate_not_passed"
    else:
        configs["ohsvgp_rbf"] = ohsvgp

    ovc = maybe_selected(capacities["ovc"], "ovc")
    if ovc is None:
        exclusions["ovc_svgp"] = "no_configuration_passed_all_development_gates"
    elif not ovc_memory_passed:
        exclusions["ovc_svgp"] = "ovc_clean_process_memory_audit_not_passed"
    else:
        configs["ovc_svgp"] = ovc

    st_svgp = maybe_selected(capacities["st_svgp"], "st_svgp")
    if st_svgp is None:
        exclusions["st_svgp"] = "no_configuration_passed_all_development_gates"
    else:
        configs["st_svgp"] = st_svgp

    shared = capacities["factorial"].get("selected", {}).get("factorial_lmc_imc_fsde_shared", {})
    shared_candidate = shared.get("candidate") if isinstance(shared, dict) else None
    if shared.get("status") != "selected" or not isinstance(shared_candidate, dict):
        for method_id in ("lmc_svgp", "imc_svgp", "fsde_svi"):
            exclusions[method_id] = "no_common_factorial_capacity_passed_all_gates"
    else:
        excluded = set(shared.get("excluded_methods", []))
        excluded_reasons = shared.get("excluded_reasons", {})
        for short, method_id in (("lmc", "lmc_svgp"), ("imc", "imc_svgp"), ("fsde", "fsde_svi")):
            if short in excluded:
                exclusions[method_id] = str(excluded_reasons.get(short, "empirical_lmc_collapse"))
                continue
            candidate = maybe_selected(online_steps, short)
            if candidate is None:
                exclusions[method_id] = "no_online_posterior_step_configuration_passed_all_gates"
                continue
            if {key: candidate[key] for key in ("temporal_inducing", "latent_rank")} != shared_candidate:
                exclusions[method_id] = "online_step_configuration_does_not_match_shared_capacity"
                continue
            configs[method_id] = candidate
    return configs, exclusions


def main() -> None:
    args = parse_args()
    output = absolute(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace fairness lock: {output}")
    roots = development_roots(args)
    capacities = {
        name: load_capacity_or_pending(root / "capacity" / "capacity_selection.json")
        for name, root in roots.items()
    }
    online_steps = load(roots["factorial"] / "online_steps" / "online_step_selection.json")
    gate = load(args.ohsvgp_gate)
    ovc_assessment = load(args.ovc_memory_assessment)
    environment = load(args.environment_lock)
    configs, exclusions = build_selected_configs(
        capacities,
        online_steps,
        ohsvgp_gate_passed=gate.get("status") == "passed",
        ovc_memory_passed=ovc_assessment.get("status") == "passed",
    )
    if not configs:
        raise ValueError("No baseline passed every development, reproduction and resource gate")
    required_environments = {ENVIRONMENT_BY_METHOD[method] for method in configs}
    if environment.get("status") != "complete" or not required_environments.issubset(environment.get("environments", {})):
        raise ValueError(f"Environment lock must contain {sorted(required_environments)}")
    hardware = load(args.hardware_fingerprint)
    if hardware.get("status") != "passed":
        raise ValueError("Cloud preflight did not pass the RTX 4090 and 120 GiB RAM gate")
    frozen = load(args.frozen_archives)
    mismatches = verify_snapshot(frozen.get("archives", []))
    if mismatches:
        raise ValueError(f"Pre-repair archives changed before formal lock: {mismatches[:3]}")
    catalog = load(args.catalog)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "locked_before_formal_seeds",
        "source_commit": git_commit(),
        "locked_code_sha256": locked_code_hashes(),
        "formal_seeds": [5, 6, 7],
        "protocol": {
            "name": "COVID long-stream Setting B",
            "task1_weeks": 52,
            "online_weeks": 143,
            "visible_current_locations": 42,
            "hidden_current_locations": 10,
            "delayed_hidden_label_absorption": "exactly once at the following week",
            "likelihood": "Gaussian",
            "precision": "float64",
            "metrics": ["RMSE", "CRPS", "Gaussian NLPD", "ECE", "Coverage90"],
        },
        "selection": {
            "seed": 0,
            "blocked_windows": ["1-28 -> 29-36", "1-36 -> 37-44", "1-44 -> 45-52"],
            "metric": "mean Gaussian NLPD on restored log1p(per-100k) scale",
            "tie_breaker": "mean RMSE, then smaller capacity",
            "convergence_by_development_root": {
                name: record.get("convergence_gate") for name, record in capacities.items()
            },
            "practical_capacity_stability_by_development_root": {
                name: record.get("practical_capacity_stability") for name, record in capacities.items()
            },
            "capacity_records": {
                name: str((roots[name] / "capacity" / "capacity_selection.json").resolve())
                for name in roots
            },
            "online_step_record": str((roots["factorial"] / "online_steps" / "online_step_selection.json").resolve()),
            "expedited_development": capacities["factorial"].get("expedited_development"),
        },
        "methods": {
            method_id: {
                "configuration": configs[method_id],
                "environment": ENVIRONMENT_BY_METHOD[method_id],
                "source_commit": source_commit(catalog, method_id),
                "execution_backend": "GPU" if method_id in GPU_METHOD_IDS else "official isolated environment",
                "resource_class": "serial_high_rss" if method_id in ("ovc_svgp", "st_svgp", "fsde_svi") else "serial",
            }
            for method_id in configs
        },
        "gate_evidence": {
            "development_roots": {name: str(root.resolve()) for name, root in roots.items()},
            "ohsvgp_official_reproduction": str(absolute(args.ohsvgp_gate)),
            "ovc_memory_assessment": str(absolute(args.ovc_memory_assessment)),
            "hardware_fingerprint": str(absolute(args.hardware_fingerprint)),
            "environment_lock": str(absolute(args.environment_lock)),
            "frozen_pre_repair_archives": str(absolute(args.frozen_archives)),
        },
        "environment_lock": environment,
        "hardware": hardware,
        "formal_result_root": str(absolute(args.formal_result_root)),
        "retained_existing_accuracy": ["persistence", "task1_lag_ridge", "bui_osgpr_controlled", "bui_osgpr_adaptive", "routeb_ordinary", "routeb_cumulative_hippo"],
        "excluded_preliminary_methods": [method for method in METHOD_IDS if method not in configs],
        "exclusion_reasons": exclusions,
    }
    payload["lock_sha256"] = canonical_json_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "methods": sorted(payload["methods"]), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
