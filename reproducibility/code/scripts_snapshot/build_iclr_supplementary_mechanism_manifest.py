#!/usr/bin/env python3
"""Build, but never execute, the ICLR supplementary mechanism job manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_ROOT = Path(
    "/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/"
    "kronecker+s2vgp/ICLR Formal experiment/iclr_era5_stage2plus/protocol"
)
DEFAULT_CALIBRATION_ROOT = (
    ROOT
    / "results/era5_stage2plus_sm_baseline_rerun_20260828_formal/calibration/"
    "routeb_kronhippo_stgp_sm"
)
DEFAULT_DATA_ROOT = ROOT / "data/era5/processed_timeseries_4"
DEFAULT_SM_CONFIG = ROOT / "configs/era5_sm_q3.json"
DEFAULT_PYTHON = ROOT / ".venv_cuda128/bin/python"
DEFAULT_OUTPUT_ROOT = ROOT / "results/iclr2027_supplementary_mechanism_planned"
DEFAULT_RUNNER = ROOT / "scripts/run_iclr_era5_routeb_strict_online.py"

CELLS = (
    ("mean_field", False),
    ("mean_field", True),
    ("structured_joint", False),
    ("structured_joint", True),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def require_directory(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def cell_name(coupling: str, transfer: bool) -> str:
    suffix = "changing_basis_transfer" if transfer else "history_reset"
    return f"{coupling}_{suffix}"


def build_jobs(
    *,
    protocol_root: Path,
    calibration_root: Path,
    data_root: Path,
    sm_config: Path,
    runner: Path,
    output_root: Path,
    python: Path,
    seeds: Iterable[int],
    mt: int,
    ms: int,
    rff_sample_size: int,
    max_blocks: int,
) -> list[dict[str, object]]:
    """Return planned short-stream jobs without creating outputs or subprocesses."""

    if mt <= 0 or ms <= 0 or rff_sample_size <= 0:
        raise ValueError("mt, ms, and rff_sample_size must be positive")
    if max_blocks < 0:
        raise ValueError("max_blocks must be non-negative")

    protocol_root = require_directory(Path(protocol_root))
    calibration_root = require_directory(Path(calibration_root))
    data_root = require_directory(Path(data_root))
    sm_config = require_file(Path(sm_config))
    runner = require_file(Path(runner))
    output_root = Path(output_root)
    python = Path(python)

    jobs: list[dict[str, object]] = []
    for seed in [int(value) for value in seeds]:
        protocol_directory = protocol_root / "task1_2" / f"seed{seed}"
        protocol_npz = require_file(protocol_directory / "protocol.npz")
        protocol_json = require_file(protocol_directory / "protocol.json")
        theta_json = require_file(calibration_root / f"seed{seed}" / "result.json")

        common_inputs = {
            "protocol_npz": str(protocol_npz),
            "protocol_json": str(protocol_json),
            "theta_json": str(theta_json),
            "spectral_mixture_json": str(sm_config),
            "runner": str(runner),
        }
        input_hashes = {
            name: sha256(Path(path)) for name, path in common_inputs.items()
        }

        for coupling, transfer in CELLS:
            cell = cell_name(coupling, transfer)
            output = output_root / "short" / "online" / cell / f"seed{seed}"
            argv = [
                str(python),
                str(runner),
                "--protocol-npz",
                str(protocol_npz),
                "--protocol-json",
                str(protocol_json),
                "--data-root",
                str(data_root),
                "--theta-json",
                str(theta_json),
                "--representation",
                "analytic_hippo_rff",
                "--mt",
                str(mt),
                "--ms",
                str(ms),
                "--rff-sample-size",
                str(rff_sample_size),
                "--temporal-kernel",
                "spectral_mixture",
                "--spectral-mixture-json",
                str(sm_config),
                "--include-conditional-residual-variance",
                "--seed",
                str(seed),
                "--coupling",
                coupling,
                "--solver-backend",
                "torch",
                "--device",
                "cuda",
                "--dtype",
                "float64",
                "--temporal-factor-device",
                "solver",
                "--output",
                str(output / "result.json"),
                "--blockwise-output",
                str(output / "blocks.csv"),
                "--predictions-output",
                str(output / "predictions.npz"),
            ]
            argv.append("--transfer" if transfer else "--no-transfer")
            if max_blocks:
                argv.extend(["--max-blocks", str(max_blocks)])

            jobs.append(
                {
                    "schema_version": 1,
                    "job_id": f"short/seed{seed}/{cell}",
                    "status": "planned_not_executed",
                    "scope": "short",
                    "seed": seed,
                    "coupling": coupling,
                    "transfer": transfer,
                    "control_semantics": (
                        "changing_basis_historical_likelihood_transfer"
                        if transfer
                        else "current_block_only_history_reset"
                    ),
                    "calibration_objective": "vfe",
                    "temporal_kernel": "spectral_mixture",
                    "predictive_variance": "full_joint_conditional",
                    "capacity": {"mt": mt, "ms": ms, "rff_sample_size": rff_sample_size},
                    "max_blocks": max_blocks,
                    "inputs": common_inputs,
                    "input_sha256": input_hashes,
                    "output_directory": str(output),
                    "expected_outputs": [
                        str(output / "result.json"),
                        str(output / "blocks.csv"),
                        str(output / "predictions.npz"),
                    ],
                    "argv": argv,
                }
            )
    return jobs


def write_manifest(jobs: list[dict[str, object]], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "status": "planned_not_executed",
        "jobs": len(jobs),
        "seeds": sorted({int(job["seed"]) for job in jobs}),
        "warning": (
            "The no-transfer cells reset all historical likelihood statistics. "
            "They are not fixed-basis, history-preserving controls and cannot isolate transfer error."
        ),
        "execution_gate": "Author GPU-hour, storage, paid-resource, and deadline approval required.",
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
    }
    manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an auditable ICLR supplementary job manifest without running it."
    )
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sm-config", type=Path, default=DEFAULT_SM_CONFIG)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--max-blocks", type=int, default=0)
    args = parser.parse_args()

    manifest = args.manifest or args.output_root / "manifests/history_retention_factorial.jsonl"
    jobs = build_jobs(
        protocol_root=args.protocol_root,
        calibration_root=args.calibration_root,
        data_root=args.data_root,
        sm_config=args.sm_config,
        runner=args.runner,
        output_root=args.output_root,
        python=args.python,
        seeds=args.seeds,
        mt=args.mt,
        ms=args.ms,
        rff_sample_size=args.rff_sample_size,
        max_blocks=args.max_blocks,
    )
    write_manifest(jobs, manifest)
    print(
        json.dumps(
            {
                "status": "planned_not_executed",
                "jobs": len(jobs),
                "manifest": str(manifest),
                "summary": str(manifest.with_suffix(".summary.json")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
