#!/usr/bin/env python3
"""Resumable Stage 2+ ERA5 benchmark orchestrator for AutoDL."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("benchmark.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expand_value(value: str) -> str:
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

    def replacement(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default or "")

    return os.path.expanduser(pattern.sub(replacement, value))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


@dataclass
class Job:
    stage: str
    scope: str
    method: str
    seed: int | None
    python: Path
    command: list[str]
    output_dir: Path
    expected: tuple[Path, ...]
    timeout_seconds: int
    device_class: str
    dependencies: tuple[Path, ...] = field(default_factory=tuple)
    legacy: bool = False

    @property
    def identifier(self) -> str:
        suffix = "" if self.seed is None else f"/seed{self.seed}"
        return f"{self.stage}/{self.scope}/{self.method}{suffix}"


def protocol_paths(benchmark: Path, scope: str, seed: int) -> tuple[Path, Path]:
    directory = benchmark / "protocol" / scope / f"seed{seed}"
    return directory / "protocol.npz", directory / "protocol.json"


def result_dir(
    benchmark: Path, scope: str, branch: str, method: str, seed: int
) -> Path:
    return benchmark / "runs" / scope / branch / method / f"seed{seed}"


def common_routeb_args(config: dict[str, Any]) -> list[str]:
    routeb = config["routeb"]
    return [
        "--mt",
        str(routeb["mt"]),
        "--ms",
        str(routeb["ms"]),
        "--iterations",
        str(routeb["iterations"]),
        "--learning-rate",
        str(routeb["learning_rate"]),
        "--validation-every",
        str(routeb["validation_every"]),
        "--rff-sample-size",
        str(routeb["rff_sample_size"]),
        "--prediction-chunk-size",
        str(routeb["prediction_chunk_size"]),
        "--warmup-steps",
        str(routeb["warmup_steps"]),
        "--profile-flops",
        "--beta-prior-variance",
        str(routeb["beta_prior_variance"]),
        "--device",
        str(config["primary_device"]),
        "--dtype",
        str(config["primary_dtype"]),
        "--evaluation-backend",
        "torch",
    ]


def calibration_jobs(
    config: dict[str, Any], benchmark: Path, python: Path
) -> list[Job]:
    jobs: list[Job] = []
    timeout = int(config["timeouts_seconds"]["calibration"])
    for representation in ("analytic_hippo_rff", "inducing_points"):
        method = f"routeb_joint_{representation}"
        for seed in config["split_seeds"]:
            protocol_npz, protocol_json = protocol_paths(
                benchmark, "task1_2", int(seed)
            )
            output = benchmark / "calibration" / method / f"seed{seed}"
            predictions = output / "predictions.npz"
            command = [
                str(python),
                "scripts/run_iclr_era5_routeb_batch.py",
                "--protocol-npz",
                str(protocol_npz),
                "--protocol-json",
                str(protocol_json),
                "--output-dir",
                str(output),
                "--predictions-output",
                str(predictions),
                "--data-part",
                "calibration",
                "--target-mode",
                "joint_xlag",
                "--representation",
                representation,
                "--split-seed",
                str(seed),
                "--model-seed",
                "0",
                *common_routeb_args(config),
            ]
            jobs.append(
                Job(
                    stage="stage2",
                    scope="calibration",
                    method=method,
                    seed=int(seed),
                    python=python,
                    command=command,
                    output_dir=output,
                    expected=(output / "result.json", predictions),
                    dependencies=(protocol_npz, protocol_json),
                    timeout_seconds=timeout,
                    device_class="modern_gpu",
                )
            )
    return jobs


def stage2_jobs(
    config: dict[str, Any], benchmark: Path, pythons: dict[str, Path]
) -> list[Job]:
    jobs = calibration_jobs(config, benchmark, pythons["routeb"])
    timeout = int(config["timeouts_seconds"]["batch"])
    for scope in config["scopes"]:
        for seed_value in config["split_seeds"]:
            seed = int(seed_value)
            protocol_npz, protocol_json = protocol_paths(benchmark, scope, seed)

            output = result_dir(benchmark, scope, "batch", "xlag_mean_only", seed)
            jobs.append(
                Job(
                    stage="stage2",
                    scope=scope,
                    method="xlag_mean_only",
                    seed=seed,
                    python=pythons["routeb"],
                    command=[
                        str(pythons["routeb"]),
                        "scripts/run_iclr_era5_xlag_mean_baselines.py",
                        "--protocol-npz",
                        str(protocol_npz),
                        "--protocol-json",
                        str(protocol_json),
                        "--output-dir",
                        str(output),
                        "--mode",
                        "batch_fixed",
                        "--seed",
                        str(seed),
                    ],
                    output_dir=output,
                    expected=(
                        output / "result.json",
                        output / "blocks.csv",
                        output / "predictions.npz",
                    ),
                    dependencies=(protocol_npz, protocol_json),
                    timeout_seconds=timeout,
                    device_class="cpu",
                )
            )

            for target, prefix in (
                ("shared_xlag_residual", "routeb_residual"),
                ("joint_xlag", "routeb_joint"),
            ):
                for representation in ("analytic_hippo_rff", "inducing_points"):
                    method = f"{prefix}_{representation}"
                    output = result_dir(benchmark, scope, "batch", method, seed)
                    command = [
                        str(pythons["routeb"]),
                        "scripts/run_iclr_era5_routeb_batch.py",
                        "--protocol-npz",
                        str(protocol_npz),
                        "--protocol-json",
                        str(protocol_json),
                        "--output-dir",
                        str(output),
                        "--predictions-output",
                        str(output / "predictions.npz"),
                        "--data-part",
                        "stream",
                        "--target-mode",
                        target,
                        "--representation",
                        representation,
                        "--split-seed",
                        str(seed),
                        "--model-seed",
                        "0",
                        *common_routeb_args(config),
                    ]
                    jobs.append(
                        Job(
                            stage="stage2",
                            scope=scope,
                            method=method,
                            seed=seed,
                            python=pythons["routeb"],
                            command=command,
                            output_dir=output,
                            expected=(output / "result.json", output / "predictions.npz"),
                            dependencies=(protocol_npz, protocol_json),
                            timeout_seconds=timeout,
                            device_class="modern_gpu",
                        )
                    )

            gpflow = config["gpflow_svgp"]
            method = f"gpflow_svgp_residual_mt{gpflow['mt']}_ms{gpflow['ms']}"
            output = result_dir(benchmark, scope, "batch", method, seed)
            jobs.append(
                Job(
                    stage="stage2",
                    scope=scope,
                    method=method,
                    seed=seed,
                    python=pythons["gpflow"],
                    command=[
                        str(pythons["gpflow"]),
                        "scripts/run_official_gpflow_svgp_era5.py",
                        "--protocol-npz",
                        str(protocol_npz),
                        "--output",
                        str(output / "result.json"),
                        "--predictions-output",
                        str(output / "predictions.npz"),
                        "--target-mode",
                        "shared_xlag_residual",
                        "--mt",
                        str(gpflow["mt"]),
                        "--ms",
                        str(gpflow["ms"]),
                        "--iterations",
                        str(gpflow["iterations"]),
                        "--batch-size",
                        str(gpflow["batch_size"]),
                        "--learning-rate",
                        str(gpflow["learning_rate"]),
                        "--natgrad-gamma",
                        str(gpflow["natgrad_gamma"]),
                        "--validation-every",
                        str(gpflow["validation_every"]),
                        "--prediction-chunk-size",
                        str(gpflow["prediction_chunk_size"]),
                        "--seed",
                        str(seed),
                        "--device",
                        str(config["primary_device"]),
                        "--dtype",
                        str(config["primary_dtype"]),
                    ],
                    output_dir=output,
                    expected=(output / "result.json", output / "predictions.npz"),
                    dependencies=(protocol_npz,),
                    timeout_seconds=timeout,
                    device_class="modern_gpu",
                )
            )
    return jobs


def legacy_stage2_jobs(
    config: dict[str, Any], benchmark: Path, pythons: dict[str, Path]
) -> list[Job]:
    jobs: list[Job] = []
    timeout = int(config["timeouts_seconds"]["legacy"])
    legacy = config["legacy"]
    for scope in config["scopes"]:
        for seed_value in config["split_seeds"]:
            seed = int(seed_value)
            protocol_npz, _ = protocol_paths(benchmark, scope, seed)
            for item in legacy["markovflow_configurations"]:
                method = (
                    f"markovflow_{item['model_kind']}_mt{item['mt']}_ms{item['ms']}"
                )
                output = result_dir(benchmark, scope, "batch", method, seed)
                jobs.append(
                    Job(
                        stage="stage2",
                        scope=scope,
                        method=method,
                        seed=seed,
                        python=pythons["markovflow"],
                        command=[
                            str(pythons["markovflow"]),
                            "scripts/run_official_markovflow_stsvgp_era5.py",
                            "--protocol-npz",
                            str(protocol_npz),
                            "--output",
                            str(output / "result.json"),
                            "--predictions-output",
                            str(output / "predictions.npz"),
                            "--model-kind",
                            str(item["model_kind"]),
                            "--target-mode",
                            "shared_xlag_residual",
                            "--mt",
                            str(item["mt"]),
                            "--ms",
                            str(item["ms"]),
                            "--iterations",
                            "100",
                            "--seed",
                            str(seed),
                        ],
                        output_dir=output,
                        expected=(output / "result.json", output / "predictions.npz"),
                        dependencies=(protocol_npz,),
                        timeout_seconds=timeout,
                        device_class="legacy_cpu",
                        legacy=True,
                    )
                )

            official_data = (
                benchmark / "official_protocol" / scope / f"seed{seed}" / "data.npz"
            )
            for ms in legacy["stsvgp_spatial_inducing"]:
                method = f"official_st_svgp_ms{ms}"
                output = result_dir(benchmark, scope, "batch", method, seed)
                jobs.append(
                    Job(
                        stage="stage2",
                        scope=scope,
                        method=method,
                        seed=seed,
                        python=pythons["stvgp"],
                        command=[
                            str(pythons["stvgp"]),
                            "scripts/run_official_stvgp_legacy.py",
                            "--model",
                            "st_svgp",
                            "--data-npz",
                            str(official_data),
                            "--num-spatial-inducing",
                            str(ms),
                            "--iterations",
                            str(legacy["stsvgp_iterations"]),
                            "--seed",
                            str(seed),
                            "--use-xlag-mean",
                            "--jit",
                            "--parallel",
                            "--trajectory-every",
                            "10",
                            "--predictions-output",
                            str(output / "predictions.npz"),
                            "--trajectory-output",
                            str(output / "trajectory.json"),
                            "--output",
                            str(output / "result.json"),
                        ],
                        output_dir=output,
                        expected=(output / "result.json", output / "predictions.npz"),
                        dependencies=(official_data,),
                        timeout_seconds=timeout,
                        device_class="legacy_cpu",
                        legacy=True,
                    )
                )
            for ms in legacy.get("mf_stsvgp_spatial_inducing", []):
                method = f"official_mf_st_svgp_ms{ms}"
                output = result_dir(benchmark, scope, "batch", method, seed)
                jobs.append(
                    Job(
                        stage="stage2",
                        scope=scope,
                        method=method,
                        seed=seed,
                        python=pythons["stvgp"],
                        command=[
                            str(pythons["stvgp"]),
                            "scripts/run_official_stvgp_legacy.py",
                            "--model",
                            "mf_st_svgp",
                            "--data-npz",
                            str(official_data),
                            "--num-spatial-inducing",
                            str(ms),
                            "--iterations",
                            str(legacy["stsvgp_iterations"]),
                            "--seed",
                            str(seed),
                            "--use-xlag-mean",
                            "--jit",
                            "--parallel",
                            "--trajectory-every",
                            "10",
                            "--predictions-output",
                            str(output / "predictions.npz"),
                            "--trajectory-output",
                            str(output / "trajectory.json"),
                            "--output",
                            str(output / "result.json"),
                        ],
                        output_dir=output,
                        expected=(output / "result.json", output / "predictions.npz"),
                        dependencies=(official_data,),
                        timeout_seconds=timeout,
                        device_class="legacy_cpu",
                        legacy=True,
                    )
                )
    return jobs


def stage3_jobs(
    config: dict[str, Any], benchmark: Path, pythons: dict[str, Path]
) -> list[Job]:
    jobs: list[Job] = []
    timeout = int(config["timeouts_seconds"]["online"])
    online = config["strict_online"]
    shared_representation = str(online["shared_theta_representation"])
    for scope in config["scopes"]:
        for seed_value in config["split_seeds"]:
            seed = int(seed_value)
            protocol_npz, protocol_json = protocol_paths(benchmark, scope, seed)
            shared_theta = (
                benchmark
                / "calibration"
                / f"routeb_joint_{shared_representation}"
                / f"seed{seed}"
                / "result.json"
            )

            for mode, method in (
                ("task1_fixed", "xlag_task1_fixed"),
                ("recursive_rls", "xlag_recursive_rls"),
            ):
                output = result_dir(benchmark, scope, "online", method, seed)
                jobs.append(
                    Job(
                        stage="stage3",
                        scope=scope,
                        method=method,
                        seed=seed,
                        python=pythons["routeb"],
                        command=[
                            str(pythons["routeb"]),
                            "scripts/run_iclr_era5_xlag_mean_baselines.py",
                            "--protocol-npz",
                            str(protocol_npz),
                            "--protocol-json",
                            str(protocol_json),
                            "--output-dir",
                            str(output),
                            "--mode",
                            mode,
                            "--seed",
                            str(seed),
                        ],
                        output_dir=output,
                        expected=(
                            output / "result.json",
                            output / "blocks.csv",
                            output / "predictions.npz",
                        ),
                        dependencies=(protocol_npz, protocol_json),
                        timeout_seconds=timeout,
                        device_class="cpu",
                    )
                )

            for representation in ("analytic_hippo_rff", "inducing_points"):
                method = f"routeb_{representation}"
                output = result_dir(benchmark, scope, "online", method, seed)
                theta = (
                    benchmark
                    / "calibration"
                    / f"routeb_joint_{representation}"
                    / f"seed{seed}"
                    / "result.json"
                )
                jobs.append(
                    Job(
                        stage="stage3",
                        scope=scope,
                        method=method,
                        seed=seed,
                        python=pythons["routeb"],
                        command=[
                            str(pythons["routeb"]),
                            "scripts/run_iclr_era5_routeb_strict_online.py",
                            "--protocol-npz",
                            str(protocol_npz),
                            "--protocol-json",
                            str(protocol_json),
                            "--theta-json",
                            str(theta),
                            "--output",
                            str(output / "result.json"),
                            "--blockwise-output",
                            str(output / "blocks.csv"),
                            "--predictions-output",
                            str(output / "predictions.npz"),
                            "--representation",
                            representation,
                            "--mt",
                            str(config["routeb"]["mt"]),
                            "--ms",
                            str(config["routeb"]["ms"]),
                            "--rff-sample-size",
                            str(config["routeb"]["rff_sample_size"]),
                            "--prediction-chunk-size",
                            str(config["routeb"]["prediction_chunk_size"]),
                            "--seed",
                            str(seed),
                            "--solver-backend",
                            "torch",
                            "--device",
                            str(config["primary_device"]),
                            "--temporal-factor-device",
                            str(
                                config["routeb"]
                                .get("temporal_factor_device", {})
                                .get(representation, "auto")
                            ),
                            "--dtype",
                            str(config["primary_dtype"]),
                        ],
                        output_dir=output,
                        expected=(
                            output / "result.json",
                            output / "blocks.csv",
                            output / "predictions.npz",
                        ),
                        dependencies=(protocol_npz, protocol_json, theta),
                        timeout_seconds=timeout,
                        device_class="modern_gpu",
                    )
                )

            method = f"bui_osgpr_mt{online['bui_mt']}_ms{online['bui_ms']}"
            output = result_dir(benchmark, scope, "online", method, seed)
            jobs.append(
                Job(
                    stage="stage3",
                    scope=scope,
                    method=method,
                    seed=seed,
                    python=pythons["gpflow"],
                    command=[
                        str(pythons["gpflow"]),
                        "scripts/run_official_bui_osgpr_era5.py",
                        "--protocol-npz",
                        str(protocol_npz),
                        "--theta-json",
                        str(shared_theta),
                        "--output",
                        str(output / "result.json"),
                        "--blockwise-output",
                        str(output / "blocks.csv"),
                        "--predictions-output",
                        str(output / "predictions.npz"),
                        "--mt",
                        str(online["bui_mt"]),
                        "--ms",
                        str(online["bui_ms"]),
                        "--seed",
                        str(seed),
                        "--device",
                        str(config["primary_device"]),
                        "--dtype",
                        str(config["primary_dtype"]),
                    ],
                    output_dir=output,
                    expected=(
                        output / "result.json",
                        output / "blocks.csv",
                        output / "predictions.npz",
                    ),
                    dependencies=(protocol_npz, shared_theta),
                    timeout_seconds=timeout,
                    device_class="modern_gpu",
                )
            )

            method = (
                f"maddox_streaming_sgpr_mt{online['maddox_mt']}_ms{online['maddox_ms']}"
            )
            output = result_dir(benchmark, scope, "online", method, seed)
            jobs.append(
                Job(
                    stage="stage3",
                    scope=scope,
                    method=method,
                    seed=seed,
                    python=pythons["maddox"],
                    command=[
                        str(pythons["maddox"]),
                        "scripts/run_official_maddox_streaming_sgpr_era5.py",
                        "--protocol-npz",
                        str(protocol_npz),
                        "--theta-json",
                        str(shared_theta),
                        "--output",
                        str(output / "result.json"),
                        "--blockwise-output",
                        str(output / "blocks.csv"),
                        "--predictions-output",
                        str(output / "predictions.npz"),
                        "--mt",
                        str(online["maddox_mt"]),
                        "--ms",
                        str(online["maddox_ms"]),
                        "--jitter",
                        str(online["maddox_jitter"]),
                        "--resample-ratio",
                        str(online["maddox_resample_ratio"]),
                        "--seed",
                        str(seed),
                        "--device",
                        str(config["primary_device"]),
                        "--dtype",
                        str(config["primary_dtype"]),
                    ],
                    output_dir=output,
                    expected=(
                        output / "result.json",
                        output / "blocks.csv",
                        output / "predictions.npz",
                    ),
                    dependencies=(protocol_npz, shared_theta),
                    timeout_seconds=timeout,
                    device_class="modern_gpu_legacy_api",
                )
            )

            method = (
                f"official_ohsvgp_m{online['ohsvgp_inducing_size']}_rff"
                f"{online['ohsvgp_rff_sample_size']}"
            )
            output = result_dir(benchmark, scope, "online", method, seed)
            jobs.append(
                Job(
                    stage="stage3",
                    scope=scope,
                    method=method,
                    seed=seed,
                    python=pythons["routeb"],
                    command=[
                        str(pythons["routeb"]),
                        "scripts/run_official_ohsvgp_era5.py",
                        "--protocol-npz",
                        str(protocol_npz),
                        "--theta-json",
                        str(shared_theta),
                        "--output",
                        str(output / "result.json"),
                        "--blockwise-output",
                        str(output / "blocks.csv"),
                        "--predictions-output",
                        str(output / "predictions.npz"),
                        "--inducing-size",
                        str(online["ohsvgp_inducing_size"]),
                        "--rff-sample-size",
                        str(online["ohsvgp_rff_sample_size"]),
                        "--microbatch-size",
                        str(online["ohsvgp_microbatch_size"]),
                        "--update-steps",
                        str(online["ohsvgp_update_steps"]),
                        "--seed",
                        str(seed),
                        "--device",
                        str(config["primary_device"]),
                        "--dtype",
                        str(config["primary_dtype"]),
                    ],
                    output_dir=output,
                    expected=(
                        output / "result.json",
                        output / "blocks.csv",
                        output / "predictions.npz",
                    ),
                    dependencies=(protocol_npz, shared_theta),
                    timeout_seconds=timeout,
                    device_class="modern_gpu",
                )
            )
    return jobs


def utility_jobs(
    config: dict[str, Any], benchmark: Path, python: Path, config_path: Path
) -> list[Job]:
    timeout = int(config["timeouts_seconds"]["report"])
    efficiency = benchmark / "efficiency"
    report = benchmark / "report"
    return [
        Job(
            stage="stage4",
            scope="all",
            method="efficiency_summary",
            seed=None,
            python=python,
            command=[
                str(python),
                "scripts/summarize_autodl_era5_efficiency.py",
                "--benchmark-root",
                str(benchmark),
                "--output-dir",
                str(efficiency),
            ],
            output_dir=efficiency,
            expected=(efficiency / "efficiency_per_run.csv", efficiency / "efficiency_summary.csv"),
            dependencies=(),
            timeout_seconds=timeout,
            device_class="cpu_aggregation",
        ),
        Job(
            stage="stage5",
            scope="all",
            method="reproducibility_audit",
            seed=None,
            python=python,
            command=[
                str(python),
                "scripts/audit_autodl_era5_benchmark.py",
                "--benchmark-root",
                str(benchmark),
                "--config",
                str(config_path),
                "--output",
                str(benchmark / "audit.json"),
            ],
            output_dir=benchmark / "audit",
            expected=(benchmark / "audit.json",),
            dependencies=(efficiency / "efficiency_summary.csv",),
            timeout_seconds=timeout,
            device_class="cpu_audit",
        ),
        Job(
            stage="stage5",
            scope="all",
            method="report_and_visualizations",
            seed=None,
            python=python,
            command=[
                str(python),
                "scripts/generate_autodl_era5_stage2plus_report.py",
                "--benchmark-root",
                str(benchmark),
                "--config",
                str(config_path),
                "--output-dir",
                str(report),
            ],
            output_dir=report,
            expected=(report / "report.md", report / "artifact_manifest.json"),
            dependencies=(benchmark / "audit.json",),
            timeout_seconds=timeout,
            device_class="cpu_report",
        ),
    ]


def classify_failure(log_path: Path, returncode: int | None, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    text = log_path.read_text(encoding="utf-8", errors="ignore").lower() if log_path.exists() else ""
    if any(token in text for token in ("out of memory", "resourceexhausted", "cuda error: out of memory", "oom")):
        return "out_of_memory"
    if "no kernel image" in text or "compute capability" in text and "not compatible" in text:
        return "gpu_incompatible"
    if "modulenotfounderror" in text or "importerror" in text:
        return "dependency_error"
    if returncode is not None and returncode < 0:
        return "terminated_by_signal"
    return "runtime_error"


class NvidiaMonitor:
    fields = [
        "timestamp_utc",
        "index",
        "name",
        "utilization_gpu_percent",
        "memory_used_mib",
        "memory_total_mib",
        "power_draw_w",
        "temperature_c",
    ]

    def __init__(self, path: Path, interval: float = 1.0) -> None:
        self.path = path
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if not shutil_which("nvidia-smi"):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(2.0, self.interval * 2.0))

    def _run(self) -> None:
        query = (
            "index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
        )
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.fields)
            while not self.stop_event.is_set():
                try:
                    output = subprocess.check_output(
                        [
                            "nvidia-smi",
                            f"--query-gpu={query}",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                    )
                    for line in output.splitlines():
                        writer.writerow(
                            [utc_now(), *[part.strip() for part in line.split(",")]]
                        )
                    handle.flush()
                except (OSError, subprocess.SubprocessError):
                    pass
                self.stop_event.wait(self.interval)


def shutil_which(command: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def python_environment(python: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"python": str(python)}
    try:
        payload["version"] = subprocess.check_output(
            [str(python), "--version"], text=True, stderr=subprocess.STDOUT, timeout=20
        ).strip()
        payload["pip_freeze"] = subprocess.check_output(
            [str(python), "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=120,
        ).splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        payload["error"] = str(exc)
    return payload


def run_job(job: Job, *, force: bool, dry_run: bool, env: dict[str, str]) -> str:
    status_path = job.output_dir / "status.json"
    if not force and all(path.is_file() and path.stat().st_size > 0 for path in job.expected):
        print(f"SKIP complete {job.identifier}")
        return "skipped"

    missing = [str(path) for path in job.dependencies if not path.exists()]
    if missing:
        payload = {
            "schema_version": 1,
            "job": job.identifier,
            "status": "blocked_dependency",
            "missing_dependencies": missing,
            "updated_at": utc_now(),
        }
        if not dry_run:
            atomic_json(status_path, payload)
        print(f"BLOCKED {job.identifier}: {missing}")
        return "blocked_dependency"

    printable = shlex.join(job.command)
    if dry_run:
        print(f"DRY-RUN {job.identifier}\n  {printable}")
        return "dry_run"

    if not job.python.is_file():
        raise FileNotFoundError(f"Python environment is missing: {job.python}")
    job.output_dir.mkdir(parents=True, exist_ok=True)
    (job.output_dir / "command.txt").write_text(printable + "\n", encoding="utf-8")
    atomic_json(job.output_dir / "environment.json", python_environment(job.python))
    log_path = job.output_dir / "run.log"
    time_path = job.output_dir / "resource_usage.txt"
    monitor = NvidiaMonitor(job.output_dir / "nvidia_smi.csv")
    command = list(job.command)
    if Path("/usr/bin/time").is_file():
        command = ["/usr/bin/time", "-v", "-o", str(time_path), *command]

    started_wall = time.perf_counter()
    started_at = utc_now()
    timed_out = False
    returncode: int | None = None
    print(f"RUN {job.identifier}")
    monitor.start()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            returncode = process.wait(timeout=job.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
        finally:
            monitor.stop()

    complete = returncode == 0 and all(
        path.is_file() and path.stat().st_size > 0 for path in job.expected
    )
    status = "complete" if complete else classify_failure(log_path, returncode, timed_out)
    payload = {
        "schema_version": 1,
        "job": job.identifier,
        "stage": job.stage,
        "scope": job.scope,
        "method": job.method,
        "seed": job.seed,
        "status": status,
        "legacy": job.legacy,
        "device_class": job.device_class,
        "command": job.command,
        "started_at": started_at,
        "finished_at": utc_now(),
        "wall_seconds": time.perf_counter() - started_wall,
        "timeout_seconds": job.timeout_seconds,
        "returncode": returncode,
        "expected_artifacts": [str(path) for path in job.expected],
        "missing_artifacts": [str(path) for path in job.expected if not path.is_file()],
        "log": str(log_path),
        "nvidia_smi": str(job.output_dir / "nvidia_smi.csv"),
        "resource_usage": str(time_path),
    }
    atomic_json(status_path, payload)
    print(f"{status.upper()} {job.identifier} ({payload['wall_seconds']:.1f} s)")
    return status


def prepare_jobs(
    config: dict[str, Any], benchmark: Path, routeb_python: Path, include_legacy: bool
) -> list[Job]:
    protocol = benchmark / "protocol"
    jobs = [
        Job(
            stage="prepare",
            scope="all",
            method="export_shared_protocol",
            seed=None,
            python=routeb_python,
            command=[
                str(routeb_python),
                "scripts/export_iclr_era5_full_benchmark_protocol.py",
                "--outdir",
                str(protocol),
                "--split-seeds",
                *[str(seed) for seed in config["split_seeds"]],
            ],
            output_dir=protocol,
            expected=(protocol / "manifest.json",),
            dependencies=(
                ROOT / config["data"]["short_root"],
                ROOT / config["data"]["long_root"] / "verification_report.json",
            ),
            timeout_seconds=int(config["timeouts_seconds"]["calibration"]),
            device_class="cpu_protocol",
        )
    ]
    if include_legacy:
        for scope in config["scopes"]:
            for seed_value in config["split_seeds"]:
                seed = int(seed_value)
                protocol_npz, _ = protocol_paths(benchmark, scope, seed)
                output = (
                    benchmark / "official_protocol" / scope / f"seed{seed}"
                )
                jobs.append(
                    Job(
                        stage="prepare",
                        scope=scope,
                        method="export_official_stvgp_protocol",
                        seed=seed,
                        python=routeb_python,
                        command=[
                            str(routeb_python),
                            "scripts/export_iclr_protocol_for_official_stvgp.py",
                            "--protocol-npz",
                            str(protocol_npz),
                            "--output",
                            str(output / "data.npz"),
                        ],
                        output_dir=output,
                        expected=(output / "data.npz",),
                        dependencies=(protocol_npz,),
                        timeout_seconds=int(config["timeouts_seconds"]["calibration"]),
                        device_class="cpu_protocol",
                    )
                )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "stage2", "stage3", "stage4", "stage5"],
        default="all",
    )
    parser.add_argument("--scope", choices=["all", "calibration", "task1_2", "task1_10"], default="all")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method-pattern")
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def filter_jobs(jobs: Iterable[Job], args: argparse.Namespace) -> list[Job]:
    selected = []
    pattern = re.compile(args.method_pattern) if args.method_pattern else None
    for job in jobs:
        if args.stage != "all" and job.stage != args.stage:
            continue
        if args.scope != "all" and job.scope != args.scope:
            continue
        if args.seed is not None and job.seed != args.seed:
            continue
        if pattern and not pattern.search(job.method):
            continue
        selected.append(job)
    return selected


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {key: Path(expand_value(value)) for key, value in config["paths"].items()}
    benchmark = paths["benchmark_root"]
    pythons = {
        "routeb": paths["routeb_python"],
        "gpflow": paths["gpflow_python"],
        "maddox": paths["maddox_python"],
        "markovflow": paths["markovflow_python"],
        "stvgp": paths["stvgp_python"],
    }
    include_legacy = bool(args.include_legacy)
    jobs = [
        *prepare_jobs(config, benchmark, pythons["routeb"], include_legacy),
        *stage2_jobs(config, benchmark, pythons),
    ]
    if include_legacy:
        jobs.extend(legacy_stage2_jobs(config, benchmark, pythons))
    jobs.extend(stage3_jobs(config, benchmark, pythons))
    jobs.extend(utility_jobs(config, benchmark, pythons["routeb"], args.config.resolve()))
    jobs = filter_jobs(jobs, args)
    if args.list:
        for job in jobs:
            print(job.identifier, job.device_class, shlex.join(job.command))
        return

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(config["cuda_visible_devices"]),
            "PYTHONHASHSEED": "0",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "TF_DETERMINISTIC_OPS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "8"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "8"),
        }
    )
    summary: dict[str, int] = {}
    for job in jobs:
        status = run_job(job, force=args.force, dry_run=args.dry_run, env=env)
        summary[status] = summary.get(status, 0) + 1
        if args.fail_fast and status not in {"complete", "skipped", "dry_run"}:
            break
    print(json.dumps({"benchmark_root": str(benchmark), "jobs": summary}, indent=2))


if __name__ == "__main__":
    main()
