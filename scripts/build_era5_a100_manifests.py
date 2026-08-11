#!/usr/bin/env python3
"""Build deterministic JSONL manifests for the unified ERA5 A100 benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloud.autodl_era5.run_benchmark import (  # noqa: E402
    Job,
    calibration_jobs,
    expand_value,
    prepare_jobs,
    protocol_paths,
    result_dir,
    stage3_jobs,
)


ENTRY_SCHEMA_VERSION = 1
ROUTE_B_BATCH_SCRIPT = "scripts/run_iclr_era5_routeb_batch.py"
ROUTE_B_ONLINE_SCRIPT = "scripts/run_iclr_era5_routeb_strict_online.py"
POSTPROCESS_SCRIPT = "scripts/summarize_task2_online_segments.py"
NCU_BATCH_SCRIPTS = {
    "scripts/run_iclr_era5_routeb_batch.py",
    "scripts/run_official_stvgp_legacy.py",
    "scripts/run_official_markovflow_stsvgp_era5.py",
    "scripts/run_official_gpflow_svgp_era5.py",
}
NCU_ONLINE_SCRIPTS = {
    "scripts/run_iclr_era5_routeb_strict_online.py",
    "scripts/run_official_bui_osgpr_era5.py",
    "scripts/run_official_maddox_streaming_sgpr_era5.py",
    "scripts/run_official_ohsvgp_era5.py",
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def resolve_path(value: str | Path, *, relative_to: Path = ROOT) -> Path:
    path = Path(expand_value(str(value)))
    return path if path.is_absolute() else relative_to / path


def load_spec(path: Path) -> dict[str, Any]:
    spec = load_json(path)
    if spec.get("schema_version") != 2:
        raise ValueError("The A100 matrix specification must use schema_version 2")
    if not isinstance(spec.get("manifest_id"), str) or not spec["manifest_id"]:
        raise ValueError("The A100 matrix specification needs a manifest_id")
    seeds = spec.get("seeds")
    if not isinstance(seeds, list) or seeds != sorted(set(seeds)):
        raise ValueError("seeds must be a sorted list of unique integers")
    if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        raise ValueError("seeds must be a sorted list of unique integers")

    gpflow = spec["short_batch"]["gpflow_feasibility"]
    selected_tier = str(gpflow["selected_tier"])
    preflight_tier = str(gpflow["preflight_tier"])
    tiers = gpflow["tiers"]
    if selected_tier not in tiers or preflight_tier not in tiers:
        raise ValueError("GPflow selected_tier and preflight_tier must name configured tiers")
    for tier_name, candidates in tiers.items():
        expected_total = int(tier_name)
        if any(int(item["mt"]) * int(item["ms"]) != expected_total for item in candidates):
            raise ValueError(f"GPflow tier {tier_name} has an invalid inducing total")
    return spec


def load_base_config(spec_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    base_value = spec.get("base_config", "benchmark.json")
    base_path = resolve_path(base_value, relative_to=spec_path.parent)
    return load_json(base_path)


def python_paths(base_config: dict[str, Any]) -> dict[str, Path]:
    paths = base_config["paths"]
    return {
        "routeb": resolve_path(paths["routeb_python"]),
        "gpflow": resolve_path(paths["gpflow_python"]),
        "maddox": resolve_path(paths["maddox_python"]),
        "markovflow": resolve_path(paths["markovflow_python"]),
        "stvgp": resolve_path(paths["stvgp_python"]),
    }


def _batch_timeout(base_config: dict[str, Any]) -> int:
    return int(base_config["timeouts_seconds"]["batch"])


def _legacy_timeout(base_config: dict[str, Any]) -> int:
    return int(base_config["timeouts_seconds"]["legacy"])


def _protocol_dependencies(benchmark: Path, scope: str, seed: int) -> tuple[Path, Path]:
    return protocol_paths(benchmark, scope, seed)


def build_protocol_jobs(
    *, spec: dict[str, Any], base_config: dict[str, Any], benchmark: Path, python: Path
) -> list[Job]:
    """Create shared and official protocol producers for the batch manifests."""
    prepared = prepare_jobs(base_config, benchmark, python, include_legacy=True)
    source = prepared[0]
    protocol_artifacts = list(source.expected)
    for scope in ("task1_2", "task1_10"):
        for seed in spec["seeds"]:
            protocol_artifacts.extend(protocol_paths(benchmark, scope, seed))
    return [replace(source, expected=tuple(protocol_artifacts)), *prepared[1:]]


def build_protocol_job(
    *, spec: dict[str, Any], base_config: dict[str, Any], benchmark: Path, python: Path
) -> Job:
    """Compatibility helper returning the shared protocol export job."""
    return build_protocol_jobs(
        spec=spec, base_config=base_config, benchmark=benchmark, python=python
    )[0]


def build_calibration_jobs(
    *, spec: dict[str, Any], base_config: dict[str, Any], benchmark: Path, python: Path
) -> list[Job]:
    jobs = calibration_jobs(base_config, benchmark, python)
    selected_seeds = set(spec["seeds"])
    result = []
    for job in jobs:
        if job.seed not in selected_seeds:
            continue
        command = list(job.command)
        if ROUTE_B_BATCH_SCRIPT in command and "--include-conditional-residual-variance" not in command:
            command.append("--include-conditional-residual-variance")
        result.append(replace(job, command=command))
    return result


def _official_job(
    *,
    benchmark: Path,
    python: Path,
    scope: str,
    seed: int,
    item: dict[str, Any],
    iterations: int,
    timeout_seconds: int,
    branch: str,
    method_prefix: str,
    device_class: str,
) -> Job:
    method = f"{method_prefix}{item['name']}"
    output = result_dir(benchmark, scope, branch, method, seed)
    data_npz = benchmark / "official_protocol" / scope / f"seed{seed}" / "data.npz"
    command = [
        str(python),
        "scripts/run_official_stvgp_legacy.py",
        "--model",
        str(item["model"]),
        "--data-npz",
        str(data_npz),
        "--iterations",
        str(iterations),
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
    ]
    if item.get("ms") is not None:
        insert_at = command.index("--iterations")
        command[insert_at:insert_at] = [
            "--num-spatial-inducing",
            str(item["ms"]),
            "--fixed-spatial-inducing",
        ]
    return Job(
        stage="stage2",
        scope=scope,
        method=method,
        seed=seed,
        python=python,
        command=command,
        output_dir=output,
        expected=(output / "result.json", output / "predictions.npz"),
        dependencies=(data_npz,),
        timeout_seconds=timeout_seconds,
        device_class=device_class,
        legacy=True,
    )


def build_xlag_jobs(
    *, spec: dict[str, Any], base_config: dict[str, Any], benchmark: Path, python: Path
) -> list[Job]:
    scope = str(spec["short_batch"]["scope"])
    jobs = []
    for seed in spec["seeds"]:
        protocol_npz, protocol_json = _protocol_dependencies(benchmark, scope, seed)
        output = result_dir(benchmark, scope, "batch", "xlag_mean_only", seed)
        jobs.append(
            Job(
                stage="stage2",
                scope=scope,
                method="xlag_mean_only",
                seed=seed,
                python=python,
                command=[
                    str(python),
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
                timeout_seconds=_batch_timeout(base_config),
                device_class="a100_cpu",
            )
        )
    return jobs


def build_official_jobs(
    *,
    spec: dict[str, Any],
    base_config: dict[str, Any],
    benchmark: Path,
    python: Path,
    scope: str,
    items: list[dict[str, Any]],
) -> list[Job]:
    return [
        _official_job(
            benchmark=benchmark,
            python=python,
            scope=scope,
            seed=seed,
            item=item,
            iterations=int(base_config["legacy"]["stsvgp_iterations"]),
            timeout_seconds=_legacy_timeout(base_config),
            branch="batch",
            method_prefix="official_",
            device_class="a100_official_full",
        )
        for item in items
        for seed in spec["seeds"]
    ]


def build_markovflow_jobs(
    *, spec: dict[str, Any], base_config: dict[str, Any], benchmark: Path, python: Path
) -> list[Job]:
    scope = str(spec["short_batch"]["scope"])
    jobs = []
    for item in spec["short_batch"]["markovflow"]:
        for seed in spec["seeds"]:
            protocol_npz, _ = _protocol_dependencies(benchmark, scope, seed)
            method = f"markovflow_{item['name']}"
            output = result_dir(benchmark, scope, "batch", method, seed)
            jobs.append(
                Job(
                    stage="stage2",
                    scope=scope,
                    method=method,
                    seed=seed,
                    python=python,
                    command=[
                        str(python),
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
                    timeout_seconds=_legacy_timeout(base_config),
                    device_class="a100_markovflow",
                    legacy=True,
                )
            )
    return jobs


def _gpflow_candidates(spec: dict[str, Any], tier: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    settings = spec["short_batch"]["gpflow_feasibility"]
    selected = str(tier or settings["selected_tier"])
    try:
        candidates = settings["tiers"][selected]
    except KeyError as exc:
        raise ValueError(f"Unknown GPflow feasibility tier: {selected}") from exc
    return selected, candidates


def _gpflow_command(
    *,
    base_config: dict[str, Any],
    settings: dict[str, Any],
    python: Path,
    protocol_npz: Path,
    output: Path,
    item: dict[str, Any],
    seed: int,
    iterations: int,
) -> list[str]:
    return [
        str(python),
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
        str(item["mt"]),
        "--ms",
        str(item["ms"]),
        "--iterations",
        str(iterations),
        "--batch-size",
        str(settings["batch_size"]),
        "--learning-rate",
        str(settings["learning_rate"]),
        "--natgrad-gamma",
        str(settings["natgrad_gamma"]),
        "--validation-every",
        str(settings["validation_every"]),
        "--prediction-chunk-size",
        str(settings["prediction_chunk_size"]),
        "--seed",
        str(seed),
        "--device",
        str(base_config["primary_device"]),
        "--dtype",
        str(base_config["primary_dtype"]),
    ]


def _gpflow_job(
    *,
    spec: dict[str, Any],
    base_config: dict[str, Any],
    benchmark: Path,
    python: Path,
    tier: str,
    item: dict[str, Any],
    seed: int,
    branch: str,
    method_prefix: str,
    iterations: int,
    device_class: str,
) -> Job:
    scope = str(spec["short_batch"]["scope"])
    settings = base_config["gpflow_svgp"]
    protocol_npz, _ = _protocol_dependencies(benchmark, scope, seed)
    method = f"{method_prefix}gpflow_feasibility_{tier}_{item['name']}"
    output = result_dir(benchmark, scope, branch, method, seed)
    return Job(
        stage="stage2",
        scope=scope,
        method=method,
        seed=seed,
        python=python,
        command=_gpflow_command(
            base_config=base_config,
            settings=settings,
            python=python,
            protocol_npz=protocol_npz,
            output=output,
            item=item,
            seed=seed,
            iterations=iterations,
        ),
        output_dir=output,
        expected=(output / "result.json", output / "predictions.npz"),
        dependencies=(protocol_npz,),
        timeout_seconds=_batch_timeout(base_config),
        device_class=device_class,
    )


def build_gpflow_preflight_jobs(
    *, spec: dict[str, Any], base_config: dict[str, Any], benchmark: Path, python: Path
) -> list[Job]:
    tier, candidates = _gpflow_candidates(spec, str(spec["short_batch"]["gpflow_feasibility"]["preflight_tier"]))
    iterations = int(spec["short_batch"]["gpflow_feasibility"]["preflight_iterations"])
    return [
        _gpflow_job(
            spec=spec,
            base_config=base_config,
            benchmark=benchmark,
            python=python,
            tier=tier,
            item=item,
            seed=0,
            branch="preflight",
            method_prefix="preflight_",
            iterations=iterations,
            device_class="a100_gpflow_preflight",
        )
        for item in candidates
    ]


def build_gpflow_jobs(
    *,
    spec: dict[str, Any],
    base_config: dict[str, Any],
    benchmark: Path,
    python: Path,
    tier: str | None = None,
) -> list[Job]:
    selected_tier, candidates = _gpflow_candidates(spec, tier)
    return [
        _gpflow_job(
            spec=spec,
            base_config=base_config,
            benchmark=benchmark,
            python=python,
            tier=selected_tier,
            item=item,
            seed=seed,
            branch="batch",
            method_prefix="",
            iterations=int(base_config["gpflow_svgp"]["iterations"]),
            device_class="a100_gpflow",
        )
        for item in candidates
        for seed in spec["seeds"]
    ]


def _routeb_batch_command(
    *,
    base_config: dict[str, Any],
    python: Path,
    protocol_npz: Path,
    protocol_json: Path,
    output: Path,
    target: str,
    representation: str,
    mt: int,
    ms: int,
    seed: int,
) -> list[str]:
    routeb = base_config["routeb"]
    return [
        str(python),
        ROUTE_B_BATCH_SCRIPT,
        "--protocol-npz",
        str(protocol_npz),
        "--protocol-json",
        str(protocol_json),
        "--output-dir",
        str(output),
        "--data-part",
        "stream",
        "--target-mode",
        target,
        "--representation",
        representation,
        "--mt",
        str(mt),
        "--ms",
        str(ms),
        "--iterations",
        str(routeb["iterations"]),
        "--learning-rate",
        str(routeb["learning_rate"]),
        "--validation-every",
        str(routeb["validation_every"]),
        "--include-conditional-residual-variance",
        "--beta-prior-variance",
        str(routeb["beta_prior_variance"]),
        "--rff-sample-size",
        str(routeb["rff_sample_size"]),
        "--xlag-length",
        "10",
        "--prediction-chunk-size",
        str(routeb["prediction_chunk_size"]),
        "--split-seed",
        str(seed),
        "--model-seed",
        "0",
        "--device",
        str(base_config["primary_device"]),
        "--dtype",
        str(base_config["primary_dtype"]),
        "--evaluation-backend",
        "torch",
        "--warmup-steps",
        str(routeb["warmup_steps"]),
        "--profile-flops",
    ]


def build_routeb_jobs(
    *, spec: dict[str, Any], base_config: dict[str, Any], benchmark: Path, python: Path
) -> list[Job]:
    scope = str(spec["short_batch"]["scope"])
    routeb_spec = spec["short_batch"]["routeb"]
    jobs = []
    for target in routeb_spec["targets"]:
        target_name = "shared_residual" if target == "shared_xlag_residual" else "joint"
        for representation in routeb_spec["representations"]:
            for capacity in routeb_spec["capacities"]:
                for seed in spec["seeds"]:
                    protocol_npz, protocol_json = _protocol_dependencies(benchmark, scope, seed)
                    method = (
                        f"routeb_{target_name}_{representation}"
                        f"_mt{capacity['mt']}_ms{capacity['ms']}"
                    )
                    output = result_dir(benchmark, scope, "batch", method, seed)
                    jobs.append(
                        Job(
                            stage="stage2",
                            scope=scope,
                            method=method,
                            seed=seed,
                            python=python,
                            command=_routeb_batch_command(
                                base_config=base_config,
                                python=python,
                                protocol_npz=protocol_npz,
                                protocol_json=protocol_json,
                                output=output,
                                target=target,
                                representation=representation,
                                mt=int(capacity["mt"]),
                                ms=int(capacity["ms"]),
                                seed=seed,
                            ),
                            output_dir=output,
                            expected=(output / "result.json", output / "predictions.npz"),
                            dependencies=(protocol_npz, protocol_json),
                            timeout_seconds=_batch_timeout(base_config),
                            device_class="a100_routeb",
                        )
                    )
    return jobs


def build_short_batch_jobs(
    *,
    spec: dict[str, Any],
    base_config: dict[str, Any],
    benchmark: Path,
    pythons: dict[str, Path],
    gpflow_tier: str | None = None,
) -> list[Job]:
    jobs: list[Job] = []
    jobs.extend(
        build_protocol_jobs(
            spec=spec,
            base_config=base_config,
            benchmark=benchmark,
            python=pythons["routeb"],
        )
    )
    jobs.extend(
        build_calibration_jobs(
            spec=spec,
            base_config=base_config,
            benchmark=benchmark,
            python=pythons["routeb"],
        )
    )
    if spec["short_batch"].get("include_xlag", False):
        jobs.extend(
            build_xlag_jobs(
                spec=spec,
                base_config=base_config,
                benchmark=benchmark,
                python=pythons["routeb"],
            )
        )
    jobs.extend(
        build_official_jobs(
            spec=spec,
            base_config=base_config,
            benchmark=benchmark,
            python=pythons["stvgp"],
            scope=str(spec["short_batch"]["scope"]),
            items=spec["short_batch"]["official_models"],
        )
    )
    jobs.extend(
        build_markovflow_jobs(
            spec=spec, base_config=base_config, benchmark=benchmark, python=pythons["markovflow"]
        )
    )
    jobs.extend(
        build_gpflow_preflight_jobs(
            spec=spec, base_config=base_config, benchmark=benchmark, python=pythons["gpflow"]
        )
    )
    jobs.extend(
        build_gpflow_jobs(
            spec=spec,
            base_config=base_config,
            benchmark=benchmark,
            python=pythons["gpflow"],
            tier=gpflow_tier,
        )
    )
    jobs.extend(
        build_routeb_jobs(
            spec=spec, base_config=base_config, benchmark=benchmark, python=pythons["routeb"]
        )
    )
    return jobs


def build_long_preflight_jobs(
    *, spec: dict[str, Any], base_config: dict[str, Any], benchmark: Path, python: Path
) -> list[Job]:
    scope = str(spec["long_batch"]["scope"])
    return [
        _official_job(
            benchmark=benchmark,
            python=python,
            scope=scope,
            seed=0,
            item=item,
            iterations=int(spec["long_batch"]["preflight_iterations"]),
            timeout_seconds=_legacy_timeout(base_config),
            branch="preflight",
            method_prefix="official_preflight_",
            device_class="a100_official_preflight",
        )
        for item in spec["long_batch"]["preflight_models"]
    ]


def build_long_full_jobs(
    *, spec: dict[str, Any], base_config: dict[str, Any], benchmark: Path, python: Path
) -> list[Job]:
    return build_official_jobs(
        spec=spec,
        base_config=base_config,
        benchmark=benchmark,
        python=python,
        scope=str(spec["long_batch"]["scope"]),
        items=spec["long_batch"]["official_full_models"],
    )


def build_long_batch_jobs(
    *, spec: dict[str, Any], base_config: dict[str, Any], benchmark: Path, pythons: dict[str, Path]
) -> list[Job]:
    """Return both long slices for callers that need the aggregate view."""
    return [
        *build_long_preflight_jobs(
            spec=spec, base_config=base_config, benchmark=benchmark, python=pythons["stvgp"]
        ),
        *build_long_full_jobs(
            spec=spec, base_config=base_config, benchmark=benchmark, python=pythons["stvgp"]
        ),
    ]


def _routeb_online_job(
    *,
    spec: dict[str, Any],
    base_config: dict[str, Any],
    benchmark: Path,
    python: Path,
    scope: str,
    representation: dict[str, str],
    capacity: dict[str, int],
    seed: int,
) -> Job:
    protocol_npz, protocol_json = _protocol_dependencies(benchmark, scope, seed)
    rep = str(representation["representation"])
    theta = benchmark / "calibration" / f"routeb_joint_{rep}" / f"seed{seed}" / "result.json"
    method = f"routeb_{representation['name']}_mt{capacity['mt']}_ms{capacity['ms']}"
    output = result_dir(benchmark, scope, "online", method, seed)
    routeb = base_config["routeb"]
    temporal_device = str(
        routeb.get("temporal_factor_device", {}).get(rep, "auto")
    )
    command = [
        str(python),
        ROUTE_B_ONLINE_SCRIPT,
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
        rep,
        "--mt",
        str(capacity["mt"]),
        "--ms",
        str(capacity["ms"]),
        "--rff-sample-size",
        str(routeb["rff_sample_size"]),
        "--prediction-chunk-size",
        str(routeb["prediction_chunk_size"]),
        "--include-conditional-residual-variance",
        "--seed",
        str(seed),
        "--solver-backend",
        "torch",
        "--device",
        str(base_config["primary_device"]),
        "--temporal-factor-device",
        temporal_device,
        "--dtype",
        str(base_config["primary_dtype"]),
    ]
    return Job(
        stage="stage3",
        scope=scope,
        method=method,
        seed=seed,
        python=python,
        command=command,
        output_dir=output,
        expected=(output / "result.json", output / "blocks.csv", output / "predictions.npz"),
        dependencies=(protocol_npz, protocol_json, theta),
        timeout_seconds=int(base_config["timeouts_seconds"]["online"]),
        device_class="a100_routeb_online",
    )


def _with_routeb_variance_flag(job: Job) -> Job:
    if ROUTE_B_ONLINE_SCRIPT not in job.command:
        return job
    if "--include-conditional-residual-variance" in job.command:
        return job
    return replace(job, command=[*job.command, "--include-conditional-residual-variance"])


def _postprocess_job(
    *, job: Job, base_config: dict[str, Any], routeb_python: Path, num_segments: int
) -> Job:
    predictions = job.output_dir / "predictions.npz"
    return Job(
        stage="stage3",
        scope=job.scope,
        method=f"postprocess_task2_online_segments_{job.method}",
        seed=job.seed,
        python=routeb_python,
        command=[
            str(routeb_python),
            POSTPROCESS_SCRIPT,
            "--predictions",
            str(predictions),
            "--output-dir",
            str(job.output_dir),
            "--num-segments",
            str(num_segments),
        ],
        output_dir=job.output_dir,
        expected=(
            job.output_dir / "task2_online_segments.csv",
            job.output_dir / "task2_online_segments.json",
        ),
        dependencies=(predictions,),
        timeout_seconds=int(base_config["timeouts_seconds"]["report"]),
        device_class="cpu_postprocess",
        status_path=job.output_dir / ".task2_online_segments" / "status.json",
    )


def build_online_jobs(
    *, spec: dict[str, Any], base_config: dict[str, Any], benchmark: Path, pythons: dict[str, Path]
) -> list[tuple[Job, str]]:
    online_spec = spec["online"]
    if not online_spec.get("include_existing_stage3", False):
        return []
    selected_scopes = set(online_spec["scopes"])
    source_jobs = stage3_jobs(base_config, benchmark, pythons)
    jobs: list[tuple[Job, str]] = []
    postprocess_segments = int(online_spec["task2_postprocess"]["num_segments"])

    for scope in sorted(selected_scopes):
        for source in source_jobs:
            if source.scope != scope or ROUTE_B_ONLINE_SCRIPT in source.command:
                continue
            online_job = _with_routeb_variance_flag(source)
            jobs.append((online_job, "online"))
            if scope == "task1_2" and online_job.output_dir / "predictions.npz" in online_job.expected:
                jobs.append(
                    (
                        _postprocess_job(
                            job=online_job,
                            base_config=base_config,
                            routeb_python=pythons["routeb"],
                            num_segments=postprocess_segments,
                        ),
                        "postprocess",
                    )
                )

        for representation in online_spec["routeb"]["representations"]:
            for capacity in online_spec["routeb"]["capacities"]:
                for seed in spec["seeds"]:
                    online_job = _routeb_online_job(
                        spec=spec,
                        base_config=base_config,
                        benchmark=benchmark,
                        python=pythons["routeb"],
                        scope=scope,
                        representation=representation,
                        capacity=capacity,
                        seed=seed,
                    )
                    jobs.append((online_job, "online"))
                    if scope == "task1_2":
                        jobs.append(
                            (
                                _postprocess_job(
                                    job=online_job,
                                    base_config=base_config,
                                    routeb_python=pythons["routeb"],
                                    num_segments=postprocess_segments,
                                ),
                                "postprocess",
                            )
                        )
    return jobs


def _efficiency_metadata(
    *,
    efficiency_spec: dict[str, Any],
    table: str,
    method_family: str,
    configuration: dict[str, Any],
    measurement_status: str,
    objective: str | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    return {
        "table": table,
        "method_family": method_family,
        "configuration": configuration,
        "measurement_status": measurement_status,
        "objective": str(objective or efficiency_spec["objective"]),
        "unit": str(unit or efficiency_spec["unit"]),
        "warmup": int(efficiency_spec["warmup"]),
        "repeats": int(efficiency_spec["repeats"]),
        "execution": {
            "gpu_count": int(efficiency_spec["gpu_count"]),
            "serial": bool(efficiency_spec["serial"]),
        },
    }


def _efficiency_probe_job(
    *,
    base_config: dict[str, Any],
    benchmark: Path,
    python: Path,
    scope: str,
    method: str,
    command: list[str],
    output: Path,
    expected: tuple[Path, ...],
    dependencies: tuple[Path, ...],
) -> Job:
    return Job(
        stage="stage4",
        scope=scope,
        method=method,
        seed=0,
        python=python,
        command=command,
        output_dir=output,
        expected=expected,
        dependencies=dependencies,
        timeout_seconds=int(base_config["timeouts_seconds"]["report"]),
        device_class="a100_efficiency_probe",
    )


def _temporal_efficiency_probe(
    *,
    base_config: dict[str, Any],
    spec: dict[str, Any],
    benchmark: Path,
    python: Path,
) -> tuple[Job, dict[str, Any]]:
    efficiency = spec["efficiency"]
    output = benchmark / "efficiency" / "probes" / "temporal_analytic_mt128"
    command = [
        str(python),
        "scripts/profile_temporal_analytic_cpu_gpu.py",
        "--output",
        str(output / "temporal_profile.json"),
        "--mt",
        "128",
        "--rff-sample-size",
        str(base_config["routeb"]["rff_sample_size"]),
        "--repeats",
        str(efficiency["repeats"]),
        "--warmup",
        str(efficiency["warmup"]),
        "--dtype",
        str(base_config["primary_dtype"]),
    ]
    job = _efficiency_probe_job(
        base_config=base_config,
        benchmark=benchmark,
        python=python,
        scope="all",
        method="probe_table2a_analytic_hippo_temporal_mt128",
        command=command,
        output=output,
        expected=(output / "temporal_profile.json",),
        dependencies=(),
    )
    metadata = _efficiency_metadata(
        efficiency_spec=efficiency,
        table="Table2A",
        method_family="analytic_hippo_temporal_factor",
        configuration={"mt": 128, "rff_sample_size": int(base_config["routeb"]["rff_sample_size"])},
        measurement_status="instrumented",
        objective="analytic HiPPO temporal basis and covariance construction wall time",
        unit="seconds_per_operation",
    )
    return job, metadata


def _batch_efficiency_probe(
    *,
    base_config: dict[str, Any],
    spec: dict[str, Any],
    benchmark: Path,
    python: Path,
) -> tuple[Job, dict[str, Any]]:
    efficiency = spec["efficiency"]
    protocol_npz, protocol_json = _protocol_dependencies(benchmark, "task1_2", 0)
    theta = benchmark / "calibration" / "routeb_joint_analytic_hippo_rff" / "seed0" / "result.json"
    data_root = resolve_path(base_config["data"]["short_root"])
    output = benchmark / "efficiency" / "probes" / "table2a_routeb_joint_analytic_mt128_ms128"
    command = [
        str(python),
        "scripts/benchmark_routeb_batch_objective.py",
        "--protocol-npz",
        str(protocol_npz),
        "--protocol-json",
        str(protocol_json),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output),
        "--scope",
        "task2_short",
        "--seed",
        "0",
        "--objective",
        "finite_dtc",
        "--versions",
        "E3",
        "--theta-json",
        str(theta),
        "--mt",
        "128",
        "--ms",
        "128",
        "--rff-sample-size",
        str(base_config["routeb"]["rff_sample_size"]),
        "--warmup",
        str(efficiency["warmup"]),
        "--repeats",
        str(efficiency["repeats"]),
    ]
    job = _efficiency_probe_job(
        base_config=base_config,
        benchmark=benchmark,
        python=python,
        scope="task1_2",
        method="probe_table2a_routeb_joint_analytic_hippo_mt128_ms128",
        command=command,
        output=output,
        expected=(output / "batch_ablation.json", output / "batch_ablation.csv"),
        dependencies=(protocol_npz, protocol_json, theta),
    )
    metadata = _efficiency_metadata(
        efficiency_spec=efficiency,
        table="Table2A",
        method_family="routeb_cumulative_hippo",
        configuration={
            "target_mode": "joint_xlag",
            "representation": "analytic_hippo_rff",
            "mt": 128,
            "ms": 128,
            "objective_source": "finite_dtc",
        },
        measurement_status="instrumented",
        objective="steady-state Route B finite-DTC objective forward/backward step",
        unit="seconds_per_step",
    )
    metadata["variance_protocol"] = "full_joint_conditional"
    metadata["explicit_runner_flag"] = "--include-conditional-residual-variance"
    return job, metadata


def _online_efficiency_probe(
    *,
    base_config: dict[str, Any],
    spec: dict[str, Any],
    benchmark: Path,
    python: Path,
    scope: str,
    representation: str,
    table: str,
) -> tuple[Job, dict[str, Any]]:
    efficiency = spec["efficiency"]
    protocol_npz, protocol_json = _protocol_dependencies(benchmark, scope, 0)
    theta = benchmark / "calibration" / f"routeb_joint_{representation}" / "seed0" / "result.json"
    name = "global_inducing" if representation == "inducing_points" else "cumulative_hippo"
    output = benchmark / "efficiency" / "probes" / f"{table.lower()}_{scope}_{name}_mt128_ms128"
    command = [
        str(python),
        "scripts/profile_routeb_backend_bottlenecks.py",
        "--protocol-npz",
        str(protocol_npz),
        "--protocol-json",
        str(protocol_json),
        "--theta-json",
        str(theta),
        "--output-dir",
        str(output),
        "--representation",
        representation,
        "--mt",
        "128",
        "--ms",
        "128",
        "--rff-sample-size",
        str(base_config["routeb"]["rff_sample_size"]),
        "--prediction-chunk-size",
        str(base_config["routeb"]["prediction_chunk_size"]),
        "--seed",
        "0",
        "--max-blocks",
        "1",
        "--repeats",
        str(efficiency["repeats"]),
    ]
    job = _efficiency_probe_job(
        base_config=base_config,
        benchmark=benchmark,
        python=python,
        scope=scope,
        method=f"probe_{table.lower()}_routeb_{name}_mt128_ms128",
        command=command,
        output=output,
        expected=(output / "summary.json", output / "summary.csv"),
        dependencies=(protocol_npz, protocol_json, theta),
    )
    metadata = _efficiency_metadata(
        efficiency_spec=efficiency,
        table=table,
        method_family=f"routeb_{name}",
        configuration={
            "representation": representation,
            "mt": 128,
            "ms": 128,
            "max_blocks": 1,
            "variance_protocol": "full_joint_conditional",
        },
        measurement_status="instrumented",
        objective="strict-online Route B update and prediction wall time",
        unit="seconds_per_block",
    )
    metadata["variance_protocol"] = "full_joint_conditional"
    metadata["explicit_runner_flag"] = "--include-conditional-residual-variance"
    return job, metadata


def _efficiency_reference_job(
    *,
    base_config: dict[str, Any],
    spec: dict[str, Any],
    benchmark: Path,
    python: Path,
    table: str,
    source_method: str,
    method_family: str,
    configuration: dict[str, Any],
    measurement_status: str,
    objective: str,
) -> tuple[Job, dict[str, Any]]:
    efficiency = spec["efficiency"]
    slug = source_method.replace("/", "_")
    output = benchmark / "efficiency" / "records" / table.lower() / slug
    expected = output / "record.json"
    payload = {
        "schema_version": ENTRY_SCHEMA_VERSION,
        "table": table,
        "method": source_method,
        "seed": int(efficiency["seed"]),
        "measurement_status": measurement_status,
        "objective": objective,
        "unit": "seconds_per_step",
        "warmup": int(efficiency["warmup"]),
        "repeats": int(efficiency["repeats"]),
        "configuration": configuration,
        "note": "No compatible implementation-level counter is exposed by this baseline wrapper.",
    }
    payload_text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    writer = (
        "from pathlib import Path; "
        f"p=Path({str(expected)!r}); "
        "p.parent.mkdir(parents=True, exist_ok=True); "
        f"p.write_text({payload_text!r}, encoding='utf-8')"
    )
    job = Job(
        stage="stage4",
        scope=table.lower(),
        method=f"record_{table.lower()}_{slug}",
        seed=int(efficiency["seed"]),
        python=python,
        command=[str(python), "-c", writer],
        output_dir=output,
        expected=(expected,),
        dependencies=(),
        timeout_seconds=int(base_config["timeouts_seconds"]["report"]),
        device_class="a100_efficiency_serial",
    )
    metadata = _efficiency_metadata(
        efficiency_spec=efficiency,
        table=table,
        method_family=method_family,
        configuration=configuration,
        measurement_status=measurement_status,
        objective=objective,
    )
    metadata["compute_source_method"] = source_method
    return job, metadata


def _method_family(method: str) -> str:
    if method.startswith("probe_table") and "routeb" in method:
        return "routeb"
    if method == "xlag_mean_only":
        return "xlag_mean"
    if method.startswith("official_mf_st_svgp"):
        return "official_mf_st_svgp"
    if method.startswith("official_st_vgp"):
        return "official_st_vgp"
    if method.startswith("official_st_svgp"):
        return "official_st_svgp"
    if method.startswith("markovflow_"):
        return "markovflow"
    if method.startswith("preflight_gpflow_"):
        return "gpflow"
    if method.startswith("gpflow_feasibility_"):
        return "gpflow"
    if method.startswith("routeb_"):
        return "routeb"
    if method.startswith("bui_"):
        return "bui_osgpr"
    if method.startswith("maddox_"):
        return "maddox_streaming_sgpr"
    if method.startswith("official_ohsvgp"):
        return "official_ohsvgp"
    if method.startswith("xlag_"):
        return "xlag_online"
    return method


def _replace_option(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    try:
        index = result.index(flag)
    except ValueError:
        result.extend((flag, value))
    else:
        result[index + 1] = value
    return result


def _profile_output_path(source: Job, profile_root: Path, value: str) -> str:
    source_root = str(source.output_dir)
    if value == source_root:
        return str(profile_root)
    if value.startswith(source_root + "/"):
        return str(profile_root / Path(value).relative_to(source.output_dir))
    return value


def _profile_job(
    *,
    source: Job,
    profile_root: Path,
    branch: str,
    timeout_seconds: int,
) -> Job:
    command = [
        _profile_output_path(source, profile_root, value)
        for value in source.command
    ]
    script = str(command[1]) if len(command) > 1 else ""
    if branch == "batch":
        command = _replace_option(command, "--iterations", "2")
    elif script == "scripts/run_official_bui_osgpr_era5.py":
        command = _replace_option(command, "--max-stream-blocks", "2")
    else:
        command = _replace_option(command, "--max-blocks", "2")
    expected = tuple(
        profile_root / path.relative_to(source.output_dir)
        if path.is_relative_to(source.output_dir)
        else path
        for path in source.expected
    )
    return replace(
        source,
        stage="stage4",
        command=command,
        output_dir=profile_root,
        expected=expected,
        timeout_seconds=timeout_seconds,
        device_class="a100_ncu_profile",
    )


def _profile_metadata(
    *,
    source: Job,
    branch: str,
    efficiency_spec: dict[str, Any],
) -> dict[str, Any]:
    script = str(source.command[1]) if len(source.command) > 1 else ""
    precision = (
        str(source.command[source.command.index("--dtype") + 1])
        if "--dtype" in source.command
        else "float64"
    )
    if branch == "batch":
        range_name = "era5_batch_update"
        stochastic = script in {
            "scripts/run_official_gpflow_svgp_era5.py",
            "scripts/run_official_markovflow_stsvgp_era5.py",
        }
        unit = "one_full_data_pass" if stochastic else "one_full_fit_optimization_update"
        native_unit = "one_minibatch_optimization_update" if stochastic else unit
        comparison_group = "stochastic_full_data_pass" if stochastic else "batch_full_fit_update"
        objective = "one steady-state batch optimization update"
    else:
        range_name = "era5_online_block"
        unit = "one_arrival_block_update_and_prediction"
        native_unit = unit
        comparison_group = "online_arrival_block"
        objective = "one steady-state strict-online block update and prediction"
    metadata = _efficiency_metadata(
        efficiency_spec=efficiency_spec,
        table=("Table2A" if branch == "batch" else "Table3A" if source.scope == "task1_2" else "Table3B"),
        method_family=_method_family(source.method),
        configuration=_job_configuration(source),
        measurement_status="scheduled_common_hardware_counter",
        objective=objective,
        unit=unit,
    )
    metadata.update(
        {
            "compute_source_method": source.method,
            "manifest_branch": branch,
            "native_work_unit": native_unit,
            "comparison_group": comparison_group,
            "precision": precision,
            "hardware_class": "NVIDIA A100",
            "warmup": 1,
            "repeats": 1,
            "ncu": {
                "enabled": True,
                "range": range_name,
                "work_unit": unit,
                "target": "last",
            },
        }
    )
    return metadata


def _job_configuration(job: Job) -> dict[str, Any]:
    configuration: dict[str, Any] = {"method": job.method}
    for flag, key in (("--mt", "mt"), ("--ms", "ms"), ("--representation", "representation")):
        if flag in job.command:
            value = job.command[job.command.index(flag) + 1]
            configuration[key] = int(value) if key in {"mt", "ms"} else value
    if "--target-mode" in job.command:
        configuration["target_mode"] = job.command[job.command.index("--target-mode") + 1]
    return configuration


def _compute_contract(
    job: Job,
    branch: str,
    *,
    source_method: str | None = None,
    cpu_only: bool = False,
) -> dict[str, Any]:
    """Declare the only FLOP comparison scope a run may enter."""

    script = str(job.command[1]) if len(job.command) > 1 else ""
    method = source_method or job.method
    family = _method_family(method)
    if cpu_only or family in {"xlag_mean", "xlag_online"}:
        return {
            "schema_version": 2,
            "baseline_family": family,
            "branch": branch,
            "data_access_unit": "cpu_only_reference",
            "measurement_scope": "not_applicable",
            "work_unit": "not_applicable",
            "native_work_unit": "not_applicable",
            "comparison_group": "not_applicable",
            "required_measurement_backend": "not_applicable",
            "comparison_status": "not_applicable",
        }
    mode = branch
    if branch == "efficiency":
        if script in {
            "scripts/benchmark_routeb_batch_objective.py",
            "scripts/run_iclr_era5_routeb_batch.py",
        } or job.method.startswith("record_table2"):
            mode = "batch"
        elif script in {
            "scripts/profile_routeb_backend_bottlenecks.py",
            "scripts/run_iclr_era5_routeb_strict_online.py",
        } or job.method.startswith("record_table3"):
            mode = "online"
    elif branch in {"official_preflight", "official_full", "shared_batch_short"}:
        mode = "batch"
    elif branch in {"online_short", "online_long", "postprocess"}:
        mode = "online"

    if mode == "batch":
        if script in {
            "scripts/run_official_gpflow_svgp_era5.py",
            "scripts/run_official_markovflow_stsvgp_era5.py",
        } or method.startswith(("gpflow_", "preflight_gpflow_", "markovflow_")):
            data_access = "stochastic_minibatch"
            unit = "one_full_data_pass"
            native_unit = "one_minibatch_optimization_update"
            comparison_group = "stochastic_full_data_pass"
        elif script in {
            ROUTE_B_BATCH_SCRIPT,
            "scripts/benchmark_routeb_batch_objective.py",
            "scripts/run_official_stvgp_legacy.py",
        } or method.startswith(("routeb_", "official_", "markovflow_")):
            data_access = "full_fit_dataset"
            unit = "one_full_fit_optimization_update"
            native_unit = unit
            comparison_group = "batch_full_fit_update"
        else:
            data_access = "undeclared"
            unit = "undeclared"
            native_unit = "undeclared"
            comparison_group = "undeclared"
        scope = "optimization_update"
    elif mode == "online":
        data_access = "arrival_block"
        unit = "one_arrival_block_update_and_prediction"
        native_unit = unit
        comparison_group = "online_arrival_block"
        scope = "block_update_and_prediction"
    else:
        data_access = "not_applicable"
        unit = "not_applicable"
        native_unit = unit
        comparison_group = unit
        scope = "not_applicable"
    return {
        "schema_version": 2,
        "baseline_family": family,
        "branch": branch,
        "data_access_unit": data_access,
        "measurement_scope": scope,
        "work_unit": unit,
        "native_work_unit": native_unit,
        "comparison_group": comparison_group,
        "required_measurement_backend": "nsight_compute_executed_gpu_flops",
        "comparison_status": (
            "pending_common_hardware_counter"
            if scope != "not_applicable"
            else "not_applicable"
        ),
    }


def build_efficiency_jobs(
    *, spec: dict[str, Any], base_config: dict[str, Any], benchmark: Path, pythons: dict[str, Path]
) -> list[tuple[Job, dict[str, Any], str]]:
    """Build seed-0 probes and isolated common-counter baseline profiles."""
    routeb_python = pythons["routeb"]
    entries: list[tuple[Job, dict[str, Any], str]] = []
    for builder in (
        _temporal_efficiency_probe,
        _batch_efficiency_probe,
    ):
        job, metadata = builder(
            base_config=base_config,
            spec=spec,
            benchmark=benchmark,
            python=routeb_python,
        )
        entries.append((job, metadata, "efficiency_probe"))
    for scope, table in (("task1_2", "Table3A"), ("task1_10", "Table3B")):
        for representation in ("analytic_hippo_rff", "inducing_points"):
            job, metadata = _online_efficiency_probe(
                base_config=base_config,
                spec=spec,
                benchmark=benchmark,
                python=routeb_python,
                scope=scope,
                representation=representation,
                table=table,
            )
            entries.append((job, metadata, "efficiency_probe"))

    short_jobs = build_short_batch_jobs(
        spec=spec,
        base_config=base_config,
        benchmark=benchmark,
        pythons=pythons,
    )
    for source in short_jobs:
        if source.seed != 0 or source.scope != "task1_2":
            continue
        script = str(source.command[1]) if len(source.command) > 1 else ""
        if (
            script in NCU_BATCH_SCRIPTS
            and script != "scripts/run_official_markovflow_stsvgp_era5.py"
            and not source.method.startswith("preflight_gpflow_")
        ):
            profile_root = (
                benchmark
                / "efficiency"
                / "profiles"
                / "batch"
                / source.scope
                / source.method
                / f"seed{source.seed}"
            )
            job = _profile_job(
                source=source,
                profile_root=profile_root,
                branch="batch",
                timeout_seconds=source.timeout_seconds,
            )
            entries.append(
                (job, _profile_metadata(source=source, branch="batch", efficiency_spec=spec["efficiency"]), "efficiency_profile")
            )
        elif source.method == "xlag_mean_only" or script == "scripts/run_official_markovflow_stsvgp_era5.py":
            objective = (
                "CPU-only linear X-lag ridge mean reference; excluded from GPU FLOP ratios"
                if source.method == "xlag_mean_only"
                else "CPU-only Markovflow/TF2.2 compatibility baseline; excluded from A100 FLOP ratios"
            )
            job, metadata = _efficiency_reference_job(
                base_config=base_config,
                spec=spec,
                benchmark=benchmark,
                python=routeb_python,
                table="Table2A",
                source_method=source.method,
                method_family=_method_family(source.method),
                configuration=_job_configuration(source),
                measurement_status="cpu_not_applicable",
                objective=objective,
            )
            entries.append((job, metadata, "efficiency_record"))

    for source in build_long_full_jobs(
        spec=spec,
        base_config=base_config,
        benchmark=benchmark,
        python=pythons["stvgp"],
    ):
        if source.seed != 0:
            continue
        profile_root = (
            benchmark
            / "efficiency"
            / "profiles"
            / "batch"
            / source.scope
            / source.method
            / f"seed{source.seed}"
        )
        job = _profile_job(
            source=source,
            profile_root=profile_root,
            branch="batch",
            timeout_seconds=source.timeout_seconds,
        )
        entries.append(
            (job, _profile_metadata(source=source, branch="batch", efficiency_spec=spec["efficiency"]), "efficiency_profile")
        )

    online_jobs = build_online_jobs(
        spec=spec,
        base_config=base_config,
        benchmark=benchmark,
        pythons=pythons,
    )
    for source, kind in online_jobs:
        if kind != "online" or source.seed != 0:
            continue
        table = "Table3A" if source.scope == "task1_2" else "Table3B"
        script = str(source.command[1]) if len(source.command) > 1 else ""
        if script in NCU_ONLINE_SCRIPTS:
            profile_root = (
                benchmark
                / "efficiency"
                / "profiles"
                / "online"
                / source.scope
                / source.method
                / f"seed{source.seed}"
            )
            job = _profile_job(
                source=source,
                profile_root=profile_root,
                branch="online",
                timeout_seconds=source.timeout_seconds,
            )
            entries.append(
                (job, _profile_metadata(source=source, branch="online", efficiency_spec=spec["efficiency"]), "efficiency_profile")
            )
        elif source.method.startswith("xlag_"):
            job, metadata = _efficiency_reference_job(
                base_config=base_config,
                spec=spec,
                benchmark=benchmark,
                python=routeb_python,
                table=table,
                source_method=source.method,
                method_family=_method_family(source.method),
                configuration=_job_configuration(source),
                measurement_status="cpu_not_applicable",
                objective="CPU-only X-lag online reference; excluded from GPU FLOP ratios",
            )
            entries.append((job, metadata, "efficiency_record"))

    output = benchmark / "efficiency"
    summary = Job(
        stage="stage4",
        scope="all",
        method="efficiency_summary",
        seed=None,
        python=routeb_python,
        command=[
            str(routeb_python),
            "scripts/summarize_era5_a100_efficiency.py",
            "--benchmark-root",
            str(benchmark),
            "--output",
            str(output / "era5_a100_efficiency.csv"),
            "--ratio-output",
            str(output / "era5_a100_flop_ratios.csv"),
            "--markdown-output",
            str(output / "era5_a100_efficiency.md"),
        ],
        output_dir=output,
        expected=(
            output / "era5_a100_efficiency.csv",
            output / "era5_a100_flop_ratios.csv",
            output / "era5_a100_efficiency.md",
        ),
        dependencies=(),
        timeout_seconds=int(base_config["timeouts_seconds"]["report"]),
        device_class="a100_efficiency_serial",
    )
    summary_metadata = _efficiency_metadata(
        efficiency_spec=spec["efficiency"],
        table="all",
        method_family="efficiency_aggregation",
        configuration={"source": "completed benchmark runs"},
        measurement_status="aggregate_only",
        objective="aggregate completed-run timing and resource artifacts",
        unit="mixed_normalized_columns",
    )
    entries.append((summary, summary_metadata, "efficiency_summary"))
    return entries


def job_entry(
    job: Job,
    *,
    manifest_id: str,
    kind: str,
    branch: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_method = None
    cpu_only = False
    if metadata:
        value = metadata.get("compute_source_method")
        if isinstance(value, str):
            source_method = value
        cpu_only = metadata.get("measurement_status") == "cpu_not_applicable"
    entry: dict[str, Any] = {
        "argv": [str(value) for value in job.command],
        "branch": branch,
        "dependencies": [str(path) for path in job.dependencies],
        "device_class": job.device_class,
        "expected": [str(path) for path in job.expected],
        "job_id": job.identifier,
        "kind": kind,
        "legacy": job.legacy,
        "manifest_id": manifest_id,
        "method": job.method,
        "output_dir": str(job.output_dir),
        "python": str(job.python),
        "schema_version": ENTRY_SCHEMA_VERSION,
        "scope": job.scope,
        "seed": job.seed,
        "stage": job.stage,
        "timeout_seconds": job.timeout_seconds,
        "compute_contract": _compute_contract(
            job,
            branch,
            source_method=source_method,
            cpu_only=cpu_only,
        ),
    }
    if job.status_path is not None:
        entry["status_path"] = str(job.status_path)
    if "scripts/run_official_stvgp_legacy.py" in job.command:
        cuda_root = job.python.parent.parent.resolve()
        entry["env"] = {
            "JAX_PLATFORM_NAME": "gpu",
            "LD_LIBRARY_PATH": str(cuda_root / "lib"),
            "PATH": f"{cuda_root / 'bin'}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "XLA_FLAGS": f"--xla_gpu_cuda_data_dir={cuda_root}",
        }
    if metadata:
        entry.update(metadata)
    return entry


def _entry_metadata(job: Job, spec: dict[str, Any], kind: str) -> dict[str, Any]:
    if job.method.startswith("preflight_gpflow_feasibility_"):
        tier, _ = _gpflow_candidates(
            spec, str(spec["short_batch"]["gpflow_feasibility"]["preflight_tier"])
        )
        return {
            "selection": {
                "family": "gpflow_feasibility",
                "role": "preflight",
                "tier": tier,
                "decision_seed": 0,
            }
        }
    if job.method.startswith("gpflow_feasibility_"):
        tier, _ = _gpflow_candidates(spec)
        return {
            "selection": {
                "family": "gpflow_feasibility",
                "role": "selected",
                "tier": tier,
                "selected_by": "seed0_preflight",
            }
        }
    if ROUTE_B_BATCH_SCRIPT in job.command or ROUTE_B_ONLINE_SCRIPT in job.command:
        return {"variance_protocol": "full_joint_conditional"}
    if kind == "efficiency":
        return {"execution": {"gpu_count": 1, "serial": True}}
    return {}


def write_jsonl(path: Path, entries: Iterable[dict[str, Any]]) -> None:
    lines = [
        json.dumps(entry, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for entry in entries
    ]
    if not lines:
        raise ValueError(f"Refusing to write an empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _argument(entry: dict[str, Any], flag: str) -> str:
    argv = entry["argv"]
    return argv[argv.index(flag) + 1]


def _routeb_entries(entries: list[dict[str, Any]], script: str) -> list[dict[str, Any]]:
    return [entry for entry in entries if len(entry["argv"]) > 1 and entry["argv"][1] == script]


def _validate_entries(
    *,
    short_entries: list[dict[str, Any]],
    long_preflight_entries: list[dict[str, Any]],
    long_full_entries: list[dict[str, Any]],
    online_short_entries: list[dict[str, Any]],
    online_long_entries: list[dict[str, Any]],
    efficiency_entries: list[dict[str, Any]],
) -> None:
    if any(
        _argument(entry, "--target-mode") == "direct"
        for entry in short_entries
        if "--target-mode" in entry["argv"]
    ):
        raise AssertionError("The shared-Xlag batch manifest cannot contain direct-target jobs")

    short_routeb = [
        entry
        for entry in _routeb_entries(short_entries, ROUTE_B_BATCH_SCRIPT)
        if entry["scope"] == "task1_2"
    ]
    if len(short_routeb) != 60:
        raise AssertionError("Short Route B batch must contain 60 jobs")
    short_capacities = {
        (int(_argument(entry, "--mt")), int(_argument(entry, "--ms")))
        for entry in short_routeb
    }
    if short_capacities != {(128, 128), (128, 64), (64, 64)}:
        raise AssertionError("Short Route B batch capacities are incomplete")
    if any("--include-conditional-residual-variance" not in entry["argv"] for entry in short_routeb):
        raise AssertionError("All Route B batch jobs need the explicit variance protocol")
    short_official = [entry for entry in short_entries if entry["kind"] == "batch" and entry["method"].startswith("official_")]
    if not any(entry["method"] == "official_st_vgp_full" for entry in short_official):
        raise AssertionError("Short official matrix must name ST-VGP as st_vgp_full")
    if any(
        entry["method"] == "official_st_vgp_full"
        and "--num-spatial-inducing" in entry["argv"]
        for entry in short_official
    ):
        raise AssertionError("Full ST-VGP must not receive a sparse spatial-inducing argument")

    protocol_entries = [entry for entry in short_entries if entry["kind"] == "protocol"]
    calibration_entries = [entry for entry in short_entries if entry["kind"] == "calibration"]
    if not protocol_entries:
        raise AssertionError("The shared batch manifest needs a protocol producer")
    if len(calibration_entries) != 10:
        raise AssertionError("The shared batch manifest needs both Route B theta producers for seeds 0-4")
    if {entry["method"] for entry in calibration_entries} != {
        "routeb_joint_analytic_hippo_rff",
        "routeb_joint_inducing_points",
    }:
        raise AssertionError("Route B calibration representations are incomplete")
    if any("--include-conditional-residual-variance" not in entry["argv"] for entry in calibration_entries):
        raise AssertionError("Calibration Route B commands need the explicit variance protocol")

    gpflow_preflight = [entry for entry in short_entries if entry["kind"] == "gpflow_feasibility_preflight"]
    gpflow_selected = [entry for entry in short_entries if entry["method"].startswith("gpflow_feasibility_")]
    if len(gpflow_preflight) != 2 or len(gpflow_selected) != 10:
        raise AssertionError("GPflow must have two seed0 preflights and one selected tier")
    if {entry["seed"] for entry in gpflow_preflight} != {0}:
        raise AssertionError("GPflow feasibility preflights must use seed0")
    if any(entry["selection"]["role"] != "preflight" for entry in gpflow_preflight):
        raise AssertionError("GPflow preflights must be marked in the selection schema")
    if any(entry["selection"]["role"] != "selected" for entry in gpflow_selected):
        raise AssertionError("GPflow main jobs must be marked as the selected tier")
    selected_tier = {entry["selection"]["tier"] for entry in gpflow_selected}
    if len(selected_tier) != 1:
        raise AssertionError("GPflow main jobs must use one selected inducing tier")
    selected_total = int(next(iter(selected_tier)))
    if any(int(_argument(entry, "--mt")) * int(_argument(entry, "--ms")) != selected_total for entry in gpflow_selected):
        raise AssertionError("GPflow main jobs must use only the selected inducing tier")

    if len(long_preflight_entries) != 3:
        raise AssertionError("Long preflight manifest must contain exactly three jobs")
    if {entry["seed"] for entry in long_preflight_entries} != {0}:
        raise AssertionError("Long preflight jobs must all use seed0")
    if {entry["method"] for entry in long_preflight_entries} != {
        "official_preflight_st_vgp_full",
        "official_preflight_st_svgp_ms64",
        "official_preflight_st_svgp_ms128",
    }:
        raise AssertionError("Long preflight models are not the requested official resource tests")
    for entry in long_preflight_entries:
        if entry["argv"][1] != "scripts/run_official_stvgp_legacy.py":
            raise AssertionError("Long preflights must be real official experiments")
        if _argument(entry, "--iterations") != "2":
            raise AssertionError("Long preflights must use short iterations")
        if "official_protocol/task1_10" not in _argument(entry, "--data-npz"):
            raise AssertionError("Long preflights must use task1_10 long data")
    if any(
        entry["method"].endswith("st_vgp_full")
        and "--num-spatial-inducing" in entry["argv"]
        for entry in long_preflight_entries
    ):
        raise AssertionError("Full ST-VGP preflight must not receive a sparse spatial-inducing argument")

    if len(long_full_entries) != 15:
        raise AssertionError("Long full manifest must contain exactly 15 jobs")
    if any(entry["kind"] != "official_full" for entry in long_full_entries):
        raise AssertionError("Long full manifest contains a non-full entry")
    if any(entry["argv"][1] != "scripts/run_official_stvgp_legacy.py" for entry in long_full_entries):
        raise AssertionError("Long full jobs must use the official ST-VGP runner")
    if {entry["method"] for entry in long_full_entries} != {
        "official_st_vgp_full",
        "official_st_svgp_ms64",
        "official_st_svgp_ms128",
    }:
        raise AssertionError("Long full method names do not match the official configurations")
    long_text = " ".join(" ".join(entry["argv"]) for entry in long_full_entries)
    if any(
        forbidden in long_text
        for forbidden in (
            ROUTE_B_BATCH_SCRIPT,
            "scripts/run_official_gpflow_svgp_era5.py",
            "scripts/run_official_markovflow_stsvgp_era5.py",
        )
    ):
        raise AssertionError("Long full batch contains a forbidden model family")

    for online_entries in (online_short_entries, online_long_entries):
        scope = online_entries[0]["scope"]
        models = [entry for entry in online_entries if entry["kind"] == "online"]
        routeb = _routeb_entries(models, ROUTE_B_ONLINE_SCRIPT)
        if len(routeb) != 30:
            raise AssertionError(f"{scope} Route B online must contain exactly 30 jobs")
        capacities = {
            (int(_argument(entry, "--mt")), int(_argument(entry, "--ms"))) for entry in routeb
        }
        if capacities != {(128, 128), (128, 64), (64, 64)}:
            raise AssertionError(f"{scope} Route B online capacities are incomplete")
        if {entry["argv"][entry["argv"].index("--representation") + 1] for entry in routeb} != {
            "analytic_hippo_rff",
            "inducing_points",
        }:
            raise AssertionError(f"{scope} Route B online representations are incomplete")
        if any("--include-conditional-residual-variance" not in entry["argv"] for entry in routeb):
            raise AssertionError(f"{scope} Route B online variance protocol is implicit")
        if any(entry.get("variance_protocol") != "full_joint_conditional" for entry in routeb):
            raise AssertionError(f"{scope} Route B online jobs are not full-joint-conditional")
        if {entry["seed"] for entry in routeb} != {0, 1, 2, 3, 4}:
            raise AssertionError(f"{scope} Route B online seeds are incomplete")
        if scope == "task1_2":
            postprocess = [entry for entry in online_entries if entry["kind"] == "postprocess"]
            if len(postprocess) != len(models):
                raise AssertionError("Task2 postprocessing does not cover every online prediction job")
            if any(entry["argv"][1] != POSTPROCESS_SCRIPT for entry in postprocess):
                raise AssertionError("Task2 postprocessing has an unexpected command")

    probes = [entry for entry in efficiency_entries if entry["kind"] == "efficiency_probe"]
    profiles = [entry for entry in efficiency_entries if entry["kind"] == "efficiency_profile"]
    records = [entry for entry in efficiency_entries if entry["kind"] == "efficiency_record"]
    summaries = [entry for entry in efficiency_entries if entry["kind"] == "efficiency_summary"]
    if len(probes) < 6 or len(profiles) < 40 or not records or len(summaries) != 1:
        raise AssertionError("Efficiency manifest needs probes, common-counter profiles, CPU references, and one summary")
    if {entry["seed"] for entry in probes} != {0}:
        raise AssertionError("Efficiency probes must use seed0")
    required_metadata = {"objective", "unit", "warmup", "repeats", "execution"}
    if any(not required_metadata.issubset(entry) for entry in probes):
        raise AssertionError("Efficiency probes are missing timing metadata")
    if any(entry["warmup"] != 10 or entry["repeats"] < 30 for entry in probes):
        raise AssertionError("Efficiency probes must declare warmup=10 and repeats>=30")
    if any(
        entry["execution"].get("serial") is not True
        or entry["execution"].get("gpu_count") != 1
        for entry in efficiency_entries
    ):
        raise AssertionError("Every efficiency entry must declare single-card serial execution")
    if any(entry["branch"] not in {"batch", "online"} for entry in profiles):
        raise AssertionError("Common-counter profile entries need an unambiguous batch or online branch")
    if any(entry.get("ncu", {}).get("enabled") is not True for entry in profiles):
        raise AssertionError("Common-counter profiles must enable Nsight Compute")
    if any(entry["measurement_status"] != "scheduled_common_hardware_counter" for entry in profiles):
        raise AssertionError("Common-counter profiles must be marked as scheduled until Nsight artifacts exist")
    if not records or any(entry["measurement_status"] != "cpu_not_applicable" for entry in records):
        raise AssertionError("Only explicit CPU references may remain outside the GPU counter protocol")


def _validate_dependency_producers(
    *,
    short_entries: list[dict[str, Any]],
    long_preflight_entries: list[dict[str, Any]],
    long_full_entries: list[dict[str, Any]],
    online_short_entries: list[dict[str, Any]],
    online_long_entries: list[dict[str, Any]],
) -> None:
    """Check that every cross-manifest dependency is declared upstream."""
    producers: set[str] = set()
    for entry in short_entries:
        if entry["method"] == "export_shared_protocol":
            producers.update(entry["expected"])
            continue
        missing = set(entry["dependencies"]) - producers
        missing = {dependency for dependency in missing if not Path(dependency).exists()}
        if missing:
            raise AssertionError(f"Shared-batch dependency has no producer: {sorted(missing)}")
        producers.update(entry["expected"])
    for entry in [*long_preflight_entries, *long_full_entries]:
        missing = set(entry["dependencies"]) - producers
        if missing:
            raise AssertionError(f"Long dependency has no producer in shared batch: {sorted(missing)}")
    for online_entries in (online_short_entries, online_long_entries):
        models = [entry for entry in online_entries if entry["kind"] == "online"]
        model_artifacts = {artifact for entry in models for artifact in entry["expected"]}
        for entry in models:
            missing = set(entry["dependencies"]) - producers
            if missing:
                raise AssertionError(f"Online dependency has no producer in shared batch: {sorted(missing)}")
        for entry in online_entries:
            if entry["kind"] != "postprocess":
                continue
            missing = set(entry["dependencies"]) - model_artifacts
            if missing:
                raise AssertionError(f"Task2 postprocess dependency has no online producer: {sorted(missing)}")


def _entries_for_jobs(
    jobs: Iterable[Job], *, manifest_id: str, kind: str, branch: str, spec: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        job_entry(
            job,
            manifest_id=manifest_id,
            kind=kind,
            branch=branch,
            metadata=_entry_metadata(job, spec, kind),
        )
        for job in jobs
    ]


def build_manifests(
    *,
    spec_path: Path,
    benchmark_root: Path | None,
    output_dir: Path | None = None,
    gpflow_tier: str | None = None,
) -> dict[str, Path]:
    spec_path = spec_path.resolve()
    spec = load_spec(spec_path)
    if gpflow_tier is not None:
        spec = json.loads(json.dumps(spec))
        spec["short_batch"]["gpflow_feasibility"]["selected_tier"] = str(gpflow_tier)
        load_spec_payload = spec
        _ = load_spec_payload
    base_config = load_base_config(spec_path, spec)
    if benchmark_root is None:
        benchmark_root = resolve_path(base_config["paths"]["benchmark_root"])
    else:
        benchmark_root = resolve_path(benchmark_root)
    pythons = python_paths(base_config)
    manifest_id = str(spec["manifest_id"])

    short_jobs = build_short_batch_jobs(
        spec=spec,
        base_config=base_config,
        benchmark=benchmark_root,
        pythons=pythons,
        gpflow_tier=gpflow_tier,
    )
    long_preflight_jobs = build_long_preflight_jobs(
        spec=spec,
        base_config=base_config,
        benchmark=benchmark_root,
        python=pythons["stvgp"],
    )
    long_full_jobs = build_long_full_jobs(
        spec=spec,
        base_config=base_config,
        benchmark=benchmark_root,
        python=pythons["stvgp"],
    )
    online_jobs = build_online_jobs(
        spec=spec, base_config=base_config, benchmark=benchmark_root, pythons=pythons
    )
    online_short_jobs = [(job, kind) for job, kind in online_jobs if job.scope == "task1_2"]
    online_long_jobs = [(job, kind) for job, kind in online_jobs if job.scope == "task1_10"]
    efficiency_jobs = build_efficiency_jobs(
        spec=spec,
        base_config=base_config,
        benchmark=benchmark_root,
        pythons=pythons,
    )

    short_entries = []
    for job in short_jobs:
        if job.method.startswith("preflight_gpflow_"):
            kind = "gpflow_feasibility_preflight"
        elif job.method.startswith("routeb_joint_") and job.scope == "calibration":
            kind = "calibration"
        elif job.method in {"export_shared_protocol", "export_official_stvgp_protocol"}:
            kind = "protocol"
        else:
            kind = "batch"
        short_entries.append(
            job_entry(
                job,
                manifest_id=manifest_id,
                kind=kind,
                branch="batch",
                metadata=_entry_metadata(job, spec, kind),
            )
        )
    long_preflight_entries = _entries_for_jobs(
        long_preflight_jobs,
        manifest_id=manifest_id,
        kind="official_preflight",
        branch="preflight",
        spec=spec,
    )
    long_full_entries = _entries_for_jobs(
        long_full_jobs,
        manifest_id=manifest_id,
        kind="official_full",
        branch="batch",
        spec=spec,
    )
    online_short_entries = [
        job_entry(
            job,
            manifest_id=manifest_id,
            kind=kind,
            branch=kind,
            metadata=_entry_metadata(job, spec, kind),
        )
        for job, kind in online_short_jobs
    ]
    online_long_entries = [
        job_entry(
            job,
            manifest_id=manifest_id,
            kind=kind,
            branch=kind,
            metadata=_entry_metadata(job, spec, kind),
        )
        for job, kind in online_long_jobs
    ]
    efficiency_entries = [
        job_entry(
            job,
            manifest_id=manifest_id,
            kind=kind,
            branch=str(metadata.get("manifest_branch", "efficiency")),
            metadata=metadata,
        )
        for job, metadata, kind in efficiency_jobs
    ]
    _validate_entries(
        short_entries=short_entries,
        long_preflight_entries=long_preflight_entries,
        long_full_entries=long_full_entries,
        online_short_entries=online_short_entries,
        online_long_entries=online_long_entries,
        efficiency_entries=efficiency_entries,
    )
    _validate_dependency_producers(
        short_entries=short_entries,
        long_preflight_entries=long_preflight_entries,
        long_full_entries=long_full_entries,
        online_short_entries=online_short_entries,
        online_long_entries=online_long_entries,
    )

    output_dir = (
        benchmark_root / str(spec.get("manifest_dir", "manifests"))
        if output_dir is None
        else resolve_path(output_dir)
    ).resolve()
    outputs = {
        "shared_batch_short": output_dir / "shared_batch_short.jsonl",
        "official_long_preflight": output_dir / "official_long_preflight.jsonl",
        "official_long_full": output_dir / "official_long_full.jsonl",
        "online_short": output_dir / "online_short.jsonl",
        "online_long": output_dir / "online_long.jsonl",
        "efficiency": output_dir / "efficiency.jsonl",
        "combined": output_dir / f"{manifest_id}.jsonl",
    }
    manifest_groups = {
        "shared_batch_short": short_entries,
        "official_long_preflight": long_preflight_entries,
        "official_long_full": long_full_entries,
        "online_short": online_short_entries,
        "online_long": online_long_entries,
        "efficiency": efficiency_entries,
    }
    for name, entries in manifest_groups.items():
        write_jsonl(outputs[name], entries)
    write_jsonl(
        outputs["combined"],
        [
            *short_entries,
            *long_preflight_entries,
            *long_full_entries,
            *online_short_entries,
            *online_long_entries,
            *efficiency_entries,
        ],
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "cloud/autodl_era5/benchmark_a100_shared_online_v2.json",
    )
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Manifest directory; defaults to ${BENCHMARK_ROOT}/manifests.",
    )
    parser.add_argument("--gpflow-tier", choices=("8192", "4096"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_manifests(
        spec_path=args.config,
        benchmark_root=args.benchmark_root,
        output_dir=args.output_dir,
        gpflow_tier=args.gpflow_tier,
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
