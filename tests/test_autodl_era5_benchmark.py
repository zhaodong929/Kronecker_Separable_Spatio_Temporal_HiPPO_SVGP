from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cloud.autodl_era5.run_benchmark import (
    legacy_stage2_jobs,
    stage2_jobs,
    stage3_jobs,
    utility_jobs,
)
from scripts.audit_autodl_era5_benchmark import audit_prediction, metric_values
from scripts.generate_autodl_era5_stage2plus_report import aggregate_runs, collect_runs
from stvgp_kronecker.benchmark_runtime import (
    SynchronizedTimer,
    atomic_write_json,
    resolve_torch_runtime,
    sha256_file,
)
from stvgp_kronecker.temporal_analytic import (
    AnalyticTemporalBuilder,
    TemporalAnalyticConfig,
    TemporalBlockSpec,
)


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return json.loads(
        (ROOT / "cloud/autodl_era5/benchmark.json").read_text(encoding="utf-8")
    )


def dummy_pythons(tmp_path: Path) -> dict[str, Path]:
    return {
        name: tmp_path / name / "bin/python"
        for name in ("routeb", "gpflow", "maddox", "markovflow", "stvgp")
    }


def test_publication_job_matrix_counts(tmp_path: Path) -> None:
    config = load_config()
    pythons = dummy_pythons(tmp_path)
    benchmark = tmp_path / "benchmark"
    assert len(config["split_seeds"]) == 5
    assert len(stage2_jobs(config, benchmark, pythons)) == 70
    assert len(stage3_jobs(config, benchmark, pythons)) == 70
    assert len(legacy_stage2_jobs(config, benchmark, pythons)) == 60


def test_routeb_publication_jobs_use_gpu_posterior_backend(tmp_path: Path) -> None:
    config = load_config()
    pythons = dummy_pythons(tmp_path)
    benchmark = tmp_path / "benchmark"
    batch = next(
        job
        for job in stage2_jobs(config, benchmark, pythons)
        if job.method == "routeb_joint_analytic_hippo_rff" and job.scope == "task1_2"
    )
    online = next(
        job
        for job in stage3_jobs(config, benchmark, pythons)
        if job.method == "routeb_analytic_hippo_rff" and job.scope == "task1_2"
    )
    assert batch.device_class == "modern_gpu"
    assert batch.command[batch.command.index("--evaluation-backend") + 1] == "torch"
    assert online.device_class == "modern_gpu"
    assert online.command[online.command.index("--solver-backend") + 1] == "torch"
    assert online.command[online.command.index("--device") + 1] == "cuda"
    assert (
        online.command[online.command.index("--temporal-factor-device") + 1]
        == "cpu"
    )

    ordinary = next(
        job
        for job in stage3_jobs(config, benchmark, pythons)
        if job.method == "routeb_inducing_points" and job.scope == "task1_2"
    )
    assert (
        ordinary.command[ordinary.command.index("--temporal-factor-device") + 1]
        == "solver"
    )


def test_utility_jobs_use_the_selected_config(tmp_path: Path) -> None:
    config = load_config()
    selected = tmp_path / "selected.json"
    jobs = utility_jobs(config, tmp_path / "benchmark", tmp_path / "python", selected)
    assert len(jobs) == 3
    for job in jobs[1:]:
        assert str(selected) in job.command


def test_runtime_helpers_on_cpu(tmp_path: Path) -> None:
    runtime = resolve_torch_runtime("cpu", "float32")
    assert runtime.device.type == "cpu"
    assert runtime.dtype == torch.float32
    with SynchronizedTimer() as timer:
        _ = sum(range(100))
    assert timer.elapsed >= 0.0
    destination = tmp_path / "nested/result.json"
    atomic_write_json(destination, {"value": 3})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"value": 3}
    assert len(sha256_file(destination)) == 64


def test_analytic_hippo_respects_float32_move() -> None:
    builder = AnalyticTemporalBuilder(
        TemporalAnalyticConfig(
            inducing_size=8,
            rff_sample_size=16,
            variance=1.0,
            lengthscale=0.3,
            kernel_type="matern32",
            dtype=torch.float64,
        )
    ).to(dtype=torch.float32)
    horizon = TemporalBlockSpec(start=-0.1, end=1.0, num_discrete_steps=11)
    basis, prior = builder.compute_temporal_basis(horizon)
    assert builder.config.dtype == torch.float32
    assert basis.dtype == torch.float32
    assert prior.dtype == torch.float32
    assert torch.isfinite(basis).all()
    assert torch.isfinite(prior).all()


def test_audit_accepts_batch_metric_in_overall_schema(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    y = np.asarray([[0.0, 1.0], [1.5, -0.5]])
    mean = np.asarray([[0.1, 0.8], [1.4, -0.3]])
    variance = np.full_like(y, 0.2)
    metrics = metric_values(y, mean, variance)
    (run / "result.json").write_text(
        json.dumps({"overall_current_block": metrics}), encoding="utf-8"
    )
    np.savez_compressed(run / "predictions.npz", y_true=y, pred_mean=mean, pred_var=variance)
    issues, details = audit_prediction(
        run / "result.json", expected_times=2, expected_space=2, branch="batch"
    )
    assert issues == []
    assert details["recomputed"]["rmse"] == metrics["rmse"]


def test_report_collector_handles_complete_and_failed_runs(tmp_path: Path) -> None:
    complete = tmp_path / "runs/task1_2/batch/xlag_mean_only/seed0"
    complete.mkdir(parents=True)
    (complete / "status.json").write_text(
        json.dumps({"status": "complete", "device_class": "cpu"}), encoding="utf-8"
    )
    (complete / "result.json").write_text(
        json.dumps(
            {
                "overall_current_block": {
                    "rmse": 0.2,
                    "nll": -0.4,
                    "coverage90": 0.88,
                    "mean_predictive_std": 0.1,
                }
            }
        ),
        encoding="utf-8",
    )
    failed = tmp_path / "runs/task1_2/batch/official_st_svgp_ms30/seed0"
    failed.mkdir(parents=True)
    (failed / "status.json").write_text(
        json.dumps(
            {
                "status": "out_of_memory",
                "device_class": "legacy_cpu",
                "legacy": True,
            }
        ),
        encoding="utf-8",
    )
    frame = collect_runs(tmp_path)
    assert set(frame.status) == {"complete", "out_of_memory"}
    summary = aggregate_runs(frame, expected_seeds=5)
    assert len(summary) == 1
    assert summary.iloc[0].completed_seeds == 1
    assert summary.iloc[0].expected_seeds == 5
