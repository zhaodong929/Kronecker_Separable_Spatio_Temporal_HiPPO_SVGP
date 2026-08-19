from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.audit_era5_a100_shared_online import audit_benchmark_root
from scripts.generate_era5_a100_shared_online_report import (
    deterministic_bootstrap_ci,
    generate_report,
)
from scripts.summarize_era5_a100_efficiency import (
    aggregate_efficiency_records,
    common_counter_flop_ratios,
    normalize_efficiency_record,
    summarize_efficiency,
)
from scripts.build_era5_a100_manifests import Job, _compute_contract, job_entry


def _write_run(
    root: Path,
    *,
    scope: str,
    branch: str,
    method: str,
    seed: int,
    rmse: float,
    status: str | None = None,
    predictions: bool = True,
    efficiency: bool = False,
    payload_branch: str | None = None,
    taskwise_metrics: list[dict[str, object]] | None = None,
    blocks: list[dict[str, object]] | None = None,
    final_task: dict[str, object] | None = None,
) -> Path:
    run = root / "runs" / scope / branch / method / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray([0.0, 1.0, 2.0])
    pred_mean = y_true - rmse
    pred_var = np.full(3, 1.0)
    result = {
        "scope": scope,
        "branch": payload_branch or branch,
        "method": method,
        "seed": seed,
        "objective": "finite_dtc",
        "overall_current_block": {
            "rmse": rmse,
            "nll": 0.5 * (np.log(2.0 * np.pi) + rmse**2),
            "coverage90": 1.0,
        },
        "final_block": {"rmse": rmse + 0.01},
        "timing": {"process_total_seconds": 2.0 + seed, "prediction_seconds": 0.25},
        "resources": {
            "peak_cuda_allocated_mib": 10.0 + seed,
            "peak_cuda_reserved_mib": 20.0 + seed,
            "peak_nvidia_mib": 30.0 + seed,
            "cpu_rss_mib": 50.0 + seed,
            "persistent_state_mib": 3.0,
        },
    }
    if taskwise_metrics is not None:
        result["taskwise_metrics"] = taskwise_metrics
    if final_task is not None:
        result["final_task"] = final_task
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if predictions:
        np.savez(run / "predictions.npz", y_true=y_true, pred_mean=pred_mean, pred_var=pred_var)
    if status is not None:
        (run / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    if blocks is not None:
        with (run / "blocks.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(blocks[0]))
            writer.writeheader()
            writer.writerows(blocks)
    if efficiency:
        (run / "online_efficiency.json").write_text(
            json.dumps(
                {
                    "scope": scope,
                    "branch": branch,
                    "method": method,
                    "seed": seed,
                    "objective": "finite_dtc",
                    "steps_or_blocks": 19,
                    "analytical_flops": 10.0,
                    "analytical_gflops_per_unit": 2.0 + seed,
                    "analytical_total_flops": (2.0 + seed) * 19e9,
                    "analytical_setup_lower_bound_flops": 100.0 + seed,
                    "analytical_block_supplement_flops": 200.0 + seed,
                    "nsight_executed_gpu_flops": 20.0 + seed,
                    "nsight_gflops_per_unit": 3.0 + seed,
                    "nsight_flops_total": (3.0 + seed) * 19e9,
                    "cpu_supplement_flops": 30.0 + seed,
                    "framework_profiler_flops": 40.0 + seed,
                    "framework_profiler_gflops_per_unit": 4.0 + seed,
                    "framework_profiler_flops_total": (4.0 + seed) * 19e9,
                    "counting_method": "profiler-counted",
                    "runtime_seconds": 2.0 + seed,
                    "compile_seconds": 0.5,
                    "peak_allocated_mib": 10.0 + seed,
                    "peak_reserved_mib": 20.0 + seed,
                    "peak_nvidia_mib": 30.0 + seed,
                    "cpu_rss_mib": 50.0 + seed,
                    "state_mib": 3.0,
                }
            ),
            encoding="utf-8",
        )
    return run


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "benchmark"
    for seed in range(3):
        official_taskwise = [
            {
                "task": f"Task{task}",
                "start": (task - 2) * 3,
                "stop": (task - 1) * 3,
                "rmse": 0.6 + seed * 0.01 + task / 1000,
                "nll": 0.8 + task / 1000,
                "coverage90": 0.9,
            }
            for task in range(2, 11)
        ]
        short_blocks = [
            {
                "block_id": block,
                "block_start": block * 3,
                "block_stop": (block + 1) * 3,
                "task": "Task2",
                "rmse": 0.3 + seed * 0.01 + block / 1000,
                "nll": 0.7 + block / 1000,
                "coverage90": 0.9,
            }
            for block in range(19)
        ]
        _write_run(
            root,
            scope="short",
            branch="batch",
            method="routeb_shared_residual_analytic_hippo_rff",
            seed=seed,
            rmse=0.4 + seed * 0.01,
        )
        _write_run(
            root,
            scope="short",
            branch="batch",
            method="routeb_joint_xlag_analytic_hippo_rff",
            seed=seed,
            rmse=0.2 + seed * 0.01,
        )
        _write_run(
            root,
            scope="short",
            branch="online",
            method="routeb_online",
            seed=seed,
            rmse=0.3 + seed * 0.01,
            efficiency=seed == 0,
            blocks=short_blocks,
        )
        _write_run(
            root,
            scope="long",
            branch="batch",
            method="official_st_svgp",
            seed=seed,
            rmse=0.6 + seed * 0.01,
            taskwise_metrics=official_taskwise if seed != 2 else None,
        )
    _write_run(
        root,
        scope="short",
        branch="batch",
        method="routeb_joint_xlag_analytic_hippo_rff",
        seed=3,
        rmse=-10.0,
        status="failed",
        predictions=False,
    )
    _write_run(
        root,
        scope="long",
        branch="batch",
        method="routeb_all_seen_batch",
        seed=0,
        rmse=0.1,
        payload_branch="online",
    )
    long_blocks = [
        {
            "block_id": task - 2,
            "block_start": (task - 2) * 3,
            "block_stop": (task - 1) * 3,
            "task": f"Task{task}",
            "rmse": 0.5 + task / 1000,
            "nll": 0.75 + task / 1000,
            "coverage90": 0.9,
        }
        for task in range(2, 11)
    ]
    _write_run(
        root,
        scope="long",
        branch="online",
        method="routeb_long_online",
        seed=0,
        rmse=0.5,
        blocks=long_blocks,
        final_task={"task": "Task10", "rmse": 0.51, "nll": 0.76, "coverage90": 0.9},
    )
    _write_run(
        root,
        scope="short",
        branch="online",
        method="routeb_online",
        seed=3,
        rmse=99.0,
        status="failed",
        predictions=False,
        efficiency=True,
    )
    _write_run(
        root,
        scope="short",
        branch="online",
        method="failed_online",
        seed=0,
        rmse=0.9,
        status="failed",
        predictions=False,
    )
    segment_dir = root / "precomputed"
    segment_dir.mkdir(parents=True)
    with (segment_dir / "online_10_segments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["segment", "scope", "branch", "method", "hours", "rmse"],
        )
        writer.writeheader()
        for segment in range(10):
            writer.writerow(
                {
                    "segment": segment,
                    "scope": "task1_2",
                    "branch": "online",
                    "method": "routeb_online",
                    "hours": 18.6,
                    "rmse": 0.3 + segment / 100,
                }
            )
    return root


def test_audit_recomputes_predictions_and_records_missing_and_failures(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    payload = audit_benchmark_root(root, expected_seeds=(0, 1, 2), include_missing_seeds=True)
    rows = payload["runs"]
    online_failed = [row for row in rows if row["method"] == "failed_online" and row["seed"] == 0]
    assert online_failed[0]["status"] == "failed"
    assert "missing_predictions.npz" in online_failed[0]["issues"]
    complete = [row for row in rows if row["method"] == "routeb_online" and row["seed"] == 0][0]
    assert complete["status"] == "complete"
    assert abs(complete["rmse_absolute_difference"]) < 1e-12
    missing = [row for row in rows if row["method"] == "routeb_long_online" and row["seed"] == 1]
    assert missing[0]["status"] == "missing"


def test_audit_reports_manifest_jobs_that_never_started(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True)
    output_dir = root / "runs/task1_10/online/never_started/seed4"
    record = {
        "scope": "task1_10",
        "branch": "online",
        "method": "never_started",
        "seed": 4,
        "output_dir": str(output_dir),
        "expected": [str(output_dir / "result.json")],
    }
    (manifest_dir / "online_long.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    payload = audit_benchmark_root(root, expected_seeds=(), include_missing_seeds=False)

    row = next(item for item in payload["runs"] if item["method"] == "never_started")
    assert row["status"] == "missing"
    assert row["issues"] == "missing_manifest_job_artifacts:online_long.jsonl"


def test_audit_keeps_preflight_failures_out_of_result_completeness(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    output_dir = root / "runs" / "task1_2" / "preflight" / "preflight_gpflow_feasibility_8192_mt128_ms64" / "seed0"
    output_dir.mkdir(parents=True)
    (output_dir / "status.json").write_text(
        json.dumps({"status": "runtime_error"}), encoding="utf-8"
    )
    manifest_dir = root / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "shared_batch_short.jsonl").write_text(
        json.dumps(
            {
                "kind": "gpflow_feasibility_preflight",
                "scope": "task1_2",
                "branch": "preflight",
                "method": "preflight_gpflow_feasibility_8192_mt128_ms64",
                "seed": 0,
                "output_dir": str(output_dir),
                "expected": [str(output_dir / "result.json")],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = audit_benchmark_root(root, expected_seeds=(), include_missing_seeds=False)

    assert payload["failed_runs"] == 0
    assert payload["missing_runs"] == 0
    assert payload["runs"] == []
    assert payload["preflight_runs"][0]["status"] == "failed"


def test_efficiency_contract_marks_uninstrumented_rows_incomparable(tmp_path: Path) -> None:
    output = tmp_path / "runs" / "task1_2" / "batch" / "gpflow" / "seed0"
    output.mkdir(parents=True)
    result = output / "result.json"
    payload = {
        "scope": "task1_2",
        "branch": "batch",
        "method": "gpflow",
        "seed": 0,
        "analytical_flops": 1.0,
    }
    result.write_text(json.dumps(payload), encoding="utf-8")
    (output / "compute_contract.json").write_text(
        json.dumps(
            {
                "contract": {
                    "schema_version": 1,
                    "baseline_family": "gpflow",
                    "data_access_unit": "stochastic_minibatch",
                    "measurement_scope": "optimization_update",
                    "work_unit": "one_minibatch_optimization_update",
                    "required_measurement_backend": "nsight_compute_executed_gpu_flops",
                    "comparison_status": "pending_common_hardware_counter",
                }
            }
        ),
        encoding="utf-8",
    )

    row = normalize_efficiency_record(payload, source_path=result)

    assert row["data_access_unit"] == "stochastic_minibatch"
    assert row["comparison_status"] == "pending_common_hardware_counter"

    row.update({"status": "complete", "artifacts_complete": True})
    aggregate = aggregate_efficiency_records([row])[0]
    assert aggregate["baseline_family"] == "gpflow"
    assert aggregate["comparison_status"] == "pending_common_hardware_counter"


def test_compute_contract_distinguishes_batch_minibatch_and_online_units() -> None:
    def job(method: str, script: str) -> Job:
        return Job(
            stage="stage4",
            scope="task1_2",
            method=method,
            seed=0,
            python=Path("python"),
            command=["python", script],
            output_dir=Path("output"),
            expected=(),
            dependencies=(),
            timeout_seconds=1,
            device_class="a100",
        )

    routeb = _compute_contract(
        job("probe_table2a_routeb_joint_analytic_hippo_mt128_ms128", "scripts/benchmark_routeb_batch_objective.py"),
        "efficiency",
    )
    gpflow = _compute_contract(
        job("gpflow_feasibility_svgp", "scripts/run_official_gpflow_svgp_era5.py"), "shared_batch_short"
    )
    online = _compute_contract(
        job("routeb_cumulative_hippo_mt128_ms128", "scripts/run_iclr_era5_routeb_strict_online.py"),
        "online_short",
    )

    assert routeb["baseline_family"] == "routeb"
    assert routeb["data_access_unit"] == "full_fit_dataset"
    assert gpflow["data_access_unit"] == "stochastic_minibatch"
    assert gpflow["work_unit"] == "one_full_data_pass"
    assert gpflow["native_work_unit"] == "one_minibatch_optimization_update"
    assert online["work_unit"] == "one_arrival_block_update_and_prediction"

    reference = job_entry(
        job("record_table2a_gpflow", "-c"),
        manifest_id="test",
        kind="efficiency_record",
        branch="efficiency",
        metadata={"compute_source_method": "gpflow_feasibility_8192_mt128_ms64"},
    )
    assert reference["compute_contract"]["baseline_family"] == "gpflow"
    assert reference["compute_contract"]["data_access_unit"] == "stochastic_minibatch"


def test_common_counter_gate_requires_metrics_and_matching_group(tmp_path: Path) -> None:
    output = tmp_path / "runs" / "task1_2" / "online" / "routeb" / "seed0"
    output.mkdir(parents=True)
    contract = {
        "schema_version": 2,
        "baseline_family": "routeb",
        "data_access_unit": "arrival_block",
        "measurement_scope": "block_update_and_prediction",
        "work_unit": "one_arrival_block_update_and_prediction",
        "native_work_unit": "one_arrival_block_update_and_prediction",
        "comparison_group": "online_arrival_block",
        "required_measurement_backend": "nsight_compute_executed_gpu_flops",
        "comparison_status": "pending_common_hardware_counter",
    }
    (output / "compute_contract.json").write_text(
        json.dumps({"contract": contract}), encoding="utf-8"
    )
    metrics = {
        "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum": 10.0,
        "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum": 20.0,
        "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum": 30.0,
    }
    payload = {
        "scope": "task1_2",
        "branch": "online",
        "method": "routeb",
        "seed": 0,
        "status": "complete",
        "measurement_backend": "nsight_compute",
        "precision": "float64",
        "hardware_class": "NVIDIA A100",
        "ncu_metric_totals": metrics,
        "nsight_executed_gpu_flops": 90.0,
        "nsight_flops_per_native_unit": 90.0,
        "nsight_flops_per_unit": 90.0,
        "nsight_flops_total": 90.0,
    }
    path = output / "ncu_flops.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    first = normalize_efficiency_record(payload, source_path=path)
    first.update({"artifacts_complete": True})
    assert first["comparison_status"] == "common_hardware_counter_complete"

    second = dict(first, method="bui", nsight_flops_per_unit=180.0, nsight_flops_total=180.0)
    second["source_path"] = "other/ncu_flops.json"
    ratios = common_counter_flop_ratios([first, second])
    assert {row["method"] for row in ratios} == {"routeb", "bui"}
    assert max(row["flop_ratio_to_group_min"] for row in ratios) == 2.0

    mismatched = dict(second, method="gpflow", comparison_group="stochastic_full_data_pass")
    ratios = common_counter_flop_ratios([first, mismatched])
    assert {row["method"] for row in ratios} == {"routeb", "gpflow"}
    assert all(row["comparison_group"] in {"online_arrival_block", "stochastic_full_data_pass"} for row in ratios)

    incomplete = dict(first)
    incomplete["comparison_status"] = "pending_common_hardware_counter"
    assert common_counter_flop_ratios([incomplete]) == []

def test_report_separates_online_from_batch_and_writes_outputs(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    output = tmp_path / "report"
    result = generate_report(root, output, expected_seeds=(0, 1, 2), bootstrap_replicates=200)

    assert all(row["branch"] == "batch" for row in result["tables"]["table2a_short_shared_xlag"])
    assert all(row["branch"] == "online" for row in result["tables"]["table3a_short_online"])
    assert all(row["branch"] == "online" for row in result["tables"]["table3b_long_online"])
    assert not any("all_seen" in row["method"] for row in result["raw_tables"]["table3b_long_online"])
    assert all(row["method"] == "official_st_svgp" for row in result["tables"]["table2a_l_official_only_long"])
    assert len(result["segments"]) == 10
    assert all(row["status"] == "complete" and float(row["total_hours"]) == 186.0 for row in result["segments"])
    assert result["deltas"]
    assert result["deltas"][0]["delta_rmse"] < 0
    assert all(row["seed"] != 3 for row in result["deltas"])

    short_seed0_blocks = [row for row in result["short_online_per_block"] if row["seed"] == 0]
    assert len(short_seed0_blocks) == 19
    assert {row["block_index"] for row in short_seed0_blocks} == {str(index) for index in range(19)}
    official_seed0 = [row for row in result["official_long_per_task"] if row["seed"] == 0]
    assert {row["task"] for row in official_seed0} == {f"Task{task}" for task in range(2, 11)}
    official_seed2 = [row for row in result["official_long_per_task"] if row["seed"] == 2]
    assert len(official_seed2) == 9
    assert all(row["failure"] == "missing_taskwise_metrics" and row["rmse"] is None for row in official_seed2)
    assert {row["task"] for row in result["long_online_per_task"]} == {f"Task{task}" for task in range(2, 11)}
    assert len(result["long_online_per_block"]) == 9
    assert result["long_online_final_task"][0]["task"] == "Task10"

    aggregate = result["tables"]["table2a_short_shared_xlag"]
    full = [row for row in aggregate if row["seed_set"] == "0-4" and row["method"].startswith("routeb_shared")][0]
    subset = [row for row in aggregate if row["seed_set"] == "0-2" and row["method"].startswith("routeb_shared")][0]
    assert full["seed_count"] == 3
    assert subset["seed_count"] == 3
    assert abs(full["rmse_mean"] - 0.41) < 1e-12
    assert abs(subset["rmse_sd"] - 0.01) < 1e-12

    for name in (
        "table2a_short_shared_xlag.csv",
        "table2a_l_official_only_long.csv",
        "table2a_l_official_only_long_per_task.csv",
        "table3a_short_online.csv",
        "table3a_10_segment.csv",
        "table3a_short_online_per_block.csv",
        "table3b_long_online.csv",
        "table3b_long_online_per_task.csv",
        "table3b_long_online_per_block.csv",
        "table3b_long_online_final_task.csv",
        "table4_efficiency.csv",
        "failure_table.csv",
        "era5_a100_shared_online_report.md",
        "era5_a100_shared_online_report.json",
    ):
        assert (output / name).is_file(), name
    markdown = (output / "era5_a100_shared_online_report.md").read_text(encoding="utf-8")
    assert "Coverage90 = (1/N) sum_i 1{y_i in [mu_i - 1.64485362695 sigma_i" in markdown
    assert "not ECE" in markdown
    assert "Table 3A: short online per block (original updates)" in markdown
    assert "Table 3B: long online final task" in markdown
    with (output / "table4_efficiency.csv").open(newline="", encoding="utf-8") as handle:
        table4_rows = list(csv.DictReader(handle))
    required_table4 = {
        "method",
        "objective",
        "scope",
        "unit",
        "steps_or_blocks_mean",
        "analytical_flops_per_unit_mean",
        "analytical_total_flops_mean",
        "counting_method",
        "runtime_seconds_mean",
        "peak_allocated_mib_mean",
        "peak_reserved_mib_mean",
        "peak_nvidia_mib_mean",
        "cpu_rss_mib_mean",
        "state_mib_mean",
        "status",
    }
    assert required_table4.issubset(table4_rows[0])
    assert {row["seed_set"] for row in table4_rows if row["method"] == "routeb_online"} == {"0-4", "0-2"}
    if result["pdf_path"] is not None:
        assert result["pdf_path"].is_file()
    assert len(result["figure_paths"]) == 3
    assert all(Path(path).is_file() for path in result["figure_paths"])


def test_efficiency_schema_keeps_flop_sources_separate(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    summary = summarize_efficiency(root)
    records = [row for row in summary["records"] if row["method"] == "routeb_online"]
    assert records
    record = records[0]
    assert record["analytical_flops"] == 10.0
    assert record["nsight_executed_gpu_flops"] == 20.0
    assert record["cpu_supplement_flops"] == 30.0
    assert record["framework_profiler_flops"] == 40.0
    assert record["analytical_flops"] != record["framework_profiler_flops"]
    assert record["objective"] == "finite_dtc"
    assert record["unit"] == "one block update+prediction"
    assert record["steps_or_blocks"] == 19.0
    assert record["analytical_flops_per_unit"] == 2e9
    assert record["analytical_total_flops"] == 38e9
    assert record["nsight_flops_per_unit"] == 3e9
    assert record["nsight_flops_total"] == 57e9
    assert record["framework_profiler_flops_per_unit"] == 4e9
    assert record["framework_profiler_flops_total"] == 76e9
    assert record["cpu_rss_mib"] == 50.0
    assert record["counting_method"] == "profiler-counted"
    assert record["status"] == "complete"
    aggregate = [
        row
        for row in summary["aggregates"]
        if row["method"] == "routeb_online" and row["seed_set"] == "0-4"
    ][0]
    assert aggregate["seed_count"] == 3
    assert aggregate["analytical_flops_mean"] == 10.0
    assert aggregate["analytical_flops_sd"] is None
    assert aggregate["analytical_flops_per_unit_mean"] == 2e9
    assert aggregate["analytical_flops_per_unit_sd"] is None
    assert aggregate["cpu_rss_mib_mean"] == 51.0
    assert aggregate["cpu_rss_mib_sd"] == 1.0
    assert aggregate["status"] == "complete"
    assert not any(row["seed_count"] == 4 for row in summary["aggregates"] if row["method"] == "routeb_online")


def test_bootstrap_ci_is_deterministic() -> None:
    first = deterministic_bootstrap_ci([1.0, 2.0, 4.0], seed=17, replicates=300)
    second = deterministic_bootstrap_ci([1.0, 2.0, 4.0], seed=17, replicates=300)
    assert first == second


def test_invalid_ten_segment_input_is_failure_na(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    invalid = root / "precomputed" / "bad_online_10_segments.csv"
    with invalid.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["segment", "scope", "branch", "hours", "rmse"])
        writer.writeheader()
        for segment in range(9):
            writer.writerow(
                {
                    "segment": segment,
                    "scope": "task1_2",
                    "branch": "batch",
                    "hours": 18.6,
                    "rmse": 123.0,
                }
            )
    result = generate_report(root, tmp_path / "report", expected_seeds=(0, 1, 2), bootstrap_replicates=50)
    failures = [row for row in result["segments"] if row.get("status") == "failure"]
    assert failures
    assert failures[0]["failure"].startswith("segment_count_invalid")
    assert failures[0].get("rmse") is None
    assert any(row.get("segment_source", "").endswith("bad_online_10_segments.csv") for row in result["failure_rows"])


def test_analytical_components_do_not_create_a_complete_total() -> None:
    record = normalize_efficiency_record(
        {
            "scope": "short",
            "branch": "online",
            "method": "component_only",
            "seed": 0,
            "steps_or_blocks": 2,
            "analytical_setup_lower_bound_flops": 7.0,
            "analytical_block_supplement_flops": 11.0,
            "analytical_gflops_per_unit": 2.0,
            "nsight_gflops_per_unit": 3.0,
            "framework_profiler_gflops_per_unit": 4.0,
        }
    )
    assert record["analytical_setup_flops"] == 7.0
    assert record["analytical_block_supplement_flops"] == 11.0
    assert record["analytical_flops"] is None
    assert record["analytical_total_flops"] is None
    assert record["analytical_flops_per_unit"] == 2e9
    assert record["nsight_flops_per_unit"] == 3e9
    assert record["framework_profiler_flops_per_unit"] == 4e9
    assert record["counting_method"] == "profiler-counted"
