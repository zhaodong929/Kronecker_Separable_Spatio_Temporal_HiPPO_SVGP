from __future__ import annotations

import csv

from scripts.parse_era5_a100_ncu_flops import build_payload, metric_totals


def test_parser_counts_fp64_sass_flops_and_normalizes_minibatches(tmp_path) -> None:
    csv_path = tmp_path / "ncu_raw.csv"
    metrics = {
        "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum": "10",
        "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum": "20",
        "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum": "30",
    }
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Metric Name", "Metric Value"])
        writer.writeheader()
        for name, value in metrics.items():
            writer.writerow({"Metric Name": name, "Metric Value": value})

    assert metric_totals(csv_path) == {name: float(value) for name, value in metrics.items()}
    payload = build_payload(
        csv_path=csv_path,
        manifest_record={
            "scope": "task1_2",
            "branch": "batch",
            "method": "gpflow",
            "seed": 0,
            "precision": "float64",
            "compute_contract": {
                "measurement_scope": "optimization_update",
                "native_work_unit": "one_minibatch_optimization_update",
                "comparison_group": "stochastic_full_data_pass",
            },
        },
        result={"args": {"batch_size": 4}, "num_time": 3, "num_train_space": 5},
        work_unit="one_full_data_pass",
    )

    assert payload["nsight_executed_gpu_flops"] == 90.0
    assert payload["nsight_flops_per_native_unit"] == 90.0
    assert payload["normalization_multiplier"] == 4
    assert payload["nsight_flops_per_unit"] == 360.0
    assert payload["nsight_flops_total"] == 360.0
