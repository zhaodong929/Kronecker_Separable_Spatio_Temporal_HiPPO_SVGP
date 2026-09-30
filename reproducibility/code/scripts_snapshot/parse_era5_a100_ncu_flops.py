#!/usr/bin/env python3
"""Parse Nsight Compute raw CSV into one auditable ERA5 FLOP artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


METRICS = {
    "dadd": "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
    "dmul": "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
    "dfma": "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
}


def _number(value: object) -> float | None:
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan", "none"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _rows(path: Path) -> list[list[str]]:
    values: list[list[str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if row and not row[0].startswith("==PROF=="):
                values.append(row)
    return values


def metric_totals(path: Path) -> dict[str, float]:
    """Support both Nsight long-form and wide-form ``--page raw`` CSV layouts."""

    totals = {name: 0.0 for name in METRICS.values()}
    seen: set[str] = set()
    rows = _rows(path)
    for row in rows:
        names = [cell.strip() for cell in row]
        if "Metric Name" in names and "Metric Value" in names:
            name_index = names.index("Metric Name")
            value_index = names.index("Metric Value")
            for data in rows[rows.index(row) + 1 :]:
                if len(data) <= max(name_index, value_index):
                    continue
                metric = data[name_index].strip()
                value = _number(data[value_index])
                if metric in totals and value is not None:
                    totals[metric] += value
                    seen.add(metric)
            break
        matched = [index for index, cell in enumerate(names) if cell in totals]
        if matched:
            for data in rows[rows.index(row) + 1 :]:
                for index in matched:
                    if index < len(data):
                        value = _number(data[index])
                        if value is not None:
                            totals[names[index]] += value
                            seen.add(names[index])
            break
    missing = [metric for metric in totals if metric not in seen]
    if missing:
        raise ValueError(f"Nsight CSV is missing or zero for required metrics: {missing}")
    return totals


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def _minibatch_multiplier(result: Mapping[str, Any]) -> int:
    args = result.get("args", {})
    if not isinstance(args, Mapping):
        raise ValueError("Result lacks args required for minibatch normalization")
    batch_size = _number(args.get("batch_size"))
    num_time = _number(result.get("num_time"))
    num_train_space = _number(result.get("num_train_space"))
    if batch_size is None or num_time is None or num_train_space is None:
        raise ValueError("Result lacks batch_size, num_time, or num_train_space")
    if batch_size <= 0.0 or num_time <= 0.0 or num_train_space <= 0.0:
        raise ValueError("Minibatch normalization inputs must be positive")
    return int(math.ceil(num_time * num_train_space / batch_size))


def build_payload(
    *,
    csv_path: Path,
    manifest_record: Mapping[str, Any],
    result: Mapping[str, Any],
    work_unit: str,
) -> dict[str, Any]:
    totals = metric_totals(csv_path)
    executed = totals[METRICS["dadd"]] + totals[METRICS["dmul"]] + 2.0 * totals[METRICS["dfma"]]
    multiplier = _minibatch_multiplier(result) if work_unit == "one_full_data_pass" else 1
    return {
        "schema_version": 1,
        "status": "complete",
        "scope": manifest_record.get("scope"),
        "branch": manifest_record.get("branch"),
        "method": manifest_record.get("method"),
        "seed": manifest_record.get("seed"),
        "measurement_backend": "nsight_compute",
        "measurement_scope": manifest_record.get("compute_contract", {}).get("measurement_scope"),
        "work_unit": work_unit,
        "native_work_unit": manifest_record.get("compute_contract", {}).get("native_work_unit"),
        "comparison_group": manifest_record.get("compute_contract", {}).get("comparison_group"),
        "precision": manifest_record.get("precision"),
        "hardware_class": manifest_record.get("hardware_class"),
        "measured_work_units": 1,
        "normalization_multiplier": multiplier,
        "normalization_rule": (
            "ceil(num_time * num_train_space / batch_size) minibatch updates per full-data pass"
            if work_unit == "one_full_data_pass"
            else "one profiled NVTX range equals one canonical work unit"
        ),
        "ncu_metric_totals": totals,
        "flop_formula": "dadd + dmul + 2 * dfma",
        "nsight_executed_gpu_flops": executed,
        "nsight_flops_per_native_unit": executed,
        "nsight_flops_per_unit": executed * multiplier,
        "nsight_flops_total": executed * multiplier,
        "ncu_csv": str(csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--manifest-record", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--work-unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(
        csv_path=args.csv,
        manifest_record=_read_json(args.manifest_record),
        result=_read_json(args.result),
        work_unit=args.work_unit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
