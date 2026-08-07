#!/usr/bin/env python3
"""Normalize and summarize ERA5 A100 efficiency artifacts.

The output deliberately keeps analytical estimates, executed GPU counters,
CPU-only supplements, and framework-profiler counters in separate columns.
They answer different questions and must not be presented as one additive FLOP
total.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:  # Running as ``python scripts/foo.py``.
    from audit_era5_a100_shared_online import audit_benchmark_root, infer_branch, infer_method, infer_scope
except ImportError:  # Running as ``import scripts.foo``.
    from scripts.audit_era5_a100_shared_online import audit_benchmark_root, infer_branch, infer_method, infer_scope


SEED_RE = re.compile(r"^seed(?P<seed>\d+)$", re.IGNORECASE)
SEED_SETS = (("0-4", tuple(range(5))), ("0-2", tuple(range(3))))

EFFICIENCY_FIELDS = (
    "steps_or_blocks",
    "analytical_flops",
    "analytical_setup_flops",
    "analytical_block_supplement_flops",
    "analytical_flops_per_unit",
    "analytical_total_flops",
    "nsight_executed_gpu_flops",
    "nsight_flops_per_unit",
    "nsight_flops_total",
    "cpu_supplement_flops",
    "framework_profiler_flops",
    "framework_profiler_flops_per_unit",
    "framework_profiler_flops_total",
    "runtime_seconds",
    "compile_seconds",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "peak_nvidia_mib",
    "cpu_rss_mib",
    "state_mib",
    "history_replay_mib",
)

CSV_HINTS = (
    "efficien",
    "flop",
    "nsight",
    "profiler",
    "resource",
    "timing",
    "compile_microbenchmark",
)
JSON_HINTS = (
    "efficien",
    "profile",
    "result",
    "timing",
    "resource",
    "status",
)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if not value:
            return None
        suffix = value.lower()
        multiplier = 1.0
        if suffix.endswith("gflops"):
            value = value[:-6]
            multiplier = 1e9
        elif suffix.endswith("mflops"):
            value = value[:-6]
            multiplier = 1e6
        elif suffix.endswith("kflops"):
            value = value[:-6]
            multiplier = 1e3
        try:
            number = float(value)
        except ValueError:
            return None
        return number * multiplier if math.isfinite(number) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first(mapping: Mapping[str, Any], keys: Iterable[str]) -> tuple[Any, str]:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key], key
    return None, ""


def _recursive_values(value: Any, keys: set[str], prefix: str = "") -> list[tuple[Any, str]]:
    found: list[tuple[Any, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in keys:
                found.append((child, path))
            found.extend(_recursive_values(child, keys, path))
    return found


def _recursive_first(payload: Mapping[str, Any], keys: Iterable[str]) -> tuple[Any, str]:
    wanted = {key.lower() for key in keys}
    values = _recursive_values(payload, wanted)
    return values[0] if values else (None, "")


def _seed_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = SEED_RE.match(part)
        if match:
            return int(match.group("seed"))
    return None


def _unit_multiplier(key: str, target: str) -> float:
    lower = key.lower()
    if "flops" in target:
        if "gflop" in lower or lower.endswith("_gflops"):
            return 1e9
        if "mflop" in lower or lower.endswith("_mflops"):
            return 1e6
        if "kflop" in lower or lower.endswith("_kflops"):
            return 1e3
    if target.endswith("_mib"):
        if "byte" in lower:
            return 1.0 / (1024.0 * 1024.0)
        if lower.endswith("_gb") or "_gb_" in lower:
            return 1024.0
        if lower.endswith("_mb") or "_mb_" in lower:
            return 1.0
    return 1.0


def _value_from_payload(payload: Mapping[str, Any], target: str) -> tuple[float | None, str]:
    aliases: dict[str, tuple[str, ...]] = {
        "steps_or_blocks": (
            "steps_or_blocks",
            "num_blocks_profiled",
            "num_blocks",
            "optimization_steps",
            "num_steps",
            "steps",
            "iterations",
        ),
        "analytical_flops": (
            "analytical_flops",
            "analytical_forward_flops",
        ),
        "analytical_setup_flops": (
            "analytical_setup_flops",
            "analytical_setup_lower_bound_flops",
        ),
        "analytical_block_supplement_flops": (
            "analytical_block_supplement_flops",
            "analytical_block_supplement_flops_total",
            "forward_only_analytical_lower_bound_flops",
        ),
        "analytical_flops_per_unit": (
            "analytical_flops_per_unit",
            "analytical_gflops_per_unit",
            "analytical_forward_supplement_flops_per_unit",
            "analytical_forward_supplement_gflops_per_unit",
        ),
        "analytical_total_flops": (
            "analytical_total_flops",
            "analytical_flops_total",
            "analytical_gflops_total",
            "analytical_forward_lower_bound_flops",
            "analytical_forward_supplement_total_flops",
            "analytical_forward_supplement_total_gflops",
        ),
        "nsight_executed_gpu_flops": (
            "nsight_executed_gpu_flops",
            "nsight_gpu_flops",
            "executed_gpu_flops",
            "gpu_executed_flops",
            "cuda_executed_flops",
            "flop_count_sp",
            "flop_count_hp",
        ),
        "nsight_flops_per_unit": (
            "nsight_flops_per_unit",
            "nsight_executed_gpu_flops_per_unit",
            "nsight_gflops_per_unit",
        ),
        "nsight_flops_total": (
            "nsight_flops_total",
            "nsight_total_flops",
            "nsight_gflops_total",
            "nsight_executed_gpu_flops_total",
            "nsight_total_gflops",
        ),
        "cpu_supplement_flops": (
            "cpu_supplement_flops",
            "cpu_flops",
            "cpu_hippo_flops",
            "cpu_factor_flops",
            "cpu_forward_flops",
        ),
        "framework_profiler_flops": (
            "framework_profiler_flops",
            "framework_profiler_total_flops",
            "profiler_counted_flops_total",
            "profiler_counted_total_flops",
            "profiler_flops",
            "aten_flops",
        ),
        "framework_profiler_flops_per_unit": (
            "framework_profiler_flops_per_unit",
            "framework_profiler_gflops_per_unit",
            "profiler_counted_flops_per_unit",
            "profiler_counted_gflops_per_unit",
        ),
        "framework_profiler_flops_total": (
            "framework_profiler_flops_total",
            "framework_profiler_total_flops",
            "framework_profiler_gflops_total",
            "profiler_counted_flops_total",
            "profiler_counted_total_flops",
            "profiler_counted_total_gflops",
        ),
        "runtime_seconds": (
            "runtime_seconds",
            "process_total_seconds",
            "end_to_end_seconds",
            "training_or_stream_runtime_seconds",
            "train_seconds",
            "wallclock_seconds",
        ),
        "compile_seconds": (
            "compile_seconds",
            "compile_time_seconds",
            "compilation_seconds",
            "jit_compile_seconds",
        ),
        "peak_allocated_mib": (
            "peak_allocated_mib",
            "peak_cuda_allocated_mib",
            "cuda_peak_allocated_mib",
            "peak_memory_allocated_mib",
            "peak_memory_allocated_bytes",
        ),
        "peak_reserved_mib": (
            "peak_reserved_mib",
            "peak_cuda_reserved_mib",
            "cuda_peak_reserved_mib",
            "peak_memory_reserved_mib",
            "peak_memory_reserved_bytes",
        ),
        "peak_nvidia_mib": (
            "peak_nvidia_mib",
            "nvidia_smi_peak_mib",
            "nvidia_peak_mib",
            "peak_gpu_memory_mib",
            "peak_nvidia_memory_mib",
            "nvidia_memory_used_mib",
        ),
        "cpu_rss_mib": (
            "cpu_rss_mib",
            "peak_cpu_rss_mib",
            "peak_rss_mib",
            "maximum_resident_set_size_mib",
            "maximum_resident_set_size_bytes",
        ),
        "state_mib": (
            "state_mib",
            "persistent_state_mib",
            "persistent_model_state_mib",
            "model_state_mib",
            "persistent_state_bytes",
            "model_state_bytes",
        ),
        "history_replay_mib": (
            "history_replay_mib",
            "history_replay_buffer_mib",
            "history_replay_buffer_bytes",
        ),
    }
    value, source = _recursive_first(payload, aliases[target])
    number = _as_float(value)
    if number is None:
        return None, source
    return number * _unit_multiplier(source, target), source


def _identity_from_payload(path: Path, payload: Mapping[str, Any], row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row = row or {}
    seed_value, _ = _first(row, ("seed", "split_seed", "random_seed"))
    if seed_value in (None, ""):
        seed_value, _ = _recursive_first(payload, ("seed", "split_seed", "random_seed"))
    seed = _as_float(seed_value)
    scope = row.get("scope") or infer_scope(path, payload)
    branch = row.get("branch") or infer_branch(path, payload)
    method = row.get("method") or row.get("model") or infer_method(path, payload)
    configuration_parts: list[str] = []
    for key in ("objective", "mode", "variant", "rank", "unit", "implementation"):
        value = row.get(key)
        if value not in (None, ""):
            configuration_parts.append(f"{key}={value}")
    if not configuration_parts:
        seed_index = next(
            (index for index in range(len(path.parts) - 1, -1, -1) if SEED_RE.match(path.parts[index])),
            None,
        )
        if seed_index is not None:
            ignored = {
                "runs",
                "online",
                "strict_online",
                "batch",
                "offline",
                "streaming",
                "training",
                "profile",
                "timing",
                "result",
            }
            method_text = str(method).lower()
            for part in reversed(path.parts[:seed_index]):
                normalized = part.lower().replace("-", "_")
                if normalized in ignored or normalized in method_text:
                    continue
                if any(token in normalized for token in ("task1_", "task2_", "tasks2_", "short", "long")):
                    continue
                configuration_parts.append(f"path={part}")
                break
    configuration = ",".join(configuration_parts)
    return {
        "scope": str(scope),
        "branch": str(branch),
        "method": str(method),
        "seed": int(seed) if seed is not None and seed.is_integer() else (int(seed) if seed is not None else None),
        "configuration": configuration,
    }


def normalize_efficiency_record(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path = "",
    row: Mapping[str, Any] | None = None,
    source_kind: str = "json",
) -> dict[str, Any]:
    """Map one JSON/CSV record into the unified efficiency schema."""

    path = Path(source_path) if source_path else Path("efficiency.json")
    identity = _identity_from_payload(path, payload, row)
    row = row or {}
    objective = (
        row.get("objective")
        or row.get("objective_source")
        or nested_value(payload, ("objective", "objective_source", "training_objective"))
        or "NA"
    )
    branch = str(identity["branch"]).lower()
    if branch == "batch":
        unit = "one objective F+B"
    elif branch == "online":
        unit = "one block update+prediction"
    else:
        unit = str(row.get("unit") or nested_value(payload, ("unit", "flops_unit")) or "NA")
    raw_counting_method = str(
        row.get("counting_method")
        or nested_value(payload, ("counting_method", "flop_counting_method"))
        or ""
    )
    result: dict[str, Any] = {
        "row_type": "raw",
        **identity,
        "objective": str(objective),
        "unit": unit,
        "source_kind": source_kind,
        "source_path": str(source_path),
        "status": str(row.get("status") if row and row.get("status") else nested_value(payload, ("status", "state", "outcome")) or "observed"),
        "flop_notes": raw_counting_method or str(nested_value(payload, ("excluded_flops",)) or ""),
    }
    for field in EFFICIENCY_FIELDS:
        value, source = _value_from_row_or_payload(payload, field, row)
        result[field] = value
        result[f"{field}_source"] = source

    # Setup and block supplements remain separate.  They are not aliases for
    # a complete analytical total, and are never silently added together.
    if result["analytical_total_flops"] is None and result["analytical_flops"] is not None:
        result["analytical_total_flops"] = result["analytical_flops"]
        result["analytical_total_flops_source"] = result["analytical_flops_source"]
    if result["analytical_flops"] is None and result["analytical_total_flops"] is not None:
        result["analytical_flops"] = result["analytical_total_flops"]
        result["analytical_flops_source"] = result["analytical_total_flops_source"]

    if result["nsight_flops_total"] is None and result["nsight_executed_gpu_flops"] is not None:
        result["nsight_flops_total"] = result["nsight_executed_gpu_flops"]
        result["nsight_flops_total_source"] = result["nsight_executed_gpu_flops_source"]
    if result["nsight_executed_gpu_flops"] is None and result["nsight_flops_total"] is not None:
        result["nsight_executed_gpu_flops"] = result["nsight_flops_total"]
        result["nsight_executed_gpu_flops_source"] = result["nsight_flops_total_source"]

    if result["framework_profiler_flops_total"] is None and result["framework_profiler_flops"] is not None:
        result["framework_profiler_flops_total"] = result["framework_profiler_flops"]
        result["framework_profiler_flops_total_source"] = result["framework_profiler_flops_source"]
    if result["framework_profiler_flops"] is None and result["framework_profiler_flops_total"] is not None:
        result["framework_profiler_flops"] = result["framework_profiler_flops_total"]
        result["framework_profiler_flops_source"] = result["framework_profiler_flops_total_source"]

    steps = result["steps_or_blocks"]
    if steps is not None and steps > 0:
        for per_unit, total in (
            ("analytical_flops_per_unit", "analytical_total_flops"),
            ("nsight_flops_per_unit", "nsight_flops_total"),
            ("framework_profiler_flops_per_unit", "framework_profiler_flops_total"),
        ):
            if result[per_unit] is None and result[total] is not None:
                result[per_unit] = result[total] / steps
                result[f"{per_unit}_source"] = f"{total}/steps_or_blocks"

    counting_lower = raw_counting_method.lower()
    profiler_fields = (
        "framework_profiler_flops",
        "framework_profiler_flops_per_unit",
        "framework_profiler_flops_total",
        "nsight_executed_gpu_flops",
        "nsight_flops_per_unit",
        "nsight_flops_total",
    )
    analytical_fields = (
        "analytical_flops",
        "analytical_flops_per_unit",
        "analytical_total_flops",
        "analytical_setup_flops",
        "analytical_block_supplement_flops",
    )
    if any(result[field] is not None for field in profiler_fields) or "profiler" in counting_lower:
        result["counting_method"] = "profiler-counted"
    elif any(result[field] is not None for field in analytical_fields) or "analytical" in counting_lower:
        result["counting_method"] = "analytical supplement"
    elif result["cpu_supplement_flops"] is not None or any(
        token in counting_lower for token in ("lower-order", "lower order", "estimate")
    ):
        result["counting_method"] = "lower-order estimate"
    else:
        result["counting_method"] = "not instrumented"
    return result


def nested_value(payload: Mapping[str, Any], keys: Iterable[str]) -> Any:
    value, _ = _recursive_first(payload, keys)
    return value


def _value_from_row_or_payload(
    payload: Mapping[str, Any], target: str, row: Mapping[str, Any] | None
) -> tuple[float | None, str]:
    if row:
        aliases = {
            target,
            target.replace("_flops", "_gflops"),
            target.replace("_mib", "_mb"),
            target.replace("_seconds", "_s"),
        }
        value, source = _first(row, aliases)
        number = _as_float(value)
        if number is not None:
            return number * _unit_multiplier(source, target), source
    return _value_from_payload(payload, target)


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _interesting(path: Path) -> bool:
    lower = path.name.lower()
    return any(token in lower for token in CSV_HINTS + JSON_HINTS)


def _record_has_efficiency(record: Mapping[str, Any]) -> bool:
    return any(record.get(field) is not None for field in EFFICIENCY_FIELDS)


def _merge_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("scope"),
            record.get("branch"),
            record.get("method"),
            record.get("seed"),
            record.get("configuration", ""),
            record.get("objective", "NA"),
            record.get("unit", "NA"),
        )
        current = merged.setdefault(key, dict(record))
        source_paths = [value for value in str(current.get("source_path", "")).split(";") if value]
        if record.get("source_path") and record["source_path"] not in source_paths:
            source_paths.append(record["source_path"])
        current["source_path"] = ";".join(sorted(source_paths))
        for field in EFFICIENCY_FIELDS:
            if current.get(field) is None and record.get(field) is not None:
                current[field] = record[field]
                current[f"{field}_source"] = record.get(f"{field}_source", "")
        if current.get("flop_notes") in (None, ""):
            current["flop_notes"] = record.get("flop_notes", "")
        record_status = str(record.get("status", "")).lower()
        if any(token in record_status for token in ("fail", "error", "cancel", "timeout")):
            current["status"] = record_status
        if current.get("counting_method") == "not instrumented" and record.get("counting_method") != "not instrumented":
            current["counting_method"] = record.get("counting_method")
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("scope")),
            str(row.get("branch")),
            str(row.get("method")),
            row.get("seed") if row.get("seed") is not None else -1,
            str(row.get("configuration")),
            str(row.get("objective")),
        ),
    )


def collect_efficiency_records(benchmark_root: str | Path) -> list[dict[str, Any]]:
    """Read efficiency-bearing JSON/CSV artifacts under ``benchmark_root``."""

    root = Path(benchmark_root).expanduser().resolve()
    candidates: list[dict[str, Any]] = []
    if not root.is_dir():
        return candidates

    for path in root.rglob("*.json"):
        if not _interesting(path):
            continue
        payload = _read_json(path)
        if payload is None:
            continue
        record = normalize_efficiency_record(payload, source_path=path.relative_to(root), source_kind="json")
        if record.get("seed") is not None and _record_has_efficiency(record):
            candidates.append(record)

    for path in root.rglob("*.csv"):
        if not _interesting(path):
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if not row:
                        continue
                    payload = dict(row)
                    record = normalize_efficiency_record(
                        payload,
                        source_path=path.relative_to(root),
                        row=row,
                        source_kind="csv",
                    )
                    if record.get("seed") is not None and _record_has_efficiency(record):
                        candidates.append(record)
        except (OSError, UnicodeError, csv.Error):
            continue
    return _merge_records(candidates)


def _annotate_audit_status(
    root: Path,
    records: Iterable[dict[str, Any]],
    audit_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    audit_rows = list(audit_payload.get("runs", []))
    output: list[dict[str, Any]] = []
    for source in records:
        record = dict(source)
        source_paths = [Path(value) for value in str(record.get("source_path", "")).split(";") if value]
        matches: list[Mapping[str, Any]] = []
        for audit_row in audit_rows:
            same_identity = all(
                str(record.get(field, "")) == str(audit_row.get(field, ""))
                for field in ("scope", "branch", "method", "seed")
            )
            if not same_identity or not audit_row.get("run_dir"):
                continue
            run_dir = Path(str(audit_row["run_dir"]))
            if any(
                (path if not path.is_absolute() else path.relative_to(root)).is_relative_to(run_dir)
                for path in source_paths
                if not path.is_absolute() or path.is_relative_to(root)
            ):
                matches.append(audit_row)
        explicit_status = str(record.get("status", "")).lower()
        explicit_failure = any(token in explicit_status for token in ("fail", "error", "cancel", "timeout"))
        complete = not explicit_failure and any(row.get("status") == "complete" for row in matches)
        record["artifacts_complete"] = complete
        record["audit_status"] = (
            "complete"
            if complete
            else ";".join(sorted({str(row.get("status", "incomplete")) for row in matches})) or "unmatched"
        )
        record["status"] = "complete" if complete else (explicit_status or record["audit_status"])
        output.append(record)
    return output


def _complete_efficiency_record(record: Mapping[str, Any]) -> bool:
    return str(record.get("status", "")).lower() == "complete" and record.get("artifacts_complete") is not False


def sample_sd(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.std(np.asarray(values), ddof=1)) if len(values) > 1 else None


def aggregate_efficiency_records(
    records: Iterable[Mapping[str, Any]],
    seed_sets: Iterable[tuple[str, Iterable[int]]] = SEED_SETS,
) -> list[dict[str, Any]]:
    raw = list(records)
    aggregates: list[dict[str, Any]] = []
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for record in raw:
        if not _complete_efficiency_record(record):
            continue
        key = (
            record.get("scope"),
            record.get("branch"),
            record.get("method"),
            record.get("configuration", ""),
            record.get("objective", "NA"),
            record.get("unit", "NA"),
        )
        groups.setdefault(key, []).append(record)
    for seed_label, expected in seed_sets:
        expected_set = set(expected)
        for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
            selected = [item for item in group if item.get("seed") in expected_set]
            if not selected:
                continue
            row: dict[str, Any] = {
                "row_type": "aggregate",
                "scope": key[0],
                "branch": key[1],
                "method": key[2],
                "configuration": key[3],
                "objective": key[4],
                "unit": key[5],
                "seed": "",
                "seed_set": seed_label,
                "source_kind": "aggregate",
                "source_path": ";".join(sorted({str(item.get("source_path", "")) for item in group})),
                "status": "complete",
                "artifacts_complete": True,
                "flop_notes": "",
            }
            row["seed_count"] = len({item.get("seed") for item in selected})
            methods = {str(item.get("counting_method", "not instrumented")) for item in selected}
            methods.discard("not instrumented")
            row["counting_method"] = "; ".join(sorted(methods)) if methods else "not instrumented"
            for field in EFFICIENCY_FIELDS:
                values = [
                    float(item[field])
                    for item in selected
                    if item.get(field) is not None and _as_float(item[field]) is not None
                ]
                row[f"{field}_mean"] = float(np.mean(values)) if values else None
                row[f"{field}_sd"] = sample_sd(values)
            aggregates.append(row)
    return aggregates


def summarize_efficiency(
    benchmark_root: str | Path,
    audit_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(benchmark_root).expanduser().resolve()
    records = collect_efficiency_records(root)
    if audit_payload is None:
        audit_payload = audit_benchmark_root(root, expected_seeds=(), include_missing_seeds=False)
    records = _annotate_audit_status(root, records, audit_payload)
    return {
        "benchmark_root": str(root),
        "records": records,
        "aggregates": aggregate_efficiency_records(records),
    }


def efficiency_fieldnames() -> list[str]:
    fields = [
        "row_type",
        "scope",
        "branch",
        "method",
        "configuration",
        "objective",
        "unit",
        "seed",
        "seed_set",
        "seed_count",
        "source_kind",
        "source_path",
        "status",
        "artifacts_complete",
        "audit_status",
        "counting_method",
        "flop_notes",
    ]
    fields.extend(EFFICIENCY_FIELDS)
    fields.extend(f"{field}_source" for field in EFFICIENCY_FIELDS)
    for field in EFFICIENCY_FIELDS:
        fields.extend((f"{field}_mean", f"{field}_sd"))
    return fields


def write_efficiency_csv(summary: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(summary.get("records", [])) + list(summary.get("aggregates", []))
    fields = efficiency_fieldnames()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_efficiency_markdown(summary: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregates = list(summary.get("aggregates", []))
    columns = [
        "scope",
        "branch",
        "method",
        "objective",
        "unit",
        "seed_set",
        "seed_count",
        "steps_or_blocks_mean",
        "steps_or_blocks_sd",
        "analytical_flops_per_unit_mean",
        "analytical_flops_per_unit_sd",
        "analytical_total_flops_mean",
        "analytical_total_flops_sd",
        "nsight_flops_per_unit_mean",
        "nsight_flops_per_unit_sd",
        "nsight_flops_total_mean",
        "nsight_flops_total_sd",
        "cpu_supplement_flops_mean",
        "cpu_supplement_flops_sd",
        "framework_profiler_flops_per_unit_mean",
        "framework_profiler_flops_per_unit_sd",
        "framework_profiler_flops_total_mean",
        "framework_profiler_flops_total_sd",
        "counting_method",
        "runtime_seconds_mean",
        "runtime_seconds_sd",
        "compile_seconds_mean",
        "compile_seconds_sd",
        "peak_allocated_mib_mean",
        "peak_allocated_mib_sd",
        "peak_reserved_mib_mean",
        "peak_reserved_mib_sd",
        "peak_nvidia_mib_mean",
        "peak_nvidia_mib_sd",
        "cpu_rss_mib_mean",
        "cpu_rss_mib_sd",
        "state_mib_mean",
        "state_mib_sd",
    ]
    lines = [
        "# ERA5 A100 efficiency",
        "",
        "FLOP columns are separate measurements or estimates; they are not additive.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in aggregates:
        lines.append("| " + " | ".join(_render(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value).replace("|", "\\|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_efficiency(args.benchmark_root)
    write_efficiency_csv(summary, args.output)
    if args.markdown_output:
        write_efficiency_markdown(summary, args.markdown_output)
    print(json.dumps({"records": len(summary["records"]), "aggregates": len(summary["aggregates"])}, indent=2))


if __name__ == "__main__":
    main()
