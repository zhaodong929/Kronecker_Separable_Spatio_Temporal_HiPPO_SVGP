#!/usr/bin/env python3
"""Generate tables and an audit report for the ERA5 A100 plan.

The report is evidence-driven.  It discovers result artifacts below a
benchmark root, audits predictions when they exist, and keeps the all-seen
batch branch out of strict-online tables and rankings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

try:  # Running as ``python scripts/foo.py``.
    from audit_era5_a100_shared_online import audit_benchmark_root, nested
    from summarize_era5_a100_efficiency import summarize_efficiency
except ImportError:  # Running as ``import scripts.foo``.
    from scripts.audit_era5_a100_shared_online import audit_benchmark_root, nested
    from scripts.summarize_era5_a100_efficiency import summarize_efficiency


SEED_SETS = (("0-4", tuple(range(5))), ("0-2", tuple(range(3))))
METRICS = (
    "rmse",
    "nll",
    "coverage90",
    "final_rmse",
    "runtime_seconds",
    "prediction_seconds",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "peak_nvidia_mib",
    "state_mib",
)
SEGMENT_NAME_RE = re.compile(r"(?:10[_-]?segment|segment[_-]?10|10segments?)", re.IGNORECASE)
TASK_NUMBERS = tuple(range(2, 11))
BLOCKS_PER_SHORT_STREAM = 19
COVERAGE90_FORMULA = (
    "Coverage90 = (1/N) sum_i 1{y_i in [mu_i - 1.64485362695 sigma_i, "
    "mu_i + 1.64485362695 sigma_i]}"
)
COVERAGE90_NOTE = "Coverage90 is predictive-interval coverage, not ECE."
TABLE4_COLUMNS = (
    "method",
    "seed_set",
    "seed_count",
    "objective",
    "scope",
    "branch",
    "unit",
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
    "status",
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
)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _sample_sd(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if _as_float(value) is not None]
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _relative(root: Path, value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value))
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _run_rows(root: Path, audit_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Turn audit rows with a result into one per-seed metric row."""

    rows: list[dict[str, Any]] = []
    for audit_row in audit_payload.get("runs", []):
        if audit_row.get("status") == "missing" or not audit_row.get("result_path"):
            continue
        result_path = root / str(audit_row["result_path"])
        payload = _read_json(result_path)
        if payload is None:
            continue
        timing = nested(payload, "timing", "timing_result.timing", "profile_result.timing")
        resources = nested(payload, "resources", "timing_result.resources", "profile_result.resources")
        if not isinstance(timing, Mapping):
            timing = {}
        if not isinstance(resources, Mapping):
            resources = {}
        label = nested(payload, "label", "display_name", "method_label") or audit_row.get("method", "")
        row: dict[str, Any] = {
            "scope": audit_row.get("scope", "unknown"),
            "branch": audit_row.get("branch", "unknown"),
            "method": audit_row.get("method", ""),
            "label": str(label),
            "seed": audit_row.get("seed"),
            "status": audit_row.get("status", ""),
            "run_dir": audit_row.get("run_dir", ""),
            "result_path": audit_row.get("result_path", ""),
            "prediction_path": audit_row.get("prediction_path", ""),
            "issues": audit_row.get("issues", ""),
            "artifacts_complete": audit_row.get("status") == "complete",
            "rmse": audit_row.get("reported_rmse"),
            "nll": audit_row.get("reported_nll"),
            "coverage90": audit_row.get("reported_coverage90"),
            "final_rmse": _as_float(
                nested(payload, "final_block.rmse", "final.rmse", "final_metrics.rmse")
            ),
            "runtime_seconds": _as_float(
                nested(
                    payload,
                    "timing.process_total_seconds",
                    "timing.end_to_end_training_seconds",
                    "timing.training_or_stream_runtime_seconds",
                    "train_seconds",
                    "runtime_seconds",
                )
            ),
            "prediction_seconds": _as_float(
                nested(
                    payload,
                    "timing.stream_prediction_seconds",
                    "timing.prediction_seconds",
                    "prediction_seconds",
                )
            ),
            "peak_allocated_mib": _as_float(
                nested(resources, "peak_cuda_allocated_mib", "peak_allocated_mib")
            ),
            "peak_reserved_mib": _as_float(
                nested(resources, "peak_cuda_reserved_mib", "peak_reserved_mib")
            ),
            "peak_nvidia_mib": _as_float(
                nested(
                    resources,
                    "peak_nvidia_mib",
                    "nvidia_smi_peak_mib",
                    "peak_gpu_memory_mib",
                )
            ),
            "state_mib": _as_float(
                nested(resources, "persistent_state_mib", "persistent_model_state_mib", "state_mib")
            ),
        }
        # CSV-like result payloads sometimes expose timing/resources at root.
        for field, aliases in {
            "runtime_seconds": ("runtime_seconds", "process_total_seconds", "train_seconds"),
            "peak_allocated_mib": ("peak_allocated_mib", "peak_cuda_allocated_mib"),
            "peak_reserved_mib": ("peak_reserved_mib", "peak_cuda_reserved_mib"),
            "state_mib": ("state_mib", "persistent_state_mib"),
        }.items():
            if row[field] is None:
                row[field] = _as_float(nested(payload, *aliases))
        rows.append(row)
    return rows


def _is_complete_run(row: Mapping[str, Any]) -> bool:
    """Allow statistical consumers to use only audited, complete runs."""

    if str(row.get("status", "")).lower() != "complete":
        return False
    if row.get("artifacts_complete") is False:
        return False
    return all(_as_float(row.get(metric)) is not None for metric in ("rmse", "nll", "coverage90"))


def _path_parts(row: Mapping[str, Any]) -> list[str]:
    values = [row.get("run_dir"), row.get("result_path"), row.get("prediction_path")]
    parts: list[str] = []
    for value in values:
        if not value:
            continue
        parts.extend(
            part.lower().replace("-", "_")
            for part in re.split(r"[/\\\\]+", str(value))
            if part
        )
    return parts


def _strict_online_path(row: Mapping[str, Any]) -> bool:
    """Require an online path marker and reject batch/all-seen markers."""

    parts = _path_parts(row)
    has_online = any(part in {"online", "strict_online", "streaming", "stream"} for part in parts)
    has_batch = any(
        part in {"batch", "offline", "all_seen", "all_seen_batch"} or "all_seen" in part
        for part in parts
    )
    return has_online and not has_batch


def _task_label(value: Any) -> str | None:
    if value is None or value == "":
        return None
    text = str(value)
    match = re.search(r"(?:task|tsk)[^0-9]*([0-9]+)", text, re.IGNORECASE)
    if match is None and text.strip().isdigit():
        number = int(text.strip())
    elif match is not None:
        number = int(match.group(1))
    else:
        return None
    return f"Task{number}" if number in TASK_NUMBERS else None


def _metric_from_mapping(mapping: Mapping[str, Any], metric: str) -> float | None:
    aliases = {
        "rmse": ("rmse", "RMSE", "task_rmse"),
        "nll": ("nll", "NLL", "task_nll"),
        "coverage90": ("coverage90", "coverage_90", "coverage", "Coverage90"),
    }
    return _as_float(nested(mapping, *aliases[metric]))


def extract_taskwise_metrics(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract taskwise_metrics from list- or mapping-shaped result payloads."""

    candidate = nested(
        payload,
        "taskwise_metrics",
        "per_task_metrics",
        "task_metrics",
        "metrics_per_task",
    )
    rows: dict[str, dict[str, Any]] = {}
    if isinstance(candidate, Mapping):
        items = candidate.items()
    elif isinstance(candidate, list):
        items = enumerate(candidate)
    else:
        return rows
    for key, value in items:
        if not isinstance(value, Mapping):
            continue
        task = _task_label(nested(value, "task", "task_id", "task_index") or key)
        if task is None:
            continue
        rows[task] = {
            "task": task,
            "rmse": _metric_from_mapping(value, "rmse"),
            "nll": _metric_from_mapping(value, "nll"),
            "coverage90": _metric_from_mapping(value, "coverage90"),
            "source": "result.json:taskwise_metrics",
        }
    return rows


def _read_blocks(root: Path, row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    run_dir = root / str(row.get("run_dir", "")) if row.get("run_dir") else None
    result_path = root / str(row.get("result_path", "")) if row.get("result_path") else None
    candidates: list[Path] = []
    if run_dir is not None and run_dir.is_dir():
        candidates.extend((run_dir / "blocks.csv", run_dir / "timing" / "blocks.csv", run_dir / "profile" / "blocks.csv"))
        candidates.extend(run_dir.rglob("blocks.csv"))
    if result_path is not None:
        candidates.append(result_path.parent / "blocks.csv")
    candidates = sorted({path for path in candidates if path.is_file()}, key=lambda path: (len(path.parts), str(path)))
    if not candidates:
        return [], None
    path = candidates[0]
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(item) for item in csv.DictReader(handle)], str(path.relative_to(root))
    except (OSError, UnicodeError, csv.Error, ValueError):
        return [], str(path.relative_to(root))


def _block_value(row: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    for alias in aliases:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    return None


def _normalise_block_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        item = dict(raw)
        item["block_index"] = _block_value(item, ("block_id", "block_index", "block", "update_block"))
        if item["block_index"] in (None, ""):
            item["block_index"] = index
        item["task"] = _task_label(_block_value(item, ("task", "task_id", "task_index", "task_name")))
        for metric in ("rmse", "nll", "coverage90"):
            item[metric] = _metric_from_mapping(item, metric)
        output.append(item)
    return output


def _metric_row(base: Mapping[str, Any], task: str | None, metrics: Mapping[str, Any], source: str, failure: str = "") -> dict[str, Any]:
    return {
        "scope": base.get("scope", ""),
        "branch": base.get("branch", ""),
        "method": base.get("method", ""),
        "label": base.get("label", ""),
        "seed": base.get("seed", ""),
        "task": task or "NA",
        "rmse": metrics.get("rmse"),
        "nll": metrics.get("nll"),
        "coverage90": metrics.get("coverage90"),
        "status": "failure" if failure else base.get("status", ""),
        "source_path": source,
        "failure": failure,
        "result_path": base.get("result_path", ""),
    }


def _failure_row(base: Mapping[str, Any], failure: str, source: str = "") -> dict[str, Any]:
    return _metric_row(base, None, {}, source, failure)


def _taskwise_table_rows(rows: Iterable[Mapping[str, Any]], root: Path, *, require_result_taskwise: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for base in rows:
        payload = _read_json(root / str(base["result_path"])) if base.get("result_path") else None
        taskwise = extract_taskwise_metrics(payload or {})
        blocks, block_source = _read_blocks(root, base)
        block_rows = _normalise_block_rows(blocks)
        by_task: dict[str, dict[str, Any]] = {}
        if not require_result_taskwise:
            for item in block_rows:
                if item.get("task"):
                    by_task[item["task"]] = item
        for task_number in TASK_NUMBERS:
            task = f"Task{task_number}"
            metrics = by_task.get(task) or taskwise.get(task)
            if metrics is None:
                failure = "missing_taskwise_metrics" if require_result_taskwise else "missing_per_task_metrics"
                row = _metric_row(base, task, {}, block_source or base.get("result_path", ""), failure)
                output.append(row)
                failures.append(dict(row))
            else:
                source = block_source if by_task.get(task) is metrics else "result.json:taskwise_metrics"
                item = _metric_row(base, task, metrics, source)
                if any(item[metric] is None for metric in ("rmse", "nll", "coverage90")):
                    item["status"] = "failure"
                    item["failure"] = "missing_per_task_metric"
                    failures.append(dict(item))
                output.append(item)
    return output, failures


def _block_table_rows(rows: Iterable[Mapping[str, Any]], root: Path, *, expected_blocks: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for base in rows:
        blocks, source = _read_blocks(root, base)
        if not source:
            failure = "missing_blocks.csv"
            output.append(_failure_row(base, failure))
            failures.append(_failure_row(base, failure))
            continue
        normalised = _normalise_block_rows(blocks)
        for item in normalised:
            row = dict(base)
            row.update(item)
            row["source_path"] = source
            row["failure"] = "" if all(item.get(metric) is not None for metric in ("rmse", "nll", "coverage90")) else "missing_block_metric"
            row["status"] = base.get("status", "") if not row["failure"] else "failure"
            output.append(row)
        if expected_blocks is not None and len(normalised) != expected_blocks:
            failure = f"block_count_mismatch:expected={expected_blocks},observed={len(normalised)}"
            failures.append(_failure_row(base, failure, source))
    return output, failures


def _final_task_rows(rows: Iterable[Mapping[str, Any]], root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for base in rows:
        payload = _read_json(root / str(base["result_path"])) if base.get("result_path") else None
        candidate = nested(payload or {}, "final_task", "final_task_metrics", "final_task_result")
        task = _task_label(nested(candidate, "task", "task_id") if isinstance(candidate, Mapping) else None)
        metrics = candidate if isinstance(candidate, Mapping) else None
        source = "result.json:final_task"
        if metrics is None or task is None:
            final_block = nested(payload or {}, "final_block", "final")
            if isinstance(final_block, Mapping):
                task = task or _task_label(nested(final_block, "task", "task_id"))
                if task is not None:
                    metrics = final_block
                    source = "result.json:final_block"
        if metrics is None or task is None:
            blocks, block_source = _read_blocks(root, base)
            normalised = _normalise_block_rows(blocks)
            if normalised:
                last = normalised[-1]
                task = task or last.get("task")
                if task is not None:
                    metrics = last
                    source = block_source or "blocks.csv"
        if metrics is None or task is None:
            failure = "missing_final_task_metrics" if metrics is None else "missing_final_task_identity"
            output.append(_metric_row(base, task, {}, source, failure))
            failures.append(_failure_row(base, failure, source))
            continue
        item = _metric_row(base, task, metrics, source)
        if any(item[metric] is None for metric in ("rmse", "nll", "coverage90")):
            item["status"] = "failure"
            item["failure"] = "missing_final_task_metric"
            failures.append(_failure_row(base, "missing_final_task_metric", source))
        output.append(item)
    return output, failures


def aggregate_metric_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_keys: tuple[str, ...] = ("scope", "branch", "method", "label"),
    seed_sets: Iterable[tuple[str, Iterable[int]]] = SEED_SETS,
) -> list[dict[str, Any]]:
    """Aggregate metrics by method using a mean and sample SD per seed set."""

    values = list(rows)
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in values:
        if not _is_complete_run(row):
            continue
        key = tuple(row.get(field, "") for field in group_keys)
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for seed_label, expected in seed_sets:
        expected_set = set(expected)
        for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
            selected = [row for row in group if row.get("seed") in expected_set]
            output_row: dict[str, Any] = dict(zip(group_keys, key))
            output_row.update(
                {
                    "seed_set": seed_label,
                    "seed_count": len({row.get("seed") for row in selected}),
                    "available_seeds": ",".join(
                        str(seed) for seed in sorted({row.get("seed") for row in selected if row.get("seed") is not None})
                    ),
                    "statuses": ";".join(sorted({str(row.get("status", "")) for row in selected})),
                    "source_paths": ";".join(sorted({str(row.get("result_path", "")) for row in selected})),
                }
            )
            for metric in METRICS:
                metric_values = [
                    float(row[metric])
                    for row in selected
                    if row.get(metric) is not None and _as_float(row[metric]) is not None
                ]
                output_row[f"{metric}_mean"] = float(np.mean(metric_values)) if metric_values else None
                output_row[f"{metric}_sd"] = _sample_sd(metric_values)
            output.append(output_row)
    return output


def _is_official_only(method: str, label: str = "") -> bool:
    text = f"{method} {label}".lower().replace("-", "_")
    if "routeb" in text or "route b" in text:
        return False
    return any(
        token in text
        for token in ("official", "gpflow", "bui", "maddox", "markovflow", "ohsvgp", "st_svgp")
    )


def _is_shared_xlag(method: str, label: str = "") -> bool:
    text = f"{method} {label}".lower().replace("-", "_")
    return any(token in text for token in ("xlag", "x_lag", "shared_residual", "shared_x"))


def _select_table(rows: Iterable[Mapping[str, Any]], table: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not _is_complete_run(row):
            continue
        scope = str(row.get("scope", ""))
        branch = str(row.get("branch", ""))
        method = str(row.get("method", ""))
        label = str(row.get("label", ""))
        if table == "table2a_short_shared_xlag":
            keep = scope == "short" and branch == "batch" and _is_shared_xlag(method, label)
        elif table == "table2a_l_official_only_long":
            keep = scope == "long" and branch == "batch" and _is_official_only(method, label)
        elif table == "table3a_short_online":
            keep = scope == "short" and branch == "online"
        elif table == "table3b_long_online":
            keep = scope == "long" and branch == "online"
        else:
            raise ValueError(f"unknown table: {table}")
        if keep and table in {"table3a_short_online", "table3b_long_online"}:
            keep = str(row.get("branch", "")) == "online" and _strict_online_path(row)
        if keep:
            selected.append(dict(row))
    # This invariant makes a future classifier regression visible immediately.
    if table in {"table3a_short_online", "table3b_long_online"}:
        assert not any(row.get("branch") == "batch" for row in selected)
    return selected


def _segment_csv(path: Path, root: Path) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error):
        return []
    source = str(path.relative_to(root))

    def value(row: Mapping[str, Any], aliases: Iterable[str]) -> Any:
        lowered = {str(key).lower(): item for key, item in row.items()}
        for alias in aliases:
            item = lowered.get(alias.lower())
            if item not in (None, ""):
                return item
        return None

    def failure(reason: str, scope: Any = None, branch: Any = None) -> list[dict[str, Any]]:
        return [
            {
                "scope": scope if scope not in (None, "") else "NA",
                "branch": branch if branch not in (None, "") else "NA",
                "segment": "NA",
                "segment_source": source,
                "segment_row": "",
                "total_hours": None,
                "status": "failure",
                "failure": reason,
            }
        ]

    if len(rows) != 10:
        return failure(f"segment_count_invalid:expected=10,observed={len(rows)}")
    segment_values = [value(row, ("segment", "segment_id", "segment_index")) for row in rows]
    if any(item in (None, "") for item in segment_values) or len(set(map(str, segment_values))) != 10:
        return failure("segment_ids_invalid")
    scopes = {str(value(row, ("scope", "task_scope", "dataset_scope"))).lower().replace("-", "_") for row in rows}
    branches = {str(value(row, ("branch", "evaluation_branch", "protocol"))).lower().replace("-", "_") for row in rows}
    allowed_scopes = {"short", "task1_2", "task1_2/short", "short/task1_2"}
    if not scopes or not scopes.issubset(allowed_scopes):
        return failure(f"segment_scope_invalid:observed={','.join(sorted(scopes))}")
    if branches != {"online"}:
        return failure(f"segment_branch_invalid:observed={','.join(sorted(branches))}")

    total_aliases = ("total_hours", "total_stream_hours", "stream_total_hours", "total_wallclock_hours")
    duration_aliases = ("hours", "segment_hours", "duration_hours", "elapsed_hours", "wallclock_hours")
    total_values = [_as_float(value(row, total_aliases)) for row in rows]
    duration_values = [_as_float(value(row, duration_aliases)) for row in rows]
    candidates: list[float] = []
    if all(item is not None for item in total_values):
        candidates.extend((float(total_values[0]), float(sum(total_values))))
    if all(item is not None for item in duration_values):
        candidates.append(float(sum(duration_values)))
    total_hours = next((candidate for candidate in candidates if math.isclose(candidate, 186.0, rel_tol=1e-9, abs_tol=1e-9)), None)
    if total_hours is None:
        observed = candidates[0] if candidates else None
        return failure(f"segment_total_hours_invalid:observed={observed}", next(iter(scopes)), next(iter(branches)))

    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        payload = dict(row)
        payload["segment_source"] = source
        payload["segment_row"] = index
        payload["total_hours"] = 186.0
        payload["status"] = "complete"
        payload["failure"] = ""
        output.append(payload)
    return output


def find_precomputed_10_segment_csvs(root: str | Path) -> list[dict[str, Any]]:
    """Read explicitly named or header-identified ten-segment CSVs."""

    root = Path(root).expanduser().resolve()
    output: list[dict[str, Any]] = []
    for path in root.rglob("*.csv"):
        name_match = SEGMENT_NAME_RE.search(path.stem)
        rows: list[dict[str, Any]] | None = None
        if name_match:
            rows = _segment_csv(path, root)
        else:
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    candidate = list(reader)
            except (OSError, UnicodeError, csv.Error):
                continue
            fields = {str(field).lower() for field in (reader.fieldnames or [])}
            if len(candidate) == 10 and any("segment" in field for field in fields):
                rows = _segment_csv(path, root)
        if rows:
            output.extend(rows)
    return output


def _protocol_family(method: str, label: str) -> tuple[str, str] | None:
    text = (method if not label or label == method else f"{method} {label}").lower().replace("-", "_")
    if "routeb" not in text and "route b" not in text:
        return None
    if "joint" in text:
        family = "joint"
    elif any(token in text for token in ("two_stage", "twostage", "shared_residual", "shared_xlag", "residual", "sequential")):
        family = "two_stage"
    else:
        return None
    base = text
    for token in (
        "structured_joint",
        "joint_xlag",
        "two_stage",
        "twostage",
        "shared_residual",
        "shared_xlag",
        "residual",
        "sequential",
        "xlag",
        "x_lag",
        "joint",
    ):
        base = base.replace(token, "")
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return family, base


def deterministic_bootstrap_ci(values: Iterable[float], *, seed: int = 20260807, replicates: int = 10000) -> tuple[float | None, float | None]:
    values = np.asarray([float(value) for value in values if _as_float(value) is not None], dtype=float)
    if values.size == 0:
        return None, None
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(max(1, int(replicates)), values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_routeb_deltas(
    rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 20260807,
    bootstrap_replicates: int = 10000,
) -> list[dict[str, Any]]:
    """Return paired joint-minus-two-stage RouteB deltas and bootstrap CIs."""

    grouped: dict[tuple[Any, ...], dict[str, dict[int, Mapping[str, Any]]]] = {}
    for row in rows:
        if not _is_complete_run(row):
            continue
        if str(row.get("branch", "")) == "online" and not _strict_online_path(row):
            continue
        family = _protocol_family(str(row.get("method", "")), str(row.get("label", "")))
        if family is None:
            continue
        protocol, base = family
        key = (row.get("scope"), row.get("branch"), base)
        grouped.setdefault(key, {}).setdefault(protocol, {})[int(row["seed"])] = row

    output: list[dict[str, Any]] = []
    for key, protocols in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        two_stage = protocols.get("two_stage", {})
        joint = protocols.get("joint", {})
        seeds = sorted(set(two_stage) & set(joint))
        if not seeds:
            continue
        pair_rows: list[dict[str, Any]] = []
        for seed in seeds:
            two = two_stage[seed]
            one = joint[seed]
            pair: dict[str, Any] = {
                "scope": key[0],
                "branch": key[1],
                "pair_key": key[2],
                "seed": seed,
                "two_stage_method": two.get("method", ""),
                "joint_method": one.get("method", ""),
            }
            for metric in ("rmse", "nll", "coverage90"):
                left = _as_float(two.get(metric))
                right = _as_float(one.get(metric))
                pair[f"two_stage_{metric}"] = left
                pair[f"joint_{metric}"] = right
                pair[f"delta_{metric}"] = right - left if left is not None and right is not None else None
            pair_rows.append(pair)
        for pair in pair_rows:
            summary = {
                "n_pairs": len(pair_rows),
                "delta_rmse_ci_low": None,
                "delta_rmse_ci_high": None,
                "delta_nll_ci_low": None,
                "delta_nll_ci_high": None,
                "delta_coverage90_ci_low": None,
                "delta_coverage90_ci_high": None,
            }
            for metric_index, metric in enumerate(("rmse", "nll", "coverage90")):
                deltas = [item[f"delta_{metric}"] for item in pair_rows if item[f"delta_{metric}"] is not None]
                low, high = deterministic_bootstrap_ci(
                    deltas,
                    seed=bootstrap_seed + metric_index,
                    replicates=bootstrap_replicates,
                )
                summary[f"delta_{metric}_ci_low"] = low
                summary[f"delta_{metric}_ci_high"] = high
            pair["bootstrap_seed"] = bootstrap_seed
            pair.update(summary)
            output.append(pair)
    return output


def summarize_paired_routeb_deltas(
    delta_rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 20260807,
    bootstrap_replicates: int = 10000,
) -> list[dict[str, Any]]:
    """Aggregate paired deltas for the full and seeds0-2 subsets."""

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in delta_rows:
        key = (row.get("scope"), row.get("branch"), row.get("pair_key"))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for seed_label, expected in SEED_SETS:
        expected_set = set(expected)
        for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
            selected = [row for row in group if row.get("seed") in expected_set]
            summary: dict[str, Any] = {
                "scope": key[0],
                "branch": key[1],
                "pair_key": key[2],
                "seed_set": seed_label,
                "seed_count": len({row.get("seed") for row in selected}),
                "available_seeds": ",".join(str(seed) for seed in sorted({row.get("seed") for row in selected})),
            }
            for metric_index, metric in enumerate(("rmse", "nll", "coverage90")):
                values = [
                    float(row[f"delta_{metric}"])
                    for row in selected
                    if row.get(f"delta_{metric}") is not None
                ]
                summary[f"delta_{metric}_mean"] = float(np.mean(values)) if values else None
                summary[f"delta_{metric}_sd"] = _sample_sd(values)
                low, high = deterministic_bootstrap_ci(
                    values,
                    seed=bootstrap_seed + metric_index,
                    replicates=bootstrap_replicates,
                )
                summary[f"delta_{metric}_ci_low"] = low
                summary[f"delta_{metric}_ci_high"] = high
            summary["bootstrap_seed"] = bootstrap_seed
            output.append(summary)
    return output


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    names.append(key)
        fieldnames = names
    names = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(rows: list[Mapping[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_render(row.get(column)) for column in columns) + " |")
    if not rows:
        lines.append("| No observed artifacts | " + " | ".join("" for _ in columns[1:]) + " |")
    return lines


def _efficiency_table_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in summary.get("aggregates", []):
        row = dict(source)
        row["scope"] = row.get("scope", "")
        row["branch"] = row.get("branch", "")
        rows.append(row)
    return rows


def _pdf_report(path: Path, sections: list[tuple[str, list[Mapping[str, Any]], list[str]]]) -> bool:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )
    styles = getSampleStyleSheet()
    story: list[Any] = [
        Paragraph("ERA5 A100 shared-online report", styles["Title"]),
        Paragraph(COVERAGE90_FORMULA, styles["BodyText"]),
        Paragraph(COVERAGE90_NOTE, styles["BodyText"]),
        Spacer(1, 5 * mm),
    ]
    for index, (title, rows, columns) in enumerate(sections):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(title, styles["Heading2"]))
        values = [columns]
        if rows:
            values.extend([_render(row.get(column)) for column in columns] for row in rows[:250])
        else:
            values.append(["No observed artifacts"] + [""] * (len(columns) - 1))
        table = Table(values, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef4")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.extend((table, Spacer(1, 4 * mm)))
    document.build(story)
    return True


def _write_figures(
    output: Path,
    *,
    short_blocks: list[Mapping[str, Any]],
    official_long_tasks: list[Mapping[str, Any]],
    long_online_tasks: list[Mapping[str, Any]],
) -> list[str]:
    """Write small diagnostic plots from audited rows, without affecting metrics."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def grouped(rows: list[Mapping[str, Any]], x_key: str, y_key: str) -> dict[str, list[tuple[float, float]]]:
        grouped_rows: dict[str, list[tuple[float, float]]] = {}
        for row in rows:
            try:
                raw_x = row[x_key]
                if isinstance(raw_x, str):
                    match = re.search(r"[0-9]+", raw_x)
                    raw_x = match.group(0) if match else raw_x
                x = float(raw_x)
                y = float(row[y_key])
            except (KeyError, TypeError, ValueError):
                continue
            grouped_rows.setdefault(str(row.get("method", "unknown")), []).append((x, y))
        return grouped_rows

    def save_line(
        filename: str,
        title: str,
        xlabel: str,
        ylabel: str,
        rows: list[Mapping[str, Any]],
        x_key: str,
        y_key: str,
    ) -> None:
        grouped_rows = grouped(rows, x_key, y_key)
        if not grouped_rows:
            return
        figure, axis = plt.subplots(figsize=(8.5, 4.8))
        for method, values in sorted(grouped_rows.items()):
            values.sort()
            axis.plot([item[0] for item in values], [item[1] for item in values], marker="o", label=method)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7, loc="best")
        figure.tight_layout()
        path = output / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        written.append(str(path))

    save_line(
        "table3a_short_online_coverage90_by_block.png",
        "Task 2 strict-online Coverage90 by block",
        "10-hour block index",
        "Coverage90",
        short_blocks,
        "block_index",
        "coverage90",
    )
    save_line(
        "table2a_l_official_rmse_by_task.png",
        "Official long batch RMSE by task",
        "Task",
        "RMSE",
        official_long_tasks,
        "task",
        "rmse",
    )
    save_line(
        "table3b_long_online_rmse_by_task.png",
        "Long strict-online RMSE by task",
        "Task",
        "RMSE",
        long_online_tasks,
        "task",
        "rmse",
    )
    return written


def generate_report(
    benchmark_root: str | Path,
    output_dir: str | Path,
    *,
    expected_seeds: Iterable[int] = tuple(range(5)),
    bootstrap_seed: int = 20260807,
    bootstrap_replicates: int = 10000,
) -> dict[str, Any]:
    """Generate all requested artifacts and return their paths and rows."""

    root = Path(benchmark_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_benchmark_root(root, expected_seeds)
    run_rows = _run_rows(root, audit)

    table_names = (
        "table2a_short_shared_xlag",
        "table2a_l_official_only_long",
        "table3a_short_online",
        "table3b_long_online",
    )
    tables: dict[str, list[dict[str, Any]]] = {}
    raw_tables: dict[str, list[dict[str, Any]]] = {}
    for table_name in table_names:
        selected = _select_table(run_rows, table_name)
        raw_tables[table_name] = selected
        tables[table_name] = aggregate_metric_rows(selected)

    segment_rows = find_precomputed_10_segment_csvs(root)
    segment_failures = [row for row in segment_rows if row.get("status") == "failure"]
    deltas = paired_routeb_deltas(
        run_rows,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    delta_summaries = summarize_paired_routeb_deltas(
        deltas,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    official_long_per_task, official_task_failures = _taskwise_table_rows(
        raw_tables["table2a_l_official_only_long"], root, require_result_taskwise=True
    )
    short_online_per_block, short_block_failures = _block_table_rows(
        raw_tables["table3a_short_online"], root, expected_blocks=BLOCKS_PER_SHORT_STREAM
    )
    long_online_per_task, long_task_failures = _taskwise_table_rows(
        raw_tables["table3b_long_online"], root, require_result_taskwise=False
    )
    long_online_per_block, long_block_failures = _block_table_rows(
        raw_tables["table3b_long_online"], root
    )
    long_online_final_task, final_task_failures = _final_task_rows(
        raw_tables["table3b_long_online"], root
    )
    efficiency = summarize_efficiency(root, audit)
    efficiency_rows = _efficiency_table_rows(efficiency)
    figure_paths = _write_figures(
        output,
        short_blocks=short_online_per_block,
        official_long_tasks=official_long_per_task,
        long_online_tasks=long_online_per_task,
    )
    failure_rows = [row for row in audit.get("runs", []) if row.get("status") != "complete"]
    failure_rows.extend(
        official_task_failures
        + short_block_failures
        + long_task_failures
        + long_block_failures
        + final_task_failures
        + segment_failures
    )

    table_fields = [
        "scope",
        "branch",
        "method",
        "label",
        "seed_set",
        "seed_count",
        "available_seeds",
        "statuses",
    ]
    for metric in METRICS:
        table_fields.extend((f"{metric}_mean", f"{metric}_sd"))
    table_fields.append("source_paths")
    for table_name, rows in tables.items():
        _write_csv(output / f"{table_name}.csv", rows, table_fields)
    _write_csv(output / "table3a_10_segment.csv", segment_rows)
    _write_csv(output / "table3a_short_online_per_block.csv", short_online_per_block)
    _write_csv(output / "table2a_l_official_only_long_per_task.csv", official_long_per_task)
    _write_csv(output / "table3b_long_online_per_task.csv", long_online_per_task)
    _write_csv(output / "table3b_long_online_per_block.csv", long_online_per_block)
    _write_csv(output / "table3b_long_online_final_task.csv", long_online_final_task)
    _write_csv(output / "paired_routeb_deltas.csv", deltas)
    _write_csv(output / "paired_routeb_delta_summary.csv", delta_summaries)
    _write_csv(output / "table4_efficiency.csv", efficiency_rows, TABLE4_COLUMNS)
    _write_csv(output / "failure_table.csv", failure_rows)
    _write_csv(output / "audit_runs.csv", audit.get("runs", []))

    section_specs = [
        (
            "Table 2A: short shared-Xlag",
            tables["table2a_short_shared_xlag"],
            table_fields[:8] + ["rmse_mean", "rmse_sd", "nll_mean", "nll_sd", "coverage90_mean", "coverage90_sd"],
        ),
        (
            "Table 2A-L: official-only long",
            tables["table2a_l_official_only_long"],
            table_fields[:8] + ["rmse_mean", "rmse_sd", "nll_mean", "nll_sd", "coverage90_mean", "coverage90_sd"],
        ),
        (
            "Table 3A: short strict online",
            tables["table3a_short_online"],
            table_fields[:8] + ["rmse_mean", "rmse_sd", "nll_mean", "nll_sd", "coverage90_mean", "coverage90_sd", "runtime_seconds_mean", "runtime_seconds_sd"],
        ),
        (
            "Table 3A: precomputed 10-segment CSV rows",
            segment_rows,
            [
                "scope",
                "branch",
                "segment",
                "segment_row",
                "total_hours",
                "status",
                "failure",
                "segment_source",
            ],
        ),
        (
            "Table 3B: long strict online",
            tables["table3b_long_online"],
            table_fields[:8] + ["rmse_mean", "rmse_sd", "nll_mean", "nll_sd", "coverage90_mean", "coverage90_sd", "runtime_seconds_mean", "runtime_seconds_sd"],
        ),
        (
            "Table 4: efficiency",
            efficiency_rows,
            list(TABLE4_COLUMNS),
        ),
        (
            "Failure table",
            failure_rows,
            [
                "scope",
                "branch",
                "method",
                "seed",
                "status",
                "issues",
                "failure",
                "source_path",
                "segment_source",
                "result_path",
                "prediction_path",
            ],
        ),
        (
            "Table 2A-L: official-only long per task",
            official_long_per_task,
            ["scope", "branch", "method", "seed", "task", "rmse", "nll", "coverage90", "status", "failure", "source_path"],
        ),
        (
            "Table 3A: short online per block (original updates)",
            short_online_per_block,
            ["scope", "branch", "method", "seed", "block_index", "task", "rmse", "nll", "coverage90", "status", "failure", "source_path"],
        ),
        (
            "Table 3B: long online per task",
            long_online_per_task,
            ["scope", "branch", "method", "seed", "task", "rmse", "nll", "coverage90", "status", "failure", "source_path"],
        ),
        (
            "Table 3B: long online per block",
            long_online_per_block,
            ["scope", "branch", "method", "seed", "block_index", "task", "rmse", "nll", "coverage90", "status", "failure", "source_path"],
        ),
        (
            "Table 3B: long online final task",
            long_online_final_task,
            ["scope", "branch", "method", "seed", "task", "rmse", "nll", "coverage90", "status", "failure", "source_path"],
        ),
        (
            "Paired two-stage vs joint RouteB deltas (joint minus two-stage)",
            delta_summaries,
            ["scope", "branch", "pair_key", "seed_set", "seed_count", "available_seeds", "delta_rmse_mean", "delta_rmse_sd", "delta_rmse_ci_low", "delta_rmse_ci_high", "delta_nll_mean", "delta_nll_sd", "delta_nll_ci_low", "delta_nll_ci_high"],
        ),
    ]
    markdown_lines = [
        "# ERA5 A100 shared-online report",
        "",
        COVERAGE90_FORMULA,
        COVERAGE90_NOTE,
        "",
        f"Benchmark root: `{root}`",
        "",
        "Seed aggregation uses observed members of seeds 0-4 and the seeds 0-2 subset; SD is sample SD. Missing artifacts are listed below and are never replaced with numeric defaults.",
        "Strict-online tables are filtered by `branch=online`; all-seen batch rows are not used for their rankings.",
        "",
    ]
    for title, rows, columns in section_specs:
        markdown_lines.extend([f"## {title}", ""])
        markdown_lines.extend(_markdown_table(rows, columns))
        markdown_lines.append("")
    pdf_path = output / "era5_a100_shared_online_report.pdf"
    pdf_available = _pdf_report(pdf_path, section_specs)
    if not pdf_available:
        pdf_path = None
        markdown_lines.extend(["PDF: reportlab is not installed; no PDF was written.", ""])
    markdown_path = output / "era5_a100_shared_online_report.md"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    manifest = {
        "benchmark_root": str(root),
        "output_dir": str(output),
        "audit": audit,
        "table_counts": {name: len(rows) for name, rows in tables.items()},
        "segment_rows": len(segment_rows),
        "table2a_l_official_only_long_per_task_rows": len(official_long_per_task),
        "table3a_short_online_per_block_rows": len(short_online_per_block),
        "table3b_long_online_per_task_rows": len(long_online_per_task),
        "table3b_long_online_per_block_rows": len(long_online_per_block),
        "table3b_long_online_final_task_rows": len(long_online_final_task),
        "paired_delta_rows": len(deltas),
        "paired_delta_summary_rows": len(delta_summaries),
        "efficiency_raw_rows": len(efficiency.get("records", [])),
        "efficiency_aggregate_rows": len(efficiency.get("aggregates", [])),
        "pdf_written": pdf_available,
        "figures": figure_paths,
        "files": {
            "markdown": str(markdown_path),
            "pdf": str(pdf_path) if pdf_path else None,
        },
    }
    (output / "era5_a100_shared_online_report.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": output,
        "audit": audit,
        "run_rows": run_rows,
        "tables": tables,
        "raw_tables": raw_tables,
        "segments": segment_rows,
        "official_long_per_task": official_long_per_task,
        "short_online_per_block": short_online_per_block,
        "long_online_per_task": long_online_per_task,
        "long_online_per_block": long_online_per_block,
        "long_online_final_task": long_online_final_task,
        "deltas": deltas,
        "delta_summaries": delta_summaries,
        "efficiency": efficiency,
        "failure_rows": failure_rows,
        "markdown_path": markdown_path,
        "pdf_path": pdf_path,
        "figure_paths": figure_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seeds", default="0,1,2,3,4")
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.expected_seeds.split(",") if value.strip())
    result = generate_report(
        args.benchmark_root,
        args.output_dir,
        expected_seeds=seeds,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result["output_dir"]),
                "table_counts": {name: len(rows) for name, rows in result["tables"].items()},
                "failure_rows": len(result["failure_rows"]),
                "pdf_written": result["pdf_path"] is not None,
            },
            indent=2,
        )
    )
if __name__ == "__main__":
    main()
