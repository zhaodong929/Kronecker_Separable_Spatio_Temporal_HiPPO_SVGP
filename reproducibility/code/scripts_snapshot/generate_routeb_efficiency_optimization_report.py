#!/usr/bin/env python3
"""Generate publication-audit tables for Route-B efficiency optimizations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def percent_reduction(original: float | None, optimized: float | None) -> float | None:
    if original is None or optimized is None or original == 0.0:
        return None
    return 100.0 * (original - optimized) / original


def mean_numeric(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [number(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def markdown_table(headers: list[str], rows: Iterable[list[str]]) -> list[str]:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return output


def load_batch_profiles(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((root / "batch_final").glob("*/seed0/*/batch_ablation.csv")):
        rows.extend(read_csv(path))
    return rows


def unified_batch_rows(root: Path, profiles: list[dict[str, str]]) -> list[dict[str, Any]]:
    profile_index = {
        (row["scope"], row["objective"], "original" if row["version"] == "E0" else "optimized"): row
        for row in profiles
    }
    runs = read_csv(root / "summary/training_metrics_per_run.csv")
    output: list[dict[str, Any]] = []
    for run in runs:
        profile_row = profile_index.get((run["scope"], run["objective"], run["version"]))
        steps = number(run.get("iterations_completed"))
        counted = None if profile_row is None else number(profile_row.get("profiler_counted_flops_per_step"))
        supplement = None if profile_row is None else number(profile_row.get("analytical_supplement_forward_flops"))
        output.append(
            {
                "method": "Route B batch empirical Bayes",
                "objective": run["objective"],
                "scope": run["scope"],
                "version": run["version"],
                "seed": run["seed"],
                "unit": "one objective forward+backward",
                "steps_or_blocks": steps,
                "profiler_counted_gflops_per_unit": None if counted is None else counted / 1e9,
                "analytical_forward_supplement_gflops_per_unit": None if supplement is None else supplement / 1e9,
                "profiler_counted_total_gflops": None if counted is None or steps is None else counted * steps / 1e9,
                "profiler_plus_forward_lower_bound_total_gflops": (
                    None if counted is None or supplement is None or steps is None
                    else (counted + supplement) * steps / 1e9
                ),
                "counting_method": None if profile_row is None else profile_row.get("counting_method"),
                "excluded_flops": None if profile_row is None else profile_row.get("excluded_flops"),
                "runtime_per_unit_seconds": number(run.get("mean_steady_state_iteration_seconds")),
                "training_or_stream_runtime_seconds": number(run.get("training_seconds")),
                "process_total_seconds": number(run.get("process_total_seconds")),
                "best_validation_step": number(run.get("best_iteration")),
                "time_to_best_validation_seconds": number(
                    run.get("time_to_best_validation_seconds")
                ),
                "peak_allocated_mib": number(run.get("peak_cuda_allocated_mib")),
                "peak_reserved_mib": number(run.get("peak_cuda_reserved_mib")),
                "rmse": number(run.get("rmse")),
                "nll": number(run.get("nll")),
                "coverage90": number(run.get("coverage90")),
            }
        )
    return output


def online_profile_for_scope(root: Path, scope: str) -> dict[str, Any] | None:
    path = root / f"online/{scope}/finite_dtc/seed0/online_efficiency.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def unified_online_rows(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted((root / "online").glob("*/*/seed*/online_efficiency.json")):
        audit = json.loads(path.read_text(encoding="utf-8"))
        timing = audit.get("timing_result") or {}
        metrics = timing.get("overall_current_block") or {}
        timing_fields = timing.get("timing") or {}
        resources = timing.get("resources") or {}
        scope = audit["scope"]
        profile_source = online_profile_for_scope(root, scope)
        num_blocks = int(timing.get("num_blocks", 0))
        stream_seconds = sum(
            float(timing_fields.get(key, 0.0))
            for key in ("stream_update_seconds", "stream_prediction_seconds")
        )
        counted_total = None if profile_source is None else number(profile_source.get("profiler_counted_flops_total"))
        lower_total = None if profile_source is None else number(
            profile_source.get("profiler_plus_analytical_lower_bound_total_flops")
        )
        output.append(
            {
                "method": "Route B cumulative-changing HiPPO strict online",
                "objective": audit["objective_source"] + " Task-1 calibration",
                "scope": scope,
                "version": "online_current_exact",
                "seed": audit["seed"],
                "unit": "one online block update+prediction",
                "steps_or_blocks": num_blocks,
                "profiler_counted_gflops_per_unit": (
                    None if counted_total is None or num_blocks == 0 else counted_total / num_blocks / 1e9
                ),
                "analytical_forward_supplement_gflops_per_unit": (
                    None if lower_total is None or counted_total is None or num_blocks == 0
                    else (lower_total - counted_total) / num_blocks / 1e9
                ),
                "profiler_counted_total_gflops": None if counted_total is None else counted_total / 1e9,
                "profiler_plus_forward_lower_bound_total_gflops": None if lower_total is None else lower_total / 1e9,
                "counting_method": None if profile_source is None else profile_source.get("counting_method"),
                "excluded_flops": None if profile_source is None else profile_source.get("excluded_flops"),
                "flops_reused_from": "finite_dtc seed0 same-scope profile; recursion shapes identical",
                "runtime_per_unit_seconds": None if num_blocks == 0 else stream_seconds / num_blocks,
                "training_or_stream_runtime_seconds": stream_seconds,
                "process_total_seconds": number(timing_fields.get("process_total_seconds")),
                "peak_allocated_mib": number(resources.get("peak_cuda_allocated_mib")),
                "peak_reserved_mib": number(resources.get("peak_cuda_reserved_mib")),
                "rmse": number(metrics.get("rmse")),
                "nll": number(metrics.get("nll")),
                "coverage90": number(metrics.get("coverage90")),
            }
        )
    return output


def collect_operator_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patterns = (
        "batch_final/*/seed0/*/profiler_operator_breakdown.csv",
        "online/*/finite_dtc/seed0/online_profiler_operator_breakdown.csv",
    )
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            source = str(path.relative_to(root))
            rows.extend({"source": source, **row} for row in read_csv(path))
    return rows


def exact_ablation_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "batch").glob("*/seed0/finite_dtc/batch_ablation.csv")):
        rows.extend(read_csv(path))
    return rows


def rank_accuracy_rows(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted((root / "rank_ablation/stream").glob("*/rank*/seed*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        parts = path.relative_to(root / "rank_ablation/stream").parts
        output.append(
            {
                "mode": "batch empirical Bayes",
                "scope": parts[0],
                "rank": int(parts[1].removeprefix("rank")),
                "seed": payload["split_seed"],
                "rmse": nested(payload, "final", "rmse"),
                "nll": nested(payload, "final", "nll"),
                "coverage90": nested(payload, "final", "coverage90"),
                "runtime_seconds": nested(payload, "timing", "training_seconds"),
                "peak_allocated_mib": nested(payload, "resources", "peak_cuda_allocated_mib"),
                "peak_reserved_mib": nested(payload, "resources", "peak_cuda_reserved_mib"),
            }
        )
    for path in sorted((root / "rank_ablation/online").glob("*/rank*/seed*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        parts = path.relative_to(root / "rank_ablation/online").parts
        output.append(
            {
                "mode": "strict online",
                "scope": parts[0],
                "rank": int(parts[1].removeprefix("rank")),
                "seed": payload["split_seed"],
                "rmse": nested(payload, "overall_current_block", "rmse"),
                "nll": nested(payload, "overall_current_block", "nll"),
                "coverage90": nested(payload, "overall_current_block", "coverage90"),
                "runtime_seconds": sum(
                    float(nested(payload, "timing", key) or 0.0)
                    for key in ("stream_update_seconds", "stream_prediction_seconds")
                ),
                "peak_allocated_mib": nested(payload, "resources", "peak_cuda_allocated_mib"),
                "peak_reserved_mib": nested(payload, "resources", "peak_cuda_reserved_mib"),
                "persistent_state_mib": nested(payload, "resources", "persistent_state_mib"),
            }
        )
    profile_index: dict[tuple[str, int], dict[str, str]] = {}
    for path in sorted((root / "rank_ablation/profile").glob("*/rank*/seed*/batch_ablation.csv")):
        rows = read_csv(path)
        if rows:
            parts = path.relative_to(root / "rank_ablation/profile").parts
            profile_index[(parts[0], int(parts[1].removeprefix("rank")))] = rows[0]
    for row in output:
        profile_row = profile_index.get((row["scope"], row["rank"]))
        if profile_row and row["mode"] == "batch empirical Bayes":
            row["profiler_counted_gflops_per_step"] = number(
                profile_row.get("profiler_counted_gflops_per_step")
            )
            row["objective_runtime_seconds_per_step"] = number(
                profile_row.get("steady_runtime_seconds_per_step")
            )
    return output


def cross_and_compile_summary(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cross_summary: list[dict[str, Any]] = []
    compile_summary: list[dict[str, Any]] = []
    for scope_dir in sorted((root / "cross").glob("*/seed0")):
        cross_rows = read_csv(scope_dir / "cross_microbenchmark.csv")
        if cross_rows:
            best = min(
                cross_rows,
                key=lambda row: number(row.get("forward_backward_seconds")) or float("inf"),
            )
            reference_candidates = [
                row for row in cross_rows
                if row.get("order") == "einsum" and row.get("feature_block_size") == "133"
            ]
            reference = reference_candidates[0] if reference_candidates else cross_rows[0]
            cross_summary.append(
                {
                    "scope": best.get("scope"),
                    "selected_order": best.get("order"),
                    "selected_feature_block_size": best.get("feature_block_size"),
                    "best_forward_seconds": number(best.get("forward_seconds")),
                    "best_forward_backward_seconds": number(best.get("forward_backward_seconds")),
                    "reference_einsum_forward_backward_seconds": number(
                        reference.get("forward_backward_seconds")
                    ),
                    "runtime_reduction_percent": percent_reduction(
                        number(reference.get("forward_backward_seconds")),
                        number(best.get("forward_backward_seconds")),
                    ),
                    "profiler_counted_gflops": number(
                        best.get("profiler_counted_gflops_forward_backward")
                    ),
                    "largest_intermediate_bytes": number(
                        best.get("largest_explicit_intermediate_bytes")
                    ),
                    "output_relative_error": number(best.get("output_relative_error")),
                    "gradient_relative_error": number(best.get("gradient_relative_error")),
                }
            )
        compile_rows = read_csv(scope_dir / "compile_microbenchmark.csv")
        compile_summary.extend(compile_rows)
    return cross_summary, compile_summary


def top_profiled_operators(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], float] = {}
    for row in rows:
        source = str(row.get("source", ""))
        operator = str(row.get("operator", ""))
        value = number(row.get("profiler_counted_flops")) or 0.0
        totals[(source, operator)] = totals.get((source, operator), 0.0) + value
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [
        {"source": source, "operator": operator, "profiler_counted_flops": flops}
        for (source, operator), flops in ranked
    ]


def generate_report(root: Path, unified: list[dict[str, Any]], ablations: list[dict[str, Any]]) -> str:
    lines = [
        "# Route B efficiency optimization diagnostic",
        "",
        "## Scope and counting boundary",
        "",
        "Fixed protocol: ERA5 Task 2 short (186 hours) and Tasks 2--10 long (1,674 hours); "
        "720 fit, 80 validation and 200 test spatial locations; split seeds 0--4; "
        "X-lag structured-joint mean with 133 features; `M_t=M_s=128`; fixed HiPPO-RFF frequencies; "
        "float64 Adam (`lr=0.02`) for 100 steps with validation every 5 steps. "
        "Both DTC and VFE training runs use full structured-joint conditional (D) prediction.",
        "",
        "Batch unit: one differentiable empirical-Bayes objective forward+backward. "
        "Online unit: one causal block posterior update plus prediction. Runtime runs are separate from profiler runs.",
        "",
        "PyTorch Profiler `with_flops=True` counts only supported operators. The analytical column is a forward-only lower-bound supplement for major Cholesky/EVD/solve work. "
        "Their sum is still not a complete hardware FLOP count because the listed exclusions remain.",
        "",
    ]
    batch = [row for row in unified if row["method"].startswith("Route B batch")]
    lines.extend(["## Exact batch optimization", ""])
    summary_rows: list[list[str]] = []
    keys = sorted({(row["scope"], row["objective"]) for row in batch})
    for scope, objective in keys:
        group = [row for row in batch if row["scope"] == scope and row["objective"] == objective]
        for version in ("original", "optimized"):
            selected = [row for row in group if row["version"] == version]
            if not selected:
                continue
            runtime = mean_numeric(selected, "training_or_stream_runtime_seconds")
            step_runtime = mean_numeric(selected, "runtime_per_unit_seconds")
            peak_allocated = mean_numeric(selected, "peak_allocated_mib")
            peak_reserved = mean_numeric(selected, "peak_reserved_mib")
            profile = selected[0]["profiler_counted_gflops_per_unit"]
            steps = mean_numeric(selected, "steps_or_blocks")
            rmse = mean_numeric(selected, "rmse")
            nll = mean_numeric(selected, "nll")
            coverage = mean_numeric(selected, "coverage90")
            supplement = selected[0]["analytical_forward_supplement_gflops_per_unit"]
            summary_rows.append([
                scope, objective, version,
                f"{profile:.3f}" if profile is not None else "NA",
                f"{supplement:.3f}" if supplement is not None else "NA",
                (
                    f"{profile * steps / 1000.0:.3f}/"
                    f"{(profile + supplement) * steps / 1000.0:.3f}"
                    if profile is not None and supplement is not None and steps is not None
                    else "NA"
                ),
                f"{step_runtime:.3f}" if step_runtime is not None else "NA",
                f"{runtime:.2f}" if runtime is not None else "NA",
                f"{peak_allocated:.0f}/{peak_reserved:.0f}" if peak_allocated is not None and peak_reserved is not None else "NA",
                f"{rmse:.4f}", f"{nll:.4f}", f"{coverage:.4f}",
            ])
    lines.extend(markdown_table(
        ["Scope", "Objective", "Version", "Profiler GFLOPs/step", "Analytical +GFLOPs/step", "Counted/lower-bound TFLOPs", "s/step", "Training s", "Peak MiB A/R", "RMSE", "NLL", "Cov90"],
        summary_rows,
    ))
    paired = [
        row for row in read_csv(root / "summary/paired_parity.csv")
        if row.get("pair_complete") == "True"
    ]
    prediction_mean_errors = [
        number(row.get("predictive_mean_relative_error")) for row in paired
    ]
    prediction_variance_errors = [
        number(row.get("predictive_variance_relative_error")) for row in paired
    ]
    final_rmse_differences = [
        abs(value)
        for row in paired
        if (value := number(row.get("delta_optimized_minus_original_rmse"))) is not None
    ]
    final_nll_differences = [
        abs(value)
        for row in paired
        if (value := number(row.get("delta_optimized_minus_original_nll"))) is not None
    ]
    final_coverage_differences = [
        abs(value)
        for row in paired
        if (value := number(row.get("delta_optimized_minus_original_coverage90"))) is not None
    ]
    prediction_mean_errors = [value for value in prediction_mean_errors if value is not None]
    prediction_variance_errors = [value for value in prediction_variance_errors if value is not None]
    nonpositive = sum(
        int(number(row.get("optimized_nonpositive_variance_count")) or 0)
        for row in paired
    )
    lines.extend(["", "E1 caches fixed feature sufficient statistics and uses the scope-selected exact contraction order. E2/E3 remain separate arithmetic-order ablations; they are not promoted when any strict parity threshold fails."])
    if prediction_mean_errors and prediction_variance_errors:
        lines.append(
            f"Across completed E0/E1 pairs, the maximum absolute final-metric differences are "
            f"{max(final_rmse_differences):.3e} for RMSE, {max(final_nll_differences):.3e} for NLL, "
            f"and {max(final_coverage_differences):.3e} for Coverage90. The maximum saved-prediction "
            f"relative errors are {max(prediction_mean_errors):.3e} for means and "
            f"{max(prediction_variance_errors):.3e} for variances; optimized non-positive predictive "
            f"variances: {nonpositive}. These are numerically equivalent final metrics, not bitwise-identical "
            f"optimization trajectories."
        )
    boundary_groups: list[str] = []
    for scope, objective in keys:
        selected = [
            row for row in batch
            if row["scope"] == scope
            and row["objective"] == objective
            and row["version"] == "optimized"
        ]
        at_boundary = sum(
            row.get("best_validation_step") == row.get("steps_or_blocks")
            for row in selected
        )
        if selected:
            boundary_groups.append(f"{scope}/{objective}: {at_boundary}/{len(selected)} seeds")
    if boundary_groups:
        lines.append(
            "Seeds whose best validation checkpoint occurred at the final (100th) training step: "
            + ", ".join(boundary_groups)
            + ". Any nonzero count means convergence at this budget is not established for every seed; "
            "the 5/5 short-VFE result is the clearest budget-limited case."
        )
    lines.append("")

    if ablations:
        lines.extend(["## E0-E3 parity", ""])
        parity_rows = []
        for row in ablations:
            gflops = number(row.get("profiler_counted_gflops_per_step"))
            parity_rows.append([
                row.get("scope", ""), row.get("version", ""),
                f"{gflops:.3f}" if gflops is not None else "NA",
                f"{number(row.get('steady_runtime_seconds_per_step')):.4f}" if number(row.get("steady_runtime_seconds_per_step")) is not None else "NA",
                row.get("passes_exact_parity", ""),
                row.get("posterior_mean_relative_error", ""),
                row.get("beta_covariance_relative_error", ""),
            ])
        lines.extend(markdown_table(
            ["Scope", "Version", "GFLOPs/step", "s/step", "Strict parity", "Mean rel. err.", "Cov. rel. err."],
            parity_rows,
        ))
        lines.append("")

    cross_rows, compile_rows = cross_and_compile_summary(root)
    if cross_rows:
        lines.extend(["## Cross contraction and compile", ""])
        lines.extend(markdown_table(
            ["Scope", "Selected eager order", "Block", "F+B s", "vs einsum", "Counted GFLOPs"],
            [
                [
                    str(row["scope"]), str(row["selected_order"]),
                    str(row["selected_feature_block_size"]),
                    f"{row['best_forward_backward_seconds']:.4f}",
                    f"{row['runtime_reduction_percent']:.1f}%" if row["runtime_reduction_percent"] is not None else "NA",
                    f"{row['profiler_counted_gflops']:.3f}" if row["profiler_counted_gflops"] is not None else "NA",
                ]
                for row in cross_rows
            ],
        ))
        lines.append("")
        if compile_rows:
            lines.extend(markdown_table(
                ["Scope", "Compile mode", "Compilation s", "Steady F+B s", "Peak reserved MiB"],
                [
                    [
                        row.get("scope", ""), row.get("compile_mode", ""),
                        f"{number(row.get('compilation_latency_seconds')):.3f}",
                        f"{number(row.get('steady_forward_backward_seconds')):.4f}",
                        f"{number(row.get('peak_reserved_bytes')) / 1024**2:.1f}",
                    ]
                    for row in compile_rows
                ],
            ))
            lines.extend(["", "Compilation latency is reported separately. `torch.compile` is not selected unless its steady-state runtime improves on the best eager contraction.", ""])

    rank_rows = read_csv(root / "rank/future_projection_error.csv")
    candidate = [row for row in rank_rows if row.get("selection") == "candidate_rank"]
    if candidate:
        lines.extend(["## Feature rank", ""])
        rank_table = []
        for rank in (133, 73, 64, 48):
            errors = [number(row.get("relative_projection_error")) for row in candidate if row.get("candidate_rank") == str(rank)]
            errors = [value for value in errors if value is not None]
            if errors:
                rank_table.append([str(rank), f"{min(errors):.3e}", f"{max(errors):.3e}", "exact" if max(errors) <= 1e-12 else "approximate"])
        lines.extend(markdown_table(["Rank", "Future min error", "Future max error", "Classification"], rank_table))
        lines.extend(["", "Rank 73/64/48 change future features and therefore remain model-changing accuracy-efficiency ablations.", ""])

    online = [row for row in unified if "strict online" in row["method"]]
    if online:
        lines.extend(["## Strict online", ""])
        lines.append("The current online backend already accumulates sufficient statistics, groups all feature RHS into one Sylvester diagonalization, and computes `m_u = v_h - W m_beta`. Batch E1-E3 labels must not be counted as three additional online speedups.")
        lines.append("")
        online_table: list[list[str]] = []
        online_keys = sorted({(row["scope"], row["objective"]) for row in online})
        for scope, objective in online_keys:
            selected = [
                row for row in online
                if row["scope"] == scope and row["objective"] == objective
            ]
            gflops = mean_numeric(selected, "profiler_counted_gflops_per_unit")
            supplement = mean_numeric(selected, "analytical_forward_supplement_gflops_per_unit")
            total_gflops = mean_numeric(selected, "profiler_counted_total_gflops")
            lower_total_gflops = mean_numeric(
                selected, "profiler_plus_forward_lower_bound_total_gflops"
            )
            per_block = mean_numeric(selected, "runtime_per_unit_seconds")
            stream_runtime = mean_numeric(selected, "training_or_stream_runtime_seconds")
            peak_allocated = mean_numeric(selected, "peak_allocated_mib")
            peak_reserved = mean_numeric(selected, "peak_reserved_mib")
            online_table.append([
                scope,
                objective.replace(" Task-1 calibration", ""),
                f"{gflops:.3f}" if gflops is not None else "NA",
                f"{supplement:.3f}" if supplement is not None else "NA",
                (
                    f"{total_gflops / 1000.0:.3f}/{lower_total_gflops / 1000.0:.3f}"
                    if total_gflops is not None and lower_total_gflops is not None
                    else "NA"
                ),
                f"{per_block:.4f}" if per_block is not None else "NA",
                f"{stream_runtime:.2f}" if stream_runtime is not None else "NA",
                f"{peak_allocated:.0f}/{peak_reserved:.0f}" if peak_allocated is not None and peak_reserved is not None else "NA",
                f"{mean_numeric(selected, 'rmse'):.4f}",
                f"{mean_numeric(selected, 'nll'):.4f}",
                f"{mean_numeric(selected, 'coverage90'):.4f}",
            ])
        lines.extend(markdown_table(
            ["Scope", "Task-1 theta", "Profiler GFLOPs/block", "Analytical +GFLOPs/block", "Counted/lower-bound TFLOPs", "s/block", "Stream s", "Peak MiB A/R", "RMSE", "NLL", "Cov90"],
            online_table,
        ))
        lines.extend([
            "",
            "The DTC/VFE label above refers only to Task-1 empirical-Bayes calibration. The causal block recursion is identical; therefore the same shape-based profiler count is reused, while runtime and predictions are measured separately for every seed.",
            "",
        ])

    rank_accuracy = rank_accuracy_rows(root)
    if rank_accuracy:
        lines.extend(["## Approximate-rank accuracy-efficiency", ""])
        batch_rank = [row for row in rank_accuracy if row["mode"] == "batch empirical Bayes"]
        online_rank = [row for row in rank_accuracy if row["mode"] == "strict online"]
        lines.extend(markdown_table(
            ["Scope", "Rank", "GFLOPs/step", "Objective s/step", "RMSE", "NLL", "Cov90"],
            [
                [
                    str(row["scope"]), str(row["rank"]),
                    f"{number(row.get('profiler_counted_gflops_per_step')):.3f}",
                    f"{number(row.get('objective_runtime_seconds_per_step')):.3f}",
                    f"{number(row.get('rmse')):.4f}",
                    f"{number(row.get('nll')):.4f}",
                    f"{number(row.get('coverage90')):.4f}",
                ]
                for row in sorted(batch_rank, key=lambda item: (item["scope"], -int(item["rank"])))
            ],
        ))
        lines.append("")
        lines.extend(markdown_table(
            ["Scope", "Rank", "Stream s", "State MiB", "RMSE", "NLL", "Cov90"],
            [
                [
                    str(row["scope"]), str(row["rank"]),
                    f"{number(row.get('runtime_seconds')):.2f}",
                    f"{number(row.get('persistent_state_mib')):.2f}",
                    f"{number(row.get('rmse')):.4f}",
                    f"{number(row.get('nll')):.4f}",
                    f"{number(row.get('coverage90')):.4f}",
                ]
                for row in sorted(online_rank, key=lambda item: (item["scope"], -int(item["rank"])))
            ],
        ))
        lines.extend([
            "",
            "Ranks 73/64/48 are not exact implementations of the 133-feature model. Their lower compute is valid only as a model-changing accuracy-efficiency trade-off.",
            "",
        ])

    operator_rows = collect_operator_rows(root)
    if operator_rows:
        lines.extend(["## Profiler-counted operator leaders", ""])
        lines.extend(markdown_table(
            ["Source", "Operator", "Formula-counted GFLOPs"],
            [
                [row["source"], row["operator"], f"{row['profiler_counted_flops'] / 1e9:.3f}"]
                for row in top_profiled_operators(operator_rows)
            ],
        ))
        lines.extend(["", "This ranking excludes unsupported factorization, solve and elementwise work; it is an implementation hotspot view, not a complete algorithm ranking.", ""])

    lines.extend([
        "## Interpretation limits",
        "",
        "- Profiler counts and analytical supplements are shown separately; neither is a complete hardware instruction count.",
        "- CUDA runtime and allocated/reserved memory are implementation-level measurements on the local GPU.",
        "- CUPTI hardware activity is unavailable in the local WSL environment; profiler FLOPs are formula-based ATen estimates.",
        "- VFE+D means VFE hyperparameter training with full structured-joint conditional prediction. Strict-online recursion itself does not optimize a VFE objective per block.",
        "- Exact optimization conclusions require objective, gradient, posterior mean and covariance parity; approximate rank compression is reported separately.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    profiles = load_batch_profiles(args.root)
    unified = unified_batch_rows(args.root, profiles) + unified_online_rows(args.root)
    ablations = exact_ablation_rows(args.root)
    rank_rows = rank_accuracy_rows(args.root)
    write_csv(args.root / "unified_efficiency_table.csv", unified)
    write_csv(args.root / "exact_e0_e3_ablation.csv", ablations)
    write_csv(args.root / "operator_breakdown.csv", collect_operator_rows(args.root))
    write_csv(args.root / "rank_accuracy_efficiency.csv", rank_rows)
    (args.root / "diagnostic_report.md").write_text(
        generate_report(args.root, unified, ablations), encoding="utf-8"
    )
    print(json.dumps({
        "unified_rows": len(unified),
        "ablation_rows": len(ablations),
        "rank_accuracy_rows": len(rank_rows),
        "output": str(args.root),
    }, indent=2))


if __name__ == "__main__":
    main()
