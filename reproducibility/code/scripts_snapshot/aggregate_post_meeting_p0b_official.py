#!/usr/bin/env python3
"""Aggregate full-protocol official Bayes-Newton P0b runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev


RESOURCE_PATTERNS = {
    "elapsed": re.compile(r"Elapsed \(wall clock\) time .*: (.+)$"),
    "peak_rss_kb": re.compile(r"Maximum resident set size \(kbytes\): (\d+)$"),
}


def parse_elapsed(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60.0 + float(seconds)
    return float(parts[0])


def resource_values(path: Path) -> dict[str, float]:
    values = {"wall_seconds": math.nan, "peak_rss_mb": math.nan}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        elapsed = RESOURCE_PATTERNS["elapsed"].search(line)
        if elapsed:
            values["wall_seconds"] = parse_elapsed(elapsed.group(1))
        peak = RESOURCE_PATTERNS["peak_rss_kb"].search(line)
        if peak:
            values["peak_rss_mb"] = float(peak.group(1)) / 1024.0
    return values


def normalized_status(status: str, run_dir: Path) -> str:
    if status != "failed":
        return status
    stderr_path = run_dir / "stderr.log"
    if stderr_path.exists():
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace").lower()
        if "resource exhausted" in stderr or "failed to allocate" in stderr:
            return "resource_exhausted"
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()

    rows: list[dict[str, object]] = []
    for status_path in sorted(root.glob("seed*/**/status.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        run_dir = status_path.parent
        result_path = run_dir / "result.json"
        status_value = normalized_status(str(status["status"]), run_dir)
        row: dict[str, object] = {
            "seed": int(status["seed"]),
            "model": status["model"],
            "num_spatial_inducing": status.get("num_spatial_inducing"),
            "status": status_value,
            "exit_code": status.get("exit_code"),
            **resource_values(run_dir / "resource_usage.txt"),
        }
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            for key in [
                "rmse", "nll", "coverage90", "mean_predictive_std", "train_seconds",
                "learned_temporal_lengthscale", "learned_likelihood_variance",
                "iterations", "iterations_requested", "stop_reason",
            ]:
                row[key] = result.get(key)
            row["learned_spatial_lengthscale_0"] = result.get("learned_spatial_lengthscales", [None, None])[0]
            row["learned_spatial_lengthscale_1"] = result.get("learned_spatial_lengthscales", [None, None])[1]
        rows.append(row)

    fields = sorted({key for row in rows for key in row})
    with (root / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    completed = [row for row in rows if row["status"] == "completed"]
    groups: dict[tuple[str, object], list[dict[str, object]]] = {}
    for row in completed:
        groups.setdefault((str(row["model"]), row["num_spatial_inducing"]), []).append(row)

    summary: list[dict[str, object]] = []
    metrics = [
        "rmse", "nll", "coverage90", "mean_predictive_std", "iterations",
        "wall_seconds", "peak_rss_mb",
    ]
    for (model, ms), group in sorted(groups.items(), key=lambda item: (item[0][0], int(item[0][1] or 0))):
        record: dict[str, object] = {
            "model": model,
            "num_spatial_inducing": ms,
            "num_completed_seeds": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric) is not None and math.isfinite(float(row[metric]))]
            record[f"{metric}_mean"] = mean(values) if values else math.nan
            record[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0 if values else math.nan
        summary.append(record)

    summary_fields = sorted({key for row in summary for key in row})
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    manifest = {
        "scope": "Full P1-aligned ERA5 task_2 variable_0 direct-target official Bayes-Newton P0b",
        "num_runs": len(rows),
        "num_completed": len(completed),
        "rows": rows,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"all_runs": str(root / "all_runs.csv"), "summary": str(root / "summary.csv"), **manifest}, indent=2))


if __name__ == "__main__":
    main()
