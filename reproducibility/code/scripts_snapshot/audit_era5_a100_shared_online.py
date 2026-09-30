#!/usr/bin/env python3
"""Audit ERA5 A100 benchmark artifacts.

The runners used by the ERA5 experiments have evolved a few directory layouts.
This module therefore discovers artifacts from their content and from the
nearest ``seedN`` directory instead of assuming one fixed runner layout.
It is intentionally conservative: missing or inconsistent evidence is
reported, never replaced with a default result.
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


SEED_RE = re.compile(r"^seed(?P<seed>\d+)$", re.IGNORECASE)
DEFAULT_SEEDS = tuple(range(5))
METRIC_NAMES = ("rmse", "nll", "coverage90")
CANONICAL_MANIFESTS = (
    "shared_batch_short.jsonl",
    "official_long_preflight.jsonl",
    "official_long_full.jsonl",
    "online_short.jsonl",
    "online_long.jsonl",
    "efficiency.jsonl",
)


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def nested(payload: Mapping[str, Any], *paths: str) -> Any:
    """Return the first non-null value found at one of the dotted paths."""

    for path in paths:
        value: Any = payload
        for key in path.split("."):
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return None


def first_metric(payload: Mapping[str, Any], metric: str) -> float | None:
    value = nested(
        payload,
        f"overall_current_block.{metric}",
        f"final_block.{metric}",
        f"overall.{metric}",
        f"metrics.{metric}",
        f"test.{metric}",
        f"final.{metric}",
        metric,
    )
    return _as_float(value)


def _seed_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = SEED_RE.match(part)
        if match:
            return int(match.group("seed"))
    return None


def _seed_dir(path: Path) -> Path | None:
    for index in range(len(path.parts) - 1, -1, -1):
        if SEED_RE.match(path.parts[index]):
            return Path(*path.parts[: index + 1])
    return None


def _lower_parts(path: Path) -> list[str]:
    return [part.lower().replace("-", "_") for part in path.parts]


def infer_scope(path: Path, payload: Mapping[str, Any] | None = None) -> str:
    """Infer ``short`` or ``long`` without assuming a numeric task size."""

    payload = payload or {}
    for value in (
        nested(payload, "scope", "task_scope", "experiment_scope", "dataset_scope"),
        nested(payload, "config.scope", "config.task_scope"),
    ):
        if value is not None:
            text = str(value).lower().replace("-", "_")
            if "long" in text or "10" in text:
                return "long"
            if "short" in text or "2" in text:
                return "short"

    parts = _lower_parts(path)
    joined = "/".join(parts)
    if any(token in joined for token in ("long", "task1_10", "tasks2_10", "task_10")):
        return "long"
    if any(token in joined for token in ("short", "task1_2", "task2_short", "task_2")):
        return "short"
    return "unknown"


def infer_branch(path: Path, payload: Mapping[str, Any] | None = None) -> str:
    """Infer the evaluation branch, keeping batch and strict online separate."""

    payload = payload or {}
    # A path-level protocol marker is authoritative.  This prevents an
    # all-seen batch artifact from entering online tables merely because its
    # result payload copied an ``online`` metadata value.
    parts = _lower_parts(path)
    for part in reversed(parts):
        if part in {"batch", "offline", "all_seen", "all_seen_batch", "all-seen", "all-seen-batch"}:
            return "batch"
        if part in {"online", "strict_online", "streaming", "stream"}:
            return "online"

    values = (
        nested(payload, "branch", "evaluation_branch", "protocol", "mode"),
        nested(payload, "config.branch", "config.protocol", "config.mode"),
    )
    for value in values:
        if value is not None:
            text = str(value).lower().replace("-", "_")
            if "online" in text or "stream" in text:
                return "online"
            if "batch" in text:
                return "batch"

    for part in reversed(parts):
        if part in {"online", "strict_online", "streaming", "stream"}:
            return "online"
        if part in {"batch", "offline", "all_seen_batch"}:
            return "batch"
        if "strict_online" in part or part.endswith("_online"):
            return "online"
        if "batch" in part:
            return "batch"
    return "unknown"


def infer_method(path: Path, payload: Mapping[str, Any] | None = None) -> str:
    payload = payload or {}
    for value in (
        nested(payload, "method", "model", "method_name", "model_name", "variant"),
        nested(payload, "config.method", "config.model", "config.variant"),
    ):
        if value is not None and str(value).strip():
            return str(value)

    parts = list(path.parts)
    seed_index = next(
        (index for index in range(len(parts) - 1, -1, -1) if SEED_RE.match(parts[index])),
        None,
    )
    if seed_index is None:
        return path.parent.name
    before_seed = parts[:seed_index]
    for index in range(len(before_seed) - 1, -1, -1):
        part = before_seed[index]
        lower = part.lower().replace("-", "_")
        if lower in {"online", "strict_online", "batch", "offline", "streaming"}:
            if index + 1 < len(before_seed):
                return before_seed[index + 1]
            break
        if lower.endswith("_online") or lower.endswith("_batch"):
            return part
    return before_seed[-1] if before_seed else path.parent.name


def _candidate_files(run_dir: Path, names: Iterable[str], patterns: Iterable[str] = ()) -> list[Path]:
    found: list[Path] = []
    for name in names:
        direct = run_dir / name
        if direct.is_file():
            found.append(direct)
    for pattern in patterns:
        found.extend(path for path in run_dir.rglob(pattern) if path.is_file())
    return sorted(set(found), key=lambda path: (len(path.parts), str(path)))


def _discover_groups(root: Path) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    """Discover logical runs keyed by scope, branch, method, and seed."""

    groups: dict[tuple[str, str, str, int], dict[str, Any]] = {}

    def add(path: Path, kind: str, payload: Mapping[str, Any] | None = None) -> None:
        seed = _seed_from_path(path)
        if seed is None:
            return
        seed_dir = _seed_dir(path)
        if seed_dir is None:
            return
        scope = infer_scope(path, payload)
        branch = infer_branch(path, payload)
        method = infer_method(path, payload)
        key = (scope, branch, method, seed)
        group = groups.setdefault(
            key,
            {
                "scope": scope,
                "branch": branch,
                "method": method,
                "seed": seed,
                "run_dir": seed_dir,
                "result_paths": [],
                "prediction_paths": [],
                "status_paths": [],
            },
        )
        if path.is_file():
            group[f"{kind}_paths"].append(path)
        if len(path.parts) < len(group["run_dir"].parts):
            group["run_dir"] = seed_dir

    for path in root.rglob("result.json"):
        payload = _json_object(path)
        add(path, "result", payload)
    for path in root.rglob("predictions.npz"):
        add(path, "prediction")
    for path in root.rglob("status.json"):
        payload = _json_object(path)
        add(path, "status", payload)
    for path in root.rglob("run_status.json"):
        payload = _json_object(path)
        add(path, "status", payload)
    for path in root.rglob("*.status"):
        add(path, "status")
    for path in root.rglob("status.txt"):
        add(path, "status")

    # A directory with a seed child but no artifact is still an expected run
    # candidate.  This catches failed jobs that only left an empty directory.
    for seed_dir in root.rglob("seed*"):
        if not seed_dir.is_dir() or _seed_from_path(seed_dir) is None:
            continue
        add(seed_dir, "result")
        key = (
            infer_scope(seed_dir),
            infer_branch(seed_dir),
            infer_method(seed_dir),
            _seed_from_path(seed_dir),
        )
        groups.setdefault(
            key,
            {
                "scope": key[0],
                "branch": key[1],
                "method": key[2],
                "seed": key[3],
                "run_dir": seed_dir,
                "result_paths": [],
                "prediction_paths": [],
                "status_paths": [],
            },
        )

    # Some probes encode the seed in their directory name rather than a
    # seedN component.  They are intentionally not promoted into result runs.
    return groups


def _select_path(paths: Iterable[Path], run_dir: Path) -> Path | None:
    values = sorted(set(paths), key=lambda path: (len(path.relative_to(run_dir).parts), str(path)))
    return values[0] if values else None


def _status_values(paths: Iterable[Path]) -> tuple[list[str], list[str]]:
    statuses: list[str] = []
    failures: list[str] = []
    for path in sorted(set(paths)):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            failures.append(f"cannot_read_status:{path.name}:{exc}")
            continue
        payload: Any = None
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                failures.append(f"invalid_status_json:{path.name}")
        value = nested(payload, "status", "state", "result", "outcome") if isinstance(payload, Mapping) else text
        if value is not None:
            status = str(value).lower()
            statuses.append(status)
            if any(token in status for token in ("fail", "error", "cancel", "timeout")):
                failures.append(f"status={status}:{path.name}")
    return statuses, failures


def _prediction_array(data: Mapping[str, Any], names: Iterable[str]) -> np.ndarray | None:
    for name in names:
        if name in data:
            return np.asarray(data[name], dtype=float).reshape(-1)
    return None


def recompute_prediction_metrics(path: Path) -> tuple[dict[str, float], list[str]]:
    """Recompute metrics from a prediction NPZ and return issues separately."""

    issues: list[str] = []
    metrics: dict[str, float] = {}
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        return {}, [f"invalid_predictions:{exc}"]
    try:
        y_true = _prediction_array(loaded, ("y_true", "targets", "target", "y"))
        pred_mean = _prediction_array(
            loaded,
            ("pred_mean", "prediction_mean", "mean", "pred", "y_pred"),
        )
        pred_var = _prediction_array(
            loaded,
            ("pred_var", "prediction_var", "variance", "pred_variance", "var"),
        )
        if y_true is None:
            issues.append("predictions_missing_y_true")
        if pred_mean is None:
            issues.append("predictions_missing_pred_mean")
        if y_true is None or pred_mean is None:
            return metrics, issues
        if y_true.size != pred_mean.size:
            issues.append(f"prediction_length_mismatch:{y_true.size}!={pred_mean.size}")
            return metrics, issues
        if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(pred_mean))):
            issues.append("predictions_nonfinite_mean_or_target")
            return metrics, issues
        error = y_true - pred_mean
        metrics["rmse"] = float(np.sqrt(np.mean(error**2)))
        if pred_var is None:
            issues.append("predictions_missing_pred_var")
            return metrics, issues
        if pred_var.size != y_true.size:
            issues.append(f"prediction_variance_length_mismatch:{pred_var.size}!={y_true.size}")
            return metrics, issues
        if not np.all(np.isfinite(pred_var)):
            issues.append("predictions_nonfinite_variance")
            return metrics, issues
        if np.any(pred_var <= 0):
            issues.append("predictions_nonpositive_variance")
        safe_var = np.maximum(pred_var, np.finfo(float).tiny)
        metrics["nll"] = float(
            np.mean(0.5 * (np.log(2.0 * np.pi * safe_var) + error**2 / safe_var))
        )
        half = 1.6448536269514722 * np.sqrt(safe_var)
        metrics["coverage90"] = float(np.mean((y_true >= pred_mean - half) & (y_true <= pred_mean + half)))
    finally:
        loaded.close()
    return metrics, issues


def _audit_group(group: Mapping[str, Any], root: Path, atol: float, rtol: float) -> dict[str, Any]:
    run_dir = Path(group["run_dir"])
    result_path = _select_path(group["result_paths"], run_dir)
    prediction_path = _select_path(group["prediction_paths"], run_dir)
    status_paths = list(group["status_paths"])
    issues: list[str] = []
    artifact_status = "complete"
    result: dict[str, Any] | None = None
    if result_path is None:
        issues.append("missing_result.json")
    else:
        result = _json_object(result_path)
        if result is None:
            issues.append("invalid_result.json")
    if prediction_path is None:
        issues.append("missing_predictions.npz")

    statuses, status_issues = _status_values(status_paths)
    issues.extend(status_issues)
    if any(any(token in status for token in ("fail", "error", "cancel", "timeout")) for status in statuses):
        issues.append("failure_status_artifact")

    reported: dict[str, float | None] = {metric: None for metric in METRIC_NAMES}
    recomputed: dict[str, float] = {}
    differences: dict[str, float] = {}
    if result is not None:
        for metric in METRIC_NAMES:
            reported[metric] = first_metric(result, metric)
        explicit_status = nested(result, "status", "state", "outcome")
        if explicit_status is not None:
            status = str(explicit_status).lower()
            statuses.append(status)
            if any(token in status for token in ("fail", "error", "cancel", "timeout")):
                issues.append(f"result_status={status}")

    if prediction_path is not None:
        recomputed, prediction_issues = recompute_prediction_metrics(prediction_path)
        issues.extend(prediction_issues)
        for metric, value in recomputed.items():
            if reported[metric] is None:
                issues.append(f"result_missing_{metric}")
                continue
            difference = abs(float(reported[metric]) - value)
            differences[metric] = difference
            if not math.isclose(float(reported[metric]), value, rel_tol=rtol, abs_tol=atol):
                issues.append(f"{metric}_mismatch:{difference:.6g}")

    if issues:
        artifact_status = "incomplete"
    if any("status=" in issue or "failure_status" in issue or "result_status" in issue for issue in issues):
        artifact_status = "failed"

    def rel(path: Path | None) -> str:
        return str(path.relative_to(root)) if path is not None and path.is_relative_to(root) else (str(path) if path else "")

    return {
        "scope": group["scope"],
        "branch": group["branch"],
        "method": group["method"],
        "seed": group["seed"],
        "run_dir": rel(run_dir),
        "result_path": rel(result_path),
        "prediction_path": rel(prediction_path),
        "status_paths": ";".join(rel(path) for path in sorted(set(status_paths))),
        "status_values": ";".join(sorted(set(statuses))),
        "status": artifact_status,
        "reported_rmse": reported["rmse"],
        "reported_nll": reported["nll"],
        "reported_coverage90": reported["coverage90"],
        "recomputed_rmse": recomputed.get("rmse"),
        "recomputed_nll": recomputed.get("nll"),
        "recomputed_coverage90": recomputed.get("coverage90"),
        "rmse_absolute_difference": differences.get("rmse"),
        "nll_absolute_difference": differences.get("nll"),
        "coverage90_absolute_difference": differences.get("coverage90"),
        "issues": ";".join(issues),
    }


def _missing_seed_rows(rows: list[dict[str, Any]], expected_seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    observed = {(row["scope"], row["branch"], row["method"]): set() for row in rows}
    for row in rows:
        observed.setdefault((row["scope"], row["branch"], row["method"]), set()).add(row["seed"])
    missing: list[dict[str, Any]] = []
    for (scope, branch, method), seeds in sorted(observed.items()):
        for seed in expected_seeds:
            if seed in seeds:
                continue
            missing.append(
                {
                    "scope": scope,
                    "branch": branch,
                    "method": method,
                    "seed": seed,
                    "run_dir": "",
                    "result_path": "",
                    "prediction_path": "",
                    "status_paths": "",
                    "status_values": "",
                    "status": "missing",
                    "reported_rmse": None,
                    "reported_nll": None,
                    "reported_coverage90": None,
                    "recomputed_rmse": None,
                    "recomputed_nll": None,
                    "recomputed_coverage90": None,
                    "rmse_absolute_difference": None,
                    "nll_absolute_difference": None,
                    "coverage90_absolute_difference": None,
                    "issues": "missing_expected_seed_artifacts",
                }
            )
    return missing


def _missing_manifest_rows(root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report manifest-declared jobs with no discoverable artifact or status."""

    observed = {
        (str(row["scope"]), str(row["branch"]), str(row["method"]), int(row["seed"]))
        for row in rows
        if row.get("seed") is not None
    }
    missing: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    manifest_dir = root / "manifests"
    for name in CANONICAL_MANIFESTS:
        path = manifest_dir / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            seed = record.get("seed")
            if not isinstance(seed, int) or isinstance(seed, bool):
                continue
            output_dir = Path(str(record.get("output_dir", manifest_dir)))
            scope = infer_scope(output_dir, record)
            branch = infer_branch(output_dir, record)
            method = str(record.get("method", ""))
            key = (scope, branch, method, seed)
            if not method or key in observed or key in seen:
                continue
            seen.add(key)
            try:
                run_dir = str(output_dir.relative_to(root))
            except ValueError:
                run_dir = str(output_dir)
            missing.append(
                {
                    "scope": scope,
                    "branch": branch,
                    "method": method,
                    "seed": seed,
                    "run_dir": run_dir,
                    "result_path": "",
                    "prediction_path": "",
                    "status_paths": "",
                    "status_values": "",
                    "status": "missing",
                    "reported_rmse": None,
                    "reported_nll": None,
                    "reported_coverage90": None,
                    "recomputed_rmse": None,
                    "recomputed_nll": None,
                    "recomputed_coverage90": None,
                    "rmse_absolute_difference": None,
                    "nll_absolute_difference": None,
                    "coverage90_absolute_difference": None,
                    "issues": f"missing_manifest_job_artifacts:{name}",
                }
            )
    return missing


def audit_benchmark_root(
    benchmark_root: str | Path,
    expected_seeds: Iterable[int] = DEFAULT_SEEDS,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    include_missing_seeds: bool = True,
) -> dict[str, Any]:
    """Return a JSON-serializable audit payload for a benchmark root."""

    root = Path(benchmark_root).expanduser().resolve()
    seeds = tuple(sorted({int(seed) for seed in expected_seeds}))
    if not root.is_dir():
        return {
            "benchmark_root": str(root),
            "expected_seeds": list(seeds),
            "complete_runs": 0,
            "incomplete_runs": 0,
            "failed_runs": 0,
            "missing_runs": 0,
            "runs": [],
            "error": "benchmark_root_not_found",
        }

    rows = [_audit_group(group, root, atol, rtol) for group in _discover_groups(root).values()]
    rows.sort(key=lambda row: (str(row["scope"]), str(row["branch"]), str(row["method"]), int(row["seed"])))
    if include_missing_seeds:
        rows.extend(_missing_seed_rows(rows, seeds))
    rows.extend(_missing_manifest_rows(root, rows))
    rows.sort(key=lambda row: (str(row["scope"]), str(row["branch"]), str(row["method"]), int(row["seed"])))
    return {
        "benchmark_root": str(root),
        "expected_seeds": list(seeds),
        "complete_runs": sum(row["status"] == "complete" for row in rows),
        "incomplete_runs": sum(row["status"] == "incomplete" for row in rows),
        "failed_runs": sum(row["status"] == "failed" for row in rows),
        "missing_runs": sum(row["status"] == "missing" for row in rows),
        "prediction_mismatch_runs": sum(
            any(token in str(row["issues"]) for token in ("_mismatch:", "prediction_length_mismatch"))
            for row in rows
        ),
        "runs": rows,
    }


def write_audit_csv(payload: Mapping[str, Any], path: str | Path) -> None:
    rows = list(payload.get("runs", []))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "branch",
        "method",
        "seed",
        "status",
        "run_dir",
        "result_path",
        "prediction_path",
        "status_paths",
        "status_values",
        "reported_rmse",
        "reported_nll",
        "reported_coverage90",
        "recomputed_rmse",
        "recomputed_nll",
        "recomputed_coverage90",
        "rmse_absolute_difference",
        "nll_absolute_difference",
        "coverage90_absolute_difference",
        "issues",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="JSON output path")
    parser.add_argument("--csv-output", type=Path, help="optional flat CSV output path")
    parser.add_argument("--expected-seeds", default="0,1,2,3,4")
    parser.add_argument("--include-missing-seeds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_seeds = tuple(int(value) for value in args.expected_seeds.split(",") if value.strip())
    payload = audit_benchmark_root(
        args.benchmark_root,
        expected_seeds,
        atol=args.atol,
        rtol=args.rtol,
        include_missing_seeds=args.include_missing_seeds,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.csv_output:
        write_audit_csv(payload, args.csv_output)


if __name__ == "__main__":
    main()
