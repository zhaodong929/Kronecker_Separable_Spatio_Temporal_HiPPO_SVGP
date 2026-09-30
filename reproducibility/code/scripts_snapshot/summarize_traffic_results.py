#!/usr/bin/env python3
"""Aggregate manifest-backed traffic result archives without mixing protocols."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


METRICS = ("rmse", "rmse_speed", "crps", "gaussian_nlpd", "ece", "coverage90")


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def method_label(method: str, mode: str) -> str:
    mapping = {
        "hippo": "KronHiPPO-STGP",
        "ordinary": "Kron-STGP",
        "mean_field": "Mean-field/Decoupled",
        "persistence": "Last-value persistence",
        "frozen_mean": "Frozen mean",
    }
    return mapping[method] if method != "hippo" or mode == "changing" else "KronHiPPO-STGP (fixed transfer)"


def summary_row(key: tuple[str, ...], group: list[dict[str, object]]) -> dict[str, object]:
    row = {name: value for name, value in zip(("dataset", "protocol", "experiment_label", "method", "mode", "horizon_steps"), key)}
    row["runs"] = len(group)
    for metric in METRICS:
        values = np.asarray([float(item[metric]) for item in group], dtype=float)
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_sample_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return row


def validate_archive(path: Path, payload: dict[str, object]) -> str | None:
    """Return a causal/numerical gate failure, or ``None`` for a valid run."""

    status_path = path.parent / "status.json"
    prediction_path = path.parent / "predictions.npz"
    if not status_path.exists() or json.loads(status_path.read_text(encoding="utf-8")).get("status") != "complete":
        return "missing or incomplete status.json"
    if not prediction_path.exists():
        return "missing predictions.npz"
    result = dict(payload.get("result", {}))
    protocol = str(result.get("protocol", "unknown"))
    guard = dict(result.get("guard", result.get("nowcasting_guard", {})))
    if int(guard.get("current_hidden_reads_before_prediction", -1)) != 0:
        return "current hidden label read before prediction"
    with np.load(prediction_path) as archive:
        if protocol == "N":
            groups = [(archive["y"], archive["mean"], archive["variance"])]
        elif protocol == "F":
            horizons = sorted(key.removeprefix("y_h") for key in archive.files if key.startswith("y_h"))
            groups = [(archive[f"y_h{h}"], archive[f"mean_h{h}"], archive[f"variance_h{h}"]) for h in horizons]
        else:
            return f"unknown protocol {protocol}"
        for y, mean, variance in groups:
            if not (np.isfinite(y).all() and np.isfinite(mean).all() and np.isfinite(variance).all()):
                return "non-finite prediction archive"
            if np.any(variance <= 0.0):
                return "non-positive predictive variance"
        stream_steps = int(archive["stream_indices"].size)
    if int(guard.get("current_hidden_reveals", -1)) != stream_steps:
        return "hidden reveal count does not match archived stream length"
    if int(guard.get("unique_delayed_hidden_absorptions", -1)) != max(0, stream_steps - 1):
        return "delayed hidden labels were not absorbed exactly once"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="*", default=[])
    args = parser.parse_args()
    files = sorted(args.results_root.rglob("result.json"))
    nowcast: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    forecast: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    failures: list[dict[str, object]] = []
    observed_seeds: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        protocol = str(result.get("protocol", "unknown"))
        experiment_label = path.parent.parent.name
        method = method_label(str(payload["method"]), str(result.get("mode", "unknown")))
        failure = validate_archive(path, payload)
        if failure is not None:
            failures.append({"path": str(path), "reason": failure})
            continue
        try:
            seed = int(path.parent.name.removeprefix("seed"))
        except ValueError:
            failures.append({"path": str(path), "reason": "output directory is not named seed<N>"})
            continue
        if protocol == "N":
            final = dict(result["final"])
            key = (str(payload["dataset"]), protocol, experiment_label, method, str(result["mode"]), "")
            nowcast[key].append(final)
            observed_seeds[key].add(seed)
        elif protocol == "F":
            for horizon, values in dict(result["final"]).items():
                key = (str(payload["dataset"]), protocol, experiment_label, method, str(result["mode"]), str(horizon))
                forecast[key].append(dict(values))
                observed_seeds[key].add(seed)
    if args.expected_seeds:
        expected = set(args.expected_seeds)
        for key, seeds in sorted(observed_seeds.items()):
            if seeds != expected:
                failures.append(
                    {
                        "group": "/".join(key),
                        "reason": f"expected seeds {sorted(expected)}, found {sorted(seeds)}",
                    }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    nowcast_rows = [summary_row(key, values) for key, values in sorted(nowcast.items())]
    forecast_rows = [summary_row(key, values) for key, values in sorted(forecast.items())]
    write_csv(nowcast_rows, args.output / "nowcasting_summary.csv")
    write_csv(forecast_rows, args.output / "forecast_summary.csv")
    audit = {
        "source_root": str(args.results_root),
        "result_archives_found": len(files),
        "nowcasting_groups": len(nowcast_rows),
        "forecast_groups": len(forecast_rows),
        "guard_failures": failures,
        "status": "pass" if not failures else "fail",
    }
    (args.output / "summary_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
