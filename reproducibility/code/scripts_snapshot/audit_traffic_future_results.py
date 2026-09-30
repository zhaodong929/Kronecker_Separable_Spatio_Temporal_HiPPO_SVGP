#!/usr/bin/env python3
"""Audit completed PEMS-BAY Protocol-F prediction archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HORIZONS = (3, 6, 12)
DETERMINISTIC_METHODS = {"graph_wavenet", "dcrnn"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_one(path: Path, expected_steps: int, method: str) -> tuple[dict[str, object], dict[int, np.ndarray]]:
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    archive = np.load(path / "predictions.npz")
    if status.get("status") != "complete" or result["result"]["protocol"] != "F":
        raise RuntimeError(f"Incomplete or non-Protocol-F result: {path}")
    stream = np.asarray(archive["stream_indices"], dtype=int)
    if stream.shape != (expected_steps,):
        raise RuntimeError(f"Unexpected stream shape at {path}: {stream.shape}")
    guard = result["result"]["nowcasting_guard"]
    expected_guard = {
        "current_hidden_reads_before_prediction": 0,
        "current_hidden_reveals": expected_steps,
        "current_visible_reads": expected_steps,
        "unique_delayed_hidden_absorptions": expected_steps - 1,
    }
    if guard != expected_guard:
        raise RuntimeError(f"Causal guard failed at {path}: {guard}")

    with (path / "per_step_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_steps * len(HORIZONS):
        raise RuntimeError(f"Unexpected metric row count at {path}: {len(rows)}")
    if any(int(row["unknown_future_exogenous_reads"]) != 0 for row in rows):
        raise RuntimeError(f"Unknown future exogenous read detected at {path}")

    truths: dict[int, np.ndarray] = {}
    for horizon in HORIZONS:
        y = np.asarray(archive[f"y_h{horizon}"], dtype=float)
        mean = np.asarray(archive[f"mean_h{horizon}"], dtype=float)
        expected_shape = (expected_steps, 65)
        if y.shape != expected_shape or mean.shape != expected_shape:
            raise RuntimeError(f"Unexpected horizon-{horizon} shape at {path}")
        if not np.isfinite(y).all() or not np.isfinite(mean).all():
            raise RuntimeError(f"Non-finite archive at {path}, horizon {horizon}")
        variance_key = f"variance_h{horizon}"
        if method in DETERMINISTIC_METHODS:
            if variance_key in archive.files:
                raise RuntimeError(f"Deterministic baseline unexpectedly reports variance at {path}")
        else:
            variance = np.asarray(archive[variance_key], dtype=float)
            if variance.shape != expected_shape or not np.isfinite(variance).all():
                raise RuntimeError(f"Invalid variance archive at {path}, horizon {horizon}")
            if not np.all(variance > 0.0):
                raise RuntimeError(f"Non-positive predictive variance at {path}, horizon {horizon}")
        truths[horizon] = y
    return (
        {
            "path": str(path),
            "status": "passed",
            "steps": expected_steps,
            "guard": guard,
            "result_sha256": sha256(path / "result.json"),
            "predictions_sha256": sha256(path / "predictions.npz"),
        },
        truths,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    reference_truth: dict[tuple[int, int], np.ndarray] = {}
    for seed in args.seeds:
        for method in args.methods:
            path = args.root / "pems_bay" / "forecast" / method / f"seed{seed}"
            record, truths = audit_one(path, args.expected_steps, method)
            for horizon, truth in truths.items():
                key = (seed, horizon)
                if key in reference_truth:
                    np.testing.assert_array_equal(reference_truth[key], truth)
                else:
                    reference_truth[key] = truth
            record.update({"seed": seed, "method": method})
            records.append(record)

    payload = {
        "schema_version": 1,
        "status": "passed",
        "expected_steps": args.expected_steps,
        "horizons": list(HORIZONS),
        "records": records,
    }
    output = args.output or (args.root / "PROTOCOL_F_ARCHIVE_AUDIT.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
