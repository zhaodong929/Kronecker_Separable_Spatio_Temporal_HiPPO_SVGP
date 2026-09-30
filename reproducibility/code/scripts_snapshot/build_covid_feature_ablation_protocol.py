#!/usr/bin/env python3
"""Create zero-mean and intercept-only copies of a COVID protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["zero_mean", "intercept_only"], required=True)
    args = parser.parse_args()
    with np.load(args.source_npz) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    if args.mode == "zero_mean":
        calibration_phi = np.zeros((*payload["calibration_y"].shape, 1), dtype=np.float32)
        stream_phi = np.zeros((*payload["stream_y"].shape, 1), dtype=np.float32)
    else:
        calibration_phi = np.ones((*payload["calibration_y"].shape, 1), dtype=np.float32)
        stream_phi = np.ones((*payload["stream_y"].shape, 1), dtype=np.float32)
    train = np.asarray(payload["train_indices"], dtype=int)
    design = calibration_phi[:, train].reshape(-1, 1).astype(np.float64)
    target = payload["calibration_y"][:, train].reshape(-1).astype(np.float64)
    beta = np.linalg.solve(design.T @ design + 1e-3 * np.eye(1), design.T @ target)
    payload["calibration_phi"] = calibration_phi
    payload["stream_phi"] = stream_phi
    payload["task1_calibration_mean"] = np.einsum("tsp,p->ts", calibration_phi, beta).astype(np.float32)
    payload["task1_stream_mean"] = np.einsum("tsp,p->ts", stream_phi, beta).astype(np.float32)
    payload["batch_stream_mean"] = payload["task1_stream_mean"].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    metadata = json.loads(args.source_json.read_text(encoding="utf-8"))
    metadata["npz"] = str(args.output.resolve())
    metadata["feature_ablation"] = args.mode
    metadata["xlag"] = {
        "mode": args.mode,
        "features": 1,
        "columns": ["zero"] if args.mode == "zero_mean" else ["intercept"],
        "test_target_lags_used": False,
    }
    metadata["npz_sha256"] = sha256(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
