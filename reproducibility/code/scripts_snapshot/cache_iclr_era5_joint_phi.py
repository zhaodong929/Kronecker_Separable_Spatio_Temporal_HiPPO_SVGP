#!/usr/bin/env python3
"""Materialize one reproducible joint-X-lag tensor for repeated benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_iclr_era5_routeb_batch import load_joint_phi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-part", choices=["calibration", "stream"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xlag-length", type=int, default=10)
    args = parser.parse_args()
    started = time.perf_counter()
    with np.load(args.protocol_npz) as arrays:
        phi, source_loading_seconds = load_joint_phi(
            arrays=arrays,
            protocol_json=args.protocol_json,
            data_root=args.data_root,
            xlag_length=args.xlag_length,
            data_part=args.data_part,
        )
    phi = np.asarray(phi, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, phi, allow_pickle=False)
    reloaded = np.load(args.output, mmap_mode="r")
    if reloaded.shape != phi.shape or reloaded.dtype != phi.dtype:
        raise RuntimeError("Cached Phi failed shape/dtype verification")
    payload = {
        "output": str(args.output),
        "shape": list(phi.shape),
        "dtype": str(phi.dtype),
        "bytes": int(phi.nbytes),
        "source_loading_seconds": source_loading_seconds,
        "process_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
