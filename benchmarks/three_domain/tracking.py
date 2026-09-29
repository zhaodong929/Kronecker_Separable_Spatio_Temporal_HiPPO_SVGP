"""Framework-independent, append-only experiment events; no network in solvers.

The external W&B supervisor consumes this journal. Recording never imports a
learning framework, reads labels, changes RNG state, or synchronizes a GPU.
"""
import json
import math
import os
import time
from pathlib import Path


def scalar_tree(value):
    if isinstance(value, dict):
        return {str(k): scalar_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar_tree(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, RuntimeError):
            return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def emit(phase, step, metrics):
    """Store every supplied event locally; dashboard downsampling is separate.

    No-op outside the campaign. File errors propagate: silently losing the
    audit journal would violate this campaign's recording contract.
    """
    path = os.environ.get("HIPPO_EVENT_PATH")
    if not path:
        return
    row = dict(schema_version=1, time_unix=time.time(), phase=str(phase),
               step=int(step), metrics=scalar_tree(metrics))
    data = (json.dumps(row, allow_nan=False, separators=(",", ":"))+"\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        written = os.write(fd, data)
        if written != len(data):
            raise OSError("Incomplete experiment event write")
    finally:
        os.close(fd)


def numeric_metrics(row, prefix=""):
    result = {}
    for key, value in row.items():
        key = "nlpd" if key == "nll" else key
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(numeric_metrics(value, name))
        elif isinstance(value, (list, tuple)) and len(value) <= 32:
            result.update(numeric_metrics({str(i): v for i, v in enumerate(value)}, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            result[name] = value
    return result
