#!/usr/bin/env python
"""Inspect the processed HiPPO-SVGP ERA5 time-series dataset.

This script is intentionally read-only. It reports the actual on-disk layout,
per-location `.npz` contents, sequence length consistency, numeric health, and
available scaler metadata.
"""

from __future__ import annotations

import argparse
import pickletools
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LAT_LON_RE = re.compile(r"lat_([-0-9.]+)_lon_([-0-9.]+)")


@dataclass(frozen=True)
class FileSummary:
    path: Path
    task: str
    scaled: bool
    lat: float
    lon: float


def parse_lat_lon(path: Path) -> tuple[float, float]:
    match = LAT_LON_RE.search(path.stem.replace("_scaled", ""))
    if match is None:
        raise ValueError(f"Cannot parse lat/lon from {path.name}")
    return float(match.group(1)), float(match.group(2))


def is_real_npz(path: Path) -> bool:
    return path.name.endswith(".npz") and ":Zone.Identifier" not in path.name


def first_values(array: np.ndarray, max_values: int = 5) -> str:
    flat = np.asarray(array).reshape(-1)
    head = flat[:max_values]
    return np.array2string(head, precision=4, separator=", ")


def summarize_array(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    summary: dict[str, Any] = {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "first_values": first_values(arr),
    }
    if np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        summary.update(
            {
                "nan_count": int(np.isnan(arr).sum()),
                "inf_count": int(np.isinf(arr).sum()),
                "finite_min": float(np.nanmin(arr[finite])) if finite.any() else None,
                "finite_max": float(np.nanmax(arr[finite])) if finite.any() else None,
            }
        )
    return summary


def summarize_scaler(path: Path) -> list[str]:
    lines = [f"### `{path.name}`", ""]
    if not path.exists():
        return lines + ["Missing.", ""]
    try:
        with path.open("rb") as handle:
            scaler = pickle.load(handle)
    except Exception as exc:  # pragma: no cover - defensive report path.
        try:
            import joblib

            scaler = joblib.load(path)
            lines.append(f"- normal pickle failed: `{type(exc).__name__}: {exc}`")
            lines.append("- loaded with `joblib.load`")
        except Exception as joblib_exc:
            return _summarize_unloadable_scaler(path, lines, exc, joblib_exc)

    return lines + _summarize_scaler_object(scaler)


def _summarize_unloadable_scaler(path: Path, lines: list[str], pickle_exc: Exception, joblib_exc: Exception) -> list[str]:
    lines.append(f"Could not load scaler with normal pickle: `{type(pickle_exc).__name__}: {pickle_exc}`")
    lines.append(f"Could not load scaler with joblib: `{type(joblib_exc).__name__}: {joblib_exc}`")
    lines.append("- pickle global references:")
    raw = path.read_bytes()
    try:
        globals_seen: list[str] = []
        with path.open("rb") as handle:
            for opcode, arg, _ in pickletools.genops(raw):
                if opcode.name in {"GLOBAL", "STACK_GLOBAL"} and arg is not None:
                    globals_seen.append(str(arg))
        if globals_seen:
            for item in sorted(set(globals_seen)):
                lines.append(f"  - `{item}`")
        else:
            lines.append("  - none found")
    except Exception as fallback_exc:
        lines.append(f"  - fallback failed: `{type(fallback_exc).__name__}: {fallback_exc}`")
    ascii_text = raw.decode("latin1", errors="ignore")
    class_hints = sorted(
        {
            hint
            for hint in re.findall(r"(sklearn[._A-Za-z0-9]+|StandardScaler|MinMaxScaler|RobustScaler)", ascii_text)
        }
    )
    if class_hints:
        lines.append("- raw pickle class/name hints:")
        for hint in class_hints[:20]:
            lines.append(f"  - `{hint}`")
    lines.append("")
    return lines


def _summarize_scaler_object(scaler: Any) -> list[str]:
    lines: list[str] = []
    lines.append(f"- type: `{type(scaler).__module__}.{type(scaler).__name__}`")
    attrs = vars(scaler) if hasattr(scaler, "__dict__") else {}
    if not attrs:
        lines.append("- attributes: none exposed through `__dict__`")
        lines.append("")
        return lines

    lines.append("- attributes:")
    for name in sorted(attrs):
        value = attrs[name]
        if isinstance(value, np.ndarray):
            info = summarize_array(value)
            lines.append(
                f"  - `{name}`: shape={info['shape']}, dtype={info['dtype']}, first={info['first_values']}"
            )
        elif np.isscalar(value) or isinstance(value, (str, bool, int, float, type(None))):
            lines.append(f"  - `{name}`: `{value}`")
        else:
            lines.append(f"  - `{name}`: `{type(value).__module__}.{type(value).__name__}`")
    lines.append("")
    return lines


def inspect_task(root: Path, task: str, sample_files: int) -> tuple[list[str], list[FileSummary]]:
    task_dir = root / task
    seq_dir = task_dir / "sequences"
    lines = [f"## {task}", ""]
    if not seq_dir.exists():
        lines += [f"Missing sequence directory: `{seq_dir}`", ""]
        return lines, []

    all_npz = sorted(path for path in seq_dir.iterdir() if path.is_file() and is_real_npz(path))
    scaled = [path for path in all_npz if path.name.endswith("_scaled.npz")]
    unscaled = [path for path in all_npz if not path.name.endswith("_scaled.npz")]
    lines += [
        f"- sequence dir: `{seq_dir}`",
        f"- all real `.npz`: {len(all_npz)}",
        f"- scaled `*_scaled.npz`: {len(scaled)}",
        f"- unscaled `.npz`: {len(unscaled)}",
    ]

    parsed: list[FileSummary] = []
    parse_errors: list[str] = []
    for path in all_npz:
        try:
            lat, lon = parse_lat_lon(path)
            parsed.append(FileSummary(path=path, task=task, scaled=path.name.endswith("_scaled.npz"), lat=lat, lon=lon))
        except ValueError as exc:
            parse_errors.append(str(exc))

    coords = {(item.lat, item.lon) for item in parsed}
    lines += [
        f"- parse errors: {len(parse_errors)}",
        f"- unique lat/lon locations: {len(coords)}",
    ]
    if coords:
        lat_vals = [lat for lat, _ in coords]
        lon_vals = [lon for _, lon in coords]
        lines += [
            f"- latitude range: {min(lat_vals):.4f} to {max(lat_vals):.4f}",
            f"- longitude range: {min(lon_vals):.4f} to {max(lon_vals):.4f}",
        ]
    lines.append("")

    length_records: dict[str, list[int]] = {}
    numeric_health: dict[str, tuple[int, int]] = {}
    sample_candidates = unscaled[:sample_files] + scaled[:sample_files]
    for path in sample_candidates:
        lines += [f"### sample `{task}/{path.name}`", ""]
        with np.load(path, allow_pickle=True) as data:
            lines.append(f"- keys: `{list(data.keys())}`")
            for key in data.keys():
                arr = np.asarray(data[key])
                info = summarize_array(arr)
                lines.append(
                    f"- `{key}`: shape={info['shape']}, dtype={info['dtype']}, first={info['first_values']}"
                )
                if "nan_count" in info:
                    lines.append(
                        f"  - numeric health: nan={info['nan_count']}, inf={info['inf_count']}, "
                        f"min={info['finite_min']}, max={info['finite_max']}"
                    )
        lines.append("")

    for path in all_npz:
        with np.load(path, allow_pickle=True) as data:
            for key in data.keys():
                arr = np.asarray(data[key])
                if key.startswith("time_"):
                    length_records.setdefault(key, []).append(int(arr.reshape(-1).shape[0]))
                elif key.startswith("data_") and arr.ndim >= 1:
                    length_records.setdefault(key, []).append(int(arr.shape[-1]))
                if np.issubdtype(arr.dtype, np.number):
                    prev_nan, prev_inf = numeric_health.get(key, (0, 0))
                    numeric_health[key] = (
                        prev_nan + int(np.isnan(arr).sum()),
                        prev_inf + int(np.isinf(arr).sum()),
                    )

    lines += ["### sequence-length consistency", ""]
    for key in sorted(length_records):
        lengths = length_records[key]
        unique_lengths = sorted(set(lengths))
        status = "OK" if len(unique_lengths) == 1 else "MISMATCH"
        lines.append(f"- `{key}`: {status}, unique lengths={unique_lengths[:10]}, files={len(lengths)}")
    lines.append("")

    lines += ["### aggregate NaN/inf check", ""]
    for key in sorted(numeric_health):
        nan_count, inf_count = numeric_health[key]
        status = "OK" if nan_count == 0 and inf_count == 0 else "CHECK"
        lines.append(f"- `{key}`: {status}, nan={nan_count}, inf={inf_count}")
    lines.append("")

    lines += summarize_scaler(task_dir / "scaler.pkl")
    return lines, parsed


def build_report(args: argparse.Namespace) -> str:
    root = Path(args.root)
    lines = [
        "# ERA5 `processed_timeseries_4` Inspection",
        "",
        f"- root: `{root}`",
        f"- inspected tasks: `{', '.join(args.tasks)}`",
        f"- sample files per task/scaledness: {args.sample_files}",
        "",
    ]
    all_parsed: list[FileSummary] = []
    for task in args.tasks:
        task_lines, parsed = inspect_task(root, task, args.sample_files)
        lines += task_lines
        all_parsed.extend(parsed)

    lines += ["## Global scaler", ""]
    lines += summarize_scaler(root / "global_scaler.pkl")

    lines += ["## Cross-task location overlap", ""]
    by_task: dict[str, set[tuple[float, float]]] = {}
    for item in all_parsed:
        by_task.setdefault(item.task, set()).add((item.lat, item.lon))
    if len(by_task) >= 2:
        tasks = list(by_task)
        overlap = set.intersection(*(by_task[task] for task in tasks))
        union = set.union(*(by_task[task] for task in tasks))
        lines += [
            f"- overlap locations across inspected tasks: {len(overlap)}",
            f"- union locations across inspected tasks: {len(union)}",
        ]
    else:
        lines.append("- fewer than two tasks inspected")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--tasks", nargs="+", default=["task_1", "task_2"])
    parser.add_argument("--sample-files", type=int, default=2)
    parser.add_argument("--out", default="docs/era5_processed_timeseries_4_inspection.md")
    args = parser.parse_args()

    report = build_report(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote report to {out_path}")


if __name__ == "__main__":
    main()
