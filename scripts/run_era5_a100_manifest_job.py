#!/usr/bin/env python3
"""Run one resumable Slurm job from a JSONL manifest.

Each manifest record contains an ``argv`` list (or the compatible ``command``
key), a working directory, and one or more completion artifacts. The command
is passed directly to ``subprocess.Popen`` with ``shell=False``; manifest
values are never interpreted by a shell.

The array index is zero based and is read from ``SLURM_ARRAY_TASK_ID`` unless
``--index`` is supplied. A record is complete only when every declared
completion artifact exists and is non-empty.

The runner intentionally uses only the Python standard library so that the
login node can build and inspect manifests without importing Torch, TensorFlow,
or JAX.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Iterable


NONFINITE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(?:nan|inf|infinity|non[- ]?finite)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
OOM_TOKENS = (
    "out of memory",
    "out_of_memory",
    "resourceexhausted",
    "oom-kill",
    "oom kill",
    "oom_kill",
    "cuda oom",
    "cuda error: out of memory",
)


class _NonFiniteJSON(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a strict JSON status record without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _path(value: str | os.PathLike[str], base: Path) -> Path:
    candidate = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return candidate if candidate.is_absolute() else (base / candidate)


def _as_string_list(value: Any, field: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not value or not all(
        isinstance(item, (str, int, float, bool)) for item in value
    ):
        raise ValueError(f"Manifest field {field!r} must be a non-empty argv list")
    return [str(item) for item in value]


def read_manifest_entry(manifest: Path, index: int) -> dict[str, Any]:
    """Return the non-blank JSONL record at zero-based ``index``."""

    if index < 0:
        raise ValueError(f"Array index must be non-negative, got {index}")
    record_index = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if record_index == index:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {manifest}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Manifest record {index} at {manifest}:{line_number} is not an object"
                    )
                return record
            record_index += 1
    raise IndexError(
        f"Array index {index} is outside {manifest} ({record_index} records)"
    )


def manifest_count(manifest: Path) -> int:
    """Count non-blank JSONL records without parsing command contents."""

    with manifest.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def record_argv(record: dict[str, Any]) -> list[str]:
    value = record.get("argv", record.get("command"))
    return _as_string_list(value, "argv")


def record_artifacts(record: dict[str, Any]) -> list[str]:
    value = record.get("complete", record.get("expected", []))
    if isinstance(value, str):
        return [value]
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Manifest field 'complete' must be a string list")
    return list(value)


def scheduler_state() -> str:
    """Read final Slurm state when sacct is available, best effort."""

    job_id = os.environ.get("SLURM_JOB_ID")
    sacct = shutil.which("sacct")
    if not job_id or not sacct:
        return ""
    try:
        result = subprocess.run(
            [sacct, "-X", "-n", "-P", "-j", job_id, "--format=State"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    states = [
        line.strip().split("|")[0]
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    return states[-1].upper() if states else ""


def _json_has_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_json_has_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_json_has_nonfinite(item) for item in value)
    return False


def _json_constant(_: str) -> Any:
    raise _NonFiniteJSON("non-finite JSON constant")


def artifact_has_nonfinite(path: Path) -> bool:
    """Detect non-finite metrics in common text artifacts."""

    if not path.is_file():
        return False
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(
                path.read_text(encoding="utf-8"), parse_constant=_json_constant
            )
            return _json_has_nonfinite(payload)
        if path.suffix.lower() in {".csv", ".tsv", ".txt", ".log"}:
            return (
                NONFINITE_TOKEN.search(
                    path.read_text(encoding="utf-8", errors="ignore")
                )
                is not None
            )
    except _NonFiniteJSON:
        return True
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return False
    return False


def log_has_nonfinite(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return NONFINITE_TOKEN.search(text) is not None


def classify_failure(
    *,
    log_path: Path,
    returncode: int | None,
    timed_out: bool = False,
    scheduler: str = "",
    artifacts: Iterable[Path] = (),
) -> str:
    """Classify a failed command as oom, timeout, signal, nonfinite, or error."""

    scheduler_upper = scheduler.upper()
    if timed_out or scheduler_upper.startswith("TIMEOUT"):
        return "timeout"
    if scheduler_upper.startswith("OUT_OF_MEMORY") or scheduler_upper.startswith("OUT_OF_MEM"):
        return "oom"
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        log_text = ""
    if any(token in log_text for token in OOM_TOKENS) or returncode == 137:
        return "oom"
    if returncode is not None and returncode < 0:
        return "signal"
    if returncode is not None and returncode >= 128 and returncode - 128 in {
        signal.SIGHUP,
        signal.SIGINT,
        signal.SIGTERM,
        signal.SIGKILL,
    }:
        return "signal"
    if NONFINITE_TOKEN.search(log_text) or any(
        artifact_has_nonfinite(path) for path in artifacts
    ):
        return "nonfinite"
    return "error"


def terminate_group(process: subprocess.Popen[str], sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(sig)
        except OSError:
            pass


def _safe_job_id(record: dict[str, Any], index: int) -> str:
    value = record.get("id", record.get("job", f"array-{index}"))
    text = str(value)
    return text if text else f"array-{index}"


def run_record(
    record: dict[str, Any],
    *,
    index: int,
    manifest: Path,
    repo: Path,
    force: bool = False,
) -> int:
    job_id = _safe_job_id(record, index)
    argv = record_argv(record)
    artifacts = [_path(item, repo) for item in record_artifacts(record)]
    cwd = _path(record.get("cwd", repo), repo)
    if not cwd.is_dir():
        raise FileNotFoundError(f"Job working directory does not exist: {cwd}")

    output_value = record.get("output_dir")
    if output_value is None:
        output_dir = (
            artifacts[0].parent
            if artifacts
            else manifest.parent / "jobs" / f"{index:06d}"
        )
    else:
        output_dir = _path(output_value, repo)
    status_path = _path(record.get("status_path", output_dir / "status.json"), repo)
    log_path = _path(record.get("log_path", output_dir / "run.log"), repo)
    complete = bool(artifacts) and all(_nonempty(path) for path in artifacts)

    common = {
        "schema_version": 1,
        "job": job_id,
        "manifest": str(manifest),
        "array_index": index,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "argv": argv,
        "cwd": str(cwd),
        "output_dir": str(output_dir),
        "completion_artifacts": [str(path) for path in artifacts],
    }
    if complete and not force:
        payload = {
            **common,
            "status": "skipped",
            "classification": "complete",
            "reason": "all completion artifacts already exist",
            "finished_at": utc_now(),
        }
        if not status_path.exists():
            atomic_json(status_path, payload)
        print(f"SKIP complete {job_id}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(shlex.join(argv) + "\n", encoding="utf-8")
    (output_dir / "manifest_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    command_env = os.environ.copy()
    overrides = record.get("env", {})
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError(f"Manifest env for {job_id} must be an object")
        for key, value in overrides.items():
            if not isinstance(key, str):
                raise ValueError(f"Manifest env key for {job_id} is not a string")
            if value is None:
                command_env.pop(key, None)
            else:
                command_env[key] = str(value)
    command_env["ERA5_A100_MANIFEST_ID"] = job_id
    timeout_value = record.get("timeout_seconds", 0)
    timeout = float(timeout_value) if timeout_value else None
    if timeout is not None and timeout < 0:
        raise ValueError(f"Negative timeout for {job_id}: {timeout}")

    started = time.perf_counter()
    started_at = utc_now()
    returncode: int | None = None
    timed_out = False
    launch_error: str | None = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"job={job_id}\n")
        log.write(f"argv={shlex.join(argv)}\n")
        log.flush()
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(cwd),
                env=command_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
                text=True,
            )
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_group(process, signal.SIGTERM)
                try:
                    returncode = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    terminate_group(process, signal.SIGKILL)
                    returncode = process.wait()
        except OSError as exc:
            launch_error = str(exc)
            log.write(f"launch error: {launch_error}\n")

    scheduler = scheduler_state()
    elapsed = time.perf_counter() - started
    complete = returncode == 0 and bool(artifacts) and all(
        _nonempty(path) for path in artifacts
    )
    if complete and not log_has_nonfinite(log_path) and not any(
        artifact_has_nonfinite(path) for path in artifacts
    ):
        status = "complete"
        classification = "complete"
    else:
        status = classification = classify_failure(
            log_path=log_path,
            returncode=returncode,
            timed_out=timed_out,
            scheduler=scheduler,
            artifacts=artifacts,
        )
        if launch_error:
            classification = status = "error"
    payload = {
        **common,
        "status": status,
        "classification": classification,
        "failure_class": None if status in {"complete", "skipped"} else classification,
        "started_at": started_at,
        "finished_at": utc_now(),
        "wall_seconds": elapsed,
        "timeout_seconds": timeout,
        "returncode": returncode,
        "scheduler_state": scheduler,
        "missing_artifacts": [str(path) for path in artifacts if not _nonempty(path)],
        "launch_error": launch_error,
        "environment_overrides": sorted(str(key) for key in (overrides or {})),
    }
    atomic_json(status_path, payload)
    if status == "complete":
        print(f"COMPLETE {job_id} ({elapsed:.1f}s)")
        return 0
    print(f"{status.upper()} {job_id} ({elapsed:.1f}s)", file=sys.stderr)
    return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--index",
        type=int,
        help="Zero-based manifest record; defaults to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("REPO_ROOT", Path.cwd())),
        help="Base directory for relative paths and default working directory.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = args.manifest.expanduser().resolve()
    repo = args.repo.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    index = args.index
    if index is None:
        raw_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw_index is None:
            raise ValueError("--index or SLURM_ARRAY_TASK_ID is required")
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID: {raw_index!r}") from exc
    record = read_manifest_entry(manifest, index)
    return run_record(record, index=index, manifest=manifest, repo=repo, force=args.force)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, IndexError, ValueError) as exc:
        print(f"manifest runner error: {exc}", file=sys.stderr)
        raise SystemExit(2)
