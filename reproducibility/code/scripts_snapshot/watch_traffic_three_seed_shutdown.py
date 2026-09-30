#!/usr/bin/env python3
"""Stop after formal seeds 1--3, summarize, and shut down the Windows host."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/traffic/formal_locked_sm_q2_road_context_v1"
METHODS = (
    "kronhippo_stgp",
    "kron_stgp",
    "joint_fixed_global",
    "decoupled_fixed_global",
    "decoupled_changing_diagnostic",
    "ignnk",
)


def status(seed: int, method: str, output: Path) -> str | None:
    path = output / "pems_bay" / "nowcast" / method / f"seed{seed}" / "status.json"
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status"))
    except (json.JSONDecodeError, OSError):
        return None


def seed_complete(seed: int, output: Path) -> bool:
    for method in METHODS:
        accepted = {"diverged"} if method == "decoupled_changing_diagnostic" else {"complete"}
        if status(seed, method, output) not in accepted:
            return False
        directory = output / "pems_bay" / "nowcast" / method / f"seed{seed}"
        if not (directory / "result.json").exists() or not (directory / "predictions.npz").exists():
            return False
    return True


def matching_processes(needles: tuple[str, ...]) -> list[int]:
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if all(needle in command for needle in needles):
            matches.append(int(entry.name))
    return matches


def terminate_seed4_queue(output: Path) -> list[int]:
    pids = set(matching_processes(("run_traffic_locked_five_split.py", "--seeds 2 4")))
    pids.update(matching_processes((str(output), "seed4")))
    pids.update(matching_processes(("pems_bay_seed4_spatial_split.json",)))
    for pid in sorted(pids, reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    return sorted(pids)


def write_record(output: Path, payload: dict[str, object]) -> None:
    (output / "THREE_SEED_STOP_STATUS.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def request_shutdown() -> None:
    subprocess.run(["sync"], check=False)
    subprocess.run(
        [
            "/mnt/c/Windows/System32/shutdown.exe",
            "/s",
            "/t",
            "60",
            "/c",
            "PEMS-BAY three-seed run stopped; available results were saved.",
        ],
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--shutdown-host", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    started = time.monotonic()
    stopped: list[int] = []

    while not seed_complete(2, output):
        if time.monotonic() - started > args.max_hours * 3600:
            write_record(
                output,
                {
                    "status": "timeout_before_seed2",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "host_shutdown_requested": bool(args.shutdown_host),
                },
            )
            if args.shutdown_host:
                request_shutdown()
            return
        time.sleep(args.poll_seconds)

    stopped.extend(terminate_seed4_queue(output))

    while not seed_complete(3, output):
        stopped.extend(terminate_seed4_queue(output))
        if time.monotonic() - started > args.max_hours * 3600:
            write_record(
                output,
                {
                    "status": "timeout_before_seed3",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "host_shutdown_requested": bool(args.shutdown_host),
                },
            )
            if args.shutdown_host:
                request_shutdown()
            return
        time.sleep(args.poll_seconds)

    stopped.extend(terminate_seed4_queue(output))
    summary = subprocess.run(
        [
            str(ROOT / ".venv_cuda128" / "bin" / "python"),
            "scripts/summarize_traffic_locked_five_split.py",
            "--seeds",
            "1",
            "2",
            "3",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    payload = {
        "status": "complete" if summary.returncode == 0 else "summary_failed",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "formal_seeds": [1, 2, 3],
        "seed2_complete": True,
        "seed3_complete": True,
        "stopped_seed4_processes": sorted(set(stopped)),
        "summary_returncode": summary.returncode,
        "summary_stdout_tail": summary.stdout[-4000:],
        "summary_stderr_tail": summary.stderr[-4000:],
        "host_shutdown_requested": bool(args.shutdown_host),
    }
    write_record(output, payload)
    if args.shutdown_host:
        request_shutdown()


if __name__ == "__main__":
    main()
