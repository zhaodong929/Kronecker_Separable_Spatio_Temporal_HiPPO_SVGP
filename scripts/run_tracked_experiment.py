#!/usr/bin/env python3
"""Run any isolated baseline interpreter with durable local and W&B records.

The supervisor, not the model, owns W&B. Every attempt has a distinct ID;
checkpoint continuation is linked via logical_id, never an offline ID collision.
No whole environment or credentials are uploaded. Input arrays remain local;
their SHA256, metadata, and split identities go into the provenance artifact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.three_domain.tracking import emit, numeric_metrics, scalar_tree


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture(command):
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=30, cwd=ROOT)
        return dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(error=str(error))


def journal_rows(path, offset):
    """Never consume a partial trailing line from a live writer."""
    if not path.exists():
        return [], offset
    rows = []
    with path.open("rb") as stream:
        stream.seek(offset)
        while True:
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                break
            rows.append(json.loads(line))
            offset = stream.tell()
    return rows, offset


def process_tree_memory(pid):
    """Linux worker tree RSS, not the small logging supervisor's RSS."""
    rss, threads, count = 0, 0, 0
    todo, seen = [pid], set()
    while todo:
        child = todo.pop()
        if child in seen:
            continue
        seen.add(child)
        try:
            root = Path('/proc') / str(child)
            fields = dict(line.split(':', 1) for line in (root / 'status').read_text().splitlines() if ':' in line)
            rss += int(fields.get('VmRSS', '0 kB').split()[0])*1024
            threads += int(fields.get('Threads', '0'))
            count += 1
            todo.extend(map(int, (root / 'task' / str(child) / 'children').read_text().split()))
        except (OSError, ValueError):
            continue
    return dict(worker_tree_rss_bytes=rss, worker_tree_threads=threads, worker_processes=count)


def validate_outputs(output, expected_steps, expected_sites):
    import numpy as np
    result = json.loads((output / "result.json").read_text())
    with np.load(output / "predictions.npz", allow_pickle=False) as arrays:
        y, mean, var = (arrays[k] for k in ("y_true", "pred_mean", "pred_var"))
        shape = (expected_steps, expected_sites)
        if y.shape != shape or mean.shape != shape or var.shape != shape:
            raise ValueError(f"Prediction shape mismatch; expected {shape}")
        if not all(np.isfinite(x).all() for x in (y, mean, var)) or not (var > 0).all():
            raise ValueError("Nonfinite predictions/targets or nonpositive variance")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("online", "offline"), default="online")
    p.add_argument("--poll-seconds", type=float, default=30)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        p.error("An experiment command is required after --")
    spec = json.loads(a.spec.read_text())
    required = {"entity", "project", "campaign", "dataset", "method", "split_seed", "training_seed", "stage"}
    if required - spec.keys():
        raise ValueError(f"Missing spec fields: {sorted(required-spec.keys())}")
    if spec["method"] not in {"kronhippo_svgp", "osgpr", "ohsvgp", "st_svgp", "mgpvae"}:
        raise ValueError("Method excluded by baseline-selection policy")
    if spec["stage"] not in {"validation", "qualification", "final", "integration", "ablation"}:
        raise ValueError("Unknown experiment stage")
    if spec["stage"] in {"final", "ablation"}:
        if not spec.get("qualification_record"):
            raise ValueError("Final runs require an explicit qualification record")
        qualification = json.loads(Path(spec["qualification_record"]).read_text())
        if qualification.get("status") != "passed" or qualification.get("method") != spec["method"] or qualification.get("dataset") != spec["dataset"]:
            raise ValueError("Qualification record does not qualify this method/dataset")
    no_hidden_release = any(json.loads(Path(f).read_text()).get('hidden_label_policy')=='never released'
        for f in spec.get('input_files',[]) if Path(f).suffix=='.json')
    if no_hidden_release and ('--delayed-observations' in command or '--delayed-observation-blocks' in command):
        raise ValueError('A no-release protocol cannot enable delayed hidden observations')
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    attempt = uuid.uuid4().hex[:12]
    record = output / "tracking" / attempt
    record.mkdir(parents=True, exist_ok=False)
    logical = "/".join(str(spec[k]) for k in ("dataset", "method", "split_seed", "training_seed", "stage"))
    provenance = dict(spec=spec, command=command, attempt=attempt, logical_id=logical,
        git=capture(["git", "rev-parse", "HEAD"]),
        git_status=capture(["git", "status", "--porcelain"]),
        scheduler={k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES") if k in os.environ},
        hardware=capture(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"]),
        input_files={str(Path(f).resolve()): digest(f) for f in spec.get("input_files", [])},
        input_metadata={str(Path(f).resolve()): json.loads(Path(f).read_text()) for f in spec.get("input_files", []) if Path(f).suffix == '.json'},
        worker_packages=capture([spec.get("worker_python", command[0]), "-c",
            "import importlib.metadata,json; print(json.dumps(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())))"]))
    (record / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (record / "code.diff").write_text(capture(["git", "diff", "HEAD"] ).get("stdout", ""))
    with tarfile.open(record / "code_snapshot.tar.gz", "w:gz") as archive:
        for folder in ("scripts", "stvgp_kronecker", "benchmarks/three_domain", "benchmarks/task_stream", "baselines", "slurm/fair_three_domain", "slurm/task_stream", "tests"):
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and not path.is_symlink() and path.suffix in {".py", ".sh", ".sbatch"}:
                    archive.add(path, arcname=str(path.relative_to(ROOT)))
    # W&B initialization is isolated from the child's numerical environment.
    import wandb
    start = time.monotonic()
    kwargs = dict(entity=spec["entity"], project=spec["project"], id=attempt,
        name=logical+"/"+attempt, group=spec["campaign"], job_type=spec["stage"],
        tags=[spec["dataset"], spec["method"], spec["stage"]],
        config=scalar_tree(provenance), dir=str(record), save_code=False,
        settings=wandb.Settings(init_timeout=30))
    try:
        run = wandb.init(mode=a.mode, **kwargs)
    except Exception as error:
        (record / "wandb-init-error.txt").write_text(type(error).__name__+": "+str(error))
        run = wandb.init(mode="offline", **kwargs)
    (record / "wandb.json").write_text(json.dumps(dict(id=attempt, url=run.url,
        mode=run.settings.mode, entity=spec["entity"], project=spec["project"]), indent=2))
    for phase in ("train", "validation", "online", "refit", "system"):
        run.define_metric(phase+"/step")
        run.define_metric(phase+"/*", step_metric=phase+"/step")
    run.summary["logical_id"] = logical
    run.summary["status"] = "running"
    run.summary["main_table_admitted"] = False
    event_path = record / "events.jsonl"
    os.environ['HIPPO_EVENT_PATH'] = str(event_path)
    env = dict(os.environ, HIPPO_EVENT_PATH=str(event_path), PYTHONUNBUFFERED="1")
    offset, dashboard_failed = 0, False
    last_by_phase = {}
    def consume(final=False):
        nonlocal offset, dashboard_failed
        rows, offset = journal_rows(event_path, offset)
        # Preserve all local rows; dashboard receives up to one point per phase
        # per polling window, independent of training throughput.
        for row in rows:
            last_by_phase[row["phase"]] = row
        if dashboard_failed:
            return
        try:
            for phase, row in last_by_phase.items():
                run.log({phase+"/step": row["step"], **numeric_metrics(row["metrics"], phase),
                         "wall/elapsed_seconds": time.monotonic()-start})
            last_by_phase.clear()
        except Exception as error:
            dashboard_failed = True
            (record / "wandb-log-error.txt").write_text(type(error).__name__+": "+str(error))
    with (record / "stdout.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        def terminate(signum, frame):
            os.killpg(process.pid, signum)
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        while process.poll() is None:
            time.sleep(a.poll_seconds)
            emit('system', int(time.monotonic()-start), process_tree_memory(process.pid))
            consume()
        code = process.returncode
    consume(final=True)
    status, validation_error = ("process_complete" if code == 0 else "failed"), None
    if code == 0 and spec.get("expected_steps") is not None:
        try:
            result = validate_outputs(output, spec["expected_steps"], spec["expected_sites"])
            if no_hidden_release and result.get('delayed_observation_rows') != 0:
                raise ValueError('No-release protocol absorbed hidden observations')
            if spec.get("hidden_delay_steps") == 1:
                expected = (spec["expected_steps"]-1)*spec["expected_sites"]
                if result.get("delayed_observation_rows") != expected:
                    raise ValueError("Incorrect delayed observation count")
            status = "completed_and_verified"
            run.summary.update(numeric_metrics(result, "result"))
        except Exception as error:
            status, validation_error, code = "invalid_outputs", str(error), 2
    terminal = dict(status=status, exit_code=code, output_validation_error=validation_error,
        elapsed_seconds=time.monotonic()-start, main_table_admitted=False,
        dashboard_failed=dashboard_failed, attempt=attempt, logical_id=logical)
    (record / "terminal.json").write_text(json.dumps(terminal, indent=2))
    run.summary.update(terminal)
    files = {}
    artifact = wandb.Artifact("run-"+attempt, type="experiment-record",
                              metadata=dict(logical_id=logical, status=status))
    # Upload completed files, not changing checkpoints; no symlinks, raw data,
    # arbitrary directory traversal, credential files, or W&B internal files.
    allowed = {".json", ".jsonl", ".csv", ".npz", ".pt", ".log", ".txt", ".diff", ".gz"}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix not in allowed:
            continue
        rel = path.relative_to(output)
        if "wandb" in rel.parts or "wandb-staging" in rel.parts:
            continue
        files[str(rel)] = dict(sha256=digest(path), bytes=path.stat().st_size)
        artifact.add_file(str(path), name=str(rel))
    (record / "artifact_manifest.json").write_text(json.dumps(files, indent=2))
    artifact.add_file(str(record / "artifact_manifest.json"), name="artifact_manifest.json")
    try:
        run.log_artifact(artifact)
        run.finish(exit_code=code)
    except Exception as error:
        (record / "wandb-finalize-error.txt").write_text(type(error).__name__+": "+str(error))
        terminal["tracking_sync_required"] = True
        (record / "terminal.json").write_text(json.dumps(terminal, indent=2))
    print(json.dumps(dict(terminal, tracking_directory=str(record))), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
