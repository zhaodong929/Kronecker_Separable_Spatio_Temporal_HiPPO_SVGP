#!/usr/bin/env python3
"""Collect an explicitly separate selected-task scalar FP64 instruction profile.

Wrap this command with the existing W&B supervisor to upload its raw artifacts.
No counter is interpreted as total model FLOPs or extrapolated to other tasks.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normal-run', type=Path, required=True)
    parser.add_argument('--profile-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', required=True, help='Explicit comma-separated zero-based task IDs')
    parser.add_argument('--csv', type=Path, help='Existing NCU long-form raw CSV; omit when executing command')
    parser.add_argument('--range-journal', type=Path, help='Existing HIPPO_PROFILE_JOURNAL; required with --csv')
    parser.add_argument('--profiler-seconds', type=float, help='Measured profiler wall time; required with --csv')
    parser.add_argument('--raw-report', type=Path, help='Optional NCU report, retained as gzip for artifact upload')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from benchmarks.task_stream.profiling import selected_tasks, parse_ncu_csv, ncu_command
    from benchmarks.task_stream.measurement import profile_record
    from benchmarks.three_domain.tracking import emit
    normal, profiled, output = map(Path.resolve, [args.normal_run, args.profile_run, args.output])
    if normal == profiled or output in (normal, profiled):
        parser.error('Normal, profiling worker and counter-output directories must be distinct')
    if output.exists() and any(p.name != 'tracking' for p in output.iterdir()):
        raise FileExistsError('Use a fresh counter output directory')
    output.mkdir(parents=True, exist_ok=True)
    nc = json.loads((normal/'configuration.json').read_text())
    nr = json.loads((normal/'result.json').read_text())
    if nr.get('status') != 'completed':
        raise ValueError('Normal reference run must have completed')
    if nr.get('ordinary_latency_valid') is not True:
        raise ValueError('Normal reference must be a non-profiled timing execution')
    tasks = selected_tasks(args.tasks, nc['tasks'])
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    raw = output/'ncu-raw.csv'
    journal = output/'profile-ranges.jsonl'
    if args.csv:
        if command or args.range_journal is None or args.profiler_seconds is None:
            parser.error('CSV ingestion requires journal and profiler duration, with no worker command')
        shutil.copyfile(args.csv, raw)
        shutil.copyfile(args.range_journal, journal)
        seconds = args.profiler_seconds
        acquisition = {'mode': 'existing_csv'}
    else:
        if not command:
            parser.error('Provide either --csv or a separate profiling worker command after --')
        env = dict(os.environ, HIPPO_PROFILE_TASKS=args.tasks, HIPPO_PROFILE_JOURNAL=str(journal))
        invocation = ncu_command(command, raw)
        (output/'acquisition.json').write_text(json.dumps(dict(command=invocation, tasks=tasks), indent=2))
        started = time.monotonic()
        with (output/'profiler-worker.log').open('w') as log:
            subprocess.run(invocation, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        seconds = time.monotonic()-started
        acquisition = {'mode': 'ncu_kernel_replay', 'command': invocation}
    pc = json.loads((profiled/'configuration.json').read_text())
    pr = json.loads((profiled/'result.json').read_text())
    if pr.get('status') != 'completed':
        raise ValueError('Profiling worker must have completed')
    if pr.get('ordinary_latency_valid') is not False or pr.get('profiled_task_ids') != list(tasks):
        raise ValueError('Profiling execution and requested task ranges disagree')
    for key in ('protocol_sha256', 'configuration'):
        if pc.get(key) != nc.get(key):
            raise ValueError(f'Normal/profile run {key} mismatch')
    for key in ('source_commit', 'source_sha256', 'configuration_sha256', 'selection_features_sha256'):
        if not nc.get('provenance', {}).get(key) or pc.get('provenance', {}).get(key) != nc['provenance'][key]:
            raise ValueError(f'Normal/profile source/configuration proof mismatch: {key}')
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    if (len(rows) != len(tasks) or {r['task'] for r in rows} != set(tasks) or
            any(r['status'] != 'complete' or r.get('range') != 'hippo_task_online' for r in rows)):
        raise ValueError('Range journal does not prove exactly the selected completed task ranges')
    counters = profile_record(parse_ncu_csv(raw), provenance=digest(raw), profiled_task_ids=list(tasks),
        expected_task_ids=list(range(nc['tasks'])), replay_policy='original_launch_ids_deduplicated',
        profiler_seconds=seconds, timing_run_id=str(normal), profile_run_id=str(profiled))
    counters.update(main_table_admitted=False, ordinary_latency_valid=False,
        replay_accounting='NCU original launch metric rows; duplicate launch+metric rejected, never silently deduplicated',
        protocol_sha256=nc['protocol_sha256'], source_commit=nc['provenance']['source_commit'],
        configuration_sha256=nc['provenance']['configuration_sha256'], acquisition=acquisition,
        normal_configuration_sha256=digest(normal/'configuration.json'),
        profile_configuration_sha256=digest(profiled/'configuration.json'),
        raw_csv_sha256=digest(raw), range_journal_sha256=digest(journal))
    if args.raw_report:
        with args.raw_report.open('rb') as source, gzip.open(output/'ncu-report.gz', 'wb') as dest:
            shutil.copyfileobj(source, dest)
        counters['raw_report_sha256'] = digest(args.raw_report)
    (output/'arithmetic.json').write_text(json.dumps(counters, indent=2))
    emit('profiling', len(tasks), counters)
    print(json.dumps(counters))


if __name__ == '__main__':
    main()
