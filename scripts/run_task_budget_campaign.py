#!/usr/bin/env python3
"""Durable three-worker validation campaign; never selects or launches final runs."""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INPUT_NAMES = ('manifest.json', 'stream.npz', 'selection-stream.npz', 'features.npz')
ENVIRONMENTS = {'kronhippo_svgp': 'env-routeb', 'ohsvgp': 'env-routeb',
                'osgpr': 'env-osgpr', 'st_svgp': 'env-jax-gpu', 'mgpvae': 'env-jax-gpu'}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_source(root, expected_commit, expected_hash):
    from benchmarks.task_stream.provenance import source_identity
    commit_file = Path(root)/'SOURCE_COMMIT'
    if not commit_file.is_file() or commit_file.read_text().strip() != expected_commit:
        raise ValueError('Immutable SOURCE_COMMIT does not match campaign release')
    if source_identity(root) != expected_hash:
        raise ValueError('Immutable source content hash does not match campaign release')


def canonical_configuration(value):
    from benchmarks.task_stream.factory import Configuration
    config = asdict(Configuration(**value))
    if not config['device'].startswith('cuda'):
        raise ValueError('DoC budget campaign requires an allocated GPU configuration')
    return config, hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def read_manifest(path):
    tasks = json.loads(Path(path).read_text())
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Manifest must be a nonempty list of validation tasks')
    ids, outputs = set(), set()
    for task in tasks:
        if not isinstance(task, dict) or any(key not in task for key in
            ('id', 'dataset', 'split_seed', 'configuration', 'prepared', 'output', 'stage')):
            raise ValueError('Manifest task lacks required fields')
        if not isinstance(task['id'], str) or not task['id'] or task['id'] in ids:
            raise ValueError('Task identities must be unique nonempty strings')
        if task['stage'] != 'validation':
            raise ValueError('Budget orchestration permits validation tasks only')
        output = str(Path(task['output']).resolve())
        if output in outputs:
            raise ValueError('Logical output roots must be distinct')
        ids.add(task['id']); outputs.add(output)
        canonical_configuration(task['configuration'])
    return tasks


class InputProofs:
    """Hash each prepared input once per worker; reject mutation while running."""
    def __init__(self):
        self.cache = {}
    def get(self, prepared):
        paths = [Path(prepared).resolve()/name for name in INPUT_NAMES]
        signatures = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)
        key = str(Path(prepared).resolve())
        if key in self.cache:
            old, hashes = self.cache[key]
            if old != signatures:
                raise ValueError('Prepared inputs changed during campaign')
            return hashes
        hashes = {str(path): sha256(path) for path in paths}
        self.cache[key] = signatures, hashes
        return hashes


def verified_completed(attempt, proof):
    """A zero supervisor exit alone is insufficient for reusing a candidate."""
    try:
        result = json.loads((attempt/'run/result.json').read_text())
        if result.get('status') != 'completed' or result.get('ordinary_latency_valid') is not True:
            return False
        provenance = result.get('provenance', {})
        if provenance.get('stage') != 'validation':
            return False
        for key in ('source_commit', 'source_sha256', 'configuration_sha256'):
            if provenance.get(key) != proof[key]:
                return False
        supplied = provenance.get('input_files', {})
        if any(supplied.get(name) != identity for name, identity in proof['input_files'].items()):
            return False
        metric = result['metrics']['nlpd']
        import math
        if not isinstance(metric, (float, int)) or not math.isfinite(metric):
            return False
        artifacts = json.loads((attempt/'run/artifacts.json').read_text())
        for name in ('predictions.npz', 'task-metrics.json', 'configuration.json'):
            if artifacts[name]['sha256'] != sha256(attempt/'run'/name):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_worker(*, tasks, manifest_hash, worker_index, state_dir, campaign, compute_root,
               expected_commit, expected_hash, root=ROOT, max_seconds=None, launch=None):
    if worker_index not in range(3):
        raise ValueError('Exactly three deterministic workers are supported')
    state_dir = Path(state_dir); state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir/f'worker-{worker_index}.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('This deterministic worker is already active')
    proof_cache = InputProofs()
    started = time.monotonic()
    assigned = [(index, task) for index, task in enumerate(tasks) if index % 3 == worker_index]
    status_path = state_dir/f'worker-{worker_index}.json'
    status = dict(status='running', worker_index=worker_index, campaign=campaign,
        manifest_sha256=manifest_hash, source_commit=expected_commit, source_sha256=expected_hash,
        assigned=[task['id'] for _, task in assigned], completed=[], failed=[], skipped=[], pending=[],
        active=None, pid=os.getpid(), slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    stopped = False
    current = None
    previous_handlers = {}
    accepted_state = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if current is not None and current.poll() is None:
            # The W&B supervisor forwards this signal to its numerical process group.
            current.send_signal(signum)
    def persist():
        status['elapsed_seconds'] = time.monotonic()-started
        atomic_json(status_path, status)
    def default_launch(command, log_path):
        nonlocal current
        with log_path.open('w') as log:
            current = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            code = current.wait()
            current = None
            return code
    launcher = launch or default_launch
    try:
        verify_source(root, expected_commit, expected_hash)
        if status_path.exists():
            previous = json.loads(status_path.read_text())
            if any(previous.get(key) != status[key] for key in ('manifest_sha256', 'source_commit', 'source_sha256')):
                raise ValueError('Worker state belongs to a different manifest or source; use a new state directory')
        accepted_state = True
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, stop)
        persist()
        for position, (_, task) in enumerate(assigned):
            if stopped or (max_seconds is not None and time.monotonic()-started >= max_seconds):
                status['pending'] = [t['id'] for _, t in assigned[position:]]
                break
            verify_source(root, expected_commit, expected_hash)
            config, config_hash = canonical_configuration(task['configuration'])
            inputs = proof_cache.get(task['prepared'])
            proof = dict(source_commit=expected_commit, source_sha256=expected_hash,
                         configuration_sha256=config_hash, input_files=inputs)
            logical = Path(task['output']).resolve()
            logical.mkdir(parents=True, exist_ok=True)
            reusable = [attempt for attempt in sorted((logical/'attempts').glob('*'))
                        if attempt.is_dir() and verified_completed(attempt, proof)]
            if reusable:
                status['skipped'].append(dict(id=task['id'], attempt=str(reusable[-1])))
                atomic_json(logical/'latest.json', dict(status='completed', attempt=str(reusable[-1]), proof=proof))
                persist(); continue
            attempt = logical/'attempts'/uuid.uuid4().hex
            attempt.mkdir(parents=True, exist_ok=False)
            atomic_json(attempt/'configuration.json', config)
            worker_python = str(Path(compute_root)/ENVIRONMENTS[config['method']]/'bin/python')
            spec = dict(entity='harrisonzhu', project='KronHiPPO-STGP', campaign=campaign,
                dataset=task['dataset'], method=config['method'], split_seed=task['split_seed'],
                training_seed=config['seed'], stage='validation', source_commit=expected_commit,
                source_sha256=expected_hash, worker_python=worker_python,
                study=task.get('study'), candidate_id=task['id'], main_table_admitted=False,
                input_files=[str(attempt/'configuration.json'), *inputs.keys()],
                timing_scope='one allocated A30 GPU; full initial-period candidate fit and validation')
            atomic_json(attempt/'spec.json', spec)
            atomic_json(attempt/'proof.json', proof)
            output = attempt/'run'
            command = [str(Path(compute_root)/'env-tracking/bin/python'),
                str(Path(root)/'scripts/run_tracked_experiment.py'), '--spec', str(attempt/'spec.json'),
                '--output', str(output), '--', worker_python, str(Path(root)/'scripts/run_task_stream.py'),
                '--prepared', str(Path(task['prepared']).resolve()), '--configuration', str(attempt/'configuration.json'),
                '--output', str(output), '--stage', 'validation']
            status['active'] = dict(id=task['id'], attempt=str(attempt)); persist()
            error = None
            try:
                code = launcher(command, attempt/'supervisor.log')
                valid = code == 0 and verified_completed(attempt, proof)
            except Exception as exception:
                code, valid, error = -1, False, f'{type(exception).__name__}: {exception}'
            terminal = dict(status='completed' if valid else 'failed', id=task['id'],
                attempt=str(attempt), exit_code=code, output_verified=valid, error=error, proof=proof)
            atomic_json(attempt/'terminal.json', terminal)
            atomic_json(logical/'latest.json', terminal)
            status['completed' if valid else 'failed'].append(dict(id=task['id'], attempt=str(attempt)))
            status['active'] = None; persist()
        status['status'] = ('pending' if status['pending'] else 'failed' if status['failed'] else 'completed')
        persist()
        return 75 if status['pending'] else 1 if status['failed'] else 0
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        if accepted_state:
            persist()
        else:
            atomic_json(state_dir/f'rejected-worker-{worker_index}-{uuid.uuid4().hex}.json', status)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--worker-index', type=int, choices=range(3), required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--compute-root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--max-seconds', type=float, help='Graceful stop BETWEEN attempts; exit 75 means pending work')
    args = parser.parse_args()
    if args.max_seconds is not None and (not 0 < args.max_seconds < float('inf')):
        parser.error('max-seconds must be finite and positive')
    return run_worker(tasks=read_manifest(args.manifest), manifest_hash=sha256(args.manifest),
        worker_index=args.worker_index, state_dir=args.state_dir, campaign=args.campaign,
        compute_root=args.compute_root, expected_commit=args.source_commit,
        expected_hash=args.source_sha256, max_seconds=args.max_seconds)


if __name__ == '__main__':
    sys.exit(main())
