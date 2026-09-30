import json
from pathlib import Path
import pytest
from scripts.run_task_budget_campaign import (run_worker, read_manifest, sha256, canonical_configuration,
                                             verify_source, InputProofs)
from benchmarks.task_stream.provenance import source_identity


def setup(tmp_path, count=7):
    root = tmp_path/'source'; root.mkdir()
    (root/'SOURCE_COMMIT').write_text('a'*40)
    prepared = tmp_path/'prepared'; prepared.mkdir()
    for name in ('manifest.json', 'stream.npz', 'selection-stream.npz', 'features.npz'):
        (prepared/name).write_text('{}')
    tasks = [dict(id=f'task-{i}', dataset='covid', split_seed=5, stage='validation', prepared=str(prepared),
        output=str(tmp_path/f'candidate-{i}'), configuration=dict(method='kronhippo_svgp',
            initial_iterations=2, learning_rate=.01, device='cuda')) for i in range(count)]
    kwargs = dict(tasks=tasks, manifest_hash='manifest-hash', worker_index=0, state_dir=tmp_path/'state',
        campaign='test', compute_root=tmp_path/'compute', expected_commit='a'*40,
        expected_hash=source_identity(root), root=root)
    return tasks, kwargs


def complete(command, log_path):
    attempt = log_path.parent
    proof = json.loads((attempt/'proof.json').read_text())
    run = attempt/'run'; run.mkdir()
    for name in ('predictions.npz', 'task-metrics.json', 'configuration.json'):
        (run/name).write_text('{}')
    (run/'artifacts.json').write_text(json.dumps({name: {'sha256': sha256(run/name)}
        for name in ('predictions.npz', 'task-metrics.json', 'configuration.json')}))
    (run/'result.json').write_text(json.dumps(dict(status='completed', ordinary_latency_valid=True,
        metrics={'nlpd': 2.}, provenance=dict(proof, stage='validation'))))
    assert command[-2:] == ['--stage', 'validation']
    spec = json.loads((attempt/'spec.json').read_text())
    assert spec['entity'] == 'harrisonzhu' and spec['project'] == 'KronHiPPO-STGP'
    return 0


def test_assignment_resume_and_tampered_outputs_create_new_preserved_attempt(tmp_path):
    tasks, kwargs = setup(tmp_path)
    calls = []
    def launch(command, log_path):
        calls.append(log_path); return complete(command, log_path)
    assert run_worker(**kwargs, launch=launch) == 0
    assert len(calls) == 3
    assert all((Path(tasks[i]['output'])/'latest.json').exists() for i in (0, 3, 6))
    assert not (Path(tasks[1]['output'])/'latest.json').exists()
    assert run_worker(**kwargs, launch=launch) == 0 and len(calls) == 3
    first = calls[0].parent
    (first/'run/predictions.npz').write_text('corrupted artifact')
    assert run_worker(**kwargs, launch=launch) == 0 and len(calls) == 4
    assert first.exists() and calls[-1].parent != first
    assert len(list((Path(tasks[0]['output'])/'attempts').iterdir())) == 2


def test_failure_continues_remaining_tasks_and_retry_keeps_history(tmp_path):
    tasks, kwargs = setup(tmp_path)
    calls = []
    def launch(command, log_path):
        calls.append(log_path)
        return 7 if len(calls) == 1 else complete(command, log_path)
    assert run_worker(**kwargs, launch=launch) == 1
    status = json.loads((kwargs['state_dir']/'worker-0.json').read_text())
    assert status['status'] == 'failed' and len(status['completed']) == 2 and len(status['failed']) == 1
    assert run_worker(**kwargs, launch=complete) == 0
    assert (calls[0].parent/'terminal.json').exists()
    assert json.loads((calls[0].parent/'terminal.json').read_text())['exit_code'] == 7


def test_source_guard_and_input_mutation_fail_before_launch(tmp_path):
    tasks, kwargs = setup(tmp_path)
    (kwargs['root']/'SOURCE_COMMIT').write_text('b'*40)
    with pytest.raises(ValueError, match='SOURCE_COMMIT'):
        run_worker(**kwargs, launch=lambda *a: pytest.fail('must not launch'))
    (kwargs['root']/'SOURCE_COMMIT').write_text('a'*40)
    with pytest.raises(ValueError, match='content hash'):
        verify_source(kwargs['root'], 'a'*40, 'wrong')
    cache = InputProofs(); cache.get(tasks[0]['prepared'])
    (Path(tasks[0]['prepared'])/'features.npz').write_text('changed input')
    with pytest.raises(ValueError, match='inputs changed'):
        cache.get(tasks[0]['prepared'])


def test_manifest_rejects_final_runs_and_graceful_time_limit_preserves_pending(tmp_path):
    tasks, kwargs = setup(tmp_path)
    path = tmp_path/'manifest.json'
    tasks[0]['stage'] = 'final'; path.write_text(json.dumps(tasks))
    with pytest.raises(ValueError, match='validation tasks only'):
        read_manifest(path)
    tasks[0]['stage'] = 'validation'
    assert run_worker(**kwargs, launch=lambda *a: pytest.fail('time limit before attempt'), max_seconds=1e-12) == 75
    status = json.loads((kwargs['state_dir']/'worker-0.json').read_text())
    assert status['status'] == 'pending' and status['pending'] == ['task-0', 'task-3', 'task-6']
