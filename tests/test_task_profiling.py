import csv
import json
import pytest
from benchmarks.task_stream.profiling import METRICS, TaskProfiler, parse_ncu_csv, selected_tasks, ncu_command
from benchmarks.task_stream.measurement import ArithmeticCount


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ID', 'Process ID', 'Context', 'Stream', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'])
        for identity, metric, unit, value in rows:
            writer.writerow([identity, '100', '1', '7', 'my kernel', METRICS.get(metric, metric), unit, value])


def test_original_launch_metrics_distinct_rows_sum_once_and_missing_is_not_zero(tmp_path):
    path = tmp_path/'raw.csv'
    rows = [('0', 'dadd', 'inst', '1,024'), ('0', 'dmul', 'inst', '2'), ('0', 'dfma', 'inst', '3'),
            ('1', 'dadd', 'inst', '0'), ('1', 'dmul', 'inst', '5'), ('1', 'dfma', 'inst', '7')]
    write_csv(path, rows)
    result = ArithmeticCount.from_kernels(parse_ncu_csv(path), provenance='test')
    assert result.fp64_add_mul_fma == 1051 and result.kernel_count == 2
    write_csv(path, rows[:-1])
    assert ArithmeticCount.from_kernels(parse_ncu_csv(path), provenance='test').status == 'incomplete'
    write_csv(path, rows + [rows[0]])
    with pytest.raises(ValueError, match='Duplicate'):
        parse_ncu_csv(path)


def test_wrong_units_or_warp_metrics_do_not_become_thread_flops(tmp_path):
    path = tmp_path/'raw.csv'
    write_csv(path, [('0', 'dadd', 'Kinst', '1.024')])
    with pytest.raises(ValueError, match='unit'):
        parse_ncu_csv(path)
    write_csv(path, [('0', 'smsp__inst_executed.sum', 'inst', '100')])
    assert parse_ncu_csv(path) == []
    write_csv(path, [('0', 'dadd', 'inst', '1.5')])
    with pytest.raises(ValueError, match='integers'):
        parse_ncu_csv(path)


def test_selected_ranges_drain_and_journal_complete_only_once(tmp_path, monkeypatch):
    import benchmarks.task_stream.profiling as profiling
    calls = []
    class NVTX:
        def nvtxRangePushA(self, name):
            calls.append(('push', name)); return 0
        def nvtxRangePop(self):
            calls.append(('pop',)); return 0
    monkeypatch.setattr(profiling, 'nvtx_library', lambda: NVTX())
    monkeypatch.setenv('HIPPO_PROFILE_TASKS', '1,3')
    monkeypatch.setenv('HIPPO_PROFILE_JOURNAL', str(tmp_path/'ranges.jsonl'))
    p = TaskProfiler(4)
    with p.range(0, lambda: calls.append(('sync',))):
        pass
    assert not calls
    with p.range(1, lambda: calls.append(('sync',))):
        calls.append(('kernel',))
    assert [row[0] for row in calls] == ['sync', 'push', 'kernel', 'sync', 'pop']
    with pytest.raises(ValueError, match='twice'):
        with p.range(1):
            pass
    with pytest.raises(RuntimeError, match='failure'):
        with p.range(3):
            raise RuntimeError('failure')
    rows = [json.loads(line) for line in (tmp_path/'ranges.jsonl').read_text().splitlines()]
    assert [r['status'] for r in rows] == ['complete', 'failed']
    assert all(r['ordinary_latency_valid'] is False for r in rows)


def test_explicit_profile_requires_valid_tasks_and_nvtx(monkeypatch):
    import benchmarks.task_stream.profiling as profiling
    for text in ('', 'all', '-1', '1,1', '9'):
        with pytest.raises(ValueError):
            selected_tasks(text, 4)
    monkeypatch.delenv('HIPPO_PROFILE_TASKS', raising=False)
    monkeypatch.setattr(profiling, 'nvtx_library', lambda: (_ for _ in ()).throw(RuntimeError('missing NVTX')))
    assert not TaskProfiler(4).enabled
    monkeypatch.setenv('HIPPO_PROFILE_TASKS', '1')
    with pytest.raises(RuntimeError, match='NVTX'):
        TaskProfiler(4)
    command = ncu_command(['python', 'worker.py'], 'raw.csv')
    assert '--print-units' in command and command[-2:] == ['python', 'worker.py']


def test_collector_links_matched_runs_and_preserves_raw_evidence(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    normal, profiled = tmp_path/'normal', tmp_path/'profiled'
    configuration = dict(tasks=4, protocol_sha256='protocol', configuration={'method': 'st_svgp'},
        provenance={'source_commit': 'source', 'source_sha256': 'source-hash',
                    'configuration_sha256': 'config', 'selection_features_sha256': 'features-hash'})
    for directory in (normal, profiled):
        directory.mkdir()
        (directory/'configuration.json').write_text(json.dumps(configuration))
        (directory/'result.json').write_text(json.dumps(dict(status='completed',
            ordinary_latency_valid=directory == normal, profiled_task_ids=[] if directory == normal else [1])))
    csv_path = tmp_path/'raw.csv'
    write_csv(csv_path, [('0', name, 'inst', str(value)) for name, value in [('dadd', 2), ('dmul', 3), ('dfma', 4)]])
    journal = tmp_path/'ranges.jsonl'
    journal.write_text(json.dumps(dict(task=1, status='complete', range='hippo_task_online'))+'\n')
    command = [sys.executable, str(Path(__file__).resolve().parents[1]/'scripts/collect_task_flops.py'),
        '--normal-run', str(normal), '--profile-run', str(profiled), '--output', str(tmp_path/'collected'),
        '--tasks', '1', '--csv', str(csv_path), '--range-journal', str(journal), '--profiler-seconds', '12.5']
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads((tmp_path/'collected/arithmetic.json').read_text())
    assert result['fp64_add_mul_fma'] == 13
    assert result['task_coverage'] == .25 and result['profiler_seconds'] == 12.5
    assert result['ordinary_latency_valid'] is False
    assert (tmp_path/'collected/ncu-raw.csv').read_bytes() == csv_path.read_bytes()
    configuration['provenance']['source_commit'] = 'different'
    (profiled/'configuration.json').write_text(json.dumps(configuration))
    command[command.index('--output') + 1] = str(tmp_path/'mismatch')
    failure = subprocess.run(command, capture_output=True, text=True)
    assert failure.returncode != 0 and 'proof mismatch' in failure.stderr


def test_pipeline_nvtx_scope_contains_release_and_prediction_but_not_scoring_or_artifacts(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import numpy as np
    import benchmarks.task_stream.pipeline as pipeline
    from benchmarks.task_stream.protocol import TaskStream
    from benchmarks.three_domain.evaluation import gaussian_scores
    active = False
    observed = []
    class Profiler:
        enabled = True
        tasks = (0,)
        def __init__(self, total):
            assert total == 2
        @contextmanager
        def range(self, index, synchronize):
            nonlocal active
            if index == 0:
                synchronize(); active = True; observed.append('push')
            try:
                yield
            finally:
                if index == 0:
                    synchronize(); observed.append('pop'); active = False
    monkeypatch.setattr(pipeline, 'TaskProfiler', Profiler)
    monkeypatch.delenv('HIPPO_EVENT_PATH', raising=False)
    stream = TaskStream(times=np.arange(5.), targets=np.arange(10.).reshape(5, 2),
        coordinates=np.array([[0., 0.], [1., 1.]]), visible=[0], hidden=[1], initial_sites=[0],
        initial_steps=2, task_steps=2, release_previous=True)
    original_task = stream.task
    def release(index):
        assert active == (index == 0)
        observed.append(f'release-{index}')
        return original_task(index)
    monkeypatch.setattr(stream, 'task', release)
    class Adapter:
        def initialize(self, initial):
            assert not active
        def predict_task(self, task):
            assert active == (task.index == 0)
            observed.append(f'prediction-{task.index}')
            self.mean, self.variance = np.zeros(task.query_shape), np.ones(task.query_shape)
            return self.mean, self.variance
    original_npz = pipeline._npz
    def serialize(path, values):
        assert not active
        return original_npz(path, values)
    monkeypatch.setattr(pipeline, '_npz', serialize)
    def score(task, truth, adapter):
        assert not active
        observed.append(f'score-{task.index}')
        return gaussian_scores(truth, adapter.mean, adapter.variance)
    result = pipeline.run(stream, Adapter(), output=tmp_path, score=score)
    assert result['ordinary_latency_valid'] is False
    assert result['profiled_task_ids'] == [0]
    assert observed[:5] == ['push', 'release-0', 'prediction-0', 'pop', 'score-0']
    configuration = json.loads((tmp_path/'configuration.json').read_text())
    assert configuration['ordinary_latency_valid'] is False
