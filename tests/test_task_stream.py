import json
import numpy as np
import pytest
from benchmarks.task_stream.protocol import TaskStream
from benchmarks.task_stream.window import TaskWindow
from benchmarks.task_stream.pipeline import run
from benchmarks.task_stream.measurement import ArithmeticCount, Measurements


def stream(y=None, release=True):
    return TaskStream(times=np.array([0., .2, .5, .7, 1., 1.4, 1.8]),
        targets=np.arange(21).reshape(7, 3) / 20 if y is None else y,
        coordinates=[[0., 0.], [.2, .5], [.7, .2]], visible=[0, 1], hidden=[2],
        initial_sites=[0, 1], initial_steps=2, task_steps=2, release_previous=release)


def test_release_boundaries_serialization_and_short_final_task(tmp_path):
    a = stream()
    assert [a.task(i).query_shape for i in range(3)] == [(2, 1), (2, 1), (1, 1)]
    assert a.task(0).delayed is None
    np.testing.assert_array_equal(a.task(1).delayed.times, a.task(0).times)
    assert not hasattr(a.task(0), 'targets')
    with pytest.raises(ValueError):
        a.task(0).visible.values[0, 0] = 123
    a.save(tmp_path/'p.npz')
    assert a.identity() == TaskStream.load(tmp_path/'p.npz').identity()
    y = a._targets.copy(); y[2:, 2] += 999
    b = stream(y)
    np.testing.assert_array_equal(a.initial().values, b.initial().values)
    np.testing.assert_array_equal(a.task(0).visible.values, b.task(0).visible.values)
    assert not np.array_equal(a.task(1).delayed.values, b.task(1).delayed.values)
    assert stream(release=False).task(1).delayed is None


def test_bounded_task_replay_matches_independent_all_released_sum():
    a = stream()
    # Exact scalar Gaussian information update, independent reference sums.
    def step(mean, precision, dt, sites, values):
        return mean + values.sum(), precision + len(values)
    state = TaskWindow(step, 0., 1.)
    state.initialize(a.initial())
    for index in range(3):
        task = a.task(index)
        trace = state.advance(task)
        expected = a.initial().values.sum()
        count = a.initial().values.size
        for earlier in range(index + 1):
            e = a.task(earlier)
            expected += e.visible.values.sum(); count += e.visible.values.size
            if e.delayed is not None:
                expected += e.delayed.values.sum(); count += e.delayed.values.size
        assert state.mean == pytest.approx(expected)
        assert state.covariance == count + 1
        assert len(trace.times) <= 2 * a.task_steps + 1
        assert len(state.previous.times) <= a.task_steps
    with pytest.raises(ValueError, match='in order'):
        state.advance(a.task(2))


def test_pipeline_weights_queries_logs_every_task_and_does_not_leak(tmp_path, monkeypatch):
    monkeypatch.setenv('HIPPO_EVENT_PATH', str(tmp_path/'events.jsonl'))
    class Adapter:
        def initialize(self, initial):
            self.level = initial.values.mean()
        def predict_task(self, task):
            return np.full(task.query_shape, self.level), np.ones(task.query_shape)
    a = stream()
    result = run(a, Adapter(), output=tmp_path/'run')
    expected = np.sqrt(np.mean((a._targets[2:, 2] - a.initial().values.mean()) ** 2))
    assert result['metrics']['rmse'] == pytest.approx(expected)
    assert result['timing']['queries'] == 5
    assert not result['main_table_admitted']
    events = [json.loads(x) for x in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [e['phase'] for e in events] == ['configuration', 'online', 'online', 'online', 'complete']


def test_counter_coverage_and_replay_are_not_silently_accepted():
    counts = ArithmeticCount.from_kernels([dict(id=1, dadd=2, dmul=3, dfma=4)], provenance='test')
    assert counts.fp64_add_mul_fma == 13
    assert counts.status == 'measured_subset'
    assert ArithmeticCount.from_kernels([], provenance='disabled').fp64_add_mul_fma is None
    assert ArithmeticCount.from_kernels([dict(id=1, dadd=0)], provenance='partial').status == 'incomplete'
    with pytest.raises(ValueError, match='Duplicate'):
        ArithmeticCount.from_kernels([dict(id=1, dadd=0, dmul=0, dfma=0)] * 2, provenance='replayed')


def test_sync_boundaries_exclude_scoring_and_reject_overlapping_phases():
    synced = []
    clock = iter([1., 3., 4., 9.])
    m = Measurements(lambda: synced.append(True), lambda: next(clock))
    with m.phase('online', task=0, queries=2):
        with pytest.raises(ValueError, match='Overlapping'):
            with m.phase('scoring'):
                pass
    with m.phase('scoring'):
        pass
    assert len(synced) == 4
    assert [r['seconds'] for r in m.rows] == [2., 5.]
