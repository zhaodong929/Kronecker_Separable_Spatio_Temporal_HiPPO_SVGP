import json
import os
import numpy as np
import pytest
from benchmarks.task_stream.pipeline import run
from benchmarks.task_stream.protocol import TaskStream
from benchmarks.three_domain.metrics import gaussian_mixture_metrics


def stream():
    return TaskStream(times=np.arange(7.), targets=np.arange(21).reshape(7, 3) / 20,
        coordinates=[[0., 0.], [.2, .5], [.7, .2]], visible=[0, 1], hidden=[2],
        initial_sites=[0, 1], initial_steps=2, task_steps=2, release_previous=True)


class Mixture:
    predictive_family = 'gaussian_mixture'
    def initialize(self, initial):
        self.step = -1
    def predict_task(self, task):
        self.step = task.index
        mu = np.broadcast_to(np.array([-1., .5, 2.])[:, None, None], (3,) + task.query_shape).copy()
        self.components = mu, .2
        self.latent = np.zeros(task.query_shape + (2,)), np.ones(task.query_shape + (2,))
        return mu.mean(0), mu.var(0) + .2
    def checkpoint_state(self):
        return dict(step=self.step, weights=np.array([1., 2.]), rng={'seed': 42})


def test_mixture_journal_artifacts_allow_exact_rescoring_short_task(tmp_path, monkeypatch):
    monkeypatch.delenv('HIPPO_EVENT_PATH', raising=False)
    result = run(stream(), Mixture(), output=tmp_path, provenance={'git_commit': 'test'})
    assert 'HIPPO_EVENT_PATH' not in os.environ
    events = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert events[-1]['phase'] == 'complete'
    scores, counts = [], []
    for directory in sorted((tmp_path / 'tasks').iterdir()):
        with np.load(directory / 'predictions.npz', allow_pickle=False) as p:
            scores.append(gaussian_mixture_metrics(p['y_true'], p['component_means'].reshape(3, -1), p['noise_variance']))
            counts.append(p['y_true'].size)
            assert 'latent_var' in p
        assert json.loads((directory / 'checkpoint.json').read_text())['auto_resume'] is False
    assert counts == [2, 2, 1]
    assert result['metrics']['nlpd'] == pytest.approx(np.average([s['nlpd'] for s in scores], weights=counts))
    assert 'tasks/000002/predictions.npz' in json.loads((tmp_path / 'artifacts.json').read_text())
    assert json.loads((tmp_path / 'timing.json').read_text())[0]['phase'] == 'initial_state'
    with pytest.raises(FileExistsError):
        run(stream(), Mixture(), output=tmp_path)


def test_failed_task_keeps_completed_predictions_and_failure_trace(tmp_path, monkeypatch):
    monkeypatch.delenv('HIPPO_EVENT_PATH', raising=False)
    class Broken(Mixture):
        def predict_task(self, task):
            if task.index == 1:
                raise RuntimeError('intentional task failure')
            return super().predict_task(task)
    with pytest.raises(RuntimeError, match='intentional'):
        run(stream(), Broken(), output=tmp_path)
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['status'] == 'failed' and result['completed_tasks'] == 1
    assert (tmp_path / 'tasks/000000/predictions.npz').is_file()
    assert not (tmp_path / 'tasks/000001/predictions.npz').exists()
    rows = json.loads((tmp_path / 'timing.json').read_text())
    assert rows[-1]['phase'] == 'online' and rows[-1]['status'] == 'failed'
    events = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert events[-1]['phase'] == 'failed'
    assert all(e['phase'] != 'complete' for e in events)


def test_serialization_failure_never_emits_complete(tmp_path, monkeypatch):
    monkeypatch.delenv('HIPPO_EVENT_PATH', raising=False)
    import benchmarks.task_stream.pipeline as pipeline
    original = pipeline._npz
    def broken(path, values):
        if path.name == 'predictions.npz' and path.parent == tmp_path:
            raise OSError('disk failure')
        return original(path, values)
    monkeypatch.setattr(pipeline, '_npz', broken)
    with pytest.raises(OSError, match='disk failure'):
        run(stream(), Mixture(), output=tmp_path)
    assert json.loads((tmp_path / 'result.json').read_text())['status'] == 'failed'
    events = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert all(e['phase'] != 'complete' for e in events)


def test_native_mixture_scores_equal_rescoring_reconstructed_physical_distribution(tmp_path, monkeypatch):
    monkeypatch.delenv('HIPPO_EVENT_PATH', raising=False)
    a = stream()
    a.metadata.update(dataset='era5', target_standardization={'mean': 280., 'scale': 7.})
    result = run(a, Mixture(), output=tmp_path)
    means, truths, components = [], [], []
    for directory in sorted((tmp_path / 'tasks').iterdir()):
        with np.load(directory / 'predictions.npz', allow_pickle=False) as data:
            center, scale = float(data['target_center']), float(data['target_scale'])
            truth = data['y_true']*scale + center
            mu = data['component_means']*scale + center
            variance = float(data['noise_variance'])*scale**2
            expected = gaussian_mixture_metrics(truth, mu.reshape(3, -1), variance)
            record = json.loads((directory/'record.json').read_text())
            for key in ('rmse', 'nlpd', 'crps'):
                assert record['metrics_native'][key] == pytest.approx(expected[key])
            np.testing.assert_allclose(mu.mean(0), data['pred_mean']*scale + center)
            np.testing.assert_allclose(mu.var(0) + variance, data['pred_var']*scale**2)
            truths.append(truth.reshape(-1)); components.append(mu.reshape(3, -1))
    expected = gaussian_mixture_metrics(np.concatenate(truths), np.concatenate(components, axis=1), variance)
    for key in ('rmse', 'nlpd', 'crps'):
        assert result['metrics_native'][key] == pytest.approx(expected[key])
    assert result['metric_scale'] == 'kelvin'
    assert result['metrics_native']['nlpd'] == pytest.approx(result['metrics']['nlpd'] + np.log(7.))
    assert result['metrics_native']['coverage'] == result['metrics']['coverage']
    assert len(json.loads((tmp_path/'task-metrics-native.json').read_text())) == 3


def test_native_transform_absent_is_identity_and_nonpositive_scale_rejected(tmp_path):
    from benchmarks.task_stream.pipeline import native_target_transform
    assert native_target_transform({})[:2] == (0., 1.)
    assert 'unavailable' in native_target_transform({})[2]
    for scale in (0., -1., np.inf, np.nan):
        with pytest.raises(ValueError):
            native_target_transform({'target_standardization': {'mean': 2., 'scale': scale}})
