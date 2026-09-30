import numpy as np
import pytest
from benchmarks.task_stream.factory import Configuration, FeatureTable, FittedTaskAdapter
from benchmarks.task_stream.protocol import TaskStream
from benchmarks.task_stream.pipeline import run


def tiny():
    t=np.arange(7.)/10
    coords=np.array([[0.,0.],[.2,.3],[.6,.1],[.8,.4]])
    s=TaskStream(times=t,targets=np.sin(t[:,None]+coords[:,0]),coordinates=coords,
        visible=[0,1,2],hidden=[3],initial_sites=[0,1,2],initial_steps=3,task_steps=2,release_previous=True)
    features=FeatureTable(t,np.stack([np.ones((7,4)),np.broadcast_to(t[:,None],(7,4))],axis=-1))
    return s,features


@pytest.mark.parametrize('method',['kronhippo_svgp','ohsvgp'])
def test_torch_initial_fit_and_stream_are_reproducible(method,tmp_path):
    import torch
    s,f=tiny()
    c=Configuration(method,initial_iterations=2,learning_rate=.001,spatial_inducing=2,
                    temporal_inducing=3,inducing_size=3,rff=16,grid_rows=4,batch_rows=4,online_iterations=1)
    predictions=[]
    for seed in [99,2]:
        torch.manual_seed(seed)
        a=FittedTaskAdapter(c,s.coordinates,s.visible,f,initial_step=.1,release_previous=True)
        out=tmp_path/str(seed)
        result=run(s,a,output=out)
        assert result['metrics']['rmse']>=0
        predictions.append(np.load(out/'predictions.npz')['pred_mean'])
    np.testing.assert_array_equal(*predictions)


def test_feature_time_lookup_and_configuration_reject_silent_changes():
    _,f=tiny()
    with pytest.raises(ValueError,match='clocks'): f([99.],[0])
    with pytest.raises(ValueError,match='excluded'): Configuration('gpvae',1,.01)
    with pytest.raises(ValueError,match='integer'): Configuration('mgpvae',1.5,.01)


def test_contribution_arms_reuse_actual_fit_and_change_only_declared_update():
    from benchmarks.task_stream.ablations import ARMS
    s, features = tiny()
    config = Configuration('kronhippo_svgp', initial_iterations=1, learning_rate=.001,
        spatial_inducing=2, temporal_inducing=3, rff=16)
    fits, predictions = [], []
    for arm in ARMS:
        adapter = FittedTaskAdapter(config, s.coordinates, s.visible, features,
            initial_step=.1, release_previous=True, arm=arm)
        adapter.initialize(s.initial())
        fits.append(adapter.fit_sha256)
        predictions.append(adapter.predict_task(s.task(0))[0])
    assert len(set(fits)) == 1
    assert not np.allclose(predictions[0], predictions[1])
    assert not np.allclose(predictions[0], predictions[2])


def test_oh_exposure_policy_drives_actual_fit_and_logs_resolution(monkeypatch):
    from dataclasses import asdict
    import benchmarks.task_stream.factory as factory
    s, features = tiny()
    config = Configuration('ohsvgp', initial_iterations=2, learning_rate=.001,
        initial_expected_passes=2., inducing_size=3, rff=16, grid_rows=4,
        batch_rows=4, online_iterations=1)
    original = asdict(config)
    events = []
    monkeypatch.setattr(factory, 'emit', lambda name, step, payload: events.append((name, step, payload)))
    adapter = FittedTaskAdapter(config, s.coordinates, s.visible, features,
        initial_step=.1, release_previous=True)
    adapter.initialize(s.initial())
    # Nine legal initial rows, four per minibatch: ceil(2*9/4)=5 actual updates.
    assert adapter.adapter.initial_steps == adapter.adapter.training_iteration == 5
    assert adapter.training_budget['observation_rows'] == 9
    assert adapter.training_budget['sampled_rows'] == 20
    budget_events = [payload for name, _, payload in events if name == 'training_budget']
    assert budget_events == [adapter.training_budget]
    configuration = [payload for name, _, payload in events if name == 'fit_configuration'][0]
    assert configuration['configuration'] == original
    assert configuration['initial_training_budget'] == adapter.training_budget
    assert asdict(config) == original


@pytest.mark.parametrize('method', ['kronhippo_svgp', 'ohsvgp'])
def test_common_time_budget_and_deterministic_selected_refit(method, monkeypatch):
    import benchmarks.task_stream.training_budget as budgets
    from dataclasses import asdict
    s, features = tiny()
    original_clock = budgets.FitClock
    def fake_clock(limit, synchronize):
        ticks = iter([0., limit+1., limit+1.])
        return original_clock(limit, synchronize, monotonic=lambda: next(ticks))
    monkeypatch.setattr(budgets, 'FitClock', fake_clock)
    config = Configuration(method, initial_iterations=1000000, learning_rate=.001,
        initial_max_seconds=60., spatial_inducing=2, temporal_inducing=3,
        inducing_size=3, rff=16, grid_rows=4, batch_rows=4, online_iterations=1)
    original_config = asdict(config)
    timed = FittedTaskAdapter(config, s.coordinates, s.visible, features,
        initial_step=.1, release_previous=True)
    timed.initialize(s.initial())
    assert timed.fit_budget_record['completed_steps'] == 1
    assert timed.fit_budget_record['stop_reason'] == 'wall_time_budget_reached'
    assert timed.fit_budget_record['overshoot_seconds'] == 1.
    assert np.isfinite(timed.predict_task(s.task(0))[0]).all()
    def forbidden_clock(*args, **kwargs):
        raise AssertionError('Frozen selected refits must not use a wall clock')
    monkeypatch.setattr(budgets, 'FitClock', forbidden_clock)
    predictions = []
    for _ in range(2):
        fixed = FittedTaskAdapter(config, s.coordinates, s.visible, features,
            initial_step=.1, release_previous=True, refit_iterations=2)
        fixed.initialize(s.initial())
        assert fixed.fit_budget_record['completed_steps'] == 2
        assert fixed.fit_budget_record['policy'] == 'selected_fixed_refit'
        assert fixed.fit_budget_record['elapsed_seconds'] is None
        assert fixed.training_budget['effective_iterations'] == 2
        predictions.append(fixed.predict_task(s.task(0))[0])
    np.testing.assert_array_equal(*predictions)
    assert asdict(config) == original_config


@pytest.mark.parametrize('iterations', [0, -1, True, 2.5])
def test_selected_refit_rejects_invalid_count(iterations):
    s, f = tiny()
    config = Configuration('ohsvgp', 1000000, .001, initial_max_seconds=60.)
    with pytest.raises(ValueError, match='positive integer'):
        FittedTaskAdapter(config, s.coordinates, s.visible, f, initial_step=.1, refit_iterations=iterations)


def test_selected_refit_requires_a_time_search_configuration():
    s, f = tiny()
    with pytest.raises(ValueError, match='wall-time search'):
        FittedTaskAdapter(Configuration('ohsvgp', 5, .001), s.coordinates, s.visible, f,
            initial_step=.1, refit_iterations=2)
