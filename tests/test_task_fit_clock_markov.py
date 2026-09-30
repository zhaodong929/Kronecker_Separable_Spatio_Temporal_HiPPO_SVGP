import os
import numpy as np
import pytest
from benchmarks.task_stream.factory import Configuration, FeatureTable, FittedTaskAdapter
from benchmarks.task_stream.protocol import TaskStream


@pytest.mark.parametrize('method', ['st_svgp', 'mgpvae'])
def test_markov_initial_clock_stops_after_completed_step_and_records_actual_budget(method):
    source = os.environ.get('MGPVAE_SOURCE', '')
    if method == 'mgpvae' and not source: pytest.skip('MGPVAE_SOURCE required')
    times = np.arange(7.)/10
    coords = np.array([[0., 0.], [.2, .4], [.6, .3], [.8, .7]])
    values = np.random.default_rng(83).normal(size=(7, 4))
    stream = TaskStream(times=times, targets=values, coordinates=coords,
        visible=[0, 1, 2], hidden=[3], initial_sites=[0, 1, 2],
        initial_steps=3, task_steps=2, release_previous=True)
    config = Configuration(method, initial_iterations=1000000, initial_max_seconds=.000001,
        learning_rate=.001, spatial_inducing=2, width=3, training_samples=2,
        prediction_samples=8, official_source=source)
    features = FeatureTable(times, np.ones((7, 4, 1)))
    adapter = FittedTaskAdapter(config, coords, stream.visible, features,
        initial_step=.1, release_previous=True)
    adapter.initialize(stream.initial())
    record = adapter.fit_budget_record.copy()
    assert record['completed_steps'] == 1
    assert record['observation_rows'] == 9
    assert record['sampled_rows'] == 9
    assert record['stop_reason'] == 'wall_time_budget_reached'
    assert record['elapsed_seconds'] > .000001
    assert not record['convergence_claimed']
    for i in range(2):
        mean, variance = adapter.predict_task(stream.task(i))
        assert np.isfinite(mean).all() and np.all(variance > 0)
    assert adapter.fit_budget_record == record
