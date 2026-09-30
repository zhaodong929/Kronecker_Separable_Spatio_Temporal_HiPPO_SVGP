"""Tiny initial fitting and ordering checks for the Markov method factories."""
import os
import numpy as np
import pytest
from benchmarks.task_stream.factory import Configuration, FeatureTable, FittedTaskAdapter
from benchmarks.task_stream.protocol import TaskStream


@pytest.mark.parametrize('method', ['st_svgp', 'mgpvae'])
def test_markov_factory_fits_then_preserves_clock_and_query_order(method):
    source = os.environ.get('MGPVAE_SOURCE', '')
    if method == 'mgpvae' and not source:
        pytest.skip('MGPVAE_SOURCE required')
    times = np.array([0., .1, .3, .4, .6, .8, 1.])
    coords = np.array([[0., 0.], [.2, .5], [.6, .3], [.9, .8]])
    values = np.random.default_rng(51).normal(size=(7, 4))
    # Initial site order differs from coordinate order. Hidden queries reverse
    # that order as well, so an accidental identity map cannot pass.
    stream = TaskStream(times=times, targets=values, coordinates=coords,
        visible=[2, 0], hidden=[3, 1], initial_sites=[2, 0], initial_steps=3,
        task_steps=2, release_previous=True)
    features = FeatureTable(times, np.ones((7, 4, 1)))
    config = Configuration(method, initial_iterations=1, learning_rate=.001,
        spatial_inducing=2, latent=2, width=3, training_samples=2,
        prediction_samples=8, official_source=source, seed=52)
    adapter = FittedTaskAdapter(config, coords, stream.visible, features,
        initial_step=.1, release_previous=True)
    adapter.initialize(stream.initial())
    if method == 'mgpvae':
        np.testing.assert_array_equal(adapter.inverse, [1, 2, 0, 3])
        np.testing.assert_array_equal(adapter.adapter.coordinates, coords[[2, 0, 1, 3]])
    else:
        assert adapter.inverse is None
    # Compare outer wrapping against one direct inner call, capturing its task
    # without advancing its state a second time.
    predict = adapter.adapter.predict_task
    seen = []
    def capture(task):
        result = predict(task)
        seen.append((task, tuple(np.array(x) for x in result)))
        return result
    adapter.adapter.predict_task = capture
    for index in range(2):
        task = stream.task(index)
        mean, variance = adapter.predict_task(task)
        inner, (raw_mean, raw_variance) = seen[-1]
        np.testing.assert_array_equal(inner.times, task.times)
        np.testing.assert_array_equal(inner.visible.values,
            task.visible.values-adapter._mean(task.times, task.visible.sites))
        expected_sites = task.query_sites if adapter.inverse is None else adapter.inverse[task.query_sites]
        np.testing.assert_array_equal(inner.query_sites, expected_sites)
        np.testing.assert_allclose(mean, raw_mean+adapter._mean(task.times, task.query_sites))
        np.testing.assert_array_equal(variance, raw_variance)
        assert mean.shape == variance.shape == (2, 2)
        assert np.isfinite(mean).all() and np.isfinite(variance).all() and (variance > 0).all()
        assert adapter.adapter.window.time == task.times[-1]
        if index:
            np.testing.assert_array_equal(inner.delayed.times, stream.task(0).times)
            expected_delayed_sites = task.delayed.sites if adapter.inverse is None else adapter.inverse[task.delayed.sites]
            np.testing.assert_array_equal(inner.delayed.sites, expected_delayed_sites)
            np.testing.assert_array_equal(inner.delayed.values,
                task.delayed.values-adapter._mean(task.delayed.times, task.delayed.sites))
        if method == 'mgpvae':
            components, noise = adapter.components
            assert components.shape == (8, 2, 2)
            np.testing.assert_allclose(mean, components.mean(0), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(variance, components.var(0)+noise, rtol=1e-12, atol=1e-12)
