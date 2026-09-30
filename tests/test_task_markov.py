import os
import numpy as np
import pytest
from benchmarks.task_stream.protocol import TaskStream


@pytest.mark.parametrize('release', [False, True])
def test_st_task_end_matches_official_full_prefix_prediction(release):
    from baselines.covid_long_setting_b.adapters.run_st_svgp import make_model, frozen_hyperparameters, assign_frozen_hyperparameters
    from benchmarks.task_stream.markov import STTaskAdapter
    import bayesnewton
    coords = np.array([[0., 0.], [.2, .5], [.6, .3], [.9, .8]])
    times = np.array([0., .1, .3, .4, .7, 1., 1.2])
    y = np.random.default_rng(47).normal(size=(7, 4))
    stream = TaskStream(times=times, targets=y, coordinates=coords, visible=[0, 1], hidden=[2, 3],
                        initial_sites=[0, 1], initial_steps=2, task_steps=2, release_previous=release)
    model = make_model(times[:2, None], np.repeat(coords[None, :2], 2, axis=0), y[:2, :2], coords[[0, 2]], trainable_inducing=False)
    kv, lv = frozen_hyperparameters(model)
    adapter = STTaskAdapter(model, coords); adapter.initialize(stream.initial())
    x, values = [], []
    def record(batch):
        for t, row in zip(batch.times, batch.values):
            for site, value in zip(batch.sites, row):
                x.append([t, *coords[site]]); values.append(value)
    record(stream.initial())
    for i in range(3):
        task = stream.task(i)
        if task.delayed is not None: record(task.delayed)
        record(task.visible)
        t, r, yy = bayesnewton.utils.create_spatiotemporal_grid(np.array(x), np.array(values)[:, None])
        reference = make_model(t, r, yy, coords[[0, 2]], trainable_inducing=False)
        assign_frozen_hyperparameters(reference, kv, lv); reference.inference(lr=1.)
        # Duplicate a singleton query for the upstream squeeze convention.
        q = task.times if len(task.times) > 1 else np.repeat(task.times, 2)
        expected = reference.predict_y(X=q[:, None], R=np.repeat(coords[None], len(q), axis=0))
        actual = adapter.predict_task(task)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, np.asarray(b)[:len(task.times), 2:], rtol=2e-6, atol=2e-7)


def test_mgp_task_smoothing_matches_official_all_observed_marginals():
    source = os.environ.get('MGPVAE_SOURCE')
    if not source: pytest.skip('MGPVAE_SOURCE required')
    from baselines.mgpvae.official import make_model
    from benchmarks.task_stream.markov import MGPTaskAdapter, mgp_task_marginals
    from benchmarks.task_stream.protocol import Observations, Task
    import jax.numpy as jnp
    coords = np.array([[0., 0.], [.4, .3], [.9, .6]])
    times = np.array([0., .2, .5, .8, 1.2, 1.7])
    y = np.random.default_rng(48).normal(size=(6, 3))
    model = make_model(source, coords, correct_spatial_covariance=True)
    adapter = MGPTaskAdapter(model, coords, samples=8)
    adapter.initialize(Observations(times[:2], np.arange(3), y[:2]))
    for i, start in enumerate([2, 4]):
        task = Task(i, start, start+2, Observations(times[start:start+2], np.arange(3), y[start:start+2]), None, np.arange(3))
        trace = adapter.window.advance(task)
        actual = mgp_task_marginals(model, trace, coords)
        expected = model.predict(jnp.asarray(times[:start+2, None]), jnp.asarray(times[start:start+2, None]),
                                 jnp.asarray(y[:start+2].T[..., None]), jnp.asarray(coords))
        for a, b in zip(actual, expected):
            b = np.asarray(b).reshape(3, 2, model.num_latent).transpose(1, 0, 2)
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('release', [False, True])
def test_mgp_bounded_partial_tasks_match_recomputed_full_prefix(release):
    source = os.environ.get('MGPVAE_SOURCE')
    if not source: pytest.skip('MGPVAE_SOURCE required')
    from baselines.mgpvae.official import make_model
    from baselines.mgpvae.selected import SelectedSiteFilter
    from benchmarks.task_stream.markov import MGPTaskAdapter, mgp_task_marginals
    from benchmarks.task_stream.window import FilterTrace
    coords = np.array([[0., 0.], [.4, .3], [.9, .6]])
    times = np.array([0., .2, .5, .8, 1.2, 1.7, 2.])
    y = np.random.default_rng(49).normal(size=(7, 3))
    stream = TaskStream(times=times, targets=y, coordinates=coords, visible=[0, 1], hidden=[2],
        initial_sites=[0, 1], initial_steps=2, task_steps=2, release_previous=release)
    model = make_model(source, coords, correct_spatial_covariance=True)
    adapter = MGPTaskAdapter(model, coords, samples=8)
    adapter.initialize(stream.initial())
    for i in range(3):
        task = stream.task(i)
        bounded = adapter.window.advance(task)
        endpoint = (np.array(adapter.window.mean), np.array(adapter.window.covariance))
        actual = mgp_task_marginals(model, bounded, coords[[2]])
        # Rebuild from the original prior with every legally released observation.
        # No TaskWindow replay state participates in this reference calculation.
        reference = SelectedSiteFilter(model)
        m, p = reference.mean, reference.covariance
        means, covariances = [], []
        for j in range(task.stop):
            sites = [0, 1, 2] if release and 2 <= j < task.start else [0, 1]
            dt = 0. if j == 0 else times[j]-times[j-1]
            m, p = reference.selected_step(m, p, dt, [(s, y[j, s]) for s in sites])
            means.append(m); covariances.append(p)
        full = FilterTrace(times[:task.stop], tuple(means), tuple(covariances), task.start)
        expected = mgp_task_marginals(model, full, coords[[2]])
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(adapter.window.mean, m, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(adapter.window.covariance, p, rtol=1e-10, atol=1e-11)
        np.testing.assert_array_equal(adapter.window.mean, endpoint[0])
        np.testing.assert_array_equal(adapter.window.covariance, endpoint[1])
        assert len(bounded.times) <= 2*stream.task_steps+1
