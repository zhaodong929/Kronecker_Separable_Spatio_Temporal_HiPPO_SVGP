import numpy as np
import pytest


@pytest.mark.parametrize('sites,inducing', [(4, 2), (5, 3)])
def test_compact_st_gaussian_vi_posterior_energy_and_all_gradients(sites, inducing):
    from baselines.covid_long_setting_b.adapters.run_st_svgp import make_model
    from baselines.st_svgp_task_training import make_compact_model
    import objax
    import jax.numpy as jnp
    rng = np.random.default_rng(73)
    coordinates = rng.uniform(size=(sites, 2))
    times = np.array([0., .001, .003, .08, .2, .41])[:, None]
    y = rng.normal(size=(6, sites))
    spatial = np.repeat(coordinates[None], len(times), axis=0)
    theta = dict(kernel_variance=1.3, ell_t=.37, ell_s=(.6, 1.2), noise_std=.41)
    kwargs = dict(trainable_inducing=False, theta=theta)
    reference = make_model(times, spatial, y, coordinates[:inducing], **kwargs)
    compact = make_compact_model(times, spatial, y, coordinates[:inducing], **kwargs)
    def gradient(model):
        derivative = objax.GradValues(model.energy, model.vars())
        @objax.Function.with_vars(model.vars())
        def calculate(): return derivative()
        return objax.Jit(calculate)()
    for iteration, rate in enumerate([1., .4, 1.]):
        # Alternating updates: sites/posterior are fixed when differentiating
        # the hyperparameter objective, exactly as official train_task1 does.
        reference.inference(lr=rate); compact.inference(lr=rate)
        for a, b in zip(compact.compute_full_pseudo_lik(), reference.compute_full_pseudo_lik()):
            np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-9)
        np.testing.assert_allclose(compact.posterior_mean.value, reference.posterior_mean.value, rtol=2e-8, atol=2e-9)
        np.testing.assert_allclose(compact.posterior_variance.value, reference.posterior_variance.value, rtol=2e-8, atol=2e-9)
        cg, cv = gradient(compact); rg, rv = gradient(reference)
        np.testing.assert_allclose(cv, rv, rtol=2e-8, atol=2e-8)
        cvars = compact.vars().subset(objax.TrainVar)
        rvars = reference.vars().subset(objax.TrainVar)
        assert len(cg) == len(rg) == len(cvars) == len(rvars)
        for (name, _), a, b in zip(cvars.items(), cg, rg):
            np.testing.assert_allclose(a, b, rtol=2e-7, atol=2e-8, err_msg=name)
        # Perturb every hyperparameter, without recomputing sites until next iteration.
        changed = [v+jnp.asarray(rng.normal(scale=.015, size=v.shape)) for v in rvars.tensors()]
        rvars.assign(changed); cvars.assign(changed)
    # The data-dependent observation sites never carry two spatial axes.
    assert compact.nat1.value.shape == (6, sites, 1)
    assert compact.precision.value.shape == ()
    assert compact.posterior_variance.value.shape == (6, inducing, inducing)


def test_compact_st_rejects_time_varying_geometry():
    from baselines.st_svgp_task_training import make_compact_model
    grid = np.zeros((3, 4, 2)); grid[1, 0, 0] = 1.
    with pytest.raises(ValueError, match='same spatial grid'):
        make_compact_model(np.arange(3.), grid, np.ones((3, 4)), grid[0, :2])


def test_compact_st_official_adam_fit_and_unchanged_online_adapter():
    from baselines.covid_long_setting_b.adapters.run_st_svgp import make_model, train_task1
    from baselines.st_svgp_task_training import make_compact_model
    from benchmarks.task_stream.markov import STTaskAdapter
    from benchmarks.task_stream.protocol import TaskStream
    import objax
    coords = np.array([[0., 0.], [.2, .5], [.6, .3], [.9, .8]])
    times = np.array([0., .1, .3, .4, .7, 1., 1.2])
    y = np.random.default_rng(74).normal(size=(7, 4))
    stream = TaskStream(times=times, targets=y, coordinates=coords,
        visible=[0, 1, 2], hidden=[3], initial_sites=[0, 1, 2], initial_steps=3,
        task_steps=2, release_previous=True)
    batch = stream.initial()
    grid = np.repeat(coords[None, batch.sites], len(batch.times), axis=0)
    models = [build(batch.times[:, None], grid, batch.values, coords[[0, 2]],
              trainable_inducing=False) for build in (make_model, make_compact_model)]
    for model in models:
        train_task1(model, iterations=2, check_interval=2, min_steps=2, plateau_checks=10,
            plateau_relative_improvement=0., adam_lr=.001, newton_lr=1.,
            checkpoint_directory=None, seed=75, spatial_inducing=2)
    for a, b in zip(models[0].vars().subset(objax.TrainVar).tensors(),
                    models[1].vars().subset(objax.TrainVar).tensors()):
        assert a.dtype == b.dtype == np.float64
        np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-9)
    adapters = [STTaskAdapter(m, coords) for m in models]
    for adapter in adapters: adapter.initialize(batch)
    for index in range(2):
        expected = adapters[0].predict_task(stream.task(index))
        actual = adapters[1].predict_task(stream.task(index))
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-9)
