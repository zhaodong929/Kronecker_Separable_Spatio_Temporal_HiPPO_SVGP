"""All-variable derivative equivalence of compact and dense corrected ELBOs."""
import os
import numpy as np
import pytest


@pytest.mark.parametrize('sites,latent', [(3, 2), (4, 3)])
def test_compact_training_loss_marginals_and_all_named_gradients(sites, latent):
    source = os.environ.get('MGPVAE_SOURCE')
    if not source: pytest.skip('MGPVAE_SOURCE required')
    from baselines.mgpvae.official import make_model
    from baselines.mgpvae.task_training import (validate_compact_training_model,
        compact_training_posterior, compact_training_energy)
    from baselines.mgpvae.spatial_moments import corrected_energy, diagonal_corrected_energy
    import jax
    import jax.numpy as jnp
    import objax
    rng = np.random.default_rng(57)
    coords = rng.uniform(size=(sites, 2))
    model = make_model(source, coords, seed=61, latent=latent, width=3,
        correct_spatial_covariance=True, compact_spatial_marginals=True, sitewise_training_filter=True)
    validate_compact_training_model(model)
    variables = model.vars().subset(objax.TrainVar)
    variables.assign([v + jnp.asarray(rng.normal(scale=.04, size=v.shape)) for v in variables.tensors()])
    values = jnp.asarray(rng.normal(size=(sites, 6, 1)))
    times = jnp.asarray([0., 0., 1/2015, 2/2015, .09, .3])[:, None]
    dt = jnp.concatenate([jnp.zeros(1), jnp.diff(times[:, 0])])
    key = jax.random.PRNGKey(63)
    expected = model.update_posterior(values, dt=dt)
    actual = compact_training_posterior(model, values, dt)
    np.testing.assert_allclose(actual[0], expected[2], rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(actual[1], jnp.diagonal(expected[3], axis1=-2, axis2=-1)[..., None], rtol=2e-9, atol=2e-10)
    for a, b in zip(actual[2:], expected[4:]):
        np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-10)
    assert actual[0].shape == actual[1].shape == (6, latent, sites, 1)
    def evaluate(energy):
        gradients = objax.GradValues(lambda: energy(model, values, key, t=times, num_samples=4), model.vars())
        @objax.Function.with_vars(model.vars())
        def calculate(): return gradients()
        return objax.Jit(calculate)()
    cg, cv = evaluate(compact_training_energy)
    for energy in (corrected_energy, diagonal_corrected_energy):
        rg, rv = evaluate(energy)
        np.testing.assert_allclose(cv, rv, rtol=2e-9, atol=2e-9)
        assert len(cg) == len(rg) == len(variables)
        for (name, _), a, b in zip(variables.items(), cg, rg):
            np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-9, err_msg=name)


def test_compact_training_requires_explicit_guard():
    from types import SimpleNamespace
    from baselines.mgpvae.task_training import compact_training_posterior
    with pytest.raises(ValueError, match='validate_compact'):
        compact_training_posterior(SimpleNamespace(), None, None)


@pytest.mark.parametrize('nonidentity,parallel', [(True, False), (False, True)])
def test_compact_training_rejects_unqualified_measurement_or_parallel(nonidentity, parallel):
    from types import SimpleNamespace
    from baselines.mgpvae.task_training import validate_compact_training_model
    h = np.kron(np.eye(3), np.array([[1., 0.]]))
    if nonidentity: h[0, 2] = .2
    kernel = SimpleNamespace(Ns=3, measurement_model=lambda: h)
    model = SimpleNamespace(kernel=kernel, parallel=parallel, time_transform=None)
    with pytest.raises(ValueError, match='identity|sequential'):
        validate_compact_training_model(model)
