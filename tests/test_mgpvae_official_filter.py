"""Qualification against the pinned external filter, not a copied reference."""
import os
import numpy as np
import pytest
from baselines.mgpvae.official import make_model
from baselines.mgpvae.filtering import OfficialVisibleFilter


def test_stateful_filter_matches_official_prefix_at_every_time():
    source = os.environ.get('MGPVAE_SOURCE')
    if not source:
        pytest.skip('set MGPVAE_SOURCE to the pinned external checkout')
    import jax.numpy as jnp
    model = make_model(source, np.array([[0., 0.], [1., 0.], [0., 1.]]))
    data = np.random.default_rng(4).normal(size=(3, 6, 1))
    times = np.arange(6.)
    adapter = OfficialVisibleFilter(model)
    for i,t in enumerate(times):
        fm, fp = adapter.observe(t, data[:, i, :])
        py, pv = model.compute_full_pseudo_lik(jnp.array(data[:, :i+1]))
        dt = jnp.array([0.] + [1.]*i)
        _, (refm, refp) = model.filter(dt, model.kernel, py, pv, parallel=False)
        np.testing.assert_allclose(fm, refm[:, -1], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(fp, refp[:, -1], rtol=1e-9, atol=1e-10)
    query = np.array([[0.5, 0.5], [0.2, 0.1], [0.8, 0.3]])
    actual_mean, actual_var = adapter.latent_at(query)
    expected_mean, expected_var = model.predict(
        jnp.asarray(times), jnp.asarray(times[-1:]), jnp.asarray(data), jnp.asarray(query))
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_var, expected_var, rtol=1e-6, atol=1e-7)
    components, noise = adapter.gaussian_components(np.array([[0.5, 0.5], [0.2, 0.1]]), seed=2)
    assert components.shape == (128, 2)
    assert np.isfinite(components).all() and noise > 0
    with pytest.raises(ValueError):
        adapter.observe(times[-1], data[:, -1])
