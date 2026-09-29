import os
import numpy as np
import pytest


def test_spatial_pushforward_matches_independent_dense_product():
    from baselines.mgpvae.spatial_moments import mix_spatial_moments
    mixing = np.array([[1., 0.], [.8, .6]])
    mean = np.array([[[[1.], [2.]]]])
    covariance = np.array([[[[.3, .05], [.05, .7]]]])
    actual_mean, actual_covariance = mix_spatial_moments(mixing, mean, covariance)
    np.testing.assert_allclose(actual_mean[0, 0], mixing @ mean[0, 0], rtol=1e-6)
    np.testing.assert_allclose(actual_covariance[0, 0], mixing @ covariance[0, 0] @ mixing.T, rtol=1e-6)
    assert not np.allclose(np.diag(mixing @ covariance[0, 0]), np.diag(actual_covariance[0, 0]))


def test_corrected_energy_keeps_kl_and_objective_decomposition():
    source = os.environ.get('MGPVAE_SOURCE')
    if not source:
        pytest.skip('set MGPVAE_SOURCE')
    from baselines.mgpvae.official import make_model
    import jax
    import jax.numpy as jnp
    coordinates = np.array([[0., 0.], [.4, .3], [.9, .6]])
    original = make_model(source, coordinates, seed=19)
    corrected = make_model(source, coordinates, seed=19, correct_spatial_covariance=True)
    corrected.vars().assign(original.vars().tensors())
    y = jnp.asarray(np.random.default_rng(2).normal(size=(3, 4, 1)))
    times = jnp.arange(4.)[:, None]
    before = np.array(original.energy(y, jax.random.PRNGKey(3), t=times, num_samples=8))
    after = np.array(corrected.energy(y, jax.random.PRNGKey(3), t=times, num_samples=8))
    assert np.isfinite(after).all()
    np.testing.assert_allclose(before[2], after[2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(after[0], after[1]+after[2], rtol=1e-12)
    assert abs(before[1]-after[1]) > 1e-8
