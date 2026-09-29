import os
import numpy as np
import pytest
from baselines.mgpvae.official import make_model
from baselines.mgpvae.filtering import OfficialVisibleFilter
from baselines.mgpvae.partial import PartialPrefixFilter


def model():
    source = os.environ.get('MGPVAE_SOURCE')
    if not source:
        pytest.skip('MGPVAE_SOURCE required')
    return make_model(source, np.array([[0.,0.], [1.,0.], [0.,1.]]))


def test_all_observed_matches_official_and_delayed_release_replays_once():
    m = model()
    y = np.random.default_rng(9).normal(size=(3,3))
    full, partial = OfficialVisibleFilter(m), PartialPrefixFilter(m)
    for t in range(3):
        full.observe(t, y[t])
        partial.release(t, [0,1,2], y[t], available_at=t)
        actual = partial.infer(t)
        np.testing.assert_allclose(actual[0], full.mean, atol=1e-9)
        np.testing.assert_allclose(actual[1], full.covariance, atol=1e-9)
    delayed, reference = PartialPrefixFilter(m), PartialPrefixFilter(m)
    delayed.release(0, [0,1], y[0,:2], available_at=0)
    reference.release(0, [0,1], y[0,:2], available_at=0)
    before = delayed.infer(0)
    delayed.release(0, [2], y[0,2:], available_at=1)
    after_registration = delayed.infer(0)
    np.testing.assert_array_equal(before[0], after_registration[0])
    delayed.release(1, [0,1], y[1,:2], available_at=1)
    reference.release(0, [2], y[0,2:], available_at=0)
    reference.release(1, [0,1], y[1,:2], available_at=1)
    actual, expected = delayed.infer(1), reference.infer(1)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-9)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-9)
    with pytest.raises(ValueError, match='duplicate'):
        delayed.release(0, [2], y[0,2:], available_at=2)


def test_hidden_and_future_values_do_not_enter_encoder():
    m = model()
    a,b = PartialPrefixFilter(m), PartialPrefixFilter(m)
    for obj, secret in ((a,1.), (b,1e12)):
        obj.release(0, [0,1], [0.2,-0.4], available_at=0)
        obj.release(0, [2], [secret], available_at=1)
        obj.release(2, [0,1,2], [secret]*3, available_at=2)
    am, ap = a.infer(0)
    bm, bp = b.infer(0)
    np.testing.assert_array_equal(am, bm)
    np.testing.assert_array_equal(ap, bp)


def test_one_observed_site_matches_dense_gaussian_conditioning():
    import jax.numpy as jnp
    from scipy.linalg import block_diag
    m = model()
    adapter = PartialPrefixFilter(m)
    adapter.release(0, [1], [0.7], available_at=0)
    means, covs = adapter.infer(0)
    encoded, variance = m.compute_full_pseudo_lik(jnp.array([[[0.7]]]))
    H = np.asarray(m.kernel.measurement_model())
    for latent, blocks in enumerate(np.asarray(adapter.Pinf)):
        prior = block_diag(*blocks)
        h = H[latent,1]
        s = h @ prior @ h + float(variance[0,latent,0])
        expected_mean = prior @ h * float(encoded[0,latent,0]) / s
        expected_cov = prior - np.outer(prior @ h, prior @ h) / s
        np.testing.assert_allclose(np.asarray(means[latent]).ravel(), expected_mean, atol=1e-9)
        for site in range(3):
            np.testing.assert_allclose(covs[latent,site], expected_cov[site*2:site*2+2,site*2:site*2+2], atol=1e-9)


def test_bounded_delay_matches_every_lawful_prefix():
    from baselines.mgpvae.partial import OneStepDelayedFilter
    m = model()
    reference, bounded = PartialPrefixFilter(m), OneStepDelayedFilter(m)
    y = np.random.default_rng(29).normal(size=(7,3))
    times = np.array([0., .3, 1., 1.7, 3., 4., 6.])
    for i,t in enumerate(times):
        reference.release(t, [0,1], y[i,:2], available_at=t)
        kwargs = {}
        if i:
            reference.release(times[i-1], [2], y[i-1,2:], available_at=t)
            kwargs = dict(delayed_time=times[i-1], delayed_sites=[2], delayed_values=y[i-1,2:])
        actual = bounded.advance(t, [0,1], y[i,:2], **kwargs)
        expected = reference.infer(t)
        np.testing.assert_allclose(actual[0], expected[0], atol=1e-9)
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-9)
    assert len(bounded._records) == 0  # no retained prefix
    assert bounded.replayed_steps == 2*len(times)-1
    with pytest.raises(ValueError, match='preceding'):
        bounded.advance(7., [0,1], [0.,0.], delayed_time=0., delayed_sites=[2], delayed_values=[1.])
