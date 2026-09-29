import os
import numpy as np
import pytest


def source():
    path = os.environ.get('MGPVAE_SOURCE')
    if not path:
        pytest.skip('set MGPVAE_SOURCE')
    return path


def test_selected_scalar_updates_match_dense_reference_with_delayed_release():
    from baselines.mgpvae.official import make_model
    from baselines.mgpvae.partial import OneStepDelayedFilter
    from baselines.mgpvae.selected import SelectedSiteFilter
    model = make_model(source(), np.array([[0., 0.], [.3, .4], [.8, .7]]))
    reference, fast = OneStepDelayedFilter(model), SelectedSiteFilter(model)
    rng = np.random.default_rng(52)
    y = rng.normal(size=(5, 3))
    times = np.array([0., .1, .4, .6, 1.2])
    for i, t in enumerate(times):
        delayed = {} if i == 0 else dict(delayed_time=times[i-1], delayed_sites=[2], delayed_values=y[i-1, 2:])
        a = reference.advance(t, [0, 1], y[i, :2], **delayed)
        b = fast.advance(t, [0, 1], y[i, :2], **delayed)
        for expected, actual in zip(a, b):
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)


def test_append_unobserved_sites_preserves_original_spatial_conditional():
    from baselines.mgpvae.official import make_model
    from baselines.mgpvae.filtering import OfficialVisibleFilter
    from baselines.mgpvae.selected import SelectedSiteFilter
    coordinates = np.array([[0., 0.], [.3, .4], [.8, .7], [.6, -.3]])
    small = make_model(source(), coordinates[:2], seed=8)
    full = make_model(source(), coordinates, seed=8)
    full.vars().assign(small.vars().tensors())
    reference, extended = OfficialVisibleFilter(small), SelectedSiteFilter(full)
    y = np.random.default_rng(11).normal(size=(4, 2))
    for i, values in enumerate(y):
        reference.observe(i*.2, values)
        extended.advance(i*.2, [0, 1], values)
        a, b = reference.latent_at(coordinates[2:]), extended.latent_at(coordinates[2:])
        for expected, actual in zip(a, b):
            np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=3e-6)
