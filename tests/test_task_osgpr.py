import numpy as np
import pytest


def test_osgpr_product_amplitude_roundtrip_and_adaptation():
    gpflow = pytest.importorskip('gpflow')
    from scripts import run_official_bui_osgpr_era5 as o
    from benchmarks.task_stream.gp import OSGPRTaskAdapter
    from benchmarks.task_stream.protocol import TaskStream
    gpflow.config.set_default_float(np.float64)
    theta = dict(ell_t=.4, ell_s=[.5, .7], kernel_variance=1.3, noise_std=.3)
    coords = np.array([[0., 0.], [.2, .4], [.7, .9]])
    x = np.column_stack([np.arange(3.)*.1, coords])
    kernel = o.make_kernel(theta, frozen=False)
    assert len(kernel.trainable_variables) == 4
    assert not kernel.kernels[1].variance.trainable
    assert not kernel.kernels[2].variance.trainable
    model = gpflow.models.SGPR((x, np.sin(x[:, :1])), kernel, x, noise_variance=.09)
    # Old checkpoints may carry spatial amplitudes. Serialization must preserve
    # their full product rather than silently retaining only the temporal term.
    kernel.kernels[1].variance.assign(1.7); kernel.kernels[2].variance.assign(.6)
    serialized = o.theta_from_model(model)
    np.testing.assert_allclose(o.make_kernel(serialized, frozen=False)(x), kernel(x), rtol=1e-12)
    targets = np.random.default_rng(38).normal(size=(6, 3))
    stream = TaskStream(times=np.arange(6.)*.1, targets=targets, coordinates=coords,
        visible=[0, 1], hidden=[2], initial_sites=[0, 1], initial_steps=2,
        task_steps=2, release_previous=True)
    adapter = OSGPRTaskAdapter(coords, theta, x, initial_steps=2, update_steps=2)
    adapter.initialize(stream.initial())
    for i in range(2):
        mean, var = adapter.predict_task(stream.task(i))
        assert mean.shape == var.shape == (2, 1)
        assert np.isfinite(mean).all() and np.isfinite(var).all() and (var > 0).all()
        np.testing.assert_allclose(adapter.model.kernel(x),
            o.make_kernel(adapter.theta, frozen=False)(x), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(adapter.old[2], adapter.model.kernel(adapter.inducing), rtol=1e-12)
        assert all(float(k.variance.numpy()) == 1. for k in adapter.model.kernel.kernels[1:])
