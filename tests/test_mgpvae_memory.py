"""Rematerialization must preserve the actual official objective and all gradients."""
import os
import numpy as np
import pytest


def test_official_energy_and_parameter_gradients_unchanged():
    source=os.environ.get('MGPVAE_SOURCE')
    if not source:pytest.skip('set MGPVAE_SOURCE')
    from baselines.mgpvae.official import make_model
    from baselines.mgpvae.memory import enable_scan_rematerialization
    import jax
    import jax.numpy as jnp
    import objax
    model=make_model(source,np.array([[0.,0.],[.4,.3],[.9,.6]]),
        seed=19,correct_spatial_covariance=True)
    import mgpvae.ops as ops
    original_scan=ops.scan
    values=jnp.asarray(np.random.default_rng(2).normal(size=(3,6,1)))
    times=jnp.arange(6.)[:,None]
    def evaluate():
        derivative=objax.GradValues(model.energy,model.vars())
        @objax.Function.with_vars(model.vars())
        def calculate():
            return derivative(values,jax.random.PRNGKey(3),t=times,num_samples=4)
        return objax.Jit(calculate)()
    reference_grad,reference_loss=evaluate()
    try:
        enable_scan_rematerialization()
        actual_grad,actual_loss=evaluate()
        np.testing.assert_allclose(actual_loss,reference_loss,rtol=1e-10,atol=1e-10)
        assert len(actual_grad)==len(reference_grad)
        for actual,expected in zip(actual_grad,reference_grad):
            np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
    finally:
        ops.scan=original_scan
