"""Qualify full posterior, objective and all gradients against pinned dense KF."""
import os
import numpy as np
import pytest


@pytest.mark.parametrize('latent',[2,4])
def test_sitewise_full_posterior_loss_and_gradients(latent):
    source=os.environ.get('MGPVAE_SOURCE')
    if not source:pytest.skip('set MGPVAE_SOURCE')
    from baselines.mgpvae.official import make_model
    import jax
    import jax.numpy as jnp
    import objax
    coordinates=np.array([[0.,0.],[.4,.3],[.9,.6]])
    options=dict(seed=17,latent=latent,correct_spatial_covariance=True,compact_spatial_marginals=True)
    reference=make_model(source,coordinates,**options)
    fast=make_model(source,coordinates,sitewise_training_filter=True,**options)
    values=jnp.asarray(np.random.default_rng(31).normal(size=(3,6,1)))
    # Include duplicate initial time, tiny PEMS steps and an irregular gap.
    times=jnp.asarray([0.,0.,1/2015,2/2015,.07,.3])[:,None]
    dt=jnp.concatenate([jnp.array([0.]),jnp.diff(times[:,0])])
    def evaluate(model):
        derivative=objax.GradValues(model.energy,model.vars())
        @objax.Function.with_vars(model.vars())
        def calculate():return derivative(values,jax.random.PRNGKey(12),t=times,num_samples=4)
        return objax.Jit(calculate)()
    for shift in [0.,.02]:
        variables=reference.vars().subset(objax.TrainVar)
        variables.assign([x+shift for x in variables.tensors()])
        fast.vars().assign(reference.vars().tensors())
        expected=reference.update_posterior(values,dt=dt)
        actual=fast.update_posterior(values,dt=dt)
        for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
        rg,rl=evaluate(reference);fg,fl=evaluate(fast)
        np.testing.assert_allclose(fl,rl,rtol=1e-10,atol=1e-10)
        assert len(rg)==len(fg)
        for a,b in zip(fg,rg):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)


def test_sitewise_rejects_unsupported_mask():
    from baselines.mgpvae.sitewise_filter import sitewise_official_filter
    with pytest.raises(ValueError,match='unmasked'):
        sitewise_official_filter(None,None,None,None,mask=np.ones(2))
