"""Compare the actual full loss and every derivative with the dense correction."""
import os
import numpy as np
import pytest


@pytest.mark.parametrize('latent',[2,4])
def test_compact_energy_and_all_gradients_match_dense(latent):
    source=os.environ.get('MGPVAE_SOURCE')
    if not source:pytest.skip('set MGPVAE_SOURCE')
    from baselines.mgpvae.official import make_model
    import jax
    import jax.numpy as jnp
    import objax
    coordinates=np.array([[0.,0.],[.4,.3],[.9,.6]])
    reference=make_model(source,coordinates,seed=17,latent=latent,correct_spatial_covariance=True)
    compact=make_model(source,coordinates,seed=17,latent=latent,correct_spatial_covariance=True,compact_spatial_marginals=True)
    values=jnp.asarray(np.random.default_rng(31).normal(size=(3,6,1)))
    times=(jnp.arange(6.)/2015)[:,None]
    def evaluate(model):
        derivative=objax.GradValues(model.energy,model.vars())
        @objax.Function.with_vars(model.vars())
        def calculate():return derivative(values,jax.random.PRNGKey(12),t=times,num_samples=4)
        return objax.Jit(calculate)()
    for shift in [0.,.02]:
        variables=reference.vars().subset(objax.TrainVar)
        variables.assign([x+shift for x in variables.tensors()])
        compact.vars().assign(reference.vars().tensors())
        _,_,_,covariance,*_=reference.update_posterior(values,dt=jnp.concatenate([jnp.array([0.]),jnp.diff(times[:,0])]))
        diagonal=np.diagonal(np.asarray(covariance),axis1=-2,axis2=-1)
        np.testing.assert_array_equal(covariance,diagonal[...,None]*np.eye(3))
        rg,rl=evaluate(reference);cg,cl=evaluate(compact)
        np.testing.assert_allclose(cl,rl,rtol=1e-10,atol=1e-10)
        assert len(rg)==len(cg)
        for a,b in zip(cg,rg):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
