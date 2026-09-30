"""Compare direct initial-time marginals with the complete official prediction path."""
import os
import numpy as np
import pytest


@pytest.mark.parametrize('latent',[2,4])
@pytest.mark.parametrize('time_scale',[1.,.005])
def test_initial_marginals_match_official_full_state_prediction(time_scale,latent):
    source=os.environ.get('MGPVAE_SOURCE')
    if not source:pytest.skip('set MGPVAE_SOURCE')
    from baselines.mgpvae.official import make_model
    from baselines.mgpvae.initial import make_initial_predictor
    import jax.numpy as jnp
    import objax
    model=make_model(source,np.array([[0.,0.],[.4,.3],[.9,.6]]),seed=19,
        correct_spatial_covariance=True,latent=latent)
    times=time_scale*jnp.array([0.,.2,.7,1.1,1.8,2.5])[:,None]
    values=jnp.asarray(np.random.default_rng(2).normal(size=(3,6,1)))
    queries=jnp.array([[.1,.2],[.8,.5]])
    predict=make_initial_predictor(model,times,values,queries)
    for shift in [0.,.03]:
        variables=model.vars().subset(objax.TrainVar)
        variables.assign([v+shift for v in variables.tensors()])
        expected=model.predict(times,times,values,queries)
        actual=predict()
        for value,reference in zip(actual,expected):
            np.testing.assert_allclose(value,np.asarray(reference).reshape(2,6,latent),rtol=1e-7,atol=1e-8)
        assert np.all(np.asarray(actual[1])>0)
