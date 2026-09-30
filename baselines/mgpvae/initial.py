"""Exact spatial prediction at the observed initial times, without dense state lifting."""
import numpy as np


def make_initial_predictor(model,times,training,query_coordinates):
    import jax.numpy as jnp
    import objax
    n=model.kernel.Ns
    h=np.asarray(model.kernel.measurement_model())
    expected=np.kron(np.eye(n),np.array([[1.,0.]]))
    if not np.array_equal(h,np.broadcast_to(expected,h.shape)):
        raise ValueError('Direct initial marginals require the pinned identity measurement')

    @objax.Function.with_vars(model.vars())
    def predict():
        dt=jnp.concatenate([jnp.array([0.]),jnp.diff(times[:,0])])
        _,_,mean,covariance,_,_,_=model.update_posterior(training,dt=dt)
        # The pinned identity measurement and block-diagonal state projection
        # make latent spatial covariance diagonal. Official spatial conditional
        # supplies the same B and residual C as model.predict.
        mixing,residual=model.kernel.spatial_conditional(times,query_coordinates)
        latent_mean=jnp.einsum('rn,tln->rtl',mixing,mean[...,0])
        latent_variance=jnp.einsum('rn,tln->rtl',mixing**2,
            jnp.diagonal(covariance,axis1=-2,axis2=-1))
        latent_variance=latent_variance+jnp.diagonal(residual,axis1=-2,axis2=-1).transpose(2,0,1)
        return latent_mean,latent_variance

    return objax.Jit(predict)


def verify_initial_prefix(model,times,training,query_coordinates):
    """Real spatial geometry, short time prefix against untouched official prediction."""
    times=times[:3];training=training[:,:3]
    expected=model.predict(times,times,training,query_coordinates)
    actual=make_initial_predictor(model,times,training,query_coordinates)()
    errors={}
    for name,value,reference in zip(['mean','variance'],actual,expected):
        value,reference=np.asarray(value),np.asarray(reference).reshape(value.shape)
        np.testing.assert_allclose(value,reference,rtol=1e-6,atol=1e-7)
        errors[name]=float(np.max(abs(value-reference)))
    return dict(status='passed',max_absolute_error=errors,time_steps=3,
        fitting_sites=int(training.shape[0]),query_sites=int(query_coordinates.shape[0]))
