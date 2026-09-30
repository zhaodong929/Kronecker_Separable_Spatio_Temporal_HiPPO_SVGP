"""Exact spatial prediction at the observed initial times, without dense state lifting."""
import numpy as np


def make_initial_predictor(model,times,training,query_coordinates):
    import jax.numpy as jnp
    import objax
    from jax import vmap
    from mgpvae.ops import rauch_tung_striebel_smoother
    from mgpvae.util import temporal_conditional
    n=model.kernel.Ns
    h=np.asarray(model.kernel.measurement_model())
    expected=np.kron(np.eye(n),np.array([[1.,0.]]))
    if not np.array_equal(h,np.broadcast_to(expected,h.shape)):
        raise ValueError('Direct initial marginals require the pinned identity measurement')

    @objax.Function.with_vars(model.vars())
    def predict():
        dt=jnp.concatenate([jnp.array([0.]),jnp.diff(times[:,0])])
        pseudo_y,pseudo_var=model.compute_full_pseudo_lik(training)
        model.kernel.precompute_spatial_mixing()
        _,(fm,fp)=model.filter(dt,model.kernel,pseudo_y,pseudo_var,parallel=model.parallel)
        smooth_dt=jnp.concatenate([dt[1:],jnp.array([0.])])
        transitions=vmap(model.kernel.state_transition)(smooth_dt)
        pinf=model.kernel.stationary_covariance()
        augmented=jnp.concatenate([jnp.array([[-1e10]]),times,jnp.array([[1e10]])])
        indices=jnp.searchsorted(augmented[:,0],times[:,0])-1
        forward=vmap(model.kernel.state_transition)(times[:,0]-augmented[indices,0])
        backward=vmap(model.kernel.state_transition)(augmented[indices+1,0]-times[:,0])
        # Official independent spatial blocks permit the exact full-state RTS
        # and bridge interpolation to be evaluated on 2x2 blocks. Retain the
        # official bridge's 1e-8 jitter even at observed query times.
        def site(fm,fp,transitions,pinf,forward,backward):
            block_index=(jnp.array([0,0,1,1]),jnp.array([0,1,0,1]))
            sm,sp,gain=rauch_tung_striebel_smoother(transitions[:,None],pinf[None],
                jnp.array([[1.,0.]]),fm[:,None],fp[:,None],block_index=block_index,
                return_full=True,spatiotemporal=True)
            pm,pv=temporal_conditional(augmented,times,sm,sp,gain,forward,backward,pinf)
            return pm[:,0,0],pv[:,0,0]
        sites=vmap(site,(1,1,1,0,1,1))
        mean,variance=vmap(sites,(0,0,1,0,1,1))(fm,fp,transitions,pinf,forward,backward)
        mixing,residual=model.kernel.spatial_conditional(times,query_coordinates)
        latent_mean=jnp.einsum('rn,lnt->rtl',mixing,mean)
        latent_variance=jnp.einsum('rn,lnt->rtl',mixing**2,variance)
        latent_variance=latent_variance+jnp.diagonal(residual,axis1=-2,axis2=-1).transpose(2,0,1)
        return latent_mean,latent_variance

    # Keep the official JAX scan kernels, but do not fuse the encoder, filter,
    # smoother and spatial projection into one outer executable. On the pinned
    # CUDA/JAX stack, the outer JIT silently changed some 720-site latent-4
    # filtered states; eager execution matches CPU and official prediction.
    # This is a compilation boundary, not a change to the posterior equations.
    return predict


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
