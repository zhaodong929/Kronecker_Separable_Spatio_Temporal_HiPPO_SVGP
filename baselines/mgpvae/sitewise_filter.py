"""Factor the guarded identity-measurement filter into official scalar filters."""


def sitewise_official_filter(dt,kernel,pseudo_y,pseudo_var,mask=None,parallel=False):
    """Keep official transition/noise/KF equations; batch independent sites.

    Used only after make_model verifies H = I_sites x [1,0]. There is no
    spatial mixing in this whitened filter; the official Lss map is downstream.
    """
    if mask is not None or parallel:
        raise ValueError('Sitewise training supports the qualified unmasked sequential filter only')
    import jax.numpy as jnp
    from jax import vmap
    from mgpvae.ops import kalman_filter
    stationary=kernel.stationary_covariance()
    transitions=vmap(kernel.state_transition)(dt)
    h=jnp.asarray([[1.,0.]],dtype=stationary.dtype)

    def latent_filter(a,p,y,r):
        def site_filter(site_a,site_p,site_y,site_r):
            return kalman_filter(site_a,site_p,jnp.zeros((2,1),dtype=p.dtype),
                h,site_y[:,None,None],site_r[:,None,None],
                mask=None,parallel=False,spatiotemporal=False)
        likelihood,(means,covs)=vmap(site_filter,(1,0,1,1))(a,p,y,r)
        return jnp.sum(likelihood),(jnp.swapaxes(means,0,1),jnp.swapaxes(covs,0,1))
    likelihood,states=vmap(latent_filter,(1,0,1,1))(
        transitions,stationary,pseudo_y,pseudo_var)
    return jnp.sum(likelihood),states
