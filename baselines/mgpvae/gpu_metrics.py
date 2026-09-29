"""Exact finite-mixture scores in JAX, avoiding CPU pairwise-CRPS bottlenecks."""
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr,logsumexp


@jax.jit
def _scores(y,mu,v):
    sd=jnp.sqrt(v)
    def absolute_normal(d,s):
        z=d/s
        return 2*s*jnp.exp(-.5*z*z)/jnp.sqrt(2*jnp.pi)+d*(2*ndtr(z)-1)
    log_density=-.5*(jnp.log(2*jnp.pi*v)+(y[None]-mu)**2/v)
    first=jnp.mean(absolute_normal(y[None]-mu,sd))
    second=.5*jnp.mean(absolute_normal(mu[:,None]-mu[None,:],jnp.sqrt(2*v)))
    pit=jnp.mean(ndtr((y[None]-mu)/sd),axis=0)
    levels=jnp.asarray(np.linspace(.05,.95,10),dtype=y.dtype)
    coverage=jnp.mean((pit[None]>=(1-levels[:,None])/2)&(pit[None]<=(1+levels[:,None])/2),axis=1,dtype=y.dtype)
    coverage90=jnp.mean((pit>=(1-.9)/2)&(pit<=(1+.9)/2),dtype=y.dtype)
    return jnp.concatenate([jnp.array([jnp.sqrt(jnp.mean((y-mu.mean(axis=0))**2)),
        -jnp.mean(logsumexp(log_density,axis=0)-jnp.log(mu.shape[0])),first-second,
        jnp.mean(jnp.abs(coverage-levels)),coverage90]),coverage])


def gaussian_mixture_scores(truth,components,noise_variance):
    y=np.asarray(truth,dtype=float).reshape(-1);mu=np.asarray(components,dtype=float);v=float(noise_variance)
    if mu.ndim!=2 or not mu.shape[0] or not y.size or mu.shape[1]!=y.size or not np.isfinite(mu).all() or not np.isfinite(y).all() or not np.isfinite(v) or v<=0:
        raise ValueError('Finite nonempty mixture and positive variance required')
    if not jax.config.read('jax_enable_x64'):raise ValueError('Float64 is required for final mixture scoring')
    values=np.asarray(_scores(jnp.asarray(y),jnp.asarray(mu),jnp.asarray(v)))
    if not np.isfinite(values).all():raise FloatingPointError('Nonfinite mixture score')
    return dict(zip(['rmse','nlpd','crps','ece','coverage90'],map(float,values[:5])),
        levels=np.linspace(.05,.95,10).tolist(),coverage=values[5:].tolist())
