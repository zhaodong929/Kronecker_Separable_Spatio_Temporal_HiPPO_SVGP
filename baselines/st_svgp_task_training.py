"""Memory-bounded Gaussian ST-SVGP VI for a fixed spatial training grid.

This is an algebraic storage specialization of BayesNewton's MarkovVariationalGP:
Gaussian likelihood sites have scalar shared precision, not dense Ns x Ns
precision matrices at each time. The inducing posterior, official filter/RTS,
KL expression and alternating site/Adam hyperparameter updates are unchanged.
"""
import numpy as np
# Install the repository's compatibility shims before importing pinned BayesNewton.
from baselines.covid_long_setting_b.adapters import run_st_svgp as official
import bayesnewton
import jax.numpy as jnp
from jax import vmap
import objax
from bayesnewton.utils import inv, gaussian_expected_log_lik


class CompactGaussianST(objax.Module):
    def __init__(self, *, kernel, likelihood, times, coordinates, targets):
        if (not isinstance(kernel, bayesnewton.kernels.SpatioTemporalKernel)
                or not isinstance(kernel.temporal_kernel, bayesnewton.kernels.Matern32)):
            raise ValueError('Compact ST training requires the qualified stationary Matern32 temporal kernel')
        if not isinstance(likelihood, bayesnewton.likelihoods.Gaussian):
            raise ValueError('Compact ST training requires Gaussian observations')
        t, r, y = map(np.asarray, (times, coordinates, targets))
        if t.ndim == 1: t = t[:, None]
        if (t.ndim != 2 or t.shape[1] != 1 or not len(t) or not np.isfinite(t).all()):
            raise ValueError('Training times must be a finite nonempty column')
        if (r.ndim != 2 or not len(r) or not np.isfinite(r).all()
                or y.shape != (len(t), len(r)) or not np.isfinite(y).all()):
            raise ValueError('Compact ST training requires finite observations on a fixed complete grid')
        if np.any(np.diff(t[:, 0]) < 0):
            raise ValueError('Training times must be ordered')
        self.kernel, self.likelihood = kernel, likelihood
        self.X, self.R, self.Y = jnp.asarray(t), jnp.asarray(r), jnp.asarray(y)
        self.dt = jnp.concatenate([jnp.zeros(1, dtype=self.X.dtype), jnp.diff(self.X[:, 0])])
        self.parallel = False
        self.num_data = len(t)
        self.nat1 = objax.StateVar(jnp.zeros((*y.shape, 1)))
        self.precision = objax.StateVar(jnp.asarray(.01))  # official initial site covariance 100 I
        m = kernel.measurement_model().shape[0]
        self.posterior_mean = objax.StateVar(jnp.zeros((len(t), m, 1)))
        self.posterior_variance = objax.StateVar(jnp.broadcast_to(jnp.eye(m), (len(t), m, m)))

    def spatial(self):
        b, c = self.kernel.spatial_conditional(self.X[:1], self.R[None])
        return b[0], c[0]

    def compute_full_pseudo_lik(self):
        b, _ = self.spatial()
        precision = b.T @ (self.precision.value*b)
        covariance = inv(precision+1e-12*jnp.eye(precision.shape[0]))
        mean = jnp.einsum('ij,tjk->tik', covariance@b.T, self.nat1.value)
        return mean, jnp.broadcast_to(covariance, (self.num_data, *covariance.shape))

    def update_posterior(self):
        pseudo_y, pseudo_var = self.compute_full_pseudo_lik()
        _, (fm, fp) = bayesnewton.ops.kalman_filter(self.dt, self.kernel, pseudo_y, pseudo_var,
            mask=None, parallel=False)
        dt = jnp.concatenate([self.dt[1:], jnp.zeros(1, dtype=self.dt.dtype)])
        mean, covariance, _ = bayesnewton.ops.rauch_tung_striebel_smoother(dt, self.kernel, fm, fp,
            parallel=False)
        self.posterior_mean.value, self.posterior_variance.value = mean, covariance

    def inference(self, lr=1., **kwargs):
        if kwargs:
            raise ValueError('Compact ST inference supports full-data Gaussian natural updates only')
        # Gaussian expected-log-likelihood Hessian is -I/noise and natural
        # mean is y/noise, independent of the old posterior mean/variance.
        noise = self.likelihood.variance
        self.nat1.value = (1-lr)*self.nat1.value+lr*self.Y[..., None]/noise
        self.precision.value = (1-lr)*self.precision.value+lr/noise
        self.update_posterior()

    def compute_kl(self):
        pseudo_y, pseudo_var = self.compute_full_pseudo_lik()
        log_lik, _ = bayesnewton.ops.kalman_filter(self.dt, self.kernel, pseudo_y, pseudo_var,
            mask=None, parallel=False)
        expected = vmap(gaussian_expected_log_lik, (0, 0, 0, 0, None))(
            pseudo_y, self.posterior_mean.value, self.posterior_variance.value, pseudo_var, None)
        return jnp.sum(expected)-log_lik

    def energy(self):
        b, c = self.spatial()
        mean = jnp.einsum('nm,tmk->tnk', b, self.posterior_mean.value)[..., 0]
        variance = jnp.einsum('ni,tij,nj->tn', b, self.posterior_variance.value, b)+jnp.diag(c)
        noise = self.likelihood.variance
        expected = -.5*(jnp.log(2*jnp.pi*noise)+((self.Y-mean)**2+variance)/noise)
        return -jnp.sum(expected)+self.compute_kl()


def make_compact_model(times, spatial_grid, targets, inducing_locations, *, trainable_inducing=False, theta=None):
    """Same official model/kernel family, without allocating dense observation sites."""
    r = np.asarray(spatial_grid)
    if (r.ndim != 3 or r.shape[0] != len(times) or not len(r)
            or not np.array_equal(r, np.broadcast_to(r[:1], r.shape))):
        raise ValueError('Compact ST training requires the same spatial grid at every time')
    theta = theta or dict(kernel_variance=1., ell_t=.2, ell_s=(1., 1.), noise_std=np.sqrt(.1))
    temporal = bayesnewton.kernels.Matern32(variance=theta['kernel_variance'], lengthscale=theta['ell_t'])
    spatial = bayesnewton.kernels.Separable([
        bayesnewton.kernels.Matern32(variance=1., lengthscale=theta['ell_s'][0]),
        bayesnewton.kernels.Matern32(variance=1., lengthscale=theta['ell_s'][1])])
    kernel = bayesnewton.kernels.SpatioTemporalKernel(temporal_kernel=temporal, spatial_kernel=spatial,
        z=inducing_locations, sparse=True, opt_z=trainable_inducing, conditional='Full')
    return CompactGaussianST(kernel=kernel,
        likelihood=bayesnewton.likelihoods.Gaussian(variance=theta['noise_std']**2),
        times=times, coordinates=r[0], targets=targets)
