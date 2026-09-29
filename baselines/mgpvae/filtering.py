"""Stateful visible-site filtering using the unmodified official MGPVAE step.

This adapter does not yet implement assimilation of delayed held-out labels.
It is therefore a qualification candidate, not a certified main-table baseline.
"""
import numpy as np


class OfficialVisibleFilter:
    def __init__(self, model):
        import jax
        import jax.numpy as jnp
        from mgpvae.ops import _sequential_kf_spatiotemporal, process_noise_covariance
        self.model = model
        kernel = model.kernel
        kernel.precompute_spatial_mixing()
        self.Pinf = kernel.stationary_covariance()
        self.mean = jnp.zeros((*self.Pinf.shape[:-1], 1))
        self.covariance = self.Pinf
        self.time = None
        H = kernel.measurement_model()

        def step(mean, covariance, dt, y):
            pseudo_y, pseudo_var = model.compute_full_pseudo_lik(y[:, None, :])
            A = kernel.state_transition(dt)
            Q = process_noise_covariance(A, self.Pinf)
            def one(m, p, a, q, h, py, pv):
                _, fm, fp = _sequential_kf_spatiotemporal(
                    a[None], q[None], h, py[None, :, None], pv[None, :, None],
                    m, p, jnp.zeros((1, 1, 1), dtype=bool), kernel.block_index)
                return fm[0], fp[0]
            return jax.vmap(one)(mean, covariance, A, Q, H, pseudo_y[0], pseudo_var[0])
        self._step = jax.jit(step)

    def observe(self, time, visible_targets):
        y = np.asarray(visible_targets, dtype=float).reshape(-1, 1)
        if y.shape[0] != self.model.kernel.Ns or not np.isfinite(y).all():
            raise ValueError('One finite observation per fixed visible site is required')
        if self.time is not None and time <= self.time:
            raise ValueError('Observations must arrive in strictly increasing time order')
        dt = 0. if self.time is None else float(time - self.time)
        self.mean, self.covariance = self._step(self.mean, self.covariance, dt, y)
        self.time = float(time)
        return self.mean, self.covariance

    def latent_at(self, coordinates):
        import jax.numpy as jnp
        if self.time is None:
            raise ValueError('Observe a time slice before prediction')
        B, C = self.model.kernel.spatial_conditional(jnp.array([self.time]), jnp.asarray(coordinates))
        mean = jnp.einsum('rs,ls->rl', B, self.mean[..., 0, 0])
        variance = jnp.einsum('rs,ls->rl', B**2, self.covariance[..., 0, 0])
        variance = variance + jnp.diagonal(C[0], axis1=-2, axis2=-1).T
        return mean, variance

    def gaussian_components(self, coordinates, *, seed, samples=128):
        """Return decoder means and noise variances, not a moment-matched score."""
        import jax
        import jax.numpy as jnp
        mean, variance = self.latent_at(coordinates)
        if not np.isfinite(variance).all() or np.any(np.asarray(variance) <= 0):
            raise ValueError('Invalid official latent predictive variance')
        z = mean[None] + jnp.sqrt(variance)[None] * jax.random.normal(
            jax.random.PRNGKey(seed), (samples, *mean.shape))
        components = self.model.likelihood.decoder(z)[..., 0]
        return np.asarray(components), float(self.model.likelihood.variance)
