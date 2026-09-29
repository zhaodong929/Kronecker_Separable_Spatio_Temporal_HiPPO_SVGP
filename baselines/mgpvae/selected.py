"""Linear-cost exact selected-row update for the pinned identity measurement H.

Upstream whitening makes measurement_model=I_space kron [1,0]; posterior
projection is spatially block diagonal. The dense selected-row solve therefore
reduces exactly to one scalar innovation per observed site and latent channel.
Parameters stay frozen. Fall back to the reference if upstream H changes.
"""
import numpy as np
from .partial import OneStepDelayedFilter


class SelectedSiteFilter(OneStepDelayedFilter):
    def __init__(self, model):
        super().__init__(model)
        import jax
        import jax.numpy as jnp
        from mgpvae.ops import process_noise_covariance
        expected = np.kron(np.eye(model.kernel.Ns), np.array([[1., 0.]]))
        h = np.asarray(model.kernel.measurement_model())
        if not np.array_equal(h, np.broadcast_to(expected, h.shape)):
            raise ValueError('Optimized selected-row update requires the pinned identity measurement')

        def step(mean, covariance, dt, sites, values):
            a = model.kernel.state_transition(dt)
            q = process_noise_covariance(a, self.Pinf)
            mean = a @ mean
            covariance = a @ covariance @ jnp.swapaxes(a, -1, -2) + q
            py, pv = model.compute_full_pseudo_lik(values[:, None, None])
            m, p = mean[:, sites], covariance[:, sites]
            gain = p[..., :, 0] / (p[..., 0, 0] + pv[0])[..., None]
            m = m + gain[..., None] * (py[0]-m[..., 0, 0])[..., None, None]
            p = p - gain[..., :, None] * p[..., 0, None, :]
            return mean.at[:, sites].set(m), covariance.at[:, sites].set(p)
        self._selected_step = jax.jit(step)

    def selected_step(self, mean, cov, dt, observations):
        import jax.numpy as jnp
        observations = sorted(observations)
        sites = jnp.asarray([s for s, _ in observations], dtype=int)
        values = jnp.asarray([v for _, v in observations], dtype=float)
        return self._selected_step(mean, cov, dt, sites, values)
