"""Causal-prefix reference for per-site missingness and delayed observations.

Preserves upstream spatial mean-field block projection after each time slice.
Selects observed measurement rows exactly; absent targets never enter encoder.
Parameters must remain frozen while comparing prefixes. Replay cost is explicit.
This inference reference does not qualify masked training or the full benchmark.
"""
import numpy as np
from .filtering import OfficialVisibleFilter


class PartialPrefixFilter(OfficialVisibleFilter):
    def __init__(self, model):
        super().__init__(model)
        self._records = {}
        self.replayed_steps = 0

    def release(self, time, site_indices, values, *, available_at):
        """Register each (historical time, site) once, at its release time."""
        time, available_at = float(time), float(available_at)
        indices = np.asarray(site_indices)
        values = np.asarray(values, dtype=float).reshape(-1)
        if not np.isfinite([time, available_at]).all() or available_at < time:
            raise ValueError('Release cannot precede observation time')
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('Site indices must be a one-dimensional integer array')
        if len(indices) != len(values) or len(set(indices)) != len(indices):
            raise ValueError('Distinct sites and matching values required')
        if np.any(indices < 0) or np.any(indices >= self.model.kernel.Ns) or not np.isfinite(values).all():
            raise ValueError('Invalid released observation')
        keys = [(time, int(i)) for i in indices]
        if any(k in self._records for k in keys):
            raise ValueError('Observation already registered; duplicate assimilation forbidden')
        self._records.update({k: (float(v), available_at) for k,v in zip(keys, values)})

    def infer(self, as_of):
        """Recompute the lawful prefix; no unreleased values reach the model."""
        import jax
        import jax.numpy as jnp
        from mgpvae.ops import process_noise_covariance
        as_of = float(as_of)
        if not np.isfinite(as_of):
            raise ValueError('Finite prediction time required')
        kernel = self.model.kernel
        H = kernel.measurement_model()
        mean, cov = jnp.zeros((*self.Pinf.shape[:-1], 1)), self.Pinf
        grouped = {}
        for (t, site), (value, release) in self._records.items():
            if t <= as_of and release <= as_of:
                grouped.setdefault(t, []).append((site, value))
        if not grouped:
            raise ValueError('No available observations')
        previous = min(grouped)
        for t in sorted(set(grouped) | {as_of}):
            A = kernel.state_transition(t - previous)
            Q = process_noise_covariance(A, self.Pinf)
            mean, cov = A @ mean, A @ cov @ jnp.swapaxes(A, -1, -2) + Q
            observations = sorted(grouped.get(t, []))
            if observations:
                indices = np.array([s for s,_ in observations], dtype=int)
                y = jnp.array([v for _,v in observations])[:, None, None]
                py, pv = self.model.compute_full_pseudo_lik(y)
                def update(m, p, h, encoded, variance):
                    ns, dim = m.shape[:2]
                    full = jnp.zeros((ns*dim, ns*dim)).at[kernel.block_index].set(p.flatten())
                    h = h[indices]
                    hp = h @ full
                    innovation = hp @ h.T + jnp.diag(variance)
                    gain = jnp.linalg.solve(innovation, hp).T
                    updated_mean = m.reshape(-1, 1) + gain @ (encoded[:,None] - h @ m.reshape(-1,1))
                    updated_cov = full - gain @ hp
                    return updated_mean.reshape(m.shape), updated_cov[kernel.block_index].reshape(p.shape)
                mean, cov = jax.vmap(update)(mean, cov, H, py[0], pv[0])
            previous = t
            self.replayed_steps += 1
        self.mean, self.covariance, self.time = mean, cov, as_of
        return mean, cov
