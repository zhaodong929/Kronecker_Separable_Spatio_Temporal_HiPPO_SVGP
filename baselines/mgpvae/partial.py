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

    def selected_step(self, mean, cov, dt, observations):
        """One upstream-equivalent projected update, selecting observed rows."""
        import jax
        import jax.numpy as jnp
        from mgpvae.ops import process_noise_covariance
        kernel = self.model.kernel
        A = kernel.state_transition(dt)
        Q = process_noise_covariance(A, self.Pinf)
        mean, cov = A @ mean, A @ cov @ jnp.swapaxes(A, -1, -2) + Q
        if not observations:
            return mean, cov
        observations = sorted(observations)
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
        return jax.vmap(update)(mean, cov, kernel.measurement_model(), py[0], pv[0])

    def infer(self, as_of):
        """Recompute the lawful prefix; no unreleased values reach the model."""
        import jax.numpy as jnp
        as_of = float(as_of)
        if not np.isfinite(as_of):
            raise ValueError('Finite prediction time required')
        mean, cov = jnp.zeros((*self.Pinf.shape[:-1], 1)), self.Pinf
        grouped = {}
        for (t, site), (value, release) in self._records.items():
            if t <= as_of and release <= as_of:
                grouped.setdefault(t, []).append((site, value))
        if not grouped:
            raise ValueError('No available observations')
        previous = min(grouped)
        for t in sorted(set(grouped) | {as_of}):
            mean, cov = self.selected_step(mean, cov, t-previous, grouped.get(t, []))
            previous = t
            self.replayed_steps += 1
        self.mean, self.covariance, self.time = mean, cov, as_of
        return mean, cov


class OneStepDelayedFilter(PartialPrefixFilter):
    """Bounded-state inference equivalent to replay for previous-step releases.

    Keep the state BEFORE the most recent update. At the next step recompute
    that update with its visible and newly released hidden observations jointly,
    then process the current observations. Thus block projection occurs once per
    time slice, matching the reference rather than double-counting visible data.
    Only frozen parameters and releases for the immediately prior step supported.
    """
    def __init__(self, model):
        super().__init__(model)
        self._previous_observations = []
        self._before_previous = None
        self._previous_dt = 0.

    def advance(self, time, sites, values, *, delayed_time=None,
                delayed_sites=(), delayed_values=()):
        time = float(time)
        if not np.isfinite(time) or (self.time is not None and time <= self.time):
            raise ValueError('Advance requires a strictly increasing finite time')
        # Reuse validation without retaining the full observation history.
        check = PartialPrefixFilter.__new__(PartialPrefixFilter)
        check.model, check._records = self.model, {}
        check.release(time, sites, values, available_at=time)
        current = [(s, v) for (_,s),(v,_) in check._records.items()]
        previous = self._previous_observations.copy()
        if delayed_time is not None:
            if self.time is None or float(delayed_time) != self.time:
                raise ValueError('Delayed observations must belong to the preceding step')
            check._records = {(self.time,s):(v,self.time) for s,v in previous}
            check.release(delayed_time, delayed_sites, delayed_values, available_at=time)
            previous = [(s,v) for (_,s),(v,_) in check._records.items()]
        elif len(delayed_sites) or len(delayed_values):
            raise ValueError('delayed_time is required')
        if delayed_time is not None:
            self.mean, self.covariance = self.selected_step(
                *self._before_previous, self._previous_dt, previous)
            self.replayed_steps += 1
        dt = 0. if self.time is None else time-self.time
        before = (self.mean, self.covariance)
        mean, cov = self.selected_step(*before, dt, current)
        self._before_previous, self._previous_dt = before, dt
        self._previous_observations = current
        self.mean, self.covariance, self.time = mean, cov, time
        self.replayed_steps += 1
        return mean, cov
