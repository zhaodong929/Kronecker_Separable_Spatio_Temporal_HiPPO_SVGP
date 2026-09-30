"""Exact corrected MGPVAE training with independent 2x2 smoothing blocks.

The official whitened measurement is I_sites x [1, 0]. Filtering and RTS
smoothing therefore factor across sites before the physical Lss mixing. This
implementation never constructs a time-indexed sites-by-sites covariance.
The static spatial covariance and its Cholesky factor remain dense O(Ns**2).
"""


def validate_compact_training_model(model):
    """Validate the mathematical preconditions outside any traced/JIT function."""
    import numpy as np
    # Construction through official.make_model checks the source pin. The
    # measurement check must still precede tracing: no implicit diagonalization.
    n = model.kernel.Ns
    h = np.asarray(model.kernel.measurement_model())
    expected = np.kron(np.eye(n), np.array([[1., 0.]]))
    if not np.array_equal(h, np.broadcast_to(expected, h.shape)):
        raise ValueError('Compact training requires the pinned identity measurement')
    if model.parallel or model.time_transform is not None:
        raise ValueError('Compact training supports untransformed sequential inference only')
    if np.asarray(model.kernel.stationary_covariance()).shape[-2:] != (2, 2):
        raise ValueError('Compact training requires the qualified two-state Matern32 blocks')
    model._compact_task_training_validated = True
    return model


def compact_training_posterior(model, Y, dt):
    """Return whitened marginal mean/variance, pseudo factors and log normalizer."""
    if not getattr(model, '_compact_task_training_validated', False):
        raise ValueError('Call validate_compact_training_model before tracing compact training')
    import jax.numpy as jnp
    from jax import vmap
    from mgpvae.ops import rauch_tung_striebel_smoother
    from .sitewise_filter import sitewise_official_filter
    pseudo_y, pseudo_var = model.compute_full_pseudo_lik(Y)
    model.kernel.precompute_spatial_mixing()
    log_lik, (fm, fp) = sitewise_official_filter(dt, model.kernel, pseudo_y, pseudo_var)
    smooth_dt = jnp.concatenate([dt[1:], jnp.zeros(1, dtype=dt.dtype)])
    transitions = vmap(model.kernel.state_transition)(smooth_dt)
    pinf = model.kernel.stationary_covariance()

    def site(mean, covariance, transition, stationary):
        block_index = (jnp.array([0, 0, 1, 1]), jnp.array([0, 1, 0, 1]))
        sm, sp, _ = rauch_tung_striebel_smoother(transition[:, None], stationary[None],
            jnp.array([[1., 0.]], dtype=mean.dtype), mean[:, None], covariance[:, None],
            block_index=block_index, return_full=False, spatiotemporal=True)
        return sm[:, 0, 0], sp[:, 0, 0]

    sites = vmap(site, (1, 1, 1, 0))
    mean, variance = vmap(sites, (0, 0, 1, 0))(fm, fp, transitions, pinf)
    mean = mean.transpose(2, 0, 1)[..., None]
    variance = variance.transpose(2, 0, 1)[..., None]
    return mean, variance, pseudo_y, pseudo_var, log_lik


def compact_training_energy(self, Y, train_rng, t=None, num_samples=1):
    """Same corrected ELBO, KL and per-site/time MC keys as the dense reference."""
    import jax.numpy as jnp
    from jax import random, vmap
    dt = jnp.concatenate([jnp.zeros(1, dtype=t.dtype), jnp.diff(t[:, 0])]) if t is not None else self.dt
    if dt is None:
        raise ValueError('Training requires explicit model times or time differences')
    mean, variance, pseudo_y, pseudo_var, log_lik = compact_training_posterior(self, Y, dt)
    kl = self.compute_kl(pseudo_y, pseudo_var, mean, variance, log_lik) * self.beta
    physical_mean = jnp.einsum('ij,tljk->tlik', self.kernel.Lss, mean)
    physical_variance = jnp.einsum('ij,tlj->tli', self.kernel.Lss**2, variance[..., 0])[..., None]
    keys = random.split(train_rng, Y.shape[0])
    expected = vmap(self.expected_density, (0, 2, 2, 0, None, None))(
        Y, physical_mean, physical_variance, keys, None, num_samples)
    negative_varexp = -jnp.nansum(expected)
    return negative_varexp + kl, negative_varexp, kl
