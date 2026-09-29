"""Explicit covariance correction to the pinned official MGPVAE objective.

Upstream model.py:832 left-multiplies posterior covariance by Lss only.
For f=Lss u, covariance must be Lss Cov(u) Lss.T. Keep the external source
untouched and expose this correction separately from causal filtering.
"""


def mix_spatial_moments(mixing, mean, covariance):
    import jax.numpy as jnp
    return (jnp.einsum('ij,tljk->tlik', mixing, mean),
            jnp.einsum('ij,tljk,nk->tlin', mixing, covariance, mixing))


def corrected_energy(self, Y, train_rng, t=None, num_samples=1):
    """Official energy expression with only the spatial covariance pushforward fixed."""
    import jax.numpy as jnp
    from jax import random, vmap
    dt = jnp.array([0] + list(jnp.diff(t, axis=0)[:, 0])) if t is not None else self.dt
    if self.time_transform is not None:
        raise NotImplementedError()
    _, _, mean, covariance, pseudo_y, pseudo_var, log_lik = self.update_posterior(Y, dt=dt)
    marginal = jnp.diagonal(covariance, axis1=-2, axis2=-1)[..., None]
    kl = self.compute_kl(pseudo_y, pseudo_var, mean, marginal, log_lik) * self.beta
    mean, covariance = mix_spatial_moments(self.kernel.Lss, mean, covariance)
    marginal = jnp.diagonal(covariance, axis1=-2, axis2=-1)[..., None]
    keys = random.split(train_rng, Y.shape[0])
    expected_density = vmap(self.expected_density, (0, 2, 2, 0, None, None))(
        Y, mean, marginal, keys, None, num_samples)
    negative_varexp = -jnp.nansum(expected_density)
    return negative_varexp + kl, negative_varexp, kl
