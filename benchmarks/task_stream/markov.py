"""Task boundaries around official observation-wise Markov inference.

Forward updates are unchanged. Task-end marginals use the pinned authors' RTS
smoother and temporal conditional, including their endpoint regularization.
The filtered endpoint, not a smoothed historical state, continues the stream.
"""
import numpy as np
from .window import TaskWindow


class STTaskAdapter:
    def __init__(self, model, coordinates):
        from baselines.st_svgp_filter import GaussianSTFilter
        import jax.numpy as jnp
        self.filter = GaussianSTFilter(model.kernel, model.likelihood, coordinates)
        f = self.filter
        self.window = TaskWindow(lambda m, p, dt, sites, values:
            f.step(m, p, dt, jnp.asarray(sites), jnp.asarray(values)), f.mean, f.covariance)

    def initialize(self, initial):
        self.window.initialize(initial)

    def predict_task(self, task):
        import jax.numpy as jnp
        from bayesnewton.ops import rauch_tung_striebel_smoother
        from bayesnewton.utils import temporal_conditional
        trace = self.window.advance(task)
        times = jnp.asarray(trace.times[:, None])
        dt = jnp.concatenate([jnp.diff(times[:, 0]), jnp.zeros(1)])
        sm, sp, gain = rauch_tung_striebel_smoother(dt, self.filter.kernel,
            jnp.stack(trace.means), jnp.stack(trace.covariances), return_full=True)
        augmented = jnp.concatenate([jnp.array([[-1e10]]), times, jnp.array([[1e10]])])
        mean, covariance = temporal_conditional(augmented, times[trace.query_start:],
                                               sm, sp, gain, self.filter.kernel)
        h = self.filter.h[task.query_sites]
        prediction = jnp.einsum('sd,tdk->tsk', h, mean)[..., 0]
        variance = jnp.einsum('sd,tde,se->ts', h, covariance, h)
        variance += self.filter.conditional[task.query_sites] + self.filter.noise
        return np.asarray(prediction), np.asarray(variance)


def mgp_task_marginals(model, trace, coordinates):
    import jax.numpy as jnp
    from jax import vmap
    from mgpvae.ops import rauch_tung_striebel_smoother
    from mgpvae.util import temporal_conditional
    times = jnp.asarray(trace.times[:, None])
    query = times[trace.query_start:]
    fm, fp = jnp.stack(trace.means, axis=1), jnp.stack(trace.covariances, axis=1)
    dt = jnp.concatenate([jnp.diff(times[:, 0]), jnp.zeros(1)])
    transitions = vmap(model.kernel.state_transition)(dt)
    pinf = model.kernel.stationary_covariance()
    augmented = jnp.concatenate([jnp.array([[-1e10]]), times, jnp.array([[1e10]])])
    indices = jnp.searchsorted(augmented[:, 0], query[:, 0]) - 1
    forward = vmap(model.kernel.state_transition)(query[:, 0] - augmented[indices, 0])
    backward = vmap(model.kernel.state_transition)(augmented[indices + 1, 0] - query[:, 0])

    def site(mean, covariance, transitions, prior, forward, backward):
        index = (jnp.array([0, 0, 1, 1]), jnp.array([0, 1, 0, 1]))
        sm, sp, gain = rauch_tung_striebel_smoother(transitions[:, None], prior[None],
            jnp.array([[1., 0.]]), mean[:, None], covariance[:, None], block_index=index,
            return_full=True, spatiotemporal=True)
        pm, pv = temporal_conditional(augmented, query, sm, sp, gain, forward, backward, prior)
        return pm[:, 0, 0], pv[:, 0, 0]

    sites = vmap(site, (1, 1, 1, 0, 1, 1))
    mean, variance = vmap(sites, (0, 0, 1, 0, 1, 1))(fm, fp, transitions, pinf, forward, backward)
    model.kernel.precompute_spatial_mixing()
    mixing, residual = model.kernel.spatial_conditional(query, jnp.asarray(coordinates))
    # time, query site, latent; use the same physical covariance pushforward.
    result_mean = jnp.einsum('rn,lnt->trl', mixing, mean)
    result_variance = jnp.einsum('rn,lnt->trl', mixing ** 2, variance)
    result_variance += jnp.diagonal(residual, axis1=-2, axis2=-1).transpose(0, 2, 1)
    return result_mean, result_variance


class MGPTaskAdapter:
    predictive_family = 'gaussian_mixture'

    def __init__(self, model, coordinates, *, samples=512, seed=0):
        from baselines.mgpvae.selected import SelectedSiteFilter
        self.filter = SelectedSiteFilter(model)
        self.coordinates = np.asarray(coordinates)
        self.samples, self.seed = int(samples), int(seed)
        if self.samples < 2:
            raise ValueError('At least two decoder samples required')
        f = self.filter
        self.window = TaskWindow(lambda m, p, dt, sites, values:
            f.selected_step(m, p, dt, list(zip(sites, values))), f.mean, f.covariance)
        self.components = None
        self.latent = None

    def initialize(self, initial):
        self.window.initialize(initial)

    def predict_task(self, task):
        import jax
        import jax.numpy as jnp
        trace = self.window.advance(task)
        mean, variance = mgp_task_marginals(self.filter.model, trace, self.coordinates[task.query_sites])
        if not np.isfinite(variance).all() or np.any(np.asarray(variance) <= 0):
            raise FloatingPointError('Invalid MGPVAE latent task variance')
        self.latent = np.asarray(mean), np.asarray(variance)
        z = mean[None] + jnp.sqrt(variance)[None] * jax.random.normal(
            jax.random.PRNGKey(self.seed + task.index), (self.samples, *mean.shape))
        components = self.filter.model.likelihood.decoder(z)[..., 0]
        noise = float(self.filter.model.likelihood.variance)
        self.components = np.asarray(components), noise
        return np.asarray(components.mean(0)), np.asarray(components.var(0)) + noise
