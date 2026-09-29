"""Scores for equal-weight Gaussian decoder mixtures (scalar outcomes)."""
import numpy as np
from scipy.special import logsumexp, ndtr


def gaussian_mixture_metrics(truth, component_means, noise_variance):
    y = np.asarray(truth, dtype=float).reshape(-1)
    mu = np.asarray(component_means, dtype=float)
    v = float(noise_variance)
    if mu.ndim != 2 or mu.shape[0] == 0 or y.size == 0 or mu.shape[1] != y.size or not np.isfinite(mu).all() or not np.isfinite(y).all() or not np.isfinite(v) or v <= 0:
        raise ValueError('Expected finite (samples, queries) component means and positive noise variance')
    def a(d, s):
        z = d/s
        return 2*s*np.exp(-.5*z*z)/np.sqrt(2*np.pi) + d*(2*ndtr(z)-1)
    lp = -.5*(np.log(2*np.pi*v) + (y[None]-mu)**2/v)
    crps = np.mean(a(y[None]-mu, np.sqrt(v)), axis=0)
    for start in range(0, y.size, 32):
        m = mu[:, start:start+32]
        crps[start:start+32] -= .5*np.mean(a(m[:,None]-m[None,:], np.sqrt(2*v)),axis=(0,1))
    return dict(rmse=float(np.sqrt(np.mean((y-mu.mean(axis=0))**2))),
                nlpd=float(-np.mean(logsumexp(lp,axis=0)-np.log(mu.shape[0]))),
                crps=float(np.mean(crps)))


def gaussian_mixture_calibration(truth, component_means, noise_variance, levels=None):
    """Central interval coverage via the exact finite-mixture CDF (PIT).

    For a continuous strictly increasing CDF, checking F(y) inside the central
    probability interval is equivalent to numerically inverting its quantiles.
    This avoids a Gaussian moment-match and adds no observation-sampling noise.
    Latent Monte Carlo error in component_means remains a separate qualification.
    """
    y = np.asarray(truth, dtype=float).reshape(-1)
    mu = np.asarray(component_means, dtype=float)
    v = float(noise_variance)
    if (mu.ndim != 2 or mu.shape[0] == 0 or y.size == 0 or mu.shape[1] != y.size
            or not np.isfinite(mu).all() or not np.isfinite(y).all()
            or not np.isfinite(v) or v <= 0):
        raise ValueError('Finite nonempty mixture and positive variance required')
    levels = np.linspace(.05, .95, 10) if levels is None else np.asarray(levels, dtype=float)
    if levels.ndim != 1 or not levels.size or not np.isfinite(levels).all() or np.any((levels <= 0) | (levels >= 1)):
        raise ValueError('Coverage levels must lie strictly between zero and one')
    pit = np.mean(ndtr((y[None] - mu)/np.sqrt(v)), axis=0)
    covered = (pit[None] >= (1-levels[:,None])/2) & (pit[None] <= (1+levels[:,None])/2)
    coverage = covered.mean(axis=1)
    return dict(ece=float(np.mean(np.abs(coverage-levels))),
                levels=levels.tolist(), coverage=coverage.tolist())
