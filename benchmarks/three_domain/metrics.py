"""Scores for equal-weight Gaussian decoder mixtures (scalar outcomes)."""
import numpy as np
from scipy.special import logsumexp, ndtr


def gaussian_mixture_metrics(truth, component_means, noise_variance):
    y = np.asarray(truth, dtype=float).reshape(-1)
    mu = np.asarray(component_means, dtype=float)
    v = float(noise_variance)
    if mu.ndim != 2 or mu.shape[1] != y.size or not np.isfinite(mu).all() or not np.isfinite(y).all() or not np.isfinite(v) or v <= 0:
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
