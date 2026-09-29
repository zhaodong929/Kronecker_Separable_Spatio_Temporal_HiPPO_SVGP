import numpy as np
import pytest
from benchmarks.three_domain.metrics import gaussian_mixture_metrics,gaussian_mixture_calibration


@pytest.mark.parametrize('samples',[1,17,64])
def test_jax_mixture_scores_equal_independent_scipy_formula(samples):
    import jax
    jax.config.update('jax_enable_x64',True)
    from baselines.mgpvae.gpu_metrics import gaussian_mixture_scores
    rng=np.random.default_rng(samples)
    y=rng.normal(size=7);mu=rng.normal(size=(samples,7));noise=.3
    expected={**gaussian_mixture_metrics(y,mu,noise),**gaussian_mixture_calibration(y,mu,noise)}
    expected['coverage90']=gaussian_mixture_calibration(y,mu,noise,[.9])['coverage'][0]
    actual=gaussian_mixture_scores(y,mu,noise)
    for key in ['rmse','nlpd','crps','ece','coverage90','coverage','levels']:
        np.testing.assert_allclose(actual[key],expected[key],rtol=1e-10,atol=1e-10)
