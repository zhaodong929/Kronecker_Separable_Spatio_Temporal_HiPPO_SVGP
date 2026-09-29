"""Check parameter gradients through the pinned authors' RFF ELBO."""
from pathlib import Path
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("previous", [0, 17, 91])
def test_lazy_legendre_matches_pinned_official_transition(previous):
    from scripts.run_traffic_ohsvgp import LazyHiPPOLegS
    from hipposvgp.hippo import HiPPO_LegS
    x = torch.randn(7, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(11))
    initial = torch.randn(6, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(12))
    official = HiPPO_LegS(4, "cpu", max_length=100)
    adapted = LazyHiPPOLegS(4, torch.device("cpu"), torch.float64)
    for fast in (False, True):
        expected = official(x, prev_discrete_steps=previous, ini=initial, fast=fast)
        actual = adapted(x, prev_discrete_steps=previous, ini=initial, fast=fast)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_official_rff_lengthscale_gradient_matches_finite_difference():
    source = Path(__file__).resolve().parents[1]/'baselines/external/harrisonzhu508_HIPPOSVGP'
    if not source.exists():
        pytest.skip('pinned HIPPOSVGP checkout required')
    from scripts.run_covid_ohsvgp_own_theta import make_model
    from hipposvgp.multidim import SE_kernel
    from hipposvgp.likelihood import GaussianLikelihood
    from hipposvgp.hippo import HiPPO_LegS
    torch.manual_seed(17)
    x = np.random.default_rng(7).normal(size=(8,3))
    x[:,0] = np.arange(8.)/8
    model = make_model(kernel=SE_kernel(3), likelihood=GaussianLikelihood(.2),
                       z_interpolate=x, rff_sample_size=32, previous_steps=0,
                       hippo=HiPPO_LegS(4, 'cpu', max_length=16), inducing_size=4,
                       old_state=None, device=torch.device('cpu'), dtype=torch.float64)
    base = (model.kernel.sample_from_spectral(32)*torch.exp(model.kernel.log_ls)[None]).detach()
    xt = torch.tensor(x,dtype=torch.float64)
    yt = torch.sin(xt[:,:1]*3)
    def loss():
        torch.manual_seed(101)  # upstream ELBO samples latent likelihood values
        w=base/torch.exp(model.kernel.log_ls)[None]
        return -model.ELBO(xt,yt,w,recompute_k=True,cache_k=False)[0]
    loss().backward()
    gradient=model.kernel.log_ls.grad.detach().clone()
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 1e-6
    eps=1e-5
    expected=[]
    with torch.no_grad():
        for j in range(3):
            model.kernel.log_ls[j] += eps
            plus=float(loss())
            model.kernel.log_ls[j] -= 2*eps
            minus=float(loss())
            model.kernel.log_ls[j] += eps
            expected.append((plus-minus)/(2*eps))
    np.testing.assert_allclose(gradient,expected,rtol=2e-3,atol=1e-4)


@pytest.mark.parametrize('variance', [.5, 1., 2.])
def test_frozen_old_covariance_is_unchanged_by_amplitude(variance):
    from scripts.run_covid_ohsvgp_own_theta import make_model, export_state
    from scripts.run_traffic_ohsvgp import LazyHiPPOLegS
    from hipposvgp.multidim import SE_kernel
    from hipposvgp.likelihood import GaussianLikelihood
    torch.manual_seed(14)
    x = np.random.default_rng(17).normal(size=(8,3))
    kernel = SE_kernel(3).to(dtype=torch.float64)
    with torch.no_grad():
        kernel.log_sf.fill_(np.log(variance))
    common = dict(kernel=kernel, likelihood=GaussianLikelihood(.2), rff_sample_size=16,
        hippo=LazyHiPPOLegS(4,torch.device('cpu'),torch.float64), inducing_size=4,
        device=torch.device('cpu'),dtype=torch.float64)
    first = make_model(**common,z_interpolate=x,previous_steps=0,old_state=None)
    w = first.kernel.sample_from_spectral(16).detach()
    state = export_state(first,w)
    second = make_model(**common,z_interpolate=x,previous_steps=8,old_state=state)
    covariance = second.KuuKfuKff_rff_se(w,torch.tensor(x))[5]
    torch.testing.assert_close(covariance*torch.exp(second.kernel.log_sf),state['Kaa'])
    objective = second.ELBO(torch.tensor(x),torch.zeros((8,1)),w)[0]
    assert torch.isfinite(objective)
