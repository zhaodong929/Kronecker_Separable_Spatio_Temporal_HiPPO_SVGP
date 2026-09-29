"""Check parameter gradients through the pinned authors' RFF ELBO."""
from pathlib import Path
import numpy as np
import pytest
import torch


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
