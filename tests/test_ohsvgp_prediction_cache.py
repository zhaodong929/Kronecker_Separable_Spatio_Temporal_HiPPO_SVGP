"""Official prediction parity and cache lifetime across query chunks and fits."""
import os
import numpy as np
import pytest
import torch


def test_scoped_basis_cache_is_exact_and_refreshes(monkeypatch):
    from scripts import run_covid_ohsvgp_own_theta as worker
    from scripts.run_traffic_ohsvgp import LazyHiPPOLegS
    from baselines.ohsvgp_prediction import cache_prediction_basis
    from hipposvgp.multidim import SE_kernel
    from hipposvgp.likelihood import GaussianLikelihood
    device=torch.device(os.environ.get('HIPPO_TEST_DEVICE','cpu'))
    torch.manual_seed(48)
    x=np.random.default_rng(2).normal(size=(32,3));x[:,0]=np.arange(32)/32
    model=worker.make_model(kernel=SE_kernel(3,device=device).to(device=device,dtype=torch.float64),
        likelihood=GaussianLikelihood(.2).to(device=device,dtype=torch.float64),
        z_interpolate=x,rff_sample_size=16,previous_steps=0,
        hippo=LazyHiPPOLegS(4,device,torch.float64),inducing_size=4,old_state=None,
        device=device,dtype=torch.float64)
    with torch.no_grad():model.mv.add_(.3)
    base=model.kernel.sample_from_spectral(16).detach()
    queries=np.random.default_rng(9).normal(size=(1030,3))
    reference=worker.predict;cached=cache_prediction_basis(reference)
    original=model.get_Z;calls=[]
    def counted(w):calls.append(1);return original(w)
    model.get_Z=counted
    for shift in [0.,.1]:
        with torch.no_grad():model.kernel.log_ls.add_(shift)
        w=base/torch.exp(model.kernel.log_ls)[None,:]
        monkeypatch.setattr(worker,'predict',reference)
        expected=reference(model,w,queries,device=device,dtype=torch.float64)
        assert len(calls)>=6
        calls.clear();monkeypatch.setattr(worker,'predict',cached)
        actual=cached(model,w,queries,device=device,dtype=torch.float64)
        assert len(calls)==1
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
        assert model.get_Z is counted and not hasattr(model,'_scoped_prediction_basis')
        calls.clear()
    def broken(*args,**kwargs):raise RuntimeError('query failed')
    with pytest.raises(RuntimeError,match='query failed'):
        cache_prediction_basis(broken)(model,w,queries)
    assert model.get_Z is counted and not hasattr(model,'_scoped_prediction_basis')
