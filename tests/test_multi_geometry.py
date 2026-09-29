import numpy as np
import pytest
import torch
from scipy.linalg import block_diag
from stvgp_kronecker.joint_ssgp_kron.multi_geometry import TorchMultiGeometryHiPPOSVGP, SumKronSolver


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('d',[0,2])
def test_multigeometry_transfer_and_prediction_match_dense_joint_gaussian(d,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    rng=np.random.default_rng(9);ms=3;mt=2;noise=.4
    ks=np.eye(ms)*1.7; beta_cov=np.eye(d)*2.;beta_mean=np.zeros(d)
    visible=rng.normal(size=(4,ms))*.2;hidden=rng.normal(size=(2,ms))*.2
    model=TorchMultiGeometryHiPPOSVGP(Ks=ks,C=visible,sigma2=noise,
          beta_prior_mean=beta_mean,beta_prior_cov=beta_cov,prior_point_variance=3.,jitter=0.,device=device)
    state=None;r=np.zeros((d+ms*mt,d+ms*mt));h=np.zeros(d+ms*mt)
    for step,c in enumerate([visible,hidden,visible,hidden,visible]):
        oldmt=mt
        if step==2:mt=3
        kt=np.eye(mt)*1.2
        l=None
        if step>=2:
            l=rng.normal(size=(oldmt,mt))*.1+np.eye(oldmt,mt)*.8
            transfer=block_diag(np.eye(d),np.kron(l,np.eye(ms)))
            r=transfer.T@r@transfer;h=transfer.T@h
        t=rng.normal(size=(2,mt))*.2
        phi=rng.normal(size=(2*len(c),d))
        y=rng.normal(size=2*len(c))
        design=np.column_stack([phi,np.kron(t,c)])
        r+=design.T@design/noise;h+=design.T@y/noise
        prior=block_diag(np.eye(d)/2.,np.kron(np.linalg.inv(kt),np.linalg.inv(ks)))
        covariance=np.linalg.inv(prior+r);mean=covariance@h
        state=model.update_block_structured_joint_ssgp_transfer(y_vec=y,Phi=phi,T_n=t,
               Kt_new=kt,C_observed=c,state=state,L_t_override=l)
        actual=np.r_[state.beta_mean.cpu().numpy(),state.M_u.cpu().numpy().reshape(-1,order='F')]
        np.testing.assert_allclose(actual,mean,rtol=2e-7,atol=1e-9)
        np.testing.assert_allclose(state.beta_cov.cpu(),covariance[:d,:d],rtol=2e-7,atol=1e-9)
        cq=rng.normal(size=(3,ms))*.1;tq=rng.normal(size=(2,mt))*.1;pq=rng.normal(size=(6,d))
        xq=np.column_stack([pq,np.kron(tq,cq)])
        gp=xq[:,d:]; prior_u=np.kron(kt,ks)
        residual=3.-np.einsum('ni,ij,nj->n',gp,prior_u,gp)
        expected_var=noise+residual+np.einsum('ni,ij,nj->n',xq,covariance,xq)
        pred,var,_=model.predict_with_C(state=state,T_eval=tq,Phi=pq,C_eval=cq,
                 include_conditional_residual_variance=True,validate_conditional_residual_variance=True)
        np.testing.assert_allclose(pred,xq@mean,rtol=2e-7,atol=1e-9)
        np.testing.assert_allclose(var,expected_var,rtol=2e-7,atol=1e-9)
        assert len(state.precision_terms)==min(step+1,2)
        assert state.tensor_bytes()>0


def test_sum_solver_checks_residual_and_refuses_unconverged_result():
    rng=np.random.default_rng(3)
    def spd(n):
        x=rng.normal(size=(n,n));return torch.tensor(x@x.T+np.eye(n))
    ki,si=spd(5),spd(3);terms=[(spd(3),spd(5)),(spd(3),spd(5))]
    rhs=torch.tensor(rng.normal(size=(15,4)));rhs[:,0]=0
    solve=SumKronSolver(ki,si,terms)
    result=solve.solve(rhs)
    dense=torch.kron(ki,si)+sum(torch.kron(b,g) for g,b in terms)
    torch.testing.assert_close(result,torch.linalg.solve(dense,rhs),rtol=1e-7,atol=1e-9)
    with pytest.raises(RuntimeError,match='did not converge'):
        SumKronSolver(ki,si,terms,max_iterations=1).solve(rhs)
