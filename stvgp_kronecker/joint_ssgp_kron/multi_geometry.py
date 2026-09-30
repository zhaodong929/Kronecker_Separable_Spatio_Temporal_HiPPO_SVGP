"""Exact finite-Gaussian updates for several fixed observation geometries.

Store a sum B_j (x) G_j, rather than incorrectly merging different G_j.
Use residual-checked, Sylvester-preconditioned conjugate gradients. This is
NOT the single-Sylvester solver or its fixed-geometry complexity guarantee.
No raw observation history or dense inducing precision is retained.
"""
from dataclasses import dataclass, field
import torch
from .torch_backend import (
    TorchJointSSGPKronHiPPOSVGP, TorchStructuredKronState, _as_tensor,
    symmetrize, inv_spd, solve_spd, vec_f, unvec_f, _transfer_beta_u,
    _unvec_f_columns, _vec_f_batch, generalized_eigh,
)


class SumKronSolver:
    def __init__(self, kt_inv, ks_inv, terms, *, tolerance=1e-9, max_iterations=512):
        self.kt_inv, self.ks_inv, self.terms = kt_inv, ks_inv, terms
        self.ms, self.mt = ks_inv.shape[0], kt_inv.shape[0]
        self.tolerance, self.max_iterations = tolerance, max_iterations
        g = sum(pair[0] for pair in terms)
        b = sum(pair[1] for pair in terms)
        evs, self.p = generalized_eigh(g, ks_inv, jitter=0.)
        evt, self.q = generalized_eigh(b, kt_inv, jitter=0.)
        self.denominator = 1 + evs[:, None]*evt[None, :]
        if not torch.isfinite(self.denominator).all() or (self.denominator <= 0).any():
            raise FloatingPointError('Nonpositive sum-Kronecker preconditioner')
        self.last_iterations = 0
        self.last_relative_residual = 0.
        self.max_relative_residual = 0.

    def apply(self, rhs):
        x = _unvec_f_columns(rhs, (self.ms,self.mt))
        out = self.ks_inv @ x @ self.kt_inv
        for g,b in self.terms:
            out = out + g @ x @ b
        return _vec_f_batch(out)

    def precondition(self, rhs):
        x = _unvec_f_columns(rhs, (self.ms,self.mt))
        return _vec_f_batch(self.p @ ((self.p.T @ x @ self.q)/self.denominator) @ self.q.T)

    def solve(self, rhs):
        vector = rhs.ndim == 1
        rhs = rhs[:,None] if vector else rhs
        if rhs.shape[1] == 0:
            return rhs
        norm = torch.linalg.vector_norm(rhs,dim=0)
        scale = norm.clamp_min(torch.finfo(rhs.dtype).tiny)
        x = torch.zeros_like(rhs)
        r = rhs.clone()
        z = self.precondition(r)
        p = z.clone()
        rz = (r*z).sum(dim=0)
        active = norm > 0
        for iteration in range(1,self.max_iterations+1):
            ap = self.apply(p)
            pap = (p*ap).sum(dim=0)
            if ((pap <= 0) & active).any() or not torch.isfinite(pap).all():
                raise FloatingPointError('Sum-Kronecker CG lost positive definiteness')
            alpha = torch.where(active,rz/torch.where(active,pap,torch.ones_like(pap)),0.)
            x = x + p*alpha
            r = r - ap*alpha
            relative = torch.linalg.vector_norm(r,dim=0)/scale
            if ((relative <= self.tolerance) | ~active).all():
                # Never trust only the recurrent CG residual.
                r = rhs-self.apply(x)
                relative = torch.linalg.vector_norm(r,dim=0)/scale
                if (relative <= self.tolerance*10).all():
                    self.last_iterations = iteration
                    self.last_relative_residual = float(relative.max())
                    self.max_relative_residual = max(self.max_relative_residual,self.last_relative_residual)
                    return x[:,0] if vector else x
                # Restart from the explicitly recomputed residual.
                active = relative > self.tolerance
                z = self.precondition(r); p=z.clone(); rz=(r*z).sum(dim=0)
                continue
            active = relative > self.tolerance
            z = self.precondition(r)
            next_rz = (r*z).sum(dim=0)
            ratio = torch.where(active,next_rz/torch.where(active,rz,torch.ones_like(rz)),0.)
            p = z+p*ratio
            p[:,~active] = 0
            rz = next_rz
        residual = float((torch.linalg.vector_norm(rhs-self.apply(x),dim=0)/scale).max())
        raise RuntimeError(f'Sum-Kronecker CG did not converge: residual={residual:g}')


@dataclass(frozen=True)
class MultiGeometryState(TorchStructuredKronState):
    precision_terms: tuple = ()
    W: torch.Tensor | None = None
    solver: object = field(default=None,repr=False)

    def tensor_bytes(self):
        seen=set()
        def visit(x):
            if torch.is_tensor(x):
                key=x.untyped_storage().data_ptr()
                if key in seen:return 0
                seen.add(key);return x.untyped_storage().nbytes()
            if isinstance(x,(tuple,list)):return sum(visit(v) for v in x)
            if isinstance(x,dict):return sum(visit(v) for v in x.values())
            if isinstance(x,SumKronSolver):return visit(vars(x))
            return 0
        return visit(vars(self))


class TorchMultiGeometryHiPPOSVGP(TorchJointSSGPKronHiPPOSVGP):
    """Same transported joint likelihood, with multiple spatial Gram factors."""
    def update_block_structured_joint_ssgp_transfer(
        self, *, y_vec, Phi, T_n, Kt_new, state=None, K_on_t=None,
        C_observed=None, beta_drift=None, no_transfer=False, L_t_override=None, zero_cross=False,
    ):
        if self.dtype != torch.float64:
            raise ValueError("Multi-geometry qualification requires float64")
        to=lambda x:_as_tensor(x,device=self.device,dtype=self.dtype)
        c=self.C if C_observed is None else to(C_observed)
        y,phi,t,kt=to(y_vec).reshape(-1),to(Phi),to(T_n),symmetrize(to(Kt_new))
        ns,ms=c.shape; nt,mt=t.shape; d=phi.shape[1]
        if ms!=self.Ks.shape[0] or y.numel()!=ns*nt or phi.shape[0]!=y.numel():
            raise ValueError('Observation geometry, targets and features disagree')
        if not all(torch.isfinite(v).all() for v in (c,y,phi,t,kt)):
            raise ValueError('Nonfinite observed input')
        gram=symmetrize(c.T@c)
        terms=[]
        rbb=phi.T@phi/self.sigma2
        rbu=phi.new_zeros((d,ms*mt))
        hb=phi.T@y/self.sigma2
        hi=c.T@unvec_f(y,(ns,nt))@t/self.sigma2
        if d:
            blocks=phi.T.reshape(d,nt,ns).transpose(1,2)
            rbu=(c.T@blocks@t).transpose(1,2).contiguous().reshape(d,ms*mt)/self.sigma2
        if state is not None and not no_transfer:
            if not isinstance(state,MultiGeometryState):
                raise TypeError('Multi-geometry update requires its matching state')
            if L_t_override is not None: l=to(L_t_override)
            elif K_on_t is not None: l=solve_spd(kt,to(K_on_t).T,jitter=self.jitter).T
            else:
                if state.mt!=mt:raise ValueError('Basis size changed without a transfer')
                l=torch.eye(mt,device=self.device,dtype=self.dtype)
            if tuple(l.shape)!=(state.mt,mt):raise ValueError('Invalid temporal transfer shape')
            rbb=rbb+state.R_beta_beta
            rbu=rbu+_transfer_beta_u(state.R_beta_u,l,ms)
            hb=hb+state.h_beta; hi=hi+state.H_info@l
            terms=[(g,symmetrize(l.T@b@l)) for g,b in state.precision_terms]
        added=symmetrize(t.T@t/self.sigma2)
        for i,(g,b) in enumerate(terms):
            # Only identical geometry is merged, never merely similarly sized.
            if torch.equal(g,gram):
                terms[i]=(g,symmetrize(b+added));break
        else:terms.append((gram,added))
        if zero_cross:
            rbu = torch.zeros_like(rbu)
        prior_cov=self.beta_prior_cov if beta_drift is None else symmetrize(self.beta_prior_cov+to(beta_drift))
        prior_precision=inv_spd(prior_cov,jitter=self.jitter) if d else phi.new_zeros((0,0))
        prior_h=prior_precision@self.beta_prior_mean
        a=symmetrize(prior_precision+rbb); h=prior_h+hb
        solver=SumKronSolver(inv_spd(kt,jitter=self.jitter),self.Ks_inv,tuple(terms))
        solved=solver.solve(torch.cat([rbu.T,vec_f(hi)[:,None]],dim=1))
        w,v=solved[:,:d],solved[:,d]
        schur=symmetrize(a-rbu@w)
        beta_cov=inv_spd(schur,jitter=self.jitter) if d else schur
        beta_mean=solve_spd(schur,h-rbu@v,jitter=self.jitter) if d else h
        return MultiGeometryState(
            beta_mean=beta_mean,beta_cov=beta_cov,M_u=unvec_f(v-w@beta_mean,(ms,mt)),
            B_temporal=sum(b for _,b in terms),H_info=hi,Kt_current=kt,Ks=self.Ks,
            G=self.G,sigma2=self.sigma2,R_beta_beta=symmetrize(rbb),R_beta_u=rbu,h_beta=hb,
            beta_prior_precision=prior_precision,beta_prior_natural=prior_h,
            Lambda_beta_given_u=schur,S_beta_beta=beta_cov,precision_terms=tuple(terms),W=w,solver=solver,
            metadata={'method':'multi_geometry_transported_joint','solver':'sylvester_preconditioned_cg',
                      'spatial_terms':len(terms),'cg_iterations':solver.last_iterations,
                      'relative_residual':solver.last_relative_residual})

    def predict_with_C(self, *, state, T_eval, Phi, C_eval, chunk_size=256,
                       include_conditional_residual_variance=False,
                       validate_conditional_residual_variance=False, return_numpy=True):
        to=lambda x:_as_tensor(x,device=self.device,dtype=self.dtype)
        t,c,phi=to(T_eval),to(C_eval),to(Phi)
        ns,nt=c.shape[0],t.shape[0];n=ns*nt
        if phi.shape[0]!=n:raise ValueError('Prediction feature rows disagree')
        means=[];variances=[];parts=[]
        for start in range(0,n,max(1,min(int(chunk_size),256))):
            stop=min(n,start+max(1,min(int(chunk_size),256)))
            idx=torch.arange(start,stop,device=self.device)
            tc,cc=t[idx//ns],c[idx%ns]
            design=(tc[:,:,None]*cc[:,None,:]).reshape(stop-start,-1)
            solved=state.solver.solve(design.T)
            u=(design*solved.T).sum(dim=1)
            adjusted=phi[start:stop]-design@state.W
            beta=torch.einsum('ni,ij,nj->n',adjusted,state.beta_cov,adjusted)
            projected=((tc@state.Kt_current)*tc).sum(dim=1)*((cc@self.Ks)*cc).sum(dim=1)
            raw=self.prior_point_variance-projected
            if validate_conditional_residual_variance and float(raw.min()) < -1e-8:
                raise FloatingPointError('Materially negative conditional residual variance')
            nu=raw.clamp_min(0.) if include_conditional_residual_variance else torch.zeros_like(raw)
            mean=phi[start:stop]@state.beta_mean+design@vec_f(state.M_u)
            variance=self.sigma2+nu+u+beta
            if not torch.isfinite(variance).all() or (variance<=0).any():
                raise FloatingPointError('Invalid multi-geometry predictive variance')
            means.append(mean);variances.append(variance);parts.append(torch.stack([nu,u,beta],dim=1))
        mean,var=torch.cat(means),torch.cat(variances);avg=torch.cat(parts).mean(dim=0)
        diagnostics=dict(avg_sigma2=self.sigma2,avg_nu_star=float(avg[0]),
                         avg_u_posterior_term=float(avg[1]),avg_beta_schur_term=float(avg[2]))
        return (mean.cpu().numpy(),var.cpu().numpy(),diagnostics) if return_numpy else (mean,var,diagnostics)
