"""Joint linear-mean SSGP Kronecker model.

The implementation here is a new NumPy/SciPy CPU path. It does not reuse or
modify the baseline PyTorch training modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .kron_utils import (
    dense_A_from_factors,
    inv_spd,
    kron_mv,
    solve_spd,
    solve_sylvester_precision,
    symmetrize,
    unvec_f,
    vec_f,
)
from .ssgp_transfer import (
    compute_Lt,
    projected_prior_transfer_dense,
    transfer_information_matrix,
    transfer_temporal_precision,
    update_information_matrix,
    update_temporal_likelihood_stat,
)
from .structured_state import StructuredKronState


@dataclass(frozen=True)
class Prediction:
    mean: float
    variance: float


class JointSSGPKronHiPPOSVGP:
    """Structured joint online GP with SSGP-style old-likelihood transfer."""

    def __init__(
        self,
        *,
        Ks: np.ndarray,
        C: np.ndarray,
        sigma2: float,
        beta_prior_mean: np.ndarray,
        beta_prior_cov: np.ndarray,
        jitter: float = 1e-6,
    ) -> None:
        self.Ks = symmetrize(np.asarray(Ks, dtype=float))
        self.C = np.asarray(C, dtype=float)
        self.sigma2 = float(sigma2)
        self.beta_prior_mean = np.asarray(beta_prior_mean, dtype=float)
        self.beta_prior_cov = symmetrize(np.asarray(beta_prior_cov, dtype=float))
        self.jitter = float(jitter)
        self.G = symmetrize(self.C.T @ self.C)
        self.Ks_inv = inv_spd(self.Ks, jitter=self.jitter)

    def initialize_state(self, Kt0: np.ndarray, M_t: int | None = None, M_s: int | None = None, d: int | None = None) -> StructuredKronState:
        Kt0 = symmetrize(np.asarray(Kt0, dtype=float))
        mt = int(M_t or Kt0.shape[0])
        ms = int(M_s or self.Ks.shape[0])
        beta_dim = int(d if d is not None else self.beta_prior_mean.shape[0])
        return StructuredKronState(
            beta_mean=np.zeros(beta_dim) if beta_dim == 0 else self.beta_prior_mean.copy(),
            beta_cov=np.zeros((0, 0)) if beta_dim == 0 else self.beta_prior_cov.copy(),
            M_u=np.zeros((ms, mt)),
            B_temporal=np.zeros((mt, mt)),
            H_info=np.zeros((ms, mt)),
            Kt_current=Kt0,
            Ks=self.Ks,
            G=self.G,
            sigma2=self.sigma2,
            metadata={"method": "initial"},
        )

    def _beta_prior_from_state(
        self,
        state: StructuredKronState | None,
        beta_drift: np.ndarray | None,
        d: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if d == 0:
            return np.zeros(0), np.zeros((0, 0))
        mean = self.beta_prior_mean.copy() if state is None else state.beta_mean.copy()
        cov = self.beta_prior_cov.copy() if state is None else state.beta_cov.copy()
        if beta_drift is not None:
            cov = cov + symmetrize(np.asarray(beta_drift, dtype=float))
        return mean, symmetrize(cov)

    def _update_beta(
        self,
        y_vec: np.ndarray,
        Phi: np.ndarray,
        pred_gp_vec: np.ndarray,
        beta_prior_mean: np.ndarray,
        beta_prior_cov: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        d = Phi.shape[1]
        if d == 0:
            return np.zeros(0), np.zeros((0, 0))
        prior_prec = inv_spd(beta_prior_cov, jitter=self.jitter)
        precision = symmetrize(prior_prec + (Phi.T @ Phi) / self.sigma2)
        rhs = prior_prec @ beta_prior_mean + Phi.T @ (y_vec - pred_gp_vec) / self.sigma2
        beta_mean = solve_spd(precision, rhs, jitter=self.jitter)
        beta_cov = inv_spd(precision, jitter=self.jitter)
        return beta_mean, beta_cov

    def _transfer_old_stats(
        self,
        state: StructuredKronState | None,
        Kt_new: np.ndarray,
        K_on_t: np.ndarray | None,
        *,
        no_transfer: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        mt_new = Kt_new.shape[0]
        if state is None or no_transfer:
            return np.zeros((mt_new, mt_new)), np.zeros((self.Ks.shape[0], mt_new))
        if K_on_t is None:
            if state.mt != mt_new:
                raise ValueError("K_on_t is required when temporal basis size changes")
            L_t = np.eye(mt_new)
        else:
            L_t = compute_Lt(K_on_t, Kt_new, jitter=self.jitter)
        return (
            transfer_temporal_precision(state.B_temporal, L_t),
            transfer_information_matrix(state.H_info, L_t),
        )

    def update_block_ssgp_transfer(
        self,
        *,
        y_vec: np.ndarray,
        Phi: np.ndarray,
        T_n: np.ndarray,
        Kt_new: np.ndarray,
        state: StructuredKronState | None = None,
        K_on_t: np.ndarray | None = None,
        beta_drift: np.ndarray | None = None,
        inner_iters: int = 2,
        no_transfer: bool = False,
    ) -> StructuredKronState:
        y_vec = np.asarray(y_vec, dtype=float).reshape(-1)
        Phi = np.asarray(Phi, dtype=float)
        T_n = np.asarray(T_n, dtype=float)
        Kt_new = symmetrize(np.asarray(Kt_new, dtype=float))
        ns, ms = self.C.shape
        nt, mt = T_n.shape
        if y_vec.size != ns * nt:
            raise ValueError(f"Expected y_vec length {ns * nt}, got {y_vec.size}")
        if Phi.shape[0] != y_vec.size:
            raise ValueError("Phi row count must match y_vec length")

        B_trans, H_trans = self._transfer_old_stats(state, Kt_new, K_on_t, no_transfer=no_transfer)
        beta_prior_mean, beta_prior_cov = self._beta_prior_from_state(state, beta_drift, Phi.shape[1])
        M_u = np.zeros((ms, mt)) if state is None or no_transfer or state.M_u.shape[1] != mt else state.M_u.copy()
        beta_mean = beta_prior_mean.copy()
        beta_cov = beta_prior_cov.copy()
        B_new = B_trans.copy()
        H_new = H_trans.copy()
        Kt_inv = inv_spd(Kt_new, jitter=self.jitter)

        for _ in range(max(1, inner_iters)):
            pred_gp_vec = kron_mv(T_n, self.C, M_u)
            beta_mean, beta_cov = self._update_beta(y_vec, Phi, pred_gp_vec, beta_prior_mean, beta_prior_cov)
            residual_vec = y_vec - Phi @ beta_mean
            residual_matrix = unvec_f(residual_vec, (ns, nt))
            B_new = update_temporal_likelihood_stat(B_trans, T_n, self.sigma2)
            H_new = update_information_matrix(H_trans, self.C, residual_matrix, T_n, self.sigma2)
            M_u = solve_sylvester_precision(Kt_inv, self.Ks_inv, B_new, self.G, H_new, jitter=self.jitter)

        return StructuredKronState(
            beta_mean=beta_mean,
            beta_cov=beta_cov,
            M_u=M_u,
            B_temporal=B_new,
            H_info=H_new,
            Kt_current=Kt_new,
            Ks=self.Ks,
            G=self.G,
            sigma2=self.sigma2,
            metadata={"method": "ssgp_transfer", "inner_iters": inner_iters},
        )

    def update_block_no_transfer(self, **kwargs: Any) -> StructuredKronState:
        kwargs = dict(kwargs)
        kwargs["no_transfer"] = True
        return self.update_block_ssgp_transfer(**kwargs).copy_with(metadata={"method": "no_transfer"})

    def update_block_projected_prior(
        self,
        *,
        y_vec: np.ndarray,
        Phi: np.ndarray,
        T_n: np.ndarray,
        Kt_new: np.ndarray,
        state: StructuredKronState | None = None,
        K_on_t: np.ndarray | None = None,
        beta_drift: np.ndarray | None = None,
        inner_iters: int = 2,
    ) -> StructuredKronState:
        """Projected-prior ablation.

        This path intentionally materializes dense matrices and is not the scalable
        SSGP transfer implementation.
        """

        if state is None:
            return self.update_block_ssgp_transfer(
                y_vec=y_vec,
                Phi=Phi,
                T_n=T_n,
                Kt_new=Kt_new,
                state=None,
                K_on_t=None,
                beta_drift=beta_drift,
                inner_iters=inner_iters,
            ).copy_with(metadata={"method": "projected_prior"})
        if K_on_t is None:
            if state.mt != Kt_new.shape[0]:
                raise ValueError("K_on_t is required for projected-prior changing-basis transfer")
            K_on_t = state.Kt_current

        y_vec = np.asarray(y_vec, dtype=float).reshape(-1)
        Phi = np.asarray(Phi, dtype=float)
        T_n = np.asarray(T_n, dtype=float)
        Kt_new = symmetrize(np.asarray(Kt_new, dtype=float))
        ns, ms = self.C.shape
        nt, mt = T_n.shape
        A = dense_A_from_factors(T_n, self.C)
        K_oo = np.kron(state.Kt_current, self.Ks)
        K_nn = np.kron(Kt_new, self.Ks)
        K_on = np.kron(K_on_t, self.Ks)
        K_no = K_on.T
        old_cov = state.metadata.get("u_cov_dense")
        if old_cov is None:
            old_cov = inv_spd(state.dense_precision(jitter=self.jitter), jitter=self.jitter)
        m_proj, S_proj = projected_prior_transfer_dense(
            state.dense_mean(),
            old_cov,
            K_oo,
            K_nn,
            K_no,
            K_on,
            jitter=self.jitter,
        )
        prior_prec_u = inv_spd(S_proj, jitter=self.jitter)
        prior_info_u = prior_prec_u @ m_proj
        beta_prior_mean, beta_prior_cov = self._beta_prior_from_state(state, beta_drift, Phi.shape[1])
        beta_mean = beta_prior_mean.copy()
        beta_cov = beta_prior_cov.copy()
        M_u = unvec_f(m_proj, (ms, mt))
        cov_u = S_proj
        for _ in range(max(1, inner_iters)):
            pred_gp_vec = A @ vec_f(M_u)
            beta_mean, beta_cov = self._update_beta(y_vec, Phi, pred_gp_vec, beta_prior_mean, beta_prior_cov)
            residual_vec = y_vec - Phi @ beta_mean
            precision_u = symmetrize(prior_prec_u + (A.T @ A) / self.sigma2)
            info_u = prior_info_u + A.T @ residual_vec / self.sigma2
            mean_u = solve_spd(precision_u, info_u, jitter=self.jitter)
            cov_u = inv_spd(precision_u, jitter=self.jitter)
            M_u = unvec_f(mean_u, (ms, mt))

        residual_matrix = unvec_f(y_vec - Phi @ beta_mean, (ns, nt))
        B_new = update_temporal_likelihood_stat(np.zeros((mt, mt)), T_n, self.sigma2)
        H_new = update_information_matrix(np.zeros((ms, mt)), self.C, residual_matrix, T_n, self.sigma2)
        return StructuredKronState(
            beta_mean=beta_mean,
            beta_cov=beta_cov,
            M_u=M_u,
            B_temporal=B_new,
            H_info=H_new,
            Kt_current=Kt_new,
            Ks=self.Ks,
            G=self.G,
            sigma2=self.sigma2,
            metadata={"method": "projected_prior", "u_cov_dense": cov_u},
        )

    def predict(
        self,
        *,
        phi_star: np.ndarray,
        t_proj_star: np.ndarray,
        c_proj_star: np.ndarray,
        state: StructuredKronState,
        include_variance: bool = True,
    ) -> Prediction:
        phi_star = np.asarray(phi_star, dtype=float).reshape(-1)
        t_proj_star = np.asarray(t_proj_star, dtype=float).reshape(-1)
        c_proj_star = np.asarray(c_proj_star, dtype=float).reshape(-1)
        beta_mean = 0.0 if phi_star.size == 0 else float(phi_star @ state.beta_mean)
        gp_mean = float(c_proj_star @ state.M_u @ t_proj_star)
        variance = self.sigma2
        if include_variance:
            if phi_star.size:
                variance += float(phi_star @ state.beta_cov @ phi_star)
            projected_prior_var = float((c_proj_star @ self.Ks @ c_proj_star) * (t_proj_star @ state.Kt_current @ t_proj_star))
            variance += max(0.0, 1.0 - projected_prior_var)
            Q = np.outer(c_proj_star, t_proj_star)
            if "u_cov_dense" in state.metadata:
                q = vec_f(Q)
                variance += float(q @ state.metadata["u_cov_dense"] @ q)
            else:
                Kt_inv = inv_spd(state.Kt_current, jitter=self.jitter)
                Z = solve_sylvester_precision(Kt_inv, self.Ks_inv, state.B_temporal, state.G, Q, jitter=self.jitter)
                variance += float(np.sum(Q * Z))
        return Prediction(mean=beta_mean + gp_mean, variance=max(float(variance), self.jitter))

    def materialize_dense_for_test(self, state: StructuredKronState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Lambda = state.dense_precision(jitter=self.jitter)
        h = state.dense_information()
        mean_dense = solve_spd(Lambda, h, jitter=self.jitter)
        return Lambda, h, mean_dense
