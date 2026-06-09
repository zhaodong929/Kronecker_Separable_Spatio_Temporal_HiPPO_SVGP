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
    dense_Du_for_tests,
    dense_joint_posterior_reference,
    inv_spd,
    kron_mv,
    schur_recover_posterior,
    solve_Du_sylvester,
    solve_spd,
    solve_sylvester_precision,
    symmetrize,
    unvec_f,
    vec_f,
)
from .ssgp_transfer import (
    compute_Lt,
    joint_likelihood_stats,
    projected_prior_transfer_dense,
    transfer_joint_old_likelihood,
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


@dataclass(frozen=True)
class VarianceDecomposition:
    sigma2: float
    nu_star: float
    u_posterior_term: float
    beta_schur_term: float
    total_variance: float


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
        prior_point_variance: float = 1.0,
        jitter: float = 1e-6,
    ) -> None:
        self.Ks = symmetrize(np.asarray(Ks, dtype=float))
        self.C = np.asarray(C, dtype=float)
        self.sigma2 = float(sigma2)
        self.beta_prior_mean = np.asarray(beta_prior_mean, dtype=float)
        self.beta_prior_cov = symmetrize(np.asarray(beta_prior_cov, dtype=float))
        self.prior_point_variance = float(prior_point_variance)
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

    def _routeB_transfer_old_stats(
        self,
        state: StructuredKronState | None,
        Kt_new: np.ndarray,
        K_on_t: np.ndarray | None,
        *,
        no_transfer: bool = False,
        beta_dim: int,
    ) -> dict[str, np.ndarray]:
        """Transfer Route-B old likelihood statistics to the new temporal basis."""

        mt_new = Kt_new.shape[0]
        ms = self.Ks.shape[0]
        if state is None or no_transfer:
            return {
                "R_beta_beta": np.zeros((beta_dim, beta_dim)),
                "R_beta_u": np.zeros((beta_dim, ms * mt_new)),
                "h_beta": np.zeros(beta_dim),
                "B_temporal": np.zeros((mt_new, mt_new)),
                "H_info": np.zeros((ms, mt_new)),
            }
        if K_on_t is None:
            if state.mt != mt_new:
                raise ValueError("K_on_t is required when temporal basis size changes")
            L_t = np.eye(mt_new)
        else:
            L_t = compute_Lt(K_on_t, Kt_new, jitter=self.jitter)

        R_beta_beta = (
            np.zeros((beta_dim, beta_dim))
            if state.R_beta_beta is None
            else np.asarray(state.R_beta_beta, dtype=float)
        )
        R_beta_u = (
            np.zeros((beta_dim, ms * state.mt))
            if state.R_beta_u is None
            else np.asarray(state.R_beta_u, dtype=float)
        )
        h_beta = np.zeros(beta_dim) if state.h_beta is None else np.asarray(state.h_beta, dtype=float)
        return transfer_joint_old_likelihood(
            R_beta_beta=R_beta_beta,
            R_beta_u=R_beta_u,
            h_beta=h_beta,
            B_temporal=state.B_temporal,
            H_info=state.H_info,
            L_t=L_t,
            M_s=ms,
        )

    def solve_Du(self, state: StructuredKronState, rhs: np.ndarray) -> np.ndarray:
        Kt_inv = inv_spd(state.Kt_current, jitter=self.jitter)
        Ks_inv = inv_spd(state.Ks, jitter=self.jitter)
        return solve_Du_sylvester(
            Kt_inv,
            Ks_inv,
            state.B_temporal,
            state.G,
            rhs,
            jitter=self.jitter,
        )

    def recover_posterior_mean_structured(
        self,
        *,
        A_beta: np.ndarray,
        B_beta_u: np.ndarray,
        h_beta: np.ndarray,
        h_u: np.ndarray,
        Kt_new: np.ndarray,
        B_temporal: np.ndarray,
    ) -> dict[str, np.ndarray]:
        Kt_inv = inv_spd(Kt_new, jitter=self.jitter)
        return schur_recover_posterior(
            A_beta,
            B_beta_u,
            h_beta,
            h_u,
            Kt_inv,
            self.Ks_inv,
            B_temporal,
            self.G,
            jitter=self.jitter,
        )

    def recover_beta_covariance(self, state: StructuredKronState) -> np.ndarray:
        if state.S_beta_beta is None:
            raise ValueError("Route-B beta covariance is not available")
        return state.S_beta_beta

    def apply_cross_covariance_beta_u(self, state: StructuredKronState, rhs_u: np.ndarray) -> np.ndarray:
        """Apply implicit Route-B ``S_beta_u`` to a vector in u-space."""

        if state.R_beta_u is None or state.S_beta_beta is None:
            raise ValueError("Route-B cross covariance is not available")
        v = self.solve_Du(state, rhs_u)
        return -state.S_beta_beta @ (state.R_beta_u @ v)

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

    def update_block_structured_joint_ssgp_transfer(
        self,
        *,
        y_vec: np.ndarray,
        Phi: np.ndarray,
        T_n: np.ndarray,
        Kt_new: np.ndarray,
        state: StructuredKronState | None = None,
        K_on_t: np.ndarray | None = None,
        beta_drift: np.ndarray | None = None,
        no_transfer: bool = False,
    ) -> StructuredKronState:
        """Route-B structured joint update.

        This is the main non-mean-field path. It stores likelihood natural
        statistics for ``z=[beta; u]``:
        ``R_beta_beta``, ``R_beta_u``, ``B_temporal kron G``, ``h_beta`` and
        ``H_info``. Posterior means and beta covariance are recovered by the
        Schur complement while ``D_u`` solves use the Sylvester-compatible
        Kronecker structure.
        """

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

        d = Phi.shape[1]
        old_stats = self._routeB_transfer_old_stats(
            state,
            Kt_new,
            K_on_t,
            no_transfer=no_transfer,
            beta_dim=d,
        )
        new_stats = joint_likelihood_stats(y_vec, Phi, T_n, self.C, self.sigma2)
        R_beta_beta = symmetrize(old_stats["R_beta_beta"] + new_stats["R_beta_beta"])
        R_beta_u = old_stats["R_beta_u"] + new_stats["R_beta_u"]
        h_beta_lik = old_stats["h_beta"] + new_stats["h_beta"]
        B_temporal = symmetrize(old_stats["B_temporal"] + new_stats["B_temporal"])
        H_info = old_stats["H_info"] + new_stats["H_info"]

        prior_mean, prior_cov = self._beta_prior_from_state(None, beta_drift, d)
        beta_prior_precision = np.zeros((0, 0)) if d == 0 else inv_spd(prior_cov, jitter=self.jitter)
        beta_prior_natural = np.zeros(0) if d == 0 else beta_prior_precision @ prior_mean
        A_beta = symmetrize(beta_prior_precision + R_beta_beta)
        h_beta_total = beta_prior_natural + h_beta_lik
        schur = self.recover_posterior_mean_structured(
            A_beta=A_beta,
            B_beta_u=R_beta_u,
            h_beta=h_beta_total,
            h_u=vec_f(H_info),
            Kt_new=Kt_new,
            B_temporal=B_temporal,
        )
        M_u = unvec_f(schur["m_u"], (ms, mt))
        return StructuredKronState(
            beta_mean=schur["m_beta"],
            beta_cov=schur["S_beta_beta"],
            M_u=M_u,
            B_temporal=B_temporal,
            H_info=H_info,
            Kt_current=Kt_new,
            Ks=self.Ks,
            G=self.G,
            sigma2=self.sigma2,
            metadata={"method": "structured_joint_ssgp_transfer"},
            R_beta_beta=R_beta_beta,
            R_beta_u=R_beta_u,
            h_beta=h_beta_lik,
            beta_prior_precision=beta_prior_precision,
            beta_prior_natural=beta_prior_natural,
            Lambda_beta_given_u=schur["Lambda_beta_given_u"],
            S_beta_beta=schur["S_beta_beta"],
        )

    def update_block_no_transfer(self, **kwargs: Any) -> StructuredKronState:
        kwargs = dict(kwargs)
        kwargs["no_transfer"] = True
        return self.update_block_ssgp_transfer(**kwargs).copy_with(metadata={"method": "no_transfer"})

    def update_block_structured_joint_no_transfer(self, **kwargs: Any) -> StructuredKronState:
        kwargs = dict(kwargs)
        kwargs["no_transfer"] = True
        return self.update_block_structured_joint_ssgp_transfer(**kwargs).copy_with(
            metadata={"method": "structured_joint_no_transfer"}
        )

    def update_block_mean_field_ssgp_transfer(self, **kwargs: Any) -> StructuredKronState:
        return self.update_block_ssgp_transfer(**kwargs).copy_with(metadata={"method": "mean_field_ssgp_transfer"})

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
            variance = self.predictive_variance_decomposition(
                phi_star=phi_star,
                t_proj_star=t_proj_star,
                c_proj_star=c_proj_star,
                state=state,
            ).total_variance
        return Prediction(mean=beta_mean + gp_mean, variance=max(float(variance), self.jitter))

    def predictive_variance_decomposition(
        self,
        *,
        phi_star: np.ndarray,
        t_proj_star: np.ndarray,
        c_proj_star: np.ndarray,
        state: StructuredKronState,
    ) -> VarianceDecomposition:
        """Return predictive variance terms without changing the prediction formula."""

        phi_star = np.asarray(phi_star, dtype=float).reshape(-1)
        t_proj_star = np.asarray(t_proj_star, dtype=float).reshape(-1)
        c_proj_star = np.asarray(c_proj_star, dtype=float).reshape(-1)
        projected_prior_var = float((c_proj_star @ self.Ks @ c_proj_star) * (t_proj_star @ state.Kt_current @ t_proj_star))
        nu_star = max(0.0, self.prior_point_variance - projected_prior_var)
        Q = np.outer(c_proj_star, t_proj_star)
        q = vec_f(Q)
        u_term = 0.0
        beta_term = 0.0
        if state.R_beta_u is not None and state.S_beta_beta is not None:
            v_star = self.solve_Du(state, q)
            u_term = float(q @ v_star)
            if phi_star.size:
                adjusted_phi = phi_star - state.R_beta_u @ v_star
                beta_term = float(adjusted_phi @ state.S_beta_beta @ adjusted_phi)
        elif "u_cov_dense" in state.metadata:
            if phi_star.size:
                beta_term = float(phi_star @ state.beta_cov @ phi_star)
            u_term = float(q @ state.metadata["u_cov_dense"] @ q)
        else:
            if phi_star.size:
                beta_term = float(phi_star @ state.beta_cov @ phi_star)
            Kt_inv = inv_spd(state.Kt_current, jitter=self.jitter)
            Z = solve_sylvester_precision(Kt_inv, self.Ks_inv, state.B_temporal, state.G, Q, jitter=self.jitter)
            u_term = float(np.sum(Q * Z))
        total = self.sigma2 + nu_star + u_term + beta_term
        return VarianceDecomposition(
            sigma2=float(self.sigma2),
            nu_star=float(nu_star),
            u_posterior_term=float(u_term),
            beta_schur_term=float(beta_term),
            total_variance=max(float(total), self.jitter),
        )

    def materialize_dense_for_test(self, state: StructuredKronState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Lambda = state.dense_precision(jitter=self.jitter)
        h = state.dense_information()
        mean_dense = solve_spd(Lambda, h, jitter=self.jitter)
        return Lambda, h, mean_dense

    def dense_joint_posterior_reference(self, state: StructuredKronState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if state.R_beta_u is None:
            raise ValueError("Route-B state is required")
        D_u = dense_Du_for_tests(
            inv_spd(state.Kt_current, jitter=self.jitter),
            inv_spd(state.Ks, jitter=self.jitter),
            state.B_temporal,
            state.G,
        )
        return dense_joint_posterior_reference(
            state.routeB_A_beta(jitter=self.jitter),
            state.R_beta_u,
            D_u,
            state.routeB_h_beta_total(jitter=self.jitter),
            vec_f(state.H_info),
            jitter=self.jitter,
        )
