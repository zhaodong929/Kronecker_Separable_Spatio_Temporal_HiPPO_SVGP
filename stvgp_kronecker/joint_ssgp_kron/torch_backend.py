"""PyTorch backend for the structured Route-B posterior and prediction.

This module mirrors the NumPy/SciPy algorithm in :mod:`kron_utils` and
:mod:`model`.  It keeps the Kronecker/Sylvester representation throughout;
CUDA acceleration does not materialize an ``(M_s M_t) x (M_s M_t)`` matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


TensorLike = torch.Tensor | np.ndarray


def _as_tensor(
    value: TensorLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def stable_jitter(matrix: torch.Tensor, jitter: float = 1e-6) -> float:
    if matrix.numel() == 0:
        return float(jitter)
    diag_mean = float(torch.diagonal(matrix).mean().detach().cpu())
    return float(jitter) * max(1.0, abs(diag_mean))


def stable_cholesky(
    matrix: torch.Tensor,
    *,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """Cholesky factor with the same scaled-jitter retry policy as SciPy."""

    matrix = symmetrize(matrix)
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    base = stable_jitter(matrix, jitter)
    last_info = -1
    for multiplier in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        factor, info = torch.linalg.cholesky_ex(
            matrix + (base * multiplier) * eye,
            check_errors=False,
        )
        last_info = int(info.max().detach().cpu())
        if last_info == 0:
            return factor
    raise torch.linalg.LinAlgError(
        f"Cholesky failed after jitter escalation from {base:g}; info={last_info}"
    )


def solve_spd(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    *,
    jitter: float = 1e-6,
) -> torch.Tensor:
    vector_input = rhs.ndim == 1
    rhs_matrix = rhs[:, None] if vector_input else rhs
    factor = stable_cholesky(matrix, jitter=jitter)
    solved = torch.cholesky_solve(rhs_matrix, factor, upper=False)
    return solved[:, 0] if vector_input else solved


def inv_spd(matrix: torch.Tensor, *, jitter: float = 1e-6) -> torch.Tensor:
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    return solve_spd(matrix, eye, jitter=jitter)


def vec_f(matrix: torch.Tensor) -> torch.Tensor:
    """Column-major vectorization using PyTorch's row-major storage."""

    return matrix.transpose(-1, -2).contiguous().reshape(-1)


def unvec_f(vector: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    rows, columns = shape
    return vector.reshape(columns, rows).transpose(0, 1)


def _unvec_f_columns(
    vectors: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Convert ``(rows*cols, n_rhs)`` into ``(n_rhs, rows, cols)``."""

    rows, columns = shape
    return vectors.transpose(0, 1).reshape(-1, columns, rows).transpose(1, 2)


def _vec_f_batch(matrices: torch.Tensor) -> torch.Tensor:
    """Convert ``(batch, rows, cols)`` into ``(rows*cols, batch)``."""

    return matrices.transpose(1, 2).contiguous().reshape(matrices.shape[0], -1).transpose(0, 1)


def generalized_eigh(
    matrix: torch.Tensor,
    metric: torch.Tensor,
    *,
    jitter: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``A p = lambda B p`` by Cholesky whitening.

    The columns of the returned eigenvector matrix satisfy
    ``P.T @ B_jittered @ P = I``.
    """

    factor = stable_cholesky(metric, jitter=jitter)
    left = torch.linalg.solve_triangular(
        factor,
        symmetrize(matrix),
        upper=False,
    )
    whitened = torch.linalg.solve_triangular(
        factor,
        left.transpose(0, 1),
        upper=False,
    ).transpose(0, 1)
    values, vectors_white = torch.linalg.eigh(symmetrize(whitened))
    vectors = torch.linalg.solve_triangular(
        factor.transpose(0, 1),
        vectors_white,
        upper=True,
    )
    return values, vectors


def solve_du_sylvester(
    kt_inv: torch.Tensor,
    ks_inv: torch.Tensor,
    b_temporal: torch.Tensor,
    g_spatial: torch.Tensor,
    rhs: torch.Tensor,
    *,
    jitter: float = 1e-8,
) -> torch.Tensor:
    """Structured solve for ``Kt^-1 kron Ks^-1 + B kron G``."""

    vector_input = rhs.ndim == 1
    rhs_matrix = rhs[:, None] if vector_input else rhs
    ms = ks_inv.shape[0]
    mt = kt_inv.shape[0]
    left_values, p = generalized_eigh(g_spatial, ks_inv, jitter=jitter)
    right_values, q = generalized_eigh(b_temporal, kt_inv, jitter=jitter)
    denominator = 1.0 + torch.outer(left_values, right_values)
    replacement = torch.where(
        denominator >= 0,
        torch.full_like(denominator, jitter),
        torch.full_like(denominator, -jitter),
    )
    denominator = torch.where(denominator.abs() < jitter, replacement, denominator)

    rhs_blocks = _unvec_f_columns(rhs_matrix, (ms, mt))
    transformed = torch.matmul(p.transpose(0, 1), rhs_blocks)
    transformed = torch.matmul(transformed, q)
    solved_blocks = torch.matmul(p, transformed / denominator)
    solved_blocks = torch.matmul(solved_blocks, q.transpose(0, 1))
    solved = _vec_f_batch(solved_blocks)
    return solved[:, 0] if vector_input else solved


def schur_recover_posterior(
    a_beta: torch.Tensor,
    b_beta_u: torch.Tensor,
    h_beta: torch.Tensor,
    h_u: torch.Tensor,
    kt_inv: torch.Tensor,
    ks_inv: torch.Tensor,
    b_temporal: torch.Tensor,
    g_spatial: torch.Tensor,
    *,
    jitter: float = 1e-8,
) -> dict[str, torch.Tensor]:
    d = h_beta.numel()
    if d == 0:
        v_h = solve_du_sylvester(
            kt_inv,
            ks_inv,
            b_temporal,
            g_spatial,
            h_u,
            jitter=jitter,
        )
        return {
            "Lambda_beta_given_u": a_beta.new_zeros((0, 0)),
            "S_beta_beta": a_beta.new_zeros((0, 0)),
            "W": h_u.new_zeros((h_u.numel(), 0)),
            "v_h": v_h,
            "m_beta": h_beta.new_zeros((0,)),
            "m_u": v_h,
        }

    joint_rhs = torch.cat((b_beta_u.transpose(0, 1), h_u[:, None]), dim=1)
    joint_solution = solve_du_sylvester(
        kt_inv,
        ks_inv,
        b_temporal,
        g_spatial,
        joint_rhs,
        jitter=jitter,
    )
    w = joint_solution[:, :d]
    v_h = joint_solution[:, d]
    lambda_beta = symmetrize(a_beta - b_beta_u @ w)
    rhs_beta = h_beta - b_beta_u @ v_h
    m_beta = solve_spd(lambda_beta, rhs_beta, jitter=jitter)
    s_beta_beta = inv_spd(lambda_beta, jitter=jitter)
    return {
        "Lambda_beta_given_u": lambda_beta,
        "S_beta_beta": s_beta_beta,
        "W": w,
        "v_h": v_h,
        "m_beta": m_beta,
        "m_u": v_h - w @ m_beta,
    }


@dataclass(frozen=True)
class TorchStructuredKronState:
    beta_mean: torch.Tensor
    beta_cov: torch.Tensor
    M_u: torch.Tensor
    B_temporal: torch.Tensor
    H_info: torch.Tensor
    Kt_current: torch.Tensor
    Ks: torch.Tensor
    G: torch.Tensor
    sigma2: float
    metadata: dict[str, Any] = field(default_factory=dict)
    R_beta_beta: torch.Tensor | None = None
    R_beta_u: torch.Tensor | None = None
    h_beta: torch.Tensor | None = None
    beta_prior_precision: torch.Tensor | None = None
    beta_prior_natural: torch.Tensor | None = None
    Lambda_beta_given_u: torch.Tensor | None = None
    S_beta_beta: torch.Tensor | None = None

    @property
    def mt(self) -> int:
        return int(self.Kt_current.shape[0])

    @property
    def ms(self) -> int:
        return int(self.Ks.shape[0])

    def tensor_bytes(self) -> int:
        total = 0
        seen: set[int] = set()
        for value in vars(self).values():
            if torch.is_tensor(value) and id(value) not in seen:
                seen.add(id(value))
                total += value.numel() * value.element_size()
        return int(total)


def _transfer_beta_u(
    r_beta_u: torch.Tensor,
    l_t: torch.Tensor,
    ms: int,
) -> torch.Tensor:
    if r_beta_u.numel() == 0:
        return r_beta_u.new_zeros((r_beta_u.shape[0], ms * l_t.shape[1]))
    d = r_beta_u.shape[0]
    old_mt = l_t.shape[0]
    blocks = r_beta_u.reshape(d, old_mt, ms).transpose(1, 2)
    transferred = blocks @ l_t
    return transferred.transpose(1, 2).contiguous().reshape(d, -1)


class TorchJointSSGPKronHiPPOSVGP:
    """CUDA-capable equivalent of the structured-joint NumPy Route-B model."""

    def __init__(
        self,
        *,
        Ks: TensorLike,
        C: TensorLike,
        sigma2: float,
        beta_prior_mean: TensorLike,
        beta_prior_cov: TensorLike,
        prior_point_variance: float = 1.0,
        jitter: float = 1e-6,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.Ks = symmetrize(_as_tensor(Ks, device=self.device, dtype=dtype))
        self.C = _as_tensor(C, device=self.device, dtype=dtype)
        self.sigma2 = float(sigma2)
        self.beta_prior_mean = _as_tensor(
            beta_prior_mean, device=self.device, dtype=dtype
        ).reshape(-1)
        self.beta_prior_cov = symmetrize(
            _as_tensor(beta_prior_cov, device=self.device, dtype=dtype)
        )
        self.prior_point_variance = float(prior_point_variance)
        self.jitter = float(jitter)
        self.G = symmetrize(self.C.transpose(0, 1) @ self.C)
        self.Ks_inv = inv_spd(self.Ks, jitter=self.jitter)

    def tensor_bytes(self) -> int:
        values = (
            self.Ks,
            self.C,
            self.beta_prior_mean,
            self.beta_prior_cov,
            self.G,
            self.Ks_inv,
        )
        return int(sum(value.numel() * value.element_size() for value in values))

    def _new_likelihood_stats(
        self,
        y_vec: torch.Tensor,
        phi: torch.Tensor,
        t_n: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ns, ms = self.C.shape
        nt, mt = t_n.shape
        d = phi.shape[1]
        y_matrix = unvec_f(y_vec, (ns, nt))
        r_beta_beta = (phi.transpose(0, 1) @ phi) / self.sigma2
        h_beta = (phi.transpose(0, 1) @ y_vec) / self.sigma2
        if d:
            phi_blocks = phi.transpose(0, 1).reshape(d, nt, ns).transpose(1, 2)
            cross = torch.matmul(self.C.transpose(0, 1), phi_blocks)
            cross = torch.matmul(cross, t_n)
            r_beta_u = cross.transpose(1, 2).contiguous().reshape(d, ms * mt) / self.sigma2
        else:
            r_beta_u = phi.new_zeros((0, ms * mt))
        return {
            "R_beta_beta": symmetrize(r_beta_beta),
            "R_beta_u": r_beta_u,
            "h_beta": h_beta,
            "B_temporal": symmetrize(t_n.transpose(0, 1) @ t_n / self.sigma2),
            "H_info": self.C.transpose(0, 1) @ y_matrix @ t_n / self.sigma2,
        }

    def _old_likelihood_stats(
        self,
        state: TorchStructuredKronState | None,
        kt_new: torch.Tensor,
        k_on_t: torch.Tensor | None,
        *,
        beta_dim: int,
        no_transfer: bool,
        l_t_override: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        mt_new = kt_new.shape[0]
        ms = self.Ks.shape[0]
        if state is None or no_transfer:
            return {
                "R_beta_beta": kt_new.new_zeros((beta_dim, beta_dim)),
                "R_beta_u": kt_new.new_zeros((beta_dim, ms * mt_new)),
                "h_beta": kt_new.new_zeros((beta_dim,)),
                "B_temporal": kt_new.new_zeros((mt_new, mt_new)),
                "H_info": kt_new.new_zeros((ms, mt_new)),
            }
        if l_t_override is not None:
            l_t = l_t_override
            if tuple(l_t.shape) != (state.mt, mt_new):
                raise ValueError(
                    f"L_t_override must have shape {(state.mt, mt_new)}, got {tuple(l_t.shape)}"
                )
        elif k_on_t is None:
            if state.mt != mt_new:
                raise ValueError("K_on_t is required when temporal basis size changes")
            l_t = torch.eye(mt_new, dtype=self.dtype, device=self.device)
        else:
            l_t = solve_spd(kt_new, k_on_t.transpose(0, 1), jitter=self.jitter).transpose(0, 1)

        r_beta_beta = (
            kt_new.new_zeros((beta_dim, beta_dim))
            if state.R_beta_beta is None
            else state.R_beta_beta
        )
        r_beta_u = (
            kt_new.new_zeros((beta_dim, ms * state.mt))
            if state.R_beta_u is None
            else state.R_beta_u
        )
        h_beta = kt_new.new_zeros((beta_dim,)) if state.h_beta is None else state.h_beta
        return {
            "R_beta_beta": r_beta_beta.clone(),
            "R_beta_u": _transfer_beta_u(r_beta_u, l_t, ms),
            "h_beta": h_beta.clone(),
            "B_temporal": symmetrize(l_t.transpose(0, 1) @ state.B_temporal @ l_t),
            "H_info": state.H_info @ l_t,
        }

    def update_block_structured_joint_ssgp_transfer(
        self,
        *,
        y_vec: TensorLike,
        Phi: TensorLike,
        T_n: TensorLike,
        Kt_new: TensorLike,
        state: TorchStructuredKronState | None = None,
        K_on_t: TensorLike | None = None,
        beta_drift: TensorLike | None = None,
        no_transfer: bool = False,
        L_t_override: TensorLike | None = None,
    ) -> TorchStructuredKronState:
        y_tensor = _as_tensor(y_vec, device=self.device, dtype=self.dtype).reshape(-1)
        phi = _as_tensor(Phi, device=self.device, dtype=self.dtype)
        t_n = _as_tensor(T_n, device=self.device, dtype=self.dtype)
        kt_new = symmetrize(_as_tensor(Kt_new, device=self.device, dtype=self.dtype))
        k_on = (
            None
            if K_on_t is None
            else _as_tensor(K_on_t, device=self.device, dtype=self.dtype)
        )
        l_override = (
            None
            if L_t_override is None
            else _as_tensor(L_t_override, device=self.device, dtype=self.dtype)
        )
        ns, ms = self.C.shape
        nt, mt = t_n.shape
        if y_tensor.numel() != ns * nt:
            raise ValueError(f"Expected y_vec length {ns * nt}, got {y_tensor.numel()}")
        if phi.shape[0] != y_tensor.numel():
            raise ValueError("Phi row count must match y_vec length")

        d = phi.shape[1]
        old = self._old_likelihood_stats(
            state,
            kt_new,
            k_on,
            beta_dim=d,
            no_transfer=no_transfer,
            l_t_override=l_override,
        )
        new = self._new_likelihood_stats(y_tensor, phi, t_n)
        r_beta_beta = symmetrize(old["R_beta_beta"] + new["R_beta_beta"])
        r_beta_u = old["R_beta_u"] + new["R_beta_u"]
        h_beta_lik = old["h_beta"] + new["h_beta"]
        b_temporal = symmetrize(old["B_temporal"] + new["B_temporal"])
        h_info = old["H_info"] + new["H_info"]

        prior_cov = self.beta_prior_cov
        if beta_drift is not None:
            prior_cov = symmetrize(
                prior_cov
                + _as_tensor(beta_drift, device=self.device, dtype=self.dtype)
            )
        beta_prior_precision = (
            kt_new.new_zeros((0, 0))
            if d == 0
            else inv_spd(prior_cov, jitter=self.jitter)
        )
        beta_prior_natural = (
            kt_new.new_zeros((0,))
            if d == 0
            else beta_prior_precision @ self.beta_prior_mean
        )
        a_beta = symmetrize(beta_prior_precision + r_beta_beta)
        h_beta_total = beta_prior_natural + h_beta_lik
        kt_inv = inv_spd(kt_new, jitter=self.jitter)
        schur = schur_recover_posterior(
            a_beta,
            r_beta_u,
            h_beta_total,
            vec_f(h_info),
            kt_inv,
            self.Ks_inv,
            b_temporal,
            self.G,
            jitter=self.jitter,
        )
        return TorchStructuredKronState(
            beta_mean=schur["m_beta"],
            beta_cov=schur["S_beta_beta"],
            M_u=unvec_f(schur["m_u"], (ms, mt)),
            B_temporal=b_temporal,
            H_info=h_info,
            Kt_current=kt_new,
            Ks=self.Ks,
            G=self.G,
            sigma2=self.sigma2,
            metadata={"method": "structured_joint_ssgp_transfer", "backend": "torch"},
            R_beta_beta=r_beta_beta,
            R_beta_u=r_beta_u,
            h_beta=h_beta_lik,
            beta_prior_precision=beta_prior_precision,
            beta_prior_natural=beta_prior_natural,
            Lambda_beta_given_u=schur["Lambda_beta_given_u"],
            S_beta_beta=schur["S_beta_beta"],
        )

    def predict_with_C(
        self,
        *,
        state: TorchStructuredKronState,
        T_eval: TensorLike,
        Phi: TensorLike,
        C_eval: TensorLike,
        chunk_size: int = 8192,
        include_conditional_residual_variance: bool = False,
        return_numpy: bool = True,
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, dict[str, float]]:
        """Chunked prediction using the same Sylvester diagonalization as NumPy."""

        t_eval = _as_tensor(T_eval, device=self.device, dtype=self.dtype)
        phi = _as_tensor(Phi, device=self.device, dtype=self.dtype)
        c_eval = _as_tensor(C_eval, device=self.device, dtype=self.dtype)
        nt = t_eval.shape[0]
        ns = c_eval.shape[0]
        n = nt * ns
        if phi.shape[0] != n:
            raise ValueError(f"Phi rows ({phi.shape[0]}) do not match T/C product ({n}).")

        kt_inv = inv_spd(state.Kt_current, jitter=self.jitter)
        left_values, p = generalized_eigh(state.G, self.Ks_inv, jitter=self.jitter)
        right_values, q = generalized_eigh(
            state.B_temporal, kt_inv, jitter=self.jitter
        )
        denominator = 1.0 + torch.outer(left_values, right_values)
        replacement = torch.where(
            denominator >= 0,
            torch.full_like(denominator, self.jitter),
            torch.full_like(denominator, -self.jitter),
        )
        denominator = torch.where(
            denominator.abs() < self.jitter, replacement, denominator
        )
        inv_denominator = denominator.reciprocal()
        c_projected_all = c_eval @ p
        t_projected_all = t_eval @ q
        s_var_all = torch.sum((c_eval @ self.Ks) * c_eval, dim=1)
        t_var_all = torch.sum((t_eval @ state.Kt_current) * t_eval, dim=1)

        r_tilde = None
        if state.R_beta_u is not None and state.S_beta_beta is not None:
            d = state.R_beta_u.shape[0]
            r_blocks = state.R_beta_u.reshape(d, state.mt, state.ms).transpose(1, 2)
            r_tilde = torch.matmul(p.transpose(0, 1), r_blocks)
            r_tilde = torch.matmul(r_tilde, q)

        mean = torch.empty(n, dtype=self.dtype, device=self.device)
        variance = torch.empty_like(mean)
        u_terms_all = torch.empty_like(mean)
        beta_terms_all = torch.empty_like(mean)
        nu_raw_all = torch.empty_like(mean)
        chunk_size = max(1, int(chunk_size))
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            flat = torch.arange(start, stop, device=self.device)
            time_index = torch.div(flat, ns, rounding_mode="floor")
            space_index = torch.remainder(flat, ns)
            c_raw = c_eval[space_index]
            t_raw = t_eval[time_index]
            c_projected = c_projected_all[space_index]
            t_projected = t_projected_all[time_index]

            gp_mean = torch.einsum("bi,ij,bj->b", c_raw, state.M_u, t_raw)
            mean[start:stop] = phi[start:stop] @ state.beta_mean + gp_mean
            u_inner = t_projected.square() @ inv_denominator.transpose(0, 1)
            u_terms = torch.sum(c_projected.square() * u_inner, dim=1)
            u_terms_all[start:stop] = u_terms

            if r_tilde is not None:
                cross = torch.einsum(
                    "bi,dij,bj->bd",
                    c_projected,
                    r_tilde * inv_denominator,
                    t_projected,
                )
                adjusted_phi = phi[start:stop] - cross
                beta_terms = torch.einsum(
                    "bi,ij,bj->b",
                    adjusted_phi,
                    state.S_beta_beta,
                    adjusted_phi,
                )
            else:
                beta_terms = torch.einsum(
                    "bi,ij,bj->b",
                    phi[start:stop],
                    state.beta_cov,
                    phi[start:stop],
                )
            beta_terms_all[start:stop] = beta_terms
            projected_prior = t_var_all[time_index] * s_var_all[space_index]
            nu_raw = torch.clamp(self.prior_point_variance - projected_prior, min=0.0)
            nu_raw_all[start:stop] = nu_raw
            nu = nu_raw if include_conditional_residual_variance else 0.0
            variance[start:stop] = torch.clamp(
                self.sigma2 + nu + u_terms + beta_terms,
                min=self.jitter,
            )

        diagnostics = {
            "avg_sigma2": self.sigma2,
            "avg_nu_star": float(
                (nu_raw_all.mean() if include_conditional_residual_variance else nu_raw_all.new_zeros(())).detach().cpu()
            ),
            "avg_nu_star_raw": float(nu_raw_all.mean().detach().cpu()),
            "avg_u_posterior_term": float(u_terms_all.mean().detach().cpu()),
            "avg_beta_schur_term": float(beta_terms_all.mean().detach().cpu()),
        }
        if not return_numpy:
            return mean, variance, diagnostics
        return (
            mean.detach().cpu().numpy(),
            variance.detach().cpu().numpy(),
            diagnostics,
        )
