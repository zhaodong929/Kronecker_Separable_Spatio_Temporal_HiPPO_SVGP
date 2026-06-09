"""SSGP-style changing-basis transfer formulas."""

from __future__ import annotations

import numpy as np

from .kron_utils import apply_Lon_to_beta_u_cross_block, inv_spd, solve_spd, symmetrize, vec_f


def compute_Lt(K_on_t: np.ndarray, K_nn_t: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """Compute ``L_t = K_on_t @ inv(K_nn_t)`` using a Cholesky solve."""

    return solve_spd(K_nn_t, K_on_t.T, jitter=jitter).T


def transfer_temporal_precision(B_old: np.ndarray, L_t: np.ndarray) -> np.ndarray:
    return symmetrize(L_t.T @ B_old @ L_t)


def transfer_R_uu_kron(B_old: np.ndarray, L_t: np.ndarray) -> np.ndarray:
    """Route-B transfer for ``R_uu = B_temporal kron G``."""

    return transfer_temporal_precision(B_old, L_t)


def transfer_information_matrix(H_old: np.ndarray, L_t: np.ndarray) -> np.ndarray:
    return H_old @ L_t


def transfer_h_u(H_old: np.ndarray, L_t: np.ndarray) -> np.ndarray:
    """Route-B transfer for ``h_u`` stored as ``H_info.shape == (M_s, M_t)``."""

    return transfer_information_matrix(H_old, L_t)


def transfer_R_beta_u(R_beta_u_old: np.ndarray, L_t: np.ndarray, M_s: int) -> np.ndarray:
    """Route-B transfer ``R_beta_u @ (L_t kron I_s)``."""

    return apply_Lon_to_beta_u_cross_block(R_beta_u_old, L_t, M_s)


def update_temporal_likelihood_stat(B_trans: np.ndarray, T_n: np.ndarray, sigma2: float) -> np.ndarray:
    return symmetrize(B_trans + (T_n.T @ T_n) / sigma2)


def update_information_matrix(
    H_trans: np.ndarray,
    C: np.ndarray,
    residual_matrix: np.ndarray,
    T_n: np.ndarray,
    sigma2: float,
) -> np.ndarray:
    return H_trans + (C.T @ residual_matrix @ T_n) / sigma2


def joint_likelihood_stats(
    y_vec: np.ndarray,
    Phi: np.ndarray,
    T_n: np.ndarray,
    C: np.ndarray,
    sigma2: float,
) -> dict[str, np.ndarray]:
    """Gaussian Route-B natural statistics for one block.

    Uses the repository convention ``A = T_n kron C`` and ``u = vec_F(M_s,M_t)``.
    Dense ``A`` is avoided for ``R_uu`` and ``h_u``; the low-dimensional
    ``R_beta_u`` is built row-wise from beta features.
    """

    y_vec = np.asarray(y_vec, dtype=float).reshape(-1)
    Phi = np.asarray(Phi, dtype=float)
    T_n = np.asarray(T_n, dtype=float)
    C = np.asarray(C, dtype=float)
    ns, ms = C.shape
    nt, mt = T_n.shape
    Y = y_vec.reshape((ns, nt), order="F")
    R_beta_beta = (Phi.T @ Phi) / sigma2
    h_beta = Phi.T @ y_vec / sigma2
    R_beta_u_rows = []
    for j in range(Phi.shape[1]):
        Phi_j = Phi[:, j].reshape((ns, nt), order="F")
        R_beta_u_rows.append(vec_f(C.T @ Phi_j @ T_n) / sigma2)
    R_beta_u = np.vstack(R_beta_u_rows) if R_beta_u_rows else np.zeros((0, ms * mt))
    B_temporal = (T_n.T @ T_n) / sigma2
    H_info = (C.T @ Y @ T_n) / sigma2
    return {
        "R_beta_beta": symmetrize(R_beta_beta),
        "R_beta_u": R_beta_u,
        "B_temporal": symmetrize(B_temporal),
        "h_beta": h_beta,
        "H_info": H_info,
    }


def transfer_joint_old_likelihood(
    *,
    R_beta_beta: np.ndarray,
    R_beta_u: np.ndarray,
    h_beta: np.ndarray,
    B_temporal: np.ndarray,
    H_info: np.ndarray,
    L_t: np.ndarray,
    M_s: int,
) -> dict[str, np.ndarray]:
    """Transfer Route-B joint old-likelihood natural statistics."""

    return {
        "R_beta_beta": np.asarray(R_beta_beta, dtype=float).copy(),
        "R_beta_u": transfer_R_beta_u(R_beta_u, L_t, M_s),
        "h_beta": np.asarray(h_beta, dtype=float).copy(),
        "B_temporal": transfer_R_uu_kron(B_temporal, L_t),
        "H_info": transfer_h_u(H_info, L_t),
    }


def projected_prior_transfer_dense(
    m_old: np.ndarray,
    S_old: np.ndarray,
    K_oo: np.ndarray,
    K_nn: np.ndarray,
    K_no: np.ndarray,
    K_on: np.ndarray,
    *,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense Gaussian projected-prior transfer for ablation/tests.

    Computes ``int p(u_n | u_o) q_old(u_o) du_o``.
    """

    Koo_inv_m = solve_spd(K_oo, m_old, jitter=jitter)
    Koo_inv_Kon = solve_spd(K_oo, K_on, jitter=jitter)
    m_proj = K_no @ Koo_inv_m
    S_proj = K_nn + K_no @ solve_spd(K_oo, S_old - K_oo, jitter=jitter) @ Koo_inv_Kon
    return m_proj, symmetrize(S_proj)


def old_likelihood_transfer_dense_for_test(
    m_old: np.ndarray,
    S_old: np.ndarray,
    K_oo: np.ndarray,
    K_on: np.ndarray,
    K_nn: np.ndarray,
    *,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense old-likelihood-ratio reference implementation for tests only."""

    S_inv = inv_spd(S_old, jitter=jitter)
    K_inv = inv_spd(K_oo, jitter=jitter)
    R_old = symmetrize(S_inv - K_inv)
    r_old = S_inv @ m_old
    L = K_on @ inv_spd(K_nn, jitter=jitter)
    Lambda_old = symmetrize(L.T @ R_old @ L)
    h_old = L.T @ r_old
    return Lambda_old, h_old


def structured_old_likelihood_dense_reference(
    B_old: np.ndarray,
    G: np.ndarray,
    H_old: np.ndarray,
    L_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize the structured transfer for small tests."""

    Lambda = np.kron(transfer_temporal_precision(B_old, L_t), G)
    h = vec_f(transfer_information_matrix(H_old, L_t))
    return Lambda, h
