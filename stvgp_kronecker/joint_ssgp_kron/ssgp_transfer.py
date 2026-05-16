"""SSGP-style changing-basis transfer formulas."""

from __future__ import annotations

import numpy as np

from .kron_utils import inv_spd, solve_spd, symmetrize, vec_f


def compute_Lt(K_on_t: np.ndarray, K_nn_t: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """Compute ``L_t = K_on_t @ inv(K_nn_t)`` using a Cholesky solve."""

    return solve_spd(K_nn_t, K_on_t.T, jitter=jitter).T


def transfer_temporal_precision(B_old: np.ndarray, L_t: np.ndarray) -> np.ndarray:
    return symmetrize(L_t.T @ B_old @ L_t)


def transfer_information_matrix(H_old: np.ndarray, L_t: np.ndarray) -> np.ndarray:
    return H_old @ L_t


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
