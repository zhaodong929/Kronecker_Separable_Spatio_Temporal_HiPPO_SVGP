"""Kronecker utilities for the joint SSGP transfer implementation."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def stable_jitter(matrix: np.ndarray, jitter: float = 1e-6) -> float:
    diag_mean = float(np.mean(np.diag(matrix))) if matrix.size else 1.0
    return jitter * max(1.0, abs(diag_mean))


def add_jitter(matrix: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    matrix = symmetrize(np.asarray(matrix, dtype=float))
    return matrix + stable_jitter(matrix, jitter) * np.eye(matrix.shape[0])


def solve_spd(matrix: np.ndarray, rhs: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    matrix_j = add_jitter(matrix, jitter)
    factor = cho_factor(matrix_j, lower=True, check_finite=False)
    return cho_solve(factor, rhs, check_finite=False)


def inv_spd(matrix: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    eye = np.eye(matrix.shape[0])
    return solve_spd(matrix, eye, jitter=jitter)


def vec_f(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix).reshape(-1, order="F")


def unvec_f(vector: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(vector).reshape(shape, order="F")


def kron_mv(
    T: np.ndarray,
    C: np.ndarray,
    U: np.ndarray,
    *,
    output: Literal["vector", "matrix"] = "vector",
) -> np.ndarray:
    """Compute ``(T kron C) vec(U)`` without materializing the Kronecker matrix."""

    result = C @ U @ T.T
    if output == "matrix":
        return result
    return vec_f(result)


def kron_t_mv(
    T: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
    *,
    output: Literal["vector", "matrix"] = "matrix",
) -> np.ndarray:
    """Compute ``(T kron C).T vec(R)`` without materializing the Kronecker matrix."""

    result = C.T @ R @ T
    if output == "vector":
        return vec_f(result)
    return result


def dense_A_from_factors(T: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Return the dense matrix ``T kron C`` for small reference tests."""

    return np.kron(T, C)


def dense_Lon_for_tests(L_t: np.ndarray, M_s: int) -> np.ndarray:
    """Return dense ``L_on = L_t kron I_s`` for small Route-B reference tests."""

    return np.kron(L_t, np.eye(M_s))


def apply_temporal_right(matrix: np.ndarray, L_t: np.ndarray) -> np.ndarray:
    """Apply ``L_t`` along the temporal mode of an ``(M_s, M_t_old)`` matrix.

    The project uses ``u = vec_F(M)`` with ``M.shape == (M_s, M_t)``. Therefore
    ``(L_t kron I_s)^T vec_F(M_old) = vec_F(M_old @ L_t)``.
    """

    return np.asarray(matrix, dtype=float) @ np.asarray(L_t, dtype=float)


def apply_Lon_to_beta_u_cross_block(R_beta_u: np.ndarray, L_t: np.ndarray, M_s: int) -> np.ndarray:
    """Compute ``R_beta_u @ (L_t kron I_s)`` without materializing ``L_on``.

    Each beta row is reshaped as an ``(M_s, M_t_old)`` matrix under the
    repository's Fortran vectorization convention, then multiplied on the
    temporal dimension by ``L_t``.
    """

    R_beta_u = np.asarray(R_beta_u, dtype=float)
    if R_beta_u.size == 0:
        return np.zeros((R_beta_u.shape[0], M_s * L_t.shape[1]))
    rows = []
    for row in R_beta_u:
        old = unvec_f(row, (M_s, L_t.shape[0]))
        rows.append(vec_f(apply_temporal_right(old, L_t)))
    return np.vstack(rows)


def dense_Du_for_tests(Kt_inv: np.ndarray, Ks_inv: np.ndarray, B: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Dense ``D_u`` reference for tests: ``Kt_inv kron Ks_inv + B kron G``."""

    return symmetrize(np.kron(Kt_inv, Ks_inv) + np.kron(B, G))


def make_spd_matrix(dim: int, jitter: float = 1e-4, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim))
    matrix = A @ A.T
    matrix += (jitter + 0.1 * dim) * np.eye(dim)
    return symmetrize(matrix)


def relative_fro_error(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B, ord="fro") / max(1.0, np.linalg.norm(B, ord="fro")))


def solve_sylvester_precision(
    Kt_inv: np.ndarray,
    Ks_inv: np.ndarray,
    B: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    *,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Solve ``Ks_inv M Kt_inv + G M B = H``.

    The solve uses generalized eigendecompositions and does not materialize the
    full Kronecker precision. With ``P.T @ Ks_inv @ P = I`` and
    ``P.T @ G @ P = diag(lambda)``, and similarly
    ``Q.T @ Kt_inv @ Q = I`` and ``Q.T @ B @ Q = diag(mu)``, the transformed
    equation is elementwise: ``X_ij * (1 + lambda_i * mu_j) = RHS_ij``.
    """

    Kt_inv = add_jitter(Kt_inv, jitter)
    Ks_inv = add_jitter(Ks_inv, jitter)
    B = symmetrize(np.asarray(B, dtype=float))
    G = symmetrize(np.asarray(G, dtype=float))
    H = np.asarray(H, dtype=float)

    left_vals, P = eigh(G, Ks_inv, check_finite=False)
    right_vals, Q = eigh(B, Kt_inv, check_finite=False)
    rhs = P.T @ H @ Q
    denom = 1.0 + np.outer(left_vals, right_vals)
    denom = np.where(np.abs(denom) < jitter, np.sign(denom) * jitter + (denom == 0) * jitter, denom)
    X = rhs / denom
    return P @ X @ Q.T


def solve_Du_sylvester(
    Kt_inv: np.ndarray,
    Ks_inv: np.ndarray,
    B: np.ndarray,
    G: np.ndarray,
    rhs: np.ndarray,
    *,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Solve ``D_u x = rhs`` with the Sylvester-compatible ``u`` precision.

    ``rhs`` may be a vector of length ``M_s*M_t`` or a dense matrix with one
    right-hand side per column. Returned shape matches the input shape.
    """

    rhs = np.asarray(rhs, dtype=float)
    M_s = Ks_inv.shape[0]
    M_t = Kt_inv.shape[0]
    if rhs.ndim == 1:
        Z = solve_sylvester_precision(
            Kt_inv,
            Ks_inv,
            B,
            G,
            unvec_f(rhs, (M_s, M_t)),
            jitter=jitter,
        )
        return vec_f(Z)
    cols = [
        solve_Du_sylvester(Kt_inv, Ks_inv, B, G, rhs[:, i], jitter=jitter)
        for i in range(rhs.shape[1])
    ]
    return np.column_stack(cols)


def schur_recover_posterior(
    A_beta: np.ndarray,
    B_beta_u: np.ndarray,
    h_beta: np.ndarray,
    h_u: np.ndarray,
    Kt_inv: np.ndarray,
    Ks_inv: np.ndarray,
    B_temporal: np.ndarray,
    G: np.ndarray,
    *,
    jitter: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Recover Route-B posterior moments from structured joint precision.

    The returned ``Lambda_beta_given_u`` is a precision matrix, not a covariance.
    """

    A_beta = symmetrize(np.asarray(A_beta, dtype=float))
    B_beta_u = np.asarray(B_beta_u, dtype=float)
    h_beta = np.asarray(h_beta, dtype=float).reshape(-1)
    h_u = np.asarray(h_u, dtype=float).reshape(-1)
    d = h_beta.shape[0]
    if d == 0:
        v_h = solve_Du_sylvester(Kt_inv, Ks_inv, B_temporal, G, h_u, jitter=jitter)
        return {
            "Lambda_beta_given_u": np.zeros((0, 0)),
            "S_beta_beta": np.zeros((0, 0)),
            "W": np.zeros((h_u.shape[0], 0)),
            "v_h": v_h,
            "m_beta": np.zeros(0),
            "m_u": v_h,
        }

    W = solve_Du_sylvester(Kt_inv, Ks_inv, B_temporal, G, B_beta_u.T, jitter=jitter)
    v_h = solve_Du_sylvester(Kt_inv, Ks_inv, B_temporal, G, h_u, jitter=jitter)
    Lambda_beta_given_u = symmetrize(A_beta - B_beta_u @ W)
    rhs_beta = h_beta - B_beta_u @ v_h
    m_beta = solve_spd(Lambda_beta_given_u, rhs_beta, jitter=jitter)
    S_beta_beta = inv_spd(Lambda_beta_given_u, jitter=jitter)
    m_u = v_h - W @ m_beta
    return {
        "Lambda_beta_given_u": Lambda_beta_given_u,
        "S_beta_beta": S_beta_beta,
        "W": W,
        "v_h": v_h,
        "m_beta": m_beta,
        "m_u": m_u,
    }


def dense_joint_posterior_reference(
    A_beta: np.ndarray,
    B_beta_u: np.ndarray,
    D_u: np.ndarray,
    h_beta: np.ndarray,
    h_u: np.ndarray,
    *,
    jitter: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense joint posterior reference for small tests.

    Returns ``(Lambda, covariance, mean)`` for ``z=[beta; u]``.
    """

    Lambda = np.block(
        [
            [np.asarray(A_beta, dtype=float), np.asarray(B_beta_u, dtype=float)],
            [np.asarray(B_beta_u, dtype=float).T, np.asarray(D_u, dtype=float)],
        ]
    )
    Lambda = symmetrize(Lambda)
    h = np.concatenate([np.asarray(h_beta, dtype=float).reshape(-1), np.asarray(h_u, dtype=float).reshape(-1)])
    mean = solve_spd(Lambda, h, jitter=jitter)
    covariance = inv_spd(Lambda, jitter=jitter)
    return Lambda, covariance, mean
