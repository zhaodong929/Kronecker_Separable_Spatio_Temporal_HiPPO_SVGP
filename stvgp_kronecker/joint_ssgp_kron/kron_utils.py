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
