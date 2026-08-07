"""Variance-mode composition for fixed-posterior Route-B diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VARIANCE_MODES = (
    "current_dtc",
    "joint_dtc",
    "gp_full_conditional",
    "full_joint_conditional",
)


@dataclass(frozen=True)
class JointVarianceTerms:
    """Pointwise terms in the structured joint ``q(beta, u)`` variance."""

    noise: float
    u_conditional: np.ndarray
    beta_marginal: np.ndarray
    u_beta_coupling: np.ndarray
    beta_u_cross: np.ndarray
    conditional_residual_raw: np.ndarray


def validated_conditional_residual(
    residual_raw: np.ndarray,
    *,
    negative_tolerance: float = 1e-8,
) -> np.ndarray:
    """Clamp rounding-scale negatives and reject a material kernel mismatch."""

    residual = np.asarray(residual_raw, dtype=float)
    if not np.all(np.isfinite(residual)):
        raise FloatingPointError("Conditional residual contains non-finite values")
    minimum = float(np.min(residual)) if residual.size else 0.0
    if minimum < -float(negative_tolerance):
        raise FloatingPointError(
            "Conditional residual is materially negative: "
            f"minimum={minimum:.6e}, tolerance={negative_tolerance:.6e}. "
            "This indicates a K_xu/K_uu/kernel-diagonal mismatch."
        )
    return np.maximum(residual, 0.0)


def compose_variance_modes(
    terms: JointVarianceTerms,
    *,
    negative_tolerance: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Compose the four requested predictive variances without double counting."""

    u_conditional = np.asarray(terms.u_conditional, dtype=float)
    beta_marginal = np.asarray(terms.beta_marginal, dtype=float)
    u_beta_coupling = np.asarray(terms.u_beta_coupling, dtype=float)
    beta_u_cross = np.asarray(terms.beta_u_cross, dtype=float)
    residual = validated_conditional_residual(
        terms.conditional_residual_raw,
        negative_tolerance=negative_tolerance,
    )
    shapes = {
        value.shape
        for value in (
            u_conditional,
            beta_marginal,
            u_beta_coupling,
            beta_u_cross,
            residual,
        )
    }
    if len(shapes) != 1:
        raise ValueError(f"Variance terms have inconsistent shapes: {sorted(shapes)}")

    # D_u^-1 is Cov(u | beta). The remaining three terms expand
    # (phi - R D_u^-1 a)^T S_bb (phi - R D_u^-1 a).
    beta_schur = beta_marginal + u_beta_coupling + beta_u_cross
    u_marginal = u_conditional + u_beta_coupling
    joint_dtc = float(terms.noise) + u_conditional + beta_schur

    variances = {
        # This is the existing implementation, written in its Schur form.
        "current_dtc": joint_dtc,
        # This is the same quantity expanded into S_bb, S_uu, and S_bu terms.
        "joint_dtc": (
            float(terms.noise)
            + beta_marginal
            + u_marginal
            + beta_u_cross
        ),
        # The marginal GP inducing uncertainty, but no beta/cross contribution.
        "gp_full_conditional": float(terms.noise) + u_marginal + residual,
        "full_joint_conditional": joint_dtc + residual,
    }
    for mode, variance in variances.items():
        if not np.all(np.isfinite(variance)):
            raise FloatingPointError(f"{mode} contains non-finite predictive variance")
        minimum = float(np.min(variance)) if variance.size else float(terms.noise)
        if minimum <= 0.0:
            raise FloatingPointError(
                f"{mode} contains non-positive predictive variance: {minimum:.6e}"
            )
    return variances
