from __future__ import annotations

import numpy as np
import pytest

from stvgp_kronecker.joint_ssgp_kron.variance_modes import (
    JointVarianceTerms,
    compose_variance_modes,
    validated_conditional_residual,
)


def test_joint_schur_variance_matches_dense_joint_covariance() -> None:
    rng = np.random.default_rng(41)
    d_beta, d_u = 3, 4
    joint_factor = rng.normal(size=(d_beta + d_u, d_beta + d_u))
    joint_precision = joint_factor @ joint_factor.T + 4.0 * np.eye(d_beta + d_u)
    a_beta = joint_precision[:d_beta, :d_beta]
    r_beta_u = joint_precision[:d_beta, d_beta:]
    d_u_precision = joint_precision[d_beta:, d_beta:]
    d_u_inverse = np.linalg.inv(d_u_precision)
    s_beta_beta = np.linalg.inv(a_beta - r_beta_u @ d_u_inverse @ r_beta_u.T)

    phi = rng.normal(size=d_beta)
    a = rng.normal(size=d_u)
    h = r_beta_u @ d_u_inverse @ a
    residual = np.array([0.17])
    noise = 0.09
    terms = JointVarianceTerms(
        noise=noise,
        u_conditional=np.array([a @ d_u_inverse @ a]),
        beta_marginal=np.array([phi @ s_beta_beta @ phi]),
        u_beta_coupling=np.array([h @ s_beta_beta @ h]),
        beta_u_cross=np.array([-2.0 * phi @ s_beta_beta @ h]),
        conditional_residual_raw=residual,
    )
    modes = compose_variance_modes(terms)

    joint_covariance = np.linalg.inv(joint_precision)
    query = np.concatenate([phi, a])
    expected_joint = noise + query @ joint_covariance @ query
    expected_gp = noise + a @ joint_covariance[d_beta:, d_beta:] @ a + residual[0]

    np.testing.assert_allclose(modes["current_dtc"], expected_joint, atol=1e-12)
    np.testing.assert_allclose(modes["joint_dtc"], expected_joint, atol=1e-12)
    np.testing.assert_allclose(modes["gp_full_conditional"], expected_gp, atol=1e-12)
    np.testing.assert_allclose(
        modes["full_joint_conditional"], expected_joint + residual[0], atol=1e-12
    )


def test_noise_and_conditional_residual_are_added_exactly_once() -> None:
    terms = JointVarianceTerms(
        noise=2.0,
        u_conditional=np.array([3.0]),
        beta_marginal=np.array([5.0]),
        u_beta_coupling=np.array([7.0]),
        beta_u_cross=np.array([-11.0]),
        conditional_residual_raw=np.array([13.0]),
    )
    modes = compose_variance_modes(terms)

    assert modes["current_dtc"][0] == pytest.approx(6.0)
    assert modes["joint_dtc"][0] == pytest.approx(6.0)
    assert modes["gp_full_conditional"][0] == pytest.approx(25.0)
    assert modes["full_joint_conditional"][0] == pytest.approx(19.0)
    assert modes["full_joint_conditional"][0] - modes["current_dtc"][0] == pytest.approx(13.0)


def test_conditional_residual_clamps_only_rounding_scale_negatives() -> None:
    residual = validated_conditional_residual(np.array([-5e-10, 0.2]))
    np.testing.assert_array_equal(residual, np.array([0.0, 0.2]))

    with pytest.raises(FloatingPointError, match="materially negative"):
        validated_conditional_residual(np.array([-2e-8, 0.2]))
