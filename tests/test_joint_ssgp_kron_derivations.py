from __future__ import annotations

from scripts.verify_joint_ssgp_kron_derivations import (
    check_fixed_basis_streaming_equals_batch,
    check_l_on_identity,
    check_no_linear_mean,
    check_no_old_data,
    check_old_likelihood_transfer,
    check_projected_prior_and_structured_transfer,
)


def test_Lon_kron_identity() -> None:
    result = check_l_on_identity()
    assert result["passed"], result


def test_old_likelihood_transfer_kron_identity() -> None:
    result = check_old_likelihood_transfer()
    assert result["passed"], result


def test_fixed_basis_streaming_equals_batch() -> None:
    result = check_fixed_basis_streaming_equals_batch()
    assert result["passed"], result


def test_no_linear_mean_reduces_to_gp_only() -> None:
    result = check_no_linear_mean()
    assert result["passed"], result


def test_no_old_data_transfer_zero() -> None:
    result = check_no_old_data()
    assert result["passed"], result


def test_projected_prior_dense_marginalization() -> None:
    result = check_projected_prior_and_structured_transfer()
    assert result["passed"], result


def test_old_likelihood_dense_vs_structured_information_vector() -> None:
    result = check_projected_prior_and_structured_transfer()
    assert result["passed"], result
