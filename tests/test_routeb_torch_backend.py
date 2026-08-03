from __future__ import annotations

import numpy as np
import pytest
import torch

from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.torch_backend import (
    TorchJointSSGPKronHiPPOSVGP,
    solve_du_sylvester,
)
from stvgp_kronecker.joint_ssgp_kron.kron_utils import (
    inv_spd,
    solve_Du_sylvester,
    vec_f,
)


def spd(rng: np.random.Generator, size: int) -> np.ndarray:
    factor = rng.normal(size=(size, size))
    return factor @ factor.T + (0.5 + size) * np.eye(size)


def assert_state_close(numpy_state, torch_state) -> None:
    names = (
        "beta_mean",
        "beta_cov",
        "M_u",
        "B_temporal",
        "H_info",
        "Kt_current",
        "Ks",
        "G",
        "R_beta_beta",
        "R_beta_u",
        "h_beta",
        "beta_prior_precision",
        "beta_prior_natural",
        "Lambda_beta_given_u",
        "S_beta_beta",
    )
    for name in names:
        expected = getattr(numpy_state, name)
        if torch.is_tensor(expected):
            expected = expected.detach().cpu().numpy()
        actual = getattr(torch_state, name).detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8, err_msg=name)


def test_torch_sylvester_matches_numpy_float64() -> None:
    rng = np.random.default_rng(7)
    ms, mt = 4, 5
    ks = spd(rng, ms)
    kt = spd(rng, mt)
    g = spd(rng, ms)
    b = spd(rng, mt)
    rhs = rng.normal(size=(ms * mt, 3))
    expected = solve_Du_sylvester(
        inv_spd(kt, jitter=1e-8),
        inv_spd(ks, jitter=1e-8),
        b,
        g,
        rhs,
        jitter=1e-8,
    )
    actual = solve_du_sylvester(
        torch.as_tensor(inv_spd(kt, jitter=1e-8)),
        torch.as_tensor(inv_spd(ks, jitter=1e-8)),
        torch.as_tensor(b),
        torch.as_tensor(g),
        torch.as_tensor(rhs),
        jitter=1e-8,
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-7, atol=1e-8)


def test_torch_routeb_fixed_and_changing_basis_match_numpy() -> None:
    rng = np.random.default_rng(11)
    ns, ms, mt, d = 6, 4, 5, 3
    sigma2 = 0.17
    ks = spd(rng, ms)
    c_train = rng.normal(size=(ns, ms))
    beta_cov = spd(rng, d)
    beta_mean = rng.normal(size=d)
    numpy_model = JointSSGPKronHiPPOSVGP(
        Ks=ks,
        C=c_train,
        sigma2=sigma2,
        beta_prior_mean=beta_mean,
        beta_prior_cov=beta_cov,
        prior_point_variance=1.3,
        jitter=1e-8,
    )
    torch_model = TorchJointSSGPKronHiPPOSVGP(
        Ks=ks,
        C=c_train,
        sigma2=sigma2,
        beta_prior_mean=beta_mean,
        beta_prior_cov=beta_cov,
        prior_point_variance=1.3,
        jitter=1e-8,
        device="cpu",
        dtype=torch.float64,
    )

    nt1 = 4
    t1 = rng.normal(size=(nt1, mt))
    kt1 = spd(rng, mt)
    phi1 = rng.normal(size=(ns * nt1, d))
    y1 = rng.normal(size=ns * nt1)
    numpy_state1 = numpy_model.update_block_structured_joint_ssgp_transfer(
        y_vec=y1, Phi=phi1, T_n=t1, Kt_new=kt1
    )
    torch_state1 = torch_model.update_block_structured_joint_ssgp_transfer(
        y_vec=y1, Phi=phi1, T_n=t1, Kt_new=kt1
    )
    assert_state_close(numpy_state1, torch_state1)

    nt2 = 3
    t2 = rng.normal(size=(nt2, mt))
    kt2 = spd(rng, mt)
    k_on = rng.normal(scale=0.1, size=(mt, mt))
    phi2 = rng.normal(size=(ns * nt2, d))
    y2 = rng.normal(size=ns * nt2)
    numpy_state2 = numpy_model.update_block_structured_joint_ssgp_transfer(
        y_vec=y2,
        Phi=phi2,
        T_n=t2,
        Kt_new=kt2,
        state=numpy_state1,
        K_on_t=k_on,
    )
    torch_state2 = torch_model.update_block_structured_joint_ssgp_transfer(
        y_vec=y2,
        Phi=phi2,
        T_n=t2,
        Kt_new=kt2,
        state=torch_state1,
        K_on_t=k_on,
    )
    assert_state_close(numpy_state2, torch_state2)

    n_eval_space, n_eval_time = 4, 3
    c_eval = rng.normal(size=(n_eval_space, ms))
    t_eval = rng.normal(size=(n_eval_time, mt))
    phi_eval = rng.normal(size=(n_eval_space * n_eval_time, d))
    expected_mean = []
    expected_variance = []
    for time_index in range(n_eval_time):
        for space_index in range(n_eval_space):
            prediction = numpy_model.predict(
                phi_star=phi_eval[time_index * n_eval_space + space_index],
                t_proj_star=t_eval[time_index],
                c_proj_star=c_eval[space_index],
                state=numpy_state2,
                include_conditional_residual_variance=False,
            )
            expected_mean.append(prediction.mean)
            expected_variance.append(prediction.variance)
    actual_mean, actual_variance, _ = torch_model.predict_with_C(
        state=torch_state2,
        T_eval=t_eval,
        Phi=phi_eval,
        C_eval=c_eval,
        chunk_size=5,
        include_conditional_residual_variance=False,
    )
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(actual_variance, expected_variance, rtol=1e-7, atol=1e-8)


def test_torch_fortran_vectorization_matches_numpy() -> None:
    rng = np.random.default_rng(19)
    matrix = rng.normal(size=(3, 5))
    from stvgp_kronecker.joint_ssgp_kron.torch_backend import vec_f as torch_vec_f

    np.testing.assert_array_equal(torch_vec_f(torch.as_tensor(matrix)).numpy(), vec_f(matrix))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_torch_cuda_matches_torch_cpu_float64() -> None:
    rng = np.random.default_rng(23)
    ns, ms, mt, nt, d = 5, 4, 6, 3, 2
    ks = spd(rng, ms)
    c_train = rng.normal(size=(ns, ms))
    beta_mean = rng.normal(size=d)
    beta_cov = spd(rng, d)
    inputs = {
        "y_vec": rng.normal(size=ns * nt),
        "Phi": rng.normal(size=(ns * nt, d)),
        "T_n": rng.normal(size=(nt, mt)),
        "Kt_new": spd(rng, mt),
    }

    def make(device: str) -> TorchJointSSGPKronHiPPOSVGP:
        return TorchJointSSGPKronHiPPOSVGP(
            Ks=ks,
            C=c_train,
            sigma2=0.2,
            beta_prior_mean=beta_mean,
            beta_prior_cov=beta_cov,
            jitter=1e-8,
            device=device,
            dtype=torch.float64,
        )

    cpu_model = make("cpu")
    cuda_model = make("cuda")
    cpu_state = cpu_model.update_block_structured_joint_ssgp_transfer(**inputs)
    cuda_state = cuda_model.update_block_structured_joint_ssgp_transfer(**inputs)
    torch.cuda.synchronize()
    assert_state_close(cpu_state, cuda_state)

    c_eval = rng.normal(size=(3, ms))
    t_eval = rng.normal(size=(2, mt))
    phi_eval = rng.normal(size=(6, d))
    cpu_mean, cpu_variance, _ = cpu_model.predict_with_C(
        state=cpu_state, T_eval=t_eval, Phi=phi_eval, C_eval=c_eval
    )
    cuda_mean, cuda_variance, _ = cuda_model.predict_with_C(
        state=cuda_state, T_eval=t_eval, Phi=phi_eval, C_eval=c_eval
    )
    np.testing.assert_allclose(cuda_mean, cpu_mean, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(cuda_variance, cpu_variance, rtol=1e-7, atol=1e-8)
