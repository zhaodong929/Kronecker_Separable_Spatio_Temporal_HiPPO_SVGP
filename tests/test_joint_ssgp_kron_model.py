from __future__ import annotations

import numpy as np

from scripts.run_hipposvgp_era5_routeb import vectorized_predict_with_C
from stvgp_kronecker.joint_ssgp_kron.kron_utils import dense_A_from_factors, vec_f
from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.kron_utils import (
    dense_Du_for_tests,
    make_spd_matrix,
    solve_Du_sylvester,
)


def test_sylvester_multi_rhs_matches_dense_solve() -> None:
    mt, ms = 4, 5
    kt_inv = make_spd_matrix(mt, seed=21)
    ks_inv = make_spd_matrix(ms, seed=22)
    b = make_spd_matrix(mt, seed=23)
    g = make_spd_matrix(ms, seed=24)
    rhs = np.random.default_rng(25).normal(size=(mt * ms, 3))
    actual = solve_Du_sylvester(kt_inv, ks_inv, b, g, rhs)
    expected = np.linalg.solve(dense_Du_for_tests(kt_inv, ks_inv, b, g), rhs)
    assert np.allclose(actual, expected, atol=1e-8, rtol=1e-8)
from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    iter_time_blocks,
    make_block_factors,
    make_spatial_projection,
    make_synthetic_dataset,
    temporal_inducing_for_block,
)


def _small_model(seed: int = 0):
    dataset = make_synthetic_dataset(num_time=12, num_space=4, noise=0.08, seed=seed)
    _, Ks, C = make_spatial_projection(dataset.spatial_coords, ms=3)
    model = JointSSGPKronHiPPOSVGP(
        Ks=Ks,
        C=C,
        sigma2=dataset.sigma2,
        beta_prior_mean=np.zeros(dataset.Phi.shape[1]),
        beta_prior_cov=10.0 * np.eye(dataset.Phi.shape[1]),
    )
    return dataset, model, C


def test_model_one_block_no_nan() -> None:
    dataset, model, C = _small_model()
    block = slice(0, 4)
    z_t = temporal_inducing_for_block(dataset.times, block, mt=3, moving=True)
    factors = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=None)
    state = model.update_block_ssgp_transfer(
        y_vec=factors.y_vec,
        Phi=factors.Phi,
        T_n=factors.T,
        Kt_new=factors.Kt,
        inner_iters=2,
    )
    mean = factors.Phi @ state.beta_mean + dense_A_from_factors(factors.T, C) @ vec_f(state.M_u)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(state.M_u))
    assert np.all(np.isfinite(state.B_temporal))
    assert np.all(np.isfinite(state.H_info))


def test_model_multi_block_no_nan() -> None:
    dataset, model, C = _small_model(seed=1)
    state = None
    old_z = None
    for block in iter_time_blocks(dataset.Y.shape[1], 4):
        z_t = temporal_inducing_for_block(dataset.times, block, mt=3, moving=True)
        factors = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=old_z)
        state = model.update_block_ssgp_transfer(
            y_vec=factors.y_vec,
            Phi=factors.Phi,
            T_n=factors.T,
            Kt_new=factors.Kt,
            K_on_t=factors.K_on_t,
            state=state,
            inner_iters=2,
        )
        mean = factors.Phi @ state.beta_mean + dense_A_from_factors(factors.T, C) @ vec_f(state.M_u)
        assert np.all(np.isfinite(mean))
        old_z = z_t


def test_conditional_residual_variance_toggle_only_changes_variance() -> None:
    dataset, model, c_mat = _small_model(seed=3)
    block = slice(0, 4)
    z_t = temporal_inducing_for_block(dataset.times, block, mt=3, moving=True)
    factors = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=None)
    state = model.update_block_structured_joint_ssgp_transfer(
        y_vec=factors.y_vec,
        Phi=factors.Phi,
        T_n=factors.T,
        Kt_new=factors.Kt,
    )
    mean_full, variance_full, diagnostics_full = vectorized_predict_with_C(
        model,
        state,
        factors,
        c_mat,
        include_conditional_residual_variance=True,
    )
    mean_finite, variance_finite, diagnostics_finite = vectorized_predict_with_C(
        model,
        state,
        factors,
        c_mat,
        include_conditional_residual_variance=False,
    )

    np.testing.assert_allclose(mean_full, mean_finite, rtol=0.0, atol=0.0)
    assert np.all(variance_full >= variance_finite)
    np.testing.assert_allclose(
        np.mean(variance_full - variance_finite),
        diagnostics_full["avg_nu_star_raw"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert diagnostics_full["avg_nu_star"] > 0.0
    assert diagnostics_finite["avg_nu_star"] == 0.0
    np.testing.assert_allclose(
        diagnostics_full["avg_nu_star_raw"],
        diagnostics_finite["avg_nu_star_raw"],
        rtol=0.0,
        atol=0.0,
    )


def test_baseline_imports_still_work() -> None:
    import stvgp_kronecker.st_model_batch as st_model_batch
    import stvgp_kronecker.st_model_online as st_model_online
    import stvgp_kronecker.train_batch as train_batch
    import stvgp_kronecker.train_online as train_online
    import stvgp_kronecker.train_online_joint as train_online_joint

    assert hasattr(st_model_batch, "BatchKroneckerSTHiPPOSVGP")
    assert hasattr(st_model_online, "OnlinePosteriorSummarySTGP")
    assert hasattr(train_batch, "main")
    assert hasattr(train_online, "main")
    assert hasattr(train_online_joint, "main")
