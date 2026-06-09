from __future__ import annotations

import numpy as np

from stvgp_kronecker.joint_ssgp_kron.kron_utils import (
    dense_A_from_factors,
    dense_Du_for_tests,
    dense_Lon_for_tests,
    dense_joint_posterior_reference,
    inv_spd,
    make_spd_matrix,
    schur_recover_posterior,
    solve_Du_sylvester,
    vec_f,
)
from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.ssgp_transfer import (
    compute_Lt,
    joint_likelihood_stats,
    transfer_R_beta_u,
    transfer_h_u,
    transfer_temporal_precision,
)
from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    iter_time_blocks,
    make_block_factors,
    make_spatial_projection,
    make_synthetic_dataset,
    temporal_inducing_for_block,
)


def _random_routeB_block(seed: int = 0):
    rng = np.random.default_rng(seed)
    ns, nt, ms, mt, d = 3, 4, 2, 3, 2
    C = rng.normal(size=(ns, ms))
    T = rng.normal(size=(nt, mt))
    Phi = rng.normal(size=(ns * nt, d))
    y = rng.normal(size=ns * nt)
    sigma2 = 0.17
    return rng, ns, nt, ms, mt, d, C, T, Phi, y, sigma2


def test_routeB_dense_vs_structured_new_block_likelihood() -> None:
    _, _, _, _, _, _, C, T, Phi, y, sigma2 = _random_routeB_block()
    A = dense_A_from_factors(T, C)
    stats = joint_likelihood_stats(y, Phi, T, C, sigma2)
    assert np.allclose(stats["R_beta_beta"], Phi.T @ Phi / sigma2)
    assert np.allclose(stats["R_beta_u"], Phi.T @ A / sigma2)
    assert np.allclose(np.kron(stats["B_temporal"], C.T @ C), A.T @ A / sigma2)
    assert np.allclose(stats["h_beta"], Phi.T @ y / sigma2)
    assert np.allclose(vec_f(stats["H_info"]), A.T @ y / sigma2)


def test_routeB_dense_vs_structured_joint_old_likelihood_transfer() -> None:
    rng = np.random.default_rng(1)
    d, ms, mt_old, mt_new = 2, 3, 4, 5
    K_on_t = rng.normal(size=(mt_old, mt_new))
    K_nn_t = make_spd_matrix(mt_new, seed=2)
    L_t = compute_Lt(K_on_t, K_nn_t, jitter=0.0)
    L_on = dense_Lon_for_tests(L_t, ms)
    B_old = make_spd_matrix(mt_old, seed=3)
    C = rng.normal(size=(5, ms))
    G = C.T @ C
    R_beta_beta = make_spd_matrix(d, seed=4)
    R_beta_u = rng.normal(size=(d, ms * mt_old))
    H_info = rng.normal(size=(ms, mt_old))
    R_old = np.block(
        [
            [R_beta_beta, R_beta_u],
            [R_beta_u.T, np.kron(B_old, G)],
        ]
    )
    T_joint = np.block(
        [
            [np.eye(d), np.zeros((d, ms * mt_new))],
            [np.zeros((ms * mt_old, d)), L_on],
        ]
    )
    R_dense = T_joint.T @ R_old @ T_joint
    assert np.allclose(R_dense[:d, :d], R_beta_beta)
    assert np.allclose(R_dense[:d, d:], transfer_R_beta_u(R_beta_u, L_t, ms))
    assert np.allclose(R_dense[d:, d:], np.kron(transfer_temporal_precision(B_old, L_t), G))
    assert np.allclose(L_on.T @ vec_f(H_info), vec_f(transfer_h_u(H_info, L_t)))


def test_routeB_schur_posterior_recovery_vs_dense_inverse() -> None:
    rng = np.random.default_rng(2)
    d, ms, mt = 2, 2, 3
    Kt = make_spd_matrix(mt, seed=10)
    Ks = make_spd_matrix(ms, seed=11)
    Kt_inv = inv_spd(Kt, jitter=0.0)
    Ks_inv = inv_spd(Ks, jitter=0.0)
    B = make_spd_matrix(mt, seed=12)
    C = rng.normal(size=(4, ms))
    G = C.T @ C
    D = dense_Du_for_tests(Kt_inv, Ks_inv, B, G)
    R_beta_u = rng.normal(scale=0.1, size=(d, ms * mt))
    A_beta = make_spd_matrix(d, seed=13) + R_beta_u @ inv_spd(D, jitter=0.0) @ R_beta_u.T
    h_beta = rng.normal(size=d)
    h_u = rng.normal(size=ms * mt)
    schur = schur_recover_posterior(A_beta, R_beta_u, h_beta, h_u, Kt_inv, Ks_inv, B, G, jitter=0.0)
    Lambda, cov, mean = dense_joint_posterior_reference(A_beta, R_beta_u, D, h_beta, h_u, jitter=0.0)
    assert np.allclose(schur["m_beta"], mean[:d], atol=1e-8)
    assert np.allclose(schur["m_u"], mean[d:], atol=1e-8)
    assert np.allclose(schur["S_beta_beta"], cov[:d, :d], atol=1e-8)
    rhs = rng.normal(size=ms * mt)
    cross_apply = -schur["S_beta_beta"] @ (R_beta_u @ solve_Du_sylvester(Kt_inv, Ks_inv, B, G, rhs, jitter=0.0))
    assert np.allclose(cross_apply, cov[:d, d:] @ rhs, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(Lambda) > 0)


def test_routeB_cross_covariance_matches_dense_reference() -> None:
    rng = np.random.default_rng(14)
    d, ms, mt = 3, 2, 3
    Kt_inv = inv_spd(make_spd_matrix(mt, seed=141), jitter=0.0)
    Ks_inv = inv_spd(make_spd_matrix(ms, seed=142), jitter=0.0)
    B = make_spd_matrix(mt, seed=143)
    C = rng.normal(size=(5, ms))
    G = C.T @ C
    D = dense_Du_for_tests(Kt_inv, Ks_inv, B, G)
    R_beta_u = rng.normal(scale=0.25, size=(d, ms * mt))
    A_beta = make_spd_matrix(d, seed=144) + R_beta_u @ inv_spd(D, jitter=0.0) @ R_beta_u.T
    h_beta = rng.normal(size=d)
    h_u = rng.normal(size=ms * mt)

    schur = schur_recover_posterior(A_beta, R_beta_u, h_beta, h_u, Kt_inv, Ks_inv, B, G, jitter=0.0)
    _, cov, _ = dense_joint_posterior_reference(A_beta, R_beta_u, D, h_beta, h_u, jitter=0.0)
    routeB_cross_cov = -schur["S_beta_beta"] @ schur["W"].T

    assert np.linalg.norm(cov[:d, d:]) > 1e-8
    assert np.allclose(routeB_cross_cov, cov[:d, d:], atol=1e-8)


def test_mean_field_has_zero_cross_covariance_and_differs_when_coupling_nonzero() -> None:
    rng = np.random.default_rng(15)
    d, ms, mt = 2, 2, 3
    Kt_inv = inv_spd(make_spd_matrix(mt, seed=151), jitter=0.0)
    Ks_inv = inv_spd(make_spd_matrix(ms, seed=152), jitter=0.0)
    B = make_spd_matrix(mt, seed=153)
    C = rng.normal(size=(6, ms))
    G = C.T @ C
    D = dense_Du_for_tests(Kt_inv, Ks_inv, B, G)
    D_inv = inv_spd(D, jitter=0.0)
    R_beta_u = rng.normal(scale=0.4, size=(d, ms * mt))
    A_beta = make_spd_matrix(d, seed=154) + R_beta_u @ D_inv @ R_beta_u.T
    h_beta = rng.normal(size=d)
    h_u = rng.normal(size=ms * mt)
    _, cov, mean = dense_joint_posterior_reference(A_beta, R_beta_u, D, h_beta, h_u, jitter=0.0)

    mean_field_cross_cov = np.zeros((d, ms * mt))
    assert np.linalg.norm(cov[:d, d:]) > 1e-8
    assert np.linalg.norm(mean_field_cross_cov - cov[:d, d:]) > 1e-8

    phi = rng.normal(size=d)
    q = rng.normal(size=ms * mt)
    x = np.concatenate([phi, q])
    dense_predictive_variance = float(x @ cov @ x)
    mean_field_predictive_variance = float(phi @ inv_spd(A_beta, jitter=0.0) @ phi + q @ D_inv @ q)
    assert abs(dense_predictive_variance - mean_field_predictive_variance) > 1e-8

    mean_field_mean = np.concatenate(
        [
            np.linalg.solve(A_beta, h_beta),
            np.linalg.solve(D, h_u),
        ]
    )
    assert np.linalg.norm(mean_field_mean - mean) > 1e-8


def test_routeB_predictive_variance_matches_dense_joint_posterior_and_differs_from_mean_field() -> None:
    rng, _, _, ms, mt, d, C, T, Phi, y, sigma2 = _random_routeB_block(seed=4)
    Ks = make_spd_matrix(ms, seed=20)
    Kt = make_spd_matrix(mt, seed=21)
    model = JointSSGPKronHiPPOSVGP(
        Ks=Ks,
        C=C,
        sigma2=sigma2,
        beta_prior_mean=np.zeros(d),
        beta_prior_cov=2.0 * np.eye(d),
        jitter=0.0,
    )
    state = model.update_block_structured_joint_ssgp_transfer(y_vec=y, Phi=Phi, T_n=T, Kt_new=Kt)
    _, cov, _ = model.dense_joint_posterior_reference(state)
    phi = rng.normal(size=d)
    c = rng.normal(size=ms)
    t = rng.normal(size=mt)
    q = vec_f(np.outer(c, t))
    pred = model.predict(phi_star=phi, c_proj_star=c, t_proj_star=t, state=state)
    nu = max(0.0, model.prior_point_variance - float((c @ Ks @ c) * (t @ Kt @ t)))
    x = np.concatenate([phi, q])
    dense_var = sigma2 + nu + float(x @ cov @ x)
    assert np.allclose(pred.variance, dense_var, atol=1e-7)
    D = state.dense_precision(jitter=0.0)
    mean_field_var = sigma2 + nu + float(phi @ state.S_beta_beta @ phi) + float(q @ inv_spd(D, jitter=0.0) @ q)
    assert abs(pred.variance - mean_field_var) > 1e-7


def test_routeB_fixed_basis_streaming_equals_batch_joint_posterior() -> None:
    dataset = make_synthetic_dataset(num_time=12, num_space=4, noise=0.06, seed=5)
    _, Ks, C = make_spatial_projection(dataset.spatial_coords, ms=3)
    z_t = temporal_inducing_for_block(dataset.times, slice(0, dataset.Y.shape[1]), mt=3, moving=False)
    Kt = make_block_factors(dataset, block=slice(0, 4), z_t=z_t, z_t_old=None).Kt
    model = JointSSGPKronHiPPOSVGP(
        Ks=Ks,
        C=C,
        sigma2=dataset.sigma2,
        beta_prior_mean=np.zeros(dataset.Phi.shape[1]),
        beta_prior_cov=10.0 * np.eye(dataset.Phi.shape[1]),
        jitter=0.0,
    )
    state = None
    H_all = []
    y_all = []
    for block in iter_time_blocks(dataset.Y.shape[1], 4):
        factors = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=z_t)
        state = model.update_block_structured_joint_ssgp_transfer(
            y_vec=factors.y_vec,
            Phi=factors.Phi,
            T_n=factors.T,
            Kt_new=factors.Kt,
            state=state,
            K_on_t=None,
        )
        H_all.append(np.hstack([factors.Phi, dense_A_from_factors(factors.T, C)]))
        y_all.append(factors.y_vec)
    H = np.vstack(H_all)
    y = np.concatenate(y_all)
    beta_dim = dataset.Phi.shape[1]
    prior = np.block(
        [
            [0.1 * np.eye(beta_dim), np.zeros((beta_dim, C.shape[1] * len(z_t)))],
            [np.zeros((C.shape[1] * len(z_t), beta_dim)), np.kron(inv_spd(Kt, jitter=0.0), inv_spd(Ks, jitter=0.0))],
        ]
    )
    Lambda_batch = prior + H.T @ H / dataset.sigma2
    h_batch = H.T @ y / dataset.sigma2
    assert state is not None
    assert np.allclose(state.routeB_dense_joint_precision(jitter=0.0), Lambda_batch, atol=1e-7)
    assert np.allclose(state.routeB_dense_joint_information(jitter=0.0), h_batch, atol=1e-7)
    _, _, mean_dense = model.dense_joint_posterior_reference(state)
    assert np.allclose(np.concatenate([state.beta_mean, vec_f(state.M_u)]), mean_dense, atol=1e-7)


def test_routeB_no_linear_mean_reduces_to_gp_only() -> None:
    _, _, _, _, _, _, C, T, _, y, sigma2 = _random_routeB_block(seed=6)
    Ks = make_spd_matrix(C.shape[1], seed=30)
    Kt = make_spd_matrix(T.shape[1], seed=31)
    model = JointSSGPKronHiPPOSVGP(
        Ks=Ks,
        C=C,
        sigma2=sigma2,
        beta_prior_mean=np.zeros(0),
        beta_prior_cov=np.zeros((0, 0)),
        jitter=0.0,
    )
    Phi = np.zeros((y.size, 0))
    routeB = model.update_block_structured_joint_ssgp_transfer(y_vec=y, Phi=Phi, T_n=T, Kt_new=Kt)
    gp_only = model.update_block_ssgp_transfer(y_vec=y, Phi=Phi, T_n=T, Kt_new=Kt, inner_iters=1)
    assert np.allclose(routeB.M_u, gp_only.M_u, atol=1e-8)
    assert np.allclose(routeB.B_temporal, gp_only.B_temporal, atol=1e-8)
    assert np.allclose(routeB.H_info, gp_only.H_info, atol=1e-8)


def test_routeB_zero_cross_feature_sanity() -> None:
    _, _, _, _, _, _, C, T, _, y, sigma2 = _random_routeB_block(seed=7)
    d = 2
    Phi = np.zeros((y.size, d))
    Ks = make_spd_matrix(C.shape[1], seed=40)
    Kt = make_spd_matrix(T.shape[1], seed=41)
    model = JointSSGPKronHiPPOSVGP(
        Ks=Ks,
        C=C,
        sigma2=sigma2,
        beta_prior_mean=np.zeros(d),
        beta_prior_cov=np.eye(d),
        jitter=0.0,
    )
    state = model.update_block_structured_joint_ssgp_transfer(y_vec=y, Phi=Phi, T_n=T, Kt_new=Kt)
    assert np.allclose(state.R_beta_u, 0.0)
    phi = np.array([0.2, -0.3])
    c = np.ones(C.shape[1]) / C.shape[1]
    t = np.ones(T.shape[1]) / T.shape[1]
    pred = model.predict(phi_star=phi, c_proj_star=c, t_proj_star=t, state=state)
    q = vec_f(np.outer(c, t))
    nu = max(0.0, model.prior_point_variance - float((c @ Ks @ c) * (t @ Kt @ t)))
    separate = sigma2 + nu + float(phi @ state.S_beta_beta @ phi) + float(q @ model.solve_Du(state, q))
    assert np.allclose(pred.variance, separate, atol=1e-8)


def test_predictive_variance_respects_kernel_amplitude() -> None:
    for kernel_variance in [2.0, 0.5]:
        dataset = make_synthetic_dataset(
            num_time=10,
            num_space=4,
            noise=0.05,
            seed=50,
            kernel_variance=kernel_variance,
        )
        _, Ks, C = make_spatial_projection(dataset.spatial_coords, ms=3)
        model = JointSSGPKronHiPPOSVGP(
            Ks=Ks,
            C=C,
            sigma2=dataset.sigma2,
            beta_prior_mean=np.zeros(dataset.Phi.shape[1]),
            beta_prior_cov=2.0 * np.eye(dataset.Phi.shape[1]),
            prior_point_variance=dataset.gp_prior_variance,
            jitter=0.0,
        )
        block = slice(0, 5)
        z_t = temporal_inducing_for_block(dataset.times, block, mt=3, moving=True)
        factors = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=None)
        state = model.update_block_structured_joint_ssgp_transfer(
            y_vec=factors.y_vec,
            Phi=factors.Phi,
            T_n=factors.T,
            Kt_new=factors.Kt,
        )
        _, cov, _ = model.dense_joint_posterior_reference(state)
        phi = factors.Phi[0]
        c = C[0]
        t = factors.T[0]
        q = vec_f(np.outer(c, t))
        pred = model.predict(phi_star=phi, c_proj_star=c, t_proj_star=t, state=state)
        nu = max(0.0, kernel_variance - float((c @ Ks @ c) * (t @ factors.Kt @ t)))
        dense_var = dataset.sigma2 + nu + float(np.concatenate([phi, q]) @ cov @ np.concatenate([phi, q]))
        assert np.allclose(pred.variance, dense_var, atol=1e-7)
