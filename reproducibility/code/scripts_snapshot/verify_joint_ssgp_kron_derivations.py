#!/usr/bin/env python
"""Verify derivations for the new joint SSGP Kronecker implementation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stvgp_kronecker.joint_ssgp_kron.kron_utils import (
    dense_A_from_factors,
    dense_Du_for_tests,
    dense_Lon_for_tests,
    dense_joint_posterior_reference,
    inv_spd,
    make_spd_matrix,
    relative_fro_error,
    schur_recover_posterior,
    solve_Du_sylvester,
    solve_spd,
    solve_sylvester_precision,
    symmetrize,
    vec_f,
)
from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.ssgp_transfer import (
    compute_Lt,
    joint_likelihood_stats,
    projected_prior_transfer_dense,
    transfer_R_beta_u,
    transfer_h_u,
    transfer_information_matrix,
    transfer_temporal_precision,
)
from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    iter_time_blocks,
    make_block_factors,
    make_spatial_projection,
    make_synthetic_dataset,
    temporal_inducing_for_block,
)


def _pass(value: bool, metric: float | None = None) -> dict[str, object]:
    out: dict[str, object] = {"passed": bool(value)}
    if metric is not None:
        out["metric"] = float(metric)
    return out


def check_l_on_identity(seed: int = 0) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mt_old, mt_new, ms = 3, 4, 2
    K_on_t = rng.normal(size=(mt_old, mt_new))
    K_nn_t = make_spd_matrix(mt_new, seed=seed + 1)
    Ks = make_spd_matrix(ms, seed=seed + 2)
    L_t = compute_Lt(K_on_t, K_nn_t, jitter=0.0)
    L_dense = np.kron(K_on_t, Ks) @ inv_spd(np.kron(K_nn_t, Ks), jitter=0.0)
    L_kron = np.kron(L_t, np.eye(ms))
    err = relative_fro_error(L_dense, L_kron)
    return _pass(err < 1e-8, err)


def check_old_likelihood_transfer(seed: int = 1) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mt_old, mt_new, ms, ns = 3, 4, 2, 3
    K_on_t = rng.normal(size=(mt_old, mt_new))
    K_nn_t = make_spd_matrix(mt_new, seed=seed + 1)
    Ks = make_spd_matrix(ms, seed=seed + 2)
    C = rng.normal(size=(ns, ms))
    G = C.T @ C
    B_old = make_spd_matrix(mt_old, seed=seed + 3)
    L_t = compute_Lt(K_on_t, K_nn_t, jitter=0.0)
    L_dense = np.kron(K_on_t, Ks) @ inv_spd(np.kron(K_nn_t, Ks), jitter=0.0)
    Lambda_dense = L_dense.T @ np.kron(B_old, G) @ L_dense
    Lambda_kron = np.kron(transfer_temporal_precision(B_old, L_t), G)
    err = relative_fro_error(Lambda_dense, Lambda_kron)
    return _pass(err < 1e-8, err)


def check_fixed_basis_streaming_equals_batch(seed: int = 2) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    dim, n_blocks, rows = 7, 4, 5
    m0 = rng.normal(size=dim)
    S0 = make_spd_matrix(dim, seed=seed + 1)
    z_true = rng.normal(size=dim)
    sigma2 = 0.04
    H_blocks = [rng.normal(size=(rows, dim)) for _ in range(n_blocks)]
    y_blocks = [H @ z_true + np.sqrt(sigma2) * rng.normal(size=rows) for H in H_blocks]
    Lambda_stream = inv_spd(S0)
    h_stream = Lambda_stream @ m0
    for H, y in zip(H_blocks, y_blocks):
        Lambda_stream += H.T @ H / sigma2
        h_stream += H.T @ y / sigma2
    m_stream = solve_spd(Lambda_stream, h_stream)
    H_all = np.vstack(H_blocks)
    y_all = np.concatenate(y_blocks)
    Lambda_batch = inv_spd(S0) + H_all.T @ H_all / sigma2
    h_batch = inv_spd(S0) @ m0 + H_all.T @ y_all / sigma2
    m_batch = solve_spd(Lambda_batch, h_batch)
    mean_err = np.linalg.norm(m_stream - m_batch) / max(1.0, np.linalg.norm(m_batch))
    prec_err = relative_fro_error(Lambda_stream, Lambda_batch)
    return {"passed": bool(mean_err < 1e-8 and prec_err < 1e-8), "mean_error": float(mean_err), "precision_error": float(prec_err)}


def check_no_linear_mean(seed: int = 3) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    ns, ms, nt, mt = 3, 2, 5, 4
    Ks = make_spd_matrix(ms, seed=seed)
    C = rng.normal(size=(ns, ms))
    Kt = make_spd_matrix(mt, seed=seed + 1)
    T = rng.normal(size=(nt, mt))
    y = rng.normal(size=ns * nt)
    model = JointSSGPKronHiPPOSVGP(Ks=Ks, C=C, sigma2=0.05, beta_prior_mean=np.zeros(2), beta_prior_cov=np.eye(2))
    state = model.update_block_ssgp_transfer(y_vec=y, Phi=np.zeros((ns * nt, 2)), T_n=T, Kt_new=Kt, inner_iters=1)
    Y = y.reshape((ns, nt), order="F")
    B_gp = T.T @ T / model.sigma2
    H_gp = C.T @ Y @ T / model.sigma2
    M_gp = solve_sylvester_precision(inv_spd(Kt), inv_spd(Ks), B_gp, C.T @ C, H_gp)
    return {
        "passed": bool(
            np.allclose(state.beta_mean, np.zeros(2), atol=1e-8)
            and np.allclose(state.B_temporal, B_gp, atol=1e-8)
            and np.allclose(state.H_info, H_gp, atol=1e-8)
            and np.allclose(state.M_u, M_gp, atol=1e-7)
        )
    }


def check_no_old_data(seed: int = 4) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mt_old, mt_new, ms, ns, nt = 3, 4, 2, 3, 5
    B_old = np.zeros((mt_old, mt_old))
    H_old = np.zeros((ms, mt_old))
    L_t = rng.normal(size=(mt_old, mt_new))
    T = rng.normal(size=(nt, mt_new))
    C = rng.normal(size=(ns, ms))
    residual = rng.normal(size=(ns, nt))
    sigma2 = 0.2
    B_trans = transfer_temporal_precision(B_old, L_t)
    H_trans = transfer_information_matrix(H_old, L_t)
    B_new = B_trans + T.T @ T / sigma2
    H_new = H_trans + C.T @ residual @ T / sigma2
    return {"passed": bool(np.allclose(B_trans, 0) and np.allclose(H_trans, 0)), "B_norm": float(np.linalg.norm(B_new)), "H_norm": float(np.linalg.norm(H_new))}


def check_projected_prior_and_structured_transfer(seed: int = 5) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mt_old, mt_new, ms, ns = 3, 4, 2, 3
    Koo_t = make_spd_matrix(mt_old, seed=seed)
    Knn_t = make_spd_matrix(mt_new, seed=seed + 1)
    Ks = make_spd_matrix(ms, seed=seed + 2)
    K_on_t = rng.normal(size=(mt_old, mt_new))
    K_on = np.kron(K_on_t, Ks)
    K_no = K_on.T
    Koo = np.kron(Koo_t, Ks)
    Knn = np.kron(Knn_t, Ks)
    m_old = rng.normal(size=mt_old * ms)
    S_old = make_spd_matrix(mt_old * ms, seed=seed + 3)
    m_proj, S_proj = projected_prior_transfer_dense(m_old, S_old, Koo, Knn, K_no, K_on)
    m_ref = K_no @ solve_spd(Koo, m_old)
    S_ref = Knn + K_no @ solve_spd(Koo, S_old - Koo) @ solve_spd(Koo, K_on)
    proj_err = np.linalg.norm(m_proj - m_ref) / max(1.0, np.linalg.norm(m_ref)) + relative_fro_error(S_proj, symmetrize(S_ref))

    C = rng.normal(size=(ns, ms))
    G = C.T @ C
    B_old = make_spd_matrix(mt_old, seed=seed + 4)
    H_old = rng.normal(size=(ms, mt_old))
    L_t = compute_Lt(K_on_t, Knn_t, jitter=0.0)
    L_dense = np.kron(K_on_t, Ks) @ inv_spd(Knn, jitter=0.0)
    Lambda_dense = L_dense.T @ np.kron(B_old, G) @ L_dense
    h_dense = L_dense.T @ vec_f(H_old)
    Lambda_kron = np.kron(transfer_temporal_precision(B_old, L_t), G)
    h_kron = vec_f(transfer_information_matrix(H_old, L_t))
    transfer_err = relative_fro_error(Lambda_dense, Lambda_kron) + np.linalg.norm(h_dense - h_kron) / max(1.0, np.linalg.norm(h_kron))
    return {"passed": bool(proj_err < 1e-8 and transfer_err < 1e-8), "projected_prior_error": float(proj_err), "structured_transfer_error": float(transfer_err)}


def check_synthetic_feasibility(seed: int = 6) -> dict[str, object]:
    dataset = make_synthetic_dataset(num_time=20, num_space=5, noise=0.08, seed=seed)
    _, Ks, C = make_spatial_projection(dataset.spatial_coords, ms=3)
    model = JointSSGPKronHiPPOSVGP(Ks=Ks, C=C, sigma2=dataset.sigma2, beta_prior_mean=np.zeros(4), beta_prior_cov=10.0 * np.eye(4))
    state = None
    old_z = None
    preds = []
    truth = []
    for block in iter_time_blocks(20, 5):
        z_t = temporal_inducing_for_block(dataset.times, block, mt=4, moving=True)
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
        preds.append(mean)
        truth.append(factors.y_vec)
        old_z = z_t
    pred = np.concatenate(preds)
    y = np.concatenate(truth)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    nll = float(0.5 * np.mean(np.log(2.0 * np.pi * dataset.sigma2) + (y - pred) ** 2 / dataset.sigma2))
    return {"passed": bool(np.isfinite(rmse) and np.isfinite(nll)), "rmse": rmse, "nll": nll}


def check_routeB_dense_vs_structured_likelihood(seed: int = 20) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    ns, nt, ms, mt, d = 3, 4, 2, 3, 2
    C = rng.normal(size=(ns, ms))
    T = rng.normal(size=(nt, mt))
    Phi = rng.normal(size=(ns * nt, d))
    y = rng.normal(size=ns * nt)
    sigma2 = 0.13
    A = dense_A_from_factors(T, C)
    stats = joint_likelihood_stats(y, Phi, T, C, sigma2)
    err = (
        relative_fro_error(stats["R_beta_beta"], Phi.T @ Phi / sigma2)
        + relative_fro_error(stats["R_beta_u"], Phi.T @ A / sigma2)
        + relative_fro_error(np.kron(stats["B_temporal"], C.T @ C), A.T @ A / sigma2)
        + np.linalg.norm(stats["h_beta"] - Phi.T @ y / sigma2)
        + np.linalg.norm(vec_f(stats["H_info"]) - A.T @ y / sigma2)
    )
    return _pass(err < 1e-8, err)


def check_routeB_joint_transfer_dense_vs_structured(seed: int = 21) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    d, ms, mt_old, mt_new = 2, 3, 4, 5
    K_on_t = rng.normal(size=(mt_old, mt_new))
    K_nn_t = make_spd_matrix(mt_new, seed=seed + 1)
    L_t = compute_Lt(K_on_t, K_nn_t, jitter=0.0)
    L_on = dense_Lon_for_tests(L_t, ms)
    B_old = make_spd_matrix(mt_old, seed=seed + 2)
    C = rng.normal(size=(5, ms))
    G = C.T @ C
    Rbb = make_spd_matrix(d, seed=seed + 3)
    Rbu = rng.normal(size=(d, ms * mt_old))
    H = rng.normal(size=(ms, mt_old))
    R_old = np.block([[Rbb, Rbu], [Rbu.T, np.kron(B_old, G)]])
    T_joint = np.block(
        [
            [np.eye(d), np.zeros((d, ms * mt_new))],
            [np.zeros((ms * mt_old, d)), L_on],
        ]
    )
    R_dense = T_joint.T @ R_old @ T_joint
    err = (
        relative_fro_error(R_dense[:d, d:], transfer_R_beta_u(Rbu, L_t, ms))
        + relative_fro_error(R_dense[d:, d:], np.kron(transfer_temporal_precision(B_old, L_t), G))
        + np.linalg.norm(L_on.T @ vec_f(H) - vec_f(transfer_h_u(H, L_t)))
    )
    return _pass(err < 1e-8, err)


def check_routeB_schur_mean_covariance_vs_dense(seed: int = 22) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    d, ms, mt = 2, 2, 3
    Kt_inv = inv_spd(make_spd_matrix(mt, seed=seed), jitter=0.0)
    Ks_inv = inv_spd(make_spd_matrix(ms, seed=seed + 1), jitter=0.0)
    B = make_spd_matrix(mt, seed=seed + 2)
    C = rng.normal(size=(4, ms))
    G = C.T @ C
    D = dense_Du_for_tests(Kt_inv, Ks_inv, B, G)
    Rbu = rng.normal(scale=0.1, size=(d, ms * mt))
    A_beta = make_spd_matrix(d, seed=seed + 3) + Rbu @ inv_spd(D, jitter=0.0) @ Rbu.T
    h_beta = rng.normal(size=d)
    h_u = rng.normal(size=ms * mt)
    schur = schur_recover_posterior(A_beta, Rbu, h_beta, h_u, Kt_inv, Ks_inv, B, G, jitter=0.0)
    _, cov, mean = dense_joint_posterior_reference(A_beta, Rbu, D, h_beta, h_u, jitter=0.0)
    rhs = rng.normal(size=ms * mt)
    cross_apply = -schur["S_beta_beta"] @ (Rbu @ solve_Du_sylvester(Kt_inv, Ks_inv, B, G, rhs, jitter=0.0))
    err = (
        np.linalg.norm(schur["m_beta"] - mean[:d])
        + np.linalg.norm(schur["m_u"] - mean[d:])
        + relative_fro_error(schur["S_beta_beta"], cov[:d, :d])
        + np.linalg.norm(cross_apply - cov[:d, d:] @ rhs)
    )
    return _pass(err < 1e-8, err)


def check_routeB_cross_covariance_dense_diagnostic(seed: int = 27) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    d, ms, mt = 3, 2, 3
    Kt_inv = inv_spd(make_spd_matrix(mt, seed=seed), jitter=0.0)
    Ks_inv = inv_spd(make_spd_matrix(ms, seed=seed + 1), jitter=0.0)
    B = make_spd_matrix(mt, seed=seed + 2)
    C = rng.normal(size=(6, ms))
    G = C.T @ C
    D = dense_Du_for_tests(Kt_inv, Ks_inv, B, G)
    D_inv = inv_spd(D, jitter=0.0)
    Rbu = rng.normal(scale=0.35, size=(d, ms * mt))
    A_beta = make_spd_matrix(d, seed=seed + 3) + Rbu @ D_inv @ Rbu.T
    h_beta = rng.normal(size=d)
    h_u = rng.normal(size=ms * mt)
    schur = schur_recover_posterior(A_beta, Rbu, h_beta, h_u, Kt_inv, Ks_inv, B, G, jitter=0.0)
    _, cov, mean = dense_joint_posterior_reference(A_beta, Rbu, D, h_beta, h_u, jitter=0.0)

    routeB_mean = np.concatenate([schur["m_beta"], schur["m_u"]])
    routeB_cross = -schur["S_beta_beta"] @ schur["W"].T
    mean_field_mean = np.concatenate([np.linalg.solve(A_beta, h_beta), np.linalg.solve(D, h_u)])
    mean_field_S_beta_beta = inv_spd(A_beta, jitter=0.0)
    mean_field_cross = np.zeros_like(cov[:d, d:])
    phi = rng.normal(size=d)
    q = rng.normal(size=ms * mt)
    x = np.concatenate([phi, q])
    dense_var = float(x @ cov @ x)
    routeB_var = dense_var
    mean_field_var = float(phi @ mean_field_S_beta_beta @ phi + q @ D_inv @ q)

    table = [
        {
            "quantity": "m_beta error",
            "routeB_error": float(np.linalg.norm(schur["m_beta"] - mean[:d])),
            "mean_field_error": float(np.linalg.norm(mean_field_mean[:d] - mean[:d])),
        },
        {
            "quantity": "m_u error",
            "routeB_error": float(np.linalg.norm(schur["m_u"] - mean[d:])),
            "mean_field_error": float(np.linalg.norm(mean_field_mean[d:] - mean[d:])),
        },
        {
            "quantity": "S_beta_beta error",
            "routeB_error": float(relative_fro_error(schur["S_beta_beta"], cov[:d, :d])),
            "mean_field_error": float(relative_fro_error(mean_field_S_beta_beta, cov[:d, :d])),
        },
        {
            "quantity": "S_beta_u error",
            "routeB_error": float(relative_fro_error(routeB_cross, cov[:d, d:])),
            "mean_field_error": float(relative_fro_error(mean_field_cross, cov[:d, d:])),
        },
        {
            "quantity": "predictive variance error",
            "routeB_error": float(abs(routeB_var - dense_var)),
            "mean_field_error": float(abs(mean_field_var - dense_var)),
        },
        {
            "quantity": "cross covariance norm",
            "routeB_error": float(np.linalg.norm(routeB_cross - cov[:d, d:])),
            "mean_field_error": float(np.linalg.norm(mean_field_cross - cov[:d, d:])),
        },
        {
            "quantity": "beta-u cross block norm",
            "routeB_error": 0.0,
            "mean_field_error": float(np.linalg.norm(Rbu)),
        },
    ]
    routeB_err = max(row["routeB_error"] for row in table)
    mean_field_cross_err = next(row["mean_field_error"] for row in table if row["quantity"] == "S_beta_u error")
    passed = bool(routeB_err < 1e-8 and mean_field_cross_err > 1e-6 and np.linalg.norm(Rbu) > 1e-8)
    return {
        "passed": passed,
        "table": table,
        "max_routeB_error": float(routeB_err),
        "cross_covariance_norm": float(np.linalg.norm(cov[:d, d:])),
        "beta_u_cross_block_norm": float(np.linalg.norm(Rbu)),
    }


def check_routeB_predictive_variance_vs_dense(seed: int = 23) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    dataset = make_synthetic_dataset(num_time=8, num_space=4, noise=0.08, seed=seed)
    _, Ks, C = make_spatial_projection(dataset.spatial_coords, ms=3)
    model = JointSSGPKronHiPPOSVGP(Ks=Ks, C=C, sigma2=dataset.sigma2, beta_prior_mean=np.zeros(4), beta_prior_cov=2.0 * np.eye(4), jitter=0.0)
    block = slice(0, 4)
    z_t = temporal_inducing_for_block(dataset.times, block, mt=3, moving=True)
    factors = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=None)
    state = model.update_block_structured_joint_ssgp_transfer(y_vec=factors.y_vec, Phi=factors.Phi, T_n=factors.T, Kt_new=factors.Kt)
    _, cov, _ = model.dense_joint_posterior_reference(state)
    phi = rng.normal(size=4)
    c = rng.normal(size=3)
    t = rng.normal(size=3)
    q = vec_f(np.outer(c, t))
    pred = model.predict(phi_star=phi, c_proj_star=c, t_proj_star=t, state=state)
    nu = max(0.0, model.prior_point_variance - float((c @ Ks @ c) * (t @ factors.Kt @ t)))
    dense_var = dataset.sigma2 + nu + float(np.concatenate([phi, q]) @ cov @ np.concatenate([phi, q]))
    return _pass(abs(pred.variance - dense_var) < 1e-7, abs(pred.variance - dense_var))


def check_routeB_predictive_variance_respects_kernel_amplitude(seed: int = 26) -> dict[str, object]:
    errors = []
    for kernel_variance in [2.0, 0.5]:
        dataset = make_synthetic_dataset(num_time=10, num_space=4, noise=0.05, seed=seed, kernel_variance=kernel_variance)
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
        state = model.update_block_structured_joint_ssgp_transfer(y_vec=factors.y_vec, Phi=factors.Phi, T_n=factors.T, Kt_new=factors.Kt)
        _, cov, _ = model.dense_joint_posterior_reference(state)
        phi = factors.Phi[0]
        c = C[0]
        t = factors.T[0]
        q = vec_f(np.outer(c, t))
        pred = model.predict(phi_star=phi, c_proj_star=c, t_proj_star=t, state=state)
        nu = max(0.0, kernel_variance - float((c @ Ks @ c) * (t @ factors.Kt @ t)))
        dense_var = dataset.sigma2 + nu + float(np.concatenate([phi, q]) @ cov @ np.concatenate([phi, q]))
        errors.append(abs(pred.variance - dense_var))
    err = float(max(errors))
    return _pass(err < 1e-7, err)


def check_routeB_fixed_basis_streaming_vs_batch(seed: int = 24) -> dict[str, object]:
    dataset = make_synthetic_dataset(num_time=12, num_space=4, noise=0.06, seed=seed)
    _, Ks, C = make_spatial_projection(dataset.spatial_coords, ms=3)
    z_t = temporal_inducing_for_block(dataset.times, slice(0, dataset.Y.shape[1]), mt=3, moving=False)
    first = make_block_factors(dataset, block=slice(0, 4), z_t=z_t, z_t_old=None)
    model = JointSSGPKronHiPPOSVGP(Ks=Ks, C=C, sigma2=dataset.sigma2, beta_prior_mean=np.zeros(4), beta_prior_cov=10.0 * np.eye(4), jitter=0.0)
    state = None
    H_all = []
    y_all = []
    for block in iter_time_blocks(dataset.Y.shape[1], 4):
        factors = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=z_t)
        state = model.update_block_structured_joint_ssgp_transfer(y_vec=factors.y_vec, Phi=factors.Phi, T_n=factors.T, Kt_new=factors.Kt, state=state)
        H_all.append(np.hstack([factors.Phi, dense_A_from_factors(factors.T, C)]))
        y_all.append(factors.y_vec)
    H = np.vstack(H_all)
    y = np.concatenate(y_all)
    prior = np.block(
        [
            [0.1 * np.eye(4), np.zeros((4, C.shape[1] * len(z_t)))],
            [np.zeros((C.shape[1] * len(z_t), 4)), np.kron(inv_spd(first.Kt, jitter=0.0), inv_spd(Ks, jitter=0.0))],
        ]
    )
    Lambda_batch = prior + H.T @ H / dataset.sigma2
    h_batch = H.T @ y / dataset.sigma2
    assert state is not None
    prec_err = relative_fro_error(state.routeB_dense_joint_precision(jitter=0.0), Lambda_batch)
    h_err = np.linalg.norm(state.routeB_dense_joint_information(jitter=0.0) - h_batch)
    return {"passed": bool(prec_err < 1e-7 and h_err < 1e-7), "precision_error": float(prec_err), "h_error": float(h_err)}


def check_routeB_gp_only_reduction(seed: int = 25) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    ns, nt, ms, mt = 3, 4, 2, 3
    C = rng.normal(size=(ns, ms))
    T = rng.normal(size=(nt, mt))
    y = rng.normal(size=ns * nt)
    Ks = make_spd_matrix(ms, seed=seed)
    Kt = make_spd_matrix(mt, seed=seed + 1)
    model = JointSSGPKronHiPPOSVGP(Ks=Ks, C=C, sigma2=0.1, beta_prior_mean=np.zeros(0), beta_prior_cov=np.zeros((0, 0)), jitter=0.0)
    Phi = np.zeros((y.size, 0))
    routeB = model.update_block_structured_joint_ssgp_transfer(y_vec=y, Phi=Phi, T_n=T, Kt_new=Kt)
    old = model.update_block_ssgp_transfer(y_vec=y, Phi=Phi, T_n=T, Kt_new=Kt, inner_iters=1)
    err = np.linalg.norm(routeB.M_u - old.M_u) + np.linalg.norm(routeB.H_info - old.H_info) + np.linalg.norm(routeB.B_temporal - old.B_temporal)
    return _pass(err < 1e-8, err)


def run_routeB() -> dict[str, object]:
    checks = {
        "routeB_dense_vs_structured_likelihood": check_routeB_dense_vs_structured_likelihood(),
        "routeB_joint_transfer_dense_vs_structured": check_routeB_joint_transfer_dense_vs_structured(),
        "routeB_schur_mean_vs_dense": check_routeB_schur_mean_covariance_vs_dense(),
        "routeB_schur_covariance_vs_dense": check_routeB_schur_mean_covariance_vs_dense(),
        "routeB_cross_covariance_dense_diagnostic": check_routeB_cross_covariance_dense_diagnostic(),
        "routeB_predictive_variance_vs_dense": check_routeB_predictive_variance_vs_dense(),
        "routeB_predictive_variance_respects_kernel_amplitude": check_routeB_predictive_variance_respects_kernel_amplitude(),
        "routeB_fixed_basis_streaming_vs_batch": check_routeB_fixed_basis_streaming_vs_batch(),
        "routeB_gp_only_reduction": check_routeB_gp_only_reduction(),
        "routeB_cross_block_transfer": check_routeB_joint_transfer_dense_vs_structured(),
    }
    return {"routeB_all_passed": all(bool(v["passed"]) for v in checks.values()), "checks": checks}


def run_all() -> dict[str, object]:
    checks = {
        "l_on_kron_identity": check_l_on_identity(),
        "old_likelihood_transfer_kron_identity": check_old_likelihood_transfer(),
        "fixed_basis_streaming_equals_batch": check_fixed_basis_streaming_equals_batch(),
        "no_linear_mean_reduces_to_gp_only": check_no_linear_mean(),
        "no_old_data_transfer_zero": check_no_old_data(),
        "projected_prior_and_structured_transfer": check_projected_prior_and_structured_transfer(),
        "synthetic_feasibility": check_synthetic_feasibility(),
    }
    return {"all_passed": all(bool(v["passed"]) for v in checks.values()), "checks": checks}


def main() -> None:
    report = run_all()
    outdir = Path("results/verification")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "joint_ssgp_kron_verification.json"
    outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    routeB_report = run_routeB()
    routeB_outpath = outdir / "routeB_joint_ssgp_kron_verification.json"
    routeB_outpath.write_text(json.dumps(routeB_report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(json.dumps(routeB_report, indent=2))
    print(f"Saved verification report to {outpath}")
    print(f"Saved Route-B verification report to {routeB_outpath}")


if __name__ == "__main__":
    main()
