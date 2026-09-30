#!/usr/bin/env python
"""Generate the 40-test Route B verification walkthrough docs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "routeB_40_math_verification_walkthrough.md"
OUTDIR = ROOT / "results" / "routeB_40_math_verification_walkthrough"
REPORT_TEX = ROOT / "results" / "routeB_experiment_report" / "routeB_experiment_report.tex"


@dataclass(frozen=True)
class VerificationItem:
    idx: int
    test: str
    group: str
    file: str
    formula: str
    code: str
    logic: str
    result: str = "PASSED"


ITEMS: list[VerificationItem] = [
    VerificationItem(
        1,
        "test_temporal_and_spatial_shape_consistency",
        "Stage-1 Kronecker STGP",
        "stvgp_kronecker/tests/test_stage1.py",
        r"$K_{uu}^{(t)}\in\mathbb{R}^{M_t\times M_t},\ K_{fu}^{(t)}\in\mathbb{R}^{T\times M_t},\ K_{zz}^{(s)}\in\mathbb{R}^{M_s\times M_s},\ K_{xz}^{(s)}\in\mathbb{R}^{S\times M_s}$",
        "temporal = model.build_temporal_covariances(times)\nspatial_cov = model.build_spatial_covariances(spatial)\nassert temporal.kuu_t.shape == (4, 4)\nassert temporal.kfu_t.shape == (5, 4)",
        "验证时间核和空间核模块输出的矩阵维度与 Kronecker STGP 的理论对象一致。",
    ),
    VerificationItem(
        2,
        "test_kronecker_projection_shapes",
        "Stage-1 Kronecker STGP",
        "stvgp_kronecker/tests/test_stage1.py",
        r"$A=A_t\otimes A_s,\quad A_t=K_{fu}^{(t)}K_{uu}^{(t)-1},\quad A_s=K_{xz}^{(s)}K_{zz}^{(s)-1}$",
        "projection = model.build_projection(times, spatial)\nassert projection.a_t.shape == (5, 4)\nassert projection.a_s.shape == (4, 4)\nassert torch.kron(projection.a_t, projection.a_s).shape == (20, 16)",
        "验证 Kronecker 投影矩阵的行数等于所有时空观测点，列数等于时间诱导点乘空间诱导点。",
    ),
    VerificationItem(
        3,
        "test_small_synthetic_batch_matches_dense_solution",
        "Stage-1 Kronecker STGP",
        "stvgp_kronecker/tests/test_stage1.py",
        r"$\Lambda=K_{uu}^{-1}+\sigma^{-2}A^\top A,\quad m=\Lambda^{-1}\sigma^{-2}A^\top y$",
        "precision = Kuu_inv + torch.reciprocal(sigma2) * (a_dense.T @ a_dense)\ninfo = torch.reciprocal(sigma2) * (a_dense.T @ y.reshape(-1))\nmean = torch.linalg.solve(precision, info)\nassert torch.allclose(output['posterior_mean_u'], mean)",
        "把模型的 batch posterior mean 与显式 dense GP 线性高斯后验公式逐元素比较。",
    ),
    VerificationItem(
        4,
        "test_small_synthetic_training_reduces_loss",
        "Stage-1 Kronecker STGP",
        "stvgp_kronecker/tests/test_stage1.py",
        r"$\mathcal{L}_{20}<\mathcal{L}_{0}$",
        "for _ in range(20):\n    output = model(times, spatial, y, cache_posterior=False)\n    output['loss'].backward()\n    optimizer.step()\nassert losses[-1] < losses[0]",
        "验证训练目标可被优化器下降，排除 loss 符号、梯度和参数注册错误。",
    ),
    VerificationItem(
        5,
        "test_blockwise_forward_returns_consistent_shapes",
        "Stage-1 Kronecker STGP",
        "stvgp_kronecker/tests/test_stage1.py",
        r"$\{B_1,\ldots,B_N\},\quad \hat{Y}_{B_n}\in\mathbb{R}^{|B_n|\times S}$",
        "blockwise = model.forward_blockwise(..., block_size=2)\nassert len(blockwise.block_outputs) == 3\nassert blockwise.block_outputs[-1]['train_mean'].shape[1] == spatial.shape[0]",
        "验证 blockwise 前向传播在分块后仍返回与空间网格匹配的预测形状。",
    ),
    VerificationItem(
        6,
        "test_online_recursion_matches_batch_solution",
        "Stage-1 Kronecker STGP",
        "stvgp_kronecker/tests/test_stage1.py",
        r"$\Lambda_N=\Lambda_0+\sum_n\sigma^{-2}A_n^\top A_n,\quad h_N=h_0+\sum_n\sigma^{-2}A_n^\top y_n$",
        "batch_output = batch_model(..., materialize_posterior_cov=True)\nfor block in blocks:\n    online_model.update_block(...)\nassert torch.allclose(batch_output['posterior_mean_u'], online_model.state.m)",
        "固定 horizon 下，online precision 累加应与一次性 batch posterior 完全一致。",
    ),
    VerificationItem(
        7,
        "test_temporal_cross_covariance_is_consistent",
        "Stage-1 Kronecker STGP",
        "stvgp_kronecker/tests/test_stage1.py",
        r"$K_{ab}=K_{ba}^{\top}$",
        "cross_ab = builder.compute_kuu_t_cross(horizon_a, horizon_b)\ncross_ba = builder.compute_kuu_t_cross(horizon_b, horizon_a)\nassert torch.allclose(cross_ab, cross_ba.T)",
        "验证不同 temporal horizons 之间的 cross covariance 满足核矩阵对称性。",
    ),
    VerificationItem(
        8,
        "test_online_local_horizon_transfer_updates_state",
        "Stage-1 Kronecker STGP",
        "stvgp_kronecker/tests/test_stage1.py",
        r"$u_o\mapsto u_n,\quad L_{on}=K_{on}K_{nn}^{-1}$",
        "first = online_model.update_block(..., horizon=first_block_horizon)\nsecond = online_model.update_block(..., horizon=second_block_horizon)\nassert first['temporal_transfer'].shape == (4, 4)",
        "验证 local horizon 改变时 transfer matrix 存在、有限，并且 online 状态能连续更新。",
    ),
    VerificationItem(
        9,
        "test_online_predictive_variance_matches_dense_precision_solver",
        "Stage-1 Kronecker STGP",
        "stvgp_kronecker/tests/test_stage1.py",
        r"$\operatorname{Var}(f_*)=k_{**}-a_*K_{uu}a_*^\top+a_*\Lambda^{-1}a_*^\top$",
        "dense_latent_var = prior_diag - projected_prior_diag + dense_posterior_correction\nassert torch.allclose(pred['latent_var'], dense_latent_var)",
        "验证 Sylvester/precision solver 给出的 latent variance 与显式 dense precision correction 一致。",
    ),
    VerificationItem(
        10,
        "test_load_processed_era5_task_aligns_locations",
        "ERA5 Data Contract",
        "stvgp_kronecker/tests/test_stage1.py",
        r"$Y\in\mathbb{R}^{T\times S}$ with shared sorted times over all selected locations",
        "task = load_processed_era5_task(...)\nassert task.train.times.tolist() == [0.0, 1.0, 2.0]\nassert task.train.observations.shape == (3, 2)",
        "验证 processed ERA5 每个 location 的乱序时间戳会被排序，并在共同时间轴上对齐。",
    ),
    VerificationItem(
        11,
        "test_load_processed_era5_task_resplit_rebuilds_longer_validation",
        "ERA5 Data Contract",
        "stvgp_kronecker/tests/test_stage1.py",
        r"$\{1,\ldots,T\}=\mathcal{T}_{train}\cup\mathcal{T}_{val}\cup\mathcal{T}_{test}$ chronologically",
        "task = load_processed_era5_task(..., resplit=True)\nassert task.train.times.tolist() == [0.0, 1.0, 2.0]\nassert task.val.times.tolist() == [3.0, 4.0]",
        "验证 chronological resplit 不打乱时间顺序，避免 online/future evaluation 泄漏。",
    ),
    VerificationItem(
        12,
        "test_load_processed_era5_tasks_concatenates_multiple_tasks",
        "ERA5 Data Contract",
        "stvgp_kronecker/tests/test_stage1.py",
        r"$Y_{1:K}=[Y^{(1)};Y^{(2)};\ldots;Y^{(K)}]$ with shared spatial coordinates",
        "task = load_processed_era5_tasks([task_1, task_2], resplit=True)\nassert task.train.observations.shape == (6, 2)\nassert task.test.times.tolist() == [9.0, 10.0, 11.0]",
        "验证多个 ERA5 task 可以按时间拼接，并保持空间位置一致。",
    ),
    VerificationItem(
        13,
        "test_discover_and_count_processed_era5_tasks",
        "ERA5 Data Contract",
        "stvgp_kronecker/tests/test_stage1.py",
        r"$|\mathcal{S}_{task}|=\#\{\text{selected location files}\}$",
        "task_dirs = discover_processed_era5_task_dirs(root, ['task_1', 'task_2'])\nassert count_processed_era5_locations(task_dirs[0]) == 2",
        "验证 task discovery 和 location counting 与磁盘上的 processed 数据结构一致。",
    ),
    VerificationItem(
        14,
        "test_spatial_inducing_fps_spreads_across_domain",
        "Spatial Inducing Contract",
        "stvgp_kronecker/tests/test_stage1.py",
        r"$Z_s=\operatorname{FPS}(X_s)$ should cover the spatial domain better than first-N selection",
        "first = select_spatial_inducing_points(..., selection_method='first')\nfps = select_spatial_inducing_points(..., selection_method='fps')\nassert float(fps[:, 0].min()) == -10.0\nassert float(fps[:, 0].max()) == 2.0",
        "验证 farthest-point spatial inducing selection 能覆盖左右空间边界。",
    ),
    VerificationItem(
        15,
        "test_Lon_kron_identity",
        "Kronecker Derivations",
        "tests/test_joint_ssgp_kron_derivations.py",
        r"$L_{on}=K_{on}K_{nn}^{-1}=(K_{on}^{t}K_{nn}^{t-1})\otimes I_s$",
        "L_t = compute_Lt(K_on_t, K_nn_t)\nL_dense = kron(K_on_t, Ks) @ inv(kron(K_nn_t, Ks))\nL_kron = kron(L_t, I_s)\nassert err < 1e-8",
        "验证 changing-basis transfer 的 dense 形式可化简为时间维 transfer 与空间单位矩阵的 Kronecker 积。",
    ),
    VerificationItem(
        16,
        "test_old_likelihood_transfer_kron_identity",
        "Kronecker Derivations",
        "tests/test_joint_ssgp_kron_derivations.py",
        r"$L_{on}^{\top}(B_o\otimes G)L_{on}=(L_t^{\top}B_oL_t)\otimes G$",
        "Lambda_dense = L_dense.T @ kron(B_old, G) @ L_dense\nLambda_kron = kron(transfer_temporal_precision(B_old, L_t), G)\nassert err < 1e-8",
        "验证旧 likelihood precision 在新时间基上投影后仍保持 Kronecker 分解。",
    ),
    VerificationItem(
        17,
        "test_fixed_basis_streaming_equals_batch",
        "Kronecker Derivations",
        "tests/test_joint_ssgp_kron_derivations.py",
        r"$\Lambda_{stream}=\Lambda_0+\sum_n H_n^\top H_n/\sigma^2=\Lambda_{batch}$",
        "for H, y in blocks:\n    Lambda_stream += H.T @ H / sigma2\n    h_stream += H.T @ y / sigma2\nassert mean_err < 1e-8 and prec_err < 1e-8",
        "验证固定 basis 下 online natural parameters 累加与 batch Gaussian posterior 完全一致。",
    ),
    VerificationItem(
        18,
        "test_no_linear_mean_reduces_to_gp_only",
        "Kronecker Derivations",
        "tests/test_joint_ssgp_kron_derivations.py",
        r"$\Phi=0\Rightarrow \beta=0,\quad H_u=C^\top YT/\sigma^2,\quad B=T^\top T/\sigma^2$",
        "state = model.update_block_ssgp_transfer(y_vec=y, Phi=zeros, ...)\nassert allclose(state.beta_mean, 0)\nassert allclose(state.B_temporal, B_gp)\nassert allclose(state.H_info, H_gp)",
        "验证线性均值为零时模型退化为 GP-only SSGP update。",
    ),
    VerificationItem(
        19,
        "test_no_old_data_transfer_zero",
        "Kronecker Derivations",
        "tests/test_joint_ssgp_kron_derivations.py",
        r"$B_o=0,\ H_o=0\Rightarrow B_{o\to n}=0,\ H_{o\to n}=0$",
        "B_trans = transfer_temporal_precision(B_old_zero, L_t)\nH_trans = transfer_information_matrix(H_old_zero, L_t)\nassert allclose(B_trans, 0) and allclose(H_trans, 0)",
        "验证没有旧数据时 transfer 项不会凭空产生旧 likelihood 信息。",
    ),
    VerificationItem(
        20,
        "test_projected_prior_dense_marginalization",
        "Kronecker Derivations",
        "tests/test_joint_ssgp_kron_derivations.py",
        r"$m_n=K_{no}K_{oo}^{-1}m_o,\quad S_n=K_{nn}+K_{no}K_{oo}^{-1}(S_o-K_{oo})K_{oo}^{-1}K_{on}$",
        "m_proj, S_proj = projected_prior_transfer_dense(...)\nassert projected_prior_error < 1e-8\nassert structured_transfer_error < 1e-8",
        "验证旧 projected-prior dense marginalization 公式和 structured old-likelihood transfer 公式都能匹配 dense reference。",
    ),
    VerificationItem(
        21,
        "test_old_likelihood_dense_vs_structured_information_vector",
        "Kronecker Derivations",
        "tests/test_joint_ssgp_kron_derivations.py",
        r"$h_{u,o\to n}=L_{on}^{\top}h_{u,o},\quad \operatorname{vec}(H_oL_t)=L_{on}^{\top}\operatorname{vec}(H_o)$",
        "h_dense = L_dense.T @ vec_f(H_old)\nh_kron = vec_f(transfer_information_matrix(H_old, L_t))\nassert norm(h_dense - h_kron) < 1e-8",
        "验证旧 information vector 的 dense transfer 与矩阵形式 $H_oL_t$ 一致。",
    ),
    VerificationItem(
        22,
        "test_model_one_block_no_nan",
        "Model Sanity",
        "tests/test_joint_ssgp_kron_model.py",
        r"$\hat y=\Phi m_\beta + A\,\operatorname{vec}(M_u)$ finite",
        "state = model.update_block_ssgp_transfer(...)\nmean = Phi @ state.beta_mean + A @ vec_f(state.M_u)\nassert np.all(np.isfinite(mean))",
        "验证单个 block 更新不会产生 NaN/inf，保护数值稳定性。",
    ),
    VerificationItem(
        23,
        "test_model_multi_block_no_nan",
        "Model Sanity",
        "tests/test_joint_ssgp_kron_model.py",
        r"$\forall n,\ \hat y_n=\Phi_nm_{\beta,n}+A_n\operatorname{vec}(M_{u,n})$ finite",
        "for block in iter_time_blocks(...):\n    state = model.update_block_ssgp_transfer(..., state=state)\n    assert np.all(np.isfinite(mean))",
        "验证多 block online transfer 后均值和状态仍有限。",
    ),
    VerificationItem(
        24,
        "test_baseline_imports_still_work",
        "Model Sanity",
        "tests/test_joint_ssgp_kron_model.py",
        r"$\text{public API remains importable after Route B additions}$",
        "import stvgp_kronecker.train_batch as train_batch\nimport stvgp_kronecker.train_online as train_online\nassert hasattr(train_batch, 'main')",
        "验证新增 Route B 代码没有破坏原 batch/online 入口和旧 API。",
    ),
    VerificationItem(
        25,
        "test_routeB_dense_vs_structured_new_block_likelihood",
        "Route B Theory",
        "tests/test_joint_ssgp_kron_routeB.py",
        r"$R_{\beta\beta}=\Phi^\top\Phi/\sigma^2,\ R_{\beta u}=\Phi^\top A/\sigma^2,\ R_{uu}=A^\top A/\sigma^2=(T^\top T)\otimes(C^\top C)/\sigma^2$",
        "stats = joint_likelihood_stats(y, Phi, T, C, sigma2)\nassert allclose(stats['R_beta_beta'], Phi.T @ Phi / sigma2)\nassert allclose(stats['R_beta_u'], Phi.T @ A / sigma2)",
        "验证新 block joint likelihood 的 dense 统计量与 structured Kronecker 统计量一致。",
    ),
    VerificationItem(
        26,
        "test_routeB_dense_vs_structured_joint_old_likelihood_transfer",
        "Route B Theory",
        "tests/test_joint_ssgp_kron_routeB.py",
        r"$R_{\beta u,o\to n}=R_{\beta u,o}L_{on},\quad R_{uu,o\to n}=L_{on}^{\top}R_{uu,o}L_{on}$",
        "R_dense = T_joint.T @ R_old @ T_joint\nassert allclose(R_dense[:d, d:], transfer_R_beta_u(R_beta_u, L_t, ms))\nassert allclose(R_dense[d:, d:], kron(transfer_temporal_precision(B_old, L_t), G))",
        "验证保留 beta-u cross block 后，旧 joint likelihood 的 basis transfer 仍与 dense 变换一致。",
    ),
    VerificationItem(
        27,
        "test_routeB_schur_posterior_recovery_vs_dense_inverse",
        "Route B Theory",
        "tests/test_joint_ssgp_kron_routeB.py",
        r"$S_{\beta|u}=(A_\beta-R_{\beta u}D_u^{-1}R_{u\beta})^{-1},\quad m_u=D_u^{-1}(h_u-R_{u\beta}m_\beta)$",
        "schur = schur_recover_posterior(...)\n_, cov, mean = dense_joint_posterior_reference(...)\nassert allclose(schur['m_beta'], mean[:d])\nassert allclose(schur['S_beta_beta'], cov[:d, :d])",
        "验证 Schur complement + Sylvester solves 恢复的 posterior mean/covariance 与 dense inverse 一致。",
    ),
    VerificationItem(
        28,
        "test_routeB_cross_covariance_matches_dense_reference",
        "Route B Theory",
        "tests/test_joint_ssgp_kron_routeB.py",
        r"$S_{\beta u}=-S_{\beta\beta}R_{\beta u}D_u^{-1}$",
        "routeB_cross_cov = -schur['S_beta_beta'] @ schur['W'].T\nassert np.linalg.norm(cov[:d, d:]) > 1e-8\nassert np.allclose(routeB_cross_cov, cov[:d, d:])",
        "验证 Route B 保留的 beta-u posterior cross covariance 与 dense reference 完全一致。",
    ),
    VerificationItem(
        29,
        "test_mean_field_has_zero_cross_covariance_and_differs_when_coupling_nonzero",
        "Route B Theory",
        "tests/test_joint_ssgp_kron_routeB.py",
        r"$S_{\beta u}^{MF}=0,\quad S_{\beta u}^{dense}\ne 0$ under nonzero coupling",
        "mean_field_cross_cov = np.zeros((d, ms * mt))\nassert norm(mean_field_cross_cov - cov[:d, d:]) > 1e-8\nassert abs(dense_predictive_variance - mean_field_predictive_variance) > 1e-8",
        "验证 mean-field 在强 coupling 下确实丢失 cross covariance，并导致预测方差/均值偏离 dense posterior。",
    ),
    VerificationItem(
        30,
        "test_routeB_predictive_variance_matches_dense_joint_posterior_and_differs_from_mean_field",
        "Route B Theory",
        "tests/test_joint_ssgp_kron_routeB.py",
        r"$\operatorname{Var}(y_*)=\sigma^2+\nu_*+[\phi_*,q_*]S[\phi_*,q_*]^\top$",
        "pred = model.predict(phi_star=phi, c_proj_star=c, t_proj_star=t, state=state)\ndense_var = sigma2 + nu + x @ cov @ x\nassert allclose(pred.variance, dense_var)\nassert abs(pred.variance - mean_field_var) > 1e-7",
        "验证 Route B 预测方差包含 dense joint posterior 的 cross term，并区别于 mean-field 方差。",
    ),
    VerificationItem(
        31,
        "test_routeB_fixed_basis_streaming_equals_batch_joint_posterior",
        "Route B Theory",
        "tests/test_joint_ssgp_kron_routeB.py",
        r"$\Lambda_N=\Lambda_0+\sum_n\sigma^{-2}[\Phi_n,A_n]^\top[\Phi_n,A_n]$",
        "state = model.update_block_structured_joint_ssgp_transfer(...)\nLambda_batch = prior + H.T @ H / sigma2\nassert allclose(state.routeB_dense_joint_precision(), Lambda_batch)",
        "验证固定 basis 下 Route B streaming joint posterior 与 batch joint posterior 一致。",
    ),
    VerificationItem(
        32,
        "test_routeB_no_linear_mean_reduces_to_gp_only",
        "Route B Theory",
        "tests/test_joint_ssgp_kron_routeB.py",
        r"$d_\beta=0\Rightarrow R_{\beta u}\ \text{empty and Route B}=GP\text{-only SSGP}$",
        "Phi = np.zeros((y.size, 0))\nrouteB = model.update_block_structured_joint_ssgp_transfer(...)\ngp_only = model.update_block_ssgp_transfer(...)\nassert allclose(routeB.M_u, gp_only.M_u)",
        "验证没有线性均值时，Route B 不引入额外行为，退化为原 GP-only 更新。",
    ),
    VerificationItem(
        33,
        "test_routeB_zero_cross_feature_sanity",
        "Route B Theory",
        "tests/test_joint_ssgp_kron_routeB.py",
        r"$\Phi=0\Rightarrow R_{\beta u}=0,\quad \operatorname{Var}(y_*)=\sigma^2+\nu_*+\phi_*^\top S_{\beta\beta}\phi_*+q_*^\top D_u^{-1}q_*$",
        "Phi = np.zeros((y.size, d))\nstate = model.update_block_structured_joint_ssgp_transfer(...)\nassert allclose(state.R_beta_u, 0.0)\nassert allclose(pred.variance, separate)",
        "验证 cross block 为零时 Route B 方差自动分解为 beta 项和 GP 项的相加形式。",
    ),
    VerificationItem(
        34,
        "test_predictive_variance_respects_kernel_amplitude",
        "Route B Theory",
        "tests/test_joint_ssgp_kron_routeB.py",
        r"$\nu_*=k(x_*,x_*)-k_{*u}K_{uu}^{-1}k_{u*}$ with $k(x_*,x_*)=\text{kernel variance}$",
        "for kernel_variance in [2.0, 0.5]:\n    model = JointSSGPKronHiPPOSVGP(..., prior_point_variance=dataset.gp_prior_variance)\n    dense_var = sigma2 + nu + x @ cov @ x\n    assert allclose(pred.variance, dense_var)",
        "验证 sparse conditional residual variance 显式尊重非单位 kernel amplitude，修复 coverage/NLL 风险点。",
    ),
    VerificationItem(
        35,
        "test_loader_shapes_and_blocks",
        "ERA5 Baseline Pipeline",
        "tests/test_hipposvgp_era5_loader_baselines.py",
        r"$Y\in\mathbb{R}^{T\times S},\quad \Phi\in\mathbb{R}^{TS\times p},\quad B_n=[t_n,t_n+b)$",
        "dataset = load_hipposvgp_era5(..., first_n_locations=2, split='all')\nassert dataset.Y.shape == (6, 2)\nassert dataset.Phi.shape[0] == 12\nassert [(b.start, b.stop) for b in blocks] == [(0, 2), (2, 4), (4, 6)]",
        "验证新的 HiPPO-SVGP ERA5 loader 输出的 Y、coords、Phi 和 online block split 与 baseline/Route B 需要的形状一致。",
    ),
    VerificationItem(
        36,
        "test_loader_converts_to_routeb_factors",
        "ERA5 Baseline Pipeline",
        "tests/test_hipposvgp_era5_loader_baselines.py",
        r"$Y_{loader}^{T\times S}\mapsto Y_{RouteB}^{S\times T},\quad (y_n,\Phi_n,T_n,K_t)$ match BlockFactors",
        "factors = make_routeb_block_factors(dataset, block=slice(0, 2), ...)\nassert factors.Y.shape == (2, 2)\nassert factors.Phi.shape[0] == 4\nassert np.isfinite(factors.y_vec).all()",
        "验证 ERA5 loader 可以转换成 Route B 的 BlockFactors，不改变主模型公式。",
    ),
    VerificationItem(
        37,
        "test_deterministic_baselines_shapes_no_leakage_and_finite_variance[PersistenceBaseline]",
        "ERA5 Baseline Pipeline",
        "tests/test_hipposvgp_era5_loader_baselines.py",
        r"$\hat y_{future}=g(\mathcal{D}_{seen},x_{future}),\quad \operatorname{Var}(\hat y)>0$",
        "baseline.fit_initial_task(times[:3], coords, Y[:3], Phi[:3])\npred_before = baseline.predict(future_times, coords, future_phi)\nY_modified_future[3:5] += 1000.0\npred_after = baseline.predict(future_times, coords, future_phi)\nassert np.allclose(pred_before.mean, pred_after.mean)\nassert np.all(pred_before.variance > 0.0)",
        "验证 persistence baseline 的 future prediction 不读取 future labels，并且残差方差有限且为正。",
    ),
    VerificationItem(
        38,
        "test_deterministic_baselines_shapes_no_leakage_and_finite_variance[ClimatologyBaseline]",
        "ERA5 Baseline Pipeline",
        "tests/test_hipposvgp_era5_loader_baselines.py",
        r"$\hat y_s=\frac{1}{|\mathcal{D}_{seen}|}\sum_{t\in seen}y_{t,s},\quad \sigma_s^2=\operatorname{Var}(y_{t,s}-\hat y_s)$",
        "baseline = ClimatologyBaseline()\nbaseline.fit_initial_task(times[:3], coords, Y[:3], build_phi_features(times[:3], coords))\npred = baseline.predict(future_times, coords, future_phi)\nassert pred.mean.shape == (2, 2)\nassert np.all(pred.variance > 0.0)",
        "验证 climatology baseline 的均值和 variance 只由 seen history 估计，future labels 修改不影响预测。",
    ),
    VerificationItem(
        39,
        "test_deterministic_baselines_shapes_no_leakage_and_finite_variance[RidgeBaseline]",
        "ERA5 Baseline Pipeline",
        "tests/test_hipposvgp_era5_loader_baselines.py",
        r"$\hat\beta=(\Phi^\top\Phi+\lambda I)^{-1}\Phi^\top y,\quad \sigma^2=\operatorname{Var}(y-\Phi\hat\beta)$",
        "baseline = RidgeBaseline()\nbaseline.fit_initial_task(times[:3], coords, Y[:3], build_phi_features(times[:3], coords))\npred = baseline.predict(future_times, coords, future_phi)\nassert pred.mean.shape == (2, 2)\nassert np.all(pred.variance > 0.0)",
        "验证 ridge baseline 的闭式解、输出形状、no-leakage future prediction 和 residual variance 都可用。",
    ),
    VerificationItem(
        40,
        "test_gpytorch_baselines_smoke_if_available",
        "ERA5 Baseline Pipeline",
        "tests/test_hipposvgp_era5_loader_baselines.py",
        r"$p(y_*|\mathcal{D})=\mathcal{N}(\mu_*,\sigma_{*,likelihood}^2),\quad \sigma_{*,likelihood}^2>0$",
        "for baseline in [IndependentTemporalGPBaseline, GPyTorchSGPRBaseline, GPyTorchSVGPBaseline]:\n    baseline.fit_initial_task(times, coords, Y, Phi)\n    pred = baseline.predict(times[:2], coords, build_phi_features(times[:2], coords))\n    assert pred.mean.shape == (2, 1)\n    assert np.all(pred.variance > 0.0)",
        "验证 independent GP、SGPR 和 SVGP 三个 GPyTorch baseline 都能训练、返回 likelihood predictive variance，并且 variance 有限为正。",
    ),
]


def markdown_escape_pipe(text: str) -> str:
    return text.replace("|", "\\|")


def syntax_notes_for(code: str) -> list[str]:
    """Return beginner-friendly Python syntax notes for a snippet."""

    notes: list[str] = []
    if "=" in code:
        notes.append("`=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。")
    if "assert " in code:
        notes.append("`assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。")
    if "assert np.allclose" in code or "assert torch.allclose" in code or "allclose(" in code:
        notes.append("`allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。")
    if "np.isfinite" in code or "torch.isfinite" in code:
        notes.append("`isfinite` 检查结果不是 `NaN` 也不是无穷大，用来确认数值计算稳定。")
    if "for " in code:
        notes.append("`for ... in ...:` 是循环；冒号后面缩进的多行代码会重复执行。")
    if "if " in code:
        notes.append("`if 条件:` 是条件分支；只有条件成立时才执行缩进块。")
    if "with " in code:
        notes.append("`with ...:` 是上下文管理语法，常用于自动打开/关闭文件或临时切换设置。")
    if "..." in code:
        notes.append("`...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。")
    if "@" in code:
        notes.append("`@` 是矩阵乘法运算符，例如 `Phi.T @ Phi` 表示转置矩阵乘矩阵。")
    if ".T" in code:
        notes.append("`.T` 表示矩阵转置。")
    if "[" in code and "]" in code:
        notes.append("方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。")
    if "shape" in code:
        notes.append("`.shape` 是数组维度；例如 `(6, 2)` 表示 6 行、2 列。")
    if "np.zeros" in code:
        notes.append("`np.zeros(shape)` 创建全 0 数组，常用来构造零特征或零 cross-covariance。")
    if "np.eye" in code:
        notes.append("`np.eye(n)` 创建 `n x n` 单位矩阵。")
    if "np.linalg" in code or "torch.linalg" in code:
        notes.append("`np.linalg` / `torch.linalg` 是线性代数工具箱，用于求解线性方程、Cholesky 分解、特征值等。")
    if "range(" in code:
        notes.append("`range(n)` 生成从 0 到 n-1 的整数序列，常与 `for` 循环配合。")
    if "lambda" in code:
        notes.append("这里的 `lambda` 若出现在变量名里只是普通文本；Python 中真正的 `lambda` 关键字表示匿名函数，本项目代码片段里基本不用。")
    if "baseline." in code or "model." in code:
        notes.append("点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。")
    if not notes:
        notes.append("这段代码主要是函数调用和变量比较；阅读时先看 `assert` 后面的条件，它就是该测试要证明的结论。")
    return notes


def explain_code_line(line: str) -> str:
    """Translate one displayed code line into a plain-language implementation explanation."""

    stripped = line.strip()
    if not stripped:
        return "空行，用来分隔代码块，让结构更清楚。"
    if stripped == "...":
        return "这里省略了与当前验证重点无关的参数或代码；完整实现可以在对应测试文件中查看。"
    if stripped.startswith("#"):
        return "这是注释，用自然语言提示下面代码的目的，不会被 Python 执行。"
    if stripped.startswith("for baseline in"):
        return "依次取出每一个 GPyTorch baseline，让同一套训练和预测检查分别作用在 independent GP、SGPR 和 SVGP 上。"
    if stripped.startswith("for ") and " in blocks" in stripped:
        return "逐个遍历 online 时间块，模拟持续学习中一块一块接收数据的过程。"
    if stripped.startswith("for ") and " in range" in stripped:
        return "重复执行固定次数，用来模拟训练迭代或分块循环。"
    if stripped.startswith("for "):
        return "开始一个循环，对集合里的每个元素重复执行缩进代码。"
    if stripped.startswith("if "):
        return "根据条件决定是否执行后面的缩进代码。"
    if stripped.startswith("with "):
        return "进入一个临时上下文，例如临时关闭梯度或打开文件；退出后自动恢复/关闭。"
    if stripped.startswith("assert np.allclose") or stripped.startswith("assert torch.allclose") or stripped.startswith("assert allclose"):
        return "检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。"
    if stripped.startswith("assert np.all(np.isfinite") or stripped.startswith("assert torch.isfinite") or "isfinite" in stripped:
        return "检查计算结果中没有 NaN 或无穷大，确认数值过程稳定。"
    if stripped.startswith("assert np.all(") and "variance > 0" in stripped:
        return "检查预测方差全部为正，避免 NLL 和 coverage 使用无效方差。"
    if stripped.startswith("assert ") and ".shape" in stripped:
        return "检查输出数组的维度是否符合理论设计的形状。"
    if stripped.startswith("assert ") and ">" in stripped:
        return "检查某个量确实大于阈值，常用于确认非零 coupling、loss 下降或方差为正。"
    if stripped.startswith("assert ") and "==" in stripped:
        return "检查实际结果是否等于预期值，通常用于验证 block 数量、shape 或字符串标记。"
    if stripped.startswith("assert "):
        return "执行一个测试检查；只要这个条件不成立，该测试就会失败。"
    if stripped.startswith("return "):
        return "把当前函数计算出的结果返回给调用者。"
    if stripped.startswith("optimizer.zero_grad()"):
        return "清空上一次训练迭代留下的梯度，避免梯度累加污染本轮更新。"
    if stripped.endswith(".backward()"):
        return "根据当前 loss 反向传播，计算每个可训练参数应该如何调整。"
    if stripped.startswith("optimizer.step()"):
        return "让优化器根据刚刚计算出的梯度更新模型参数。"
    if stripped.startswith("losses.append"):
        return "把本轮 loss 数值记录下来，后面用来检查训练是否真的让 loss 下降。"
    if stripped.startswith("pred_before"):
        return "在修改 future 标签之前先做一次预测，作为 no-leakage 检查的基准。"
    if stripped.startswith("pred_after"):
        return "修改 future 标签后再预测一次；如果预测没变，说明模型没有偷看 future 标签。"
    if "Y_modified_future" in stripped and "+=" in stripped:
        return "故意把 future 标签改得非常大，用来测试预测函数是否会错误读取 future ground truth。"
    if stripped.startswith("baseline.fit_initial_task"):
        return "用初始任务/已见数据训练 baseline；此时只允许使用 seen history。"
    if stripped.startswith("baseline.predict"):
        return "让 baseline 在给定时间和空间位置上输出预测均值与预测方差。"
    if stripped.startswith("pred = baseline.predict") or stripped.startswith("pred = model.predict"):
        return "调用模型的预测函数，得到后面要检查的预测均值和方差。"
    if stripped.startswith("state = model.update") or stripped.startswith("routeB = model.update") or stripped.startswith("gp_only = model.update"):
        return "把当前 block 的观测数据送进模型，更新 posterior 状态；返回的 state 保存更新后的自然参数、均值和方差结构。"
    if stripped.startswith("online_model.update_block"):
        return "把一个 online block 送入旧 online 模型，检查递推更新是否和 batch 解一致。"
    if stripped.startswith("batch_output"):
        return "一次性用所有数据训练 batch 模型，作为 online 递推结果的 dense/batch 参考答案。"
    if stripped.startswith("online_model"):
        return "构造或使用 online 模型，用来模拟持续学习的一块块更新。"
    if stripped.startswith("dataset = load_hipposvgp_era5"):
        return "从 processed ERA5 文件夹读取一个小数据集，并堆叠成时间 x 空间的矩阵。"
    if stripped.startswith("dataset = make_synthetic_dataset"):
        return "生成一个受控 synthetic 时空数据集，用来验证公式而不是依赖真实数据。"
    if stripped.startswith("factors = make_routeb_block_factors") or stripped.startswith("factors = make_block_factors"):
        return "把一个时间 block 转换成 Route B 更新需要的因子：观测向量、线性特征、时间投影矩阵和核矩阵。"
    if stripped.startswith("temporal ="):
        return "根据输入时间点构造时间方向的核矩阵和交叉核矩阵。"
    if stripped.startswith("spatial_cov ="):
        return "根据空间坐标构造空间方向的核矩阵和交叉核矩阵。"
    if stripped.startswith("projection ="):
        return "计算时间投影和空间投影，后面可以通过 Kronecker 积组成完整时空投影。"
    if stripped.startswith("precision ="):
        return "按理论公式构造 posterior precision 矩阵，也就是高斯后验的精度矩阵。"
    if stripped.startswith("info ="):
        return "按理论公式构造 information vector，也就是 precision 形式里的右端项。"
    if stripped.startswith("mean ="):
        return "解线性方程得到 posterior mean，作为模型输出要对照的 dense reference。"
    if stripped.startswith("dense_latent_var"):
        return "按 dense precision solver 的公式显式计算 latent predictive variance。"
    if stripped.startswith("dense_var"):
        return "按 dense joint posterior 公式计算预测方差参考值。"
    if stripped.startswith("mean_field_var"):
        return "按 mean-field 假设计算预测方差，用来证明它和保留 cross covariance 的 Route B 不同。"
    if stripped.startswith("separate"):
        return "在 cross block 为零时，手动把 beta 方差项和 GP 方差项相加，作为检查参考。"
    if stripped.startswith("stats = joint_likelihood_stats"):
        return "调用 structured 统计量函数，计算 Route B 新 block likelihood 的各个自然参数块。"
    if stripped.startswith("A = dense_A_from_factors"):
        return "把时间投影 T 和空间投影 C 展开成完整 dense 时空设计矩阵 A，作为对照 reference。"
    if stripped.startswith("R_dense"):
        return "用完整 dense joint transform 显式计算旧 likelihood 转移后的 precision，作为 structured transfer 的参考答案。"
    if stripped.startswith("schur ="):
        return "用 Schur complement 和 Sylvester solve 恢复 joint posterior 的均值与协方差块。"
    if stripped.startswith("routeB_cross_cov"):
        return "按 Route B 公式计算 beta 与 u 的 posterior cross covariance。"
    if stripped.startswith("mean_field_cross_cov"):
        return "构造 mean-field 的 cross covariance；mean-field 假设下这一块被强制设为 0。"
    if stripped.startswith("Phi = np.zeros"):
        return "构造全 0 的线性特征，用来测试没有线性均值或没有 beta-u coupling 时模型是否正确退化。"
    if stripped.startswith("task = load_processed_era5"):
        return "读取 processed ERA5 task，并检查它是否正确排序、对齐或拼接多个任务。"
    if stripped.startswith("task_dirs"):
        return "发现指定的 ERA5 task 文件夹，确认 loader 能找到正确数据目录。"
    if stripped.startswith("first = select_spatial_inducing_points"):
        return "用简单 first-N 方式选择空间诱导点，作为对比基线。"
    if stripped.startswith("fps = select_spatial_inducing_points"):
        return "用 farthest-point sampling 选择空间诱导点，检查它是否覆盖空间边界。"
    if "=" in stripped:
        left, right = stripped.split("=", 1)
        return f"计算右侧 `{right.strip()}`，并把结果保存到变量 `{left.strip()}`，供后续检查使用。"
    return "执行这一行代码对应的函数调用或检查；它是该验证步骤中的一个中间操作。"


def line_by_line_explanations(code: str) -> list[str]:
    lines = []
    for i, line in enumerate(code.splitlines(), start=1):
        if not line.strip():
            continue
        lines.append(f"{i}. `{line}`：{explain_code_line(line)}")
    return lines


def write_markdown() -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Route B 40 个数学理论与基线管线验证实验：公式-代码-逻辑对照",
        "",
        "本文档面向代码基础较弱的读者，逐个解释当前项目中 40 个验证测试。前 34 个是 Route B/Stage-1 的核心数学与实现验证；后 6 个是新增 ERA5 loader 与 baseline 管线验证。每个块都说明：验证哪个数学公式或数据契约、对应代码片段、以及为什么这个测试能证明实现没有偏离理论或实验协议。",
        "",
        "验证命令：",
        "",
        "```bash",
        "uv run --no-sync pytest -q",
        "uv run --no-sync python scripts/verify_joint_ssgp_kron_derivations.py",
        "```",
        "",
        "结果：`40 passed, 1 warning`。warning 来自本机 CUDA driver 版本提示，不影响 CPU 数值验证。",
        "",
        "## Python 语法速查",
        "",
        "下面 40 个测试都只用到少量常见 Python/Numpy/PyTorch 语法。先掌握这些符号，后面每个代码块会更容易读：",
        "",
        "- `=`：赋值，把右边的结果存到左边变量。",
        "- `assert 条件`：测试断言，条件不成立时测试失败。",
        "- `for ... in ...:`：循环，冒号后缩进的代码会重复执行。",
        "- `if ...:`：条件判断，只有条件成立才执行缩进块。",
        "- `A @ B`：矩阵乘法。",
        "- `A.T`：矩阵转置。",
        "- `x[:3]`：切片，取前 3 个元素；`x[3:5]` 取第 3 到第 4 个元素。",
        "- `dict['key']`：从字典里按名字取值。",
        "- `.shape`：数组形状，例如 `(T, S)`。",
        "- `np.allclose(a,b)` / `torch.allclose(a,b)`：判断两个浮点数组是否近似相等。",
        "- `np.isfinite(x)`：检查结果不是 NaN 或无穷大。",
        "- `...`：文档中的省略号，表示省略了不影响理解的完整参数。",
        "",
        "## 总览表",
        "",
        "| # | 测试 | 分组 | 主要验证对象 | 结果 |",
        "|---:|---|---|---|---|",
    ]
    for item in ITEMS:
        lines.append(
            f"| {item.idx} | `{item.test}` | {item.group} | {markdown_escape_pipe(item.logic)} | {item.result} |"
        )
    lines += ["", "## 逐项讲解", ""]
    for item in ITEMS:
        lines += [
            f"### {item.idx}. `{item.test}`",
            "",
            f"- 文件：`{item.file}`",
            f"- 分组：{item.group}",
            f"- 结果：`{item.result}`",
            "",
            "**数学公式片段**",
            "",
            item.formula,
            "",
            "**代码片段**",
            "",
            "```python",
            item.code,
            "```",
            "",
            "**代码逐行实现逻辑翻译**",
            "",
        ]
        for note in line_by_line_explanations(item.code):
            lines.append(f"- {note}")
        lines += ["", "**涉及的基础语法提示**", ""]
        for note in syntax_notes_for(item.code):
            lines.append(f"- {note}")
        lines += [
            "",
            "**验证逻辑**",
            "",
            item.logic,
            "",
        ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def tex_escape_test_name(text: str) -> str:
    return tex_escape(text).replace(r"\_", r"\_\allowbreak{}")


def make_report_section() -> str:
    rows = []
    for item in ITEMS:
        rows.append(
            f"{item.idx} & {tex_escape_test_name(item.test)} & {item.formula} & {item.result} \\\\"
        )
    table = "\n".join(rows)
    return rf"""
\section{{40 mathematical and pipeline verification tests}}
The verification suite now contains 40 tests: 14 Stage-1 Kronecker STGP tests, 7 derivation tests, 3 model/API sanity tests, 10 Route B tests, and 6 ERA5 loader/baseline pipeline tests. I reran the full suite with:
\begin{{verbatim}}
uv run --no-sync pytest -q
\end{{verbatim}}
The result is \textbf{{40 passed, 1 warning}}. The warning is a local CUDA driver initialization warning from PyTorch and does not affect the CPU numerical checks. The derivation script was also rerun and reported \texttt{{all\_passed=true}} and \texttt{{routeB\_all\_passed=true}}.

\paragraph{{What is verified.}} The first 34 tests verify the Kronecker projection shapes, dense batch posterior identity, online-vs-batch natural parameter equivalence, changing-basis transfer identities, Schur-complement posterior recovery, beta-u cross covariance, dense predictive variance, GP-only reduction, zero-cross-feature reduction, and non-unit kernel-amplitude predictive variance. The last 6 tests verify the ERA5 loader shape contract, conversion to Route B block factors, deterministic baseline no-leakage/finite-variance behavior, and GPyTorch likelihood predictive variance. A detailed formula-code walkthrough is available under \texttt{{results/routeB\_40\_math\_verification\_walkthrough/}}.

\begingroup
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{longtable}}{{r >{{\raggedright\arraybackslash}}p{{0.36\linewidth}} >{{\raggedright\arraybackslash}}p{{0.50\linewidth}} l}}
\toprule
\# & Test & Formula or contract being checked & Result \\
\midrule
{table}
\bottomrule
\end{{longtable}}
\endgroup
"""


def inject_report_section() -> None:
    if not REPORT_TEX.exists():
        return
    text = REPORT_TEX.read_text(encoding="utf-8")
    if r"\usepackage{amsmath}" not in text:
        text = text.replace(r"\usepackage{booktabs}", "\\usepackage{booktabs}\n\\usepackage{amsmath}")
    if r"\usepackage{amssymb}" not in text:
        text = text.replace(r"\usepackage{amsmath}", "\\usepackage{amsmath}\n\\usepackage{amssymb}")
    start = text.find(r"\section{40 mathematical and pipeline verification tests}")
    old_start = text.find(r"\section{34 mathematical verification tests}")
    marker = r"\section{Original short synthetic sanity result}"
    section = make_report_section()
    if start < 0 and old_start >= 0:
        start = old_start
    if start >= 0:
        end = text.find(marker, start)
        if end < 0:
            raise RuntimeError("Could not find insertion end marker after existing verification section")
        text = text[:start] + section + "\n" + text[end:]
    else:
        insert = text.find(marker)
        if insert < 0:
            raise RuntimeError("Could not find insertion marker in report tex")
        text = text[:insert] + section + "\n" + text[insert:]
    REPORT_TEX.write_text(text, encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_markdown()
    inject_report_section()
    print(f"Wrote {DOC}")
    print(f"Updated {REPORT_TEX}")


if __name__ == "__main__":
    main()
