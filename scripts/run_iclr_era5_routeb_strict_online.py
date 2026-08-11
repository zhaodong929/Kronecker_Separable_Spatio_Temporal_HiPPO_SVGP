#!/usr/bin/env python3
"""Strict-online Route B on the shared Task-1 calibration ERA5 protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hipposvgp_era5_routeb import (
    augment_dataset_phi,
    selected_locations_from_dataset,
    vectorized_predict_with_C,
)
from scripts.run_routeb_batch_empirical_bayes import object_array_bytes
from scripts.run_routeb_online_parity_ladder import spatial_projection, temporal_factors
from stvgp_kronecker.benchmark_runtime import (
    SynchronizedTimer,
    host_snapshot,
    peak_rss_mib,
    resolve_torch_runtime,
)
from stvgp_kronecker.temporal_kernel_config import (
    load_spectral_mixture_config,
    temporal_kernel_metadata,
)
from scripts.era5_ncu_ranges import pop_range, profile_this_index, push_range
from stvgp_kronecker.data.hipposvgp_era5 import load_hipposvgp_era5
from stvgp_kronecker.joint_ssgp_kron.kron_utils import solve_spd, vec_f
from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.torch_backend import (
    TorchJointSSGPKronHiPPOSVGP,
    solve_spd as torch_solve_spd,
)
from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    BlockFactors,
    make_analytic_temporal_builder,
    temporal_spec_for_block,
)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(y_true, mean, variance):
    y = np.asarray(y_true, dtype=float).reshape(-1)
    mu = np.asarray(mean, dtype=float).reshape(-1)
    var = np.maximum(np.asarray(variance, dtype=float).reshape(-1), 1e-10)
    std = np.sqrt(var)
    result = {
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "nll": float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var))),
        "mean_predictive_std": float(np.mean(std)),
        "mean_interval_width90": float(np.mean(2.0 * 1.6448536269514722 * std)),
    }
    for level, z_score in (
        (50, 0.6744897501960817),
        (80, 1.2815515655446004),
        (90, 1.6448536269514722),
        (95, 1.959963984540054),
    ):
        result[f"coverage{level}"] = float(np.mean(np.abs(y - mu) <= z_score * std))
    return result


def matern32_1d(x1, x2, lengthscale, variance):
    distance = np.abs(np.asarray(x1)[:, None] - np.asarray(x2)[None, :]) / lengthscale
    scaled = np.sqrt(3.0) * distance
    return variance * (1.0 + scaled) * np.exp(-scaled)


def matern32_1d_torch(x1, x2, lengthscale, variance):
    distance = torch.abs(x1[:, None] - x2[None, :]) / lengthscale
    scaled = np.sqrt(3.0) * distance
    return variance * (1.0 + scaled) * torch.exp(-scaled)


def temporal_factors_torch(
    *,
    builder,
    times,
    query,
    basis,
    old_basis,
    old_temporal_basis,
    device,
    dtype,
):
    """CUDA equivalent of cumulative-changing ``temporal_factors``."""

    spec = temporal_spec_for_block(times, basis, moving=True)
    old_spec = (
        None
        if old_basis is None or old_temporal_basis is not None
        else temporal_spec_for_block(times, old_basis, moving=True)
    )
    with torch.no_grad():
        query_times = torch.as_tensor(times[query], device=device, dtype=dtype)
        kfu, kt, k_on_t, new_temporal_basis = (
            builder.compute_block_covariances_with_basis(
                query_times,
                spec,
                old_spec,
                old_basis=old_temporal_basis,
            )
        )
        kt = builder.add_jitter(kt)
        t_mat = torch_solve_spd(kt, kfu.transpose(0, 1), jitter=1e-12).transpose(0, 1)
    return t_mat, kt, k_on_t, new_temporal_basis


class TaskPhiCache:
    def __init__(self, root, stream_y, xlag_length):
        self.root = Path(root)
        calibration = load_hipposvgp_era5(
            self.root, tasks=("task_1",), variable_index=0, split="all"
        )
        self.selected_locations = selected_locations_from_dataset(calibration)
        self.stream_y = np.asarray(stream_y)
        self.xlag_length = int(xlag_length)
        self.task_index = None
        self.phi = None
        self.loading_seconds = 0.0

    def block(self, block):
        task_index = 2 + int(block.start) // 186
        task_start = (task_index - 2) * 186
        local = slice(block.start - task_start, block.stop - task_start)
        if local.start < 0 or local.stop > 186:
            raise ValueError(f"Block {block} crosses an ERA5 task boundary")
        if self.task_index != task_index:
            started = time.perf_counter()
            raw = load_hipposvgp_era5(
                self.root,
                tasks=(f"task_{task_index}",),
                variable_index=0,
                split="all",
                selected_locations=self.selected_locations,
            )
            augmented = augment_dataset_phi(
                raw, phi_mode="medium_era5_xlag", xlag_length=self.xlag_length
            )
            self.phi = np.asarray(augmented.Phi, dtype=np.float64).reshape(
                raw.Y.shape[0], raw.Y.shape[1], -1
            )
            expected = self.stream_y[task_start : task_start + raw.Y.shape[0]]
            np.testing.assert_allclose(raw.Y, expected, atol=2e-6, rtol=0.0)
            self.task_index = task_index
            self.loading_seconds += time.perf_counter() - started
        return self.phi[local], task_index


class ProtocolPhiCache:
    """Read generic causal features embedded in a materialised protocol."""

    def __init__(self, phi, stream_shape):
        self.phi = np.asarray(phi, dtype=np.float64)
        if self.phi.ndim != 3 or self.phi.shape[:2] != tuple(stream_shape):
            raise ValueError(
                "stream_phi must have shape (time, space, features): "
                f"target={tuple(stream_shape)}, phi={self.phi.shape}"
            )
        self.loading_seconds = 0.0

    def block(self, block):
        return self.phi[block], "stream"


def make_factors(y_block, phi_block, spatial_indices, t_mat, kt, k_on_t, block, backend):
    indices = np.asarray(spatial_indices, dtype=int)
    y_matrix = np.asarray(y_block[:, indices].T, dtype=float)
    phi = np.asarray(phi_block[:, indices, :], dtype=float).reshape(-1, phi_block.shape[-1])
    return BlockFactors(
        y_vec=vec_f(y_matrix),
        Phi=phi,
        Y=y_matrix,
        T=t_mat,
        Kt=kt,
        K_on_t=k_on_t,
        block_slice=block,
        inducing_times=np.empty(0, dtype=float),
        temporal_backend=backend,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--theta-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blockwise-output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--representation", choices=["analytic_hippo_rff", "inducing_points"], required=True)
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument(
        "--temporal-kernel",
        choices=["matern32", "spectral_mixture"],
        default="matern32",
    )
    parser.add_argument(
        "--spectral-mixture-json",
        type=Path,
        help="The same fixed mixture configuration used during calibration.",
    )
    parser.add_argument("--prediction-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--include-conditional-residual-variance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include k_xx - K_xu K_uu^-1 K_ux in the predictive variance. "
            "Use --no-include-conditional-residual-variance only to reproduce "
            "the historical finite-DTC protocol."
        ),
    )
    parser.add_argument("--beta-prior-variance", type=float, default=1000.0)
    parser.add_argument(
        "--use-protocol-beta-prior",
        action="store_true",
        help="Use a diagonal beta prior variance stored under beta_prior_variance in the protocol.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument(
        "--delayed-observations",
        action="store_true",
        help="Absorb the previous block's held-out observations before the current visible block.",
    )
    parser.add_argument(
        "--delayed-observation-blocks",
        type=int,
        default=0,
        help=(
            "Number of complete online blocks before a held-out label arrives. "
            "Zero preserves the legacy setting; --delayed-observations is a "
            "backward-compatible synonym for one block."
        ),
    )
    parser.add_argument(
        "--task1-posterior-init",
        action="store_true",
        help=(
            "Absorb all Task-1 visible locations before the online stream and continue "
            "on the same temporal grid."
        ),
    )
    parser.add_argument(
        "--feature-projection-npz",
        type=Path,
        help="Optional shared orthonormal feature basis stored under the 'basis' key.",
    )
    parser.add_argument(
        "--spatial-projection-npz",
        type=Path,
        help="Optional precomputed spatial Ks/C matrices for the selected kernel mode.",
    )
    parser.add_argument(
        "--spatial-kernel-mode",
        choices=("geo", "graph", "geo_graph"),
        default="geo",
    )
    parser.add_argument(
        "--solver-backend",
        choices=["auto", "numpy", "torch"],
        default="auto",
        help="auto uses PyTorch on CUDA and the NumPy/SciPy reference on CPU.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument(
        "--temporal-factor-device",
        choices=["auto", "cpu", "solver"],
        default="auto",
        help=(
            "Device for temporal kernels/HiPPO-RFF/Bessel factors. auto uses "
            "CPU for analytic HiPPO on a CUDA solver and the solver device "
            "otherwise."
        ),
    )
    args = parser.parse_args()

    if args.delayed_observation_blocks < 0:
        raise ValueError("--delayed-observation-blocks must be non-negative")
    if args.delayed_observations and args.delayed_observation_blocks not in (0, 1):
        raise ValueError(
            "--delayed-observations conflicts with --delayed-observation-blocks > 1"
        )
    delayed_observation_blocks = (
        args.delayed_observation_blocks
        if args.delayed_observation_blocks > 0
        else int(args.delayed_observations)
    )

    spectral_mixture = load_spectral_mixture_config(args.spectral_mixture_json)
    if args.temporal_kernel == "spectral_mixture":
        if args.representation != "analytic_hippo_rff":
            raise ValueError("The fixed spectral-mixture screen is only supported by HiPPO-RFF")
        if spectral_mixture is None:
            raise ValueError("--spectral-mixture-json is required for spectral_mixture")
    elif spectral_mixture is not None:
        raise ValueError("--spectral-mixture-json requires --temporal-kernel spectral_mixture")

    process_started = time.perf_counter()
    if args.solver_backend == "auto":
        requested_device = args.device.lower()
        wants_cuda = requested_device.startswith("cuda") or (
            requested_device == "auto" and torch.cuda.is_available()
        )
        solver_backend = "torch" if wants_cuda else "numpy"
    else:
        solver_backend = args.solver_backend
    if solver_backend == "numpy":
        if args.device.lower().startswith("cuda"):
            raise ValueError("--solver-backend numpy only supports --device cpu/auto")
        if args.dtype != "float64":
            raise ValueError("The NumPy/SciPy reference backend is fixed to float64")
        runtime = None
        synchronize = lambda: None
        solver_device = "cpu"
        solver_dtype = "float64"
    else:
        runtime = resolve_torch_runtime(args.device, args.dtype)
        synchronize = runtime.synchronize
        solver_device = str(runtime.device)
        solver_dtype = args.dtype
    if args.temporal_factor_device == "cpu":
        temporal_factor_device = torch.device("cpu")
    elif args.temporal_factor_device == "solver":
        temporal_factor_device = (
            runtime.device if runtime is not None else torch.device("cpu")
        )
    elif (
        args.representation == "analytic_hippo_rff"
        and runtime is not None
        and runtime.uses_cuda
    ):
        temporal_factor_device = torch.device("cpu")
    else:
        temporal_factor_device = (
            runtime.device if runtime is not None else torch.device("cpu")
        )
    temporal_factor_dtype = (
        runtime.dtype if runtime is not None else torch.float64
    )
    np.random.seed(args.seed)
    torch.manual_seed(0)
    arrays = np.load(args.protocol_npz)
    metadata = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    theta_payload = json.loads(args.theta_json.read_text(encoding="utf-8"))
    theta = theta_payload["learned_theta"]
    protocol_phi = np.asarray(arrays["stream_phi"]) if "stream_phi" in arrays else None
    original_beta_dimension = 133 if protocol_phi is None else int(protocol_phi.shape[-1])
    feature_projection = None
    if args.feature_projection_npz is not None:
        with np.load(args.feature_projection_npz) as projection_payload:
            feature_projection = np.asarray(
                projection_payload["basis"], dtype=np.float64
            )
        if (
            feature_projection.ndim != 2
            or feature_projection.shape[0] != original_beta_dimension
        ):
            raise ValueError(
                f"Feature projection must have shape ({original_beta_dimension}, rank), got "
                f"{feature_projection.shape}"
            )
    beta_dimension = (
        original_beta_dimension
        if feature_projection is None
        else feature_projection.shape[1]
    )
    if args.use_protocol_beta_prior:
        if "beta_prior_variance" not in arrays:
            raise ValueError("--use-protocol-beta-prior requires beta_prior_variance in the protocol")
        protocol_beta_variance = np.asarray(arrays["beta_prior_variance"], dtype=float)
        if protocol_beta_variance.shape != (original_beta_dimension,):
            raise ValueError(
                "beta_prior_variance must have shape "
                f"({original_beta_dimension},), got {protocol_beta_variance.shape}"
            )
        if not np.isfinite(protocol_beta_variance).all() or (protocol_beta_variance <= 0.0).any():
            raise ValueError("beta_prior_variance must be finite and positive")
        beta_prior_cov = np.diag(protocol_beta_variance)
        if feature_projection is not None:
            beta_prior_cov = feature_projection.T @ beta_prior_cov @ feature_projection
    else:
        beta_prior_cov = args.beta_prior_variance * np.eye(beta_dimension)
    stream_times = np.asarray(arrays["stream_times"], dtype=float)
    y = np.asarray(arrays["stream_y"], dtype=float)
    coordinates = np.asarray(arrays["coordinates"], dtype=float)
    train_indices = np.asarray(arrays["train_indices"], dtype=int)
    test_indices = np.asarray(arrays["test_indices"], dtype=int)
    blocks = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(arrays["block_start"], arrays["block_stop"])
    )
    if args.max_blocks > 0:
        blocks = blocks[: args.max_blocks]
    calibration_y = None
    calibration_phi = None
    temporal_offset = 0
    if args.task1_posterior_init:
        if protocol_phi is None or "calibration_phi" not in arrays:
            raise ValueError("--task1-posterior-init requires protocol calibration_phi")
        calibration_y = np.asarray(arrays["calibration_y"], dtype=float)
        calibration_phi = np.asarray(arrays["calibration_phi"], dtype=float)
        calibration_times = np.asarray(arrays["calibration_times"], dtype=float)
        if calibration_y.shape != calibration_phi.shape[:2]:
            raise ValueError("calibration_y/calibration_phi shape mismatch")
        if calibration_times.shape != (calibration_y.shape[0],):
            raise ValueError("calibration_times length must match calibration_y")
        if calibration_times.size < 2:
            raise ValueError("Task-1 posterior initialization requires at least two time points")
        time_step = float(np.median(np.diff(calibration_times)))
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("Task-1 calibration times must be strictly increasing")
        temporal_times = np.concatenate(
            [
                calibration_times,
                calibration_times[-1] + time_step * np.arange(1, y.shape[0] + 1),
            ]
        )
        temporal_offset = calibration_y.shape[0]
    else:
        temporal_times = stream_times
    if args.task1_posterior_init and solver_backend != "torch":
        raise ValueError("Task-1 posterior initialization currently requires --solver-backend torch")
    spatial_setup_started = time.perf_counter()
    if args.spatial_projection_npz is None or args.spatial_kernel_mode == "geo":
        spatial_inducing = np.asarray(arrays[f"inducing_coords_ms{args.ms}"], dtype=float)
        ks, c_all = spatial_projection(coordinates, spatial_inducing, theta["ell_s"])
    else:
        with np.load(args.spatial_projection_npz) as spatial_payload:
            ks = np.asarray(
                spatial_payload[f"ks_{args.spatial_kernel_mode}_ms{args.ms}"],
                dtype=float,
            )
            c_all = np.asarray(
                spatial_payload[f"c_{args.spatial_kernel_mode}_ms{args.ms}"],
                dtype=float,
            )
        if ks.shape != (args.ms, args.ms) or c_all.shape != (coordinates.shape[0], args.ms):
            raise ValueError("Precomputed spatial projection has incompatible shape")
    c_train = c_all[train_indices]
    c_test = c_all[test_indices]
    spatial_setup_seconds = time.perf_counter() - spatial_setup_started
    sigma2 = float(theta["noise_std"]) ** 2
    solver_setup_started = time.perf_counter()
    if solver_backend == "torch":
        assert runtime is not None
        model = TorchJointSSGPKronHiPPOSVGP(
            Ks=ks,
            C=c_train,
            sigma2=sigma2,
            beta_prior_mean=np.zeros(beta_dimension),
            beta_prior_cov=beta_prior_cov,
            prior_point_variance=float(theta["kernel_variance"]),
            device=runtime.device,
            dtype=runtime.dtype,
        )
        c_test_backend = torch.as_tensor(
            c_test, device=runtime.device, dtype=runtime.dtype
        )
        c_train_backend = torch.as_tensor(
            c_train, device=runtime.device, dtype=runtime.dtype
        )
    else:
        model = JointSSGPKronHiPPOSVGP(
            Ks=ks,
            C=c_train,
            sigma2=sigma2,
            beta_prior_mean=np.zeros(beta_dimension),
            beta_prior_cov=beta_prior_cov,
            prior_point_variance=float(theta["kernel_variance"]),
        )
        c_test_backend = c_test
        c_train_backend = c_train
    synchronize()
    solver_setup_seconds = time.perf_counter() - solver_setup_started
    temporal_setup_started = time.perf_counter()
    builder = None
    ordinary_t = None
    ordinary_kt = None
    if args.representation == "analytic_hippo_rff":
        builder = make_analytic_temporal_builder(
            mt=args.mt,
            lengthscale=float(theta["ell_t"]),
            variance=float(theta["kernel_variance"]),
            rff_sample_size=args.rff_sample_size,
            seed=0,
            jitter=1e-7,
            kernel_type=args.temporal_kernel,
            spectral_mixture_weights=spectral_mixture["weights"] if spectral_mixture else None,
            spectral_mixture_means=spectral_mixture["means"] if spectral_mixture else None,
            spectral_mixture_scales=spectral_mixture["scales"] if spectral_mixture else None,
        )
        builder = builder.to(
            device=temporal_factor_device,
            dtype=temporal_factor_dtype,
        )
    else:
        if solver_backend == "torch":
            assert runtime is not None
            z_t = torch.linspace(
                float(temporal_times.min()),
                float(temporal_times.max()),
                args.mt,
                device=temporal_factor_device,
                dtype=temporal_factor_dtype,
            )
            times_tensor = torch.as_tensor(
                temporal_times,
                device=temporal_factor_device,
                dtype=temporal_factor_dtype,
            )
            ordinary_kt = matern32_1d_torch(
                z_t,
                z_t,
                float(theta["ell_t"]),
                float(theta["kernel_variance"]),
            )
            ordinary_kt = (
                0.5 * (ordinary_kt + ordinary_kt.transpose(0, 1))
                + 1e-7
                * torch.eye(
                    args.mt,
                    device=temporal_factor_device,
                    dtype=temporal_factor_dtype,
                )
            )
            kfu = matern32_1d_torch(
                times_tensor,
                z_t,
                float(theta["ell_t"]),
                float(theta["kernel_variance"]),
            )
            ordinary_t = torch_solve_spd(
                ordinary_kt, kfu.transpose(0, 1), jitter=1e-12
            ).transpose(0, 1)
            ordinary_t = ordinary_t.to(device=runtime.device, dtype=runtime.dtype)
            ordinary_kt = ordinary_kt.to(device=runtime.device, dtype=runtime.dtype)
        else:
            z_t = np.linspace(
                float(temporal_times.min()), float(temporal_times.max()), args.mt
            )
            ordinary_kt = matern32_1d(
                z_t, z_t, float(theta["ell_t"]), float(theta["kernel_variance"])
            )
            ordinary_kt = 0.5 * (ordinary_kt + ordinary_kt.T) + 1e-7 * np.eye(args.mt)
            kfu = matern32_1d(
                temporal_times,
                z_t,
                float(theta["ell_t"]),
                float(theta["kernel_variance"]),
            )
            ordinary_t = solve_spd(ordinary_kt, kfu.T, jitter=1e-12).T
    synchronize()
    temporal_setup_seconds = time.perf_counter() - temporal_setup_started

    if protocol_phi is not None:
        phi_cache = ProtocolPhiCache(protocol_phi, y.shape)
    else:
        phi_cache = TaskPhiCache(
            metadata["root"] if args.data_root is None else args.data_root,
            y,
            metadata["xlag"]["length"],
        )
    state = None
    previous_basis = None
    previous_temporal_basis = None
    rows = []
    all_true = []
    all_mean = []
    all_var = []
    prediction_grid = np.empty((stream_times.size, test_indices.size), dtype=np.float64)
    variance_grid = np.empty_like(prediction_grid)
    total_update_seconds = 0.0
    total_prediction_seconds = 0.0
    total_factor_seconds = 0.0
    total_feature_seconds = 0.0
    pending_test_contexts = []
    delayed_observation_rows = 0
    task1_initialization_rows = 0
    task1_initialization_seconds = 0.0
    task1_initialization_summary = None
    if runtime is not None:
        runtime.reset_peak_memory()

    if args.task1_posterior_init:
        assert calibration_y is not None and calibration_phi is not None
        assert runtime is not None
        if feature_projection is not None:
            calibration_phi = np.einsum(
                "tsp,pr->tsr",
                calibration_phi,
                feature_projection,
                optimize=True,
            )
        task1_slice = slice(0, temporal_offset)
        with SynchronizedTimer(synchronize) as initialization_timer:
            if args.representation == "analytic_hippo_rff":
                assert builder is not None
                t_mat, kt, _, previous_temporal_basis = temporal_factors_torch(
                    builder=builder,
                    times=temporal_times,
                    query=task1_slice,
                    basis=task1_slice,
                    old_basis=None,
                    old_temporal_basis=None,
                    device=temporal_factor_device,
                    dtype=temporal_factor_dtype,
                )
                t_mat = t_mat.to(device=runtime.device, dtype=runtime.dtype)
                kt = kt.to(device=runtime.device, dtype=runtime.dtype)
            else:
                t_mat = ordinary_t[task1_slice]
                kt = ordinary_kt
                previous_temporal_basis = None
            task1_factors = make_factors(
                calibration_y,
                calibration_phi,
                train_indices,
                t_mat,
                kt,
                None,
                task1_slice,
                args.representation,
            )
            state = model.update_block_structured_joint_ssgp_transfer(
                y_vec=task1_factors.y_vec,
                Phi=task1_factors.Phi,
                T_n=task1_factors.T,
                Kt_new=task1_factors.Kt,
                state=None,
                K_on_t=None,
                C_observed=c_train_backend,
            )
        task1_initialization_seconds = initialization_timer.elapsed
        task1_initialization_rows = int(task1_factors.y_vec.size)
        previous_basis = task1_slice
        task1_initialization_summary = {
            "beta_mean_norm": float(torch.linalg.vector_norm(state.beta_mean).detach().cpu()),
            "beta_mean_max_abs": float(state.beta_mean.abs().max().detach().cpu()),
            "u_mean_norm": float(torch.linalg.vector_norm(state.M_u).detach().cpu()),
            "u_mean_max_abs": float(state.M_u.abs().max().detach().cpu()),
            "beta_cov_max_abs": float(state.beta_cov.abs().max().detach().cpu()),
        }
        posterior_values = torch.cat(
            [state.beta_mean.reshape(-1), state.beta_cov.reshape(-1), state.M_u.reshape(-1)]
        )
        if not torch.isfinite(posterior_values).all():
            raise ValueError("Task-1 posterior initialization produced non-finite state values")

    for block_id, block in enumerate(blocks):
        profile_range = profile_this_index(block_id, len(blocks))
        profile_open = push_range("era5_online_block", profile_range)
        feature_started = time.perf_counter()
        phi_block, task_index = phi_cache.block(block)
        task_label = (
            f"task_{task_index}" if isinstance(task_index, (int, np.integer)) else str(task_index)
        )
        if feature_projection is not None:
            phi_block = np.einsum(
                "tsp,pr->tsr",
                np.asarray(phi_block, dtype=np.float64),
                feature_projection,
                optimize=True,
            )
        feature_seconds = time.perf_counter() - feature_started
        y_block = y[block]
        temporal_block = slice(
            temporal_offset + block.start,
            temporal_offset + block.stop,
        )
        with SynchronizedTimer(synchronize) as factor_timer:
            if (
                delayed_observation_blocks > 0
                and len(pending_test_contexts) >= delayed_observation_blocks
            ):
                if solver_backend != "torch":
                    raise ValueError("Delayed observations currently require --solver-backend torch")
                pending = pending_test_contexts.pop(0)
                delayed_block = pending["block"]
                delayed_temporal_block = slice(
                    temporal_offset + delayed_block.start,
                    temporal_offset + delayed_block.stop,
                )
                if args.representation == "analytic_hippo_rff":
                    assert builder is not None and previous_basis is not None
                    delayed_t, delayed_kt, _, _ = temporal_factors_torch(
                        builder=builder,
                        times=temporal_times,
                        query=delayed_temporal_block,
                        basis=previous_basis,
                        old_basis=None,
                        old_temporal_basis=None,
                        device=temporal_factor_device,
                        dtype=temporal_factor_dtype,
                    )
                    assert runtime is not None
                    delayed_t = delayed_t.to(device=runtime.device, dtype=runtime.dtype)
                    delayed_kt = delayed_kt.to(device=runtime.device, dtype=runtime.dtype)
                else:
                    delayed_t = ordinary_t[delayed_temporal_block]
                    delayed_kt = ordinary_kt
                delayed_factors = make_factors(
                    pending["y_block"],
                    pending["phi_block"],
                    test_indices,
                    delayed_t,
                    delayed_kt,
                    None,
                    delayed_block,
                    args.representation,
                )
                state = model.update_block_structured_joint_ssgp_transfer(
                    y_vec=delayed_factors.y_vec,
                    Phi=delayed_factors.Phi,
                    T_n=delayed_factors.T,
                    Kt_new=delayed_factors.Kt,
                    state=state,
                    K_on_t=None,
                    C_observed=c_test_backend,
                )
                delayed_observation_rows += int(delayed_factors.y_vec.size)
            if args.representation == "analytic_hippo_rff":
                basis = slice(0, temporal_block.stop)
                if solver_backend == "torch":
                    assert runtime is not None
                    t_mat, kt, k_on_t, new_temporal_basis = temporal_factors_torch(
                        builder=builder,
                        times=temporal_times,
                        query=temporal_block,
                        basis=basis,
                        old_basis=previous_basis,
                        old_temporal_basis=previous_temporal_basis,
                        device=temporal_factor_device,
                        dtype=temporal_factor_dtype,
                    )
                    t_mat = t_mat.to(device=runtime.device, dtype=runtime.dtype)
                    kt = kt.to(device=runtime.device, dtype=runtime.dtype)
                    k_on_t = (
                        None
                        if k_on_t is None
                        else k_on_t.to(device=runtime.device, dtype=runtime.dtype)
                    )
                else:
                    t_mat, kt, k_on_t, new_temporal_basis = temporal_factors(
                        builder=builder,
                        times=temporal_times,
                        query=temporal_block,
                        basis=basis,
                        old_basis=previous_basis,
                        basis_mode="cumulative_changing",
                        old_temporal_basis=previous_temporal_basis,
                        return_temporal_basis=True,
                    )
                previous_basis = basis
                previous_temporal_basis = new_temporal_basis
            else:
                t_mat = ordinary_t[temporal_block]
                kt = ordinary_kt
                k_on_t = None
            train_factors = make_factors(
                y_block,
                phi_block,
                train_indices,
                t_mat,
                kt,
                k_on_t,
                block,
                args.representation,
            )
            test_factors = make_factors(
                y_block,
                phi_block,
                test_indices,
                t_mat,
                kt,
                None,
                block,
                args.representation,
            )
        factor_seconds = factor_timer.elapsed
        with SynchronizedTimer(synchronize) as update_timer:
            state = model.update_block_structured_joint_ssgp_transfer(
                y_vec=train_factors.y_vec,
                Phi=train_factors.Phi,
                T_n=train_factors.T,
                Kt_new=train_factors.Kt,
                state=state,
                K_on_t=train_factors.K_on_t,
            )
        update_seconds = update_timer.elapsed
        with SynchronizedTimer(synchronize) as prediction_timer:
            if solver_backend == "torch":
                mean, variance, _ = model.predict_with_C(
                    state=state,
                    T_eval=test_factors.T,
                    Phi=test_factors.Phi,
                    C_eval=c_test_backend,
                    chunk_size=args.prediction_chunk_size,
                    include_conditional_residual_variance=(
                        args.include_conditional_residual_variance
                    ),
                    validate_conditional_residual_variance=(
                        args.include_conditional_residual_variance
                    ),
                )
            else:
                mean, variance, _ = vectorized_predict_with_C(
                    model,
                    state,
                    test_factors,
                    c_test_backend,
                    prediction_mode="streaming_sylvester",
                    chunk_size=args.prediction_chunk_size,
                    include_conditional_residual_variance=(
                        args.include_conditional_residual_variance
                    ),
                )
        prediction_seconds = prediction_timer.elapsed
        pop_range(profile_open)
        block_metrics = metrics(test_factors.y_vec, mean, variance)
        mean_matrix = np.asarray(mean).reshape(block.stop - block.start, test_indices.size)
        var_matrix = np.asarray(variance).reshape(block.stop - block.start, test_indices.size)
        prediction_grid[block] = mean_matrix
        variance_grid[block] = var_matrix
        all_true.append(test_factors.y_vec)
        all_mean.append(mean)
        all_var.append(variance)
        total_update_seconds += update_seconds
        total_prediction_seconds += prediction_seconds
        total_factor_seconds += factor_seconds
        total_feature_seconds += feature_seconds
        persistent_state_bytes = object_array_bytes(
            {
                "model": model,
                "state": state,
                "builder": builder,
                "ordinary_t": ordinary_t,
                "ordinary_kt": ordinary_kt,
                "c_test": c_test_backend,
            }
        )
        row = {
            "block_id": block_id,
            "task": task_label,
            "block_start": block.start,
            "block_stop": block.stop,
            "hours": block.stop - block.start,
            "feature_loading_seconds": feature_seconds,
            "factor_preparation_seconds": factor_seconds,
            "update_seconds": update_seconds,
            "prediction_seconds": prediction_seconds,
            "persistent_state_bytes": persistent_state_bytes,
            "solver_backend": solver_backend,
            "solver_device": solver_device,
            "solver_dtype": solver_dtype,
            "temporal_factor_device": str(temporal_factor_device),
            **block_metrics,
        }
        if runtime is not None and runtime.uses_cuda:
            row["peak_cuda_allocated_mib"] = (
                torch.cuda.max_memory_allocated(runtime.device) / 1024.0**2
            )
        rows.append(row)
        if delayed_observation_blocks > 0:
            pending_test_contexts.append(
                {"block": block, "y_block": y_block, "phi_block": phi_block}
            )
        print(json.dumps(row), flush=True)

    overall = metrics(np.concatenate(all_true), np.concatenate(all_mean), np.concatenate(all_var))
    final_block = {
        key: rows[-1][key]
        for key in (
            "rmse",
            "nll",
            "coverage50",
            "coverage80",
            "coverage90",
            "coverage95",
            "mean_predictive_std",
            "mean_interval_width90",
        )
    }
    persistent_bytes = rows[-1]["persistent_state_bytes"]
    payload = {
        "implementation": "Route B strict-online structured joint beta-GP",
        "protocol": (
            "Task-1 empirical-Bayes calibration and all-visible-location posterior initialization; "
            "frozen theta; delayed-observation strict online stream"
            if args.task1_posterior_init
            else "Task-1 Route-B empirical-Bayes calibration; frozen theta; new-block-only Task-2(+) streaming"
        ),
        "delayed_observations": bool(delayed_observation_blocks > 0),
        "delayed_observation_blocks": int(delayed_observation_blocks),
        "delayed_observation_rows": delayed_observation_rows,
        "task1_posterior_initialization": bool(args.task1_posterior_init),
        "task1_posterior_initialization_rows": task1_initialization_rows,
        "task1_posterior_initialization_summary": task1_initialization_summary,
        "target_mode": "joint X-lag mean and GP posterior",
        "temporal_representation": args.representation,
        "temporal_kernel": temporal_kernel_metadata(args.temporal_kernel, spectral_mixture),
        "solver_backend": solver_backend,
        "temporal_factor_device": str(temporal_factor_device),
        "temporal_factor_policy": args.temporal_factor_device,
        "basis_protocol": (
            "cumulative-changing HiPPO horizon [t0,tk]"
            if args.representation == "analytic_hippo_rff"
            else "fixed global uniformly spaced temporal inducing locations (known horizon)"
        ),
        "split_seed": args.seed,
        "mt": args.mt,
        "ms": args.ms,
        "spatial_kernel_mode": args.spatial_kernel_mode,
        "num_stream_times": int(stream_times.size),
        "num_blocks": len(blocks),
        "num_train_space": int(train_indices.size),
        "num_test_space": int(test_indices.size),
        "learned_theta": theta,
        "hyperparameters": "learned on Task 1 by the matching Route B representation, then frozen",
        "feature_projection": {
            "path": (
                None
                if args.feature_projection_npz is None
                else str(args.feature_projection_npz)
            ),
            "original_dimension": int(original_beta_dimension),
            "active_dimension": int(beta_dimension),
        },
        "predictive_variance": (
            "full structured-joint conditional; conditional residual included"
            if args.include_conditional_residual_variance
            else "finite projected DTC; no conditional residual variance"
        ),
        "overall_current_block": overall,
        "final_block": final_block,
        "timing": {
            "spatial_projection_setup_seconds": spatial_setup_seconds,
            "solver_setup_seconds": solver_setup_seconds,
            "temporal_static_setup_seconds": temporal_setup_seconds,
            "task1_posterior_initialization_seconds": task1_initialization_seconds,
            "stream_feature_loading_seconds": total_feature_seconds,
            "stream_factor_preparation_seconds": total_factor_seconds,
            "stream_update_seconds": total_update_seconds,
            "stream_prediction_seconds": total_prediction_seconds,
            "mean_block_update_seconds": float(np.mean([row["update_seconds"] for row in rows])),
            "mean_block_factor_preparation_seconds": float(
                np.mean([row["factor_preparation_seconds"] for row in rows])
            ),
            "mean_block_prediction_seconds": float(np.mean([row["prediction_seconds"] for row in rows])),
            "first_block_update_seconds": float(rows[0]["update_seconds"]),
            "mean_steady_state_block_update_seconds": float(
                np.mean([row["update_seconds"] for row in rows[1:]] or [rows[0]["update_seconds"]])
            ),
            "xlag_feature_loading_seconds": phi_cache.loading_seconds,
            "process_total_seconds": time.perf_counter() - process_started,
        },
        "resources": {
            **({"peak_rss_mib": peak_rss_mib()} if runtime is None else runtime.resources()),
            "persistent_state_bytes": persistent_bytes,
            "persistent_state_mib": persistent_bytes / 1024.0**2,
            "history_replay_buffer_bytes": 0,
            "device": solver_device,
            "dtype": solver_dtype,
            "gpu_accelerated": bool(runtime is not None and runtime.uses_cuda),
            "posterior_update_device": solver_device,
            "prediction_device": solver_device,
            "temporal_factor_device": str(temporal_factor_device),
            "spherical_bessel_device": (
                str(temporal_factor_device)
                if args.representation == "analytic_hippo_rff"
                else None
            ),
            "host_to_device_block_factors_in_update_timing": solver_backend == "torch",
        },
        "environment": host_snapshot(ROOT),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(rows, args.blockwise_output)
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        evaluated_stop = blocks[-1].stop
        np.savez_compressed(
            args.predictions_output,
            y_true=y[:evaluated_stop, test_indices],
            pred_mean=prediction_grid[:evaluated_stop],
            pred_var=variance_grid[:evaluated_stop],
            test_indices=test_indices,
            times=stream_times[:evaluated_stop],
            variance_mode=np.asarray(
                "full_joint_conditional"
                if args.include_conditional_residual_variance
                else "current_dtc"
            ),
            mt=np.asarray(args.mt),
            ms=np.asarray(args.ms),
        )
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
