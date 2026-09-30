#!/usr/bin/env python3
"""ICLR 2027 ERA5 transfer-mechanism experiments.

This runner is intentionally separate from the production benchmark runner.  It
reuses the materialised strict-online protocol and calibrated VFE/SM
hyperparameters, and writes only to a caller-provided result directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_iclr_era5_routeb_strict_online import (  # noqa: E402
    ProtocolPhiCache,
    TaskPhiCache,
    heldout_targets_for_evaluation,
    make_factors,
    make_prediction_factors,
    metrics,
)
from scripts.run_hipposvgp_era5_routeb import vectorized_predict_with_C  # noqa: E402
from scripts.run_routeb_online_parity_ladder import (  # noqa: E402
    spatial_projection,
    temporal_factors,
)
from stvgp_kronecker.joint_ssgp_kron.kron_utils import (  # noqa: E402
    dense_Du_for_tests,
    inv_spd,
    solve_spd,
    vec_f,
)
from stvgp_kronecker.joint_ssgp_kron.model import (  # noqa: E402
    JointSSGPKronHiPPOSVGP,
)
from stvgp_kronecker.joint_ssgp_kron.synthetic import (  # noqa: E402
    make_analytic_temporal_builder,
)
from stvgp_kronecker.joint_ssgp_kron.torch_backend import (  # noqa: E402
    TorchJointSSGPKronHiPPOSVGP,
)
from stvgp_kronecker.temporal_kernel_config import (  # noqa: E402
    load_spectral_mixture_config,
)


MODES = (
    "structured_changing",
    "identity_reuse_changing",
    "structured_fixed",
    "mean_field_changing",
    "zero_cross_changing",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _relative_error(left: Any, right: Any, floor: float = 1e-14) -> float:
    a = _numpy(left)
    b = _numpy(right)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), floor))


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _select_inducing(arrays: Any, ms: int) -> np.ndarray:
    key = f"inducing_coords_ms{ms}"
    if key in arrays:
        return np.asarray(arrays[key], dtype=np.float64)
    base = np.asarray(arrays["inducing_coords_ms128"], dtype=np.float64)
    if ms > base.shape[0]:
        raise ValueError(f"Cannot derive M_s={ms} from {base.shape[0]} stored landmarks")
    index = np.linspace(0, base.shape[0] - 1, ms, dtype=int)
    if np.unique(index).size != ms:
        raise ValueError("Deterministic inducing-point subset contains duplicates")
    return base[index]


def _state_diagnostics(state: Any) -> dict[str, float]:
    delta_beta = state.beta_mean
    cross = state.R_beta_u
    return {
        "beta_norm": float(torch.linalg.vector_norm(delta_beta).detach().cpu()),
        "u_norm": float(torch.linalg.vector_norm(state.M_u).detach().cpu()),
        "cross_precision_norm": (
            0.0
            if cross is None
            else float(torch.linalg.vector_norm(cross).detach().cpu())
        ),
        "persistent_tensor_bytes": int(state.tensor_bytes()),
    }


def _state_gap(state: Any, reference: Any) -> dict[str, float]:
    fields = (
        "beta_mean",
        "beta_cov",
        "M_u",
        "R_beta_beta",
        "R_beta_u",
        "h_beta",
        "B_temporal",
        "H_info",
    )
    return {
        f"reference_{field}_relerr": _relative_error(
            getattr(state, field), getattr(reference, field)
        )
        for field in fields
    }


def _conditional_trace_ratio(
    old_kt: Any | None,
    new_kt: Any,
    k_on: Any | None,
) -> tuple[float, float]:
    if old_kt is None or k_on is None:
        return 0.0, 0.0
    koo = _numpy(old_kt)
    knn = _numpy(new_kt)
    kon = _numpy(k_on)
    conditional = koo - kon @ solve_spd(knn, kon.T, jitter=1e-12)
    conditional = 0.5 * (conditional + conditional.T)
    minimum = float(np.linalg.eigvalsh(conditional).min())
    ratio = float(np.trace(conditional) / max(float(np.trace(koo)), 1e-14))
    return ratio, minimum


def _gaussian_kl_diag(
    mean_p: np.ndarray,
    var_p: np.ndarray,
    mean_q: np.ndarray,
    var_q: np.ndarray,
) -> float:
    vp = np.maximum(np.asarray(var_p, dtype=float), 1e-12)
    vq = np.maximum(np.asarray(var_q, dtype=float), 1e-12)
    dm = np.asarray(mean_p, dtype=float) - np.asarray(mean_q, dtype=float)
    return float(np.mean(0.5 * (np.log(vq / vp) + (vp + dm * dm) / vq - 1.0)))


def _dense_reference(
    model: TorchJointSSGPKronHiPPOSVGP,
    state: Any,
    phi: np.ndarray,
    t_eval: np.ndarray,
    c_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialise the small canary posterior and its predictive moments."""

    kt = _numpy(state.Kt_current)
    ks = _numpy(state.Ks)
    b = _numpy(state.B_temporal)
    g = _numpy(state.G)
    rbb = _numpy(state.R_beta_beta)
    rbu = _numpy(state.R_beta_u)
    hb = _numpy(state.h_beta)
    hu = vec_f(_numpy(state.H_info))
    prior_precision = _numpy(state.beta_prior_precision)
    prior_natural = _numpy(state.beta_prior_natural)
    du = dense_Du_for_tests(
        inv_spd(kt, jitter=model.jitter),
        inv_spd(ks, jitter=model.jitter),
        b,
        g,
    )
    precision = np.block(
        [
            [prior_precision + rbb, rbu],
            [rbu.T, du],
        ]
    )
    information = np.concatenate([prior_natural + hb, hu])
    covariance = inv_spd(precision, jitter=model.jitter)
    mean = covariance @ information

    n_space = c_eval.shape[0]
    residual_design = np.einsum(
        "tj,si->tsji", t_eval, c_eval, optimize=True
    ).reshape(t_eval.shape[0] * n_space, -1)
    design = np.concatenate([phi, residual_design], axis=1)
    predictive_mean = design @ mean
    posterior_variance = np.einsum("ni,ij,nj->n", design, covariance, design)
    s_var = np.einsum("si,ij,sj->s", c_eval, ks, c_eval)
    t_var = np.einsum("ti,ij,tj->t", t_eval, kt, t_eval)
    projected = np.outer(t_var, s_var).reshape(-1)
    conditional = np.maximum(model.prior_point_variance - projected, 0.0)
    predictive_variance = np.maximum(
        model.sigma2 + conditional + posterior_variance, 1e-12
    )
    return mean, covariance, predictive_mean, predictive_variance


def _predict(
    model: TorchJointSSGPKronHiPPOSVGP,
    state: Any,
    factors: Any,
    c_test: torch.Tensor,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean, variance, _ = model.predict_with_C(
        state=state,
        T_eval=factors.T,
        Phi=factors.Phi,
        C_eval=c_test,
        chunk_size=chunk_size,
        include_conditional_residual_variance=True,
        validate_conditional_residual_variance=True,
    )
    return np.asarray(mean), np.asarray(variance)


def _update(
    model: TorchJointSSGPKronHiPPOSVGP,
    mode: str,
    factors: Any,
    state: Any | None,
    c_train: torch.Tensor,
    *,
    identity_transfer: bool = False,
) -> Any:
    common = dict(
        y_vec=factors.y_vec,
        Phi=factors.Phi,
        T_n=factors.T,
        Kt_new=factors.Kt,
        state=state,
        K_on_t=factors.K_on_t,
        C_observed=c_train,
    )
    if mode == "mean_field_changing":
        return model.update_block_ssgp_transfer(**common)
    if identity_transfer and state is not None:
        common["L_t_override"] = np.eye(factors.Kt.shape[0])
    return model.update_block_structured_joint_ssgp_transfer(
        **common,
        zero_cross=(mode == "zero_cross_changing"),
    )


def _reference_state(
    model: TorchJointSSGPKronHiPPOSVGP,
    history: list[tuple[slice, np.ndarray, np.ndarray]],
    *,
    builder: Any,
    times: np.ndarray,
    basis: slice,
    kt: np.ndarray,
    train_indices: np.ndarray,
    c_train: torch.Tensor,
) -> Any:
    """Recompute all visible history in the current basis without transfer."""

    reference = None
    identity = np.eye(kt.shape[0])
    for query, y_block, phi_block in history:
        t_hist, kt_hist, _ = temporal_factors(
            builder=builder,
            times=times,
            query=query,
            basis=basis,
            old_basis=None,
            basis_mode="cumulative_changing",
        )
        np.testing.assert_allclose(kt_hist, kt, rtol=2e-8, atol=2e-10)
        factors = make_factors(
            y_block,
            phi_block,
            train_indices,
            t_hist,
            kt_hist,
            None if reference is None else kt_hist,
            query,
            "analytic_hippo_rff",
        )
        reference = model.update_block_structured_joint_ssgp_transfer(
            y_vec=factors.y_vec,
            Phi=factors.Phi,
            T_n=factors.T,
            Kt_new=factors.Kt,
            state=reference,
            K_on_t=factors.K_on_t,
            L_t_override=None if reference is None else identity,
            C_observed=c_train,
        )
    return reference


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.mode not in MODES:
        raise ValueError(args.mode)
    if args.batch_reference and args.mode not in {
        "structured_changing",
        "identity_reuse_changing",
        "structured_fixed",
    }:
        raise ValueError("Batch reference is defined only for structured arms")
    if args.mode == "structured_fixed" and not args.batch_reference and args.canary:
        raise ValueError("The fixed-basis canary requires --batch-reference")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        device_index = 0 if device.index is None else device.index
        torch.cuda.set_device(device_index)
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device_index)

    arrays = np.load(args.protocol_npz)
    metadata = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    theta_payload = json.loads(args.theta_json.read_text(encoding="utf-8"))
    theta = theta_payload["learned_theta"]
    mixture = load_spectral_mixture_config(args.spectral_mixture_json)
    if mixture is None:
        raise ValueError("A spectral-mixture configuration is required")

    times = np.asarray(arrays["stream_times"], dtype=np.float64)
    y = np.asarray(arrays["stream_y"], dtype=np.float64)
    coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
    train_indices = np.asarray(arrays["train_indices"], dtype=int)
    test_indices = np.asarray(arrays["test_indices"], dtype=int)
    blocks = [
        slice(int(a), int(b))
        for a, b in zip(arrays["block_start"], arrays["block_stop"])
    ]
    if args.max_blocks:
        blocks = blocks[: args.max_blocks]
    inducing = _select_inducing(arrays, args.ms)
    ks, c_all = spatial_projection(coordinates, inducing, theta["ell_s"])
    c_train_np = c_all[train_indices]
    c_test_np = c_all[test_indices]
    c_train = torch.as_tensor(c_train_np, device=device, dtype=torch.float64)
    c_test = torch.as_tensor(c_test_np, device=device, dtype=torch.float64)

    protocol_phi = np.asarray(arrays["stream_phi"]) if "stream_phi" in arrays else None
    if protocol_phi is None:
        phi_cache = TaskPhiCache(
            metadata["root"] if args.data_root is None else args.data_root,
            y,
            metadata["xlag"]["length"],
        )
        beta_dimension = 133
    else:
        phi_cache = ProtocolPhiCache(protocol_phi, y.shape)
        beta_dimension = int(protocol_phi.shape[-1])
    original_beta_dimension = beta_dimension
    if args.canary and args.canary_beta_dim > 0:
        beta_dimension = min(beta_dimension, args.canary_beta_dim)

    model = TorchJointSSGPKronHiPPOSVGP(
        Ks=ks,
        C=c_train,
        sigma2=float(theta["noise_std"]) ** 2,
        beta_prior_mean=np.zeros(beta_dimension),
        beta_prior_cov=args.beta_prior_variance * np.eye(beta_dimension),
        prior_point_variance=float(theta["kernel_variance"]),
        jitter=args.jitter,
        device=device,
        dtype=torch.float64,
    )
    canary_model = None
    canary_state = None
    if args.canary:
        canary_model = JointSSGPKronHiPPOSVGP(
            Ks=ks,
            C=c_train_np,
            sigma2=float(theta["noise_std"]) ** 2,
            beta_prior_mean=np.zeros(beta_dimension),
            beta_prior_cov=args.beta_prior_variance * np.eye(beta_dimension),
            prior_point_variance=float(theta["kernel_variance"]),
            jitter=0.0,
        )
    builder = make_analytic_temporal_builder(
        mt=args.mt,
        lengthscale=float(theta["ell_t"]),
        variance=float(theta["kernel_variance"]),
        rff_sample_size=args.rff_sample_size,
        seed=0,
        jitter=1e-7,
        kernel_type="spectral_mixture",
        spectral_mixture_weights=mixture["weights"],
        spectral_mixture_means=mixture["means"],
        spectral_mixture_scales=mixture["scales"],
    ).to(device="cpu", dtype=torch.float64)

    state = None
    old_basis = None
    old_temporal_basis = None
    old_beta = None
    old_kt = None
    history: list[tuple[slice, np.ndarray, np.ndarray]] = []
    block_rows: list[dict[str, Any]] = []
    all_y: list[np.ndarray] = []
    all_mean: list[np.ndarray] = []
    all_var: list[np.ndarray] = []
    fixed_basis = slice(0, times.size)

    for block_id, block in enumerate(blocks):
        phi_block, task_index = phi_cache.block(block)
        phi_block = np.asarray(phi_block, dtype=np.float64)[..., :beta_dimension]
        y_block = y[block]
        history.append((block, y_block, phi_block))
        basis = fixed_basis if args.mode == "structured_fixed" else slice(0, block.stop)
        temporal_started = time.perf_counter()
        t_mat, kt, k_on, new_temporal_basis = temporal_factors(
            builder=builder,
            times=times,
            query=block,
            basis=basis,
            old_basis=(None if args.mode == "structured_fixed" else old_basis),
            basis_mode="cumulative_changing",
            old_temporal_basis=(
                None if args.mode == "structured_fixed" else old_temporal_basis
            ),
            return_temporal_basis=True,
        )
        if args.mode == "structured_fixed" and state is not None:
            k_on = kt
        temporal_seconds = time.perf_counter() - temporal_started
        train_factors = make_factors(
            y_block,
            phi_block,
            train_indices,
            t_mat,
            kt,
            k_on,
            block,
            "analytic_hippo_rff",
        )
        test_factors = make_prediction_factors(
            phi_block,
            test_indices,
            t_mat,
            kt,
            block,
            "analytic_hippo_rff",
        )
        hidden_label_reads = 0

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_started = time.perf_counter()
        state = _update(
            model,
            args.mode,
            train_factors,
            state,
            c_train,
            identity_transfer=(
                args.mode in {"structured_fixed", "identity_reuse_changing"}
            ),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_seconds = time.perf_counter() - update_started
        mean, variance = _predict(
            model, state, test_factors, c_test, args.prediction_chunk_size
        )
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise FloatingPointError("Non-finite predictive moment")
        if float(np.min(variance)) < 0.0:
            raise FloatingPointError("Negative predictive variance")
        test_y_vec = heldout_targets_for_evaluation(
            y, block, test_indices, prediction_complete=True
        )

        conditional_ratio, conditional_min_eig = _conditional_trace_ratio(
            old_kt, kt, k_on
        )
        row: dict[str, Any] = {
            "seed": args.seed,
            "mode": args.mode,
            "ms": args.ms,
            "mt": args.mt,
            "block_id": block_id,
            "block_start": block.start,
            "block_stop": block.stop,
            "task": task_index,
            "hidden_label_reads": hidden_label_reads,
            "temporal_factor_seconds": temporal_seconds,
            "update_seconds": update_seconds,
            "conditional_trace_ratio": conditional_ratio,
            "conditional_min_eigenvalue": conditional_min_eig,
            "predictive_variance_min": float(np.min(variance)),
            "beta_delta_norm": (
                0.0
                if old_beta is None
                else float(np.linalg.norm(_numpy(state.beta_mean) - old_beta))
            ),
            "cross_block_contribution_norm": (
                0.0
                if state.R_beta_u is None or old_beta is None
                else float(
                    np.linalg.norm(
                        _numpy(state.R_beta_u).T
                        @ (_numpy(state.beta_mean) - old_beta)
                    )
                )
            ),
            **_state_diagnostics(state),
            **metrics(test_y_vec, mean, variance),
        }

        if args.batch_reference:
            reference_started = time.perf_counter()
            reference = _reference_state(
                model,
                history,
                builder=builder,
                times=times,
                basis=basis,
                kt=kt,
                train_indices=train_indices,
                c_train=c_train,
            )
            reference_mean, reference_variance = _predict(
                model,
                reference,
                test_factors,
                c_test,
                args.prediction_chunk_size,
            )
            row.update(_state_gap(state, reference))
            row.update(
                {
                    "reference_seconds": time.perf_counter() - reference_started,
                    "reference_predictive_mean_relerr": _relative_error(
                        mean, reference_mean
                    ),
                    "reference_predictive_variance_relerr": _relative_error(
                        variance, reference_variance
                    ),
                    "reference_predictive_gaussian_kl": _gaussian_kl_diag(
                        mean, variance, reference_mean, reference_variance
                    ),
                    "reference_posterior_mean_relerr": _relative_error(
                        np.concatenate(
                            [_numpy(state.beta_mean), vec_f(_numpy(state.M_u))]
                        ),
                        np.concatenate(
                            [
                                _numpy(reference.beta_mean),
                                vec_f(_numpy(reference.M_u)),
                            ]
                        ),
                    ),
                    "reference_posterior_cov_relerr": _relative_error(
                        state.beta_cov, reference.beta_cov
                    ),
                }
            )

        if args.canary:
            if args.ms > 16 or args.mt > 16:
                raise ValueError("Dense canary is limited to M_s,M_t <= 16")
            assert canary_model is not None
            canary_kwargs = dict(
                y_vec=train_factors.y_vec,
                Phi=train_factors.Phi,
                T_n=train_factors.T,
                Kt_new=train_factors.Kt,
                state=canary_state,
                K_on_t=train_factors.K_on_t,
            )
            if args.mode == "structured_fixed" and canary_state is not None:
                canary_kwargs["L_t_override"] = np.eye(args.mt)
            canary_state = canary_model.update_block_structured_joint_ssgp_transfer(
                **canary_kwargs
            )
            dense_mean, dense_cov, dense_pred_mean, dense_pred_var = _dense_reference(
                canary_model,
                canary_state,
                test_factors.Phi,
                test_factors.T,
                c_test_np,
            )
            canary_pred_mean, canary_pred_var, _ = vectorized_predict_with_C(
                canary_model,
                canary_state,
                test_factors,
                c_test_np,
                prediction_mode="streaming_sylvester",
                chunk_size=args.prediction_chunk_size,
                include_conditional_residual_variance=True,
            )
            structured_mean = np.concatenate(
                [canary_state.beta_mean, vec_f(canary_state.M_u)]
            )
            row.update(
                {
                    "dense_state_mean_relerr": _relative_error(
                        structured_mean, dense_mean
                    ),
                    "dense_beta_cov_relerr": _relative_error(
                        canary_state.beta_cov,
                        dense_cov[:beta_dimension, :beta_dimension],
                    ),
                    "dense_predictive_mean_relerr": _relative_error(
                        canary_pred_mean, dense_pred_mean
                    ),
                    "dense_predictive_variance_relerr": _relative_error(
                        canary_pred_var, dense_pred_var
                    ),
                    "numpy_torch_predictive_mean_relerr": _relative_error(
                        mean, canary_pred_mean
                    ),
                    "numpy_torch_predictive_variance_relerr": _relative_error(
                        variance, canary_pred_var
                    ),
                }
            )

        block_rows.append(row)
        all_y.append(np.asarray(test_y_vec))
        all_mean.append(mean)
        all_var.append(variance)
        old_beta = _numpy(state.beta_mean).copy()
        old_kt = _numpy(state.Kt_current).copy()
        old_basis = basis
        old_temporal_basis = new_temporal_basis

    aggregate = metrics(
        np.concatenate(all_y), np.concatenate(all_mean), np.concatenate(all_var)
    )
    canary_fields = [
        key
        for key in block_rows[0]
        if key.endswith("relerr")
        and (
            key.startswith("dense_")
            or (args.mode == "structured_fixed" and key.startswith("reference_"))
        )
    ]
    maximum_canary_error = max(
        (float(row[key]) for row in block_rows for key in canary_fields),
        default=0.0,
    )
    canary_status = "not_requested"
    if args.canary:
        canary_status = "pass" if maximum_canary_error <= args.canary_tolerance else "fail"
        if maximum_canary_error > args.canary_stop_threshold:
            canary_status = "fatal"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(block_rows, args.output_dir / "blocks.csv")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y=np.concatenate(all_y),
        mean=np.concatenate(all_mean),
        variance=np.concatenate(all_var),
    )
    git_status = _git_value("status", "--short")
    result = {
        "status": "complete",
        "mode": args.mode,
        "seed": args.seed,
        "ms": args.ms,
        "mt": args.mt,
        "beta_dimension": beta_dimension,
        "original_beta_dimension": original_beta_dimension,
        "protocol": "ERA5 short Task 1--2 strict-online",
        "objective": "VFE",
        "temporal_kernel": "spectral_mixture",
        "predictive_variance": "full_conditional",
        "batch_reference": bool(args.batch_reference),
        "historical_transport": (
            "identity_reuse_in_successive_changing_hippo_functionals"
            if args.mode == "identity_reuse_changing"
            else "conditional_transport"
            if args.mode == "structured_changing"
            else "fixed_global_identity"
            if args.mode == "structured_fixed"
            else "conditional_transport"
        ),
        "canary": {
            "requested": bool(args.canary),
            "status": canary_status,
            "maximum_relative_error": maximum_canary_error,
            "tolerance": args.canary_tolerance,
            "stop_threshold": args.canary_stop_threshold,
        },
        "aggregate": aggregate,
        "diagnostics": {
            "blocks": len(block_rows),
            "hidden_label_reads": int(sum(r["hidden_label_reads"] for r in block_rows)),
            "minimum_predictive_variance": float(
                min(r["predictive_variance_min"] for r in block_rows)
            ),
            "maximum_beta_delta_norm": float(
                max(r["beta_delta_norm"] for r in block_rows)
            ),
            "persistent_state_bytes_unique": sorted(
                {int(r["persistent_tensor_bytes"]) for r in block_rows}
            ),
        },
        "timing": {
            "wall_seconds": time.perf_counter() - started,
            "update_seconds": float(sum(r["update_seconds"] for r in block_rows)),
            "temporal_factor_seconds": float(
                sum(r["temporal_factor_seconds"] for r in block_rows)
            ),
            "timing_scope": "RTX 5070 Laptop mechanism diagnostic; not comparable to RTX 4090 tables",
        },
        "resources": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 1024.0**2
                if device.type == "cuda"
                else 0.0
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "provenance": {
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_status_entry_count": len(git_status.splitlines()),
            "git_status_sha256": hashlib.sha256(git_status.encode("utf-8")).hexdigest(),
            "protocol_npz": str(args.protocol_npz.resolve()),
            "protocol_npz_sha256": _sha256(args.protocol_npz),
            "protocol_json": str(args.protocol_json.resolve()),
            "protocol_json_sha256": _sha256(args.protocol_json),
            "theta_json": str(args.theta_json.resolve()),
            "theta_json_sha256": _sha256(args.theta_json),
            "spectral_mixture_json": str(args.spectral_mixture_json.resolve()),
            "spectral_mixture_json_sha256": _sha256(args.spectral_mixture_json),
            "runner_sha256": _sha256(Path(__file__)),
            "command": " ".join(sys.argv),
            "cwd": os.getcwd(),
        },
    }
    _write_json(result, args.output_dir / "result.json")
    if canary_status in {"fail", "fatal"}:
        raise RuntimeError(
            f"Canary {canary_status}: maximum relative error {maximum_canary_error:.3e}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--theta-json", type=Path, required=True)
    parser.add_argument("--spectral-mixture-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--prediction-chunk-size", type=int, default=8192)
    parser.add_argument("--beta-prior-variance", type=float, default=1000.0)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-reference", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument(
        "--canary-beta-dim",
        type=int,
        default=4,
        help="Small feature rank used only by the numerical correctness canary.",
    )
    parser.add_argument("--canary-tolerance", type=float, default=1e-8)
    parser.add_argument("--canary-stop-threshold", type=float, default=1e-6)
    return parser.parse_args()


if __name__ == "__main__":
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
