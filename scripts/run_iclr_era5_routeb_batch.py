#!/usr/bin/env python3
"""Controlled batch empirical-Bayes Route B on the shared ERA5 protocol."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_routeb_batch_empirical_bayes import (
    GridData,
    evaluate,
    object_array_bytes,
    serialized_training_state_bytes,
    tensor_training_data,
    write_csv,
)
from scripts.run_hipposvgp_era5_routeb import augment_dataset_phi
from scripts.run_iclr_era5_routeb_strict_online import TaskPhiCache
from stvgp_kronecker.benchmark_runtime import (
    SynchronizedTimer,
    host_snapshot,
    resolve_torch_runtime,
)
from stvgp_kronecker.data.hipposvgp_era5 import load_hipposvgp_era5
from stvgp_kronecker.routeb_empirical_bayes import BatchRouteBEmpiricalBayes


def load_protocol(
    path: Path, ms: int, data_part: str
) -> tuple[GridData, np.ndarray, np.ndarray]:
    arrays = np.load(path)
    time_key = "calibration_times" if data_part == "calibration" else "stream_times"
    target_key = "calibration_y" if data_part == "calibration" else "stream_y"
    y = np.asarray(arrays[target_key], dtype=np.float64)
    zero_phi = np.zeros((*y.shape, 0), dtype=np.float64)
    inducing_key = f"inducing_coords_ms{ms}"
    if inducing_key not in arrays:
        raise KeyError(f"{path} does not contain {inducing_key}")
    data = GridData(
        times=np.asarray(arrays[time_key], dtype=np.float64),
        coordinates=np.asarray(arrays["coordinates"], dtype=np.float64),
        y=y,
        phi=zero_phi,
        train_indices=np.asarray(arrays["train_indices"], dtype=int),
        test_indices=np.asarray(arrays["test_indices"], dtype=int),
        spatial_inducing=np.asarray(arrays[inducing_key], dtype=np.float64),
    )
    return (
        data,
        np.asarray(arrays["fit_indices"], dtype=int),
        np.asarray(arrays["validation_indices"], dtype=int),
    )


def load_joint_phi(
    *,
    arrays: np.lib.npyio.NpzFile,
    protocol_json: Path,
    xlag_length: int,
    data_part: str,
) -> tuple[np.ndarray, float]:
    metadata = json.loads(protocol_json.read_text(encoding="utf-8"))
    if data_part == "calibration":
        started = time.perf_counter()
        raw = load_hipposvgp_era5(
            metadata["root"], tasks=("task_1",), variable_index=0, split="all"
        )
        augmented = augment_dataset_phi(
            raw, phi_mode="medium_era5_xlag", xlag_length=xlag_length
        )
        expected = np.asarray(arrays["calibration_y"], dtype=np.float64)
        np.testing.assert_allclose(augmented.Y, expected, atol=2e-6, rtol=0.0)
        phi = np.asarray(augmented.Phi, dtype=np.float32).reshape(
            expected.shape[0], expected.shape[1], -1
        )
        return phi, time.perf_counter() - started

    y = np.asarray(arrays["stream_y"], dtype=np.float64)
    cache = TaskPhiCache(metadata["root"], y, xlag_length)
    blocks = [
        slice(int(start), int(stop))
        for start, stop in zip(arrays["block_start"], arrays["block_stop"])
    ]
    phi = None
    for block in blocks:
        block_phi, _ = cache.block(block)
        if phi is None:
            phi = np.empty((*y.shape, block_phi.shape[-1]), dtype=np.float32)
        phi[block] = np.asarray(block_phi, dtype=np.float32)
    if phi is None:
        raise ValueError("Protocol does not contain any streaming blocks")
    return phi, cache.loading_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-part", choices=["calibration", "stream"], default="stream"
    )
    parser.add_argument(
        "--target-mode",
        choices=["direct", "shared_xlag_residual", "joint_xlag"],
        required=True,
    )
    parser.add_argument("--representation", choices=["analytic_hippo_rff", "inducing_points"], required=True)
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--beta-prior-variance", type=float, default=1000.0)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--xlag-length", type=int, default=10)
    parser.add_argument("--initial-ell-t", type=float, default=0.05)
    parser.add_argument("--initial-ell-s", nargs=2, type=float, default=[0.35, 0.35])
    parser.add_argument("--initial-kernel-variance", type=float, default=1.0)
    parser.add_argument("--initial-noise", type=float, default=0.1)
    parser.add_argument("--prediction-chunk-size", type=int, default=8192)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--save-pointwise", action="store_true")
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument(
        "--evaluation-backend",
        choices=["auto", "numpy", "torch"],
        default="auto",
        help="Backend for validation/final posterior recovery and prediction.",
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--profile-flops", action="store_true")
    args = parser.parse_args()

    process_started = time.perf_counter()
    runtime = resolve_torch_runtime(args.device, args.dtype)
    if args.evaluation_backend == "auto":
        evaluation_backend = "torch" if runtime.uses_cuda else "numpy"
    else:
        evaluation_backend = args.evaluation_backend
    if evaluation_backend == "numpy" and args.dtype != "float64":
        raise ValueError(
            "NumPy evaluation is the float64 reference; use --evaluation-backend torch "
            "for a float32 experiment."
        )
    torch.manual_seed(args.model_seed)
    if runtime.uses_cuda:
        torch.cuda.manual_seed_all(args.model_seed)
    np.random.seed(args.model_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, fit_indices, validation_indices = load_protocol(
        args.protocol_npz, args.ms, args.data_part
    )
    arrays = np.load(args.protocol_npz)
    feature_loading_seconds = 0.0
    if args.target_mode == "joint_xlag":
        if args.protocol_json is None:
            raise ValueError("--protocol-json is required for joint_xlag")
        data.phi, feature_loading_seconds = load_joint_phi(
            arrays=arrays,
            protocol_json=args.protocol_json,
            xlag_length=args.xlag_length,
            data_part=args.data_part,
        )
    if args.target_mode == "shared_xlag_residual":
        offset_key = (
            "task1_calibration_mean"
            if args.data_part == "calibration"
            else "batch_stream_mean"
        )
        offset = np.asarray(arrays[offset_key], dtype=np.float64)
    else:
        offset = np.zeros_like(data.y)
    posterior_y = data.y - offset
    zero_phi = data.phi
    y_train, phi_train, coordinates_train = tensor_training_data(
        data,
        fit_indices,
        y_override=posterior_y,
        phi_override=zero_phi,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    model = BatchRouteBEmpiricalBayes(
        times=data.times,
        spatial_inducing=data.spatial_inducing,
        mt=args.mt,
        representation=args.representation,
        initial_ell_t=args.initial_ell_t,
        initial_ell_s=tuple(args.initial_ell_s),
        initial_kernel_variance=args.initial_kernel_variance,
        initial_noise_std=args.initial_noise,
        rff_sample_size=args.rff_sample_size,
        seed=args.model_seed,
        objective_type="finite_dtc",
    ).to(device=runtime.device, dtype=runtime.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initial_spatial_inducing = model.spatial_inducing.detach().clone()
    initial_temporal_support = model.temporal.z_t.detach().clone()
    initial_rff = (
        model.temporal.builder.base_frequencies.detach().clone()
        if model.temporal.builder is not None
        else None
    )

    profiled_step_flops = None
    flop_profile_seconds = 0.0
    flop_profile_error = None
    if args.profile_flops:
        try:
            from torch.utils.flop_counter import FlopCounterMode

            with SynchronizedTimer(runtime.synchronize) as flop_timer:
                optimizer.zero_grad(set_to_none=True)
                with FlopCounterMode(display=False) as counter:
                    profile_objective = model.objective(
                        y_matrix=y_train,
                        phi_tensor=phi_train,
                        spatial_coordinates=coordinates_train,
                        beta_prior_variance=args.beta_prior_variance,
                    )
                    profile_objective.nlml_per_observation.backward()
                profiled_step_flops = int(counter.get_total_flops())
                optimizer.zero_grad(set_to_none=True)
            flop_profile_seconds = flop_timer.elapsed
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            flop_profile_error = f"{type(exc).__name__}: {exc}"

    warmup_seconds = 0.0
    if args.warmup_steps > 0:
        with SynchronizedTimer(runtime.synchronize) as warmup_timer:
            for _ in range(args.warmup_steps):
                optimizer.zero_grad(set_to_none=True)
                warmup_objective = model.objective(
                    y_matrix=y_train,
                    phi_tensor=phi_train,
                    spatial_coordinates=coordinates_train,
                    beta_prior_variance=args.beta_prior_variance,
                )
                warmup_objective.nlml_per_observation.backward()
            optimizer.zero_grad(set_to_none=True)
        warmup_seconds = warmup_timer.elapsed
    runtime.reset_peak_memory()

    best_nll = float("inf")
    best_iteration = 0
    best_elapsed = None
    best_state = None
    trace = []
    iteration_times = []
    validation_seconds = 0.0
    training_started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        with SynchronizedTimer(runtime.synchronize) as iteration_timer:
            optimizer.zero_grad(set_to_none=True)
            objective = model.objective(
                y_matrix=y_train,
                phi_tensor=phi_train,
                spatial_coordinates=coordinates_train,
                beta_prior_variance=args.beta_prior_variance,
            )
            loss = objective.nlml_per_observation
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite NLML at iteration {iteration}")
            loss.backward()
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 20.0
            )
            optimizer.step()
            model.clamp_parameters()
        iteration_seconds = iteration_timer.elapsed
        gradient_norm = float(gradient_norm_tensor.detach().cpu())
        iteration_times.append(iteration_seconds)
        row = {
            "iteration": iteration,
            "train_nlml_per_observation": float(loss.detach()),
            "gradient_norm_before_clip": gradient_norm,
            "iteration_seconds": iteration_seconds,
            **model.theta(),
        }
        if iteration == 1 or iteration % args.validation_every == 0 or iteration == args.iterations:
            started = time.perf_counter()
            validation, _, _ = evaluate(
                empirical_model=model,
                data=data,
                posterior_indices=fit_indices,
                evaluation_indices=validation_indices,
                representation=args.representation,
                beta_prior_variance=args.beta_prior_variance,
                prediction_chunk_size=args.prediction_chunk_size,
                posterior_y_override=posterior_y,
                posterior_phi_override=zero_phi,
                evaluation_phi_override=zero_phi,
                evaluation_mean_offset=offset,
                include_conditional_residual_variance=False,
                collect_pointwise=False,
                solver_backend=evaluation_backend,
                solver_device=runtime.device,
                solver_dtype=runtime.dtype,
                synchronize=runtime.synchronize,
            )
            elapsed = time.perf_counter() - started
            validation_seconds += elapsed
            row.update(
                validation_nll=validation["nll"],
                validation_rmse=validation["rmse"],
                validation_seconds=elapsed,
            )
            if validation["nll"] < best_nll:
                best_nll = validation["nll"]
                best_iteration = iteration
                best_elapsed = time.perf_counter() - training_started
                best_state = copy.deepcopy(model.state_dict())
        trace.append(row)
        if iteration == 1 or iteration % args.validation_every == 0 or iteration == args.iterations:
            print(json.dumps(row), flush=True)

    if best_state is None:
        raise RuntimeError("No validation checkpoint was recorded")
    training_seconds = time.perf_counter() - training_started
    model.load_state_dict(best_state)
    torch.testing.assert_close(model.spatial_inducing, initial_spatial_inducing, rtol=0.0, atol=0.0)
    torch.testing.assert_close(model.temporal.z_t, initial_temporal_support, rtol=0.0, atol=0.0)
    if initial_rff is not None:
        torch.testing.assert_close(
            model.temporal.builder.base_frequencies, initial_rff, rtol=0.0, atol=0.0
        )

    collect_pointwise = args.save_pointwise or args.predictions_output is not None
    final, pointwise, persistent_bytes = evaluate(
        empirical_model=model,
        data=data,
        posterior_indices=data.train_indices,
        evaluation_indices=data.test_indices,
        representation=args.representation,
        beta_prior_variance=args.beta_prior_variance,
        prediction_chunk_size=args.prediction_chunk_size,
        posterior_y_override=posterior_y,
        posterior_phi_override=zero_phi,
        evaluation_phi_override=zero_phi,
        evaluation_mean_offset=offset,
        include_conditional_residual_variance=False,
        collect_pointwise=collect_pointwise,
        solver_backend=evaluation_backend,
        solver_device=runtime.device,
        solver_dtype=runtime.dtype,
        synchronize=runtime.synchronize,
    )
    checkpoint_bytes = serialized_training_state_bytes(model, optimizer)
    payload = {
        "implementation": "Route B finite-DTC structured empirical Bayes",
        "protocol": "controlled batch/full-history with spatial held-out validation",
        "data_part": args.data_part,
        "target_mode": args.target_mode,
        "temporal_representation": args.representation,
        "evaluation_backend": evaluation_backend,
        "temporal_factor_device": str(runtime.device),
        "split_seed": args.split_seed,
        "mt": args.mt,
        "ms": args.ms,
        "num_time": int(data.times.size),
        "num_train_space": int(data.train_indices.size),
        "num_validation_space": int(validation_indices.size),
        "num_test_space": int(data.test_indices.size),
        "num_xlag_features": int(data.phi.shape[-1]),
        "active_joint_mean_features": int(phi_train.shape[-1]),
        "hyperparameters": "learned by Route B marginal likelihood; fixed inducing locations",
        "predictive_variance": "finite projected DTC; no conditional residual variance",
        "best_iteration": best_iteration,
        "best_validation_nll": best_nll,
        "time_to_best_validation_seconds": best_elapsed,
        "learned_theta": model.theta(),
        "final": final,
        "timing": {
            "warmup_seconds": warmup_seconds,
            "flop_profile_seconds": flop_profile_seconds,
            "training_seconds": training_seconds,
            "mean_iteration_seconds": float(np.mean(iteration_times)),
            "median_iteration_seconds": float(np.median(iteration_times)),
            "mean_steady_state_iteration_seconds": float(
                np.mean(iteration_times[1:] or iteration_times)
            ),
            "validation_seconds": validation_seconds,
            "xlag_feature_loading_seconds": feature_loading_seconds,
            "posterior_setup_seconds": final["posterior_update_seconds"],
            "prediction_seconds": final["prediction_seconds"],
            "process_total_seconds": time.perf_counter() - process_started,
        },
        "resources": {
            **runtime.resources(),
            "persistent_model_state_bytes": persistent_bytes,
            "persistent_model_state_mib": persistent_bytes / 1024.0**2,
            "serialized_training_checkpoint_bytes": checkpoint_bytes,
            "history_replay_buffer_bytes": int(data.y[:, data.train_indices].nbytes),
            "profiled_forward_backward_flops_per_step": profiled_step_flops,
            "estimated_training_flops": (
                profiled_step_flops * args.iterations
                if profiled_step_flops is not None
                else None
            ),
            "flops_scope": (
                "PyTorch-supported operations in one Route-B objective forward/backward; "
                "optimizer, validation, posterior recovery and prediction excluded"
            ),
            "flop_profile_error": flop_profile_error,
            "training_device": str(runtime.device),
            "temporal_factor_device": str(runtime.device),
            "spherical_bessel_device": (
                str(runtime.device)
                if args.representation == "analytic_hippo_rff"
                else None
            ),
            "posterior_update_device": (
                str(runtime.device) if evaluation_backend == "torch" else "cpu"
            ),
            "prediction_device": (
                str(runtime.device) if evaluation_backend == "torch" else "cpu"
            ),
            "posterior_solver_backend": evaluation_backend,
        },
        "environment": host_snapshot(ROOT),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    write_csv(trace, args.output_dir / "training_trace.csv")
    if pointwise:
        if args.save_pointwise:
            write_csv(pointwise, args.output_dir / "pointwise_predictions.csv")
        if args.predictions_output is not None:
            shape = (data.times.size, data.test_indices.size)
            args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.predictions_output,
                y_true=np.asarray([row["y_true"] for row in pointwise]).reshape(shape),
                pred_mean=np.asarray([row["pred_mean"] for row in pointwise]).reshape(shape),
                pred_var=np.asarray([row["pred_var"] for row in pointwise]).reshape(shape),
                test_indices=data.test_indices,
                times=data.times,
            )
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
