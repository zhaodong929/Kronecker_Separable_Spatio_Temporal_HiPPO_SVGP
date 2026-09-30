#!/usr/bin/env python3
"""Periodic empirical-Bayes rebasing for online structured-joint Route B.

Hyperparameters are frozen between rebases, so ordinary block-to-block posterior
transfer is used for most blocks.  At each rebase, proximal empirical Bayes is
fit using only observations that arrived since the previous rebase, followed by
an all-seen posterior rebuild under the updated feature map.  The procedure is
causal, but the replay buffer means it is not a strict bounded-memory method.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hipposvgp_era5_routeb import vectorized_predict_with_C
from scripts.run_routeb_batch_empirical_bayes import (
    evaluate,
    inner_training_validation_split,
    load_controlled_grid,
    object_array_bytes,
    write_csv,
)
from scripts.run_routeb_online_incremental_empirical_bayes import (
    batch_reference_metrics,
    calibration_path,
    optimize_hyperparameters,
    sample_sd,
    theta_log_vector,
    time_slice_grid,
)
from scripts.run_routeb_online_parity_ladder import (
    block_factors,
    predictive_metrics,
    spatial_projection,
    state_mean,
)
from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.synthetic import temporal_spec_for_block
from stvgp_kronecker.routeb_empirical_bayes import BatchRouteBEmpiricalBayes


def rebase_due(block_id: int, rebase_interval: int) -> bool:
    if block_id < 0:
        raise ValueError("block_id must be non-negative")
    if rebase_interval <= 0:
        raise ValueError("rebase_interval must be positive")
    return (block_id + 1) % rebase_interval == 0


def adaptation_slice(last_rebase_stop: int, block: slice) -> slice:
    """Return data that arrived after the previous rebase, including this block."""

    start = int(last_rebase_stop)
    stop = int(block.stop)
    if start < 0 or start >= stop:
        raise ValueError(
            f"Invalid adaptation interval [{start}, {stop}) for block {block}"
        )
    return slice(start, stop)


def history_replay_buffer_bytes(data: Any, stop: int) -> int:
    """Bytes required to retain all seen training observations and covariates."""

    indices = np.asarray(data.train_indices, dtype=int)
    payload = {
        "times": np.asarray(data.times[:stop]),
        "y": np.asarray(data.y[:stop][:, indices]),
        "phi": np.asarray(data.phi[:stop][:, indices]),
    }
    return object_array_bytes(payload)


def temporal_factors_from_model(
    *,
    empirical_model: BatchRouteBEmpiricalBayes,
    all_times: np.ndarray,
    query: slice,
    basis: slice,
    old_basis: slice | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Build DTC factors and an optional old-to-new HiPPO cross covariance."""

    builder = empirical_model.temporal.builder
    if builder is None:
        raise ValueError("Periodic changing-basis transfer requires analytic HiPPO")
    current_spec = temporal_spec_for_block(all_times, basis, moving=True)
    empirical_model.set_temporal_query(
        np.asarray(all_times[query], dtype=float), temporal_horizon=current_spec
    )
    with torch.no_grad():
        t_mat, kt = empirical_model.temporal.factors()
        k_on_t = None
        if old_basis is not None:
            old_spec = temporal_spec_for_block(all_times, old_basis, moving=True)
            k_on_t = builder.compute_kuu_t_cross(old_spec, current_spec)
    return (
        np.asarray(t_mat.cpu(), dtype=float),
        np.asarray(kt.cpu(), dtype=float),
        None if k_on_t is None else np.asarray(k_on_t.cpu(), dtype=float),
    )


def inference_model(
    *,
    empirical_model: BatchRouteBEmpiricalBayes,
    data: Any,
    beta_prior_variance: float,
) -> tuple[JointSSGPKronHiPPOSVGP, np.ndarray, np.ndarray, int]:
    theta = empirical_model.theta()
    ks, c_all = spatial_projection(
        data.coordinates,
        data.spatial_inducing,
        [float(value) for value in theta["ell_s"]],
    )
    c_train = c_all[data.train_indices]
    c_test = c_all[data.test_indices]
    model = JointSSGPKronHiPPOSVGP(
        Ks=ks,
        C=c_train,
        sigma2=float(theta["noise_std"]) ** 2,
        beta_prior_mean=np.zeros(data.phi.shape[-1]),
        beta_prior_cov=beta_prior_variance * np.eye(data.phi.shape[-1]),
        prior_point_variance=float(theta["kernel_variance"]),
    )
    static_bytes = object_array_bytes(
        {
            "Ks": model.Ks,
            "Ks_inv": model.Ks_inv,
            "C": model.C,
            "spatial_inducing": empirical_model.spatial_inducing,
            "temporal_buffers": dict(empirical_model.temporal.named_buffers()),
            "theta": dict(empirical_model.named_parameters()),
        }
    )
    return model, c_train, c_test, static_bytes


def run_seed(
    *,
    base: Path,
    outdir: Path,
    data_root: str,
    split_seed: int,
    mt: int,
    ms: int,
    block_size: int,
    rebase_interval: int,
    eb_steps: int,
    learning_rate: float,
    validation_every: int,
    validation_fraction: float,
    prediction_chunk_size: int,
    parameter_policy: str,
    proximal_lambda: float,
    kernel_variance_lr_multiplier: float,
    noise_lr_multiplier: float,
    noise_floor: float,
    nonfinite_gradient_policy: str,
) -> dict[str, Any]:
    calibration = json.loads(
        calibration_path(base, split_seed).read_text(encoding="utf-8")
    )
    calibration_args = calibration["args"]
    if mt != int(calibration_args["mt"]) or ms != int(calibration_args["ms"]):
        raise ValueError(
            "Periodic EB warm-starts Task-1 calibration and requires matching "
            f"supports M_t={calibration_args['mt']}, M_s={calibration_args['ms']}"
        )
    controlled_npz = (
        base
        / "phase_d_joint_xlag_controlled"
        / f"seed{split_seed}"
        / f"era5_xlag_seed{split_seed}.npz"
    )
    data = load_controlled_grid(
        root=data_root,
        task="task_2",
        controlled_npz=controlled_npz,
        ms=ms,
        xlag_length=int(calibration_args.get("xlag_length", 10)),
    )
    inner_train, inner_validation = inner_training_validation_split(
        data.train_indices,
        split_seed=split_seed,
        validation_fraction=validation_fraction,
    )
    blocks = [
        slice(start, min(data.times.size, start + block_size))
        for start in range(0, data.times.size, block_size)
    ]
    first_horizon = temporal_spec_for_block(data.times, blocks[0], moving=True)
    model_seed = int(calibration_args.get("model_seed", 0))
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    empirical_model = BatchRouteBEmpiricalBayes(
        times=data.times[blocks[0]],
        spatial_inducing=data.spatial_inducing,
        mt=mt,
        representation="analytic_hippo_rff",
        initial_ell_t=float(calibration["learned_theta"]["ell_t"]),
        initial_ell_s=tuple(calibration["learned_theta"]["ell_s"]),
        initial_kernel_variance=float(
            calibration["learned_theta"]["kernel_variance"]
        ),
        initial_noise_std=float(calibration["learned_theta"]["noise_std"]),
        rff_sample_size=int(calibration_args.get("rff_sample_size", 256)),
        seed=model_seed,
        objective_type="finite_dtc",
        temporal_horizon=first_horizon,
    )
    empirical_model.set_theta(calibration["learned_theta"])
    initial_frequencies = (
        empirical_model.temporal.builder.base_frequencies.detach().clone()
    )
    initial_spatial_inducing = empirical_model.spatial_inducing.detach().clone()
    beta_prior_variance = float(
        calibration_args.get("beta_prior_variance", 1000.0)
    )

    online_model = None
    online_state = None
    c_test = None
    static_state_bytes = 0
    previous_basis: slice | None = None
    last_rebase_stop = 0
    rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    process_started = time.perf_counter()

    for block_id, block in enumerate(blocks):
        seen = slice(0, block.stop)
        current_basis = seen
        horizon = temporal_spec_for_block(data.times, current_basis, moving=True)
        apply_rebase = rebase_due(block_id, rebase_interval)
        theta_before = empirical_model.theta()
        eb_summary = {
            "best_iteration": 0,
            "best_validation_nll": float("nan"),
            "total_seconds": 0.0,
            "iteration_seconds_total": 0.0,
            "mean_iteration_seconds": 0.0,
            "validation_seconds_total": 0.0,
        }
        adaptation = None
        if apply_rebase:
            adaptation = adaptation_slice(last_rebase_stop, block)
            fit_data = time_slice_grid(data, adaptation)
            block_trace, eb_summary = optimize_hyperparameters(
                model=empirical_model,
                fit_data=fit_data,
                validation_data=fit_data,
                temporal_horizon=horizon,
                inner_train=inner_train,
                inner_validation=inner_validation,
                steps=eb_steps,
                learning_rate=learning_rate,
                validation_every=validation_every,
                beta_prior_variance=beta_prior_variance,
                prediction_chunk_size=prediction_chunk_size,
                split_seed=split_seed,
                block_id=block_id,
                strategy="periodic_rebase",
                parameter_policy=parameter_policy,
                proximal_lambda=proximal_lambda,
                kernel_variance_lr_multiplier=kernel_variance_lr_multiplier,
                noise_lr_multiplier=noise_lr_multiplier,
                noise_floor=noise_floor,
                nonfinite_gradient_policy=nonfinite_gradient_policy,
            )
            trace.extend(block_trace)

        theta_after = empirical_model.theta()
        theta_step_norm = float(
            np.linalg.norm(
                theta_log_vector(theta_after) - theta_log_vector(theta_before)
            )
        )
        if online_model is None or apply_rebase:
            online_model, _, c_test, static_state_bytes = inference_model(
                empirical_model=empirical_model,
                data=data,
                beta_prior_variance=beta_prior_variance,
            )

        incremental_update_seconds = 0.0
        rebase_rebuild_seconds = 0.0
        if apply_rebase:
            t_seen, kt_seen, _ = temporal_factors_from_model(
                empirical_model=empirical_model,
                all_times=data.times,
                query=seen,
                basis=current_basis,
                old_basis=None,
            )
            train_seen = block_factors(
                data=data,
                query=seen,
                spatial_indices=data.train_indices,
                t_mat=t_seen,
                kt=kt_seen,
                k_on_t=None,
            )
            started = time.perf_counter()
            online_state = online_model.update_block_structured_joint_ssgp_transfer(
                y_vec=train_seen.y_vec,
                Phi=train_seen.Phi,
                T_n=train_seen.T,
                Kt_new=train_seen.Kt,
                state=None,
                K_on_t=None,
            )
            rebase_rebuild_seconds = time.perf_counter() - started
            last_rebase_stop = int(block.stop)
        else:
            t_new, kt_new, k_on_t = temporal_factors_from_model(
                empirical_model=empirical_model,
                all_times=data.times,
                query=block,
                basis=current_basis,
                old_basis=previous_basis,
            )
            new_factors = block_factors(
                data=data,
                query=block,
                spatial_indices=data.train_indices,
                t_mat=t_new,
                kt=kt_new,
                k_on_t=k_on_t,
            )
            started = time.perf_counter()
            online_state = online_model.update_block_structured_joint_ssgp_transfer(
                y_vec=new_factors.y_vec,
                Phi=new_factors.Phi,
                T_n=new_factors.T,
                Kt_new=new_factors.Kt,
                state=online_state,
                K_on_t=new_factors.K_on_t,
            )
            incremental_update_seconds = time.perf_counter() - started

        t_seen, kt_seen, _ = temporal_factors_from_model(
            empirical_model=empirical_model,
            all_times=data.times,
            query=seen,
            basis=current_basis,
            old_basis=None,
        )
        train_seen = block_factors(
            data=data,
            query=seen,
            spatial_indices=data.train_indices,
            t_mat=t_seen,
            kt=kt_seen,
            k_on_t=None,
        )
        reference_started = time.perf_counter()
        batch_state = online_model.update_block_structured_joint_ssgp_transfer(
            y_vec=train_seen.y_vec,
            Phi=train_seen.Phi,
            T_n=train_seen.T,
            Kt_new=train_seen.Kt,
            state=None,
            K_on_t=None,
        )
        batch_reference_seconds = time.perf_counter() - reference_started
        eval_seen = block_factors(
            data=data,
            query=seen,
            spatial_indices=data.test_indices,
            t_mat=t_seen,
            kt=kt_seen,
            k_on_t=None,
        )
        prediction_started = time.perf_counter()
        online_mean, online_variance, _ = vectorized_predict_with_C(
            online_model,
            online_state,
            eval_seen,
            c_test,
            prediction_mode="streaming_sylvester",
            chunk_size=prediction_chunk_size,
            include_conditional_residual_variance=False,
        )
        online_prediction_seconds = time.perf_counter() - prediction_started
        reference_prediction_started = time.perf_counter()
        batch_mean, batch_variance, _ = vectorized_predict_with_C(
            online_model,
            batch_state,
            eval_seen,
            c_test,
            prediction_mode="streaming_sylvester",
            chunk_size=prediction_chunk_size,
            include_conditional_residual_variance=False,
        )
        batch_prediction_seconds = time.perf_counter() - reference_prediction_started
        y_eval = np.asarray(eval_seen.y_vec, dtype=float)
        online_metrics = predictive_metrics(y_eval, online_mean, online_variance)
        batch_metrics = predictive_metrics(y_eval, batch_mean, batch_variance)
        online_state_mean = state_mean(online_state)
        batch_state_mean = state_mean(batch_state)
        relative_state_error = float(
            np.linalg.norm(online_state_mean - batch_state_mean)
            / max(np.linalg.norm(batch_state_mean), 1e-15)
        )
        relative_prediction_error = float(
            np.linalg.norm(online_mean - batch_mean)
            / max(np.linalg.norm(batch_mean), 1e-15)
        )
        bounded_state_bytes = static_state_bytes + object_array_bytes(online_state)
        replay_bytes = history_replay_buffer_bytes(data, int(block.stop))
        row = {
            "split_seed": split_seed,
            "block_id": block_id,
            "block_start": int(block.start),
            "block_stop": int(block.stop),
            "num_seen_times": int(block.stop),
            "rebase_applied": int(apply_rebase),
            "adaptation_start": -1 if adaptation is None else int(adaptation.start),
            "adaptation_stop": -1 if adaptation is None else int(adaptation.stop),
            "adaptation_num_times": 0 if adaptation is None else int(adaptation.stop - adaptation.start),
            "future_times_used_for_eb": 0,
            "best_eb_iteration": int(eb_summary["best_iteration"]),
            "best_validation_nll": float(eb_summary["best_validation_nll"]),
            "eb_total_seconds": float(eb_summary["total_seconds"]),
            "eb_mean_iteration_seconds": float(eb_summary["mean_iteration_seconds"]),
            "incremental_update_seconds": incremental_update_seconds,
            "rebase_rebuild_seconds": rebase_rebuild_seconds,
            "online_prediction_seconds": online_prediction_seconds,
            "batch_reference_seconds_excluded": batch_reference_seconds,
            "batch_prediction_seconds_excluded": batch_prediction_seconds,
            "relative_state_mean_error": relative_state_error,
            "relative_prediction_mean_error": relative_prediction_error,
            "online_rmse": online_metrics["rmse"],
            "online_nll": online_metrics["nll"],
            "online_coverage90": online_metrics["coverage90"],
            "online_ece": online_metrics["ece"],
            "batch_rmse": batch_metrics["rmse"],
            "batch_nll": batch_metrics["nll"],
            "batch_coverage90": batch_metrics["coverage90"],
            "online_minus_batch_rmse": online_metrics["rmse"] - batch_metrics["rmse"],
            "online_minus_batch_nll": online_metrics["nll"] - batch_metrics["nll"],
            "theta_log_step_norm": theta_step_norm,
            "ell_t": float(theta_after["ell_t"]),
            "ell_s_0": float(theta_after["ell_s"][0]),
            "ell_s_1": float(theta_after["ell_s"][1]),
            "kernel_variance": float(theta_after["kernel_variance"]),
            "noise_std": float(theta_after["noise_std"]),
            "bounded_state_bytes": bounded_state_bytes,
            "history_replay_buffer_bytes": replay_bytes,
            "total_persistent_bytes": bounded_state_bytes + replay_bytes,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        previous_basis = current_basis

    torch.testing.assert_close(
        empirical_model.temporal.builder.base_frequencies,
        initial_frequencies,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        empirical_model.spatial_inducing,
        initial_spatial_inducing,
        rtol=0.0,
        atol=0.0,
    )
    seed_dir = outdir / f"seed{split_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, seed_dir / "blockwise.csv")
    if trace:
        write_csv(trace, seed_dir / "training_trace.csv")
    final_validation_started = time.perf_counter()
    final_validation_metrics, _, _ = evaluate(
        empirical_model=empirical_model,
        data=time_slice_grid(data, slice(0, data.times.size)),
        posterior_indices=inner_train,
        evaluation_indices=inner_validation,
        representation="analytic_hippo_rff",
        beta_prior_variance=beta_prior_variance,
        prediction_chunk_size=prediction_chunk_size,
        include_conditional_residual_variance=False,
        collect_pointwise=False,
    )
    process_seconds = time.perf_counter() - process_started
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    final = rows[-1]
    payload = {
        "method": "structured_joint_routeb_periodic_proximal_eb_rebase",
        "strict_bounded_memory": False,
        "causal_no_future_access": True,
        "history_replay_required": True,
        "split_seed": split_seed,
        "mt": mt,
        "ms": ms,
        "block_size": block_size,
        "num_blocks": len(blocks),
        "rebase_interval": rebase_interval,
        "num_rebases": int(sum(row["rebase_applied"] for row in rows)),
        "parameter_policy": parameter_policy,
        "proximal_lambda": proximal_lambda,
        "kernel_variance_lr_multiplier": kernel_variance_lr_multiplier,
        "noise_lr_multiplier": noise_lr_multiplier,
        "noise_floor": noise_floor,
        "nonfinite_gradient_policy": nonfinite_gradient_policy,
        "initial_theta": calibration["learned_theta"],
        "final_theta": empirical_model.theta(),
        "final": final,
        "final_theta_inner_validation_batch_recompute": final_validation_metrics,
        "timing": {
            "process_total_seconds_including_diagnostics": process_seconds,
            "algorithm_seconds": float(
                np.sum(
                    [
                        row["eb_total_seconds"]
                        + row["incremental_update_seconds"]
                        + row["rebase_rebuild_seconds"]
                        + row["online_prediction_seconds"]
                        for row in rows
                    ]
                )
            ),
            "cumulative_eb_seconds": float(np.sum([row["eb_total_seconds"] for row in rows])),
            "cumulative_incremental_update_seconds": float(
                np.sum([row["incremental_update_seconds"] for row in rows])
            ),
            "cumulative_rebase_rebuild_seconds": float(
                np.sum([row["rebase_rebuild_seconds"] for row in rows])
            ),
            "cumulative_online_prediction_seconds": float(
                np.sum([row["online_prediction_seconds"] for row in rows])
            ),
            "diagnostic_batch_reference_seconds_excluded": float(
                np.sum([row["batch_reference_seconds_excluded"] for row in rows])
            ),
            "final_validation_seconds_excluded": time.perf_counter()
            - final_validation_started,
        },
        "resources": {
            "peak_rss_mib_internal": float(peak_rss_kib / 1024.0),
            "bounded_state_mib": float(final["bounded_state_bytes"] / 1024.0**2),
            "history_replay_buffer_mib": float(
                final["history_replay_buffer_bytes"] / 1024.0**2
            ),
            "total_persistent_mib": float(
                final["total_persistent_bytes"] / 1024.0**2
            ),
            "device": "cpu",
            "dtype": "float64",
        },
        "protocol": {
            "task": "ERA5 task_2 variable 0",
            "spatial_split": "800 train / 200 held out",
            "mean": "joint X-lag covariates, L=10",
            "objective": "finite/DTC marginal likelihood",
            "prediction_variance": "strict projected variance; no conditional residual",
            "basis": "cumulative changing HiPPO horizon [t_0,t_k]",
            "supports": "fixed RFF coordinates and fixed spatial inducing locations",
            "adaptation": (
                "proximal EB on observations since the previous rebase; no future "
                "times; inner-spatial validation NLL checkpoint"
            ),
            "between_rebases": "theta frozen; block-to-block conditional posterior transfer",
            "at_rebase": "all-seen posterior rebuild under updated theta",
            "memory_class": (
                "causal periodic history replay, not strict bounded-memory streaming"
            ),
        },
    }
    (seed_dir / "result.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def aggregate(
    *, base: Path, outdir: Path, split_seeds: list[int], args: argparse.Namespace
) -> dict[str, Any]:
    results = [
        json.loads((outdir / f"seed{seed}" / "result.json").read_text(encoding="utf-8"))
        for seed in split_seeds
    ]
    metric_paths = {
        "final_online_rmse": lambda result: result["final"]["online_rmse"],
        "final_online_nll": lambda result: result["final"]["online_nll"],
        "final_online_coverage90": lambda result: result["final"]["online_coverage90"],
        "final_batch_rmse": lambda result: result["final"]["batch_rmse"],
        "final_batch_nll": lambda result: result["final"]["batch_nll"],
        "final_online_minus_batch_rmse": lambda result: result["final"]["online_minus_batch_rmse"],
        "algorithm_seconds": lambda result: result["timing"]["algorithm_seconds"],
        "peak_rss_mib": lambda result: result["resources"]["peak_rss_mib_internal"],
        "bounded_state_mib": lambda result: result["resources"]["bounded_state_mib"],
        "history_replay_buffer_mib": lambda result: result["resources"]["history_replay_buffer_mib"],
    }
    summary: dict[str, Any] = {"n_seeds": len(results)}
    for name, getter in metric_paths.items():
        values = [float(getter(result)) for result in results]
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_sd"] = sample_sd(values)
    block_mean_gaps = []
    max_prediction_errors = []
    for seed in split_seeds:
        rows = np.genfromtxt(
            outdir / f"seed{seed}" / "blockwise.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        block_mean_gaps.append(float(np.mean(np.abs(rows["online_minus_batch_rmse"]))))
        max_prediction_errors.append(float(np.max(rows["relative_prediction_mean_error"])))
    summary["block_mean_absolute_rmse_gap_mean"] = float(np.mean(block_mean_gaps))
    summary["block_mean_absolute_rmse_gap_sd"] = sample_sd(block_mean_gaps)
    summary["max_relative_prediction_error_mean"] = float(np.mean(max_prediction_errors))
    summary["max_relative_prediction_error_sd"] = sample_sd(max_prediction_errors)

    batch_rows = []
    for seed in split_seeds:
        batch_rows.append(batch_reference_metrics(base, seed))
    write_csv(batch_rows, outdir / "task2_batch_eb_reference.csv")
    payload = {
        "experiment": "Route-B periodic proximal-EB rebase",
        "settings": {
            "mt": args.mt,
            "ms": args.ms,
            "block_size": args.block_size,
            "rebase_interval": args.rebase_interval,
            "eb_steps": args.eb_steps,
            "learning_rate": args.learning_rate,
            "parameter_policy": args.parameter_policy,
            "proximal_lambda": args.proximal_lambda,
            "kernel_variance_lr_multiplier": args.kernel_variance_lr_multiplier,
            "noise_lr_multiplier": args.noise_lr_multiplier,
            "noise_floor": args.noise_floor,
            "nonfinite_gradient_policy": args.nonfinite_gradient_policy,
            "split_seeds": split_seeds,
        },
        "summary": summary,
        "task2_full_batch_eb_reference": {
            "rmse_mean": float(np.mean([row["rmse"] for row in batch_rows])),
            "rmse_sd": sample_sd([row["rmse"] for row in batch_rows]),
            "nll_mean": float(np.mean([row["nll"] for row in batch_rows])),
            "nll_sd": sample_sd([row["nll"] for row in batch_rows]),
        },
    }
    (outdir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--rebase-interval", type=int, default=5)
    parser.add_argument("--eb-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--prediction-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--parameter-policy",
        choices=["all", "lengthscales_only"],
        default="lengthscales_only",
    )
    parser.add_argument("--proximal-lambda", type=float, default=0.1)
    parser.add_argument("--kernel-variance-lr-multiplier", type=float, default=0.25)
    parser.add_argument("--noise-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--noise-floor", type=float, default=0.06)
    parser.add_argument(
        "--nonfinite-gradient-policy",
        choices=["reject_step", "zero_parameter"],
        default="zero_parameter",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.eb_steps <= 0:
        raise ValueError("--eb-steps must be positive")
    if args.rebase_interval <= 0:
        raise ValueError("--rebase-interval must be positive")
    base = args.base.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    for split_seed in args.split_seeds:
        result_path = outdir / f"seed{split_seed}" / "result.json"
        if args.skip_existing and result_path.exists():
            print(f"Reusing {result_path}", flush=True)
            continue
        run_seed(
            base=base,
            outdir=outdir,
            data_root=args.root,
            split_seed=split_seed,
            mt=args.mt,
            ms=args.ms,
            block_size=args.block_size,
            rebase_interval=args.rebase_interval,
            eb_steps=args.eb_steps,
            learning_rate=args.learning_rate,
            validation_every=args.validation_every,
            validation_fraction=args.validation_fraction,
            prediction_chunk_size=args.prediction_chunk_size,
            parameter_policy=args.parameter_policy,
            proximal_lambda=args.proximal_lambda,
            kernel_variance_lr_multiplier=args.kernel_variance_lr_multiplier,
            noise_lr_multiplier=args.noise_lr_multiplier,
            noise_floor=args.noise_floor,
            nonfinite_gradient_policy=args.nonfinite_gradient_policy,
        )
    aggregate(base=base, outdir=outdir, split_seeds=args.split_seeds, args=args)


if __name__ == "__main__":
    main()
