#!/usr/bin/env python3
"""Run a causal online-hyperparameter adaptation ablation for COVID Route B.

The fixed Route-B archives remain the primary comparison.  This runner adds a
matched diagnostic in which Task-1 empirical-Bayes parameters are warm-started
and updated every few online weeks.  The parameter objective uses only the 42
current-visible jurisdictions; delayed hidden labels are used only by the
posterior replay.  Therefore the current hidden labels are never used before
the prediction is written.

Because changing theta invalidates the old sufficient statistics, the
posterior is causally rebuilt from legal observations after each online week.
This is an adaptation ablation, not a bounded-memory runtime baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.covid_long_setting_b.archive import PredictionArchive
from baselines.covid_long_setting_b.protocol import COVIDSettingBProtocol
from scripts.compute_covid_long_target_likelihood_crps import normal_crps
from scripts.compute_covid_long_target_likelihood_ece import (
    ECE_COVERAGE_LEVELS,
    empirical_normal_intervals,
    interval_calibration,
)
from scripts.run_iclr_era5_routeb_strict_online import (
    make_factors,
    matern32_1d,
)
from scripts.run_routeb_online_parity_ladder import spatial_projection, temporal_factors
from stvgp_kronecker.joint_ssgp_kron.kron_utils import solve_spd
from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    make_analytic_temporal_builder,
    temporal_spec_for_block,
)
from stvgp_kronecker.joint_ssgp_kron.torch_backend import TorchJointSSGPKronHiPPOSVGP
from stvgp_kronecker.routeb_empirical_bayes import BatchRouteBEmpiricalBayes


REPRESENTATIONS = {
    "ordinary": {
        "routeb_name": "routeb_ordinary",
        "kernel": "matern32",
        "spectral_mixture": None,
        "rff_sample_size": 256,
    },
    "cumulative": {
        "routeb_name": "routeb_cumulative",
        "kernel": "spectral_mixture",
        "spectral_mixture": "configs/covid_sm_q2.json",
        # Match the formal frozen Route-B calibration archive.
        "rff_sample_size": 256,
    },
}


def read_spectral_mixture(path: Path | None) -> dict[str, tuple[float, ...]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: tuple(float(value) for value in payload[key])
        for key in ("weights", "means", "scales")
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def theta_log_vector(theta: dict[str, object]) -> np.ndarray:
    return np.log(
        np.asarray(
            [
                float(theta["ell_t"]),
                *[float(value) for value in theta["ell_s"]],
                float(theta["kernel_variance"]),
                float(theta["noise_std"]),
            ],
            dtype=np.float64,
        )
    )


def project_theta_to_reference(
    theta: dict[str, object],
    reference: dict[str, object],
    max_log_deviation: float,
) -> dict[str, object]:
    """Keep adaptation in a multiplicative trust region around Task-1 EB."""

    if max_log_deviation <= 0.0:
        raise ValueError("max_log_deviation must be positive")

    def project(value: float, anchor: float) -> float:
        log_ratio = np.clip(
            np.log(float(value) / float(anchor)),
            -float(max_log_deviation),
            float(max_log_deviation),
        )
        return float(float(anchor) * np.exp(log_ratio))

    return {
        "ell_t": project(theta["ell_t"], reference["ell_t"]),
        "ell_s": [
            project(value, anchor)
            for value, anchor in zip(theta["ell_s"], reference["ell_s"])
        ],
        "kernel_variance": project(
            theta["kernel_variance"], reference["kernel_variance"]
        ),
        "noise_std": project(theta["noise_std"], reference["noise_std"]),
    }


def common_metrics(
    y_true_standardized: np.ndarray,
    pred_mean_standardized: np.ndarray,
    pred_var_standardized: np.ndarray,
    metadata: dict[str, Any],
    *,
    seed: int,
) -> dict[str, float]:
    standardization = metadata["target_standardization"]
    scale = float(standardization["scale"])
    offset = float(standardization["mean"])
    truth = np.asarray(y_true_standardized, dtype=np.float64) * scale + offset
    mean = np.asarray(pred_mean_standardized, dtype=np.float64) * scale + offset
    variance = np.maximum(np.asarray(pred_var_standardized, dtype=np.float64) * scale**2, 1e-12)
    lower, upper = empirical_normal_intervals(
        mean, variance, samples=100, seed=1_000_000 + int(seed)
    )
    ece, _, _ = interval_calibration(truth, lower, upper, ECE_COVERAGE_LEVELS)
    error = truth - mean
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "crps": float(np.mean(normal_crps(truth, mean, variance))),
        "gaussian_nlpd": float(
            np.mean(0.5 * (np.log(2.0 * np.pi * variance) + error**2 / variance))
        ),
        "ece": float(ece),
        "coverage90": float(
            np.mean(np.abs(error) <= 1.6448536269514722 * np.sqrt(variance))
        ),
    }


def make_eb_model(
    *,
    full_times: np.ndarray,
    inducing: np.ndarray,
    theta: dict[str, object],
    representation: str,
    spectral_mixture: dict[str, tuple[float, ...]] | None,
    rff_sample_size: int,
    mt: int,
) -> BatchRouteBEmpiricalBayes:
    info = REPRESENTATIONS[representation]
    horizon = temporal_spec_for_block(
        full_times, slice(0, full_times.size), moving=True
    )
    model = BatchRouteBEmpiricalBayes(
        times=full_times,
        spatial_inducing=inducing,
        mt=mt,
        representation="analytic_hippo_rff" if representation == "cumulative" else "inducing_points",
        initial_ell_t=float(theta["ell_t"]),
        initial_ell_s=tuple(float(value) for value in theta["ell_s"]),
        initial_kernel_variance=float(theta["kernel_variance"]),
        initial_noise_std=float(theta["noise_std"]),
        rff_sample_size=rff_sample_size,
        seed=0,
        objective_type="finite_dtc",
        temporal_horizon=horizon,
        temporal_kernel=info["kernel"],
        spectral_mixture=spectral_mixture,
    )
    model.set_theta(theta)
    return model


def adapt_theta(
    *,
    full_times: np.ndarray,
    prefix_end: int,
    y: np.ndarray,
    phi: np.ndarray,
    coordinates_all: np.ndarray,
    visible: np.ndarray,
    inducing: np.ndarray,
    theta: dict[str, object],
    reference_theta: dict[str, object],
    representation: str,
    spectral_mixture: dict[str, tuple[float, ...]] | None,
    rff_sample_size: int,
    mt: int,
    steps: int,
    learning_rate: float,
    proximal_lambda: float,
    max_log_deviation: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Update theta using only visible states through the current week."""

    model = make_eb_model(
        full_times=full_times,
        inducing=inducing,
        theta=theta,
        representation=representation,
        spectral_mixture=spectral_mixture,
        rff_sample_size=rff_sample_size,
        mt=mt,
    )
    query_times = full_times[:prefix_end]
    horizon = temporal_spec_for_block(
        full_times, slice(0, prefix_end), moving=True
    )
    model.set_temporal_query(query_times, temporal_horizon=horizon)
    y_train = torch.as_tensor(
        y[:prefix_end, visible].T, dtype=torch.float64
    )
    phi_train = torch.as_tensor(
        phi[:prefix_end, visible].transpose(1, 0, 2), dtype=torch.float64
    )
    coordinates = torch.as_tensor(
        np.asarray(coordinates_all, dtype=np.float64)[visible], dtype=torch.float64
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    anchors = [parameter.detach().clone() for parameter in model.parameters()]
    trace: list[dict[str, object]] = []
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        diagnostics = model.objective(
            y_matrix=y_train,
            phi_tensor=phi_train,
            spatial_coordinates=coordinates,
            beta_prior_variance=1000.0,
        )
        penalty = torch.zeros((), dtype=torch.float64)
        for parameter, anchor in zip(model.parameters(), anchors):
            penalty = penalty + torch.sum((parameter - anchor) ** 2)
        loss = diagnostics.nlml_per_observation + 0.5 * float(proximal_lambda) * penalty
        if not bool(torch.isfinite(loss)):
            trace.append({"step": step, "finite": False, "reason": "nonfinite_objective"})
            break
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not gradients_finite:
            trace.append({"step": step, "finite": False, "reason": "nonfinite_gradient"})
            break
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=10.0)
        )
        optimizer.step()
        model.clamp_parameters()
        model.set_theta(
            project_theta_to_reference(
                model.theta(), reference_theta, max_log_deviation
            )
        )
        trace.append(
            {
                "step": step,
                "finite": True,
                "objective": float(loss.detach()),
                "nlml_per_observation": float(diagnostics.nlml_per_observation.detach()),
                "gradient_norm": gradient_norm,
                **model.theta(),
            }
        )
    return model.theta(), trace


def ordinary_temporal_factors(
    *, full_times: np.ndarray, query: slice, theta: dict[str, object], mt: int
) -> tuple[np.ndarray, np.ndarray]:
    z_t = np.linspace(float(full_times.min()), float(full_times.max()), int(mt))
    kt = matern32_1d(
        z_t, z_t, float(theta["ell_t"]), float(theta["kernel_variance"])
    )
    kt = 0.5 * (kt + kt.T) + 1e-7 * np.eye(int(mt))
    kfu = matern32_1d(
        full_times[query], z_t, float(theta["ell_t"]), float(theta["kernel_variance"])
    )
    return solve_spd(kt, kfu.T, jitter=1e-12).T, kt


def build_state(
    *,
    full_times: np.ndarray,
    prefix_end: int,
    combined_y: np.ndarray,
    combined_phi: np.ndarray,
    visible: np.ndarray,
    hidden: np.ndarray,
    coordinates: np.ndarray,
    inducing: np.ndarray,
    theta: dict[str, object],
    representation: str,
    spectral_mixture: dict[str, tuple[float, ...]] | None,
    rff_sample_size: int,
    mt: int,
) -> tuple[TorchJointSSGPKronHiPPOSVGP, object, np.ndarray, np.ndarray, dict[str, object]]:
    ks, c_all = spatial_projection(
        coordinates, inducing, [float(value) for value in theta["ell_s"]]
    )
    c_train = c_all[visible]
    c_test = c_all[hidden]
    state_model = TorchJointSSGPKronHiPPOSVGP(
        Ks=ks,
        C=c_train,
        sigma2=float(theta["noise_std"]) ** 2,
        beta_prior_mean=np.zeros(combined_phi.shape[-1]),
        beta_prior_cov=1000.0 * np.eye(combined_phi.shape[-1]),
        prior_point_variance=float(theta["kernel_variance"]),
        device="cpu",
        dtype=torch.float64,
    )
    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    basis = slice(0, prefix_end)
    builder = None
    if representation == "cumulative":
        builder = make_analytic_temporal_builder(
            mt=mt,
            lengthscale=float(theta["ell_t"]),
            variance=float(theta["kernel_variance"]),
            rff_sample_size=rff_sample_size,
            seed=0,
            jitter=1e-7,
            kernel_type="spectral_mixture",
            spectral_mixture_weights=spectral_mixture["weights"] if spectral_mixture else None,
            spectral_mixture_means=spectral_mixture["means"] if spectral_mixture else None,
            spectral_mixture_scales=spectral_mixture["scales"] if spectral_mixture else None,
        )

    def factors(query: slice) -> tuple[np.ndarray, np.ndarray]:
        key = (int(query.start), int(query.stop))
        if key not in cache:
            if representation == "cumulative":
                t_mat, kt, _ = temporal_factors(
                    builder=builder,
                    times=full_times,
                    query=query,
                    basis=basis,
                    old_basis=None,
                    basis_mode="cumulative_changing",
                )
            else:
                t_mat, kt = ordinary_temporal_factors(
                    full_times=full_times, query=query, theta=theta, mt=mt
                )
            cache[key] = (np.asarray(t_mat, dtype=np.float64), np.asarray(kt, dtype=np.float64))
        return cache[key]

    def update(
        state: object | None,
        row_index: int,
        locations: np.ndarray,
        c_observed: np.ndarray,
    ) -> object:
        t_mat, kt = factors(slice(row_index, row_index + 1))
        factors_block = make_factors(
            combined_y[row_index : row_index + 1],
            combined_phi[row_index : row_index + 1],
            locations,
            t_mat,
            kt,
            None,
            slice(0, 1),
            "analytic_hippo_rff" if representation == "cumulative" else "inducing_points",
        )
        return state_model.update_block_structured_joint_ssgp_transfer(
            y_vec=factors_block.y_vec,
            Phi=factors_block.Phi,
            T_n=factors_block.T,
            Kt_new=factors_block.Kt,
            state=state,
            K_on_t=None,
            C_observed=c_observed,
        )

    task_t, task_kt = factors(slice(0, 52))
    task_factors = make_factors(
        combined_y[:52],
        combined_phi[:52],
        visible,
        task_t,
        task_kt,
        None,
        slice(0, 52),
        "analytic_hippo_rff" if representation == "cumulative" else "inducing_points",
    )
    state = state_model.update_block_structured_joint_ssgp_transfer(
        y_vec=task_factors.y_vec,
        Phi=task_factors.Phi,
        T_n=task_factors.T,
        Kt_new=task_factors.Kt,
        state=None,
        K_on_t=None,
        C_observed=c_train,
    )
    stream_weeks = prefix_end - 52
    for week in range(stream_weeks):
        if week > 0:
            state = update(state, 52 + week - 1, hidden, c_test)
        state = update(state, 52 + week, visible, c_train)
    return state_model, state, c_test, factors(slice(52 + stream_weeks - 1, 53 + stream_weeks - 1))[0], {"state_rows": int(task_factors.y_vec.size)}


def run_seed(args: argparse.Namespace) -> dict[str, object]:
    protocol_path = args.protocol_root / f"seed{args.seed}" / "protocol.npz"
    protocol = COVIDSettingBProtocol(protocol_path)
    metadata = protocol.metadata
    with np.load(protocol_path, allow_pickle=False) as arrays:
        calibration_y = np.asarray(arrays["calibration_y"], dtype=np.float64)
        stream_y = np.asarray(arrays["stream_y"], dtype=np.float64)
        calibration_phi = np.asarray(arrays["calibration_phi"], dtype=np.float64)
        stream_phi = np.asarray(arrays["stream_phi"], dtype=np.float64)
        coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
        inducing = np.asarray(arrays[f"inducing_coords_ms{args.ms}"], dtype=np.float64)
        calibration_times = np.asarray(arrays["calibration_times"], dtype=np.float64)
    time_step = float(np.median(np.diff(calibration_times)))
    online_times = calibration_times[-1] + time_step * np.arange(1, protocol.online_weeks + 1)
    full_times = np.concatenate([calibration_times, online_times])
    combined_y = np.vstack([calibration_y, stream_y])
    combined_phi = np.concatenate([calibration_phi, stream_phi], axis=0)
    visible = protocol.visible_locations
    hidden = protocol.hidden_locations
    representation_info = REPRESENTATIONS[args.representation]
    spectral_mixture = read_spectral_mixture(
        ROOT / representation_info["spectral_mixture"]
        if representation_info["spectral_mixture"]
        else None
    )
    calibration_path = (
        ROOT
        / "results/diagnostics/covid_long_stream_2020_2024_mandatory"
        / f"seed{args.seed}"
        / representation_info["routeb_name"]
        / "calibration/result.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    theta: dict[str, object] = calibration["learned_theta"]
    reference_theta: dict[str, object] = json.loads(
        json.dumps(calibration["learned_theta"])
    )
    output = args.output_root / f"seed{args.seed}" / args.representation
    output.mkdir(parents=True, exist_ok=True)
    archive = PredictionArchive(
        protocol, method=f"routeb_{args.representation}_online_adaptive", seed=args.seed
    )
    rows: list[dict[str, object]] = []
    theta_updates: list[dict[str, object]] = []
    started = time.perf_counter()
    current_hidden_reads = 0
    delayed_hidden_rows = 0
    current_visible_rows = 0
    for week in range(protocol.online_weeks):
        prefix_end = 52 + week + 1
        should_adapt = week == 0 or ((week + 1) % int(args.adapt_every) == 0)
        theta_before = dict(theta)
        adaptation_trace: list[dict[str, object]] = []
        adaptation_seconds = 0.0
        if should_adapt:
            adaptation_started = time.perf_counter()
            theta, adaptation_trace = adapt_theta(
                full_times=full_times,
                prefix_end=prefix_end,
                y=combined_y,
                phi=combined_phi,
                coordinates_all=coordinates,
                visible=visible,
                inducing=inducing,
                theta=theta,
                reference_theta=reference_theta,
                representation=args.representation,
                spectral_mixture=spectral_mixture,
                rff_sample_size=int(representation_info["rff_sample_size"]),
                mt=args.mt,
                steps=args.adapt_steps,
                learning_rate=args.learning_rate,
                proximal_lambda=args.proximal_lambda,
                max_log_deviation=args.max_log_deviation,
            )
            adaptation_seconds = time.perf_counter() - adaptation_started
            theta_updates.append(
                {
                    "stream_week": week,
                    "global_week": 52 + week,
                    "steps": len(adaptation_trace),
                    "seconds": adaptation_seconds,
                    "theta_before": theta_before,
                    "theta_after": theta,
                    "trace": adaptation_trace,
                }
            )
        rebuild_started = time.perf_counter()
        state_model, state, c_test, t_eval, state_info = build_state(
            full_times=full_times,
            prefix_end=prefix_end,
            combined_y=combined_y,
            combined_phi=combined_phi,
            visible=visible,
            hidden=hidden,
            coordinates=coordinates,
            inducing=inducing,
            theta=theta,
            representation=args.representation,
            spectral_mixture=spectral_mixture,
            rff_sample_size=int(representation_info["rff_sample_size"]),
            mt=args.mt,
        )
        rebuild_seconds = time.perf_counter() - rebuild_started
        row_index = 52 + week
        mean, variance, diagnostics = state_model.predict_with_C(
            state=state,
            T_eval=t_eval,
            Phi=combined_phi[row_index, hidden].reshape(-1, combined_phi.shape[-1]),
            C_eval=c_test,
            chunk_size=8192,
            include_conditional_residual_variance=True,
            validate_conditional_residual_variance=True,
        )
        adaptation_rejected = False
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(variance).all()
            or np.any(variance <= 0.0)
            or not np.isfinite(float(diagnostics["avg_beta_schur_term"]))
            or float(diagnostics["avg_beta_schur_term"]) > args.max_beta_schur_term
        ) and should_adapt:
            # A candidate hyperparameter update must not poison the stream.
            # Revert to the last accepted theta and rebuild causally.
            adaptation_rejected = True
            theta = theta_before
            state_model, state, c_test, t_eval, state_info = build_state(
                full_times=full_times,
                prefix_end=prefix_end,
                combined_y=combined_y,
                combined_phi=combined_phi,
                visible=visible,
                hidden=hidden,
                coordinates=coordinates,
                inducing=inducing,
                theta=theta,
                representation=args.representation,
                spectral_mixture=spectral_mixture,
                rff_sample_size=int(representation_info["rff_sample_size"]),
                mt=args.mt,
            )
            mean, variance, diagnostics = state_model.predict_with_C(
                state=state,
                T_eval=t_eval,
                Phi=combined_phi[row_index, hidden].reshape(-1, combined_phi.shape[-1]),
                C_eval=c_test,
                chunk_size=8192,
                include_conditional_residual_variance=True,
                validate_conditional_residual_variance=True,
            )
        info = protocol.week(week)
        archive.append(info, mean, variance)
        current_visible_rows += int(visible.size)
        if week > 0:
            delayed_hidden_rows += int(hidden.size)
        scale = float(metadata["target_standardization"]["scale"])
        offset = float(metadata["target_standardization"]["mean"])
        truth = stream_y[week, hidden] * scale + offset
        point_metrics = common_metrics(
            stream_y[week : week + 1, hidden],
            mean.reshape(1, -1),
            variance.reshape(1, -1),
            metadata,
            seed=args.seed + week,
        )
        rows.append(
            {
                "stream_week": week,
                "global_week": 52 + week,
                "adapted": int(should_adapt),
                "adaptation_steps": len(adaptation_trace),
                "adaptation_rejected": int(adaptation_rejected),
                "adaptation_seconds": adaptation_seconds,
                "posterior_rebuild_seconds": rebuild_seconds,
                "theta_log_step_norm": float(
                    np.linalg.norm(theta_log_vector(theta) - theta_log_vector(theta_before))
                ),
                "current_hidden_reads": 0,
                "delayed_hidden_rows_total": delayed_hidden_rows,
                "current_visible_rows_total": current_visible_rows,
                "state_rows": state_info["state_rows"],
                **point_metrics,
                **{f"predictive_{key}": value for key, value in diagnostics.items()},
                **theta,
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    prediction_path = output / "predictions.npz"
    audit = archive.write(
        prediction_path,
        extra_metadata={
            "representation": args.representation,
            "mt": args.mt,
            "ms": args.ms,
            "parameter_adaptation": "causal visible-only EB update with posterior replay",
            "adapt_every": args.adapt_every,
            "adapt_steps": args.adapt_steps,
            "learning_rate": args.learning_rate,
            "proximal_lambda": args.proximal_lambda,
            "max_log_deviation": args.max_log_deviation,
            "max_beta_schur_term": args.max_beta_schur_term,
            "current_hidden_labels_used_for_adaptation": False,
            "current_hidden_labels_used_for_prediction": False,
        },
    )
    with np.load(prediction_path, allow_pickle=False) as archive_arrays:
        final_metrics = common_metrics(
            np.asarray(archive_arrays["y_true"]),
            np.asarray(archive_arrays["pred_mean"]),
            np.asarray(archive_arrays["pred_var"]),
            metadata,
            seed=args.seed,
        )
    write_csv(rows, output / "blockwise.csv")
    payload: dict[str, object] = {
        "status": "complete",
        "method": f"Route B {args.representation} online adaptive",
        "seed": args.seed,
        "protocol": "Task-1 EB initialization; causal parameter adaptation; delayed hidden labels and current visible labels only",
        "representation": args.representation,
        "capacity": {"mt": args.mt, "ms": args.ms},
        "initial_theta": calibration["learned_theta"],
        "final_theta": theta,
        "parameter_adaptation": {
            "updated_parameters": ["ell_t", "ell_s", "kernel_variance", "noise_std"],
            "frequency": f"every {args.adapt_every} online weeks, including week 0",
            "steps_per_update": args.adapt_steps,
            "learning_rate": args.learning_rate,
            "proximal_lambda": args.proximal_lambda,
            "fit_information": "current-visible 42 jurisdictions through the current week",
            "posterior_information": "Task-1 visible observations, all prior delayed hidden labels, and current visible observations",
        },
        "overall_current_block": final_metrics,
        "theta_updates": theta_updates,
        "audit": {
            **audit,
            "current_hidden_reads": current_hidden_reads,
            "delayed_hidden_rows": delayed_hidden_rows,
            "expected_delayed_hidden_rows": (protocol.online_weeks - 1) * int(hidden.size),
            "current_visible_rows": current_visible_rows,
            "expected_current_visible_rows": protocol.online_weeks * int(visible.size),
            "hidden_predictions": protocol.online_weeks * int(hidden.size),
            "passed": current_hidden_reads == 0 and delayed_hidden_rows == (protocol.online_weeks - 1) * int(hidden.size),
        },
        "timing": {
            "process_total_seconds": time.perf_counter() - started,
            "adaptation_seconds_total": float(sum(float(item["seconds"]) for item in theta_updates)),
            "posterior_rebuild_seconds_total": float(sum(float(row["posterior_rebuild_seconds"]) for row in rows)),
        },
        "artifacts": {
            "predictions": str(prediction_path),
            "blockwise": str(output / "blockwise.csv"),
            "calibration": str(calibration_path),
        },
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/diagnostics/covid_long_routeb_online_adaptive"),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--representation", choices=tuple(REPRESENTATIONS), required=True)
    parser.add_argument("--mt", type=int, default=32)
    parser.add_argument("--ms", type=int, default=32)
    parser.add_argument("--adapt-every", type=int, default=4)
    parser.add_argument("--adapt-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--proximal-lambda", type=float, default=0.1)
    parser.add_argument(
        "--max-log-deviation",
        type=float,
        default=0.10,
        help="Multiplicative trust-region radius around Task-1 EB parameters.",
    )
    parser.add_argument(
        "--max-beta-schur-term",
        type=float,
        default=10.0,
        help="Reject an update if the predictive beta Schur term exceeds this bound.",
    )
    args = parser.parse_args()
    args.protocol_root = (ROOT / args.protocol_root).resolve()
    args.output_root = (ROOT / args.output_root).resolve()
    if args.adapt_every < 1 or args.adapt_steps < 1:
        raise ValueError("adapt-every and adapt-steps must be positive")
    if args.max_log_deviation <= 0.0 or args.max_beta_schur_term <= 0.0:
        raise ValueError("trust-region and Schur-term bounds must be positive")
    if args.mt != 32 or args.ms != 32:
        raise ValueError("The matched adaptive ablation is fixed at Mt=Ms=32")
    result = run_seed(args)
    print(json.dumps({"status": result["status"], "metrics": result["overall_current_block"], "output": result["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
