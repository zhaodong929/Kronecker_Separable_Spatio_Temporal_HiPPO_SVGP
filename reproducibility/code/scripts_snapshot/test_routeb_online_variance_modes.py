#!/usr/bin/env python3
"""Fixed-posterior variance diagnostic for long-stream Route-B HiPPO."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import NormalDist
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_iclr_era5_routeb_strict_online import (  # noqa: E402
    TaskPhiCache,
    make_factors,
    temporal_factors_torch,
)
from scripts.run_routeb_online_parity_ladder import spatial_projection  # noqa: E402
from stvgp_kronecker.joint_ssgp_kron.torch_backend import (  # noqa: E402
    TorchJointSSGPKronHiPPOSVGP,
    generalized_eigh,
    inv_spd,
)
from stvgp_kronecker.joint_ssgp_kron.synthetic import (  # noqa: E402
    make_analytic_temporal_builder,
)
from stvgp_kronecker.joint_ssgp_kron.variance_modes import (  # noqa: E402
    JointVarianceTerms,
    VARIANCE_MODES,
    compose_variance_modes,
)


COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
CALIBRATION_LEVELS = tuple(np.linspace(0.10, 0.95, 18))
COLORS = {
    "current_dtc": "#4C78A8",
    "joint_dtc": "#72B7B2",
    "gp_full_conditional": "#F58518",
    "full_joint_conditional": "#C23B3B",
}
PLOT_MODES = (
    "current_dtc",
    "gp_full_conditional",
    "full_joint_conditional",
)
PLOT_LABELS = {
    "current_dtc": "current_dtc = joint_dtc",
    "gp_full_conditional": "gp_full_conditional",
    "full_joint_conditional": "full_joint_conditional",
}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def prediction_metrics(
    y_true: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    mu = np.asarray(mean, dtype=float).reshape(-1)
    var = np.asarray(variance, dtype=float).reshape(-1)
    if not np.all(np.isfinite(var)) or np.any(var <= 0.0):
        raise FloatingPointError("Predictive variance must be finite and positive")
    std = np.sqrt(var)
    result = {
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "nll": float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var))),
        "mean_predictive_std": float(np.mean(std)),
    }
    for level in COVERAGE_LEVELS:
        z_value = NormalDist().inv_cdf(0.5 + level / 2.0)
        result[f"coverage{int(level * 100)}"] = float(
            np.mean(np.abs(y - mu) <= z_value * std)
        )
    z90 = NormalDist().inv_cdf(0.95)
    result["mean_interval_width90"] = float(np.mean(2.0 * z90 * std))
    return result


def calibration_coverage(
    y_true: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    level: float,
) -> float:
    z_value = NormalDist().inv_cdf(0.5 + float(level) / 2.0)
    return float(
        np.mean(
            np.abs(np.asarray(y_true).reshape(-1) - np.asarray(mean).reshape(-1))
            <= z_value * np.sqrt(np.asarray(variance).reshape(-1))
        )
    )


@torch.no_grad()
def joint_variance_terms_with_c(
    *,
    model: TorchJointSSGPKronHiPPOSVGP,
    state: Any,
    t_eval: torch.Tensor,
    phi: np.ndarray,
    c_eval: torch.Tensor,
    chunk_size: int,
) -> tuple[np.ndarray, JointVarianceTerms]:
    """Return an explicit S_bb/S_uu/S_bu decomposition for one block."""

    t_eval = torch.as_tensor(t_eval, device=model.device, dtype=model.dtype)
    phi_tensor = torch.as_tensor(phi, device=model.device, dtype=model.dtype)
    c_eval = torch.as_tensor(c_eval, device=model.device, dtype=model.dtype)
    nt, ns = int(t_eval.shape[0]), int(c_eval.shape[0])
    num_points = nt * ns
    if int(phi_tensor.shape[0]) != num_points:
        raise ValueError("Phi rows do not match the time-space prediction grid")
    if state.R_beta_u is None or state.S_beta_beta is None:
        raise ValueError("The diagnostic requires a structured joint Route-B state")

    kt_inv = inv_spd(state.Kt_current, jitter=model.jitter)
    left_values, p = generalized_eigh(state.G, model.Ks_inv, jitter=model.jitter)
    right_values, q = generalized_eigh(
        state.B_temporal,
        kt_inv,
        jitter=model.jitter,
    )
    denominator = 1.0 + torch.outer(left_values, right_values)
    if bool(torch.any(denominator.abs() < model.jitter)):
        replacement = torch.where(
            denominator >= 0,
            torch.full_like(denominator, model.jitter),
            torch.full_like(denominator, -model.jitter),
        )
        denominator = torch.where(
            denominator.abs() < model.jitter,
            replacement,
            denominator,
        )
    inv_denominator = denominator.reciprocal()
    c_projected_all = c_eval @ p
    t_projected_all = t_eval @ q
    spatial_projected_prior = torch.sum((c_eval @ model.Ks) * c_eval, dim=1)
    temporal_projected_prior = torch.sum(
        (t_eval @ state.Kt_current) * t_eval,
        dim=1,
    )

    d_beta = int(state.R_beta_u.shape[0])
    r_blocks = state.R_beta_u.reshape(d_beta, state.mt, state.ms).transpose(1, 2)
    r_tilde = torch.matmul(p.transpose(0, 1), r_blocks)
    r_tilde = torch.matmul(r_tilde, q)

    mean = torch.empty(num_points, dtype=model.dtype, device=model.device)
    u_conditional = torch.empty_like(mean)
    beta_marginal = torch.empty_like(mean)
    u_beta_coupling = torch.empty_like(mean)
    beta_u_cross = torch.empty_like(mean)
    residual_raw = torch.empty_like(mean)

    for start in range(0, num_points, max(1, int(chunk_size))):
        stop = min(num_points, start + max(1, int(chunk_size)))
        flat = torch.arange(start, stop, device=model.device)
        time_index = torch.div(flat, ns, rounding_mode="floor")
        space_index = torch.remainder(flat, ns)
        c_raw = c_eval[space_index]
        t_raw = t_eval[time_index]
        c_projected = c_projected_all[space_index]
        t_projected = t_projected_all[time_index]
        phi_chunk = phi_tensor[start:stop]

        mean[start:stop] = (
            phi_chunk @ state.beta_mean
            + torch.einsum("bi,ij,bj->b", c_raw, state.M_u, t_raw)
        )
        u_inner = t_projected.square() @ inv_denominator.transpose(0, 1)
        u_conditional[start:stop] = torch.sum(c_projected.square() * u_inner, dim=1)

        h = torch.einsum(
            "bi,dij,bj->bd",
            c_projected,
            r_tilde * inv_denominator,
            t_projected,
        )
        beta_marginal[start:stop] = torch.einsum(
            "bi,ij,bj->b", phi_chunk, state.S_beta_beta, phi_chunk
        )
        u_beta_coupling[start:stop] = torch.einsum(
            "bi,ij,bj->b", h, state.S_beta_beta, h
        )
        beta_u_cross[start:stop] = -2.0 * torch.einsum(
            "bi,ij,bj->b", phi_chunk, state.S_beta_beta, h
        )

        projected_prior = (
            temporal_projected_prior[time_index]
            * spatial_projected_prior[space_index]
        )
        residual_raw[start:stop] = model.prior_point_variance - projected_prior

    to_numpy = lambda value: value.detach().cpu().numpy()  # noqa: E731
    return to_numpy(mean), JointVarianceTerms(
        noise=float(model.sigma2),
        u_conditional=to_numpy(u_conditional),
        beta_marginal=to_numpy(beta_marginal),
        u_beta_coupling=to_numpy(u_beta_coupling),
        beta_u_cross=to_numpy(beta_u_cross),
        conditional_residual_raw=to_numpy(residual_raw),
    )


def summarize_metrics(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    metric_names = (
        "rmse",
        "nll",
        "coverage50",
        "coverage80",
        "coverage90",
        "coverage95",
        "mean_predictive_std",
        "mean_interval_width90",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    output = []
    for group, members in grouped.items():
        item = dict(zip(group_keys, group))
        item["seeds"] = len(members)
        for metric in metric_names:
            values = np.asarray([float(member[metric]) for member in members])
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_sd"] = sample_sd(values)
        output.append(item)
    return output


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    protocol_dir = args.benchmark_root / "protocol" / "task1_10" / f"seed{seed}"
    theta_path = (
        args.benchmark_root
        / "calibration"
        / "routeb_joint_analytic_hippo_rff"
        / f"seed{seed}"
        / "result.json"
    )
    saved_dir = (
        args.benchmark_root
        / "runs"
        / "task1_10"
        / "online"
        / "routeb_analytic_hippo_rff"
        / f"seed{seed}"
    )
    required = (
        protocol_dir / "protocol.npz",
        protocol_dir / "protocol.json",
        theta_path,
        saved_dir / "predictions.npz",
        saved_dir / "result.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing diagnostic inputs:\n" + "\n".join(missing))

    with np.load(protocol_dir / "protocol.npz") as arrays:
        times = np.asarray(arrays["stream_times"], dtype=float)
        y = np.asarray(arrays["stream_y"], dtype=float)
        coordinates = np.asarray(arrays["coordinates"], dtype=float)
        train_indices = np.asarray(arrays["train_indices"], dtype=int)
        test_indices = np.asarray(arrays["test_indices"], dtype=int)
        blocks = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(arrays["block_start"], arrays["block_stop"])
        )
        spatial_inducing = np.asarray(arrays[f"inducing_coords_ms{args.ms}"], dtype=float)
    metadata = json.loads((protocol_dir / "protocol.json").read_text(encoding="utf-8"))
    theta = json.loads(theta_path.read_text(encoding="utf-8"))["learned_theta"]
    with np.load(saved_dir / "predictions.npz") as saved:
        saved_y = np.asarray(saved["y_true"], dtype=float)
        saved_mean = np.asarray(saved["pred_mean"], dtype=float)
        saved_variance = np.asarray(saved["pred_var"], dtype=float)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float64
    torch.manual_seed(0)
    np.random.seed(seed)

    ks, c_all = spatial_projection(coordinates, spatial_inducing, theta["ell_s"])
    c_train = c_all[train_indices]
    c_test = c_all[test_indices]
    model = TorchJointSSGPKronHiPPOSVGP(
        Ks=ks,
        C=c_train,
        sigma2=float(theta["noise_std"]) ** 2,
        beta_prior_mean=np.zeros(int(metadata["xlag"]["features"])),
        beta_prior_cov=args.beta_prior_variance
        * np.eye(int(metadata["xlag"]["features"])),
        prior_point_variance=float(theta["kernel_variance"]),
        device=device,
        dtype=dtype,
    )
    c_test_tensor = torch.as_tensor(c_test, device=device, dtype=dtype)
    temporal_device = torch.device("cpu") if args.temporal_device == "cpu" else device
    builder = make_analytic_temporal_builder(
        mt=args.mt,
        lengthscale=float(theta["ell_t"]),
        variance=float(theta["kernel_variance"]),
        rff_sample_size=args.rff_sample_size,
        seed=0,
        jitter=1e-7,
        kernel_type="matern32",
    ).to(device=temporal_device, dtype=dtype)
    phi_cache = TaskPhiCache(args.data_root, y, int(metadata["xlag"]["length"]))

    state = None
    previous_basis = None
    previous_temporal_basis = None
    y_chunks: list[np.ndarray] = []
    mean_chunks: list[np.ndarray] = []
    task_chunks: list[np.ndarray] = []
    variance_chunks = {mode: [] for mode in VARIANCE_MODES}
    residual_sum = 0.0
    residual_count = 0
    residual_min = float("inf")
    residual_max = float("-inf")
    residual_negative_count = 0
    max_mean_difference = 0.0
    max_current_variance_difference = 0.0
    max_a_b_difference = 0.0

    for block_id, block in enumerate(blocks):
        phi_block, task_index = phi_cache.block(block)
        basis = slice(0, block.stop)
        t_mat, kt, k_on_t, new_temporal_basis = temporal_factors_torch(
            builder=builder,
            times=times,
            query=block,
            basis=basis,
            old_basis=previous_basis,
            old_temporal_basis=previous_temporal_basis,
            device=temporal_device,
            dtype=dtype,
        )
        previous_basis = basis
        previous_temporal_basis = new_temporal_basis
        t_model = t_mat.to(device=device, dtype=dtype)
        kt_model = kt.to(device=device, dtype=dtype)
        k_on_model = None if k_on_t is None else k_on_t.to(device=device, dtype=dtype)
        train_factors = make_factors(
            y[block],
            phi_block,
            train_indices,
            t_model,
            kt_model,
            k_on_model,
            block,
            "analytic_hippo_rff",
        )
        test_factors = make_factors(
            y[block],
            phi_block,
            test_indices,
            t_model,
            kt_model,
            None,
            block,
            "analytic_hippo_rff",
        )
        with torch.no_grad():
            state = model.update_block_structured_joint_ssgp_transfer(
                y_vec=train_factors.y_vec,
                Phi=train_factors.Phi,
                T_n=train_factors.T,
                Kt_new=train_factors.Kt,
                state=state,
                K_on_t=train_factors.K_on_t,
            )
            reconstructed_mean, terms = joint_variance_terms_with_c(
                model=model,
                state=state,
                t_eval=test_factors.T,
                phi=test_factors.Phi,
                c_eval=c_test_tensor,
                chunk_size=args.prediction_chunk_size,
            )
        mode_variances = compose_variance_modes(
            terms,
            negative_tolerance=args.negative_residual_tolerance,
        )

        saved_block_mean = saved_mean[block].reshape(-1)
        saved_block_variance = saved_variance[block].reshape(-1)
        saved_block_y = saved_y[block].reshape(-1)
        expected_y = np.asarray(test_factors.y_vec, dtype=float).reshape(-1)
        np.testing.assert_allclose(expected_y, saved_block_y, atol=0.0, rtol=0.0)
        max_mean_difference = max(
            max_mean_difference,
            float(np.max(np.abs(reconstructed_mean - saved_block_mean))),
        )
        max_current_variance_difference = max(
            max_current_variance_difference,
            float(
                np.max(
                    np.abs(mode_variances["current_dtc"] - saved_block_variance)
                )
            ),
        )
        max_a_b_difference = max(
            max_a_b_difference,
            float(
                np.max(
                    np.abs(
                        mode_variances["current_dtc"]
                        - mode_variances["joint_dtc"]
                    )
                )
            ),
        )
        if max_mean_difference > args.reproduction_atol:
            raise AssertionError(
                f"Seed {seed} block {block_id}: reconstructed mean differs from saved "
                f"mean by {max_mean_difference:.3e}"
            )
        if max_current_variance_difference > args.reproduction_atol:
            raise AssertionError(
                f"Seed {seed} block {block_id}: current DTC variance differs from saved "
                f"variance by {max_current_variance_difference:.3e}"
            )
        if max_a_b_difference > 1e-9:
            raise AssertionError(f"A/B joint variance mismatch: {max_a_b_difference:.3e}")

        residual_raw = np.asarray(terms.conditional_residual_raw)
        residual_sum += float(np.sum(residual_raw))
        residual_count += int(residual_raw.size)
        residual_min = min(residual_min, float(np.min(residual_raw)))
        residual_max = max(residual_max, float(np.max(residual_raw)))
        residual_negative_count += int(np.count_nonzero(residual_raw < 0.0))
        y_chunks.append(saved_block_y)
        mean_chunks.append(saved_block_mean)
        task_chunks.append(np.full(saved_block_y.shape, task_index, dtype=int))
        for mode in VARIANCE_MODES:
            variance_chunks[mode].append(mode_variances[mode])
        print(
            json.dumps(
                {
                    "seed": seed,
                    "block": block_id,
                    "task": task_index,
                    "mean_residual": float(np.mean(residual_raw)),
                    "min_residual": float(np.min(residual_raw)),
                    "max_mean_diff": max_mean_difference,
                }
            ),
            flush=True,
        )

    y_all = np.concatenate(y_chunks)
    mean_all = np.concatenate(mean_chunks)
    task_all = np.concatenate(task_chunks)
    variances = {
        mode: np.concatenate(chunks) for mode, chunks in variance_chunks.items()
    }
    per_seed_rows = []
    per_task_seed_rows = []
    calibration_rows = []
    for mode in VARIANCE_MODES:
        per_seed_rows.append(
            {"seed": seed, "scope": "task2_10", "mode": mode, **prediction_metrics(y_all, mean_all, variances[mode])}
        )
        for task_index in range(2, 11):
            selected = task_all == task_index
            per_task_seed_rows.append(
                {
                    "seed": seed,
                    "task": f"task_{task_index}",
                    "mode": mode,
                    **prediction_metrics(
                        y_all[selected], mean_all[selected], variances[mode][selected]
                    ),
                }
            )
        for level in CALIBRATION_LEVELS:
            calibration_rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "nominal_coverage": level,
                    "empirical_coverage": calibration_coverage(
                        y_all, mean_all, variances[mode], level
                    ),
                }
            )
    return {
        "per_seed_rows": per_seed_rows,
        "per_task_seed_rows": per_task_seed_rows,
        "calibration_rows": calibration_rows,
        "residual": {
            "seed": seed,
            "mean": residual_sum / residual_count,
            "minimum": residual_min,
            "maximum": residual_max,
            "negative_count": residual_negative_count,
            "count": residual_count,
        },
        "validation": {
            "seed": seed,
            "max_saved_mean_abs_difference": max_mean_difference,
            "max_saved_current_variance_abs_difference": max_current_variance_difference,
            "max_current_vs_joint_variance_abs_difference": max_a_b_difference,
        },
    }


def plot_calibration(rows: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot([0, 1], [0, 1], color="black", linewidth=1.0, linestyle="--", label="Ideal")
    for mode in PLOT_MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        levels = sorted({float(row["nominal_coverage"]) for row in mode_rows})
        empirical = [
            np.mean(
                [
                    float(row["empirical_coverage"])
                    for row in mode_rows
                    if float(row["nominal_coverage"]) == level
                ]
            )
            for level in levels
        ]
        ax.plot(
            levels,
            empirical,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=COLORS[mode],
            label=PLOT_LABELS[mode],
        )
    ax.set(xlabel="Nominal coverage", ylabel="Empirical coverage", xlim=(0.08, 0.97), ylim=(0.08, 0.97))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_task_metric(
    task_rows: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    output: Path,
    reference: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for mode in PLOT_MODES:
        rows = sorted(
            [row for row in task_rows if row["mode"] == mode],
            key=lambda row: int(str(row["task"]).split("_")[-1]),
        )
        tasks = [int(str(row["task"]).split("_")[-1]) for row in rows]
        ax.plot(
            tasks,
            [float(row[f"{metric}_mean"]) for row in rows],
            marker="o",
            linewidth=1.6,
            color=COLORS[mode],
            label=PLOT_LABELS[mode],
        )
    if reference is not None:
        ax.axhline(reference, color="black", linewidth=1.0, linestyle="--", label="Nominal 0.90")
    ax.set(xlabel="Streaming task", ylabel=ylabel, xticks=range(2, 11))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def format_metric(row: dict[str, Any], metric: str) -> str:
    return f"{float(row[f'{metric}_mean']):.4f} +/- {float(row[f'{metric}_sd']):.4f}"


def write_report(
    output: Path,
    overall: list[dict[str, Any]],
    per_task: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    command: str,
) -> None:
    by_mode = {row["mode"]: row for row in overall}
    task_lookup = {(row["task"], row["mode"]): row for row in per_task}
    current = by_mode["current_dtc"]
    full = by_mode["full_joint_conditional"]
    gap = 0.9 - float(current["coverage90_mean"])
    closed = (
        (float(full["coverage90_mean"]) - float(current["coverage90_mean"])) / gap
        if gap > 0
        else 0.0
    )
    if abs(float(full["coverage90_mean"]) - 0.9) <= 0.03 and float(full["nll_mean"]) < float(current["nll_mean"]):
        verdict = "The omitted conditional residual largely explains the low long-stream coverage."
    elif closed >= 0.5:
        verdict = "The omitted conditional residual explains part, but not all, of the low long-stream coverage."
    else:
        verdict = "The omitted conditional residual is not the main explanation for the low long-stream coverage."

    lines = [
        "# Route-B task1_10 fixed-posterior variance diagnostic",
        "",
        "## What the original implementation contains",
        "",
        "The original `current_dtc` variance already contains observation noise, the conditional GP-state uncertainty, beta uncertainty, the beta-u cross term, and the extra GP marginal uncertainty induced by beta-u coupling. The latter three are evaluated in the equivalent Schur form `(phi - R D_u^-1 a)^T S_beta_beta (phi - R D_u^-1 a)`. Therefore modes A and B are mathematically identical; their numerical agreement is a correctness check, not a new method improvement.",
        "",
        "The only omitted GP term is the interdomain conditional residual `k_xx - K_xu K_uu^-1 K_ux`. Observation noise is added exactly once in every mode.",
        "",
        "## Overall results",
        "",
        "| Mode | RMSE | NLL | Cov50 | Cov80 | Cov90 | Cov95 | Mean std | Mean width90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in VARIANCE_MODES:
        row = by_mode[mode]
        lines.append(
            "| {mode} | {rmse} | {nll} | {c50} | {c80} | {c90} | {c95} | {std} | {width} |".format(
                mode=mode,
                rmse=format_metric(row, "rmse"),
                nll=format_metric(row, "nll"),
                c50=format_metric(row, "coverage50"),
                c80=format_metric(row, "coverage80"),
                c90=format_metric(row, "coverage90"),
                c95=format_metric(row, "coverage95"),
                std=format_metric(row, "mean_predictive_std"),
                width=format_metric(row, "mean_interval_width90"),
            )
        )
    residual_count = sum(int(row["count"]) for row in residual_rows)
    residual_mean = sum(float(row["mean"]) * int(row["count"]) for row in residual_rows) / residual_count
    lines.extend(
        [
            "",
            "## Conditional residual diagnostics",
            "",
            f"- Mean: `{residual_mean:.8g}`",
            f"- Minimum: `{min(float(row['minimum']) for row in residual_rows):.8g}`",
            f"- Maximum: `{max(float(row['maximum']) for row in residual_rows):.8g}`",
            f"- Negative count: `{sum(int(row['negative_count']) for row in residual_rows)}` / `{residual_count}`",
            "",
            "## Interpretation",
            "",
            f"Mode D changes Coverage90 by `{float(full['coverage90_mean']) - float(current['coverage90_mean']):+.4f}` and NLL by `{float(full['nll_mean']) - float(current['nll_mean']):+.4f}` relative to the current DTC variance. It closes approximately `{100.0 * closed:.1f}%` of the gap from the current coverage to 0.90.",
            "",
            verdict,
            "",
            "The aggregate correction is not uniformly calibrated over time. For `full_joint_conditional`, Coverage90 is "
            f"`{float(task_lookup[('task_2', 'full_joint_conditional')]['coverage90_mean']):.4f}` on Task 2 but only "
            f"`{float(task_lookup[('task_10', 'full_joint_conditional')]['coverage90_mean']):.4f}` on Task 10. Thus the residual fixes most of the aggregate DTC under-dispersion, while later-task drift, frozen Task-1 hyperparameters, or finite-state/transfer error still require separate treatment.",
            "",
            "RMSE is identical across all modes because the saved predictive means are reused unchanged. This is a fixed-posterior diagnostic. It does not retrain a VFE/full-SVGP objective and does not show what hyperparameters would be learned under that objective.",
            "",
            "## Validation",
            "",
            f"- Largest reconstructed-vs-saved mean difference: `{max(float(row['max_saved_mean_abs_difference']) for row in validation_rows):.3e}`",
            f"- Largest reconstructed-vs-saved current variance difference: `{max(float(row['max_saved_current_variance_abs_difference']) for row in validation_rows):.3e}`",
            f"- Largest A-vs-B variance difference: `{max(float(row['max_current_vs_joint_variance_abs_difference']) for row in validation_rows):.3e}`",
            "- Every variance was checked to be finite and positive.",
            "",
            "## Command",
            "",
            "```bash",
            command,
            "```",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_benchmark = os.environ.get("BENCHMARK_ROOT")
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path(default_benchmark) if default_benchmark else None,
        required=default_benchmark is None,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data/era5/processed_timeseries_4_task1_10_extension",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/diagnostics/routeb_task1_10_variance",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temporal-device", choices=["cpu", "solver"], default="cpu")
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--prediction-chunk-size", type=int, default=8192)
    parser.add_argument("--beta-prior-variance", type=float, default=1000.0)
    parser.add_argument("--negative-residual-tolerance", type=float, default=1e-8)
    parser.add_argument("--reproduction-atol", type=float, default=2e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_seed_rows: list[dict[str, Any]] = []
    all_task_seed_rows: list[dict[str, Any]] = []
    all_calibration_rows: list[dict[str, Any]] = []
    residual_rows = []
    validation_rows = []
    for seed in args.seeds:
        result = run_seed(args, seed)
        all_seed_rows.extend(result["per_seed_rows"])
        all_task_seed_rows.extend(result["per_task_seed_rows"])
        all_calibration_rows.extend(result["calibration_rows"])
        residual_rows.append(result["residual"])
        validation_rows.append(result["validation"])

    overall = summarize_metrics(all_seed_rows, ("mode",))
    per_task = summarize_metrics(all_task_seed_rows, ("task", "mode"))
    overall.sort(key=lambda row: VARIANCE_MODES.index(str(row["mode"])))
    per_task.sort(
        key=lambda row: (
            int(str(row["task"]).split("_")[-1]),
            VARIANCE_MODES.index(str(row["mode"])),
        )
    )
    write_csv(overall, args.output_dir / "metrics_overall.csv")
    write_csv(per_task, args.output_dir / "metrics_per_task.csv")
    write_csv(all_seed_rows, args.output_dir / "metrics_per_seed.csv")
    plot_calibration(all_calibration_rows, args.output_dir / "calibration_curve.png")
    plot_task_metric(
        per_task,
        metric="mean_predictive_std",
        ylabel="Mean predictive standard deviation",
        output=args.output_dir / "predictive_std_by_task.png",
    )
    plot_task_metric(
        per_task,
        metric="coverage90",
        ylabel="Coverage90",
        output=args.output_dir / "coverage90_by_task.png",
        reference=0.9,
    )

    by_mode = {row["mode"]: row for row in overall}
    if sorted(args.seeds) == [0, 1, 2, 3, 4]:
        expected = {
            "rmse_mean": 0.1388,
            "nll_mean": -0.1804,
            "coverage90_mean": 0.7468,
        }
        for metric, target in expected.items():
            actual = float(by_mode["current_dtc"][metric])
            if abs(actual - target) > 5e-4:
                raise AssertionError(
                    f"Mode A failed the saved-result check for {metric}: {actual} vs {target}"
                )
    for mode in VARIANCE_MODES[1:]:
        np.testing.assert_allclose(
            float(by_mode[mode]["rmse_mean"]),
            float(by_mode["current_dtc"]["rmse_mean"]),
            rtol=0.0,
            atol=1e-12,
        )

    command = " ".join(sys.argv)
    write_report(
        args.output_dir / "diagnostic_report.md",
        overall,
        per_task,
        residual_rows,
        validation_rows,
        command,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
