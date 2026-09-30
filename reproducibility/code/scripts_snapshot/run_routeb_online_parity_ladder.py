#!/usr/bin/env python3
"""Online Route-B parity ladder against all-seen batch recomputation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hipposvgp_era5_routeb import (
    coverage90,
    ece_gaussian,
    gaussian_nll,
    vectorized_predict_with_C,
)
from scripts.run_routeb_batch_empirical_bayes import load_controlled_grid, object_array_bytes
from stvgp_kronecker.joint_ssgp_kron.kron_utils import solve_spd, vec_f
from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.ssgp_transfer import compute_whitened_orthogonal_Lt
from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    BlockFactors,
    make_analytic_temporal_builder,
    temporal_spec_for_block,
)
from stvgp_kronecker.routeb_empirical_bayes import DTYPE, matern32_separable, robust_cholesky
from stvgp_kronecker.temporal_analytic import TemporalBlockSpec


BASIS_MODES = ("global_fixed", "cumulative_changing", "local_block", "shared_rff_fixed")
DEFAULT_BASIS_MODES = ("global_fixed", "cumulative_changing")
TRANSFER_PROTOCOLS = ("conditional", "whitened_orthogonal", "periodic_rebase")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def basis_slice(mode: str, block: slice, *, total_time: int) -> slice:
    if mode in {"global_fixed", "shared_rff_fixed"}:
        return slice(0, total_time)
    if mode == "cumulative_changing":
        return slice(0, block.stop)
    if mode == "local_block":
        return block
    raise ValueError(mode)


def spatial_projection(
    coordinates: np.ndarray,
    inducing: np.ndarray,
    lengthscales: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    coords = torch.as_tensor(coordinates, dtype=DTYPE)
    z_s = torch.as_tensor(inducing, dtype=DTYPE)
    ell_s = torch.as_tensor(lengthscales, dtype=DTYPE)
    with torch.no_grad():
        ks = matern32_separable(z_s, z_s, ell_s)
        kxs = matern32_separable(coords, z_s, ell_s)
        chol = robust_cholesky(ks)
        c_mat = torch.cholesky_solve(kxs.transpose(0, 1), chol).transpose(0, 1)
        ks = 0.5 * (ks + ks.transpose(0, 1)) + 1e-7 * torch.eye(
            ks.shape[0], dtype=DTYPE
        )
    return np.asarray(ks, dtype=float), np.asarray(c_mat, dtype=float)


def temporal_factors(
    *,
    builder: Any,
    times: np.ndarray,
    query: slice,
    basis: slice,
    old_basis: slice | None,
    basis_mode: str,
    old_temporal_basis: np.ndarray | torch.Tensor | None = None,
    return_temporal_basis: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | tuple[
    np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None
]:
    if basis_mode == "shared_rff_fixed":
        if builder.config.inducing_size % 2:
            raise ValueError("shared_rff_fixed requires an even M_t for sin/cos pairs")
        num_frequencies = builder.config.inducing_size // 2
        with torch.no_grad():
            frequencies = builder.current_frequencies()[0, :num_frequencies]
            query_times = torch.as_tensor(times[query], dtype=frequencies.dtype).reshape(-1, 1)
            scale = torch.sqrt(builder.variance / float(num_frequencies))
            features = scale * torch.cat(
                [torch.sin(query_times * frequencies), torch.cos(query_times * frequencies)],
                dim=1,
            )
        mt = builder.config.inducing_size
        result = (np.asarray(features, dtype=float), np.eye(mt), None)
        return (*result, None) if return_temporal_basis else result

    def make_spec(block: slice) -> TemporalBlockSpec:
        spec = temporal_spec_for_block(times, block, moving=True)
        if basis_mode != "local_block":
            return spec
        # A genuine windowed baseline resets both HiPPO time and Fourier phase.
        # The legacy runner retained ``prev_discrete_steps=block.start`` and was
        # therefore algebraically cumulative despite receiving a local slice.
        return TemporalBlockSpec(
            start=spec.start,
            end=spec.end,
            num_discrete_steps=spec.num_discrete_steps,
            prev_discrete_steps=0,
            phase_origin=spec.start,
        )

    spec = make_spec(basis)
    old_spec = None if old_basis is None else make_spec(old_basis)
    with torch.no_grad():
        kfu_tensor, kt_tensor, k_on_tensor, new_basis_tensor = (
            builder.compute_block_covariances_with_basis(
                times[query],
                spec,
                old_spec,
                old_basis=old_temporal_basis,
            )
        )
        kt = builder.add_jitter(kt_tensor).detach().cpu().numpy()
        kfu = kfu_tensor.detach().cpu().numpy()
        k_on_t = (
            None if k_on_tensor is None else k_on_tensor.detach().cpu().numpy()
        )
        new_temporal_basis = new_basis_tensor.detach().cpu().numpy()
    t_mat = solve_spd(kt, kfu.T, jitter=1e-12).T
    result = (t_mat, kt, k_on_t)
    return (*result, new_temporal_basis) if return_temporal_basis else result


def block_factors(
    *,
    data: Any,
    query: slice,
    spatial_indices: np.ndarray,
    t_mat: np.ndarray,
    kt: np.ndarray,
    k_on_t: np.ndarray | None,
) -> BlockFactors:
    spatial_indices = np.asarray(spatial_indices, dtype=int)
    y_matrix = np.asarray(data.y[query][:, spatial_indices].T, dtype=float)
    phi = np.asarray(data.phi[query][:, spatial_indices, :], dtype=float).reshape(
        -1, data.phi.shape[-1]
    )
    return BlockFactors(
        y_vec=vec_f(y_matrix),
        Phi=phi,
        Y=y_matrix,
        T=t_mat,
        Kt=kt,
        K_on_t=k_on_t,
        block_slice=query,
        inducing_times=np.empty(0, dtype=float),
        temporal_backend="analytic_hippo_rff",
    )


def state_mean(state: Any) -> np.ndarray:
    return np.concatenate([np.asarray(state.beta_mean).reshape(-1), vec_f(state.M_u)])


def predictive_metrics(y: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "nll": gaussian_nll(y, mean, variance),
        "coverage90": coverage90(y, mean, variance),
        "ece": ece_gaussian(y, mean, variance),
        "mean_predictive_std": float(np.mean(np.sqrt(variance))),
    }


def run_configuration(
    *,
    base: Path,
    data_root: str,
    split_seed: int,
    basis_mode: str,
    mt: int,
    ms: int,
    block_size: int,
    prediction_chunk_size: int,
    transfer_protocol: str,
    rebase_interval: int,
    temporal_jitter: float,
) -> list[dict[str, Any]]:
    if transfer_protocol not in TRANSFER_PROTOCOLS:
        raise ValueError(f"Unknown transfer protocol: {transfer_protocol}")
    if basis_mode != "cumulative_changing" and transfer_protocol != "conditional":
        raise ValueError(
            f"{transfer_protocol} is only defined for cumulative_changing, got {basis_mode}"
        )
    calibration_result = (
        base
        / "phase_m_routeb_empirical_bayes"
        / "task1_calibration_then_freeze"
        / "analytic_hippo_rff"
        / f"seed{split_seed}"
        / "result.json"
    )
    calibration = json.loads(calibration_result.read_text(encoding="utf-8"))
    theta = calibration["learned_theta"]
    run_args = calibration["args"]
    controlled = (
        base
        / "phase_d_joint_xlag_controlled"
        / f"seed{split_seed}"
        / f"era5_xlag_seed{split_seed}.npz"
    )
    data = load_controlled_grid(
        root=data_root,
        task="task_2",
        controlled_npz=controlled,
        ms=ms,
        xlag_length=int(run_args.get("xlag_length", 10)),
    )
    ks, c_all = spatial_projection(
        data.coordinates,
        data.spatial_inducing,
        [float(value) for value in theta["ell_s"]],
    )
    c_train = c_all[data.train_indices]
    c_test = c_all[data.test_indices]
    sigma2 = float(theta["noise_std"]) ** 2
    beta_prior_variance = float(run_args.get("beta_prior_variance", 1000.0))
    torch.manual_seed(int(run_args.get("model_seed", 0)))
    builder = make_analytic_temporal_builder(
        mt=mt,
        lengthscale=float(theta["ell_t"]),
        variance=float(theta["kernel_variance"]),
        rff_sample_size=int(run_args.get("rff_sample_size", 256)),
        seed=int(run_args.get("model_seed", 0)),
        jitter=temporal_jitter,
        kernel_type="matern32",
    )
    model = JointSSGPKronHiPPOSVGP(
        Ks=ks,
        C=c_train,
        sigma2=sigma2,
        beta_prior_mean=np.zeros(data.phi.shape[-1]),
        beta_prior_cov=beta_prior_variance * np.eye(data.phi.shape[-1]),
        prior_point_variance=float(theta["kernel_variance"]),
    )
    blocks = [
        slice(start, min(data.times.size, start + block_size))
        for start in range(0, data.times.size, block_size)
    ]
    online_state = None
    previous_basis: slice | None = None
    rows: list[dict[str, Any]] = []
    static_state_bytes = object_array_bytes(
        {"Ks": model.Ks, "Ks_inv": model.Ks_inv, "C": model.C, "builder": builder}
    )

    for block_id, block in enumerate(blocks):
        current_basis = basis_slice(basis_mode, block, total_time=data.times.size)
        old_basis = None if basis_mode == "global_fixed" else previous_basis
        t_new, kt, k_on_t = temporal_factors(
            builder=builder,
            times=data.times,
            query=block,
            basis=current_basis,
            old_basis=old_basis,
            basis_mode=basis_mode,
        )
        new_factors = block_factors(
            data=data,
            query=block,
            spatial_indices=data.train_indices,
            t_mat=t_new,
            kt=kt,
            k_on_t=k_on_t,
        )
        L_t_override = None
        if (
            online_state is not None
            and transfer_protocol == "whitened_orthogonal"
            and k_on_t is not None
        ):
            L_t_override = compute_whitened_orthogonal_Lt(
                online_state.Kt_current,
                k_on_t,
                kt,
                jitter=model.jitter,
            )
        started = time.perf_counter()
        online_state = model.update_block_structured_joint_ssgp_transfer(
            y_vec=new_factors.y_vec,
            Phi=new_factors.Phi,
            T_n=new_factors.T,
            Kt_new=new_factors.Kt,
            state=online_state,
            K_on_t=new_factors.K_on_t,
            L_t_override=L_t_override,
        )
        online_update_seconds = time.perf_counter() - started

        seen = slice(0, block.stop)
        t_seen, kt_seen, _ = temporal_factors(
            builder=builder,
            times=data.times,
            query=seen,
            basis=current_basis,
            old_basis=None,
            basis_mode=basis_mode,
        )
        batch_train_factors = block_factors(
            data=data,
            query=seen,
            spatial_indices=data.train_indices,
            t_mat=t_seen,
            kt=kt_seen,
            k_on_t=None,
        )
        started = time.perf_counter()
        batch_state = model.update_block_structured_joint_ssgp_transfer(
            y_vec=batch_train_factors.y_vec,
            Phi=batch_train_factors.Phi,
            T_n=batch_train_factors.T,
            Kt_new=batch_train_factors.Kt,
            state=None,
            K_on_t=None,
        )
        batch_recompute_seconds = time.perf_counter() - started

        rebase_applied = bool(
            transfer_protocol == "periodic_rebase"
            and (block_id + 1) % max(1, rebase_interval) == 0
            and block_id + 1 < len(blocks)
        )
        rebase_seconds = batch_recompute_seconds if rebase_applied else 0.0
        if rebase_applied:
            online_state = batch_state

        eval_factors = block_factors(
            data=data,
            query=seen,
            spatial_indices=data.test_indices,
            t_mat=t_seen,
            kt=kt_seen,
            k_on_t=None,
        )
        started = time.perf_counter()
        online_mean, online_var, _ = vectorized_predict_with_C(
            model,
            online_state,
            eval_factors,
            c_test,
            prediction_mode="streaming_sylvester",
            chunk_size=prediction_chunk_size,
        )
        online_prediction_seconds = time.perf_counter() - started
        started = time.perf_counter()
        batch_mean, batch_var, _ = vectorized_predict_with_C(
            model,
            batch_state,
            eval_factors,
            c_test,
            prediction_mode="streaming_sylvester",
            chunk_size=prediction_chunk_size,
        )
        batch_prediction_seconds = time.perf_counter() - started
        y_eval = np.asarray(eval_factors.y_vec, dtype=float)
        online_metrics = predictive_metrics(y_eval, online_mean, online_var)
        batch_metrics = predictive_metrics(y_eval, batch_mean, batch_var)

        online_m = state_mean(online_state)
        batch_m = state_mean(batch_state)
        relative_state_mean_error = float(
            np.linalg.norm(online_m - batch_m) / max(np.linalg.norm(batch_m), 1e-15)
        )
        relative_prediction_mean_error = float(
            np.linalg.norm(online_mean - batch_mean)
            / max(np.linalg.norm(batch_mean), 1e-15)
        )
        relative_prediction_variance_error = float(
            np.linalg.norm(online_var - batch_var)
            / max(np.linalg.norm(batch_var), 1e-15)
        )
        row: dict[str, Any] = {
            "basis_mode": basis_mode,
            "transfer_protocol": transfer_protocol,
            "configuration": f"{basis_mode}__{transfer_protocol}",
            "split_seed": split_seed,
            "block_id": block_id,
            "block_start": block.start,
            "block_stop": block.stop,
            "num_seen_times": block.stop,
            "mt": mt,
            "ms": ms,
            "relative_state_mean_error": relative_state_mean_error,
            "relative_prediction_mean_error": relative_prediction_mean_error,
            "relative_prediction_variance_error": relative_prediction_variance_error,
            "online_rmse": online_metrics["rmse"],
            "batch_rmse": batch_metrics["rmse"],
            "online_minus_batch_rmse": online_metrics["rmse"] - batch_metrics["rmse"],
            "online_nll": online_metrics["nll"],
            "batch_nll": batch_metrics["nll"],
            "online_minus_batch_nll": online_metrics["nll"] - batch_metrics["nll"],
            "online_coverage90": online_metrics["coverage90"],
            "batch_coverage90": batch_metrics["coverage90"],
            "online_ece": online_metrics["ece"],
            "batch_ece": batch_metrics["ece"],
            "online_update_seconds": online_update_seconds,
            "batch_recompute_seconds": batch_recompute_seconds,
            "rebase_applied": int(rebase_applied),
            "rebase_seconds": rebase_seconds,
            "online_total_seconds_including_rebase": online_update_seconds + rebase_seconds,
            "online_prediction_seconds": online_prediction_seconds,
            "batch_prediction_seconds": batch_prediction_seconds,
            "dynamic_state_bytes": object_array_bytes(online_state),
            "persistent_state_bytes": static_state_bytes + object_array_bytes(online_state),
            "ell_t": float(theta["ell_t"]),
            "ell_s_0": float(theta["ell_s"][0]),
            "ell_s_1": float(theta["ell_s"][1]),
            "kernel_variance": float(theta["kernel_variance"]),
            "noise_variance": sigma2,
            "calibration_mt": int(run_args.get("mt", -1)),
            "calibration_ms": int(run_args.get("ms", -1)),
            "temporal_jitter": temporal_jitter,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        previous_basis = current_basis

    if basis_mode in {"global_fixed", "shared_rff_fixed"}:
        max_state_error = max(float(row["relative_state_mean_error"]) for row in rows)
        max_prediction_error = max(float(row["relative_prediction_mean_error"]) for row in rows)
        if max_state_error > 1e-6 or max_prediction_error > 1e-8:
            raise RuntimeError(
                "Fixed-basis online accumulation failed parity: "
                f"state={max_state_error:.3e}, prediction={max_prediction_error:.3e}"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--prediction-chunk-size", type=int, default=8192)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--basis-modes",
        nargs="+",
        choices=BASIS_MODES,
        default=list(DEFAULT_BASIS_MODES),
        help=(
            "Basis protocols to run. local_block is an explicit windowed diagnostic and "
            "is intentionally excluded from the default main-method comparison."
        ),
    )
    parser.add_argument(
        "--transfer-protocols",
        nargs="+",
        choices=TRANSFER_PROTOCOLS,
        default=["conditional"],
    )
    parser.add_argument("--rebase-interval", type=int, default=5)
    parser.add_argument(
        "--temporal-jitter",
        type=float,
        default=1e-7,
        help="Temporal K_uu jitter; 1e-7 matches the batch empirical-Bayes runner.",
    )
    args = parser.parse_args()

    base = args.base.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    for split_seed in args.split_seeds:
        for mode in args.basis_modes:
            for transfer_protocol in args.transfer_protocols:
                if mode != "cumulative_changing" and transfer_protocol != "conditional":
                    continue
                rows = run_configuration(
                    base=base,
                    data_root=args.root,
                    split_seed=split_seed,
                    basis_mode=mode,
                    mt=args.mt,
                    ms=args.ms,
                    block_size=args.block_size,
                    prediction_chunk_size=args.prediction_chunk_size,
                    transfer_protocol=transfer_protocol,
                    rebase_interval=args.rebase_interval,
                    temporal_jitter=args.temporal_jitter,
                )
                write_csv(
                    rows,
                    outdir / f"seed{split_seed}_{mode}__{transfer_protocol}.csv",
                )

    all_rows: list[dict[str, Any]] = []
    available_configurations: list[str] = []
    for mode in BASIS_MODES:
        for transfer_protocol in TRANSFER_PROTOCOLS:
            configuration = f"{mode}__{transfer_protocol}"
            mode_found = False
            for split_seed in args.split_seeds:
                path = outdir / f"seed{split_seed}_{configuration}.csv"
                if not path.exists():
                    continue
                with path.open(newline="", encoding="utf-8") as handle:
                    all_rows.extend(csv.DictReader(handle))
                mode_found = True
            if mode_found:
                available_configurations.append(configuration)

    write_csv(all_rows, outdir / "blockwise.csv")
    final_rows = [row for row in all_rows if int(row["num_seen_times"]) == 186]
    summary: list[dict[str, Any]] = []
    metrics = (
        "relative_state_mean_error",
        "relative_prediction_mean_error",
        "relative_prediction_variance_error",
        "online_rmse",
        "batch_rmse",
        "online_minus_batch_rmse",
        "online_nll",
        "batch_nll",
        "online_minus_batch_nll",
        "online_update_seconds",
        "batch_recompute_seconds",
        "rebase_seconds",
        "online_total_seconds_including_rebase",
        "dynamic_state_bytes",
        "persistent_state_bytes",
    )
    for configuration in available_configurations:
        group = [row for row in final_rows if row["configuration"] == configuration]
        item: dict[str, Any] = {
            "configuration": configuration,
            "basis_mode": group[0]["basis_mode"],
            "transfer_protocol": group[0]["transfer_protocol"],
            "n_seeds": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            item[f"final_{metric}_mean"] = float(np.mean(values))
            item[f"final_{metric}_sd"] = sample_sd(values)
        all_mode_rows = [row for row in all_rows if row["configuration"] == configuration]
        item["block_mean_online_update_seconds"] = float(
            np.mean([float(row["online_update_seconds"]) for row in all_mode_rows])
        )
        item["block_mean_batch_recompute_seconds"] = float(
            np.mean([float(row["batch_recompute_seconds"]) for row in all_mode_rows])
        )
        item["block_mean_total_seconds_including_rebase"] = float(
            np.mean([float(row["online_total_seconds_including_rebase"]) for row in all_mode_rows])
        )
        item["total_rebase_seconds"] = float(
            np.sum([float(row["rebase_seconds"]) for row in all_mode_rows])
        )
        item["mean_rebase_seconds_per_seed"] = item["total_rebase_seconds"] / max(
            int(item["n_seeds"]), 1
        )
        item["max_relative_state_mean_error"] = float(
            np.max([float(row["relative_state_mean_error"]) for row in all_mode_rows])
        )
        summary.append(item)
    write_csv(summary, outdir / "summary.csv")
    payload = {
        "protocol": {
            "task": "ERA5 task_2 variable 0",
            "hyperparameters": (
                "Route-B task_1 empirical-Bayes calibration at M_t=M_s=128, "
                "frozen before task_2"
            ),
            "capacity_sweep_control": (
                "all evaluated M_t values reuse the same split-specific M_t=M_s=128 "
                "calibration to isolate representation and transfer effects"
            ),
            "temporal_jitter": args.temporal_jitter,
            "mean": "joint X-lag covariates, L=10",
            "kernel": "Matérn-3/2 temporal and spatial",
            "mt": args.mt,
            "ms": args.ms,
            "block_size": args.block_size,
            "num_blocks": int(np.ceil(186 / args.block_size)),
            "evaluation": "all seen times at 200 spatially held-out locations",
            "predictive_variance": "strict finite/DTC; no conditional residual variance",
            "batch_reference": "recomputed from every seen observation under the current basis",
            "periodic_rebase": (
                f"every {args.rebase_interval} blocks; requires retained all-seen history and is not "
                "strict bounded-history streaming"
            ),
        },
        "summary": summary,
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
