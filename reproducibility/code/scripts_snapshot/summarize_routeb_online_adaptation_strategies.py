#!/usr/bin/env python3
"""Aggregate Route-B online hyperparameter adaptation experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def log_theta(theta: dict[str, Any]) -> np.ndarray:
    return np.log(
        np.asarray(
            [
                float(theta["ell_t"]),
                *[float(value) for value in theta["ell_s"]],
                float(theta["kernel_variance"]),
                float(theta["noise_std"]),
            ]
        )
    )


def summarize_values(
    row: dict[str, Any], name: str, values: list[float]
) -> None:
    row[f"{name}_mean"] = float(np.mean(values))
    row[f"{name}_sd"] = sample_sd(values)
    row[f"{name}_median"] = float(np.median(values))


def summarize_replay_strategy(
    *,
    label: str,
    directory: Path,
    memory_class: str,
    parameter_policy: str,
    timing_note: str = "",
) -> dict[str, Any]:
    results = [
        read_json(directory / "new_block" / f"seed{seed}" / "result.json")
        for seed in (0, 1, 2)
    ]
    block_rows = [
        read_csv(directory / "new_block" / f"seed{seed}" / "blockwise.csv")
        for seed in (0, 1, 2)
    ]
    row: dict[str, Any] = {
        "method": label,
        "memory_class": memory_class,
        "parameter_policy": parameter_policy,
        "posterior_protocol": "all-seen posterior refresh after every theta decision",
        "timing_note": timing_note,
    }
    summarize_values(row, "rmse", [float(result["final"]["rmse"]) for result in results])
    summarize_values(row, "nll", [float(result["final"]["nll"]) for result in results])
    summarize_values(
        row,
        "coverage90",
        [float(result["final"]["coverage90"]) for result in results],
    )
    algorithm_seconds = [
        float(
            np.sum(
                [
                    float(item["eb_total_seconds"])
                    + float(item["posterior_refresh_seconds"])
                    + float(item["prediction_seconds"])
                    for item in rows
                ]
            )
        )
        for rows in block_rows
    ]
    summarize_values(row, "algorithm_seconds", algorithm_seconds)
    summarize_values(
        row,
        "peak_rss_mib",
        [float(result["resources"]["peak_rss_mib_internal"]) for result in results],
    )
    summarize_values(
        row,
        "bounded_state_mib",
        [
            float(result["resources"]["persistent_state_mib"])
            for result in results
        ],
    )
    summarize_values(
        row,
        "endpoint_log_theta_drift",
        [
            float(
                np.linalg.norm(
                    log_theta(result["final_theta"])
                    - log_theta(result["initial_theta"])
                )
            )
            for result in results
        ],
    )
    summarize_values(
        row,
        "cumulative_log_theta_drift",
        [
            float(np.sum([float(item["theta_log_step_norm"]) for item in rows]))
            for rows in block_rows
        ],
    )
    row["history_replay_buffer_mib_mean"] = float("nan")
    row["history_replay_buffer_mib_sd"] = float("nan")
    return row


def summarize_frozen_online(base: Path) -> dict[str, Any]:
    directory = base / "phase_r_online_hippo_capacity_dtc" / "mt128_ms128"
    seed_rows = [
        read_csv(directory / f"seed{seed}_cumulative_changing__conditional.csv")
        for seed in (0, 1, 2)
    ]
    finals = [rows[-1] for rows in seed_rows]
    row: dict[str, Any] = {
        "method": "Task-1 freeze",
        "memory_class": "strict bounded-state online",
        "parameter_policy": "Task-1 Route-B EB theta frozen on Task 2",
        "posterior_protocol": "cumulative changing-basis block transfer",
        "timing_note": "Task-1 calibration is amortized and excluded",
    }
    for name, column in (
        ("rmse", "online_rmse"),
        ("nll", "online_nll"),
        ("coverage90", "online_coverage90"),
    ):
        summarize_values(row, name, [float(item[column]) for item in finals])
    summarize_values(
        row,
        "algorithm_seconds",
        [
            float(
                np.sum(
                    [
                        float(item["online_total_seconds_including_rebase"])
                        + float(item["online_prediction_seconds"])
                        for item in rows
                    ]
                )
            )
            for rows in seed_rows
        ],
    )
    summarize_values(
        row,
        "bounded_state_mib",
        [float(item["persistent_state_bytes"]) / 1024.0**2 for item in finals],
    )
    for name in ("endpoint_log_theta_drift", "cumulative_log_theta_drift"):
        summarize_values(row, name, [0.0, 0.0, 0.0])
    row["peak_rss_mib_mean"] = float("nan")
    row["peak_rss_mib_sd"] = float("nan")
    row["peak_rss_mib_median"] = float("nan")
    row["history_replay_buffer_mib_mean"] = 0.0
    row["history_replay_buffer_mib_sd"] = 0.0
    return row


def summarize_periodic(base: Path) -> dict[str, Any]:
    directory = (
        base
        / "phase_t_proximal_online_eb"
        / "periodic_rebase5_lengthscales_lambda0p1_lr0p005_woodbury"
    )
    results = [read_json(directory / f"seed{seed}" / "result.json") for seed in (0, 1, 2)]
    block_rows = [read_csv(directory / f"seed{seed}" / "blockwise.csv") for seed in (0, 1, 2)]
    row: dict[str, Any] = {
        "method": "Periodic rebase-5 lengthscale EB",
        "memory_class": "causal periodic history replay",
        "parameter_policy": "proximal lengthscale-only EB every 5 blocks",
        "posterior_protocol": "online transfer between all-seen rebases",
        "timing_note": "batch-reference diagnostics excluded",
    }
    for name, key in (
        ("rmse", "online_rmse"),
        ("nll", "online_nll"),
        ("coverage90", "online_coverage90"),
    ):
        summarize_values(row, name, [float(result["final"][key]) for result in results])
    summarize_values(
        row,
        "algorithm_seconds",
        [float(result["timing"]["algorithm_seconds"]) for result in results],
    )
    for name, key in (
        ("peak_rss_mib", "peak_rss_mib_internal"),
        ("bounded_state_mib", "bounded_state_mib"),
        ("history_replay_buffer_mib", "history_replay_buffer_mib"),
    ):
        summarize_values(row, name, [float(result["resources"][key]) for result in results])
    summarize_values(
        row,
        "endpoint_log_theta_drift",
        [
            float(
                np.linalg.norm(
                    log_theta(result["final_theta"])
                    - log_theta(result["initial_theta"])
                )
            )
            for result in results
        ],
    )
    summarize_values(
        row,
        "cumulative_log_theta_drift",
        [
            float(np.sum([float(item["theta_log_step_norm"]) for item in rows]))
            for rows in block_rows
        ],
    )
    row["final_online_minus_batch_rmse_mean"] = float(
        np.mean([result["final"]["online_minus_batch_rmse"] for result in results])
    )
    return row


def summarize_batch(base: Path) -> dict[str, Any]:
    directory = base / "phase_t_proximal_online_eb" / "task2_batch_eb_strict_dtc_reevaluation"
    strict = read_json(directory / "summary.json")
    source_results = [
        read_json(
            base
            / "phase_m_routeb_empirical_bayes"
            / "task2_empirical_bayes"
            / "analytic_hippo_rff"
            / f"seed{seed}"
            / "result.json"
        )
        for seed in (0, 1, 2)
    ]
    row: dict[str, Any] = {
        "method": "Full Task-2 batch EB",
        "memory_class": "full-history batch",
        "parameter_policy": "all five theta parameters, 100 Adam steps",
        "posterior_protocol": "single full-history posterior",
        "timing_note": "training plus strict-DTC final evaluation",
    }
    for name in ("rmse", "nll", "coverage90"):
        row[f"{name}_mean"] = float(strict[f"{name}_mean"])
        row[f"{name}_sd"] = float(strict[f"{name}_sd"])
        row[f"{name}_median"] = float(
            np.median(
                [
                    read_json(directory / f"seed{seed}" / "result.json")["metrics"][name]
                    for seed in (0, 1, 2)
                ]
            )
        )
    algorithm_seconds = [
        float(result["timing"]["training_total_seconds"])
        + float(
            read_json(directory / f"seed{seed}" / "result.json")["evaluation_seconds"]
        )
        for seed, result in enumerate(source_results)
    ]
    summarize_values(row, "algorithm_seconds", algorithm_seconds)
    summarize_values(
        row,
        "peak_rss_mib",
        [float(result["resources"]["peak_rss_mib_internal"]) for result in source_results],
    )
    summarize_values(
        row,
        "bounded_state_mib",
        [float(result["resources"]["persistent_model_state_mib"]) for result in source_results],
    )
    summarize_values(
        row,
        "endpoint_log_theta_drift",
        [
            float(
                np.linalg.norm(
                    log_theta(result["learned_theta"])
                    - log_theta(result["initial_theta"])
                )
            )
            for result in source_results
        ],
    )
    row["cumulative_log_theta_drift_mean"] = float("nan")
    row["cumulative_log_theta_drift_sd"] = float("nan")
    row["cumulative_log_theta_drift_median"] = float("nan")
    row["history_replay_buffer_mib_mean"] = float("nan")
    row["history_replay_buffer_mib_sd"] = float("nan")
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Strategy | Class | RMSE | NLL | Coverage90 | Runtime median (s) | Endpoint drift |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {memory_class} | {rmse_mean:.4f} +/- {rmse_sd:.4f} | "
            "{nll_mean:.4f} +/- {nll_sd:.4f} | {coverage90_mean:.4f} +/- "
            "{coverage90_sd:.4f} | {algorithm_seconds_median:.1f} | "
            "{endpoint_log_theta_drift_mean:.3f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [
        summarize_frozen_online(base),
        summarize_replay_strategy(
            label="Aggressive full new-block EB",
            directory=base / "phase_s_online_incremental_eb" / "strict_new_block_steps5",
            memory_class="causal every-block history replay",
            parameter_policy="all five theta parameters",
        ),
        summarize_replay_strategy(
            label="New-block EB + all-seen validation safeguard",
            directory=base
            / "phase_s_online_incremental_eb"
            / "pilot_steps5_allseen_checkpoint",
            memory_class="causal every-block history replay",
            parameter_policy="all five theta parameters",
        ),
        summarize_replay_strategy(
            label="Lengthscale-only proximal EB",
            directory=base / "phase_t_proximal_online_eb" / "lengthscales_lambda0p1",
            memory_class="causal every-block history replay",
            parameter_policy="ell_t and ell_s only; lambda=0.1",
        ),
        summarize_replay_strategy(
            label="Conservative full proximal EB",
            directory=base
            / "phase_t_proximal_online_eb"
            / "full_lambda0p1_varlr0p5_noiselr0p1",
            memory_class="causal every-block history replay",
            parameter_policy="all theta; reduced variance/noise learning rates",
            timing_note="seed 0 timing contaminated by concurrent pilot; use median only",
        ),
        summarize_periodic(base),
        summarize_batch(base),
    ]
    write_csv(rows, outdir / "strategy_comparison.csv")
    (outdir / "strategy_comparison.json").write_text(
        json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8"
    )
    table = markdown_table(rows)
    (outdir / "strategy_comparison.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
