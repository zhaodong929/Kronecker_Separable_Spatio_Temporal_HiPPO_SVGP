from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stvgp_kronecker.joint_ssgp_kron.kron_utils import (
    dense_Du_for_tests,
    inv_spd,
    make_spd_matrix,
    solve_Du_sylvester,
)

OUTDIR = ROOT / "results" / "routeB_experiment_report"
PLOTDIR = OUTDIR / "plots"
TABLEDIR = OUTDIR / "tables"


METHOD_LABELS = {
    "no_transfer": "no_transfer",
    "mean_field_ssgp_transfer": "mean-field",
    "structured_joint_ssgp_transfer": "Route B",
    "projected_prior": "projected_prior",
    "ssgp_transfer": "ssgp_transfer",
    "dense_reference_fixed_basis": "dense reference",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def latex_escape(text: object) -> str:
    s = str(text)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def metric_cell(summary: dict[str, Any], metric: str, digits: int = 4) -> str:
    item = summary.get(metric, {})
    mean = item.get("mean", float("nan"))
    se = item.get("se", float("nan"))
    if not math.isfinite(mean):
        return "NA"
    if math.isfinite(se):
        return f"{mean:.{digits}f} $\\pm$ {se:.{digits}f}"
    return f"{mean:.{digits}f}"


def seed_summary(report: dict[str, Any], method: str, eval_mode: str) -> dict[str, Any]:
    key = f"method={method}|eval_mode={eval_mode}"
    return report["seed_level_summary_by_method_eval_mode"][key]


def table_to_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(x) for x in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latex_table(header: list[str], rows: list[list[object]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    def cell(value: object) -> str:
        text = str(value)
        if "$" in text:
            return text
        return latex_escape(text)

    lines = [r"\begin{tabular}{" + align + r"}", r"\toprule"]
    lines.append(" & ".join(latex_escape(h) for h in header) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(cell(x) for x in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def sylvester_dense_validation_rows() -> list[list[object]]:
    """Small numerical benchmark for the structured Du Sylvester solve.

    This table is intentionally a solver validation only. The dense solve is
    the numerical reference for the same linear system D_u z = q; it is not a
    model or statistical baseline.
    """

    rows: list[list[object]] = []
    for mt, ms, seed in [(8, 6, 102), (20, 12, 201), (40, 25, 203)]:
        rng = np.random.default_rng(seed)
        Kt = make_spd_matrix(mt, seed=seed)
        Ks = make_spd_matrix(ms, seed=seed + 10)
        Kt_inv = inv_spd(Kt, jitter=0.0)
        Ks_inv = inv_spd(Ks, jitter=0.0)
        B = make_spd_matrix(mt, seed=seed + 20)
        G = make_spd_matrix(ms, seed=seed + 30)
        q = rng.normal(size=mt * ms)
        D_u = dense_Du_for_tests(Kt_inv, Ks_inv, B, G)

        start = time.perf_counter()
        z_syl = solve_Du_sylvester(Kt_inv, Ks_inv, B, G, q, jitter=0.0)
        syl_time = time.perf_counter() - start

        start = time.perf_counter()
        z_dense = np.linalg.solve(D_u, q)
        dense_time = time.perf_counter() - start

        rel_solution_error = np.linalg.norm(z_syl - z_dense) / max(np.linalg.norm(z_dense), 1e-12)
        rel_residual = np.linalg.norm(D_u @ z_syl - q) / max(np.linalg.norm(q), 1e-12)
        dense_memory_mb = D_u.nbytes / (1024.0 * 1024.0)
        speedup = dense_time / max(syl_time, 1e-12)
        rows.append(
            [
                f"{mt} x {ms}",
                mt * ms,
                f"{rel_solution_error:.3e}",
                f"{rel_residual:.3e}",
                f"{syl_time * 1e3:.3f}",
                f"{dense_time * 1e3:.3f}",
                f"{speedup:.2f}x",
                f"{dense_memory_mb:.3f}",
            ]
        )
    return rows


def _mean_se(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), float(std / np.sqrt(arr.size))


def _metric_pair(summary_rows: list[dict[str, str]], metric: str, prefix: str) -> str:
    mean, se = _mean_se([float(r[f"{prefix}_{metric}"]) for r in summary_rows])
    return f"{mean:.4f} $\\pm$ {se:.4f}"


def report_metric_cell(report: dict[str, Any], method: str, eval_mode: str, metric: str, digits: int = 4) -> str:
    key = f"method={method}|eval_mode={eval_mode}"
    seed_summary_by_mode = report.get("seed_level_summary_by_method_eval_mode", {})
    if key in seed_summary_by_mode and metric in seed_summary_by_mode[key]:
        item = seed_summary_by_mode[key][metric]
        mean = item.get("mean", float("nan"))
        se = item.get("se", float("nan"))
        if math.isfinite(mean) and math.isfinite(se):
            return f"{mean:.{digits}f} $\\pm$ {se:.{digits}f}"
    vals = report["mean_by_method_eval_mode"][f"{method}|{eval_mode}"]
    return f"{vals[metric]:.{digits}f}"


def summarize_selected_values(metrics_path: Path) -> tuple[list[list[object]], list[list[object]]]:
    rows = read_csv_dicts(metrics_path)
    by_seed: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("eval_mode") == "seen_history" and row.get("method") == "no_transfer":
            by_seed.setdefault(row["seed"], row)
    selected_rows: list[list[object]] = []
    for seed, row in sorted(by_seed.items(), key=lambda item: int(item[0])):
        selected_rows.append(
            [
                row["regime_name"],
                seed,
                row.get("data_ell_t", ""),
                row.get("selected_ell_t", row.get("fitted_ell_t", "")),
                row.get("selected_candidate_score", ""),
            ]
        )
    regime_rows: list[list[object]] = []
    by_regime: dict[str, list[dict[str, str]]] = {}
    for row in by_seed.values():
        by_regime.setdefault(row["regime_name"], []).append(row)
    for regime, selected in sorted(by_regime.items()):
        counts = Counter(row.get("selected_ell_t", row.get("fitted_ell_t", "")) for row in selected)
        example = selected[0]
        selected_values = ", ".join(f"{value} ({count})" for value, count in sorted(counts.items(), key=lambda item: float(item[0])))
        initial_task = f"{example.get('initial_task_blocks', '')} blocks; fraction {example.get('initial_task_fraction', '')}"
        regime_rows.append(
            [
                regime,
                example.get("data_ell_t", ""),
                "model ell_t",
                selected_values,
                initial_task,
                f"coupling={example.get('beta_u_correlation_design', '')}; noise=0.08; kernel var=1.0",
            ]
        )
    return regime_rows, selected_rows


def routeb_all_metric_win(row: dict[str, str]) -> bool:
    return (
        float(row["routeB_rmse"]) < float(row["mean_field_rmse"])
        and float(row["routeB_nll"]) < float(row["mean_field_nll"])
        and float(row["routeB_cov90"]) > float(row["mean_field_cov90"])
        and float(row["routeB_rmse_forgetting"]) < float(row["mean_field_rmse_forgetting"])
        and float(row["routeB_nll_forgetting"]) < float(row["mean_field_nll_forgetting"])
    )


def plot_standard_confirmatory(summary_rows: list[dict[str, str]]) -> None:
    methods = [("mean_field", "mean-field"), ("routeB", "Route B")]
    metrics = [
        ("rmse", "RMSE"),
        ("nll", "NLL"),
        ("cov90", "90% coverage"),
        ("rmse_forgetting", "RMSE forgetting"),
        ("nll_forgetting", "NLL forgetting"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.2))
    for ax, (metric, title) in zip(axes, metrics):
        means = []
        ses = []
        for prefix, _ in methods:
            mean, se = _mean_se([float(r[f"{prefix}_{metric}"]) for r in summary_rows])
            means.append(mean)
            ses.append(se)
        xs = np.arange(len(methods))
        ax.bar(xs, means, yerr=ses, capsize=3, color=["#66c2a5", "#fc8d62"])
        ax.set_xticks(xs)
        ax.set_xticklabels([label for _, label in methods], rotation=30, ha="right")
        ax.set_title(title)
        if metric == "cov90":
            ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
    fig.suptitle(f"Standard diagnostic ablation, seen history (mean +/- SE over {len(summary_rows)} medium/strong configurations)")
    fig.tight_layout()
    fig.savefig(PLOTDIR / "standard_confirmatory_seen_history.png", dpi=200)
    plt.close(fig)


def collect_ablation_rows() -> list[dict[str, Any]]:
    rows = []
    base = ROOT / "results" / "experiments_routeB_long_memory_ablation"
    for report_path in sorted(base.glob("mt_*_bs_*_ell_*_noise_*/joint_ssgp_kron_synthetic_report.json")):
        match = re.search(r"mt_(\d+)_bs_(\d+)_ell_([0-9.]+)_noise_([0-9.]+)", str(report_path))
        if not match:
            continue
        mt, block_size, ell_t, noise = match.groups()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for method in ["no_transfer", "mean_field_ssgp_transfer", "structured_joint_ssgp_transfer"]:
            s = seed_summary(report, method, "seen_history")
            row = {
                "mt": int(mt),
                "block_size": int(block_size),
                "ell_t": float(ell_t),
                "noise": float(noise),
                "method": method,
            }
            for metric in ["rmse", "nll", "coverage90", "rmse_forgetting", "nll_forgetting", "avg_predictive_variance", "avg_interval_width90"]:
                row[f"{metric}_mean"] = s[metric]["mean"]
                row[f"{metric}_se"] = s[metric]["se"]
            rows.append(row)
    return rows


def plot_long_memory_ablation(rows: list[dict[str, Any]]) -> None:
    methods = ["no_transfer", "mean_field_ssgp_transfer", "structured_joint_ssgp_transfer"]
    colors = {"no_transfer": "#8da0cb", "mean_field_ssgp_transfer": "#66c2a5", "structured_joint_ssgp_transfer": "#fc8d62"}
    for metric, ylabel in [("rmse", "Seen-history RMSE"), ("nll", "Seen-history NLL"), ("avg_predictive_variance", "Avg predictive variance")]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        for ax, ell_t in zip(axes[0], [0.5, 0.8]):
            for method in methods:
                subset = [r for r in rows if r["block_size"] == 5 and r["ell_t"] == ell_t and r["noise"] == 0.08 and r["method"] == method]
                subset = sorted(subset, key=lambda r: r["mt"])
                ax.errorbar([r["mt"] for r in subset], [r[f"{metric}_mean"] for r in subset], yerr=[r[f"{metric}_se"] for r in subset], marker="o", label=METHOD_LABELS[method], color=colors[method])
            ax.set_title(f"block=5, ell_t={ell_t}, noise=0.08")
            ax.set_ylabel(ylabel)
        for ax, noise in zip(axes[1], [0.08, 0.10]):
            for method in methods:
                subset = [r for r in rows if r["block_size"] == 10 and r["ell_t"] == 0.8 and r["noise"] == noise and r["method"] == method]
                subset = sorted(subset, key=lambda r: r["mt"])
                ax.errorbar([r["mt"] for r in subset], [r[f"{metric}_mean"] for r in subset], yerr=[r[f"{metric}_se"] for r in subset], marker="o", label=METHOD_LABELS[method], color=colors[method])
            ax.set_title(f"block=10, ell_t=0.8, noise={noise}")
            ax.set_xlabel("M_t")
            ax.set_ylabel(ylabel)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.suptitle(f"Long-memory ablation: {ylabel}")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(PLOTDIR / f"long_memory_ablation_{metric}.png", dpi=200)
        plt.close(fig)


def summarize_best_ablation(rows: list[dict[str, Any]]) -> list[list[object]]:
    routeb = [r for r in rows if r["method"] == "structured_joint_ssgp_transfer"]
    routeb_best_rmse = min(routeb, key=lambda r: r["rmse_mean"])
    routeb_best_nll = min(routeb, key=lambda r: r["nll_mean"])
    matched = []
    for tag, rb in [("Route B best RMSE", routeb_best_rmse), ("Route B best NLL", routeb_best_nll)]:
        same_cfg = [r for r in rows if r["mt"] == rb["mt"] and r["block_size"] == rb["block_size"] and r["ell_t"] == rb["ell_t"] and r["noise"] == rb["noise"]]
        for r in same_cfg:
            matched.append(
                [
                    tag,
                    r["mt"],
                    r["block_size"],
                    r["ell_t"],
                    r["noise"],
                    METHOD_LABELS[r["method"]],
                    f"{r['rmse_mean']:.4f} $\\pm$ {r['rmse_se']:.4f}",
                    f"{r['nll_mean']:.4f} $\\pm$ {r['nll_se']:.4f}",
                    f"{r['coverage90_mean']:.4f} $\\pm$ {r['coverage90_se']:.4f}",
                    f"{r['avg_predictive_variance_mean']:.4f}",
                ]
            )
    return matched


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PLOTDIR.mkdir(parents=True, exist_ok=True)
    TABLEDIR.mkdir(parents=True, exist_ok=True)

    old_standard = load_json("results/experiments_routeB/joint_ssgp_kron_synthetic_report.json")
    standard_diagnostic_rows_all = read_csv_dicts(ROOT / "results" / "experiments_routeB_standard_diagnostic_ablation" / "standard_diagnostic_summary.csv")
    standard_diagnostic_rows = [r for r in standard_diagnostic_rows_all if r.get("beta_u_correlation_design") != "weak"]
    regime_reports = {
        "standard": load_json("results/experiments_routeB_continual_standard/joint_ssgp_kron_synthetic_report.json"),
        "long_memory": load_json("results/experiments_routeB_continual_long_memory/joint_ssgp_kron_synthetic_report.json"),
        "sparse_current": load_json("results/experiments_routeB_continual_sparse_current/joint_ssgp_kron_synthetic_report.json"),
        "old_region": load_json("results/experiments_routeB_continual_old_region/joint_ssgp_kron_synthetic_report.json"),
    }
    verification = load_json("results/verification/routeB_joint_ssgp_kron_verification.json")
    sweep = load_json("results/experiments_routeB_continual_sweep/joint_ssgp_kron_synthetic_report.json")
    calibration_sweep = load_json(
        "results/experiments_routeB_calibration_sweep_rerun_current_routeB/joint_ssgp_kron_synthetic_report.json"
    )

    plot_standard_confirmatory(standard_diagnostic_rows)
    ablation_rows = collect_ablation_rows()
    plot_long_memory_ablation(ablation_rows)

    standard_rows = [
        [
            "mean-field",
            _metric_pair(standard_diagnostic_rows, "rmse", "mean_field"),
            _metric_pair(standard_diagnostic_rows, "nll", "mean_field"),
            _metric_pair(standard_diagnostic_rows, "cov90", "mean_field"),
            _metric_pair(standard_diagnostic_rows, "rmse_forgetting", "mean_field"),
            _metric_pair(standard_diagnostic_rows, "nll_forgetting", "mean_field"),
        ],
        [
            "Route B",
            _metric_pair(standard_diagnostic_rows, "rmse", "routeB"),
            _metric_pair(standard_diagnostic_rows, "nll", "routeB"),
            _metric_pair(standard_diagnostic_rows, "cov90", "routeB"),
            _metric_pair(standard_diagnostic_rows, "rmse_forgetting", "routeB"),
            _metric_pair(standard_diagnostic_rows, "nll_forgetting", "routeB"),
        ],
    ]
    table_to_csv(TABLEDIR / "standard_confirmatory_seen_history.csv", ["method", "rmse", "nll", "coverage90", "rmse_forgetting", "nll_forgetting"], standard_rows)

    old_standard_rows = []
    for method, vals in old_standard["mean_by_method"].items():
        old_standard_rows.append(
            [
                METHOD_LABELS.get(method, method),
                f"{vals['rmse']:.4f}",
                f"{vals['mae']:.4f}",
                f"{vals['nll']:.4f}",
                f"{vals['coverage90']:.4f}",
                f"{vals.get('runtime_per_block', vals.get('runtime_sec', float('nan'))):.6f}",
            ]
        )

    regime_rows = []
    for regime, report in regime_reports.items():
        for method in ["no_transfer", "mean_field_ssgp_transfer", "structured_joint_ssgp_transfer"]:
            key = f"{method}|seen_history"
            vals = report["mean_by_method_eval_mode"][key]
            regime_rows.append(
                [
                    regime,
                    METHOD_LABELS[method],
                    f"{vals['rmse']:.4f}",
                    f"{vals['nll']:.4f}",
                    f"{vals['coverage90']:.4f}",
                    f"{vals['rmse_forgetting']:.4f}",
                    f"{vals['nll_forgetting']:.4f}",
                ]
            )

    initial_task_gp_reports = {
        "standard": load_json("results/experiments_routeB_standard_initial_task_fullgp_ellt_fit_strong/joint_ssgp_kron_synthetic_report.json"),
        "long_memory": load_json(
            "results/experiments_routeB_long_memory_initial_task_ellt_fit_strong/joint_ssgp_kron_synthetic_report.json"
        ),
    }
    initial_task_metrics_paths = [
        ROOT / "results" / "experiments_routeB_standard_initial_task_fullgp_ellt_fit_strong" / "joint_ssgp_kron_synthetic_metrics.csv",
        ROOT / "results" / "experiments_routeB_long_memory_initial_task_ellt_fit_strong" / "joint_ssgp_kron_synthetic_metrics.csv",
    ]
    initial_task_gp_rows = []
    initial_task_gp_current_rows = []
    for regime, report in initial_task_gp_reports.items():
        for method in ["no_transfer", "mean_field_ssgp_transfer", "structured_joint_ssgp_transfer"]:
            initial_task_gp_rows.append(
                [
                    regime,
                    METHOD_LABELS[method],
                    report_metric_cell(report, method, "seen_history", "rmse"),
                    report_metric_cell(report, method, "seen_history", "nll"),
                    report_metric_cell(report, method, "seen_history", "coverage90"),
                    report_metric_cell(report, method, "seen_history", "rmse_forgetting"),
                    report_metric_cell(report, method, "seen_history", "nll_forgetting"),
                ]
            )
            initial_task_gp_current_rows.append(
                [
                    regime,
                    METHOD_LABELS[method],
                    report_metric_cell(report, method, "current", "rmse"),
                    report_metric_cell(report, method, "current", "nll"),
                    report_metric_cell(report, method, "current", "coverage90"),
                    report_metric_cell(report, method, "current", "avg_predictive_variance"),
                    report_metric_cell(report, method, "current", "avg_interval_width90"),
                ]
            )
    initial_task_param_rows = []
    initial_task_selected_rows = []
    for metrics_path in initial_task_metrics_paths:
        param_rows, selected_rows = summarize_selected_values(metrics_path)
        initial_task_param_rows.extend(param_rows)
        initial_task_selected_rows.extend(selected_rows)
    table_to_csv(
        TABLEDIR / "tableC_initial_task_fullgp_mll_seen_history.csv",
        ["regime", "method", "rmse", "nll", "coverage90", "rmse_forgetting", "nll_forgetting"],
        initial_task_gp_rows,
    )
    table_to_csv(
        TABLEDIR / "tableC_initial_task_fullgp_mll_current.csv",
        ["regime", "method", "rmse", "nll", "coverage90", "avg_predictive_variance", "avg_interval_width90"],
        initial_task_gp_current_rows,
    )
    table_to_csv(
        TABLEDIR / "tableC_initial_task_fullgp_mll_protocol.csv",
        ["regime", "data_ell_t", "selected_parameter", "selected_values", "initial_task", "fixed_parameters"],
        initial_task_param_rows,
    )
    table_to_csv(
        TABLEDIR / "tableC_initial_task_fullgp_mll_selected_values.csv",
        ["regime", "seed", "data_ell_t", "selected_model_ell_t", "selected_nlml"],
        initial_task_selected_rows,
    )

    cross_rows = []
    for row in verification["checks"]["routeB_cross_covariance_dense_diagnostic"]["table"]:
        cross_rows.append([row["quantity"], f"{row['routeB_error']:.3e}", f"{row['mean_field_error']:.3e}"])
    sylvester_rows = sylvester_dense_validation_rows()
    table_to_csv(
        TABLEDIR / "sylvester_dense_numerical_validation.csv",
        [
            "M_t x M_s",
            "dim",
            "relative_solution_error",
            "relative_residual",
            "sylvester_time_ms",
            "dense_time_ms",
            "dense_over_sylvester_speedup",
            "dense_matrix_memory_mb",
        ],
        sylvester_rows,
    )

    strong_rows = []
    for noise in [0.03, 0.05, 0.08, 0.10]:
        for method in ["mean_field_ssgp_transfer", "structured_joint_ssgp_transfer"]:
            vals = sweep["mean_by_noise_coupling_method_eval"][f"noise={noise}|coupling=strong|{method}|seen_history"]
            strong_rows.append(
                [
                    f"{noise:.2f}",
                    METHOD_LABELS[method],
                    f"{vals['rmse']:.4f}",
                    f"{vals['nll']:.4f}",
                    f"{vals['coverage90']:.4f}",
                    f"{vals['avg_predictive_variance']:.4f}",
                    f"{vals['avg_interval_width90']:.4f}",
                ]
            )

    def calibration_vals(noise: float, method: str, eval_mode: str) -> dict:
        key = f"noise={noise}|model_ell_t=0.25|coupling=strong|{method}|{eval_mode}"
        if key in calibration_sweep["mean_by_noise_coupling_method_eval"]:
            return calibration_sweep["mean_by_noise_coupling_method_eval"][key]
        legacy_key = f"noise={noise}|coupling=strong|{method}|{eval_mode}"
        return calibration_sweep["mean_by_noise_coupling_method_eval"][legacy_key]

    calibration_current_rows = []
    calibration_seen_rows = []
    for eval_mode, target_rows in [("current", calibration_current_rows), ("seen_history", calibration_seen_rows)]:
        for noise in [0.03, 0.05, 0.08, 0.10]:
            for method in ["mean_field_ssgp_transfer", "structured_joint_ssgp_transfer"]:
                vals = calibration_vals(noise, method, eval_mode)
                target_rows.append(
                    [
                        f"{noise:.2f}",
                        METHOD_LABELS[method],
                        f"{vals['rmse']:.4f}",
                        f"{vals['nll']:.4f}",
                        f"{vals['coverage90']:.4f}",
                        f"{vals['avg_predictive_variance']:.4f}",
                        f"{vals['avg_interval_width90']:.4f}",
                        f"{vals['avg_beta_schur_term']:.6f}",
                    ]
                )

    ablation_best_rows = summarize_best_ablation(ablation_rows)
    table_to_csv(
        TABLEDIR / "long_memory_ablation_representative_best_routeB.csv",
        ["tag", "mt", "block_size", "ell_t", "noise", "method", "rmse", "nll", "coverage90", "avg_predictive_variance"],
        ablation_best_rows,
    )
    table_to_csv(
        TABLEDIR / "long_memory_ablation_all_seen_history.csv",
        ["mt", "block_size", "ell_t", "noise", "method", "rmse_mean", "rmse_se", "nll_mean", "nll_se", "coverage90_mean", "coverage90_se", "rmse_forgetting_mean", "nll_forgetting_mean", "avg_predictive_variance_mean"],
        [
            [
                r["mt"],
                r["block_size"],
                r["ell_t"],
                r["noise"],
                METHOD_LABELS[r["method"]],
                r["rmse_mean"],
                r["rmse_se"],
                r["nll_mean"],
                r["nll_se"],
                r["coverage90_mean"],
                r["coverage90_se"],
                r["rmse_forgetting_mean"],
                r["nll_forgetting_mean"],
                r["avg_predictive_variance_mean"],
            ]
            for r in ablation_rows
        ],
    )

    sylvester_table = latex_table(
        ["M_t x M_s", "Dim", "Rel. solution error", "Rel. residual", "Sylv. ms", "Dense ms", "Speedup", "Dense MB"],
        sylvester_rows,
    )
    cross_table = latex_table(["Quantity", "Route B error", "Mean-field error"], cross_rows)
    old_standard_table = latex_table(["Method", "RMSE", "MAE", "NLL", "90% coverage", "Runtime/block"], old_standard_rows)
    standard_table = latex_table(["Method", "RMSE", "NLL", "Coverage90", "RMSE forgetting", "NLL forgetting"], standard_rows)
    standard_win_rows = [r for r in standard_diagnostic_rows if routeb_all_metric_win(r)]
    standard_win_distribution_rows = [
        ["All five metrics", f"{len(standard_win_rows)} / {len(standard_diagnostic_rows)}"],
        ["block_size=10", f"{Counter(r['block_size'] for r in standard_win_rows)['10']} / {len(standard_win_rows)}"],
        ["block_size=5", f"{Counter(r['block_size'] for r in standard_win_rows)['5']} / {len(standard_win_rows)}"],
        ["coupling=medium", f"{Counter(r['beta_u_correlation_design'] for r in standard_win_rows)['medium']} / {len(standard_win_rows)}"],
        ["coupling=strong", f"{Counter(r['beta_u_correlation_design'] for r in standard_win_rows)['strong']} / {len(standard_win_rows)}"],
        ["hyperfit=noise_kernel", f"{Counter(r['hyperfit_mode'] for r in standard_win_rows)['noise_kernel']} / {len(standard_win_rows)}"],
        ["hyperfit=ell_only", f"{Counter(r['hyperfit_mode'] for r in standard_win_rows)['ell_only']} / {len(standard_win_rows)}"],
    ]
    standard_stable_setting_rows = [
        ["5/8/12/16", "10", "1.0", "medium", "ell_only or noise_kernel"],
        ["5/8/12/16", "10", "1.0", "strong", "noise_kernel"],
        ["5/8/12/16", "10", "0.5", "medium", "ell_only or noise_kernel"],
        ["5/8/12/16", "10", "0.5", "strong", "noise_kernel, mainly M_t >= 12"],
        ["5/8", "10", "1.5", "medium", "ell_only, partly noise_kernel"],
        ["5", "10", "1.5", "strong", "noise_kernel"],
    ]

    def standard_row_for(mt: str, block: str, gp: str, coupling: str, hyperfit: str) -> dict[str, str]:
        for row in standard_diagnostic_rows:
            if (
                row["mt"] == mt
                and row["block_size"] == block
                and row["gp_signal_strength"] == gp
                and row["beta_u_correlation_design"] == coupling
                and row["hyperfit_mode"] == hyperfit
            ):
                return row
        raise KeyError((mt, block, gp, coupling, hyperfit))

    standard_representative_specs = [
        ("5", "10", "1.0", "medium", "ell_only"),
        ("8", "10", "1.0", "medium", "noise_kernel"),
        ("12", "10", "1.0", "strong", "noise_kernel"),
        ("5", "10", "1.5", "medium", "noise_kernel"),
        ("16", "10", "0.5", "medium", "noise_kernel"),
    ]
    standard_representative_rows = []
    for spec in standard_representative_specs:
        row = standard_row_for(*spec)
        standard_representative_rows.append(
            [
                row["mt"],
                row["block_size"],
                row["gp_signal_strength"],
                row["beta_u_correlation_design"],
                row["hyperfit_mode"],
                f"{float(row['routeB_rmse']):.4f}",
                f"{float(row['mean_field_rmse']):.4f}",
                f"{float(row['routeB_nll']):.4f}",
                f"{float(row['mean_field_nll']):.4f}",
            ]
        )
    standard_win_distribution_table = latex_table(["Factor", "All-metric-win count"], standard_win_distribution_rows)
    standard_stable_setting_table = latex_table(["M_t", "block_size", "gp_signal", "coupling", "hyperfit"], standard_stable_setting_rows)
    standard_representative_table = latex_table(
        ["M_t", "Block", "GP", "Coupling", "Fit", "Route B RMSE", "MF RMSE", "Route B NLL", "MF NLL"],
        standard_representative_rows,
    )
    regime_table = latex_table(["Regime", "Method", "RMSE", "NLL", "Cov90", "RMSE forget", "NLL forget"], regime_rows)
    gp_generative_table = latex_table(
        ["Regime", "Method", "RMSE", "NLL", "Cov90", "RMSE forget", "NLL forget"],
        initial_task_gp_rows,
    )
    gp_generative_current_table = latex_table(
        ["Regime", "Method", "RMSE", "NLL", "Cov90", "Avg var", "Width90"],
        initial_task_gp_current_rows,
    )
    gp_generative_param_table = latex_table(
        [
            "Regime",
            "data ell_t",
            "Selected parameter",
            "Selected values",
            "Initial task",
            "Fixed parameters",
        ],
        initial_task_param_rows,
        align=r"@{}p{0.12\linewidth}p{0.08\linewidth}p{0.13\linewidth}p{0.15\linewidth}p{0.15\linewidth}p{0.17\linewidth}@{}",
    )
    gp_generative_selected_table = latex_table(
        ["Regime", "Seed", "data ell_t", "selected model ell_t", "selected NLML"],
        initial_task_selected_rows,
    )
    strong_table = latex_table(["Noise", "Method", "RMSE", "NLL", "Cov90", "Avg var", "Width90"], strong_rows)
    calibration_current_table = latex_table(
        ["Noise", "Method", "RMSE", "NLL", "Cov90", "Avg var", "Width90", "beta/Schur"],
        calibration_current_rows,
    )
    calibration_seen_table = latex_table(
        ["Noise", "Method", "RMSE", "NLL", "Cov90", "Avg var", "Width90", "beta/Schur"],
        calibration_seen_rows,
    )
    ablation_best_table = latex_table(["Tag", "M_t", "Block", "ell_t", "Noise", "Method", "RMSE", "NLL", "Cov90", "Avg var"], ablation_best_rows)
    model_ell_average_rows = []
    for row in read_csv_dicts(TABLEDIR / "model_ell_ablation_average_over_mt.csv"):
        model_ell_average_rows.append(
            [
                row["method"],
                row["model_ell_t"],
                f"{float(row['rmse_mean_over_mt']):.4f}",
                f"{float(row['nll_mean_over_mt']):.4f}",
                f"{float(row['coverage90_mean_over_mt']):.4f}",
                f"{float(row['rmse_forgetting_mean_over_mt']):.4f}",
                f"{float(row['nll_forgetting_mean_over_mt']):.4f}",
            ]
        )
    model_ell_average_table = latex_table(
        ["Method", "model ell_t", "RMSE", "NLL", "Cov90", "RMSE forget", "NLL forget"],
        model_ell_average_rows,
    )
    ellt_fit_long_memory_rows = [
        ["mismatch", "0.8", "fixed 0.25", "negative temporal mismatch diagnostic"],
        ["oracle matched", "0.8", "fixed 0.8", "upper-bound diagnostic only"],
        ["initial-task full-GP MLL", "0.8", "independent full-GP marginal NLL", "primary non-oracle protocol"],
    ]
    initial_task_fullgp_long_report_path = ROOT / "results" / "experiments_routeB_long_memory_initial_task_ellt_fit" / "initial_task_fullgp" / "joint_ssgp_kron_synthetic_report.json"
    initial_task_fullgp_standard_report_path = ROOT / "results" / "experiments_routeB_standard_initial_task_fullgp_ellt_fit" / "joint_ssgp_kron_synthetic_report.json"

    def optional_initial_task_rows(path: Path) -> list[list[str]]:
        if not path.exists():
            return [["not yet generated", "NA", "NA", "NA", "NA", "NA"]]
        report = json.loads(path.read_text(encoding="utf-8"))
        out = []
        for method in ["no_transfer", "mean_field_ssgp_transfer", "structured_joint_ssgp_transfer"]:
            vals = report["mean_by_method_eval_mode"][f"{method}|seen_history"]
            out.append(
                [
                    METHOD_LABELS[method],
                    f"{vals['rmse']:.4f}",
                    f"{vals['nll']:.4f}",
                    f"{vals['coverage90']:.4f}",
                    f"{vals['rmse_forgetting']:.4f}",
                    f"{vals['nll_forgetting']:.4f}",
                ]
            )
        return out

    def fit_comparison_rows(regime: str) -> list[list[str]]:
        rows = read_csv_dicts(TABLEDIR / "ellt_fit_protocol_summary.csv")
        out = []
        for row in rows:
            if row["regime"] != regime:
                continue
            if row["setting"] != "initial_task_fullgp_mll":
                continue
            if row["method"] not in {"mean-field", "Route B"}:
                continue
            out.append(
                [
                    row["setting"].replace("initial_task_", "").replace("_", " "),
                    row["method"],
                    f"{float(row['rmse_mean']):.4f} $\\pm$ {float(row['rmse_se']):.4f}",
                    f"{float(row['nll_mean']):.4f} $\\pm$ {float(row['nll_se']):.4f}",
                    f"{float(row['coverage90_mean']):.4f} $\\pm$ {float(row['coverage90_se']):.4f}",
                    f"{float(row['rmse_forgetting_mean']):.4f}",
                    f"{float(row['nll_forgetting_mean']):.4f}",
                    row["fitted_ell_t_values"],
                ]
            )
        return out or [["not generated", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]]

    selected_ell_rows = [
        [
            row["regime"],
            row["setting"].replace("initial_task_", "").replace("_", " "),
            row["seed"],
            row["fitted_ell_t"],
            f"{float(row['selected_candidate_score']):.4f}",
        ]
        for row in read_csv_dicts(TABLEDIR / "ellt_fit_selected_values.csv")
        if row["setting"] == "initial_task_fullgp_mll"
    ]
    ellt_fit_protocol_table = latex_table(["Setting", "data ell_t", "model ell_t protocol", "Purpose"], ellt_fit_long_memory_rows)
    ellt_fit_initial_task_table = latex_table(["Fitter", "Method", "RMSE", "NLL", "Cov90", "RMSE forget", "NLL forget", "fitted ell_t"], fit_comparison_rows("long_memory"))
    ellt_fit_standard_table = latex_table(["Fitter", "Method", "RMSE", "NLL", "Cov90", "RMSE forget", "NLL forget", "fitted ell_t"], fit_comparison_rows("standard"))
    selected_ell_table = latex_table(["Experiment", "Fitter", "Seed", "fitted model ell_t", "selected score"], selected_ell_rows)

    standard_config_count = len(standard_diagnostic_rows)
    standard_win_count = len(standard_win_rows)
    standard_block10_wins = Counter(r["block_size"] for r in standard_win_rows)["10"]
    standard_medium_wins = Counter(r["beta_u_correlation_design"] for r in standard_win_rows)["medium"]
    standard_strong_wins = Counter(r["beta_u_correlation_design"] for r in standard_win_rows)["strong"]
    standard_noise_kernel_wins = Counter(r["hyperfit_mode"] for r in standard_win_rows)["noise_kernel"]

    tex = rf"""
\documentclass[10pt]{{article}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{caption}}
\usepackage{{longtable}}
\usepackage{{array}}
\title{{Route B Structured Joint SSGP: Continual-Learning Experiments}}
\author{{Generated experiment report}}
\date{{}}
\begin{{document}}
\maketitle

\section{{Executive Summary}}
This report consolidates the Route B structured joint SSGP experiments. The validation evidence is separated into three different categories. First, the Sylvester experiments are numerical linear-solver checks against dense solves of the same system. Second, the dense posterior experiment checks whether the structured Schur/Sylvester recovery matches the exact posterior of the same finite-dimensional Gaussian approximation. Third, the synthetic continual-learning experiments use data generated as $f\sim GP(0,k)$ and $y=\Phi\beta+f+\epsilon$; those experiments are the appropriate evidence for prediction, calibration, and forgetting claims. The continual-learning evidence is mixed. The earlier short standard sanity experiment favored Route B, but the stronger standard diagnostic scan does not support a blanket Route B advantage; Route B is most useful when beta-u coupling and residual signal are non-negligible.

\section{{Validation taxonomy and ground-truth interpretation}}
The word ``reference'' means different things in different experiments. In the Sylvester test, the reference is the dense numerical solution of $D_u z=q$. In the dense posterior test, the reference is $\Lambda_{{dense}}^{{-1}}h_{{dense}}$, the exact posterior of the same finite-dimensional Gaussian approximation. This is not the unknown data-generating truth. In the GP-generative prediction tests, the data are generated from a full Kronecker GP prior and the natural targets are RMSE, NLL, coverage, ECE, and forgetting; $m_u$ is not treated as a physical ground-truth quantity there because $u$ is an inducing representation.

\section{{Table A: Numerical equivalence of Sylvester solves and dense solves}}
This table validates only the linear algebra solver. The system is $D_u z=q$ with
$D_u=K_t^{{-1}}\otimes K_s^{{-1}}+B\otimes G$. The dense solve is the numerical reference, and mean-field is intentionally not included because it is a model approximation, not a solver baseline.
\begin{{table}}[H]
\centering
\small
{sylvester_table}
\caption{{Numerical equivalence of Sylvester solves and dense solves. Rel. solution error is $\|z_{{sylvester}}-z_{{dense}}\|/\|z_{{dense}}\|$; rel. residual is $\|D_u z_{{sylvester}}-q\|/\|q\|$.}}
\end{{table}}

\section{{Table B: Dense finite-dimensional posterior validation}}
This experiment checks whether the structured Route B posterior recovery matches the exact dense posterior of the same finite-dimensional Gaussian model. The dense posterior is not the unknown data-generating truth; it is the exact posterior of the finite-dimensional approximation with joint precision blocks $\Lambda_{{\beta\beta}}$, $\Lambda_{{\beta u}}$, and $\Lambda_{{uu}}$. Route B directly preserves the beta-u cross block. Mean-field sets it to zero, so it is expected to differ when beta-u coupling is nonzero.
\begin{{table}}[H]
\centering
{cross_table}
\caption{{Dense finite-dimensional posterior validation. The reference is $\Lambda_{{dense}}^{{-1}}h_{{dense}}$ for the same finite-dimensional Gaussian approximation, not a true GP data-generating posterior.}}
\end{{table}}

\section{{Table C: GP-generative synthetic prediction and calibration}}
For statistical prediction claims, the synthetic generator samples the latent residual on the full observed grid: $f\sim GP(0,k_t\otimes k_s)$, then $y=\Phi\beta+f+\epsilon$. Therefore RMSE, NLL, coverage, ECE, and forgetting are the appropriate claim-level metrics. If the true $\beta$ is known, $\|m_\beta-\beta_{{true}}\|$ can be reported as an auxiliary parameter-recovery diagnostic; $m_u$ error is not a natural ground-truth metric in this setting.

The main fair comparison now uses initial-task fitted full-GP marginal likelihood with strong beta-u coupling. Medium coupling remains available as a conservative option, but the default experiment uses strong coupling because it directly stresses the Route B mechanism: retaining the beta-u posterior cross covariance induced by $R_{{\beta u}}=\sigma^{{-2}}\Phi^\top A$. The only selected hyperparameter in this run is the model temporal lengthscale $\ell_t$. Observation noise is fixed at 0.08 and kernel variance is fixed at 1.0. The initial task is the first 20\% of the time series, corresponding to four blocks for block\_size=5 and num\_time=100. Time normalization is expected-horizon with time\_scale=1.0. The fitted $\ell_t$ is selected once per seed by full-GP negative log marginal likelihood on the initial task and then frozen for all later online blocks. All methods share the same selected value within each seed. The candidate grid is
\[
\ell_t \in \{{0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.6\}}.
\]

\begin{{table}}[H]
\centering
\scriptsize
{gp_generative_param_table}
\caption{{Initial-task full-GP MLL protocol used for Table C.}}
\end{{table}}

\begin{{table}}[H]
\centering
\scriptsize
{gp_generative_selected_table}
\caption{{Actual model temporal lengthscale selected by initial-task full-GP MLL. Lower selected NLML is better within each seed's candidate grid.}}
\end{{table}}

\begin{{table}}[H]
\centering
\small
{gp_generative_table}
\caption{{GP-generative synthetic prediction and calibration under the initial-task fitted full-GP MLL protocol. These are seen-history continual-learning metrics on data generated as $f\sim GP(0,k_t\otimes k_s)$ and $y=\Phi\beta+f+\epsilon$.}}
\end{{table}}

The current-block evaluation is reported below as a diagnostic, not as the main continual-learning claim. It evaluates only the just-trained block, so no-transfer can be competitive or stronger because it does not need to retain old tasks. Seen-history remains the claim-level continual-learning mode because it measures retention and forgetting over all previously seen blocks.
\begin{{table}}[H]
\centering
\small
{gp_generative_current_table}
\caption{{Current-block diagnostic for the same Table C protocol. This table is included to show short-term fit behavior; it is not the main continual-learning retention metric.}}
\end{{table}}

\section{{Original short synthetic sanity result}}
This is the earlier short synthetic continual-learning sanity result that motivated the larger confirmatory experiment.
\begin{{table}}[H]
\centering
{old_standard_table}
\caption{{Original short synthetic sanity experiment.}}
\end{{table}}

\section{{Calibration diagnostics rerun}}
This section reports the calibration/noise sweep rerun under the current implementation. The setting is synthetic standard data, num\_time=20, num\_space=6, block\_size=5, M\_t=5, M\_s=4, two seeds, linear dimension 2, noise in \{{0.03,0.05,0.08,0.10\}}, beta-u coupling in \{{weak,medium,strong\}}, and evaluation modes current, seen\_history, and future. The tables below report the strong-coupling slice because it is the most direct stress test of beta-u posterior coupling. The rerun uses ell\_t fitting disabled, so it follows the fixed model ell\_t=0.25 diagnostic protocol rather than the later initial-task full-GP MLL protocol.

\begin{{table}}[H]
\centering
\small
{calibration_current_table}
\caption{{Calibration diagnostics rerun: strong coupling, current block.}}
\end{{table}}

\begin{{table}}[H]
\centering
\small
{calibration_seen_table}
\caption{{Calibration diagnostics rerun: strong coupling, seen history.}}
\end{{table}}

\section{{Standard confirmatory experiment}}
The standard result is now reported as a mechanism diagnostic rather than a single fixed setting. The experiment uses the standard synthetic regime with num\_time=100, num\_space=6, M\_s=4, observation noise=0.08, eval modes current and seen\_history, and the main continual-learning metric is seen\_history. The grid is M\_t in \{{5,8,12,16\}}, block\_size in \{{5,10\}}, GP residual signal strength in \{{0.5,1.0,1.5\}}, beta-u coupling in \{{medium,strong\}}, and two full-GP MLL hyperparameter protocols: ell\_only and noise\_kernel. The noise\_kernel protocol fits ell\_t, observation noise, and kernel variance on the initial task, then freezes the selected values for all later online tasks and all methods. This gives {standard_config_count} medium/strong configurations. The table reports mean $\pm$ standard error over configurations, not over random seeds; the diagnostic run uses one seed and should be interpreted as a mechanism scan, not a final significance test.
\begin{{table}}[H]
\centering
{standard_table}
\caption{{Standard diagnostic ablation, seen-history means over 144 configurations.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{plots/standard_confirmatory_seen_history.png}}
\caption{{Standard diagnostic ablation seen-history metrics with standard error over configurations. The figure keeps the same bar-chart format as the previous standard confirmatory plot, but now summarizes the full diagnostic grid.}}
\end{{figure}}

\begin{{table}}[H]
\centering
{standard_win_distribution_table}
\caption{{Where Route B wins all five metrics against mean-field. A win requires lower RMSE, lower NLL, higher 90\% coverage, lower RMSE forgetting, and lower NLL forgetting.}}
\end{{table}}

\begin{{table}}[H]
\centering
\small
{standard_stable_setting_table}
\caption{{Stable mechanism conditions where Route B most often wins all metrics in the standard diagnostic ablation.}}
\end{{table}}

\begin{{table}}[H]
\centering
\small
{standard_representative_table}
\caption{{Representative strong Route B wins from the standard diagnostic ablation.}}
\end{{table}}

\paragraph{{Route B wins.}} After fixing the NaN forgetting aggregation and focusing on the medium/strong coupling regimes that match the Route B mechanism, {standard_win_count} / {standard_config_count} standard diagnostic configurations satisfy all five Route B advantages over mean-field: lower RMSE, lower NLL, higher coverage, lower RMSE forgetting, and lower NLL forgetting. These wins are not uniformly distributed. They concentrate at block\_size=10 ({standard_block10_wins} / {standard_win_count} wins), medium coupling ({standard_medium_wins} / {standard_win_count}), strong coupling ({standard_strong_wins} / {standard_win_count}), and the full-GP MLL noise\_kernel hyperfit protocol ({standard_noise_kernel_wins} / {standard_win_count}).

\paragraph{{Interpretation.}} Route B is most consistently beneficial when beta-u coupling is non-negligible and the online block is less myopic, especially with block\_size=10 and medium/strong coupling. Full-GP MLL fitting of noise and kernel variance further increases the frequency of all-metric wins. GP residual signal strength does not need to be maximal: gp\_signal=1.0 and 0.5 are more stable than 1.5 in this scan. Weak-coupling standard settings remain mean-field-favorable or mixed. Simply increasing M\_t is not sufficient; the all-metric win count is actually largest at M\_t=5 in this one-seed diagnostic, so the standard failure mode is not only temporal basis capacity.

\section{{Continual benchmark regimes}}
The following table summarizes the previously run continual regimes with seen\_history as the main mode. These are block-averaged means from the corresponding experiment reports.
\begin{{table}}[H]
\centering
\small
{regime_table}
\caption{{Seen-history summary across continual regimes.}}
\end{{table}}

\section{{Noise sweep and calibration}}
Noise 0.08 remains the recommended balanced synthetic setting, but current-block coverage at low noise can be below 0.9. The strong-coupling seen-history sweep shows Route B often improves NLL against mean-field, while coverage and future/current behavior remain diagnostic rather than claim-level evidence.
\begin{{table}}[H]
\centering
{strong_table}
\caption{{Strong coupling, seen-history noise sweep.}}
\end{{table}}

\section{{Long-memory ablation}}
Configuration grid: M\_t in \{{5,8,12,16\}}, block\_size in \{{5,10\}}, ell\_t in \{{0.5,0.8\}}, noise in \{{0.08,0.10\}}, 3 seeds, num\_time=100, num\_space=10. The goal is to test whether the earlier long-memory failure is caused by insufficient temporal basis capacity or by the transfer structure itself.

\subsection{{Linear basis and time-dependence visualization}}
The synthetic linear regression basis is
\[
\Phi(t,s) = [1,\ t_{{\mathrm{{scaled}}}},\ s_{{\mathrm{{centered}}}},\ \sin(2\pi t_{{\mathrm{{scaled}}}})].
\]
The true coefficient vector is $[0.4,-0.7,0.25,0.15]$. This basis can represent a global trend, a spatial offset, and one seasonal sinusoid, but it cannot by itself represent arbitrary long-memory residual trajectories. The following plots compare one observed location against the final online predictions. The first uses the current long-memory experiment convention where data are generated with ell\_t=0.8 but the model block factors still use temporal lengthscale 0.25. The second uses a matched model lengthscale 0.8 as a diagnostic.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{long_memory_time_dependence/long_memory_location_time_dependence.png}}
\caption{{Long-memory time dependence at one location: data ell\_t=0.8, model ell\_t=0.25.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{long_memory_time_dependence_matched_ell/long_memory_location_time_dependence.png}}
\caption{{Long-memory time dependence at one location: data ell\_t=0.8, model ell\_t=0.8 diagnostic.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{plots/long_memory_ablation_rmse.png}}
\caption{{Long-memory ablation: seen-history RMSE vs M\_t.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{plots/long_memory_ablation_nll.png}}
\caption{{Long-memory ablation: seen-history NLL vs M\_t.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{plots/long_memory_ablation_avg_predictive_variance.png}}
\caption{{Long-memory ablation: average predictive variance vs M\_t.}}
\end{{figure}}

\begin{{table}}[H]
\centering
\small
{ablation_best_table}
\caption{{Representative best Route B long-memory ablation settings and matched baselines. Full CSV is saved with the report.}}
\end{{table}}

\paragraph{{Interpretation.}} If Route B improves with larger M\_t in a subpanel, that indicates temporal basis capacity was a limiting factor. Where Route B remains worse than mean-field/no\_transfer, the failure is more likely caused by calibration or transfer mismatch rather than only basis capacity.

\subsection{{Model temporal lengthscale ablation}}
The previous long-memory experiments used model-side temporal lengthscale 0.25 in the block-factor construction. The follow-up ablation fixes data ell\_t=0.8 and sweeps model ell\_t in \{{0.25,0.5,0.8\}} and M\_t in \{{5,8,12,16\}}. The table averages seen-history metrics over M\_t.

\begin{{table}}[H]
\centering
\small
{model_ell_average_table}
\caption{{Long-memory model temporal lengthscale ablation, averaged over M\_t.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{plots/model_ell_ablation_rmse.png}}
\caption{{Model ell\_t ablation: seen-history RMSE.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{plots/model_ell_ablation_nll.png}}
\caption{{Model ell\_t ablation: seen-history NLL.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{plots/model_ell_ablation_rmse_forgetting.png}}
\caption{{Model ell\_t ablation: RMSE forgetting.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{plots/model_ell_ablation_nll_forgetting.png}}
\caption{{Model ell\_t ablation: NLL forgetting.}}
\end{{figure}}

\paragraph{{Interpretation.}} The model ell\_t=0.25 setting is clearly mismatched for long-memory. Route B improves substantially at model ell\_t=0.5 and 0.8. Model ell\_t=0.8 gives the best Route B NLL and NLL forgetting, while model ell\_t=0.5 gives the best Route B RMSE and RMSE forgetting.

\subsection{{Initial-task fitted temporal lengthscale}}
The previous short-K fitting experiment is not used as the main protocol. To avoid using the true synthetic temporal lengthscale as an oracle, the runner now fits model ell\_t once on an initial task and freezes that value for all later online tasks. The selector is an independent batch/full-GP marginal likelihood on the initial task, integrating out beta under the Gaussian beta prior. No structured validation-NLL fitter is used in the main protocol. No future online blocks or test labels are used. The selected ell\_t is shared by no\_transfer, mean-field, and Route B; it is not tuned separately per method.

Time normalization is a general dataset-independent layer. With custom or expected-horizon normalization, $t_{{scaled}}=(t_{{raw}}-t_0)/time\_scale$; initial-task normalization uses the initial-task span but can bias selection toward short memory when the initial task is short. With the time-scale grid, normalized model-time candidates are \{{0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.6\}}. Synthetic data may set time\_scale=1.0, but this is the same general mechanism used for other datasets, not a synthetic-specific rule.

\begin{{table}}[H]
\centering
\small
{ellt_fit_protocol_table}
\caption{{Long-memory ell\_t fitting protocol settings.}}
\end{{table}}

\begin{{table}}[H]
\centering
\small
{ellt_fit_initial_task_table}
\caption{{Long-memory initial-task fitted ell\_t protocol, seen-history means.}}
\end{{table}}

\paragraph{{Interpretation.}} The mismatch setting remains a negative diagnostic and the matched setting is an upper-bound diagnostic only. Full-GP marginal likelihood is the main non-oracle protocol because it is method independent and gives the stronger long-memory continual result: Route B has lower RMSE, lower NLL, and lower forgetting than mean-field.

\begin{{table}}[H]
\centering
\small
{ellt_fit_standard_table}
\caption{{Initial-task fitted ell\_t confirmatory experiment in the standard regime, seen-history means.}}
\end{{table}}

\paragraph{{Reporting rule.}} All methods share the fitted ell\_t and the value is frozen after the initial task. Results from the prior short-K fitted experiment are excluded from this comparison. The standard confirmatory result remains mixed: Route B improves NLL and coverage against mean-field under full-GP MLL fitting, but mean-field still has lower RMSE and lower forgetting.

\begin{{table}}[H]
\centering
\small
{selected_ell_table}
\caption{{Actual model ell\_t selected by the initial-task fitter. These values are shared by all methods within each seed.}}
\end{{table}}

\section{{Negative Results and Caveats}}
\begin{{itemize}}
\item The stronger standard confirmatory experiment does not support the earlier short-run claim that Route B is uniformly better.
\item Under initial-task full-GP MLL fitting, Route B beats mean-field on long-memory RMSE and NLL, but not on standard-regime RMSE.
\item Long-memory and sparse-current regimes show Route B can improve NLL while losing RMSE or coverage.
\item Route B is sharper because it preserves beta-u covariance; this is theoretically correct but can hurt coverage when noise/calibration is mismatched.
\item The old-region retention regime currently uses fixed spatial factors and left-region evaluation only. A true left-to-right masked training schedule requires block-specific spatial factors or separate region states.
\item projected\_prior is an old diagnostic ablation and is not a principled Route B baseline.
\item ERA5 remains a lightweight probe; no superiority claim is made there.
\end{{itemize}}

\section{{Artifacts}}
Main output directories:
\begin{{itemize}}
\item results/experiments\_routeB\_standard\_confirmatory\_all/
\item results/experiments\_routeB\_long\_memory\_ablation/
\item results/experiments\_routeB\_continual\_sweep/
\item results/routeB\_experiment\_report/
\end{{itemize}}

\end{{document}}
"""
    tex_path = OUTDIR / "routeB_experiment_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    if shutil.which("pdflatex"):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name], cwd=OUTDIR, check=True)
        print(f"Saved report PDF to {OUTDIR / 'routeB_experiment_report.pdf'}")
    else:
        print(f"pdflatex not found in this environment; saved LaTeX source to {tex_path}")


if __name__ == "__main__":
    main()
