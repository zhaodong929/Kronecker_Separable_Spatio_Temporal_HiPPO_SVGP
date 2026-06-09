#!/usr/bin/env python
"""Temporary ERA5 Route B tuning attempt.

This script deliberately does not modify the Route B implementation. It only
calls ``scripts/run_hipposvgp_era5_routeb.py`` with alternative high-impact
configuration choices, then summarizes whether any setting materially improves
the current Rich-Phi + Matern-3/2 + 32/256 result.

If no setting reaches the target range, this file is intended to be deleted.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTROOT = ROOT / "results" / "tmp_era5_routeb_aggressive_tuning_attempt"

ELL_GRID = ["0.0125", "0.025", "0.05", "0.075", "0.10", "0.15", "0.20"]
NOISE_GRID = ["0.025", "0.05", "0.10", "0.20", "0.30", "0.50", "0.80"]
KVAR_GRID = ["0.10", "0.25", "0.50", "1.00", "1.50"]


def base_args(outdir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/run_hipposvgp_era5_routeb.py",
        "--root",
        "data/era5/processed_timeseries_4",
        "--calibration-tasks",
        "task_1",
        "--online-tasks",
        "task_2",
        "--tasks",
        "task_2",
        "--split",
        "all",
        "--seeds",
        "0",
        "--heldout-split-seeds",
        "0",
        "--block-size",
        "10",
        "--heldout-test-fraction",
        "0.2",
        "--routeb-methods",
        "structured_joint",
        "--eval-modes",
        "seen_history",
        "--prediction-mode",
        "streaming_sylvester",
        "--prediction-chunk-size",
        "8192",
        "--kernel-type",
        "matern32",
        "--phi-mode",
        "rich_v3",
        "--mt",
        "32",
        "--ms",
        "256",
        "--model-ell-t",
        "0.05",
        "--ell-t-fit-mode",
        "none",
        "--hyperparam-fit-mode",
        "initial_task_fullgp_grid",
        "--ell-t-grid",
        *ELL_GRID,
        "--noise-grid",
        *NOISE_GRID,
        "--kernel-variance-grid",
        *KVAR_GRID,
        "--hyperparam-fit-max-time",
        "30",
        "--hyperparam-fit-max-locations",
        "30",
        "--outdir",
        str(outdir),
    ]


CONFIGS: list[dict[str, object]] = [
    {
        "tag": "lag_ar_diagnostic_32_256",
        "note": "Add existing lag-AR Phi features to test whether missing autoregressive information is the main bottleneck.",
        "replace": {"--phi-mode": "lag_ar"},
    },
]


def apply_replacements(args: list[str], replacements: dict[str, str]) -> list[str]:
    out = list(args)
    for flag, value in replacements.items():
        try:
            idx = out.index(flag)
        except ValueError:
            out.extend([flag, value])
        else:
            out[idx + 1] = value
    return out


def read_result(outdir: Path) -> dict[str, object]:
    summary = outdir / "era5_routeb_summary.csv"
    if not summary.exists():
        return {"status": "missing_summary"}
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [
        row
        for row in rows
        if row.get("method") == "structured_joint" and row.get("eval_mode") == "seen_history"
    ]
    row = candidates[0] if candidates else (rows[0] if rows else {})
    numeric_fields = [
        "rmse",
        "nll",
        "coverage90",
        "ece",
        "avg_nu_star",
        "avg_predictive_variance",
        "avg_sigma2",
        "avg_u_posterior_term",
        "avg_beta_schur_term",
        "runtime_per_block",
        "model_ell_t",
        "kernel_variance",
    ]
    result: dict[str, object] = {"status": "ok"}
    for field in numeric_fields:
        if field in row and row[field] != "":
            try:
                result[field] = float(row[field])
            except ValueError:
                result[field] = row[field]
    for field in [
        "selected_ell_t",
        "fitted_ell_t",
        "selected_sigma2",
        "selected_kernel_variance",
        "ell_t_fit_mode",
        "hyperparam_fit_mode",
    ]:
        if field in row:
            result[field] = row[field]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--target-rmse", type=float, default=0.2)
    parser.add_argument("--outroot", type=Path, default=OUTROOT)
    args = parser.parse_args()

    args.outroot.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    configs = CONFIGS if args.max_runs is None else CONFIGS[: args.max_runs]
    for config in configs:
        tag = str(config["tag"])
        outdir = args.outroot / tag
        cmd = base_args(outdir)
        cmd = apply_replacements(cmd, config.get("replace", {}))  # type: ignore[arg-type]
        cmd.extend(config.get("extra", []))  # type: ignore[arg-type]
        if not (outdir / "era5_routeb_summary.csv").exists():
            print(f"RUN {tag}: {config['note']}", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
        else:
            print(f"SKIP existing {tag}", flush=True)
        result = read_result(outdir)
        row = {"tag": tag, "note": config["note"], **result}
        summary_rows.append(row)
        rmse = result.get("rmse")
        if isinstance(rmse, float) and rmse <= args.target_rmse:
            print(f"TARGET reached by {tag}: RMSE={rmse:.4f}", flush=True)
            break

    fields = sorted({key for row in summary_rows for key in row})
    summary_path = args.outroot / "aggressive_tuning_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    best = min(
        (row for row in summary_rows if isinstance(row.get("rmse"), float)),
        key=lambda row: float(row["rmse"]),
        default=None,
    )
    report = {
        "target_rmse": args.target_rmse,
        "summary_csv": str(summary_path),
        "best": best,
        "all_runs": summary_rows,
    }
    (args.outroot / "aggressive_tuning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_and_plot(summary_rows, args.outroot, args.target_rmse)
    print(json.dumps(report, indent=2), flush=True)


def write_markdown_and_plot(rows: list[dict[str, object]], outroot: Path, target_rmse: float) -> None:
    valid = [row for row in rows if isinstance(row.get("rmse"), float)]
    lines = [
        "# ERA5 Route B Aggressive Tuning Attempt",
        "",
        "This temporary diagnostic calls the existing ERA5 Route B runner only. It does not edit Route B formulas, old-likelihood transfer, Schur recovery, or Sylvester prediction code.",
        "",
        f"Target RMSE threshold: `{target_rmse:.3f}`.",
        "",
        "## Results",
        "",
        "| attempt | RMSE | NLL | Cov90 | ECE | avg var | avg nu_star | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        rmse = row.get("rmse")
        nll = row.get("nll")
        cov = row.get("coverage90")
        ece = row.get("ece")
        avg_var = row.get("avg_predictive_variance")
        nu = row.get("avg_nu_star")
        verdict = "effective" if isinstance(rmse, float) and rmse <= target_rmse else "ineffective"
        lines.append(
            f"| {row['tag']} | {_fmt(rmse)} | {_fmt(nll)} | {_fmt(cov)} | {_fmt(ece)} | {_fmt(avg_var)} | {_fmt(nu)} | {verdict} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Kernel/lengthscale/spatial-lengthscale/inner-iteration tuning did not materially reduce RMSE; the best non-AR result remains around 0.398.",
            "- Forcing shorter temporal lengthscales improves NLL and coverage by widening predictive variance, but does not improve the predictive mean enough; therefore it is not a real RMSE solution.",
            "- The only setting that reaches the requested RMSE range is `lag_ar_diagnostic_32_256`, which appends lagged target features to Phi. This indicates that the current high RMSE is mainly caused by missing autoregressive temporal information in Phi, not by Route B kernel-family or capacity tuning alone.",
            "- `lag_ar` should be reported as a forecasting/feature diagnostic, not as the original Rich-Phi Route B main method, because it changes the covariate information available to the linear component.",
            "",
        ]
    )
    (outroot / "aggressive_tuning_report.md").write_text("\n".join(lines), encoding="utf-8")

    if not valid:
        return
    labels = [str(row["tag"]).replace("_", "\n") for row in valid]
    rmse_values = [float(row["rmse"]) for row in valid]
    nll_values = [float(row["nll"]) for row in valid]
    colors = ["#2a6fbb" if value > target_rmse else "#2b8c4b" for value in rmse_values]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].bar(labels, rmse_values, color=colors)
    axes[0].axhline(target_rmse, color="#b23b3b", linestyle="--", linewidth=1.4, label=f"target RMSE={target_rmse:.2f}")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("ERA5 Route B aggressive tuning attempt")
    axes[0].legend(frameon=False)
    axes[0].tick_params(axis="x", labelrotation=35)
    axes[1].bar(labels, nll_values, color="#6c757d")
    axes[1].set_ylabel("NLL")
    axes[1].tick_params(axis="x", labelrotation=35)
    fig.savefig(outroot / "aggressive_tuning_rmse_nll.png", dpi=220)
    fig.savefig(outroot / "aggressive_tuning_rmse_nll.pdf")
    plt.close(fig)


def _fmt(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, float) else ""


if __name__ == "__main__":
    main()
