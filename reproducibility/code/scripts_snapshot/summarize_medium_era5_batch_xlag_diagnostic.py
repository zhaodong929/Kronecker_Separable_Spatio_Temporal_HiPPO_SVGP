#!/usr/bin/env python3
"""Summarize the medium-ERA5 batch/x-lag diagnostic runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    base = Path("results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready")
    outdir = base / "medium_era5_batch_xlag_diagnostic"
    outdir.mkdir(parents=True, exist_ok=True)

    def row(
        label: str,
        scope: str,
        path: Path,
        method: str = "structured_joint",
        eval_mode: str | None = None,
        phi_mode: str | None = None,
    ) -> dict[str, object]:
        df = pd.read_csv(path)
        if "method" in df.columns:
            df = df[df["method"] == method]
        if eval_mode is not None and "eval_mode" in df.columns:
            df = df[df["eval_mode"] == eval_mode]
        if phi_mode is not None and "phi_mode" in df.columns:
            df = df[df["phi_mode"] == phi_mode]
        if len(df) != 1:
            raise RuntimeError(f"{label}: expected 1 row, got {len(df)} from {path}")
        r = df.iloc[0]
        return {
            "label": label,
            "scope": scope,
            "method": method,
            "eval_mode": r.get("eval_mode", eval_mode or ""),
            "phi_mode": r.get("phi_mode", ""),
            "rmse": float(r["rmse"]),
            "rmse_uncertainty": float(r.get("rmse_se", r.get("rmse_ci95", 0.0))),
            "nll": float(r["nll"]),
            "nll_uncertainty": float(r.get("nll_se", r.get("nll_ci95", 0.0))),
            "coverage90": float(r["coverage90"]),
            "ece": float(r["ece"]),
            "avg_var": float(r.get("avg_var", r.get("avg_predictive_variance", float("nan")))),
            "source": str(path),
        }

    rows: list[dict[str, object]] = []
    phi_summary = base / "analytic_hippo_rff_fixed_rerun/tables/phi_mode_summary.csv"
    rows.append(row("original medium safe-lag", "3 split main report", phi_summary, "structured_joint", phi_mode="medium_era5"))
    rows[-1]["eval_mode"] = "seen_history"
    rows[-1]["phi_mode"] = "medium_era5"

    rows.append(
        row(
            "x-lag no target lag",
            "3 split diagnostic",
            outdir / "medium_era5_xlag_structured_3split/era5_routeb_summary.csv",
            "structured_joint",
            "seen_history",
        )
    )
    rows.append(
        row(
            "safe recursive batch",
            "split0 diagnostic",
            outdir / "medium_era5_batch_structured_split0/era5_routeb_summary.csv",
            "structured_joint",
            "batch",
        )
    )
    rows.append(
        row(
            "oracle y-lag batch",
            "split0 upper bound",
            outdir / "medium_era5_oracle_ylag_batch_split0/era5_routeb_summary.csv",
            "structured_joint",
            "batch",
        )
    )

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "medium_era5_batch_xlag_summary.csv", index=False)

    all_methods = pd.read_csv(outdir / "medium_era5_xlag_all_methods_split0/era5_routeb_summary.csv")
    all_methods = all_methods[
        ["method", "eval_mode", "phi_mode", "rmse", "nll", "coverage90", "ece", "avg_var", "runtime_per_block"]
    ]
    all_methods.to_csv(outdir / "medium_era5_xlag_all_methods_split0_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    colors = ["#4c78a8", "#54a24b", "#f58518", "#b279a2"]
    for ax, metric, title in zip(axes, ["rmse", "nll"], ["RMSE lower is better", "NLL/NLPD lower is better"]):
        bars = ax.bar(range(len(summary)), summary[metric], color=colors)
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(summary["label"], rotation=22, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, summary[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Medium-ERA5 y-lag vs x-lag and batch/oracle diagnostics")
    fig.savefig(outdir / "medium_era5_batch_xlag_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ordered = all_methods.set_index("method").loc[["no_transfer", "mean_field", "structured_joint"]].reset_index()
    bars = ax.bar(ordered["method"], ordered["rmse"], color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_title("medium_era5_xlag split-0 method comparison")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, ordered["rmse"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(outdir / "medium_era5_xlag_method_rmse_split0.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    md = """# Medium-ERA5 batch and x-lag diagnostic

All runs use the analytic HiPPO-RFF Route B setting inherited from the current medium-ERA5 main experiment: task_1 calibration, task_2 online evaluation, RBF kernel, Mt=8, Ms=64, ell_t=0.05, sigma=0.1, kernel variance=1.0.

## Key results

| Setting | Scope | RMSE | NLL/NLPD | Cov90 | ECE | Avg var |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
"""
    for r in rows:
        md += (
            f"| {r['label']} | {r['scope']} | {float(r['rmse']):.4f} | {float(r['nll']):.4f} | "
            f"{float(r['coverage90']):.4f} | {float(r['ece']):.4f} | {float(r['avg_var']):.4f} |\n"
        )
    md += """

## Interpretation

1. The x-lag feature map removes target-lag leakage and rollout complexity by replacing y_{t-1}, y_{t-2}, y_{t-1}-y_{t-2} with current, one-step lagged, and differenced exogenous ERA5 surface covariates. It improves structured-joint RMSE from the main safe-lag medium result (0.2779) to 0.1989 over three held-out splits.
2. The oracle y-lag batch row is an explicit upper-bound/cheating diagnostic because held-out target lags are read from the target sequence. Its RMSE is 0.1294 on split 0, confirming that true target inertia contains very strong information.
3. The safe recursive batch row is not an upper bound; it uses the final posterior but still has to cold-start and recursively fill held-out target lags over the whole task. Its worse RMSE indicates that the recursive y-lag rollout, not only posterior transfer capacity, is a major error source.
4. In the x-lag split-0 method comparison, structured joint remains best, so the Route B structured coupling is still useful after removing target-lag recursion.

## Files

- Summary CSV: medium_era5_batch_xlag_summary.csv
- x-lag all-method split-0 CSV: medium_era5_xlag_all_methods_split0_summary.csv
- Main plot: medium_era5_batch_xlag_metrics.png
- Method plot: medium_era5_xlag_method_rmse_split0.png
"""
    (outdir / "medium_era5_batch_xlag_diagnostic_report.md").write_text(md, encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nall methods split0")
    print(all_methods.to_string(index=False))
    print(outdir / "medium_era5_batch_xlag_diagnostic_report.md")


if __name__ == "__main__":
    main()
