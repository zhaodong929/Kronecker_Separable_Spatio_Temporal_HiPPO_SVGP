#!/usr/bin/env python3
"""Run Medium-ERA5 x-lag lag-length and kernel validation diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path("results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready")
OUT = ROOT / "xlag_lag_kernel_validation"
RUNS = OUT / "runs"
FIGS = OUT / "figures"
PYTHON = Path(".venv/bin/python")

LAG_GRID = [1, 2, 3, 4, 5, 6, 8, 10]
VAL_SEED = 0
TEST_SEED = 1


BASE_ARGS = [
    "--root",
    "data/era5/processed_timeseries_4",
    "--calibration-tasks",
    "task_1",
    "--online-tasks",
    "task_2",
    "--variable-index",
    "0",
    "--split",
    "all",
    "--block-size",
    "10",
    "--routeb-methods",
    "structured_joint",
    "--eval-modes",
    "seen_history",
    "--ohsvgp-heldout-eval",
    "--seeds",
    "0",
    "--mt",
    "8",
    "--ms",
    "64",
    "--prediction-mode",
    "streaming_sylvester",
    "--prediction-chunk-size",
    "8192",
    "--hyperparam-fit-mode",
    "none",
    "--ell-t-fit-mode",
    "none",
    "--model-ell-t",
    "0.05",
    "--routeb-noise",
    "0.1",
    "--kernel-variance",
    "1.0",
]


def run_command(name: str, args: list[str], *, force: bool = False) -> Path:
    outdir = RUNS / name
    summary = outdir / "era5_routeb_summary.csv"
    if summary.exists() and not force:
        return outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), "scripts/run_hipposvgp_era5_routeb.py", "--outdir", str(outdir), *args]
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (outdir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (outdir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (outdir / "walltime_seconds.txt").write_text(f"{time.perf_counter() - started:.6f}\n", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with code {proc.returncode}. See {outdir / 'stdout.log'}")
    return outdir


def read_summary(outdir: Path) -> dict[str, float]:
    df = pd.read_csv(outdir / "era5_routeb_summary.csv")
    df = df[(df["method"] == "structured_joint") & (df["eval_mode"] == "seen_history")]
    if len(df) != 1:
        raise RuntimeError(f"Expected one structured_joint seen_history row in {outdir}")
    row = df.iloc[0].to_dict()
    wall = float((outdir / "walltime_seconds.txt").read_text().strip()) if (outdir / "walltime_seconds.txt").exists() else float("nan")
    return {
        "rmse": float(row["rmse"]),
        "nll": float(row["nll"]),
        "coverage90": float(row["coverage90"]),
        "ece": float(row["ece"]),
        "runtime": wall,
    }


def run_routeb(
    *,
    name: str,
    phi_mode: str,
    kernel_type: str,
    split_seed: int,
    lag_length: int = 1,
    num_mixtures: int = 4,
    save_predictions: bool = False,
) -> Path:
    args = [
        *BASE_ARGS,
        "--heldout-split-seeds",
        str(split_seed),
        "--phi-mode",
        phi_mode,
        "--kernel-type",
        kernel_type,
        "--xlag-length",
        str(lag_length),
        "--num-mixtures",
        str(num_mixtures),
    ]
    if save_predictions:
        args.append("--save-per-location-predictions")
    return run_command(name, args)


def lag_validation() -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, Any]] = []
    for lag in LAG_GRID:
        val_dir = run_routeb(
            name=f"lag_validation/L{lag}_val_seed{VAL_SEED}_rbf",
            phi_mode="medium_era5_xlag",
            kernel_type="rbf",
            split_seed=VAL_SEED,
            lag_length=lag,
        )
        test_dir = run_routeb(
            name=f"lag_validation/L{lag}_test_seed{TEST_SEED}_rbf",
            phi_mode="medium_era5_xlag",
            kernel_type="rbf",
            split_seed=TEST_SEED,
            lag_length=lag,
        )
        val = read_summary(val_dir)
        test = read_summary(test_dir)
        rows.append(
            {
                "lag_length": lag,
                "kernel_type": "rbf",
                "num_mixtures": 0,
                "val_seed": VAL_SEED,
                "test_seed": TEST_SEED,
                "val_rmse": val["rmse"],
                "val_nll": val["nll"],
                "test_rmse": test["rmse"],
                "test_nll": test["nll"],
                "val_runtime": val["runtime"],
                "test_runtime": test["runtime"],
                "runtime": val["runtime"] + test["runtime"],
                "stability_comment": "",
            }
        )
    df = pd.DataFrame(rows)
    order = df.sort_values(["val_nll", "val_rmse"]).reset_index(drop=True)
    selected = int(order.iloc[0]["lag_length"])
    comments = []
    for _, r in df.iterrows():
        if int(r["lag_length"]) == selected:
            comments.append("selected by validation NLL")
        elif r["val_nll"] > order.iloc[0]["val_nll"] + 0.05:
            comments.append("worse validation NLL")
        elif r["val_rmse"] > order.iloc[0]["val_rmse"] + 0.02:
            comments.append("higher validation RMSE")
        else:
            comments.append("similar")
    df["stability_comment"] = comments
    df.to_csv(OUT / "lag_length_validation_results.csv", index=False)
    (OUT / "selected_lag.json").write_text(json.dumps({"selected_lag": selected, "criterion": "min validation NLL, then RMSE"}, indent=2), encoding="utf-8")
    return df, selected


def kernel_ablation(selected_lag: int) -> pd.DataFrame:
    configs = [
        ("X-lag", "medium_era5_xlag", "rbf", selected_lag, 0),
        ("X-lag", "medium_era5_xlag", "matern32", selected_lag, 0),
        ("X-lag", "medium_era5_xlag", "spectral_mixture", selected_lag, 2),
        ("X-lag", "medium_era5_xlag", "spectral_mixture", selected_lag, 4),
        ("X-lag", "medium_era5_xlag", "spectral_mixture", selected_lag, 8),
        ("recursive Y-lag", "medium_era5", "rbf", 2, 0),
        ("oracle Y-lag", "medium_era5_oracle_ylag", "rbf", 2, 0),
    ]
    rows: list[dict[str, Any]] = []
    for method, phi_mode, kernel, lag, q in configs:
        q_arg = q if q else 4
        label = f"{method.replace(' ', '_').replace('-', '').lower()}_{kernel}_L{lag}_Q{q}"
        val_dir = run_routeb(
            name=f"kernel_ablation/{label}_val_seed{VAL_SEED}",
            phi_mode=phi_mode,
            kernel_type=kernel,
            split_seed=VAL_SEED,
            lag_length=lag if phi_mode == "medium_era5_xlag" else 1,
            num_mixtures=q_arg,
        )
        test_dir = run_routeb(
            name=f"kernel_ablation/{label}_test_seed{TEST_SEED}",
            phi_mode=phi_mode,
            kernel_type=kernel,
            split_seed=TEST_SEED,
            lag_length=lag if phi_mode == "medium_era5_xlag" else 1,
            num_mixtures=q_arg,
        )
        val = read_summary(val_dir)
        test = read_summary(test_dir)
        rows.append(
            {
                "method": method,
                "kernel_type": kernel,
                "lag_length": lag,
                "num_mixtures": q,
                "seed": f"val={VAL_SEED};test={TEST_SEED}",
                "val_rmse": val["rmse"],
                "val_nll": val["nll"],
                "test_rmse": test["rmse"],
                "test_nll": test["nll"],
                "runtime": val["runtime"] + test["runtime"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "kernel_ablation_results.csv", index=False)
    return df


def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "era5_routeb_per_location_predictions.csv")


def prediction_exports(selected_lag: int, kernel_df: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    # Use the best validation x-lag kernel for the qualitative plots.
    xlag_rows = kernel_df[kernel_df["method"] == "X-lag"].sort_values(["val_nll", "val_rmse"])
    best = xlag_rows.iloc[0]
    kernel = str(best["kernel_type"])
    q = int(best["num_mixtures"]) if int(best["num_mixtures"]) > 0 else 4
    xlag_dir = run_routeb(
        name="visualization/best_xlag_test_predictions",
        phi_mode="medium_era5_xlag",
        kernel_type=kernel,
        split_seed=TEST_SEED,
        lag_length=selected_lag,
        num_mixtures=q,
        save_predictions=True,
    )
    recursive_dir = run_routeb(
        name="visualization/recursive_ylag_test_predictions",
        phi_mode="medium_era5",
        kernel_type="rbf",
        split_seed=TEST_SEED,
        save_predictions=True,
    )
    oracle_dir = run_routeb(
        name="visualization/oracle_ylag_test_predictions",
        phi_mode="medium_era5_oracle_ylag",
        kernel_type="rbf",
        split_seed=TEST_SEED,
        save_predictions=True,
    )
    preds = {
        "xlag": load_predictions(xlag_dir),
        "recursive": load_predictions(recursive_dir),
        "oracle": load_predictions(oracle_dir),
    }
    rows = []
    common = set(preds["xlag"]["location_index"].unique())
    for df in preds.values():
        common &= set(df["location_index"].unique())
    for loc in sorted(common):
        row: dict[str, Any] = {"location_index": int(loc)}
        for key, df in preds.items():
            g = df[df["location_index"] == loc]
            row[f"{key}_rmse"] = float(np.sqrt(np.mean((g["y_true"] - g["pred_mean"]) ** 2)))
        row["xlag_gain_vs_recursive"] = row["recursive_rmse"] - row["xlag_rmse"]
        rows.append(row)
    loc_df = pd.DataFrame(rows)
    success = int(loc_df.sort_values("xlag_gain_vs_recursive", ascending=False).iloc[0]["location_index"])
    failure = int(loc_df.sort_values("xlag_gain_vs_recursive", ascending=True).iloc[0]["location_index"])
    loc_df.to_csv(OUT / "location_success_failure_scores.csv", index=False)
    make_prediction_figure(preds, success, failure, selected_lag, kernel, q)
    return success, failure, loc_df


def make_prediction_figure(preds: dict[str, pd.DataFrame], success: int, failure: int, lag: int, kernel: str, q: int) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    colors_map = {"xlag": "#4C8A4A", "recursive": "#D2842A", "oracle": "#8E5C87"}
    labels = {"xlag": f"X-lag, L={lag}, {kernel}" + (f" Q={q}" if kernel == "spectral_mixture" else ""), "recursive": "Recursive Y-lag, RBF", "oracle": "Oracle Y-lag, RBF"}
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True, constrained_layout=True)
    for ax, loc, title in zip(axes, [success, failure], ["Representative success case", "Representative failure case"]):
        base = preds["xlag"][preds["xlag"]["location_index"] == loc].sort_values("time_index")
        x = base["time_index"].to_numpy()
        ax.axvspan(x.min(), x.max(), color="#F5F7FA", zorder=0, label="held-out/test time series")
        ax.plot(x, base["y_true"], color="#222222", lw=1.7, label="ground truth")
        for key in ["recursive", "xlag", "oracle"]:
            g = preds[key][preds[key]["location_index"] == loc].sort_values("time_index")
            mean = g["pred_mean"].to_numpy()
            std = np.sqrt(np.maximum(g["pred_var_y"].to_numpy(), 1e-10))
            ax.plot(x, mean, color=colors_map[key], lw=1.1, label=labels[key])
            ax.fill_between(x, mean - 1.645 * std, mean + 1.645 * std, color=colors_map[key], alpha=0.10, linewidth=0)
        ax.set_title(f"{title}: held-out location {loc}", loc="left")
        ax.set_ylabel("scaled target")
        ax.grid(alpha=0.22)
        ax.text(
            0.01,
            0.04,
            "Training region: other spatial locations used for online updates; shown location is held out.",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
            ha="left",
            va="bottom",
        )
    axes[0].legend(ncol=2, fontsize=8, loc="upper right")
    axes[1].set_xlabel("time index in task_2")
    fig.savefig(FIGS / "representative_success_failure_fit.png", dpi=260, bbox_inches="tight")
    fig.savefig(FIGS / "representative_success_failure_fit.pdf", bbox_inches="tight")
    plt.close(fig)


def make_summary_figures(lag_df: pd.DataFrame, kernel_df: pd.DataFrame) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    axes[0].plot(lag_df["lag_length"], lag_df["val_rmse"], marker="o", color="#3B6EA8", label="validation")
    axes[0].plot(lag_df["lag_length"], lag_df["test_rmse"], marker="s", color="#4C8A4A", label="test")
    axes[0].set_xlabel("lag length L")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Lag-length validation")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(lag_df["lag_length"], lag_df["val_nll"], marker="o", color="#3B6EA8", label="validation")
    axes[1].plot(lag_df["lag_length"], lag_df["test_nll"], marker="s", color="#4C8A4A", label="test")
    axes[1].set_xlabel("lag length L")
    axes[1].set_ylabel("NLL/NLPD")
    axes[1].set_title("Probabilistic score")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(FIGS / "lag_length_validation.png", dpi=260, bbox_inches="tight")
    fig.savefig(FIGS / "lag_length_validation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.2), constrained_layout=True)
    labels = [
        f"{r.method}\n{r.kernel_type}" + (f"\nQ={int(r.num_mixtures)}" if int(r.num_mixtures) else "")
        for r in kernel_df.itertuples()
    ]
    colors = ["#4C8A4A" if m == "X-lag" else "#D2842A" if m == "recursive Y-lag" else "#8E5C87" for m in kernel_df["method"]]
    bars = ax.bar(np.arange(len(kernel_df)), kernel_df["test_rmse"], color=colors)
    ax.set_xticks(np.arange(len(kernel_df)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("test RMSE")
    ax.set_title("Kernel ablation at selected lag")
    ax.grid(axis="y", alpha=0.25)
    for b, value in zip(bars, kernel_df["test_rmse"]):
        ax.text(b.get_x() + b.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(FIGS / "kernel_ablation_test_rmse.png", dpi=260, bbox_inches="tight")
    fig.savefig(FIGS / "kernel_ablation_test_rmse.pdf", bbox_inches="tight")
    plt.close(fig)


def register_fonts() -> str:
    arial = Path("/mnt/c/Windows/Fonts/arial.ttf")
    if arial.exists():
        pdfmetrics.registerFont(TTFont("ArialLocal", str(arial)))
        return "ArialLocal"
    return "Helvetica"


def table(data: list[list[Any]], widths: list[float], font: str, fontsize: float = 7.0) -> Table:
    style = ParagraphStyle("cell", fontName=font, fontSize=fontsize, leading=fontsize + 2, alignment=TA_LEFT, wordWrap="CJK")
    wrapped = [[Paragraph(str(c), style) for c in row] for row in data]
    t = Table(wrapped, colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F6")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#222222")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#D6DCE2")),
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def make_report(lag_df: pd.DataFrame, kernel_df: pd.DataFrame, selected: int, success: int, failure: int) -> None:
    font = register_fonts()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Title2", fontName=font, fontSize=17, leading=22, spaceAfter=10))
    styles.add(ParagraphStyle("H2", fontName=font, fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle("Body2", fontName=font, fontSize=8.7, leading=12.2, spaceAfter=5))
    styles.add(ParagraphStyle("Caption2", fontName=font, fontSize=7.2, leading=9.5, textColor=colors.HexColor("#444444"), spaceAfter=8))
    pdf = OUT / "medium_era5_xlag_lag_kernel_validation_report.pdf"
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=1.35 * cm, rightMargin=1.35 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm)
    story = [
        Paragraph("Medium-ERA5 X-lag lag-length and kernel validation", styles["Title2"]),
        Paragraph("This diagnostic tests whether the exogenous-lag feature map should use more than one lag and whether changing the GP kernel improves predictive RMSE or NLL. Validation is held-out split seed 0 and test is held-out split seed 1. The protocol is deliberately lightweight and keeps the Route B training loop, likelihood, optimizer, split logic and metrics unchanged.", styles["Body2"]),
        Paragraph("Lag-length validation", styles["H2"]),
    ]
    lag_table = [["L", "Val RMSE", "Val NLL", "Test RMSE", "Test NLL", "Runtime (s)", "Comment"]]
    for r in lag_df.itertuples():
        lag_table.append([int(r.lag_length), f"{r.val_rmse:.4f}", f"{r.val_nll:.4f}", f"{r.test_rmse:.4f}", f"{r.test_nll:.4f}", f"{r.runtime:.1f}", r.stability_comment])
    story.append(table(lag_table, [0.9 * cm, 2.0 * cm, 2.0 * cm, 2.1 * cm, 2.0 * cm, 2.0 * cm, 5.7 * cm], font))
    story.append(Paragraph(f"Selected lag length: L={selected}, chosen by validation NLL with validation RMSE used as the tie-breaker.", styles["Body2"]))
    story.append(Image(str(FIGS / "lag_length_validation.png"), width=16.6 * cm, height=6.6 * cm))
    story.append(Paragraph("Figure 1. Validation and test metrics across exogenous lag length L.", styles["Caption2"]))
    story.append(Paragraph("Kernel ablation", styles["H2"]))
    kernel_table = [["Method", "Kernel", "L", "Q", "Val RMSE", "Val NLL", "Test RMSE", "Test NLL", "Runtime (s)"]]
    for r in kernel_df.itertuples():
        kernel_table.append([r.method, r.kernel_type, int(r.lag_length), int(r.num_mixtures), f"{r.val_rmse:.4f}", f"{r.val_nll:.4f}", f"{r.test_rmse:.4f}", f"{r.test_nll:.4f}", f"{r.runtime:.1f}"])
    story.append(table(kernel_table, [2.7 * cm, 2.8 * cm, 0.8 * cm, 0.8 * cm, 1.8 * cm, 1.8 * cm, 1.9 * cm, 1.8 * cm, 1.8 * cm], font, fontsize=6.6))
    story.append(Paragraph("The recursive Y-lag row is the corrected non-cheating target-lag protocol. The oracle Y-lag row is included only as a leakage upper bound and should not be interpreted as a deployable model.", styles["Body2"]))
    story.append(Image(str(FIGS / "kernel_ablation_test_rmse.png"), width=15.5 * cm, height=7.2 * cm))
    story.append(Paragraph("Figure 2. Test RMSE for kernel and protocol ablations at the selected lag length.", styles["Caption2"]))
    story.append(Paragraph("Standardized visualization", styles["H2"]))
    story.append(Paragraph(f"The success and failure cases are selected by the per-location RMSE gain of X-lag over recursive Y-lag on test split seed {TEST_SEED}. Success location: {success}. Failure location: {failure}. Each panel marks the shown sequence as a held-out/test location; online updates use the remaining training locations.", styles["Body2"]))
    story.append(Image(str(FIGS / "representative_success_failure_fit.png"), width=16.8 * cm, height=12.0 * cm))
    story.append(Paragraph("Figure 3. Representative success and failure cases with ground truth, predictive means and 90% uncertainty bands.", styles["Caption2"]))
    story.append(Paragraph("Code changes", styles["H2"]))
    story.append(Paragraph("The exact code diff is saved as code_diff.patch in this report directory. The main changes are: a trainable torch ARD spectral-mixture kernel with softplus constraints and tests; configurable X-lag length; and configurable default spectral-mixture component count. RBF remains the default.", styles["Body2"]))
    doc.build(story)


def save_code_diff() -> None:
    tracked = subprocess.run(
        ["git", "diff", "--", "scripts/run_hipposvgp_era5_routeb.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout
    new_file_diffs = []
    for path in [
        "stvgp_kronecker/spectral_mixture_kernel.py",
        "stvgp_kronecker/tests/test_spectral_mixture_kernel.py",
        "scripts/run_medium_era5_xlag_lag_kernel_validation.py",
    ]:
        proc = subprocess.run(
            ["git", "diff", "--no-index", "--", "/dev/null", path],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        new_file_diffs.append(proc.stdout)
    (OUT / "code_diff.patch").write_text(tracked + "\n".join(new_file_diffs), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    lag_df, selected = lag_validation()
    kernel_df = kernel_ablation(selected)
    make_summary_figures(lag_df, kernel_df)
    success, failure, _ = prediction_exports(selected, kernel_df)
    save_code_diff()
    make_report(lag_df, kernel_df, selected, success, failure)
    print(f"Selected lag length L={selected}")
    print(f"Report: {OUT / 'medium_era5_xlag_lag_kernel_validation_report.pdf'}")
    print(f"Lag CSV: {OUT / 'lag_length_validation_results.csv'}")
    print(f"Kernel CSV: {OUT / 'kernel_ablation_results.csv'}")


if __name__ == "__main__":
    main()
