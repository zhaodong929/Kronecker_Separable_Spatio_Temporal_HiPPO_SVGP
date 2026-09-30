#!/usr/bin/env python3
"""Append fairness diagnostics to the ICLR Formal STVGP baseline report."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

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
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path("results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment")
FAIR = ROOT / "fairness_diagnostics"
BASE_REPORT = ROOT / "iclr_formal_stvgp_baseline_report.pdf"
APPENDIX = FAIR / "iclr_formal_fairness_appendix.pdf"
FINAL = ROOT / "iclr_formal_stvgp_baseline_report_with_fairness_appendix.pdf"
FIGS = FAIR / "figures"


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def table(data: list[list[object]], widths: list[float] | None = None) -> Table:
    tbl = Table(data, colWidths=widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8B8B8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.0),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def parse_routeb_name(name: str) -> tuple[int, int, float]:
    m = re.search(r"Mt(\d+)_Ms(\d+)_ls(\d+)", name)
    if not m:
        return 0, 0, float("nan")
    mt = int(m.group(1))
    ms = int(m.group(2))
    ls_raw = m.group(3)
    ls = float(ls_raw) / (100.0 if len(ls_raw) == 3 else 1.0)
    return mt, ms, ls


def load_routeb_batch() -> pd.DataFrame:
    rows = []
    for path in sorted((FAIR / "routeb_batch_upper").glob("*/era5_routeb_summary.csv")):
        row = pd.read_csv(path).iloc[0].to_dict()
        mt, ms, ls = parse_routeb_name(path.parent.name)
        rows.append(
            {
                "setting": path.parent.name,
                "mt": mt,
                "ms": ms,
                "spatial_lengthscale": ls,
                "rmse": float(row["rmse"]),
                "nll": float(row["nll"]),
                "coverage90": float(row["coverage90"]),
                "avg_std": float(row["avg_std"]),
                "runtime_per_block": float(row["runtime_per_block"]),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(FAIR / "routeb_batch_upper_summary.csv", index=False)
    return df


def load_stvgp_schedules() -> pd.DataFrame:
    df = pd.read_csv(FAIR / "stvgp_schedules/stvgp_era5_baseline_summary.csv")
    df = df[["method", "eval_mode", "mean_mode", "rmse", "nll", "coverage90", "avg_std", "runtime_per_block"]].copy()
    df.to_csv(FAIR / "stvgp_schedule_summary.csv", index=False)
    return df


def build_residual_table(routeb: pd.DataFrame, stvgp: pd.DataFrame) -> pd.DataFrame:
    best_routeb = routeb.sort_values("rmse").iloc[0]
    st_exact = stvgp[(stvgp["method"] == "stvgp_exact_xlag_ridge") & (stvgp["eval_mode"] == "seen_history")].iloc[0]
    mean_only = stvgp[
        (stvgp["method"].str.contains("xlag_mean_only"))
        & (stvgp["eval_mode"] == "seen_history")
    ].iloc[0]
    rows = [
        {
            "residual_model": "X-lag ridge mean only",
            "protocol": "seen-history refit mean, no residual GP",
            "rmse": float(mean_only["rmse"]),
            "nll": float(mean_only["nll"]),
        },
        {
            "residual_model": "Route B X-lag + RBF",
            "protocol": "original streaming/Kronecker reference",
            "rmse": 0.1899366567512086,
            "nll": -0.035034236524608,
        },
        {
            "residual_model": "Route B batch high capacity",
            "protocol": str(best_routeb["setting"]),
            "rmse": float(best_routeb["rmse"]),
            "nll": float(best_routeb["nll"]),
        },
        {
            "residual_model": "STVGP exact separable residual",
            "protocol": "seen-history exact refit",
            "rmse": float(st_exact["rmse"]),
            "nll": float(st_exact["nll"]),
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(FAIR / "residual_only_diagnostic_summary.csv", index=False)
    return df


def make_figures(routeb: pd.DataFrame, stvgp: pd.DataFrame, residual: pd.DataFrame) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    rb = routeb.sort_values("rmse")
    labels = rb["setting"].str.replace("_", "\n").tolist()
    x = np.arange(len(rb))
    axes[0].bar(x, rb["rmse"], color="#6B8BA4")
    axes[0].axhline(0.0814793791921591, color="black", linestyle="--", linewidth=1, label="STVGP residual")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Route B batch/final-state upper-bound")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].bar(x, rb["nll"], color="#8A6F9E")
    axes[1].axhline(-1.126186239925016, color="black", linestyle="--", linewidth=1, label="STVGP residual")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("NLL")
    axes[1].set_title("Probabilistic score")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.savefig(FIGS / "routeb_batch_upper_bound.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    subset = stvgp[stvgp["method"].eq("stvgp_exact_xlag_ridge")].copy()
    labels = subset["eval_mode"].tolist()
    x = np.arange(len(subset))
    axes[0].bar(x, subset["rmse"], color="#527BA8")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("STVGP under streaming constraints")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, subset["nll"], color="#8A6F9E")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("NLL")
    axes[1].set_title("Calibration changes")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(FIGS / "stvgp_streaming_constraint.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    x = np.arange(len(residual))
    ax.bar(x, residual["rmse"], color=["#999999", "#3D6B45", "#6B8BA4", "#8A6F9E"])
    ax.set_xticks(x)
    ax.set_xticklabels(residual["residual_model"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("Residual-only diagnostic")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIGS / "residual_only_diagnostic.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_appendix(routeb: pd.DataFrame, stvgp: pd.DataFrame, residual: pd.DataFrame) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=16, leading=20, spaceAfter=10)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=12.2, leading=15, spaceBefore=8, spaceAfter=5)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.8, leading=11.2, alignment=TA_LEFT, spaceAfter=5)
    small = ParagraphStyle("Small", parent=body, fontSize=7.0, leading=8.3)
    doc = SimpleDocTemplate(
        str(APPENDIX),
        pagesize=A4,
        leftMargin=1.55 * cm,
        rightMargin=1.55 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
    )
    story: list[object] = []
    story.append(Paragraph("Fairness diagnostics: refit advantage, streaming constraints, and residual structure", title))
    story.append(
        para(
            "This appendix adds the three fairness checks requested after the first STVGP comparison. The goal is not to declare "
            "Route B failed, but to isolate whether the observed gap comes from full seen-history refitting, residual spatial "
            "interpolation, or the Route B streaming/Kronecker approximation.",
            body,
        )
    )
    story.append(Paragraph("1. Route B Batch/Final-State Upper Bound", h1))
    story.append(Image(str(FIGS / "routeb_batch_upper_bound.png"), width=16.5 * cm, height=6.2 * cm))
    rows = [["Setting", "Mt", "Ms", "space ell", "RMSE", "NLL", "Cov90", "avg std"]]
    for _, r in routeb.sort_values("rmse").iterrows():
        rows.append([r["setting"], int(r["mt"]), int(r["ms"]), f"{r['spatial_lengthscale']:.2f}", f"{r['rmse']:.4f}", f"{r['nll']:.4f}", f"{r['coverage90']:.4f}", f"{r['avg_std']:.4f}"])
    story.append(table(rows))
    story.append(
        para(
            "Increasing Route B capacity from Mt=8, Ms=64 to Mt=16, Ms=256 improves the final-state batch result from RMSE 0.2794 "
            "to 0.1839 and improves NLL to -0.2175. However, it remains far from the exact STVGP residual refit result "
            "(RMSE 0.0815, NLL -1.1262). This indicates that capacity helps, but the gap is not explained only by the small "
            "default inducing budget.",
            body,
        )
    )
    story.append(Paragraph("2. Streaming-Constrained STVGP", h1))
    story.append(Image(str(FIGS / "stvgp_streaming_constraint.png"), width=16.5 * cm, height=6.2 * cm))
    rows = [["Method", "Eval mode", "RMSE", "NLL", "Cov90", "avg std"]]
    for _, r in stvgp[stvgp["method"].eq("stvgp_exact_xlag_ridge")].iterrows():
        rows.append([r["method"], r["eval_mode"], f"{r['rmse']:.4f}", f"{r['nll']:.4f}", f"{r['coverage90']:.4f}", f"{r['avg_std']:.4f}"])
    story.append(table(rows))
    story.append(
        para(
            "The constrained result is sharper than expected. With X-lag mean plus STVGP residual, current-block refit gives "
            "RMSE 0.0777, slightly better than the seen-history refit RMSE 0.0815, although its NLL/coverage are less well "
            "calibrated. In contrast, the initial-only variant collapses. This means the STVGP advantage is not primarily a "
            "full-history advantage; it mainly comes from same-time spatial interpolation using the currently observed training "
            "locations. The task is therefore strongly spatial-kriging dominated.",
            body,
        )
    )
    story.append(Paragraph("3. Residual-Only Diagnostic", h1))
    story.append(Image(str(FIGS / "residual_only_diagnostic.png"), width=14.8 * cm, height=5.9 * cm))
    rows = [["Residual model", "Protocol", "RMSE", "NLL"]]
    for _, r in residual.iterrows():
        rows.append([para(str(r["residual_model"]), small), para(str(r["protocol"]), small), f"{r['rmse']:.4f}", f"{r['nll']:.4f}"])
    story.append(table(rows, widths=[4.0 * cm, 7.3 * cm, 2.0 * cm, 2.0 * cm]))
    story.append(
        para(
            "The X-lag mean-only model has RMSE 0.2999 under the seen-history refit protocol, so the strong STVGP result is not "
            "coming from the linear X-lag mean alone. Exact STVGP residual modelling improves this to 0.0815. Route B high-capacity "
            "batch improves over the original Route B reference, but remains far from the exact residual baseline. This points to "
            "the Route B residual/spatial projection as the main bottleneck under this held-out spatial interpolation protocol.",
            body,
        )
    )
    story.append(Paragraph("Conclusion", h1))
    story.append(
        para(
            "The fairer interpretation is now sharper: Route B is not losing because X-lag features are weak, nor is the gap "
            "explained mainly by full-history refitting. The dominant gap comes from residual same-time spatial interpolation. "
            "The next experiment should therefore redesign or strengthen Route B's spatial residual projection so that it better "
            "preserves local kriging information at held-out locations, then compare under a matched no-full-refit budget.",
            body,
        )
    )
    doc.build(story)


def main() -> None:
    FAIR.mkdir(parents=True, exist_ok=True)
    routeb = load_routeb_batch()
    stvgp = load_stvgp_schedules()
    residual = build_residual_table(routeb, stvgp)
    make_figures(routeb, stvgp, residual)
    build_appendix(routeb, stvgp, residual)
    subprocess.run(["pdfunite", str(BASE_REPORT), str(APPENDIX), str(FINAL)], check=True)
    print(FINAL)


if __name__ == "__main__":
    main()
