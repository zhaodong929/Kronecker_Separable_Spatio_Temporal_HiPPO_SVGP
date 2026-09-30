#!/usr/bin/env python3
"""Generate the ICLR Formal experiment report for STVGP baseline integration."""

from __future__ import annotations

import json
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
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path("results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment")
BASELINE = ROOT / "stvgp_era5_baseline"
PDF = ROOT / "iclr_formal_stvgp_baseline_report.pdf"
MD = ROOT / "iclr_formal_stvgp_baseline_report.md"
FIGS = ROOT / "figures"


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def table(data: list[list[object]], col_widths: list[float] | None = None) -> Table:
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8B8B8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def read_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    st = pd.read_csv(BASELINE / "stvgp_era5_baseline_summary.csv")
    xlag = pd.read_csv(
        "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/xlag_lag_kernel_validation/kernel_ablation_results.csv"
    )
    return st, xlag


def build_comparison(st: pd.DataFrame, xlag: pd.DataFrame) -> pd.DataFrame:
    xref = xlag[(xlag["method"] == "X-lag") & (xlag["kernel_type"] == "rbf")].iloc[0]
    rows = [
        {
            "method": "Route B X-lag + RBF",
            "type": "current streaming/Kronecker reference",
            "val_rmse": float(xref["val_rmse"]),
            "val_nll": float(xref["val_nll"]),
            "test_rmse": float(xref["test_rmse"]),
            "test_nll": float(xref["test_nll"]),
            "coverage90": np.nan,
            "runtime_per_block": np.nan,
        }
    ]
    labels = {
        "stvgp_exact_zero": "STVGP-style separable GP",
        "stvgp_exact_xlag_ridge": "STVGP-style residual + X-lag mean",
    }
    for method, group in st.groupby("method"):
        val = group[group["heldout_split_seed"] == 0].iloc[0]
        test = group[group["heldout_split_seed"] == 1].iloc[0]
        rows.append(
            {
                "method": labels.get(method, method),
                "type": "literature-style exact separable seen-history refit",
                "val_rmse": float(val["rmse"]),
                "val_nll": float(val["nll"]),
                "test_rmse": float(test["rmse"]),
                "test_nll": float(test["nll"]),
                "coverage90": float(test["coverage90"]),
                "runtime_per_block": float(test["runtime_per_block"]),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "iclr_formal_stvgp_comparison.csv", index=False)
    return df


def make_figures(comp: pd.DataFrame) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    labels = comp["method"].tolist()
    x = np.arange(len(labels))
    colors = ["#3D6B45", "#527BA8", "#8A6F9E"][: len(labels)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), constrained_layout=True)
    axes[0].bar(x, comp["test_rmse"], color=colors)
    axes[0].set_ylabel("test RMSE")
    axes[0].set_title("Held-out seen-history mean accuracy")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, comp["test_nll"], color=colors)
    axes[1].set_ylabel("test NLL/NLPD")
    axes[1].set_title("Held-out seen-history probabilistic score")
    axes[1].grid(axis="y", alpha=0.25)
    for ax, metric in zip(axes, ["test_rmse", "test_nll"]):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        for i, v in enumerate(comp[metric]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(FIGS / "formal_stvgp_vs_routeb.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIGS / "formal_stvgp_vs_routeb.pdf", bbox_inches="tight")
    plt.close(fig)


def write_markdown(comp: pd.DataFrame) -> None:
    def simple_markdown_table(df: pd.DataFrame) -> str:
        rows = []
        cols = list(df.columns)
        rows.append("| " + " | ".join(cols) + " |")
        rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    vals.append("" if pd.isna(value) else f"{value:.4f}")
                else:
                    vals.append(str(value))
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join(rows)

    lines = [
        "# ICLR Formal experiment: STVGP literature baseline for Medium-ERA5",
        "",
        "## Reference method",
        "",
        "The current default reference is Route B structured joint with Medium-ERA5 X-lag features and an RBF residual kernel: test RMSE 0.1899, test NLL -0.0350.",
        "",
        "## External baseline status",
        "",
        "The official AaltoML/spatio-temporal-GPs repository was cloned under `baselines/external/aaltoml_spatio_temporal_gps` at commit `c5b929e`. Its official Bayes-Newton experiment script was smoke-tested and failed in the current Python environment because `bayesnewton` is unavailable; the official requirements target Python 3.7-era JAX/Objax/Bayes-Newton versions.",
        "",
        "## Comparable local baseline",
        "",
        "A local STVGP-style baseline was implemented with a separable Matern-3/2 temporal/spatial covariance and Kronecker eigensystem inference. It is a literature-style seen-history refit baseline, not a streaming method.",
        "",
        simple_markdown_table(comp),
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")


def build_pdf(comp: pd.DataFrame) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=10)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=12.2, leading=15, spaceBefore=8, spaceAfter=5)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.8, leading=11.2, alignment=TA_LEFT, spaceAfter=5)
    small = ParagraphStyle("Small", parent=body, fontSize=7.2, leading=8.6)

    doc = SimpleDocTemplate(
        str(PDF),
        pagesize=A4,
        leftMargin=1.55 * cm,
        rightMargin=1.55 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
    )
    story: list[object] = []
    story.append(Paragraph("ICLR Formal experiment: STVGP literature baseline for Medium-ERA5", title))
    story.append(
        para(
            "This report starts the transition from internal temporal architecture diagnostics to a literature-facing "
            "spatio-temporal GP comparison. Spectral-mixture experiments are stopped here; the default reference is "
            "Route B structured joint with Medium-ERA5 X-lag features and an RBF residual kernel.",
            body,
        )
    )
    story.append(Paragraph("Protocol", h1))
    story.append(
        para(
            "Reference result: X-lag + RBF, validation split seed 0 and test split seed 1. The reference test metrics "
            "are RMSE 0.1899 and NLL -0.0350. The literature baseline uses the same processed ERA5 task_2, the same "
            "20% held-out spatial split convention, block size 10, and seen-history evaluation.",
            body,
        )
    )
    story.append(Paragraph("AaltoML / Solin Group Baseline", h1))
    story.append(
        para(
            "The likely target paper is Hamelijnck et al., Spatio-Temporal Variational Gaussian Processes, NeurIPS 2021. "
            "The official repository AaltoML/spatio-temporal-GPs was cloned to baselines/external/aaltoml_spatio_temporal_gps "
            "at commit c5b929e. The Bayes-Newton experiment defines a separable spatio-temporal model with a Matern-3/2 "
            "temporal kernel, separable Matern-3/2 spatial kernel, k-means spatial inducing points, and MarkovVariationalGP "
            "or MarkovVariationalMeanFieldGP inference.",
            body,
        )
    )
    story.append(
        para(
            "Official-code smoke check: the original script fails in this environment at import bayesnewton. This is expected "
            "because the official requirements pin Python 3.7-era JAX, Objax, Bayes-Newton, GPflow and TensorFlow versions, "
            "whereas the current project environment is Python 3.11 and only has GPyTorch among those GP libraries. The failure "
            "is logged in official_stvgp_smoke_stdout.log.",
            body,
        )
    )
    story.append(Paragraph("Local STVGP-Style Adapter", h1))
    story.append(
        para(
            "To obtain a comparable ERA5 number now, I implemented a local STVGP-style exact separable baseline. It uses "
            "Matern-3/2 kernels in time and space and exploits the full grid structure through Kronecker eigensystems. Two "
            "variants are reported: raw separable STVGP, and an X-lag ridge mean followed by a separable GP on the residual. "
            "This is a strong seen-history refit baseline, not a streaming update method.",
            body,
        )
    )
    story.append(Image(str(FIGS / "formal_stvgp_vs_routeb.png"), width=16.5 * cm, height=5.9 * cm))
    story.append(Paragraph("Main Results", h1))
    rows = [[para("Method", small), para("Type", small), "Val RMSE", "Val NLL", "Test RMSE", "Test NLL", "Cov90", "RT/block"]]
    for _, row in comp.iterrows():
        rows.append(
            [
                para(str(row["method"]), small),
                para(str(row["type"]), small),
                f"{row['val_rmse']:.4f}",
                f"{row['val_nll']:.4f}",
                f"{row['test_rmse']:.4f}",
                f"{row['test_nll']:.4f}",
                "" if pd.isna(row["coverage90"]) else f"{row['coverage90']:.4f}",
                "" if pd.isna(row["runtime_per_block"]) else f"{row['runtime_per_block']:.4f}",
            ]
        )
    story.append(table(rows, col_widths=[3.6 * cm, 4.05 * cm, 1.55 * cm, 1.55 * cm, 1.65 * cm, 1.55 * cm, 1.35 * cm, 1.55 * cm]))
    story.append(Paragraph("Interpretation", h1))
    story.append(
        para(
            "The raw separable STVGP baseline is already stronger than the current Route B X-lag + RBF reference on this "
            "held-out seen-history task. This is plausible because it performs direct spatial interpolation at the same "
            "observed times using the full grid covariance. Adding the same X-lag information as a ridge mean makes the "
            "baseline much stronger, reaching test RMSE 0.0815 and NLL -1.1262. This result is not a failure of the Route B "
            "idea; it clarifies that the next comparison must distinguish batch seen-history spatial interpolation from "
            "streaming transfer with memory constraints.",
            body,
        )
    )
    story.append(Paragraph("Next Steps", h1))
    story.append(
        para(
            "The next formal step is to run the same table on reduced compute-controlled settings: fixed inducing budget, "
            "no full refit per block, and then Route B with analytic HiPPO-RFF temporal interdomain features. A second line "
            "should implement the actual Bayes-Newton STVGP once a Python 3.7-compatible environment is available, so the "
            "official natural-gradient filtering implementation can be compared directly against our streaming Kronecker method.",
            body,
        )
    )
    doc.build(story)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    st, xlag = read_results()
    comp = build_comparison(st, xlag)
    make_figures(comp)
    write_markdown(comp)
    build_pdf(comp)
    print(PDF)


if __name__ == "__main__":
    main()
