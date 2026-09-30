#!/usr/bin/env python3
"""Generate the four-step ICLR-style priority experiment report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path("results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/four_step_priority_experiments")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def first_summary_row(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    return rows[0] if rows else {}


def collect_step1() -> list[dict[str, Any]]:
    rows = []
    for mt, ms in [(8, 64), (16, 64), (16, 128), (32, 128), (32, 256)]:
        row = first_summary_row(ROOT / "step1_routeb_capacity" / f"Mt{mt}_Ms{ms}" / "era5_routeb_summary.csv")
        if not row:
            continue
        rows.append(
            {
                "setting": f"Mt={mt}, Ms={ms}",
                "mt": mt,
                "ms": ms,
                "rmse": float(row["rmse"]),
                "nll": float(row["nll"]),
                "coverage90": float(row["coverage90"]),
                "avg_std": float(row["avg_std"]),
                "runtime_per_block": float(row["runtime_per_block"]),
            }
        )
    return rows


def collect_step2(step1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    baseline = next((row for row in step1_rows if row["mt"] == 16 and row["ms"] == 128), None)
    if baseline is not None:
        rows.append(
            {
                "setting": "linspace, RBF, ls=0.35",
                "selection": "linspace",
                "kernel": "rbf",
                "lengthscale": 0.35,
                **{k: baseline[k] for k in ["rmse", "nll", "coverage90", "avg_std", "runtime_per_block"]},
            }
        )
    mapping = {
        "farthest_rbf_ls035": ("farthest, RBF, ls=0.35", "farthest", "rbf", 0.35),
        "kmeans_rbf_ls035": ("k-means, RBF, ls=0.35", "kmeans", "rbf", 0.35),
        "linspace_rbf_ls100": ("linspace, RBF, ls=1.0", "linspace", "rbf", 1.0),
        "farthest_matern32_ls035": ("farthest, Matern32, ls=0.35", "farthest", "matern32", 0.35),
        "farthest_matern32_ls100": ("farthest, Matern32, ls=1.0", "farthest", "matern32", 1.0),
    }
    for dirname, meta in mapping.items():
        row = first_summary_row(ROOT / "step2_spatial_enhancement" / dirname / "era5_routeb_summary.csv")
        if not row:
            continue
        label, selection, kernel, ls = meta
        rows.append(
            {
                "setting": label,
                "selection": selection,
                "kernel": kernel,
                "lengthscale": ls,
                "rmse": float(row["rmse"]),
                "nll": float(row["nll"]),
                "coverage90": float(row["coverage90"]),
                "avg_std": float(row["avg_std"]),
                "runtime_per_block": float(row["runtime_per_block"]),
            }
        )
    local_rows = read_csv(ROOT / "step2_spatial_enhancement" / "nearest_neighbor_local_correction_summary.csv")
    for row in local_rows:
        rows.append(
            {
                "setting": "nearest-neighbour local correction (final block)",
                "selection": "local NN",
                "kernel": "none",
                "lengthscale": "",
                "rmse": float(row["rmse"]),
                "nll": float(row["nll"]),
                "coverage90": float(row["coverage90"]),
                "avg_std": float(row["avg_std"]),
                "runtime_per_block": float(row["runtime"]),
            }
        )
    return rows


def collect_step3() -> list[dict[str, Any]]:
    return [
        {
            "method": row["method"],
            "rmse": float(row["rmse"]),
            "nll": float(row["nll"]),
            "coverage90": float(row["coverage90"]),
            "avg_std": float(row["avg_std"]),
            "runtime": float(row["runtime"]),
        }
        for row in read_csv(ROOT / "step3_joint_stvgp" / "joint_stvgp_final_block_summary.csv")
    ]


def make_bar(rows: list[dict[str, Any]], label_key: str, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row[label_key]) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    axes[0].bar(x, [float(row["rmse"]) for row in rows], color="#6B8791")
    axes[0].set_title(title + " RMSE")
    axes[0].set_ylabel("RMSE")
    axes[1].bar(x, [float(row["nll"]) for row in rows], color="#9A7B72")
    axes[1].set_title(title + " NLL")
    axes[1].set_ylabel("NLL")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def add_table(story: list[Any], rows: list[list[Any]], widths: list[float]) -> None:
    table = Table(rows, colWidths=[w * inch for w in widths])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF2")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C8CDD2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)


def build_report(step1: list[dict[str, Any]], step2: list[dict[str, Any]], step3: list[dict[str, Any]]) -> Path:
    outdir = ROOT
    figdir = outdir / "figures"
    make_bar(step1, "setting", "Step 1: Route B capacity", figdir / "step1_capacity.png")
    make_bar(step2, "setting", "Step 2: Spatial enhancement", figdir / "step2_spatial.png")
    make_bar(step3, "method", "Step 3: Exact STVGP joint mean", figdir / "step3_joint.png")

    best_routeb = min(step1, key=lambda row: row["rmse"])
    exact_upper = {"method": "Exact separable STVGP residual + X-lag mean", "rmse": 0.08425008985346832, "nll": -1.0919581426461562}
    exact_joint = min(step3, key=lambda row: row["rmse"])
    step4 = [
        {"method": "Exact STVGP two-stage upper bound", **exact_upper},
        {"method": "Joint exact STVGP upper bound", "rmse": exact_joint["rmse"], "nll": exact_joint["nll"]},
        {"method": f"Best streaming Route B ({best_routeb['setting']})", "rmse": best_routeb["rmse"], "nll": best_routeb["nll"]},
    ]
    make_bar(step4, "method", "Step 4: Streaming gap", figdir / "step4_streaming_gap.png")

    write_csv(step1, outdir / "step1_routeb_capacity_summary.csv")
    write_csv(step2, outdir / "step2_spatial_enhancement_summary.csv")
    write_csv(step3, outdir / "step3_joint_stvgp_summary.csv")
    write_csv(step4, outdir / "step4_streaming_gap_summary.csv")

    pdf_path = outdir / "iclr_four_step_priority_experiment_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch)
    styles = getSampleStyleSheet()
    story: list[Any] = []
    story.append(Paragraph("Four-step priority experiments for HiPPO-STVGP", styles["Title"]))
    story.append(Paragraph("Re-centering Route B around the spatio-temporal residual posterior", styles["Heading2"]))
    story.append(Paragraph("The experiments below separate three questions: whether Route B improves with inducing capacity, whether the bottleneck is spatial residual kriging, and whether the structured joint mean-GP idea becomes stronger when attached to an exact STVGP backbone. X-lag is treated throughout as a meteorological mean/covariate component rather than as the proposed method itself.", styles["BodyText"]))

    story.append(Paragraph("Step 1: Route B capacity path", styles["Heading2"]))
    story.append(Paragraph("We first increased the temporal and spatial inducing budgets under the same streaming Route B protocol. The monotone RMSE improvement shows that the previous 0.19 result was not a hard methodological limit; a substantial part of the gap to exact STVGP is sparse-capacity related. NLL improves up to the high-capacity regime but coverage decreases, indicating that variance calibration becomes the next issue.", styles["BodyText"]))
    add_table(story, [["Setting", "RMSE", "NLL", "Cov90", "Std", "sec/block"]] + [[r["setting"], fmt(r["rmse"]), fmt(r["nll"]), fmt(r["coverage90"]), fmt(r["avg_std"]), fmt(r["runtime_per_block"])] for r in step1], [1.4, 0.65, 0.65, 0.6, 0.55, 0.65])
    story.append(Spacer(1, 0.08 * inch))
    story.append(Image(str(figdir / "step1_capacity.png"), width=7.0 * inch, height=2.25 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Step 2: Spatial enhancement", styles["Heading2"]))
    story.append(Paragraph("At fixed capacity (Mt=16, Ms=128), we varied the spatial inducing geometry, spatial lengthscale and kernel family. K-means and farthest-point inducing locations give small RMSE gains over linspace, but the gains are much smaller than those obtained by increasing capacity. A larger spatial lengthscale hurts RMSE, while Matérn-3/2 mainly inflates predictive variance and does not close the gap to exact STVGP. A nearest-neighbour local residual diagnostic is strong in the final-block setting, suggesting that local same-time spatial information is valuable and should be incorporated more principledly.", styles["BodyText"]))
    add_table(story, [["Setting", "RMSE", "NLL", "Cov90", "Std"]] + [[r["setting"], fmt(r["rmse"]), fmt(r["nll"]), fmt(r["coverage90"]), fmt(r["avg_std"])] for r in step2], [2.7, 0.65, 0.65, 0.6, 0.55])
    story.append(Spacer(1, 0.08 * inch))
    story.append(Image(str(figdir / "step2_spatial.png"), width=7.0 * inch, height=2.25 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Step 3: STVGP + structured joint mean", styles["Heading2"]))
    story.append(Paragraph("We then tested the more ambitious idea: attach the structured joint mean-GP coupling to the exact STVGP backbone rather than treating beta as a two-stage ridge fit. In a final seen-history diagnostic, the joint exact STVGP model improves both RMSE and NLL relative to the exact STVGP residual two-stage baseline. This is the strongest new signal in the experiment: the joint beta-GP coupling is not merely useful for Route B transfer, but can strengthen the strongest STVGP backbone itself.", styles["BodyText"]))
    add_table(story, [["Method", "RMSE", "NLL", "Cov90", "Std", "runtime"]] + [[r["method"], fmt(r["rmse"]), fmt(r["nll"]), fmt(r["coverage90"]), fmt(r["avg_std"]), fmt(r["runtime"])] for r in step3], [2.8, 0.65, 0.65, 0.6, 0.55, 0.65])
    story.append(Spacer(1, 0.08 * inch))
    story.append(Image(str(figdir / "step3_joint.png"), width=7.0 * inch, height=2.25 * inch))

    story.append(Paragraph("Step 4: Returning to streaming", styles["Heading2"]))
    story.append(Paragraph("Because Step 3 shows a real gain from joint coupling, the next principled target is an online/streaming version of joint STVGP. The current streaming realization is Route B with sparse HiPPO temporal and sparse spatial inducing variables. At Mt=32, Ms=256 it reaches RMSE 0.1259, considerably closer to the exact STVGP upper bound than the original 0.1899 setting, but still above the joint exact STVGP diagnostic. This motivates a streaming joint-STVGP extension with stronger spatial/local residual correction and explicit variance calibration.", styles["BodyText"]))
    add_table(story, [["Reference", "RMSE", "NLL"]] + [[r["method"], fmt(r["rmse"]), fmt(r["nll"])] for r in step4], [3.6, 0.75, 0.75])
    story.append(Spacer(1, 0.08 * inch))
    story.append(Image(str(figdir / "step4_streaming_gap.png"), width=7.0 * inch, height=2.25 * inch))

    story.append(Paragraph("Overall conclusion", styles["Heading2"]))
    story.append(Paragraph("The evidence now supports a revised direction. Route B should be defended under sparse/streaming constraints, where capacity scaling already moves it from 0.1899 to 0.1259 RMSE. However, the most promising research result is that structured joint beta-GP coupling improves the exact STVGP backbone itself in the final-block diagnostic. The next development should therefore combine both directions: keep Route B as the sparse streaming approximation, but formulate a stronger joint-STVGP upper-bound model and then design a streaming approximation to it.", styles["BodyText"]))
    doc.build(story)

    (outdir / "iclr_four_step_priority_report.json").write_text(json.dumps({"pdf": str(pdf_path), "step1": step1, "step2": step2, "step3": step3, "step4": step4}, indent=2), encoding="utf-8")
    return pdf_path


def main() -> None:
    step1 = collect_step1()
    step2 = collect_step2(step1)
    step3 = collect_step3()
    pdf = build_report(step1, step2, step3)
    print(pdf)


if __name__ == "__main__":
    main()
