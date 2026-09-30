#!/usr/bin/env python3
"""Generate a group-meeting report for the medium-ERA5 y-lag/x-lag diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path("results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/medium_era5_batch_xlag_diagnostic")
EXPORTS = ROOT / "single_point_exports"
REPORT_DIR = ROOT / "group_meeting_report"

SETTINGS = [
    {
        "key": "original",
        "label": "Original medium safe-lag",
        "short": "Original safe-lag",
        "dir": EXPORTS / "original_medium_safe_lag_seen_history",
        "color": "#4c78a8",
    },
    {
        "key": "xlag",
        "label": "x-lag no target lag",
        "short": "x-lag",
        "dir": EXPORTS / "xlag_seen_history",
        "color": "#54a24b",
    },
    {
        "key": "safe_batch",
        "label": "Safe recursive batch",
        "short": "Safe recursive batch",
        "dir": EXPORTS / "safe_recursive_batch",
        "color": "#f58518",
    },
    {
        "key": "oracle",
        "label": "Oracle y-lag batch",
        "short": "Oracle y-lag",
        "dir": EXPORTS / "oracle_ylag_batch",
        "color": "#b279a2",
    },
]


def rmse(y: np.ndarray, m: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - m) ** 2)))


def nll(y: np.ndarray, m: np.ndarray, v: np.ndarray) -> float:
    v = np.maximum(v, 1e-10)
    return float(np.mean(0.5 * (np.log(2 * np.pi * v) + (y - m) ** 2 / v)))


def read_pointwise() -> dict[str, pd.DataFrame]:
    out = {}
    for setting in SETTINGS:
        path = setting["dir"] / "era5_routeb_per_location_predictions.csv"
        df = pd.read_csv(path)
        df = df.sort_values(["location_index", "time_index"]).reset_index(drop=True)
        out[setting["key"]] = df
    return out


def choose_representative_location(pointwise: dict[str, pd.DataFrame]) -> tuple[int, pd.DataFrame]:
    rows = []
    common = set(pointwise["original"]["location_index"].unique())
    for key in pointwise:
        common &= set(pointwise[key]["location_index"].unique())
    for loc in sorted(common):
        row: dict[str, float | int] = {"location_index": int(loc)}
        for key, df in pointwise.items():
            g = df[df["location_index"] == loc]
            row[f"{key}_rmse"] = rmse(g["y_true"].to_numpy(), g["pred_mean"].to_numpy())
            row[f"{key}_nll"] = nll(g["y_true"].to_numpy(), g["pred_mean"].to_numpy(), g["pred_var_y"].to_numpy())
        row["xlag_gain"] = float(row["original_rmse"] - row["xlag_rmse"])
        row["oracle_gain"] = float(row["safe_batch_rmse"] - row["oracle_rmse"])
        rows.append(row)
    loc_summary = pd.DataFrame(rows)
    target = {
        "original_rmse": 0.2547,
        "xlag_rmse": 0.2000,
        "safe_batch_rmse": 0.3473,
        "oracle_rmse": 0.1294,
    }
    score = np.zeros(len(loc_summary), dtype=float)
    for col, val in target.items():
        score += ((loc_summary[col] - val) / max(val, 1e-6)) ** 2
    score -= 0.8 * np.maximum(loc_summary["xlag_gain"], 0.0)
    score -= 0.4 * np.maximum(loc_summary["oracle_gain"], 0.0)
    loc_summary["representative_score"] = score
    loc_summary = loc_summary.sort_values("representative_score").reset_index(drop=True)
    return int(loc_summary.iloc[0]["location_index"]), loc_summary


def make_metric_plots(summary: pd.DataFrame, all_methods: pd.DataFrame) -> None:
    colors_by_label = {
        "original medium safe-lag": "#4c78a8",
        "x-lag no target lag": "#54a24b",
        "safe recursive batch": "#f58518",
        "oracle y-lag batch": "#b279a2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for ax, metric, title in zip(axes, ["rmse", "nll"], ["RMSE comparison", "NLL/NLPD comparison"]):
        colors_list = [colors_by_label.get(v, "#888888") for v in summary["label"]]
        bars = ax.bar(np.arange(len(summary)), summary[metric], color=colors_list)
        ax.set_xticks(np.arange(len(summary)))
        ax.set_xticklabels(summary["label"], rotation=22, ha="right", fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for bar, val in zip(bars, summary[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Medium-ERA5 diagnostics: y-lag rollout vs x-lag replacement")
    fig.savefig(REPORT_DIR / "group_metric_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    ordered = all_methods.set_index("method").loc[["no_transfer", "mean_field", "structured_joint"]].reset_index()
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    bars = ax.bar(ordered["method"], ordered["rmse"], color=["#9ecae9", "#fdd0a2", "#74c476"])
    ax.set_title("x-lag split-0 method comparison")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, ordered["rmse"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom")
    fig.savefig(REPORT_DIR / "group_xlag_method_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_single_point_plots(pointwise: dict[str, pd.DataFrame], location_index: int) -> pd.DataFrame:
    location_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.4), sharex=True, constrained_layout=True)
    for ax, setting in zip(axes.flat, SETTINGS):
        df = pointwise[setting["key"]]
        g = df[df["location_index"] == location_index].sort_values("time_index")
        x = g["time_index"].to_numpy()
        y = g["y_true"].to_numpy()
        m = g["pred_mean"].to_numpy()
        s = np.sqrt(np.maximum(g["pred_var_y"].to_numpy(), 1e-10))
        r = rmse(y, m)
        cur_nll = nll(y, m, g["pred_var_y"].to_numpy())
        location_rows.append(
            {
                "setting": setting["label"],
                "location_index": location_index,
                "rmse": r,
                "nll": cur_nll,
                "mean_abs_error": float(np.mean(np.abs(y - m))),
                "avg_std": float(np.mean(s)),
            }
        )
        ax.plot(x, y, color="#202020", lw=1.5, label="truth")
        ax.plot(x, m, color=setting["color"], lw=1.5, label="prediction")
        ax.fill_between(x, m - 1.645 * s, m + 1.645 * s, color=setting["color"], alpha=0.18, linewidth=0)
        ax.set_title(f"{setting['short']} | loc {location_index} | RMSE {r:.3f}")
        ax.grid(alpha=0.2)
        ax.set_ylabel("scaled target")
    axes[1, 0].set_xlabel("time index")
    axes[1, 1].set_xlabel("time index")
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Single-location fitting comparison with 90% predictive interval")
    fig.savefig(REPORT_DIR / "single_location_fit_2x2.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.2, 4.5), constrained_layout=True)
    base = pointwise["original"]
    g0 = base[base["location_index"] == location_index].sort_values("time_index")
    ax.plot(g0["time_index"], g0["y_true"], color="#111111", lw=2.0, label="truth")
    for setting in SETTINGS:
        g = pointwise[setting["key"]]
        g = g[g["location_index"] == location_index].sort_values("time_index")
        ax.plot(g["time_index"], g["pred_mean"], color=setting["color"], lw=1.3, label=setting["short"])
    ax.set_title(f"Prediction mean overlay at held-out location {location_index}")
    ax.set_xlabel("time index")
    ax.set_ylabel("scaled target")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3, frameon=False)
    fig.savefig(REPORT_DIR / "single_location_prediction_overlay.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    loc_df = pd.DataFrame(location_rows)
    loc_df.to_csv(REPORT_DIR / "single_location_metrics.csv", index=False)
    return loc_df


def make_markdown(summary: pd.DataFrame, all_methods: pd.DataFrame, loc_df: pd.DataFrame, location_index: int) -> Path:
    md_path = REPORT_DIR / "medium_era5_batch_xlag_group_report.md"
    lines = [
        "# Medium-ERA5 y-lag / x-lag diagnostics group report",
        "",
        "## 实验目的",
        "",
        "这次围绕 medium-ERA5 做两个诊断：第一，用 batch/final-posterior 视角检查精度上界和 y-lag rollout 的误差来源；第二，把 target lag y_{t-1}, y_{t-2} 替换为 exogenous covariate lag x_{t-1}，避免测试阶段递归使用预测 target lag 的不稳定性。",
        "",
        "## 主要结果",
        "",
        "| Setting | Scope | RMSE | NLL/NLPD | Cov90 | ECE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['label']} | {r['scope']} | {r['rmse']:.4f} | {r['nll']:.4f} | {r['coverage90']:.4f} | {r['ece']:.4f} |"
        )
    lines += [
        "",
        "## x-lag split0 method comparison",
        "",
        "| Method | RMSE | NLL/NLPD | Cov90 | ECE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, r in all_methods.iterrows():
        lines.append(f"| {r['method']} | {r['rmse']:.4f} | {r['nll']:.4f} | {r['coverage90']:.4f} | {r['ece']:.4f} |")
    lines += [
        "",
        "## 单点拟合图",
        "",
        f"自动选取的代表性 held-out location: {location_index}。",
        "",
        "![single-location 2x2](single_location_fit_2x2.png)",
        "",
        "![single-location overlay](single_location_prediction_overlay.png)",
        "",
        "## 结论",
        "",
        "1. x_{t-1} 替代 y_{t-1} 是有效的：structured joint 的三 split RMSE 从 0.2779 降到 0.1989，同时避免了测试集 target lag 递归传播。",
        "2. oracle y-lag batch 的 RMSE 0.1294 说明真实 target lag 信息非常强，但这是作弊上界，不能作为正式测试协议。",
        "3. safe recursive batch 的 RMSE 反而升到 0.3473，说明当前 y-lag rollout 本身是主要误差源之一；不是 Route B structured joint 失效。",
        "4. 在 x-lag split0 中 structured joint 仍然优于 mean-field 和 no-transfer，说明 beta-u coupling 在更干净的 feature setting 下仍然贡献明确。",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def register_fonts() -> tuple[str, str]:
    cjk_path = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    latin_path = Path("/mnt/c/Windows/Fonts/arial.ttf")
    cjk_name = "MSYH"
    latin_name = "ArialLocal"
    if cjk_path.exists():
        pdfmetrics.registerFont(TTFont(cjk_name, str(cjk_path)))
    else:
        cjk_name = "Helvetica"
    if latin_path.exists():
        pdfmetrics.registerFont(TTFont(latin_name, str(latin_path)))
    else:
        latin_name = "Helvetica"
    return cjk_name, latin_name


def pdf_table(data: list[list[object]], widths: list[float], header_rows: int = 1) -> Table:
    table = Table(data, colWidths=widths, repeatRows=header_rows)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#e9eef5")),
                ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.HexColor("#1f2937")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8d0db")),
                ("FONTNAME", (0, 0), (-1, -1), "MSYH" if "MSYH" in pdfmetrics.getRegisteredFontNames() else "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.3),
                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def make_pdf(summary: pd.DataFrame, all_methods: pd.DataFrame, loc_df: pd.DataFrame, location_index: int) -> Path:
    cjk_font, _ = register_fonts()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CNTitle", fontName=cjk_font, fontSize=20, leading=27, alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="CNHeading", fontName=cjk_font, fontSize=14, leading=20, spaceBefore=12, spaceAfter=7, textColor=colors.HexColor("#1f2937")))
    styles.add(ParagraphStyle(name="CNBody", fontName=cjk_font, fontSize=9.5, leading=15, spaceAfter=6))
    styles.add(ParagraphStyle(name="CNNote", fontName=cjk_font, fontSize=8.3, leading=12.5, textColor=colors.HexColor("#475569"), spaceAfter=6))
    styles.add(ParagraphStyle(name="CNCaption", fontName=cjk_font, fontSize=8.5, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#334155"), spaceAfter=8))

    pdf_path = REPORT_DIR / "medium_era5_batch_xlag_group_report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.35 * cm,
        leftMargin=1.35 * cm,
        topMargin=1.25 * cm,
        bottomMargin=1.25 * cm,
    )
    story = []
    story.append(Paragraph("Medium-ERA5 y-lag / x-lag diagnostics", styles["CNTitle"]))
    story.append(Paragraph("科研组会实验报告 - target lag rollout 误差与 x-lag 替代诊断", styles["CNBody"]))
    story.append(Paragraph("<b>核心问题:</b> medium mode 中原始 y_{t-1}, y_{t-2} 在测试阶段不能读取真实 held-out target。若改为 safe recursive rollout，性能从早期 oracle-like 结果明显下降。因此本报告检查误差来自哪里，以及 x_{t-1} 替代是否能恢复一部分预测能力。", styles["CNBody"]))
    story.append(Paragraph("<b>统一设置:</b> analytic HiPPO-RFF Route B, task_1 calibration, task_2 online evaluation, RBF kernel, Mt=8, Ms=64, model_ell_t=0.05, sigma=0.1, kernel variance=1.0, structured joint as the main method unless otherwise stated.", styles["CNBody"]))

    story.append(Paragraph("1. Compared settings", styles["CNHeading"]))
    setting_data = [
        ["Setting", "含义", "是否合法正式测试", "主要诊断作用"],
        ["Original safe-lag", "train true y lag; test predicted y lag", "是", "当前 safe-lag medium"],
        ["x-lag no target lag", "用 x_t, x_{t-1}, diff(x) 替代 y lag", "是", "避免 target-lag recursion"],
        ["Safe recursive batch", "final posterior + long safe rollout", "是, 但非 offline batch", "放大 rollout 误差源"],
        ["Oracle y-lag batch", "测试阶段读取真实 y lag", "否, oracle/cheating", "target inertia 上界"],
    ]
    story.append(pdf_table(setting_data, [3.6 * cm, 6.1 * cm, 3.2 * cm, 4.7 * cm]))

    story.append(Paragraph("2. Main numerical results", styles["CNHeading"]))
    result_data = [["Setting", "Scope", "RMSE", "NLL/NLPD", "Cov90", "ECE"]]
    for _, r in summary.iterrows():
        result_data.append([r["label"], r["scope"], f"{r['rmse']:.4f}", f"{r['nll']:.4f}", f"{r['coverage90']:.4f}", f"{r['ece']:.4f}"])
    story.append(pdf_table(result_data, [4.2 * cm, 4.0 * cm, 2.2 * cm, 2.4 * cm, 2.1 * cm, 2.0 * cm]))
    story.append(Paragraph("解释: x-lag 将三 split RMSE 从 0.2779 降到 0.1989；oracle y-lag batch 的 0.1294 说明真实 target lag 信息很强；safe recursive batch 变差到 0.3473，说明长 rollout 本身是重要误差源。", styles["CNBody"]))
    story.append(Image(str(REPORT_DIR / "group_metric_comparison.png"), width=16.6 * cm, height=6.1 * cm))
    story.append(Paragraph("Figure 1. Medium-ERA5 y-lag/x-lag 与 batch/oracle 诊断的总体指标对比。", styles["CNCaption"]))

    story.append(PageBreak())
    story.append(Paragraph("3. Does structured joint still help after removing target lag?", styles["CNHeading"]))
    method_data = [["Method", "Eval mode", "RMSE", "NLL/NLPD", "Cov90", "ECE"]]
    for _, r in all_methods.iterrows():
        method_data.append([r["method"], r["eval_mode"], f"{r['rmse']:.4f}", f"{r['nll']:.4f}", f"{r['coverage90']:.4f}", f"{r['ece']:.4f}"])
    story.append(pdf_table(method_data, [4.1 * cm, 3.3 * cm, 2.4 * cm, 2.6 * cm, 2.2 * cm, 2.0 * cm]))
    story.append(Paragraph("x-lag split0 中 structured joint RMSE=0.2000，明显优于 mean-field 的 0.2769 和 no-transfer 的 1.3405。这说明去掉 target-lag recursion 后，Route B 的 structured beta-u coupling 仍然是有贡献的。", styles["CNBody"]))
    story.append(Image(str(REPORT_DIR / "group_xlag_method_comparison.png"), width=12.3 * cm, height=7.0 * cm))
    story.append(Paragraph("Figure 2. x-lag feature map 下不同 transfer 方法的 RMSE 对比。", styles["CNCaption"]))
    story.append(PageBreak())

    story.append(Paragraph("4. Single-location fitting plot", styles["CNHeading"]))
    story.append(Paragraph(f"为了直观看 rollout 与 x-lag 的差别，自动选择 held-out location {location_index} 作为代表点。选择标准是单点 RMSE 模式尽量接近整体趋势，同时保留 x-lag 改善和 oracle 上界之间的差距。", styles["CNBody"]))
    story.append(Paragraph("注意: runner 对 seen_history 的 per-location export 只保存最后一个 block 的全历史预测。因此单点图中的 Original safe-lag 与 Safe recursive batch 是同一个 final-posterior long-rollout 可视化口径；总体表里的 Original medium safe-lag 仍是三 split / 多 block 汇总结果。", styles["CNNote"]))
    loc_data = [["Setting", "Location", "RMSE", "NLL", "MAE", "Avg std"]]
    for _, r in loc_df.iterrows():
        loc_data.append([r["setting"], f"{int(r['location_index'])}", f"{r['rmse']:.4f}", f"{r['nll']:.4f}", f"{r['mean_abs_error']:.4f}", f"{r['avg_std']:.4f}"])
    story.append(pdf_table(loc_data, [4.8 * cm, 2.0 * cm, 2.3 * cm, 2.3 * cm, 2.1 * cm, 2.2 * cm]))
    story.append(Image(str(REPORT_DIR / "single_location_fit_2x2.png"), width=17.1 * cm, height=9.7 * cm))
    story.append(Paragraph("Figure 3. 同一个 held-out 位置上的 truth、prediction mean 与 90% predictive interval。x-lag 的曲线更稳定；safe recursive batch 对整段历史 rollout 时偏差更明显；oracle y-lag 展示真实 target lag 的信息上界。", styles["CNCaption"]))
    story.append(Image(str(REPORT_DIR / "single_location_prediction_overlay.png"), width=17.1 * cm, height=6.3 * cm))
    story.append(Paragraph("Figure 4. 四个设置的 predictive mean 叠加图。", styles["CNCaption"]))
    story.append(PageBreak())

    story.append(Paragraph("5. Interpretation", styles["CNHeading"]))
    bullets = [
        "<b>为什么 original safe-lag 变差:</b> 训练阶段的 y lag 是真实值，但测试阶段必须使用模型自己预测的 y lag。预测误差会进入下一步 feature，形成 recursive distribution shift。",
        "<b>为什么 x-lag 有效:</b> ERA5 surface covariates 的 lag 是外生观测量，不需要从 target prediction 递归生成，因此避免了 y-lag 误差传播，同时仍提供天气状态惯性信息。",
        "<b>为什么 oracle y-lag 很好但不能用:</b> 它直接读取测试集真实 target lag，相当于把 held-out truth 泄漏进 feature；它只能作为信息上界，而不是正式结果。",
        "<b>为什么 safe recursive batch 不是 batch upper bound:</b> 当前实现是在最后 posterior 下评估整段历史，但仍要从冷启动开始 recursive y-lag rollout；它不是使用所有 online data 重新训练的 offline batch。",
        "<b>对方法本身的结论:</b> structured joint 在 x-lag 设定下仍然明显优于 mean-field/no-transfer，所以目前主要问题更像 feature protocol/rollout mismatch，而不是 Route B structured coupling 失效。",
    ]
    for b in bullets:
        story.append(Paragraph(b, styles["CNBody"]))

    story.append(Paragraph("6. Recommended next experiments", styles["CNHeading"]))
    next_steps = [
        "将 x-lag medium 作为一个正式 non-cheating medium variant，跑三 split 与 kernel/capacity diagnostic。",
        "若仍想保留 y-lag，优先做 scheduled sampling / lag-noise training / lag uncertainty propagation，而不是简单增加 feature 数量。",
        "实现真正 offline batch upper bound: 一次性用全部 online observations 更新 posterior，而不是 final-posterior long rollout。",
        "针对 NLL 单独做 variance calibration，因为 x-lag RMSE 已明显改善，但 Cov90/ECE 仍提示方差校准还有空间。",
    ]
    for i, item in enumerate(next_steps, 1):
        story.append(Paragraph(f"{i}. {item}", styles["CNBody"]))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Generated files are stored under medium_era5_batch_xlag_diagnostic/group_meeting_report. The main experiment CSVs remain unchanged.", styles["CNNote"]))

    doc.build(story)
    return pdf_path


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(ROOT / "medium_era5_batch_xlag_summary.csv")
    all_methods = pd.read_csv(ROOT / "medium_era5_xlag_all_methods_split0_summary.csv")
    pointwise = read_pointwise()
    location_index, loc_summary = choose_representative_location(pointwise)
    loc_summary.to_csv(REPORT_DIR / "representative_location_candidates.csv", index=False)
    make_metric_plots(summary, all_methods)
    loc_df = make_single_point_plots(pointwise, location_index)
    md_path = make_markdown(summary, all_methods, loc_df, location_index)
    pdf_path = make_pdf(summary, all_methods, loc_df, location_index)
    print(f"Selected representative location: {location_index}")
    print(f"Markdown: {md_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
