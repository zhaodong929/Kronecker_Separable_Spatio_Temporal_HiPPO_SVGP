#!/usr/bin/env python3
"""Generate a Chinese PDF report for the Route B efficiency diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean, stdev
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#245A7A")
TEAL = colors.HexColor("#2A7F79")
PALE_BLUE = colors.HexColor("#EAF2F7")
PALE_GREEN = colors.HexColor("#E8F3EF")
PALE_RED = colors.HexColor("#FBECEC")
LIGHT_GREY = colors.HexColor("#F5F7F8")
MID_GREY = colors.HexColor("#D6DEE3")
DARK_GREY = colors.HexColor("#33434D")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def average(rows: list[dict[str, str]], field: str) -> float | None:
    values = [number(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return mean(finite) if finite else None


def fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def sci(value: float | None) -> str:
    if value is None:
        return "NA"
    if value == 0.0:
        return "0"
    return f"{value:.2e}"


def mean_sd(rows: list[dict[str, str]], field: str, digits: int = 4) -> str:
    values = [number(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    if not finite:
        return "NA"
    spread = stdev(finite) if len(finite) > 1 else 0.0
    return f"{mean(finite):.{digits}f} +/- {spread:.{digits}f}"


def register_fonts() -> tuple[str, str]:
    candidates = [
        (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/msyhbd.ttc")),
        (Path("/mnt/c/Windows/Fonts/msyh.ttc"), Path("/mnt/c/Windows/Fonts/msyhbd.ttc")),
    ]
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont("ReportCJK", str(regular), subfontIndex=0))
            pdfmetrics.registerFont(TTFont("ReportCJKBold", str(bold), subfontIndex=0))
            return "ReportCJK", "ReportCJKBold"
    raise FileNotFoundError("Microsoft YaHei font was not found")


def p(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)).replace("\n", "<br/>"), style)


def build_styles(font: str, bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN", parent=base["Title"], fontName=bold, fontSize=24,
            leading=31, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8 * mm,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN", parent=base["Normal"], fontName=font, fontSize=11,
            leading=17, textColor=DARK_GREY, spaceAfter=4 * mm,
        ),
        "h1": ParagraphStyle(
            "H1CN", parent=base["Heading1"], fontName=bold, fontSize=16,
            leading=21, textColor=NAVY, spaceBefore=3 * mm, spaceAfter=3 * mm,
        ),
        "h2": ParagraphStyle(
            "H2CN", parent=base["Heading2"], fontName=bold, fontSize=11.5,
            leading=15, textColor=BLUE, spaceBefore=2.5 * mm, spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "BodyCN", parent=base["BodyText"], fontName=font, fontSize=9.2,
            leading=14, textColor=DARK_GREY, spaceAfter=2.5 * mm,
        ),
        "small": ParagraphStyle(
            "SmallCN", parent=base["BodyText"], fontName=font, fontSize=7.5,
            leading=10, textColor=DARK_GREY,
        ),
        "table": ParagraphStyle(
            "TableCN", parent=base["BodyText"], fontName=font, fontSize=6.6,
            leading=8.2, textColor=colors.HexColor("#1D2B33"), alignment=TA_CENTER,
        ),
        "table_left": ParagraphStyle(
            "TableLeftCN", parent=base["BodyText"], fontName=font, fontSize=6.6,
            leading=8.2, textColor=colors.HexColor("#1D2B33"), alignment=TA_LEFT,
        ),
        "table_head": ParagraphStyle(
            "TableHeadCN", parent=base["BodyText"], fontName=bold, fontSize=6.5,
            leading=8, textColor=colors.white, alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "CalloutCN", parent=base["BodyText"], fontName=bold, fontSize=10.5,
            leading=15, textColor=NAVY, alignment=TA_LEFT,
        ),
    }


def report_table(
    headers: list[str],
    rows: list[list[object]],
    styles: dict[str, ParagraphStyle],
    widths: list[float] | None = None,
    left_columns: tuple[int, ...] = (0,),
    highlights: dict[tuple[int, int], colors.Color] | None = None,
) -> Table:
    data: list[list[Paragraph]] = [
        [p(header, styles["table_head"]) for header in headers]
    ]
    for row in rows:
        data.append([
            p(cell, styles["table_left"] if index in left_columns else styles["table"])
            for index, cell in enumerate(row)
        ])
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
    commands: list[tuple] = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.35, MID_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
    ]
    for row_index in range(1, len(data)):
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index),
                         colors.white if row_index % 2 else LIGHT_GREY))
    for column in left_columns:
        commands.append(("ALIGN", (column, 1), (column, -1), "LEFT"))
    for (row_index, column), color in (highlights or {}).items():
        commands.append(("BACKGROUND", (column, row_index + 1), (column, row_index + 1), color))
    table.setStyle(TableStyle(commands))
    return table


def page_decor(canvas, doc) -> None:
    canvas.saveState()
    width, height = landscape(A4)
    canvas.setStrokeColor(MID_GREY)
    canvas.setLineWidth(0.45)
    canvas.line(16 * mm, height - 12 * mm, width - 16 * mm, height - 12 * mm)
    canvas.setFont("ReportCJK", 7.5)
    canvas.setFillColor(colors.HexColor("#60727D"))
    canvas.drawString(16 * mm, height - 9 * mm, "Route B 效率优化完整实验报告")
    canvas.drawRightString(width - 16 * mm, 8 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def grouped(rows: list[dict[str, str]], fields: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    result: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def generate(root: Path, output: Path) -> None:
    font, bold = register_fonts()
    styles = build_styles(font, bold)
    unified = read_csv(root / "unified_efficiency_table.csv")
    parity = read_csv(root / "summary/paired_parity.csv")
    e0e3 = read_csv(root / "exact_e0_e3_ablation.csv")
    rank_accuracy = read_csv(root / "rank_accuracy_efficiency.csv")
    rank_projection = read_csv(root / "rank/future_projection_error.csv")
    hyperparameters = read_csv(root / "summary/learned_hyperparameters.csv")
    operator_rows = read_csv(root / "operator_breakdown.csv")

    output.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = landscape(A4)
    doc = BaseDocTemplate(
        str(output), pagesize=landscape(A4),
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=17 * mm, bottomMargin=14 * mm,
        title="Route B 效率优化完整实验报告",
        author="Route B experiment audit",
        subject="ERA5 batch and strict-online efficiency ablations",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=page_decor)])
    story: list[object] = []

    story.extend([
        Spacer(1, 13 * mm),
        p("Route B 效率优化完整实验报告", styles["title"]),
        p("ERA5 Task 2 short 与 Tasks 2-10 long：Batch empirical Bayes、strict-online cumulative HiPPO、FLOPs、显存与近似 rank 消融", styles["subtitle"]),
        Spacer(1, 4 * mm),
    ])
    summary_boxes = [
        ["最终精确版本", "E1 缓存固定统计量 + 自动 contraction"],
        ["Long DTC FLOPs", "163.887 -> 108.774 GFLOPs/step (-33.6%)"],
        ["Long VFE 总训练时间", "207.23 -> 162.56 s (-21.6%)"],
        ["严格在线 long", "13.917 GFLOPs/block；DTC 校准 RMSE 0.1376"],
    ]
    box_table = Table(
        [[p(label, styles["callout"]), p(value, styles["body"])] for label, value in summary_boxes],
        colWidths=[48 * mm, 132 * mm], hAlign="LEFT",
    )
    box_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), PALE_BLUE),
        ("BACKGROUND", (1, 0), (1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.7, MID_GREY),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, MID_GREY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.extend([
        box_table,
        Spacer(1, 8 * mm),
        p("Coverage90 的数学定义（不是 ECE）", styles["h2"]),
        p(
            "Coverage90 = (1/N) sum_i 1{ y_i 位于 [mu_i - 1.64485 sigma_i, mu_i + 1.64485 sigma_i] }。"
            "其中 mu_i 和 sigma_i 是第 i 个测试点的预测均值和预测标准差，1.64485 是标准正态分布的 95% 分位数。"
            "因此 Coverage90 直接统计真实值落入名义 90% 预测区间的比例；理想值接近 0.90。",
            styles["body"],
        ),
        p(
            "ECE 是把多个置信水平下的名义覆盖率与经验覆盖率进行汇总后的校准误差，是另一个指标。"
            "本报告表格中的 Cov90/Coverage90 均使用上面的单一 90% 区间覆盖率公式，不是 ECE。",
            styles["body"],
        ),
        Spacer(1, 2 * mm),
        p("报告日期：2026-08-06。硬件：NVIDIA GeForce RTX 5070 Laptop GPU（8 GB），PyTorch 2.11.0+cu128，WSL2，float64。所有正式运行与 profiler 分离。", styles["body"]),
        p("重要口径：PyTorch Profiler with_flops=True 只统计支持的 ATen 算子。解析补充只覆盖主要 forward Cholesky/EVD/solve；两者之和仍是 lower bound，不是完整硬件 FLOPs。", styles["body"]),
        PageBreak(),
    ])

    story.extend([p("1. 实验协议与验收标准", styles["h1"]), p(
        "本实验固定数据、空间划分、mask、初始化、超参数优化器和随机基频，只改变计算实现或训练目标。Batch 每个版本先 warm-up 10 次，再独立测量至少 30 次 objective forward+backward；正式训练为 Adam 100 steps，学习率 0.02，每 5 steps 按 validation NLL 选择 checkpoint。",
        styles["body"],
    )])
    protocol_rows = [
        ["数据范围", "Task 2 short", "186 小时"],
        ["数据范围", "Tasks 2-10 long", "1,674 小时"],
        ["空间划分", "fit / validation / test", "720 / 80 / 200 locations"],
        ["随机种子", "split seeds", "0, 1, 2, 3, 4"],
        ["Route B 状态", "M_t / M_s / X-lag features", "128 / 128 / 133"],
        ["训练目标", "finite DTC 与 VFE", "均使用 full structured-joint conditional (D) prediction"],
        ["在线协议", "Task-1 EB calibration 后冻结 theta", "Task 2: 19 blocks；Tasks 2-10: 171 blocks；block size 10"],
        ["计时", "runtime 与 profiler 分开", "10 warm-up；30 timed repeats；GPU 同步"],
    ]
    story.append(report_table(["类别", "设置", "取值"], protocol_rows, styles,
                              widths=[42 * mm, 85 * mm, 135 * mm], left_columns=(0, 1)))
    story.extend([
        Spacer(1, 3 * mm),
        p("精确一致性阈值：objective 相对误差 < 1e-8；gradient 相对误差 < 1e-7；posterior mean 相对误差 < 1e-8，并同时检查 covariance、NLL、conditional residual、有限值与负方差。", styles["body"]),
        PageBreak(),
    ])

    batch = [row for row in unified if row["method"].startswith("Route B batch")]
    batch_groups = grouped(batch, ("scope", "objective", "version"))
    batch_rows: list[list[object]] = []
    batch_order = [
        (scope, objective, version)
        for scope in ("task2_short", "tasks2_10_long")
        for objective in ("finite_dtc", "vfe")
        for version in ("original", "optimized")
    ]
    for key in batch_order:
        scope, objective, version = key
        rows = batch_groups[key]
        counted = average(rows, "profiler_counted_gflops_per_unit")
        supplement = average(rows, "analytical_forward_supplement_gflops_per_unit")
        counted_total = average(rows, "profiler_counted_total_gflops")
        lower_total = average(rows, "profiler_plus_forward_lower_bound_total_gflops")
        batch_rows.append([
            "Task 2" if scope == "task2_short" else "Tasks 2-10",
            "DTC" if objective == "finite_dtc" else "VFE+D",
            "Original" if version == "original" else "Optimized",
            fmt(counted), fmt(supplement),
            f"{fmt(counted_total / 1000 if counted_total else None)}/{fmt(lower_total / 1000 if lower_total else None)}",
            fmt(average(rows, "runtime_per_unit_seconds")),
            fmt(average(rows, "training_or_stream_runtime_seconds"), 2),
            f"{fmt(average(rows, 'peak_allocated_mib'), 0)}/{fmt(average(rows, 'peak_reserved_mib'), 0)}",
            fmt(average(rows, "rmse"), 4), fmt(average(rows, "nll"), 4),
            fmt(average(rows, "coverage90"), 4),
        ])
    story.extend([
        p("2. 最终 Batch 精确优化结果", styles["h1"]),
        p("表 1 报告 seeds 0-4 平均。Counted/lower-bound TFLOPs 分别表示 profiler 总量与加上解析 forward 补充后的 lower bound。", styles["body"]),
        report_table(
            ["Scope", "Objective", "Version", "Profiler\nGFLOPs/step", "Analytical +\nGFLOPs/step", "Counted/lower\nTFLOPs", "s/step", "Training s", "Peak MiB\nA/R", "RMSE", "NLL", "Cov90"],
            batch_rows, styles,
            widths=[25*mm, 21*mm, 24*mm, 24*mm, 24*mm, 29*mm, 19*mm, 22*mm, 25*mm, 18*mm, 18*mm, 18*mm],
            left_columns=(0, 1, 2),
        ),
        Spacer(1, 3 * mm),
    ])

    change_rows: list[list[object]] = []
    for scope in ("task2_short", "tasks2_10_long"):
        for objective in ("finite_dtc", "vfe"):
            original = batch_groups[(scope, objective, "original")]
            optimized = batch_groups[(scope, objective, "optimized")]
            old_flops = average(original, "profiler_counted_gflops_per_unit")
            new_flops = average(optimized, "profiler_counted_gflops_per_unit")
            old_time = average(original, "training_or_stream_runtime_seconds")
            new_time = average(optimized, "training_or_stream_runtime_seconds")
            change_rows.append([
                "Task 2" if scope == "task2_short" else "Tasks 2-10",
                "DTC" if objective == "finite_dtc" else "VFE+D",
                f"{100*(old_flops-new_flops)/old_flops:.1f}%",
                f"{100*(old_time-new_time)/old_time:.1f}%",
                fmt(average(optimized, "rmse") - average(original, "rmse"), 7),
                fmt(average(optimized, "nll") - average(original, "nll"), 7),
            ])
    story.append(report_table(
        ["Scope", "Objective", "Profiler FLOPs 降低", "总训练时间降低", "Delta RMSE", "Delta NLL"],
        change_rows, styles, widths=[35*mm, 30*mm, 42*mm, 42*mm, 36*mm, 36*mm], left_columns=(0, 1),
    ))
    story.extend([
        Spacer(1, 3 * mm),
        p("结论：缓存固定 Phi^T Phi、Phi^T y、y^T y 对 long setting 最有效。最终指标差异处于数值误差范围，但 long optimized 的 reserved memory 增至约 7,672 MiB，说明更快的 contraction 使用了更大的 CUDA workspace。", styles["body"]),
        PageBreak(),
    ])

    story.extend([p("3. E0-E3 逐项精确消融", styles["h1"]), p(
        "该表为 seed 0 单步 objective 微基准。E2/E3 在 long scope 通过，但在 Task 2 未通过统一 covariance 验收，因此不能作为跨 scope 的论文默认精确实现。",
        styles["body"],
    )])
    e_rows: list[list[object]] = []
    flags = {"E0": ("x", "x", "x"), "E1": ("yes", "x", "x"),
             "E2": ("yes", "yes", "x"), "E3": ("yes", "yes", "yes")}
    for row in e0e3:
        cached, combined, removed = flags[row["version"]]
        e_rows.append([
            "Task 2" if row["scope"] == "task2_short" else "Tasks 2-10",
            row["version"], cached, combined, removed,
            fmt(number(row["profiler_counted_gflops_per_step"])),
            fmt(number(row["steady_runtime_seconds_per_step"]), 4),
            sci(number(row["objective_relative_error"])), sci(number(row["gradient_relative_error"])),
            sci(number(row["posterior_mean_relative_error"])), sci(number(row["beta_covariance_relative_error"])),
            "PASS" if row["passes_exact_parity"] == "True" else "FAIL",
        ])
    story.append(report_table(
        ["Scope", "Ver.", "Cache", "Combined", "Remove solve", "GFLOPs/step", "s/step", "Obj. err", "Grad err", "Mean err", "Cov. err", "Parity"],
        e_rows, styles,
        widths=[30*mm, 15*mm, 18*mm, 22*mm, 23*mm, 25*mm, 20*mm, 23*mm, 23*mm, 23*mm, 23*mm, 19*mm],
        left_columns=(0,),
        highlights={(index, 11): PALE_GREEN if row[-1] == "PASS" else PALE_RED for index, row in enumerate(e_rows)},
    ))
    story.extend([
        Spacer(1, 3 * mm),
        p("最终选择：E1 缓存固定统计量，并在同一步 objective 内复用当前 kernel 对应的变换；cross tensor 由 scope 自适应选择 contraction 顺序。E2/E3 保留为算术顺序研究，不进入主结果。", styles["body"]),
        PageBreak(),
    ])

    story.extend([p("4. Cross tensor contraction 与 torch.compile", styles["h1"]), p(
        "下表先给出每个 scope 的最佳 eager contraction，再在附录列出全部 24 个 order/block 组合。",
        styles["body"],
    )])
    cross_best: list[list[object]] = []
    compile_rows: list[list[object]] = []
    for scope in ("task2_short", "tasks2_10_long"):
        cross = read_csv(root / f"cross/{scope}/seed0/cross_microbenchmark.csv")
        best = min(cross, key=lambda row: float(row["forward_backward_seconds"]))
        reference = next(row for row in cross if row["order"] == "einsum" and row["feature_block_size"] == "133")
        cross_best.append([
            "Task 2" if scope == "task2_short" else "Tasks 2-10",
            best["order"], best["feature_block_size"],
            fmt(number(best["forward_seconds"]), 4), fmt(number(best["forward_backward_seconds"]), 4),
            f"{100*(float(reference['forward_backward_seconds'])-float(best['forward_backward_seconds']))/float(reference['forward_backward_seconds']):.1f}%",
            fmt(number(best["profiler_counted_gflops_forward_backward"])),
            f"{float(best['peak_allocated_bytes'])/1024**2:.1f}/{float(best['peak_reserved_bytes'])/1024**2:.1f}",
            sci(number(best["output_relative_error"])), sci(number(best["gradient_relative_error"])),
        ])
        for row in read_csv(root / f"cross/{scope}/seed0/compile_microbenchmark.csv"):
            compile_rows.append([
                "Task 2" if scope == "task2_short" else "Tasks 2-10",
                row["compile_mode"], fmt(number(row["compilation_latency_seconds"]), 3),
                fmt(number(row["steady_forward_backward_seconds"]), 4),
                f"{float(row['peak_allocated_bytes'])/1024**2:.1f}/{float(row['peak_reserved_bytes'])/1024**2:.1f}",
                sci(number(row["output_relative_error"])), sci(number(row["gradient_relative_error"])),
            ])
    story.append(report_table(
        ["Scope", "Best order", "Block", "Forward s", "F+B s", "vs einsum", "GFLOPs", "Peak MiB A/R", "Output err", "Grad err"],
        cross_best, styles, widths=[31*mm, 34*mm, 18*mm, 24*mm, 24*mm, 25*mm, 23*mm, 34*mm, 25*mm, 25*mm], left_columns=(0, 1),
    ))
    story.extend([Spacer(1, 4 * mm), p("torch.compile 微基准", styles["h2"]), report_table(
        ["Scope", "Mode", "Compile s", "Steady F+B s", "Peak MiB A/R", "Output err", "Grad err"],
        compile_rows, styles, widths=[40*mm, 45*mm, 35*mm, 42*mm, 45*mm, 35*mm, 35*mm], left_columns=(0, 1),
    ), Spacer(1, 3 * mm), p(
        "结论：Task 2 选择 spatial-first；long scope 选择 temporal-first。torch.compile 的稳定时间没有形成可靠优势，且 reduce-overhead 在 long scope 将 reserved memory 提高到约 4,130 MiB，因此不采用。",
        styles["body"],
    ), PageBreak()])

    online = [row for row in unified if "strict online" in row["method"]]
    online_groups = grouped(online, ("scope", "objective"))
    online_rows: list[list[object]] = []
    for key in sorted(online_groups):
        scope, objective = key
        rows = online_groups[key]
        counted = average(rows, "profiler_counted_gflops_per_unit")
        supplement = average(rows, "analytical_forward_supplement_gflops_per_unit")
        counted_total = average(rows, "profiler_counted_total_gflops")
        lower_total = average(rows, "profiler_plus_forward_lower_bound_total_gflops")
        online_rows.append([
            "Task 2" if scope == "task2_short" else "Tasks 2-10",
            "DTC EB" if objective.startswith("finite_dtc") else "VFE EB",
            "19" if scope == "task2_short" else "171",
            fmt(counted), fmt(supplement),
            f"{fmt(counted_total/1000 if counted_total else None)}/{fmt(lower_total/1000 if lower_total else None)}",
            fmt(average(rows, "runtime_per_unit_seconds"), 4),
            fmt(average(rows, "training_or_stream_runtime_seconds"), 2),
            f"{fmt(average(rows, 'peak_allocated_mib'), 0)}/{fmt(average(rows, 'peak_reserved_mib'), 0)}",
            fmt(average(rows, "rmse"), 4), fmt(average(rows, "nll"), 4), fmt(average(rows, "coverage90"), 4),
        ])
    story.extend([p("5. Strict-online cumulative HiPPO", styles["h1"]), p(
        "表 5 为 seeds 0-4 平均。DTC/VFE 只表示 Task-1 empirical-Bayes 超参数来源；每个 causal block 的 posterior recursion、状态维度与 FLOPs 相同。",
        styles["body"],
    ), report_table(
        ["Scope", "Task-1 theta", "Blocks", "Profiler\nGFLOPs/block", "Analytical +\nGFLOPs/block", "Counted/lower\nTFLOPs", "s/block", "Stream s", "Peak MiB\nA/R", "RMSE", "NLL", "Cov90"],
        online_rows, styles,
        widths=[28*mm, 28*mm, 18*mm, 27*mm, 27*mm, 31*mm, 21*mm, 22*mm, 25*mm, 19*mm, 19*mm, 19*mm], left_columns=(0, 1),
    ), Spacer(1, 3*mm), p(
        "DTC Task-1 calibration 在 short 和 long online 中都优于 VFE calibration。Long DTC 的 RMSE/NLL/Coverage90 为 0.1376/-0.5033/0.8317；VFE calibration 为 0.1416/-0.3349/0.7830。VFE 在 long batch 上更好，并不意味着它提供了更适合 streaming transfer 的 Task-1 theta。",
        styles["body"],
    ), PageBreak()])

    candidate_rows = [row for row in rank_projection if row["selection"] == "candidate_rank"]
    projection_groups = grouped(candidate_rows, ("candidate_rank",))
    projection_table: list[list[object]] = []
    for rank in ("133", "73", "64", "48"):
        rows = projection_groups[(rank,)]
        errors = [float(row["relative_projection_error"]) for row in rows]
        projection_table.append([
            rank, sci(min(errors)), sci(max(errors)),
            "精确" if max(errors) <= 1e-12 else "近似/改变模型",
        ])
    batch_rank = [row for row in rank_accuracy if row["mode"] == "batch empirical Bayes"]
    online_rank = [row for row in rank_accuracy if row["mode"] == "strict online"]
    batch_rank_rows = [[
        "Task 2" if row["scope"] == "task2_short" else "Tasks 2-10", row["rank"],
        fmt(number(row.get("profiler_counted_gflops_per_step"))),
        fmt(number(row.get("objective_runtime_seconds_per_step"))),
        fmt(number(row["runtime_seconds"]), 2), fmt(number(row["peak_allocated_mib"]), 0),
        fmt(number(row["rmse"]), 4), fmt(number(row["nll"]), 4), fmt(number(row["coverage90"]), 4),
    ] for row in sorted(batch_rank, key=lambda item: (item["scope"], -int(item["rank"])))]
    online_rank_rows = [[
        "Task 2" if row["scope"] == "task2_short" else "Tasks 2-10", row["rank"],
        fmt(number(row["runtime_seconds"]), 2), fmt(number(row["persistent_state_mib"]), 2),
        fmt(number(row["peak_allocated_mib"]), 0), fmt(number(row["rmse"]), 4),
        fmt(number(row["nll"]), 4), fmt(number(row["coverage90"]), 4),
    ] for row in sorted(online_rank, key=lambda item: (item["scope"], -int(item["rank"])))]
    story.extend([p("6. Feature rank 诊断与精度-效率消融", styles["h1"]), p(
        "投影 basis 只由 Task-1 fit features 拟合，再检查 Tasks 2-10 的全部 1000 个未来位置。Rank 73/64/48 的未来误差明显非零，因此只能作为 approximate rank compression。",
        styles["body"],
    ), report_table(["Rank", "Future min error", "Future max error", "分类"], projection_table, styles,
                    widths=[35*mm, 60*mm, 60*mm, 75*mm], left_columns=(3,)),
        Spacer(1, 4*mm), p("Batch rank 消融（seed 0）", styles["h2"]),
        report_table(["Scope", "Rank", "GFLOPs/step", "Objective s/step", "Training s", "Peak alloc MiB", "RMSE", "NLL", "Cov90"],
                     batch_rank_rows, styles, widths=[37*mm, 20*mm, 35*mm, 38*mm, 31*mm, 35*mm, 28*mm, 28*mm, 28*mm], left_columns=(0,)),
        Spacer(1, 4*mm), p("Strict-online rank 消融（seed 0）", styles["h2"]),
        report_table(["Scope", "Rank", "Stream s", "State MiB", "Peak alloc MiB", "RMSE", "NLL", "Cov90"],
                     online_rank_rows, styles, widths=[42*mm, 22*mm, 38*mm, 34*mm, 40*mm, 30*mm, 30*mm, 30*mm], left_columns=(0,)),
        PageBreak()])

    hp_groups = grouped(hyperparameters, ("scope", "objective", "version"))
    hp_rows: list[list[object]] = []
    hp_order = [
        (scope, objective, version)
        for scope in ("task2_short", "tasks2_10_long")
        for objective in ("finite_dtc", "vfe")
        for version in ("original", "optimized")
    ]
    for key in hp_order:
        scope, objective, version = key
        rows = hp_groups[key]
        hp_rows.append([
            "Task 2" if scope == "task2_short" else "Tasks 2-10",
            "DTC" if objective == "finite_dtc" else "VFE",
            "Original" if version == "original" else "Optimized",
            mean_sd(rows, "ell_t"), mean_sd(rows, "ell_s_1"), mean_sd(rows, "ell_s_2"),
            mean_sd(rows, "kernel_variance"), mean_sd(rows, "noise_variance"),
        ])
    boundary_rows: list[list[object]] = []
    parity_groups = grouped([row for row in parity if row["pair_complete"] == "True"], ("scope", "objective"))
    for key in sorted(parity_groups):
        scope, objective = key
        rows = parity_groups[key]
        at_limit = sum(row["optimized_best_iteration"] == row["optimized_iterations_completed"] for row in rows)
        boundary_rows.append([
            "Task 2" if scope == "task2_short" else "Tasks 2-10",
            "DTC" if objective == "finite_dtc" else "VFE",
            f"{at_limit}/{len(rows)}", ", ".join(row["optimized_best_iteration"] for row in rows),
            sci(max(abs(float(row["delta_optimized_minus_original_rmse"])) for row in rows)),
            sci(max(abs(float(row["delta_optimized_minus_original_nll"])) for row in rows)),
            sci(max(abs(float(row["delta_optimized_minus_original_coverage90"])) for row in rows)),
        ])
    story.extend([p("7. 学习到的超参数与 validation 边界", styles["h1"]), p(
        "五个超参数均由 Route B empirical Bayes 学习；表中为 seeds 0-4 的 mean +/- standard deviation。Original 与 optimized 应在浮点容差内一致。",
        styles["body"],
    ), report_table(
        ["Scope", "Objective", "Version", "ell_t", "ell_s1", "ell_s2", "sigma_f^2", "sigma_y^2"],
        hp_rows, styles, widths=[25*mm, 20*mm, 24*mm, 39*mm, 39*mm, 39*mm, 39*mm, 39*mm], left_columns=(0, 1, 2),
    ), Spacer(1, 4*mm), p("Validation checkpoint 与 E0/E1 最终指标差异", styles["h2"]),
        report_table(
            ["Scope", "Objective", "Best=100", "Best steps by seed", "Max Delta RMSE", "Max Delta NLL", "Max Delta Cov90"],
            boundary_rows, styles, widths=[38*mm, 30*mm, 30*mm, 55*mm, 43*mm, 43*mm, 43*mm], left_columns=(0, 1),
        ), Spacer(1, 3*mm), p(
            "Task 2 VFE 的 5/5 seeds 都在第 100 步达到最佳 validation NLL，说明 100-step 对它是固定预算比较，不能写成已经完全收敛。Long VFE 为 2/5。最大 E0/E1 差异来自少数 long VFE 浮点训练轨迹，但最终指标仍数值等价。",
            styles["body"],
        ), PageBreak()])

    story.extend([p("附录 A. 全部 contraction 微基准", styles["h1"]), p(
        "每个 scope 包含 einsum、spatial-first、temporal-first，以及 feature block 16/32/64/133。Blocking 主要改变运行时间和显存，不改变理论 FLOPs。",
        styles["body"],
    )])
    for scope in ("task2_short", "tasks2_10_long"):
        rows = read_csv(root / f"cross/{scope}/seed0/cross_microbenchmark.csv")
        full_rows = [[
            row["order"], row["feature_block_size"], fmt(number(row["forward_seconds"]), 4),
            fmt(number(row["forward_backward_seconds"]), 4), fmt(number(row["profiler_counted_gflops_forward_backward"])),
            f"{float(row['largest_explicit_intermediate_bytes'])/1024**2:.1f}",
            f"{float(row['peak_allocated_bytes'])/1024**2:.1f}/{float(row['peak_reserved_bytes'])/1024**2:.1f}",
            sci(number(row["output_relative_error"])), sci(number(row["gradient_relative_error"])),
        ] for row in rows]
        story.extend([
            p("Task 2 short" if scope == "task2_short" else "Tasks 2-10 long", styles["h2"]),
            report_table(
                ["Order", "Block", "Forward s", "F+B s", "GFLOPs", "Largest MiB", "Peak MiB A/R", "Output err", "Grad err"],
                full_rows, styles, widths=[43*mm, 20*mm, 30*mm, 30*mm, 30*mm, 32*mm, 40*mm, 31*mm, 31*mm], left_columns=(0,),
            ), Spacer(1, 4*mm),
        ])
    totals: dict[tuple[str, str], float] = defaultdict(float)
    for row in operator_rows:
        totals[(row["source"], row["operator"])] += float(row["profiler_counted_flops"])
    top = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:12]
    operator_table = [[
        source.replace("online/tasks2_10_long/finite_dtc/seed0/", "online long/")
              .replace("online/task2_short/finite_dtc/seed0/", "online short/")
              .replace("batch_final/tasks2_10_long/seed0/", "batch long/")
              .replace("batch_final/task2_short/seed0/", "batch short/"),
        operator, f"{flops/1e9:.3f}",
    ] for (source, operator), flops in top]
    story.extend([p("附录 B. Profiler 算子热点与解释边界", styles["h1"]), report_table(
        ["Source", "Operator", "Formula-counted GFLOPs"], operator_table, styles,
        widths=[150*mm, 55*mm, 55*mm], left_columns=(0, 1),
    ), Spacer(1, 4*mm), p(
        "这些数字用于定位实现热点，不可直接跨 batch/online 排名，因为 online 行是整个 stream 的累计值，而 batch 行通常是一轮 forward+backward。Profiler 不覆盖 factorization/solve backward、elementwise kernels、transcendental functions、部分 reductions、optimizer、validation 和 prediction。WSL 环境中的 CUPTI hardware activity 返回不可用，因此没有硬件指令级 FLOPs。",
        styles["body"],
    ), p("最终论文建议", styles["h1"]), p(
        "1. 主实现采用 E1 + scope-adaptive eager contraction。2. Batch 表同时报告 DTC+D 与 VFE+D；long batch 中 VFE 更好，但 short VFE 仍可能未收敛。3. Strict-online 主协议继续采用 Task-1 Route-B DTC empirical-Bayes calibration 后冻结 theta，因为当前 DTC calibration 的 long online RMSE/NLL/Coverage90 全部优于 VFE calibration。4. Rank 73/64/48 只作为近似效率-精度消融。5. GFLOPs 必须与 runtime、peak allocated/reserved memory 并列，并明确 profiler 与 analytical supplement 的统计边界。",
        styles["body"],
    ), p("验证状态", styles["h2"]), p(
        "正式结果完整性：40/40 batch training，8/8 learned-theta profiler，20/20 online audits，4/4 rank calibrations，8/8 rank batch，8/8 rank online，8/8 rank profiler。相关自动化测试 25 passed；Python 与 shell 语法检查通过；预测方差全部 finite 且无 non-positive variance。",
        styles["body"],
    )])

    doc.build(story)
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("results/diagnostics/routeb_efficiency_optimization"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/pdf/routeb_efficiency_optimization_full_report_zh.pdf"),
    )
    args = parser.parse_args()
    generate(args.root, args.output)


if __name__ == "__main__":
    main()
