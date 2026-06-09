from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "results" / "tmp_era5_routeb_aggressive_tuning_attempt" / "lag_ar_single_location_loc99_theory_report.md"
PDF_PATH = MD_PATH.with_suffix(".pdf")


def convert_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def table_from(lines: list[str], body_style: ParagraphStyle) -> Table:
    rows: list[list[Paragraph]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append([Paragraph(convert_inline(cell), body_style) for cell in cells])
    table = Table(rows, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2933")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cfd8dc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_story() -> list:
    registerFont(UnicodeCIDFont("STSong-Light"))
    sample = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle(
            "h1",
            parent=sample["Heading1"],
            fontName="STSong-Light",
            fontSize=18,
            leading=24,
            spaceAfter=12,
            alignment=TA_CENTER,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=sample["Heading2"],
            fontName="STSong-Light",
            fontSize=13.5,
            leading=18,
            spaceBefore=12,
            spaceAfter=7,
            textColor=colors.HexColor("#1f2933"),
        ),
        "body": ParagraphStyle(
            "body",
            parent=sample["BodyText"],
            fontName="STSong-Light",
            fontSize=10,
            leading=15,
            spaceAfter=6,
            alignment=TA_LEFT,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=sample["BodyText"],
            fontName="STSong-Light",
            fontSize=9.8,
            leading=14,
            leftIndent=10,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "code",
            parent=sample["Code"],
            fontName="Courier",
            fontSize=8.6,
            leading=11,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.HexColor("#f6f8fa"),
            borderColor=colors.HexColor("#e5e7eb"),
            borderWidth=0.4,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=sample["BodyText"],
            fontName="STSong-Light",
            fontSize=8.8,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#54616a"),
            spaceAfter=8,
        ),
    }

    lines = MD_PATH.read_text(encoding="utf-8").splitlines()
    story: list = []
    i = 0
    in_code = False
    code_lines: list[str] = []
    bullet_items: list[ListItem] = []

    def flush_bullets() -> None:
        nonlocal bullet_items
        if bullet_items:
            story.append(ListFlowable(bullet_items, bulletType="bullet", start="circle", leftIndent=14))
            bullet_items = []
            story.append(Spacer(1, 3))

    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["code"]))
                code_lines = []
                in_code = False
            else:
                flush_bullets()
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not line.strip():
            flush_bullets()
            story.append(Spacer(1, 3))
            i += 1
            continue

        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image_match:
            flush_bullets()
            image_path = Path(image_match.group(2))
            if image_path.exists():
                img = Image(str(image_path))
                max_width = 17.0 * cm
                max_height = 12.2 * cm
                scale = min(max_width / img.imageWidth, max_height / img.imageHeight)
                img.drawWidth = img.imageWidth * scale
                img.drawHeight = img.imageHeight * scale
                story.append(KeepTogether([img, Paragraph("图 1. ERA5 location 99 seen-history diagnostic.", styles["caption"])]))
            i += 1
            continue

        if line.startswith("|"):
            flush_bullets()
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.append(table_from(table_lines, styles["body"]))
            story.append(Spacer(1, 8))
            continue

        if line.startswith("# "):
            flush_bullets()
            story.append(Paragraph(convert_inline(line[2:].strip()), styles["h1"]))
            i += 1
            continue

        if line.startswith("## "):
            flush_bullets()
            title = line[3:].strip()
            if title.startswith("5. "):
                story.append(PageBreak())
            story.append(Paragraph(convert_inline(title), styles["h2"]))
            i += 1
            continue

        if line.startswith("- "):
            bullet_items.append(ListItem(Paragraph(convert_inline(line[2:].strip()), styles["bullet"])))
            i += 1
            continue

        flush_bullets()
        story.append(Paragraph(convert_inline(line), styles["body"]))
        i += 1

    flush_bullets()
    return story


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("STSong-Light", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(2 * cm, 1.2 * cm, "ERA5 Route B lag-AR 单点诊断实验报告")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


def main() -> None:
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.8 * cm,
        title="ERA5 Route B lag-AR 单点诊断实验报告",
        author="Codex",
    )
    doc.build(build_story(), onFirstPage=footer, onLaterPages=footer)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
