"""Generate an annotated Route B theory document without modifying the source TeX.

The source theory file remains unchanged. This script appends an English
formula-by-formula explanation appendix extracted from
docs/routeB_theory_derivation_bilingual.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEX = ROOT / "docs" / "new_main_joint_training_ssgp_kron_routeB_refined2_with_sylvester_appendix.tex"
GUIDE_MD = ROOT / "docs" / "routeB_theory_derivation_bilingual.md"
OUT_TEX = ROOT / "docs" / "new_main_joint_training_ssgp_kron_routeB_refined2_with_sylvester_appendix_annotated_en.tex"


@dataclass
class FormulaNote:
    section: str
    title: str
    formulas: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)
    speaking_point: str = ""


def latex_escape_text(text: str) -> str:
    """Escape ordinary prose while keeping inline math delimited by \( ... \)."""

    parts = re.split(r"(\\\(.*?\\\))", text)
    escaped: list[str] = []
    for part in parts:
        if part.startswith(r"\(") and part.endswith(r"\)"):
            escaped.append(part)
        else:
            escaped.append(
                part.replace("\\", r"\textbackslash{}")
                .replace("&", r"\&")
                .replace("%", r"\%")
                .replace("$", r"\$")
                .replace("#", r"\#")
                .replace("_", r"\_")
                .replace("{", r"\{")
                .replace("}", r"\}")
                .replace("~", r"\textasciitilde{}")
                .replace("^", r"\textasciicircum{}")
            )
    return "".join(escaped)


def clean_formula(formula: str) -> str:
    # Pandoc-style markdown equations use raw LaTeX inside $$ blocks; keep them raw.
    return formula.strip()


def english_heading(text: str) -> str:
    """Keep the English side of bilingual markdown headings for pdflatex output."""

    if " / " in text:
        return text.split(" / ", 1)[0].strip()
    return text.strip()


def parse_formula_notes(markdown: str) -> list[FormulaNote]:
    lines = markdown.splitlines()
    notes: list[FormulaNote] = []
    section = "Overview"
    current: FormulaNote | None = None
    mode: str | None = None
    formula_buf: list[str] = []
    in_formula = False

    def finish_formula() -> None:
        nonlocal formula_buf
        if current is not None and formula_buf:
            current.formulas.append(clean_formula("\n".join(formula_buf)))
        formula_buf = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("## ") and not line.startswith("### "):
            section = english_heading(line[3:].strip())
            mode = None
            continue

        if line.startswith("### Formula"):
            finish_formula()
            title = english_heading(line[4:].strip())
            current = FormulaNote(section=section, title=title)
            notes.append(current)
            mode = None
            continue

        if current is None:
            continue

        if line.strip() == "$$":
            if not in_formula:
                in_formula = True
                formula_buf = []
            else:
                in_formula = False
                finish_formula()
            continue

        if in_formula:
            formula_buf.append(line)
            continue

        if line.startswith("**English explanation**"):
            mode = "english"
            continue
        if "Speaking point" in line:
            mode = "speaking"
            continue
        if line.startswith("**") or line.startswith("---"):
            mode = None
            continue
        if line.startswith("### ") or line.startswith("## "):
            mode = None
            continue

        if mode == "english":
            stripped = line.strip()
            if not stripped:
                continue
            match = re.match(r"^\d+\.\s+(.*)$", stripped)
            current.explanation.append(match.group(1) if match else stripped)
        elif mode == "speaking":
            stripped = line.strip()
            if stripped:
                current.speaking_point = stripped
                mode = None

    finish_formula()
    return [n for n in notes if n.formulas or n.explanation or n.speaking_point]


def notes_to_tex(notes: list[FormulaNote]) -> str:
    out: list[str] = []
    out.append(r"\clearpage")
    out.append(r"\section{Formula-by-Formula English Explanation and Speaking Notes}")
    out.append(
        "This appendix is an annotated companion to the theory document. "
        "It keeps the main formulas in their original order and adds a plain-English "
        "derivation guide for discussion with a supervisor. The original source TeX "
        "file is not modified; this is a generated annotated copy."
    )
    out.append("")

    last_section = None
    for note in notes:
        if note.section != last_section:
            out.append(rf"\subsection{{{latex_escape_text(note.section)}}}")
            last_section = note.section

        out.append(rf"\paragraph{{{latex_escape_text(note.title)}}}")
        for formula in note.formulas:
            out.append(r"\[")
            out.append(formula)
            out.append(r"\]")

        if note.explanation:
            out.append(r"\textbf{English explanation.}")
            out.append(r"\begin{enumerate}[leftmargin=2em]")
            for item in note.explanation:
                out.append(rf"\item {latex_escape_text(item)}")
            out.append(r"\end{enumerate}")

        if note.speaking_point:
            out.append(r"\textbf{Speaking point.} " + latex_escape_text(note.speaking_point))
        out.append("")
    return "\n".join(out)


def main() -> None:
    source = SOURCE_TEX.read_text(encoding="utf-8")
    markdown = GUIDE_MD.read_text(encoding="utf-8")
    notes = parse_formula_notes(markdown)
    appendix = notes_to_tex(notes)

    if r"\end{document}" not in source:
        raise RuntimeError(f"Could not find \\end{{document}} in {SOURCE_TEX}")

    annotated = source.replace(
        r"\title{Joint Online Gaussian Process with Linear Mean Model\\",
        r"\title{Annotated Joint Online Gaussian Process with Linear Mean Model\\",
        1,
    )
    annotated = annotated.replace(
        r"\date{March 2026}",
        r"\date{March 2026\\Annotated English explanation version}",
        1,
    )
    annotated = annotated.replace(r"\end{document}", appendix + "\n\n" + r"\end{document}", 1)
    OUT_TEX.write_text(annotated, encoding="utf-8")
    print(f"Wrote {OUT_TEX}")
    print(f"Inserted {len(notes)} formula explanation blocks")


if __name__ == "__main__":
    main()
