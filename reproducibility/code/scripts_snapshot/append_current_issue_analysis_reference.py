"""Append a detailed analysis reference section to docs/current issue.docx.

This script uses direct OOXML editing so it does not depend on python-docx.
It preserves the original document and writes a new copy with an appended
structured analysis section.
"""

from __future__ import annotations

import copy
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W_NS)


def qn(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def make_p(text: str, style: str | None = None, bold: bool = False) -> ET.Element:
    p = ET.Element(qn("p"))
    if style:
        ppr = ET.SubElement(p, qn("pPr"))
        pstyle = ET.SubElement(ppr, qn("pStyle"))
        pstyle.set(qn("val"), style)
    r = ET.SubElement(p, qn("r"))
    if bold:
        rpr = ET.SubElement(r, qn("rPr"))
        ET.SubElement(rpr, qn("b"))
    t = ET.SubElement(r, qn("t"))
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return p


def make_table(rows: list[list[str]]) -> ET.Element:
    tbl = ET.Element(qn("tbl"))
    tbl_pr = ET.SubElement(tbl, qn("tblPr"))
    tbl_w = ET.SubElement(tbl_pr, qn("tblW"))
    tbl_w.set(qn("w"), "0")
    tbl_w.set(qn("type"), "auto")
    borders = ET.SubElement(tbl_pr, qn("tblBorders"))
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = ET.SubElement(borders, qn(edge))
        b.set(qn("val"), "single")
        b.set(qn("sz"), "4")
        b.set(qn("space"), "0")
        b.set(qn("color"), "D9E2F3")
    for row_idx, row in enumerate(rows):
        tr = ET.SubElement(tbl, qn("tr"))
        for cell in row:
            tc = ET.SubElement(tr, qn("tc"))
            tc_pr = ET.SubElement(tc, qn("tcPr"))
            shd = ET.SubElement(tc_pr, qn("shd"))
            shd.set(qn("fill"), "EAF2FF" if row_idx == 0 else "FFFFFF")
            tc.append(make_p(cell, bold=(row_idx == 0)))
    return tbl


SECTION_BLOCKS: list[ET.Element] = []


def add_paragraph(text: str, style: str | None = None, bold: bool = False) -> None:
    SECTION_BLOCKS.append(make_p(text, style=style, bold=bold))


def add_table(rows: list[list[str]]) -> None:
    SECTION_BLOCKS.append(make_table(rows))


add_paragraph("Detailed analysis reference for the ERA5 variance issue", "Heading1")
add_paragraph(
    "This section is intended as a supervisor-facing reference. It separates the observed phenomenon, the variance formula, the numerical evidence, the likely causes, and the safe interpretation boundary. The main point is that the wide single-location bands are best interpreted as conservative observation-level uncertainty, not as direct evidence that the structured Route B posterior update is wrong."
)

add_paragraph("1. What exactly is being plotted?", "Heading2")
add_paragraph(
    "The plotted band is the predictive interval for the noisy observation y*, not only the latent GP response. In the Route B implementation, the observation-level predictive variance can be read as:"
)
add_paragraph(
    "Var(y*) = sigma2 + nu_star + u posterior term + beta/Schur term.",
    bold=True,
)
add_paragraph(
    "Here sigma2 is the observation-noise variance used by the likelihood; nu_star is the sparse conditional residual variance left after projecting the test point onto the inducing/basis representation; the u posterior term is the uncertainty propagated through the inducing GP coefficients; and the beta/Schur term is the uncertainty from the linear beta part after accounting for beta-u coupling. Therefore, a wide interval can be caused by a large noise estimate or by a large sparse residual, even if the structured beta-u posterior correction itself is numerically stable."
)

add_paragraph("2. Numerical evidence from the current ERA5 run", "Heading2")
add_table(
    [
        [
            "Setting",
            "sigma2",
            "avg nu_star",
            "posterior terms",
            "avg total variance",
            "Cov90",
            "Interpretation",
        ],
        [
            "base Phi structured_joint",
            "0.8364",
            "0.6188",
            "~0.0033",
            "1.4584",
            "0.9964",
            "Very conservative observation interval",
        ],
        [
            "rich Phi structured_joint",
            "0.3813",
            "0.6188",
            "~0.0031",
            "1.0031",
            "0.9991",
            "Feature map explains more signal, but nu_star remains large",
        ],
        [
            "fixed small sigma2=0.09",
            "0.0900",
            "0.6188",
            "small",
            "0.7092",
            "0.9719",
            "Bands shrink and NLL improves while coverage remains high",
        ],
        [
            "subset full-GP MLL",
            "0.0100",
            "0.3108",
            "small",
            "0.3209",
            "0.8476",
            "Much sharper, but under-covers in this subset diagnostic",
        ],
    ]
)
add_paragraph(
    "The base-Phi structured_joint variance is dominated by sigma2 + nu_star = 0.8364 + 0.6188 = 1.4552, which almost exactly matches the reported average total variance 1.4584. The u posterior term and beta/Schur term are tiny by comparison. This is the strongest evidence that the band width is not mainly caused by the Route B cross-covariance correction."
)

add_paragraph("3. Why sigma2 is likely conservative", "Heading2")
add_paragraph(
    "The current sigma2 is estimated from calibration-task beta-only ridge residuals. But under the model y = Phi beta + f + epsilon, the residual y - Phi beta_hat is not pure measurement noise. It also contains structured GP residuals, unmodelled seasonal-spatial effects, basis mismatch, and any systematic signal that the base feature map cannot explain. Treating this entire residual variance as observation noise inflates sigma2. This is why the observation-level predictive bands become wide before the posterior terms even matter."
)
add_paragraph(
    "This interpretation is supported by the rich-Phi diagnostic. When the feature map is enriched with additional seasonal-spatial and interaction terms, the estimated sigma2 decreases from 0.8364 to 0.3813. That reduction means part of the original noise estimate was actually explainable structured signal."
)

add_paragraph("4. Why nu_star remains important", "Heading2")
add_paragraph(
    "Even after improving Phi, avg nu_star remains around 0.6188 in the current diagnostic. This means the sparse inducing/basis representation still leaves a sizeable conditional residual variance at the ERA5 test points. In practical terms, the current M_t=8 and M_s=64 representation may not fully cover the 1000-location spatio-temporal grid. This does not mean the posterior update is wrong; it means the sparse approximation is still conservative at prediction time."
)
add_paragraph(
    "If the goal is to reduce interval width without simply forcing sigma2 downward, the next natural checks are increasing M_t/M_s, improving spatial inducing placement, or changing the temporal/spatial basis design. These checks target nu_star directly."
)

add_paragraph("5. What should and should not be claimed", "Heading2")
add_table(
    [
        ["Claim type", "Safe wording"],
        [
            "Main Route B retention claim",
            "structured_joint improves OHSVGP-style held-out seen-history RMSE/NLL/forgetting over internal Route B ablations.",
        ],
        [
            "Uncertainty statement",
            "The current observation-level intervals are conservative; calibration remains an open issue.",
        ],
        [
            "What not to overclaim",
            "Do not say the wide bands prove Route B uncertainty is intrinsically better. They mainly show conservative sigma2 and nu_star.",
        ],
        [
            "Rich Phi diagnostic",
            "Rich Phi reduces residual noise and improves fit, but does not by itself solve sparse residual variance.",
        ],
    ]
)

add_paragraph("6. Recommended next experiments", "Heading2")
add_paragraph(
    "First, always report latent-function intervals and observation-level intervals separately. This makes clear whether the visual band width comes from sigma2 or from latent posterior/sparse residual uncertainty."
)
add_paragraph(
    "Second, run a calibration sweep over sigma2, kernel variance, and ell_t using the initial task only. The comparison should report NLL, Cov90, ECE, avg variance, and avg nu_star. A useful set of conditions is: beta-only residual sigma2; fixed smaller sigma2; initial-task full-GP MLL sigma2; and full-GP MLL plus rich Phi."
)
add_paragraph(
    "Third, run a sparse-basis capacity diagnostic over M_t and M_s. If avg nu_star decreases while RMSE/NLL and calibration improve, then the remaining band width is mainly a sparse-basis capacity issue rather than a Route B posterior issue."
)

add_paragraph("7. One-paragraph summary for discussion", "Heading2")
add_paragraph(
    "The ERA5 single-location bands are wide because the plotted interval is for noisy observations and is dominated by sigma2 plus nu_star. In the base-Phi run, sigma2=0.8364 and nu_star=0.6188 already explain almost all of the total variance, while the Route B posterior terms are very small. Therefore, the issue is mainly conservative noise/residual calibration rather than failure of the structured beta-u posterior transfer. Rich Phi reduces sigma2 and improves fit, but nu_star remains large, so the next step is to separate latent and observation intervals, tune noise/kernel hyperparameters on the initial task, and test whether larger or better sparse bases reduce nu_star."
)


def append_blocks(input_path: Path, output_path: Path) -> None:
    with zipfile.ZipFile(input_path, "r") as zin:
        entries = {name: zin.read(name) for name in zin.namelist()}

    document_xml = entries["word/document.xml"]
    root = ET.fromstring(document_xml)
    body = root.find(qn("body"))
    if body is None:
        raise RuntimeError("Could not find Word document body")

    sect_pr = body.find(qn("sectPr"))
    insert_at = list(body).index(sect_pr) if sect_pr is not None else len(list(body))
    for block in reversed([copy.deepcopy(b) for b in SECTION_BLOCKS]):
        body.insert(insert_at, block)

    entries["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in entries.items():
            zout.writestr(name, data)


def main() -> None:
    input_path = Path("docs/current issue.docx")
    output_path = Path("docs/current issue - detailed analysis reference.docx")
    append_blocks(input_path, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
