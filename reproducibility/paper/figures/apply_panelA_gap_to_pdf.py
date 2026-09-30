"""Insert a vector fill between the authored Figure 1 trend curves.

The original PDF content is retained and the new path is prepended to the
page content, so the existing axes, observations, labels, and trend strokes
remain in their original z-order above the tint.
"""

from pathlib import Path
from lxml import etree
from pptx import Presentation
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject


ROOT = Path(__file__).resolve().parent
PPTX = ROOT / "Figure1_Publication_Overview.pptx"
PDF_IN = ROOT / "Figure1_Publication_Overview.pdf"
PDF_OUT = ROOT / "Figure1_Publication_Overview_gap.pdf"


def vertices(shape):
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    path = shape._element.find(".//a:path", ns)
    if path is None:
        raise RuntimeError(f"Missing custom path for {shape.name}")
    pts = []
    for pt in path.findall(".//a:moveTo/a:pt", ns) + path.findall(".//a:lnTo/a:pt", ns):
        pts.append((shape.left + int(pt.get("x")), shape.top + int(pt.get("y"))))
    return pts


def main():
    prs = Presentation(str(PPTX))
    slide = prs.slides[0]
    old = next(sh for sh in slide.shapes if sh.name == "old trend")
    revised = next(sh for sh in slide.shapes if sh.name == "revised trend")
    top = vertices(revised)
    bottom = vertices(old)
    if len(top) != len(bottom):
        raise RuntimeError("Trend paths use different sampling grids")

    # Figure1_Publication_Overview.pdf is 1125 x 345 pt and the PPTX slide is
    # 14287500 x 4381500 EMU, hence 12700 EMU per PDF point.
    page_h = 345.0
    scale = 12700.0
    poly = [(x / scale, page_h - y / scale) for x, y in top]
    poly.extend((x / scale, page_h - y / scale) for x, y in reversed(bottom))

    # Insert a pale warm off-white fill immediately after the page background;
    # the original PDF then redraws every authored stroke on top of it.
    # Keep the band as a secondary cue: this is intentionally much closer to
    # white than the first revision, while the authored strokes remain above it.
    ops = [b"q", b"0.965 0.937 0.902 rg"]
    x0, y0 = poly[0]
    ops.append(f"{x0:.4f} {y0:.4f} m".encode("ascii"))
    for x, y in poly[1:]:
        ops.append(f"{x:.4f} {y:.4f} l".encode("ascii"))
    ops.extend([b"h", b"f", b"Q", b""])
    fill_stream = b"\n".join(ops)

    reader = PdfReader(str(PDF_IN))
    page = reader.pages[0]
    original = page.get_contents().get_data()
    stream = DecodedStreamObject()
    marker = b"Q\r\n EMC"
    split = original.find(marker)
    if split < 0:
        marker = b"Q\n EMC"
        split = original.find(marker)
    if split < 0:
        raise RuntimeError("Could not locate the opening background group")
    cut = split + len(marker)
    stream.set_data(original[:cut] + b"\n" + fill_stream + b"\n" + original[cut:])
    page.replace_contents(stream)

    writer = PdfWriter()
    writer.add_page(page)
    with PDF_OUT.open("wb") as fh:
        writer.write(fh)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
