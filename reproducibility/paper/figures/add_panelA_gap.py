"""Add a restrained pale warm-brown fill between the two trend curves in Figure 1.

The existing editable PowerPoint artwork is preserved by the caller before this
script runs.  The fill is built from the already-authored curve vertices, so it
does not redraw or perturb either trend line.
"""

from pathlib import Path
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor


ROOT = Path(__file__).resolve().parent
PPTX_IN = ROOT / "Figure1_Publication_Overview.pptx"
PPTX_OUT = ROOT / "Figure1_Publication_Overview_gap.pptx"


def curve_vertices(shape):
    """Return absolute EMU vertices from a custom-geometry line path."""
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    path = shape._element.find(".//a:path", ns)
    if path is None:
        raise RuntimeError(f"No custom path found for {shape.name!r}")
    x0, y0 = shape.left, shape.top
    pts = []
    for pt in path.findall(".//a:moveTo/a:pt", ns) + path.findall(".//a:lnTo/a:pt", ns):
        pts.append((x0 + int(pt.get("x")), y0 + int(pt.get("y"))))
    if len(pts) < 2:
        raise RuntimeError(f"Too few vertices for {shape.name!r}")
    return pts


def main():
    prs = Presentation(str(PPTX_IN))
    slide = prs.slides[0]
    old_curve = next(sh for sh in slide.shapes if sh.name == "old trend")
    revised_curve = next(sh for sh in slide.shapes if sh.name == "revised trend")

    revised_pts = curve_vertices(revised_curve)
    old_pts = curve_vertices(old_curve)
    if len(revised_pts) != len(old_pts):
        raise RuntimeError("Trend curves do not share a common sampling grid")

    # Both curves use the same x-grid.  Follow the revised curve left-to-right
    # and the old curve right-to-left to form the exact visible gap.
    vertices = revised_pts + list(reversed(old_pts))
    fb = slide.shapes.build_freeform(vertices[0][0], vertices[0][1])
    fb.add_line_segments(vertices[1:], close=True)
    gap = fb.convert_to_shape()
    gap.name = "revised-old trend transition"

    # A quiet warm off-white tint, with no outline, keeps the gap legible without
    # competing with the blue revised trend or the gray dashed old trend.
    gap.fill.solid()
    gap.fill.fore_color.rgb = RGBColor(246, 239, 230)
    gap.line.fill.background()

    # Place the new shape immediately behind the two authored curves, while
    # leaving axes, observations, labels, and annotations unchanged.
    sp_tree = slide.shapes._spTree
    new_el = gap._element
    sp_tree.remove(new_el)
    old_el = old_curve._element
    insert_at = list(sp_tree).index(old_el)
    sp_tree.insert(insert_at, new_el)

    prs.save(str(PPTX_OUT))
    print(PPTX_OUT)


if __name__ == "__main__":
    main()
