"""Create the ICLR overview diagram as a vector PDF."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
GRAY = "#4D4D4D"
LIGHT = "#F5F7FA"


def rounded(ax, xy, width, height, text, edge, face="white", fontsize=8.5):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=GRAY,
    )
    return patch


def arrow(ax, start, end, color=GRAY, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=1.2,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(Rectangle((0.005, 0.01), 0.99, 0.98, facecolor=LIGHT, edgecolor="#D9DDE3", linewidth=0.8))

    ax = axes[0]
    ax.set_title("(a) Changing representation", loc="left", fontweight="bold", fontsize=8.8)
    rounded(ax, (0.06, 0.51), 0.34, 0.18, "$u_{k-1}$\n$\\mathcal{H}_{k-1}$", BLUE, "white")
    rounded(ax, (0.60, 0.51), 0.34, 0.18, "$u_k$\n$\\mathcal{H}_k$", BLUE, "white")
    arrow(ax, (0.59, 0.60), (0.41, 0.60), BLUE)
    ax.text(
        0.50,
        0.83,
        r"$\mathbb{E}[u_{k-1}\mid u_k]$",
        ha="center",
        color=BLUE,
        fontsize=7.6,
    )
    ax.text(0.50, 0.75, r"$=(L_t\otimes I_s)u_k$", ha="center", color=BLUE, fontsize=7.6)
    ax.plot([0.10, 0.88], [0.38, 0.38], color=GRAY, linewidth=1.0)
    ax.scatter([0.13, 0.42, 0.89], [0.38] * 3, s=14, color=[GRAY, BLUE, ORANGE], zorder=3)
    ax.text(0.13, 0.28, r"$t_0$", ha="center")
    ax.text(0.42, 0.28, r"$t_{k-1}$", ha="center", color=BLUE)
    ax.text(0.89, 0.28, r"$t_k$", ha="center", color=ORANGE)
    ax.text(0.50, 0.09, "same dimension, different GP functionals", ha="center", fontsize=7.8, color=GRAY)

    ax = axes[1]
    ax.set_title("(b) Transfer likelihood evidence", loc="left", fontweight="bold", fontsize=8.8)
    rounded(
        ax,
        (0.08, 0.57),
        0.34,
        0.18,
        r"$\widetilde{\ell}_{k-1}$" + "\n" + r"$(\beta,u_{k-1})$",
        GREEN,
        "white",
        8.2,
    )
    rounded(
        ax,
        (0.58, 0.57),
        0.34,
        0.18,
        r"$\widetilde{\ell}_{k-1\to k}$" + "\n" + r"$(\beta,u_k)$",
        GREEN,
        "#E7F5EF",
        8.2,
    )
    arrow(ax, (0.43, 0.66), (0.57, 0.66), GREEN)
    ax.text(0.50, 0.82, "likelihood transport", ha="center", color=GREEN, fontsize=7.8)
    ax.text(0.50, 0.43, r"$R' = M^{\mathrm{T}}RM,\quad r'=M^{\mathrm{T}}r$", ha="center", fontsize=7.8)
    ax.text(0.50, 0.28, r"retains $R_{\beta u}$ and $B_k\otimes G$", ha="center", fontsize=7.6, color=GRAY)
    ax.text(0.50, 0.10, "transfer evidence, not latent coordinates", ha="center", fontsize=7.7, color=GRAY)

    ax = axes[2]
    ax.set_title("(c) Structured recovery", loc="left", fontweight="bold", fontsize=8.8)
    rounded(
        ax,
        (0.08, 0.62),
        0.84,
        0.15,
        r"$K_t^{-1}\!\otimes K_s^{-1}+B_k\!\otimes G$",
        ORANGE,
        "white",
        9,
    )
    arrow(ax, (0.50, 0.60), (0.50, 0.49), ORANGE)
    rounded(ax, (0.18, 0.29), 0.64, 0.18, "Schur + Sylvester\nsmall eigensystems", ORANGE, "#FFF1EA")
    arrow(ax, (0.50, 0.28), (0.50, 0.19), ORANGE)
    ax.text(0.50, 0.10, r"$O(M^3)$ recovery, $O(M^2)$ state", ha="center", fontsize=8.2, color=GRAY)

    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.02, top=0.92, wspace=0.08)
    fig.savefig("bounded_state_overview.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig("bounded_state_overview.png", dpi=300, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
