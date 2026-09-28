"""
Shared publication-quality matplotlib style for the surrogacy paper.

Apply once at the top of any figure-generation script:

    from figstyle import apply_style, COLORS, SIZES
    apply_style()

Design choices
--------------
- Serif body font with light tracking, rendered at 9pt for tick labels and 10pt
  for axis labels so the figure survives the 0.6x reduction in two-column layout.
- Editorial four-color palette tuned for legibility on greyscale photocopies
  and for color-vision-deficient readers (no red/green collisions).
- Top/right spines stripped, axis spines a touch lighter than ink black, and
  ticks pointing outward (academic convention).
- Subtle dashed gridlines drawn behind the data.
- Tight margins; PDFs saved at 600 dpi for raster fallback.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# Editorial palette, deepened slightly for more vibrancy on the printed page
# while staying away from cartoonish primaries. Tuned to remain legible in
# greyscale and to avoid red/green collisions for color-vision-deficient readers.
COLORS = {
    "primary": "#1f6db5",
    "accent": "#c92a2a",
    "secondary": "#1b9e8a",
    "tertiary": "#e8633f",
    "neutral": "#4a4a4a",
    "theory": "#6a3d9a",
    "muted": "#8a9099",
    "grid": "#d3d4d6",
    "spine": "#3c3c3c",
}

# All single-column figures are authored at the ACM sigconf column width
# (~3.35in) and displayed at \linewidth so they render 1:1 (no downscaling),
# giving one consistent on-page font size across every figure in the paper.
# `*_09` sizes are authored at 0.9\linewidth for figures included at that width.
SIZES = {
    "single_col": (3.35, 2.15),
    "single_col_tall": (3.35, 2.7),
    "single_col_2panel": (3.35, 1.62),
    "single_col_09": (3.0, 1.95),
    "single_col_small": (2.55, 1.75),
    "twopanel": (7.0, 2.35),
    "full_2panel": (7.0, 2.5),
    "wide_half": (5.0, 3.2),
    "double_col": (7.0, 3.0),
    "double_col_tall": (7.0, 3.4),
}


def apply_style() -> None:
    """Apply the shared rcParams. Idempotent; safe to call multiple times."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "Times", "DejaVu Serif", "STIXGeneral", "serif",
        ],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.titleweight": "regular",
        "axes.labelsize": 8,
        "axes.labelpad": 2.5,
        "axes.linewidth": 0.7,
        "axes.edgecolor": COLORS["spine"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["grid"],
        "grid.linestyle": (0, (1, 2)),
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.color": COLORS["spine"],
        "ytick.color": COLORS["spine"],
        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.45,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.3,
        "legend.columnspacing": 1.0,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.6,
        "lines.markeredgewidth": 1.0,
        "patch.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        # No bbox="tight": author each figure at its on-page width so the saved
        # canvas equals the display width and fonts render 1:1 (consistent across
        # the paper). Margins are handled per figure via tight_layout.
        "savefig.bbox": "standard",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axes(ax) -> None:
    """Light per-axes refinements after content has been drawn."""
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["spine"])
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(which="both", colors=COLORS["spine"])
    ax.title.set_color(COLORS["spine"])
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_color(COLORS["spine"])


def panel_label(ax, text: str, *, x: float = 0.0, y: float = 1.04) -> None:
    """Attach an inline panel label, e.g. '(a)', in axis coordinates.

    Font size tracks the active title size so the label scales correctly when a
    figure is authored under a scaled rc-context (e.g. wide figures meant to be
    downscaled into a single column)."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=mpl.rcParams["axes.titlesize"], fontweight="bold",
            color=COLORS["spine"], ha="left", va="bottom")
