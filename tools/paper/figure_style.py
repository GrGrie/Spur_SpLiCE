"""Shared look of the paper figures: one colour per method and quiet axes.

A method keeps its colour in every figure. The palette passes the categorical colour checks
(lightness band, chroma floor, colour-vision-deficiency separation); its lighter members sit below
3:1 contrast on white, so each figure names its lines in a legend or with direct labels.
"""

from __future__ import annotations

from pathlib import Path

from cospro.tracking.artifacts import PROJECT_ROOT

FIGURE_DIR = PROJECT_ROOT / "paper" / "figures"

COLORS = {
    "CoSpRo": "#2a78d6",
    "Raw CLIP graph": "#eb6834",
    "SimCLR": "#1baf7a",
    "Semantic SpLiCE graph": "#eda100",
    "Direct SpLiCE reconstruction": "#e87ba4",
}
#: Categorical slots in a fixed order, for methods the dictionary above does not name.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")
INK = "#333333"
MUTED = "#8a8f98"
GRID = "#ececec"


def color_for(name: str, index: int) -> str:
    """The colour of a method: its own if it has one, otherwise the next categorical slot."""

    return COLORS.get(name, PALETTE[index % len(PALETTE)])


def use_paper_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": "#555555", "ytick.color": "#555555", "pdf.fonttype": 42,
    })


def save(figure, output_dir: Path, name: str) -> None:
    """Write ``name``.pdf for LaTeX and ``name``.png for quick viewing."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    print(f"[INFO] Wrote {output_dir / name}.pdf and .png")
