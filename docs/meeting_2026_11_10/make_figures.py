"""Figures of the concept-factors branch report (docs/meeting_2026_11_10/factors_report.tex).

Every number comes from the study summaries under outputs/reports/factors_<dataset>/summary.md, the
research log docs/notes/CONCEPT_FACTORS_2026_09_26.md and the per-seed tables of
paper/SpatialConcepts.tex. Validation split, mean of the last four probes (epochs 425 to 500).

    python docs/meeting_2026_11_10/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
INK, MUTED, GRID = "#333333", "#8a8f98", "#ececec"
BLUE, ORANGE, GREEN, YELLOW, PINK, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"
FAMILY = {
    "baseline": MUTED,
    "image-level": YELLOW,
    "spatial (SCD)": BLUE,
    "spatial + prior-free extension": PURPLE,
    "spatial + object prior": ORANGE,
    "control": "#c4c8ce",
}

plt.rcParams.update({
    "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": "#555555", "ytick.color": "#555555", "pdf.fonttype": 42,
})


def save(figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {name}")


# (label, family, WGA mean, WGA sd, seeds); sd None for single seeds.
WATERBIRDS = [
    ("SimCLR", "baseline", 46.7, 1.9, 6),
    ("CoSpRo paper", "baseline", 50.0, 3.2, 4),
    ("LateTVG, prune 0.9", "baseline", 47.9, None, 1),
    ("F2: trained linear head", "image-level", 41.6, 6.1, 4),
    ("xfit: cross-fitted ridge, w10", "image-level", 51.5, 5.5, 4),
    ("SCD: spatial concept maps", "spatial (SCD)", 52.8, 4.7, 6),
    ("  control: shuffled maps", "control", 47.5, 2.5, 6),
    ("  control: CLIP PCA maps", "control", 53.6, 1.9, 4),
    ("Region contrast (DetCon)", "spatial + prior-free extension", 55.3, 3.1, 4),
    ("Symmetric mixing", "spatial + prior-free extension", 49.2, 2.5, 4),
    ("Per-group heads (LooC)", "spatial + prior-free extension", 52.4, 2.9, 4),
    ("Compositing (object prior)", "spatial + object prior", 59.1, 7.1, 6),
    ("  control: random silhouette", "control", 54.2, 6.1, 6),
    ("  control: context kept", "control", 52.7, 2.1, 4),
]
METASHIFT = [
    ("SimCLR", "baseline", 44.3, 7.3, 6),
    ("CoSpRo paper", "baseline", 42.9, 2.7, 10),
    ("F2: trained linear head, w3", "image-level", 46.6, 5.7, 4),
    ("F3: concept blocks", "image-level", 43.8, 4.1, 4),
    ("xfit: cross-fitted ridge, w10", "image-level", 47.6, 1.0, 2),
    ("SCD: spatial concept maps", "spatial (SCD)", 51.1, 5.4, 6),
    ("  control: shuffled maps", "control", 45.1, 2.9, 6),
    ("  control: CLIP PCA maps", "control", 44.4, 7.4, 4),
    ("Region contrast (DetCon)", "spatial + prior-free extension", 50.0, 6.5, 4),
    ("Symmetric mixing", "spatial + prior-free extension", 51.0, 8.5, 4),
    ("Per-group heads (LooC)", "spatial + prior-free extension", 46.9, 3.3, 4),
    ("Compositing (object prior)", "spatial + object prior", 52.3, 1.9, 4),
    ("  control: random silhouette", "control", 50.6, 3.7, 4),
    ("  control: context kept", "control", 49.3, 1.9, 4),
]


def scoreboard() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 4.6), sharey=True)
    for axis, (title, rows, simclr) in zip(axes, [("Waterbirds", WATERBIRDS, 46.7), ("MetaShift", METASHIFT, 44.3)]):
        positions = np.arange(len(rows))[::-1]
        for y, (label, family, mean, sd, seeds) in zip(positions, rows):
            color = FAMILY[family]
            axis.barh(y, mean - 30, left=30, height=0.62, color=color, alpha=0.9 if family != "control" else 1.0)
            if sd is not None:
                axis.errorbar(mean, y, xerr=sd, color=INK, lw=0.8, capsize=2)
            right = mean + (sd or 0.0) + 0.6
            axis.text(right, y, f"{mean:.1f} ({seeds}s)", va="center", ha="left", fontsize=6.6, color=INK)
        axis.axvline(simclr, color=INK, lw=0.8, ls="--")
        axis.set_xlim(30, 74)
        axis.set_title(title, fontsize=10, color=INK)
        axis.set_xlabel("val. WGA (%), dashed: SimCLR")
        axis.grid(axis="x", color=GRID)
        axis.set_axisbelow(True)
    labels = [row[0] for row in WATERBIRDS]
    axes[0].set_yticks(np.arange(len(labels))[::-1])
    axes[0].set_yticklabels(labels)
    # MetaShift rows differ in two baseline labels; label its own axis on the right.
    meta = axes[1].twinx()
    meta.set_ylim(axes[1].get_ylim())
    meta.set_yticks(np.arange(len(METASHIFT))[::-1])
    meta.set_yticklabels([row[0] for row in METASHIFT], fontsize=7.5)
    meta.spines["right"].set_visible(False)
    meta.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in FAMILY.values()]
    figure.legend(handles, FAMILY.keys(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))
    figure.tight_layout()
    save(figure, "scoreboard")


def spur_cifar() -> None:
    rows = [
        ("SimCLR", "baseline", 15.6, 1.0),
        ("CoSpRo (1 seed)", "baseline", 19.6, None),
        ("F2 trained head", "image-level", 32.9, 1.6),
        ("  F2, shuffled targets", "control", 24.2, 0.4),
        ("xfit cross-fitted ridge", "image-level", 33.6, 2.3),
        ("  xfit, shuffled targets", "control", 18.2, 3.2),
        ("xfit, CLIP PCA targets", "image-level", 37.0, 0.3),
        ("SCD spatial maps", "spatial (SCD)", 38.1, 0.5),
        ("  SCD, shuffled maps", "control", 17.2, 2.0),
    ]
    figure, axis = plt.subplots(figsize=(4.6, 2.5))
    positions = np.arange(len(rows))[::-1]
    for y, (label, family, mean, sd) in zip(positions, rows):
        axis.barh(y, mean, height=0.62, color=FAMILY[family])
        if sd is not None:
            axis.errorbar(mean, y, xerr=sd, color=INK, lw=0.8, capsize=2)
        axis.text(mean + 1.2, y, f"{mean:.1f}", va="center", fontsize=7)
    axis.set_yticks(positions)
    axis.set_yticklabels([row[0] for row in rows])
    axis.axvline(15.6, color=INK, lw=0.8, ls="--")
    axis.set_xlabel("validation WGA (%), seeds 1 and 2")
    axis.set_title("Spur-CIFAR10 (100 groups)", fontsize=9.5)
    axis.grid(axis="x", color=GRID)
    axis.set_axisbelow(True)
    save(figure, "spur_cifar10")


def paired_seeds() -> None:
    # Per-seed WGA, mean of the last four probes (paper/SpatialConcepts.tex, appendix B).
    data = {
        "Waterbirds": {
            "SimCLR": [44.97, 43.79, 47.43, 46.99, 48.18, 48.61],
            "Shuffled maps": [50.19, 47.54, 44.91, 44.27, 48.07, 50.00],
            "SCD": [57.26, 51.13, 50.75, 52.96, 58.65, 45.82],
        },
        "MetaShift": {
            "SimCLR": [50.69, 44.79, 31.60, 52.08, 44.44, 42.36],
            "Shuffled maps": [43.06, 42.01, 49.31, 46.87, 46.53, 42.71],
            "SCD": [60.42, 50.69, 54.17, 46.87, 48.61, 45.83],
        },
    }
    colors = {"SimCLR": GREEN, "Shuffled maps": "#9aa0a8", "SCD": BLUE}
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for axis, (title, arms) in zip(axes, data.items()):
        seeds = np.arange(1, 7)
        for arm, values in arms.items():
            axis.plot(seeds, values, "o-", color=colors[arm], lw=1.2, ms=4, label=arm)
        axis.set_title(title, fontsize=9.5)
        axis.set_xlabel("student seed")
        axis.set_xticks(seeds)
        axis.grid(color=GRID)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("validation WGA (%)")
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower left")
    figure.tight_layout()
    save(figure, "paired_seeds")


def memorization() -> None:
    # Explained variance of the factors on images inside the concept loss (seen) and on the held-out
    # fifth (unseen); CLIP unseen is the score of frozen CLIP features on the same unseen images.
    rows = [
        ("MetaShift\nimage-level\nxfit w10", 0.60, 0.04, 0.26),
        ("MetaShift\nimage-level\nshuffled", 0.01, 0.02, 0.26),
        ("Waterbirds\nimage-level\nxfit w10", 0.30, 0.09, 0.27),
        ("Spur-CIFAR10\nimage-level\nxfit w1", 0.31, 0.27, 0.42),
        ("MetaShift\nSCD\n(spatial)", 0.23, 0.22, 0.74),
        ("Waterbirds\nSCD\n(spatial)", 0.26, 0.26, 0.88),
    ]
    figure, axis = plt.subplots(figsize=(7.4, 2.8))
    x = np.arange(len(rows))
    axis.bar(x - 0.18, [row[1] for row in rows], width=0.34, color=ORANGE, label="seen (in the concept loss)")
    axis.bar(x + 0.18, [row[2] for row in rows], width=0.34, color=BLUE, label="unseen (held-out fifth)")
    for xi, row in zip(x, rows):
        axis.text(xi - 0.18, row[1] + 0.01, f"{row[1]:.2f}", ha="center", fontsize=6.8)
        axis.text(xi + 0.18, row[2] + 0.01, f"{row[2]:.2f}", ha="center", fontsize=6.8)
    axis.set_xticks(x)
    axis.set_xticklabels([row[0] for row in rows], fontsize=7.2)
    axis.set_ylabel("explained variance\nof concept factors")
    axis.set_ylim(0, 0.7)
    axis.grid(axis="y", color=GRID)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=7.5, loc="upper right")
    save(figure, "memorization")


def decoupling() -> None:
    groups = ["cat\nindoor", "cat\noutdoor", "dog\nindoor", "dog\noutdoor"]
    arms = [
        ("CLIP teacher", [0.92, 0.88, 0.83, 0.91], GREEN),
        ("SCD student", [0.47, 0.07, 0.10, 0.47], BLUE),
        ("Region contrast", [0.61, 0.12, None, None], PURPLE),
        ("Shuffled-map student", [0.24, 0.08, 0.19, 0.29], "#9aa0a8"),
    ]
    figure, axis = plt.subplots(figsize=(5.2, 2.5))
    x = np.arange(4)
    width = 0.2
    for index, (name, values, color) in enumerate(arms):
        offset = (index - 1.5) * width
        for xi, value in zip(x, values):
            if value is not None:
                axis.bar(xi + offset, value, width=width, color=color, label=name if xi == 0 else None)
    axis.set_xticks(x)
    axis.set_xticklabels(groups)
    axis.set_ylabel("EV of the cat factor\non unseen images")
    axis.axhline(0, color=MUTED, lw=0.6)
    axis.grid(axis="y", color=GRID)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    axis.annotate("minority", xy=(1, 0.14), ha="center", fontsize=7, color=ORANGE)
    axis.annotate("minority", xy=(2, 0.24), ha="center", fontsize=7, color=ORANGE)
    save(figure, "decoupling")


def group_accuracies() -> None:
    data = {
        "Waterbirds": (["landbird\nland", "landbird\nwater", "waterbird\nland", "waterbird\nwater"], {
            "SimCLR": [48, 54, 54, 63],
            "SCD": [54, 65, 60, 67],
            "Compositing": [61, 67, 68, 68],
            "Symmetric mixing": [49, 58, 67, 61],
        }, [1, 2]),
        "MetaShift": (["cat\nindoor", "cat\noutdoor", "dog\nindoor", "dog\noutdoor"], {
            "SimCLR": [57, 47, 53, 64],
            "SCD": [64, 51, 57, 73],
            "Compositing": [70, 55, 57, 71],
            "Symmetric mixing": [59, 53, 61, 72],
        }, [1, 2]),
    }
    colors = {"SimCLR": GREEN, "SCD": BLUE, "Compositing": ORANGE, "Symmetric mixing": PURPLE}
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for axis, (title, (names, arms, minority)) in zip(axes, data.items()):
        x = np.arange(4)
        for index, (arm, values) in enumerate(arms.items()):
            axis.bar(x + (index - 1.5) * 0.2, values, width=0.2, color=colors[arm], label=arm)
        for m in minority:
            axis.axvspan(m - 0.45, m + 0.45, color="#fff3e0", zorder=-1)
        axis.set_xticks(x)
        axis.set_xticklabels(names, fontsize=7.5)
        axis.set_ylim(40, 78)
        axis.set_title(title + " (shaded: minority groups)", fontsize=9)
        axis.grid(axis="y", color=GRID)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("validation accuracy (%)")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    figure.tight_layout()
    save(figure, "group_accuracies")


def grouping_signals() -> None:
    pairs = ["path /\nwalkway", "heron /\negret", "ocean /\nsea", "forest /\nwoodland",
             "bamboo /\njungle", "gull /\nseascape", "heron /\nmarsh"]
    signals = {
        "centred SpLiCE text": [0.23, 0.70, 0.71, 0.49, 0.24, 0.16, 0.08],
        "raw CLIP text": [0.85, 0.87, 0.94, 0.90, 0.67, 0.65, 0.67],
        "co-activation": [0.37, 0.09, 0.00, 0.08, 0.10, 0.02, 0.06],
        "image response": [0.66, 0.79, 0.94, 0.89, 0.67, 0.58, 0.11],
    }
    colors = [MUTED, BLUE, ORANGE, GREEN]
    figure, axis = plt.subplots(figsize=(7.0, 2.5))
    x = np.arange(len(pairs))
    for index, ((name, values), color) in enumerate(zip(signals.items(), colors)):
        axis.bar(x + (index - 1.5) * 0.2, values, width=0.2, color=color, label=name)
    axis.axhline(0.85, color=BLUE, lw=0.9, ls="--")
    axis.text(6.6, 0.87, "meaning threshold 0.85", color=BLUE, fontsize=7, ha="right")
    axis.axvline(3.5, color=INK, lw=0.6)
    axis.text(1.5, 1.04, "synonyms: should merge", ha="center", fontsize=8)
    axis.text(5.0, 1.04, "object / context: must stay apart", ha="center", fontsize=8)
    axis.set_xticks(x)
    axis.set_xticklabels(pairs, fontsize=7.5)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("similarity")
    axis.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    axis.grid(axis="y", color=GRID)
    axis.set_axisbelow(True)
    save(figure, "grouping_signals")


def image_level_rounds() -> None:
    # MetaShift: every image-level attempt raised average accuracy, none raised the minority groups.
    rows = [
        ("SimCLR", 55.2, 47, 53),
        ("F2 w1", 58.2, 42, 58),
        ("F2 w3", 61.5, 47, 54),
        ("F2 whitened w3", 56.9, 51, 55),
        ("F2 atypicality w3", 58.8, 52, 55),
        ("F2 balancing w3", 59.0, 44, 55),
        ("F3 blocks", 60.0, 45, 51),
        ("xfit w10", 61.0, 51, 48),
        ("SCD (spatial)", 61.2, 51, 57),
    ]
    figure, axis = plt.subplots(figsize=(7.0, 2.5))
    x = np.arange(len(rows))
    axis.plot(x, [r[1] for r in rows], "o-", color=INK, label="average accuracy")
    axis.plot(x, [r[2] for r in rows], "s--", color=ORANGE, label="cat outdoor (minority)")
    axis.plot(x, [r[3] for r in rows], "^--", color=PURPLE, label="dog indoor (minority)")
    axis.set_xticks(x)
    axis.set_xticklabels([r[0] for r in rows], fontsize=7.2, rotation=20)
    axis.set_ylabel("validation acc. (%)")
    axis.grid(color=GRID)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    axis.set_ylim(38, 68)
    save(figure, "metashift_rounds")


if __name__ == "__main__":
    scoreboard()
    spur_cifar()
    paired_seeds()
    memorization()
    decoupling()
    group_accuracies()
    grouping_signals()
    image_level_rounds()
