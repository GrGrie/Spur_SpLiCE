"""Retained CoSpRo teacher edges: four same-class, cross-background pairs and one wrong-class failure.

Reads the audit JSON that ``tools.paper.build_submission_figure`` writes (``evidence.json``). The
four examples are the retained same-class, cross-background pairs with the largest projection gain,
one per concept group; the failure is the heaviest retained wrong-class edge from another group.
Labels are post-hoc and illustrate the edges; they play no part in graph construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.paper.figure_style import FIGURE_DIR, INK, MUTED, save, use_paper_style

CLASSES = {0: "landbird", 1: "waterbird"}
PLACES = {0: "land", 1: "water"}


def choose(pairs: list[dict]) -> tuple[list[dict], dict]:
    good = sorted((p for p in pairs if p["stratum"] == "same_y_different_a" and p["retained"]), key=lambda p: -p["gain"])
    chosen, used = [], set()
    for pair in good:
        if pair["group_id"] not in used:
            chosen.append(pair)
            used.add(pair["group_id"])
        if len(chosen) == 4:
            break
    failures = [p for p in pairs if p["stratum"].startswith("different_y") and p["retained"] and p["group_id"] not in used]
    return chosen, max(failures, key=lambda p: p["final_edge_weight"])


def square_tile(path: Path, size: int = 320):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size))


def local_image(annotation: dict, dataset_root: Path) -> Path:
    """The image under ``dataset_root``; the audit stores the path of the machine that built it."""

    parts = Path(annotation["image"]).parts
    return dataset_root / parts[-2] / parts[-1]


def label(annotation: dict) -> str:
    return f"{CLASSES[annotation['target']]} · {PLACES[annotation['spurious_attribute']]}"


def plot(evidence: Path, dataset_root: Path, output_dir: Path) -> None:
    use_paper_style()
    import matplotlib.pyplot as plt

    chosen, failure = choose(json.loads(evidence.read_text(encoding="utf-8"))["all_pairs"])
    columns = [(pair, False) for pair in chosen] + [(failure, True)]
    figure, axes = plt.subplots(2, 5, figsize=(9.2, 4.3), gridspec_kw={"wspace": 0.08, "hspace": 0.12})
    for column, (pair, is_failure) in enumerate(columns):
        for row, side in enumerate(("left_annotation", "right_annotation")):
            axis = axes[row, column]
            axis.imshow(square_tile(local_image(pair[side], dataset_root)))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(INK if is_failure else MUTED)
                spine.set_linewidth(1.6 if is_failure else 0.8)
                spine.set_linestyle((0, (3, 2)) if is_failure else "-")
            axis.set_xlabel(label(pair[side]), fontsize=8, color=INK, labelpad=3)
        concepts = ", ".join(pair["concepts"]).replace(" photography", "\nphotography").replace(" of the snow", "\nof the snow")
        axes[0, column].set_title(f"removed: {concepts}", fontsize=8, color="#222222")
    axes[0, 0].set_ylabel("anchor", fontsize=9, color=INK)
    axes[1, 0].set_ylabel("teacher neighbour", fontsize=9, color=INK)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    to_figure = figure.transFigure.inverted()
    # Sit the group labels above the tallest column title, two-line titles included.
    title_top = max(to_figure.transform(axis.title.get_window_extent(renderer))[1][1] for axis in axes[0])

    def group_label(first: int, last: int, text: str) -> None:
        left, right = axes[0, first].get_position(), axes[0, last].get_position()
        y = title_top + 0.03
        figure.add_artist(plt.Line2D([left.x0, right.x1], [y - 0.012, y - 0.012], color=MUTED, linewidth=0.8))
        figure.text((left.x0 + right.x1) / 2, y, text, ha="center", va="bottom", fontsize=9, color="#222222")

    group_label(0, 3, "Same class, other background")
    group_label(4, 4, "Wrong class")
    save(figure, output_dir, "edge_examples")
    for pair, is_failure in columns:
        print(f"{'failure' if is_failure else 'example'}: {pair['concepts']} {label(pair['left_annotation'])} -> "
              f"{label(pair['right_annotation'])} (gain {pair['gain']:.4f}, weight {pair['final_edge_weight']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True, help="evidence.json from build_submission_figure")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Waterbirds image directory")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    plot(args.evidence, args.dataset_root, args.output_dir)


if __name__ == "__main__":
    main()
