"""Retained teacher edges of any dataset: cross-context pairs and one wrong-class failure.

Reads a teacher graph and the dataset adapter that produced its sample IDs, so it works for every
registered dataset. The examples are the retained edges with the largest confidence times teacher
preference that keep the class and change the spurious attribute, one per concept group, followed
by the heaviest retained wrong-class edge from another group. Labels are post-hoc: the graph was
built without them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cospro.config.settings import data_folder
from cospro.data.registry import canonical_dataset_name, dataset_class
from cospro.diagnostics.labels import load_labels
from tools.paper.figure_style import FIGURE_DIR, INK, MUTED, save, use_paper_style


def edges(graph: dict) -> list[dict]:
    """Every retained edge with its weight, confidence and concept group."""

    indices = np.asarray(graph["neighbor_indices"])
    weights = np.asarray(graph["weights"], dtype=float)
    confidence = np.asarray(graph["anchor_confidence"], dtype=float)
    group_ids = np.asarray(graph.get("group_ids", np.full(indices.shape, -1)))
    rows, columns = np.where((indices >= 0) & (weights > 0))
    return [{"anchor": int(row), "neighbour": int(indices[row, column]), "weight": float(weights[row, column]),
             "mass": float(confidence[row] * weights[row, column]), "group": int(group_ids[row, column])}
            for row, column in zip(rows, columns)]


def choose(graph: dict, y: np.ndarray, a: np.ndarray, examples: int) -> tuple[list[dict], dict | None]:
    """The strongest cross-context edge of each concept group, and one wrong-class edge."""

    ranked = sorted(edges(graph), key=lambda edge: -edge["mass"])
    chosen, used = [], set()
    for edge in ranked:
        anchor, neighbour = edge["anchor"], edge["neighbour"]
        if y[anchor] == y[neighbour] and a[anchor] != a[neighbour] and edge["group"] not in used:
            chosen.append(edge)
            used.add(edge["group"])
        if len(chosen) == examples:
            break
    failures = [edge for edge in ranked if y[edge["anchor"]] != y[edge["neighbour"]] and edge["group"] not in used]
    return chosen, failures[0] if failures else None


def square_tile(image, size: int = 320):
    """A centre crop, so images of different shapes line up in the grid."""

    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    return image.convert("RGB").crop((left, top, left + side, top + side)).resize((size, size))


def plot(dataset: str, graph_path: Path, dataset_folder: Path, output_dir: Path, name: str, examples: int) -> None:
    use_paper_style()
    import matplotlib.pyplot as plt

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    labels = load_labels(dataset, dataset_folder)
    sample_ids = [str(value) for value in graph["sample_ids"]]
    rows = [int(str(sample_id).partition(":")[2]) for sample_id in sample_ids]
    y, a = labels.for_ids(sample_ids)
    adapter = dataset_class(dataset)
    images = adapter.from_config(adapter.Config(root_dir=str(dataset_folder)))
    concepts = {group["group_id"]: group["concepts"] for group in graph.get("groups", [])}
    chosen, failure = choose(graph, np.asarray(y), np.asarray(a), examples)
    columns = [(edge, False) for edge in chosen] + ([(failure, True)] if failure else [])

    def caption(index: int) -> str:
        names = labels.group_names()[int(y[index]) * labels.n_attributes + int(a[index])]
        return names.replace(" / ", " · ")

    figure, axes = plt.subplots(2, len(columns), figsize=(1.85 * len(columns), 4.3),
                                gridspec_kw={"wspace": 0.08, "hspace": 0.12}, squeeze=False)
    for column, (edge, is_failure) in enumerate(columns):
        for row, index in enumerate((edge["anchor"], edge["neighbour"])):
            axis = axes[row, column]
            axis.imshow(square_tile(images.get_input(rows[index])))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(INK if is_failure else MUTED)
                spine.set_linewidth(1.6 if is_failure else 0.8)
                spine.set_linestyle((0, (3, 2)) if is_failure else "-")
            axis.set_xlabel(caption(index), fontsize=8, color=INK, labelpad=3)
        title = ", ".join(concepts.get(edge["group"], [f"group {edge['group']}"]))
        axes[0, column].set_title(f"removed: {title}", fontsize=8, color="#222222")
    axes[0, 0].set_ylabel("anchor", fontsize=9, color=INK)
    axes[1, 0].set_ylabel("teacher neighbour", fontsize=9, color=INK)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    to_figure = figure.transFigure.inverted()
    title_top = max(to_figure.transform(axis.title.get_window_extent(renderer))[1][1] for axis in axes[0])

    def group_label(first: int, last: int, text: str) -> None:
        left, right = axes[0, first].get_position(), axes[0, last].get_position()
        height = title_top + 0.03
        figure.add_artist(plt.Line2D([left.x0, right.x1], [height - 0.012, height - 0.012], color=MUTED, linewidth=0.8))
        figure.text((left.x0 + right.x1) / 2, height, text, ha="center", va="bottom", fontsize=9, color="#222222")

    group_label(0, len(chosen) - 1, "Same class, other context")
    if failure:
        group_label(len(chosen), len(chosen), "Wrong class")
    save(figure, output_dir, name)
    for edge, is_failure in columns:
        print(f"{'failure' if is_failure else 'example'}: {concepts.get(edge['group'])} "
              f"{caption(edge['anchor'])} -> {caption(edge['neighbour'])} (weight {edge['weight']:.3f}, mass {edge['mass']:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=canonical_dataset_name, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--data-folder", type=Path, default=data_folder())
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--name", default="", help="Figure file name; defaults to <dataset>_edge_examples.")
    parser.add_argument("--examples", type=int, default=4, help="Cross-context pairs to show.")
    args = parser.parse_args()
    plot(args.dataset, args.graph, args.data_folder, args.output_dir,
         args.name or f"{args.dataset}_edge_examples", args.examples)


if __name__ == "__main__":
    main()
