"""Teacher-edge mass between (class, attribute) groups: two graphs and their difference.

Each anchor contributes its confidence times its teacher preference over neighbours, as in the
CoSpRo objective, and every row is normalized to 100%. Dashed cells keep the class and change the
spurious attribute, the relations CoSpRo aims to add. Labels are post-hoc: they describe a graph
that was built without them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cospro.config.settings import data_folder
from cospro.data.registry import canonical_dataset_name
from cospro.diagnostics.labels import load_labels
from cospro.tracking.artifacts import resolve_output_root
from tools.paper.figure_style import FIGURE_DIR, save, use_paper_style


def transition(graph_path: Path, labels) -> np.ndarray:
    """Row-normalized edge mass from each anchor group to each neighbour group, in percent."""

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    y, a = labels.for_ids([str(value) for value in graph["sample_ids"]])
    group = np.asarray(y) * labels.n_attributes + np.asarray(a)
    size = (int(np.max(y)) + 1) * labels.n_attributes
    mass = np.zeros((size, size))
    for anchor, (neighbours, weights, confidence) in enumerate(
        zip(graph["neighbor_indices"], graph["weights"], graph["anchor_confidence"])
    ):
        for neighbour, weight in zip(neighbours, weights):
            if neighbour >= 0 and weight > 0:
                mass[group[anchor], group[neighbour]] += confidence * weight
    rows = mass.sum(1, keepdims=True)
    return 100 * mass / np.where(rows > 0, rows, 1)


def counterfactual_cells(n_attributes: int, size: int) -> list[tuple[int, int]]:
    """Cells that keep the class and change the spurious attribute."""

    return [(row, column) for row in range(size) for column in range(size)
            if row // n_attributes == column // n_attributes and row != column]


def plot(dataset: str, graphs: list[tuple[str, Path]], dataset_folder: Path, output_dir: Path, name: str) -> None:
    use_paper_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    labels = load_labels(dataset, dataset_folder)
    matrices = [(title, transition(path, labels)) for title, path in graphs]
    if len(matrices) != 2:
        raise ValueError("Pass exactly two graphs: the control first and the graph of interest second.")
    difference = matrices[1][1] - matrices[0][1]
    group_names = [name.replace(" / ", "\n") for name in labels.group_names()]
    size = len(group_names)
    blues = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
    diverging = LinearSegmentedColormap.from_list("diverging", ["#c2451a", "#eb6834", "#f2f2f0", "#2a78d6", "#184f95"])
    panels = [(matrices[0][1], matrices[0][0], blues, None), (matrices[1][1], matrices[1][0], blues, None),
              (difference, f"{matrices[1][0]} − {matrices[0][0]} (points)", diverging, TwoSlopeNorm(0, -20, 20))]
    figure, axes = plt.subplots(1, 3, figsize=(3.2 * size + 0.4, 0.75 * size + 1.4), gridspec_kw={"wspace": 0.18})
    for axis, (matrix, title, cmap, norm) in zip(axes, panels):
        image = axis.imshow(matrix, cmap=cmap, norm=norm, vmin=None if norm else 0, vmax=None if norm else 100)
        for (row, column), value in np.ndenumerate(matrix):
            dark = abs(value) > 12 if norm else value > 55
            axis.text(column, row, f"{value:+.1f}" if norm else f"{value:.1f}", ha="center", va="center",
                      fontsize=8, color="white" if dark else "#222222")
        for row, column in counterfactual_cells(labels.n_attributes, size):
            axis.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#222222",
                                         linewidth=1.6, linestyle=(0, (3, 2))))
        axis.set_xticks(range(size), group_names, fontsize=8)
        axis.set_yticks(range(size), group_names if axis is axes[0] else [], fontsize=8)
        axis.set_xlabel("neighbour group")
        axis.set_title(title, fontsize=9, loc="left")
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03).outline.set_visible(False)
    axes[0].set_ylabel("anchor group")
    save(figure, output_dir, name)
    for title, matrix in matrices:
        print(f"{title} (%):\n{np.round(matrix, 1)}")


def named_graph(value: str) -> tuple[str, Path]:
    title, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(f"Expected TITLE=PATH, got {value!r}")
    return title, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=canonical_dataset_name, default="waterbirds")
    parser.add_argument("--graph", type=named_graph, action="append", default=[], metavar="TITLE=PATH",
                        help="Pass twice: the control first, the graph of interest second.")
    parser.add_argument("--data-folder", type=Path, default=data_folder(), help="Dataset root (for y and a)")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--name", default="", help="Figure file name; defaults to <dataset>_transition_matrices.")
    args = parser.parse_args()
    graphs = args.graph
    if not graphs:
        directory = resolve_output_root() / "shared" / "waterbirds" / "graphs"
        graphs = [("Raw CLIP graph", directory / "raw_clip_graph.json"), ("CoSpRo graph", directory / "crp_graph.json")]
    name = args.name or ("transition_matrices" if args.dataset == "waterbirds" else f"{args.dataset}_transition_matrices")
    plot(args.dataset, graphs, args.data_folder, args.output_dir, name)


if __name__ == "__main__":
    main()
