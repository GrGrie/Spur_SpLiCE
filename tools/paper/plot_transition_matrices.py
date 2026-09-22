"""Teacher-edge mass between (class, background) groups: raw CLIP, CoSpRo and their difference.

Each anchor contributes its confidence times its teacher preference over neighbours, as in the
CoSpRo objective, and every row is normalized to 100%. These are the numbers the graph-audit text
quotes (for example 59.2% versus 45.0% for landbirds on water). Labels are post-hoc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cospro.config.settings import data_folder
from cospro.diagnostics.labels import load_labels
from cospro.tracking.artifacts import resolve_output_root
from tools.paper.figure_style import FIGURE_DIR, save, use_paper_style

GROUPS = ["LB\nland", "LB\nwater", "WB\nland", "WB\nwater"]
# Cells that keep the class and flip the background.
COUNTERFACTUAL_CELLS = ((0, 1), (1, 0), (2, 3), (3, 2))


def transition(graph_path: Path, labels) -> np.ndarray:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    y, a = labels.for_ids([str(value) for value in graph["sample_ids"]])
    group = 2 * np.asarray(y) + np.asarray(a)
    mass = np.zeros((4, 4))
    for anchor, (neighbours, weights, confidence) in enumerate(
        zip(graph["neighbor_indices"], graph["weights"], graph["anchor_confidence"])
    ):
        for neighbour, weight in zip(neighbours, weights):
            if neighbour >= 0 and weight > 0:
                mass[group[anchor], group[neighbour]] += confidence * weight
    return 100 * mass / mass.sum(1, keepdims=True)


def plot(cospro_graph: Path, raw_graph: Path, dataset_folder: Path, output_dir: Path) -> None:
    use_paper_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    labels = load_labels("waterbirds", dataset_folder)
    raw, cospro = transition(raw_graph, labels), transition(cospro_graph, labels)
    blues = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
    diverging = LinearSegmentedColormap.from_list("diverging", ["#c2451a", "#eb6834", "#f2f2f0", "#2a78d6", "#184f95"])
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), gridspec_kw={"wspace": 0.18})
    panels = (
        (raw, "Raw CLIP graph", blues, None),
        (cospro, "CoSpRo graph", blues, None),
        (cospro - raw, "CoSpRo − raw CLIP (points)", diverging, TwoSlopeNorm(0, -20, 20)),
    )
    for axis, (matrix, title, cmap, norm) in zip(axes, panels):
        image = axis.imshow(matrix, cmap=cmap, norm=norm, vmin=None if norm else 0, vmax=None if norm else 100)
        for (row, column), value in np.ndenumerate(matrix):
            dark = abs(value) > 12 if norm else value > 55
            axis.text(column, row, f"{value:+.1f}" if norm else f"{value:.1f}", ha="center", va="center",
                      fontsize=8, color="white" if dark else "#222222")
        for row, column in COUNTERFACTUAL_CELLS:
            axis.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#222222",
                                         linewidth=1.6, linestyle=(0, (3, 2))))
        axis.set_xticks(range(4), GROUPS, fontsize=8)
        axis.set_yticks(range(4), GROUPS if axis is axes[0] else [], fontsize=8)
        axis.set_xlabel("neighbour group")
        axis.set_title(title, fontsize=9, loc="left")
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03).outline.set_visible(False)
    axes[0].set_ylabel("anchor group")
    save(figure, output_dir, "transition_matrices")
    print("raw CLIP (%):\n", np.round(raw, 1), "\nCoSpRo (%):\n", np.round(cospro, 1))


def main() -> None:
    graphs = resolve_output_root() / "shared" / "waterbirds" / "graphs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cospro-graph", type=Path, default=graphs / "crp_graph.json")
    parser.add_argument("--raw-graph", type=Path, default=graphs / "raw_clip_graph.json")
    parser.add_argument("--data-folder", type=Path, default=data_folder(), help="Dataset root (for y and a)")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    plot(args.cospro_graph, args.raw_graph, args.data_folder, args.output_dir)


if __name__ == "__main__":
    main()
