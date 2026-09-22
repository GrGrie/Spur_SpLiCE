"""Validation worst-group and average accuracy over SSL epochs for the Table 2 methods.

Each line is the mean over seeds 1-4 of the periodic linear probes, smoothed with a 3-point centred
moving average. The unsmoothed epoch-500 means are the Table 2 values; the tool prints them so the
figure can be checked against the table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cospro.tracking.artifacts import resolve_output_root
from tools.paper.figure_style import COLORS, FIGURE_DIR, GRID, save, use_paper_style

# Table 2 method -> run records below <artifact root>/seeds.
METHODS = {
    "CoSpRo": "paper_completion_2026_09_08_core/seed_0*/splice_crp_kl/*/run.json",
    "Semantic SpLiCE graph": "next_actions_after_transfer_2026_09_07_graph_ablation/seed_0*/semantic_splice/*/run.json",
    "Raw CLIP graph": "paper_completion_2026_09_08_core/seed_0*/raw_clip_kl/*/run.json",
    "Direct SpLiCE reconstruction": (
        "next_actions_after_transfer_2026_09_07_direct_transfer/seed_0*/splice_reconstruction/*/run.json"
    ),
    "SimCLR": "paper_completion_2026_09_08_core/seed_0*/simclr/*/run.json",
}
PANELS = (
    ("Last linear val worst-group acc", "Worst-group accuracy (%)"),
    ("Last linear val acc", "Average accuracy (%)"),
)


def probe_history(run: Path, key: str) -> list[tuple[int, float]]:
    """(SSL epoch, metric) for every periodic probe. Older records log it under ``linear_probe``."""

    events = json.loads(run.read_text(encoding="utf-8"))["metrics"]
    final = [(e["step"], e["values"]["metrics"][key]) for e in events if e["stage"] == "linear_probe_final"]
    return sorted(final or [(e["step"], e["values"][key]) for e in events if e["stage"] == "linear_probe"])


def seed_matrix(seeds_root: Path, pattern: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Probe epochs and a (seed, epoch) matrix of the metric."""

    runs = [probe_history(path, key) for path in sorted(seeds_root.glob(pattern))]
    if len(runs) != 4 or any([step for step, _ in run] != [step for step, _ in runs[0]] for run in runs):
        raise ValueError(f"{pattern}: expected four seeds with matching probe epochs")
    return np.array([step for step, _ in runs[0]]), np.array([[value for _, value in run] for run in runs])


def smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Centred moving average; the ends average over the points that exist."""

    half = window // 2
    return np.array([values[max(0, i - half): i + half + 1].mean() for i in range(len(values))])


def plot(seeds_root: Path, output_dir: Path) -> None:
    use_paper_style()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True, gridspec_kw={"hspace": 0.2})
    handles = {}
    for axis, (key, label) in zip(axes, PANELS):
        for name, pattern in METHODS.items():
            epochs, values = seed_matrix(seeds_root, pattern, key)
            (handles[name],) = axis.plot(epochs, smooth(values.mean(0)), color=COLORS[name], linewidth=2.2,
                                         solid_capstyle="round", zorder=3 if name == "CoSpRo" else 2)
        axis.set_ylabel(label)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_xlim(0, 510)
        axis.set_xticks(range(0, 501, 100))
        axis.tick_params(labelbottom=True)
    axes[1].set_xlabel("SSL epoch")
    # One legend for both panels, in METHODS order: epoch-500 worst-group accuracy, best first.
    figure.legend(list(handles.values()), list(handles), loc="upper right", bbox_to_anchor=(0.985, 1.0), ncol=5,
                  frameon=False, fontsize=8.5, handlelength=1.8, columnspacing=1.2)
    figure.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.08)
    figure.align_ylabels(axes)
    save(figure, output_dir, "training_curves")
    for name, pattern in METHODS.items():
        finals = [seed_matrix(seeds_root, pattern, key)[1][:, -1].mean() for key, _ in PANELS]
        print(f"{name:<30} epoch-500 mean: WGA {finals[0]:.2f}, Avg {finals[1]:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, help="Output tree holding seeds/ (default: outputs/).")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    plot(resolve_output_root(args.artifact_root) / "seeds", args.output_dir)


if __name__ == "__main__":
    main()
