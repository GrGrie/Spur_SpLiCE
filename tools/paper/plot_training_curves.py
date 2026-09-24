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
from tools.paper.figure_style import FIGURE_DIR, GRID, color_for, save, use_paper_style

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


def probe_history(record: dict, key: str) -> list[tuple[int, float]]:
    """(SSL epoch, metric) for every periodic probe. Older records log it under ``linear_probe``."""

    events = record["metrics"]
    final = [(e["step"], e["values"]["metrics"][key]) for e in events if e["stage"] == "linear_probe_final"]
    return sorted(final or [(e["step"], e["values"][key]) for e in events if e["stage"] == "linear_probe"])


def latest_per_seed(seeds_root: Path, pattern: str) -> list[dict]:
    """The most recent complete record of every seed the pattern matches.

    A seed that was run again keeps both attempt directories, so the newest one wins, as it does
    in cospro.cli.collect_results.
    """

    newest: dict[str, tuple[str, dict]] = {}
    for path in sorted(seeds_root.glob(pattern)):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            continue
        seed = str(record.get("identity", {}).get("seed", path.parent.parent.name))
        updated = str(record.get("timestamps", {}).get("updated_at", ""))
        if seed not in newest or updated >= newest[seed][0]:
            newest[seed] = (updated, record)
    return [record for _, record in newest.values()]


def seed_matrix(seeds_root: Path, pattern: str, key: str, seeds: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Probe epochs and a (seed, epoch) matrix of the metric."""

    runs = [history for record in latest_per_seed(seeds_root, pattern) if (history := probe_history(record, key))]
    if (seeds and len(runs) != seeds) or not runs or any([step for step, _ in run] != [step for step, _ in runs[0]] for run in runs):
        raise ValueError(f"{pattern}: expected {seeds or 'matching'} seeds with matching probe epochs, found {len(runs)}")
    return np.array([step for step, _ in runs[0]]), np.array([[value for _, value in run] for run in runs])


def smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Centred moving average; the ends average over the points that exist."""

    half = window // 2
    return np.array([values[max(0, i - half): i + half + 1].mean() for i in range(len(values))])


def plot(seeds_root: Path, output_dir: Path, methods: dict[str, str] = METHODS, name: str = "training_curves",
         seeds: int = 4) -> None:
    use_paper_style()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True, gridspec_kw={"hspace": 0.2})
    handles = {}
    for axis, (key, label) in zip(axes, PANELS):
        for position, (method, pattern) in enumerate(methods.items()):
            epochs, values = seed_matrix(seeds_root, pattern, key, seeds)
            (handles[method],) = axis.plot(epochs, smooth(values.mean(0)), color=color_for(method, position), linewidth=2.2,
                                           solid_capstyle="round", zorder=3 if method.startswith("CoSpRo") else 2)
        axis.set_ylabel(label)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_xlim(0, 510)
        axis.set_xticks(range(0, 501, 100))
        axis.tick_params(labelbottom=True)
    axes[1].set_xlabel("SSL epoch")
    # One legend for both panels, in the order the methods were passed.
    figure.legend(list(handles.values()), list(handles), loc="upper right", bbox_to_anchor=(0.985, 1.0),
                  ncol=len(handles),
                  frameon=False, fontsize=8.5, handlelength=1.8, columnspacing=1.2)
    figure.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.08)
    figure.align_ylabels(axes)
    save(figure, output_dir, name)
    for method, pattern in methods.items():
        finals = [seed_matrix(seeds_root, pattern, key, seeds)[1][:, -1].mean() for key, _ in PANELS]
        print(f"{method:<30} epoch-500 mean: WGA {finals[0]:.2f}, Avg {finals[1]:.2f}")


def named_runs(value: str) -> tuple[str, str]:
    label, separator, pattern = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(f"Expected LABEL=GLOB, got {value!r}")
    return label, pattern


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, help="Output tree holding seeds/ (default: outputs/).")
    parser.add_argument("--run", type=named_runs, action="append", default=[], metavar="LABEL=GLOB",
                        help="A method and the run records of its seeds, below <artifact root>/seeds. "
                             "Repeat in the order the legend should read; defaults to the Table 2 methods.")
    parser.add_argument("--seeds", type=int, default=0, help="Seeds each method must have; 0 accepts what it finds.")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--name", default="training_curves")
    args = parser.parse_args()
    methods = dict(args.run) if args.run else METHODS
    plot(resolve_output_root(args.artifact_root) / "seeds", args.output_dir, methods, args.name, args.seeds)


if __name__ == "__main__":
    main()
