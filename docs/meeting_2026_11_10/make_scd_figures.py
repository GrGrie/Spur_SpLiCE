"""Figures of the standalone SCD report (scd_report.tex): SimCLR, CoSpRo and SCD.

* ``scd_paired_seeds``: validation WGA per student seed, mean of the last four probes.
* ``scd_training_curves``: validation WGA and average accuracy over SSL epochs, mean over seeds 1-4,
  3-point smoothing.
* ``scd_group_accuracies``: validation accuracy per (class, attribute) group at the epoch-500 probe,
  mean and sample standard deviation over the available seeds.

Every value is read from the run records under outputs/seeds. Run from the repository root:

    python docs/meeting_2026_11_10/make_scd_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.paper.figure_style import GRID, INK, save, use_paper_style  # noqa: E402
from tools.paper.plot_training_curves import smooth  # noqa: E402

SEEDS = ROOT / "outputs" / "seeds"
OUT = Path(__file__).resolve().parent / "figures"
#: Method -> (run-record glob per dataset, colour).
METHODS = {
    "SimCLR": ({"waterbirds": "factors_waterbirds/seed_0*/simclr/*/run.json",
                "metashift": "factors_metashift/seed_0*/simclr/*/run.json"}, "#1baf7a"),
    "CoSpRo": ({"waterbirds": "paper_completion_2026_09_08_core/seed_0*/splice_crp_kl/*/run.json",
                "metashift": "metashift_cospro/seed_*/cospro/*/run.json"}, "#eb6834"),
    "SCD": ({"waterbirds": "factors_waterbirds/seed_0*/spatial_w5_holdout/*/run.json",
             "metashift": "factors_metashift/seed_0*/spatial_w5_holdout/*/run.json"}, "#2a78d6"),
}
DATASETS = {"waterbirds": "Waterbirds", "metashift": "MetaShift"}
GROUPS = {
    "waterbirds": ("landbird\nland", "landbird\nwater", "waterbird\nland", "waterbird\nwater"),
    "metashift": ("cat\nindoor", "cat\noutdoor", "dog\nindoor", "dog\noutdoor"),
}


def probes(record: dict) -> list[dict]:
    """Periodic probe metrics in epoch order; newer records nest them under ``linear_probe_final``."""

    events = record["metrics"]
    final = [(e["step"], e["values"]["metrics"]) for e in events if e["stage"] == "linear_probe_final"]
    chosen = final or [(e["step"], e["values"]) for e in events if e["stage"] == "linear_probe"]
    return [dict(values, epoch=step) for step, values in sorted(chosen, key=lambda item: item[0])]


def runs(pattern: str) -> dict[int, list[dict]]:
    """Seed -> probes of its newest complete record."""

    newest: dict[int, tuple[str, dict]] = {}
    for path in sorted(SEEDS.glob(pattern)):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            continue
        seed = int(record["identity"]["seed"])
        updated = str(record.get("timestamps", {}).get("updated_at", ""))
        if seed not in newest or updated >= newest[seed][0]:
            newest[seed] = (updated, record)
    return {seed: probes(record) for seed, (_, record) in sorted(newest.items())}


def wga_last4(history: list[dict]) -> float:
    return float(np.mean([p["Linear val worst-group acc"] for p in history[-4:]]))


def paired_seeds(data: dict) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True, gridspec_kw={"wspace": 0.06})
    for axis, (dataset, title) in zip(axes, DATASETS.items()):
        for method, (_, color) in METHODS.items():
            seeds = [seed for seed in data[method][dataset] if seed <= 6]
            values = [wga_last4(data[method][dataset][seed]) for seed in seeds]
            axis.plot(seeds, values, marker="o", color=color, linewidth=1.6, markersize=4.5, label=method)
            print(f"{dataset:<11} {method:<7} seeds {seeds} last-4 WGA {np.round(values, 2).tolist()}")
        axis.set_title(title, fontsize=10, color=INK)
        axis.set_xticks(range(1, 7))
        axis.set_xlabel("student seed")
        axis.grid(color=GRID, linewidth=0.6)
    axes[0].set_ylabel("validation WGA (%)")
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower left")
    figure.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.17)
    save(figure, OUT, "scd_paired_seeds")


def training_curves(data: dict) -> None:
    import matplotlib.pyplot as plt

    panels = (("Linear val worst-group acc", "Worst-group accuracy (%)"), ("Linear val acc", "Average accuracy (%)"))
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 5.4), sharex=True, gridspec_kw={"hspace": 0.18, "wspace": 0.22})
    handles = {}
    for column, (dataset, title) in enumerate(DATASETS.items()):
        for row, (key, label) in enumerate(panels):
            axis = axes[row, column]
            for method, (_, color) in METHODS.items():
                histories = [data[method][dataset][seed] for seed in range(1, 5)]
                epochs = np.array([p["epoch"] for p in histories[0]])
                values = np.array([[p[key] for p in history] for history in histories])
                (handles[method],) = axis.plot(epochs, smooth(values.mean(0)), color=color, linewidth=2.0,
                                               zorder=3 if method == "SCD" else 2)
            axis.grid(axis="y", color=GRID, linewidth=0.6)
            axis.set_xlim(0, 510)
            axis.set_xticks(range(0, 501, 100))
            if column == 0:
                axis.set_ylabel(label)
            if row == 0:
                axis.set_title(title, fontsize=10, color=INK)
            else:
                axis.set_xlabel("SSL epoch")
    figure.legend(list(handles.values()), list(handles), loc="upper center", bbox_to_anchor=(0.5, 1.02),
                  ncol=len(handles), frameon=False, fontsize=8.5)
    figure.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.08)
    save(figure, OUT, "scd_training_curves")


def group_accuracies(data: dict) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 2.9), sharey=True, gridspec_kw={"wspace": 0.06})
    width = 0.26
    for axis, (dataset, title) in zip(axes, DATASETS.items()):
        positions = np.arange(4)
        for offset, (method, (_, color)) in enumerate(METHODS.items()):
            values = np.array([h[-1]["Linear val group accuracies"] for h in data[method][dataset].values()])
            x = positions + (offset - 1) * width
            axis.bar(x, values.mean(0), width * 0.92, color=color, label=method, zorder=2)
            axis.errorbar(x, values.mean(0), yerr=values.std(0, ddof=1), fmt="none", ecolor=INK, elinewidth=0.8,
                          capsize=1.8, zorder=3)
            print(f"{dataset:<11} {method:<7} seeds {len(values)} groups {np.round(values.mean(0), 1).tolist()}")
        axis.set_xticks(positions)
        axis.set_xticklabels(GROUPS[dataset], fontsize=8)
        axis.set_title(title, fontsize=10, color=INK)
        axis.set_ylim(30, 80)
        axis.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    axes[0].set_ylabel("Validation accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False, fontsize=8.5)
    figure.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.17)
    save(figure, OUT, "scd_group_accuracies")


def main() -> None:
    use_paper_style()
    data = {method: {dataset: runs(pattern) for dataset, pattern in patterns.items()}
            for method, (patterns, _) in METHODS.items()}
    paired_seeds(data)
    training_curves(data)
    group_accuracies(data)


if __name__ == "__main__":
    main()
