"""Figures of the spatial concept paper (paper/SpatialConcepts.tex).

* ``spatial_training_curves``: validation worst-group and average accuracy over SSL epochs on
  Waterbirds and MetaShift, mean over seeds 1-4 of the periodic probes, 3-point smoothing.
* ``spatial_group_accuracies``: validation accuracy of every (class, attribute) group, mean of the
  last four probes over seeds 1-6.
* ``spatial_map_examples``: MetaShift concept maps of one training image per group, with a SimCLR
  crop and the 7 x 7 targets it receives. It needs the stored maps and the images
  (``--maps``, ``--data-folder``); the other two figures read only the study summaries.

    python -m tools.paper.plot_spatial_concept_figures --maps <concept_maps.pt> --data-folder E:/Datasets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cospro.tracking.artifacts import PROJECT_ROOT
from tools.paper.figure_style import FIGURE_DIR, GRID, INK, MUTED, save, use_paper_style
from tools.paper.plot_training_curves import smooth

ARMS = {
    "SCD": ("spatial_w5_holdout", "#2a78d6"),
    "Shuffled maps": ("spatial_w5_shuffled_holdout", "#eb6834"),
    "CLIP principal-component maps": ("spatial_pca_w5_holdout", "#eda100"),
    "SimCLR": ("simclr", "#1baf7a"),
}
DATASETS = {"waterbirds": "Waterbirds", "metashift": "MetaShift"}
GROUPS = {
    "waterbirds": ("landbird\nland", "landbird\nwater", "waterbird\nland", "waterbird\nwater"),
    "metashift": ("cat\nindoor", "cat\noutdoor", "dog\nindoor", "dog\noutdoor"),
}
#: Probe metrics in the study summaries' trajectories.
PANELS = (("wga", "Worst-group accuracy (%)"), ("acc", "Average accuracy (%)"))
#: Map cell colours of the three concepts that win most cells of an image.
CONCEPT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def study_runs(dataset: str) -> list[dict]:
    path = PROJECT_ROOT / "outputs" / "reports" / f"factors_{dataset}" / "summary.json"
    return [run for run in json.loads(path.read_text(encoding="utf-8"))["runs"] if run.get("status") == "complete"]


def arm_runs(runs: list[dict], arm: str, seeds: range) -> list[dict]:
    chosen = {run["seed"]: run for run in runs if run["arm"] == arm and run["seed"] in seeds}
    return [chosen[seed] for seed in sorted(chosen)]


def plot_curves(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(8.4, 5.4), sharex=True, gridspec_kw={"hspace": 0.18, "wspace": 0.22})
    handles = {}
    for column, (dataset, title) in enumerate(DATASETS.items()):
        runs = study_runs(dataset)
        for row, (key, label) in enumerate(PANELS):
            axis = axes[row, column]
            for method, (arm, color) in ARMS.items():
                chosen = arm_runs(runs, arm, range(1, 5))
                if len(chosen) != 4:
                    raise ValueError(f"{dataset} {arm}: expected seeds 1-4, found {[run['seed'] for run in chosen]}")
                epochs = np.array([point["epoch"] for point in chosen[0]["trajectory"]])
                values = np.array([[point[key] for point in run["trajectory"]] for run in chosen])
                (handles[method],) = axis.plot(epochs, smooth(values.mean(0)), color=color, linewidth=2.0,
                                               solid_capstyle="round", zorder=3 if arm == "spatial_w5_holdout" else 2)
                if row == 0:
                    print(f"{dataset:<11} {method:<30} last-4 mean WGA {values[:, -4:].mean():.2f}")
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
                  ncol=len(handles), frameon=False, fontsize=8.5, handlelength=1.8, columnspacing=1.2)
    figure.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.08)
    save(figure, output_dir, "spatial_training_curves")


def plot_groups(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    shown = {name: ARMS[name] for name in ("SimCLR", "Shuffled maps", "SCD")}
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 2.9), sharey=True, gridspec_kw={"wspace": 0.06})
    width = 0.26
    for axis, (dataset, title) in zip(axes, DATASETS.items()):
        runs = study_runs(dataset)
        positions = np.arange(4)
        for offset, (method, (arm, color)) in enumerate(shown.items()):
            values = np.array([run["val_group_accuracies"] for run in arm_runs(runs, arm, range(1, 7))])
            x = positions + (offset - 1) * width
            axis.bar(x, values.mean(0), width * 0.92, color=color, label=method, zorder=2)
            axis.errorbar(x, values.mean(0), yerr=values.std(0, ddof=1), fmt="none", ecolor=INK, elinewidth=0.8,
                          capsize=1.8, zorder=3)
            print(f"{dataset:<11} {method:<22} seeds {len(values)} groups {np.round(values.mean(0), 1).tolist()}")
        axis.set_xticks(positions)
        axis.set_xticklabels(GROUPS[dataset], fontsize=8)
        axis.set_title(title, fontsize=10, color=INK)
        axis.set_ylim(30, 80)
        axis.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    axes[0].set_ylabel("Validation accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False, fontsize=8.5)
    figure.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.17)
    save(figure, output_dir, "spatial_group_accuracies")


def plot_examples(output_dir: Path, maps_path: Path, data_folder: str) -> None:
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    from cospro.data.registry import dataset_class
    from cospro.pipeline.concept_maps import warp_maps

    stored = torch.load(maps_path, map_location="cpu", weights_only=False)
    maps = stored["maps"].float()
    names = [name.split(" | ")[-1].split(" (+")[0] for name in stored["factor_names"]]
    rows = {int(sample.rpartition(":")[2]): row for row, sample in enumerate(stored["sample_ids"])}
    cls = dataset_class("metashift")
    dataset = cls.from_config(cls.Config(root_dir=data_folder))
    metadata = dataset.metadata_array
    train = [int(index) for index in dataset.get_subset("train").indices]
    animal = {"cat": ("cat", "cats", "kitten"), "dog": ("dog", "puppies", "pup")}

    def top(index: int, count: int = 3) -> list[int]:
        winners = maps[rows[index]].argmax(dim=0)
        return torch.bincount(winners.flatten(), minlength=len(names)).argsort(descending=True)[:count].tolist()

    def pick(target: int, attribute: int) -> int:
        """For illustration: the image of the group whose three leading concepts cover most cells and name its animal."""

        words = animal["cat" if target == 0 else "dog"]
        found = [index for index in train if int(metadata[index, 1]) == target and int(metadata[index, 0]) == attribute]
        scored = []
        for index in found:
            leading = top(index)
            if not any(part in words for part in stored["factor_names"][leading[0]].split(" | ") + [names[leading[0]]]
                       ) and not any(names[factor] in words for factor in leading):
                continue
            winners = maps[rows[index]].argmax(dim=0)
            scored.append((float(sum((winners == factor).float().mean() for factor in leading)), index))
        return max(scored)[1]

    labels = {(0, 0): "cat, indoor", (0, 1): "cat, outdoor", (1, 0): "dog, indoor", (1, 1): "dog, outdoor"}
    chosen = [pick(0, 0), pick(0, 1), pick(1, 0), pick(1, 1)]
    crops = torch.tensor([[0.18, 0.10, 0.62, 0.62, 0.0], [0.30, 0.22, 0.58, 0.58, 0.0],
                          [0.12, 0.25, 0.64, 0.64, 0.0], [0.25, 0.15, 0.60, 0.60, 0.0]])
    cmap = ListedColormap(list(CONCEPT_COLORS) + ["#00000000"])
    figure, axes = plt.subplots(3, 4, figsize=(8.4, 7.0), gridspec_kw={"hspace": 0.32, "wspace": 0.06})
    for column, index in enumerate(chosen):
        image = dataset.get_input(index).convert("RGB").resize((224, 224))
        leading = top(index)
        winners = maps[rows[index]].argmax(dim=0)
        coded = torch.full_like(winners, 3)
        for rank, factor in enumerate(leading):
            coded[winners == factor] = rank
        full = F.interpolate(coded[None, None].float(), size=(224, 224), mode="nearest")[0, 0].numpy()
        top_, left, height, width, _ = crops[column].tolist()
        axes[0, column].imshow(image)
        axes[0, column].add_patch(Rectangle((left * 224, top_ * 224), width * 224, height * 224, fill=False,
                                            edgecolor="white", linewidth=1.6, linestyle="--"))
        axes[0, column].set_title(labels[(int(metadata[index, 1]), int(metadata[index, 0]))], fontsize=9.5, color=INK)
        axes[1, column].imshow(image)
        axes[1, column].imshow(full, cmap=cmap, vmin=0, vmax=3, alpha=0.55, interpolation="nearest")
        warped = warp_maps(maps[rows[index]][None], crops[column][None], (7, 7))[0].argmax(dim=0)
        cell = torch.full_like(warped, 3)
        for rank, factor in enumerate(leading):
            cell[warped == factor] = rank
        crop = image.crop((left * 224, top_ * 224, (left + width) * 224, (top_ + height) * 224)).resize((224, 224))
        axes[2, column].imshow(crop)
        axes[2, column].imshow(F.interpolate(cell[None, None].float(), size=(224, 224), mode="nearest")[0, 0].numpy(),
                               cmap=cmap, vmin=0, vmax=3, alpha=0.55, interpolation="nearest")
        for line in range(1, 7):
            axes[2, column].axhline(line * 32 - 0.5, color="white", linewidth=0.5, alpha=0.8)
            axes[2, column].axvline(line * 32 - 0.5, color="white", linewidth=0.5, alpha=0.8)
        axes[2, column].legend(handles=[Patch(color=CONCEPT_COLORS[rank], label=names[factor])
                                        for rank, factor in enumerate(leading)],
                               loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False, fontsize=7.5,
                               handlelength=1.0, labelspacing=0.25)
        for axis in axes[:, column]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
    axes[0, 0].set_ylabel("image and crop", fontsize=9, color=INK)
    axes[1, 0].set_ylabel("concept map, 14 x 14", fontsize=9, color=INK)
    axes[2, 0].set_ylabel("crop target, 7 x 7", fontsize=9, color=INK)
    print("example images", chosen, [[names[factor] for factor in top(index)] for index in chosen])
    save(figure, output_dir, "spatial_map_examples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", type=Path, help="MetaShift concept_maps.pt; without it the example figure is skipped.")
    parser.add_argument("--data-folder", default="", help="Dataset root holding MetaShift.")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    use_paper_style()
    plot_curves(args.output_dir)
    plot_groups(args.output_dir)
    if args.maps:
        plot_examples(args.output_dir, args.maps, args.data_folder)


if __name__ == "__main__":
    main()
