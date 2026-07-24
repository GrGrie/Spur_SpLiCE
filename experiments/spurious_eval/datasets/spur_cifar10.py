from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import datasets, transforms

from experiments.spurious_eval.metrics import compute_group_metrics
from experiments.spurious_eval.datasets.augmentation import (
    StrongAugmentationConfig,
    build_standard_and_strong_ssl_transforms,
)
from experiments.spurious_eval.datasets.paths import resolve_dataset_root
from experiments.spurious_eval.datasets.transforms import (
    ConceptAwareSSLSubset,
    ConceptAwareTwoCropTransform,
    TwoCropTransform,
    build_augmentation_routing,
)
from experiments.spurious_eval.datasets.wilds_compat import (
    CombinatorialGrouper,
    WILDSDataset,
    get_eval_loader,
    get_ssl_train_loader,
    get_train_loader,
)
from splice.ssl_regularization import dataset_score_cache_key


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
LINE_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
]


class RecolorHorizontalLine:
    """Break the synthetic class/line-colour shortcut in a counterfactual view."""

    def __init__(self, line_width: int, colors=LINE_COLORS) -> None:
        self.line_width = max(1, int(line_width))
        self.colors = list(colors)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.copy()
        draw = ImageDraw.Draw(image)
        width, height = image.size
        half_width = self.line_width // 2
        center_y = height // 2
        y0 = max(0, center_y - half_width)
        y1 = min(height - 1, y0 + self.line_width - 1)
        color_index = int(torch.randint(len(self.colors), size=(1,)).item())
        draw.rectangle([0, y0, width - 1, y1], fill=self.colors[color_index])
        return image


@dataclass(frozen=True)
class SpurCIFAR10Config(StrongAugmentationConfig):
    root_dir: str = "./datasets"
    image_size: int = 32
    train_split: str = "ds_train"
    eval_split: str = "val"
    ssl_crop_min: float = 0.2
    val_fraction: float = 0.1
    train_spurious_correlation: float = 0.95
    eval_spurious_correlation: float = 0.1
    spurious_seed: int = 0
    line_width: int = 2
    download: bool = True


def spur_cifar10_transforms(
    image_size: int = 32,
    ssl_crop_min: float = 0.2,
    strong_config: SpurCIFAR10Config | None = None,
) -> tuple[transforms.Compose, transforms.Compose, transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    strong_config = strong_config or SpurCIFAR10Config(image_size=image_size, ssl_crop_min=ssl_crop_min)
    ssl_train_transform, strong_ssl_train_transform = build_standard_and_strong_ssl_transforms(
        image_size=image_size,
        ssl_crop_min=ssl_crop_min,
        normalize=normalize,
        strong_config=strong_config,
    )
    if strong_config.splice_strong_line_recolor:
        strong_ssl_train_transform = transforms.Compose(
            [RecolorHorizontalLine(strong_config.line_width), strong_ssl_train_transform]
        )
    linear_train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return ssl_train_transform, strong_ssl_train_transform, linear_train_transform, eval_transform


class SpurCIFAR10Dataset(WILDSDataset):
    """CIFAR-10 with one class-associated horizontal-line color per class."""

    _dataset_name = "spur_cifar10"

    def __init__(
        self,
        root_dir: str = "./datasets",
        split_scheme: str = "official",
        val_fraction: float = 0.1,
        train_spurious_correlation: float = 0.95,
        eval_spurious_correlation: float = 0.1,
        spurious_seed: int = 0,
        line_width: int = 2,
        download: bool = True,
    ) -> None:
        requested_root = Path(root_dir).expanduser()
        self.root_dir = self._find_cifar_root(requested_root)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = self.root_dir
        self.line_width = line_width
        self.train_spurious_correlation = train_spurious_correlation
        self.spurious_seed = spurious_seed
        self.line_colors = LINE_COLORS

        self.train_dataset = datasets.CIFAR10(str(self.root_dir), train=True, download=download)
        self.test_dataset = datasets.CIFAR10(str(self.root_dir), train=False, download=download)
        train_labels = np.asarray(self.train_dataset.targets, dtype=np.int64)
        test_labels = np.asarray(self.test_dataset.targets, dtype=np.int64)

        train_count = len(train_labels)
        val_count = int(round(train_count * val_fraction))
        train_split_count = train_count - val_count
        split_train = np.zeros(train_split_count, dtype=np.int64)
        split_val = np.ones(val_count, dtype=np.int64)
        split_test = np.full(len(test_labels), 2, dtype=np.int64)
        self._split_array = np.concatenate([split_train, split_val, split_test])

        labels = np.concatenate([train_labels, test_labels])
        self._y_array = torch.LongTensor(labels)
        self._y_size = 1
        self._n_classes = 10
        train_spurious = self._make_spurious_values(train_labels[:train_split_count], train_spurious_correlation, spurious_seed)
        val_spurious = self._make_spurious_values(train_labels[train_split_count:], eval_spurious_correlation, spurious_seed + 1)
        test_spurious = self._make_spurious_values(test_labels, eval_spurious_correlation, spurious_seed + 2)
        spurious = np.concatenate([train_spurious, val_spurious, test_spurious])
        self._metadata_array = torch.stack((torch.LongTensor(spurious), self._y_array), dim=1)
        metadata_digest = hashlib.sha256(self._metadata_array.numpy().tobytes()).hexdigest()[:16]
        print(
            f"[INFO] SpurCIFAR10 resolved root={self.root_dir} "
            f"metadata_sha256={metadata_digest} spurious_seed={spurious_seed}",
            flush=True,
        )
        self._metadata_fields = ["line_color", "y"]
        self._metadata_map = {
            "line_color": [f"{class_name}_color" for class_name in CIFAR10_CLASSES],
            "y": CIFAR10_CLASSES,
        }
        self._source_is_train = np.concatenate(
            [np.ones(train_count, dtype=bool), np.zeros(len(test_labels), dtype=bool)]
        )
        self._source_indices = np.concatenate([np.arange(train_count), np.arange(len(test_labels))])
        self._split_scheme = split_scheme
        if self._split_scheme != "official":
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=["line_color", "y"])
        super().__init__(root_dir, split_scheme)

    @staticmethod
    def _find_cifar_root(root_dir: Path) -> Path:
        try:
            return resolve_dataset_root(root_dir, "spur_cifar10", ["cifar-10-batches-py"])
        except FileNotFoundError:
            pass

        try:
            return resolve_dataset_root(root_dir, "cifar10", ["cifar-10-batches-py"])
        except FileNotFoundError:
            pass

        return root_dir

    @staticmethod
    def _make_spurious_values(labels: np.ndarray, correlation: float, seed: int) -> np.ndarray:
        if not 0 <= correlation <= 1:
            raise ValueError("Spurious correlation must be in the interval [0, 1].")
        rng = np.random.RandomState(seed)
        matches = rng.rand(len(labels)) < correlation
        alternative_offsets = rng.randint(1, len(CIFAR10_CLASSES), size=len(labels))
        alternative_colors = (labels + alternative_offsets) % len(CIFAR10_CLASSES)
        return np.where(matches, labels, alternative_colors).astype(np.int64)

    def get_input(self, idx: int):
        source_idx = int(self._source_indices[idx])
        if self._source_is_train[idx]:
            image, _ = self.train_dataset[source_idx]
        else:
            image, _ = self.test_dataset[source_idx]
        image = image.convert("RGB")
        line_color = int(self.metadata_array[idx, 0].item())
        draw = ImageDraw.Draw(image)
        width, height = image.size
        half_width = max(1, self.line_width) // 2
        center_y = height // 2
        y0 = max(0, center_y - half_width)
        y1 = min(height - 1, y0 + max(1, self.line_width) - 1)
        draw.rectangle([0, y0, width - 1, y1], fill=self.line_colors[line_color])
        return image

    def eval(self, y_pred: torch.Tensor, y_true: torch.Tensor, metadata: torch.Tensor):
        metrics = compute_group_metrics(y_pred, y_true, metadata)
        lines = [f"Average acc: {metrics.average:.3f}"]
        for idx, (acc, count) in enumerate(zip(metrics.group_accuracy, metrics.group_counts)):
            if count > 0:
                lines.append(f"  group {idx} [n = {count:6.0f}]:\tacc = {acc:5.3f}")
        lines.append(f"Worst-group acc: {metrics.worst_group:.3f}")
        lines.append(f"Best-group  acc: {metrics.best_group:.3f}")
        return metrics.as_spurssl_dict(), "\n".join(lines)


def make_spur_cifar10_loaders(
    config: SpurCIFAR10Config,
    batch_size: int,
    num_workers: int | None = None,
    train_loader_kwargs: dict | None = None,
    eval_loader_kwargs: dict | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_loader_kwargs = train_loader_kwargs or {}
    eval_loader_kwargs = eval_loader_kwargs or {}
    if num_workers is not None:
        train_loader_kwargs = {"num_workers": num_workers, "pin_memory": True, **train_loader_kwargs}
        eval_loader_kwargs = {"num_workers": num_workers, "pin_memory": True, **eval_loader_kwargs}
    _, _, linear_train_transform, eval_transform = spur_cifar10_transforms(config.image_size)
    full_dataset = SpurCIFAR10Dataset(
        config.root_dir,
        val_fraction=config.val_fraction,
        train_spurious_correlation=config.train_spurious_correlation,
        eval_spurious_correlation=config.eval_spurious_correlation,
        spurious_seed=config.spurious_seed,
        line_width=config.line_width,
        download=config.download,
    )
    train_dataset = full_dataset.get_subset(config.train_split, transform=linear_train_transform)
    eval_dataset = full_dataset.get_subset(config.eval_split, transform=eval_transform)
    train_loader = get_train_loader("standard", train_dataset, batch_size=batch_size, drop_last=False, **train_loader_kwargs)
    eval_loader = get_eval_loader("standard", eval_dataset, batch_size=batch_size, drop_last=False, **eval_loader_kwargs)
    return train_loader, eval_loader


def make_spur_cifar10_ssl_loader(
    config: SpurCIFAR10Config,
    batch_size: int,
    num_workers: int | None = None,
    concept_scorer=None,
    splice_mode: str = "none",
    splice_score_threshold: float | None = None,
    splice_score_quantile: float = 0.75,
    splice_routing_mode: str = "semantic",
    splice_routing_seed: int = 0,
    **loader_kwargs,
) -> torch.utils.data.DataLoader:
    if num_workers is not None:
        loader_kwargs = {"num_workers": num_workers, "pin_memory": True, **loader_kwargs}
    ssl_train_transform, strong_ssl_train_transform, _, _ = spur_cifar10_transforms(
        config.image_size,
        ssl_crop_min=config.ssl_crop_min,
        strong_config=config,
    )
    full_dataset = SpurCIFAR10Dataset(
        config.root_dir,
        val_fraction=config.val_fraction,
        train_spurious_correlation=config.train_spurious_correlation,
        eval_spurious_correlation=config.eval_spurious_correlation,
        spurious_seed=config.spurious_seed,
        line_width=config.line_width,
        download=config.download,
    )
    if splice_mode in {"augment", "corr_reg", "augment_corr_reg", "synthesis_distill"}:
        if concept_scorer is None:
            raise ValueError("SpLiCE modes require a SpLiCE concept scorer.")
        score_subset = full_dataset.get_subset("train", transform=None)
        cache_key = dataset_score_cache_key("spur_cifar10", full_dataset, "train")
        if splice_mode == "synthesis_distill":
            concept_weights = concept_scorer.synthesis_targets_dataset(
                score_subset,
                cache_key=cache_key,
                spurious_metadata_index=0,
            )
            scores = torch.zeros(len(score_subset))
        else:
            concept_weights = concept_scorer.concept_weights_dataset(score_subset, cache_key=cache_key)
            scores = concept_scorer.reduce_selected_weights(concept_weights)
        uses_augmentation = splice_mode in {"augment", "augment_corr_reg"}
        uses_regularizer = splice_mode in {"corr_reg", "augment_corr_reg", "synthesis_distill"}
        if uses_augmentation:
            routing_scores, resolved_threshold, semantic_threshold = build_augmentation_routing(
                scores,
                splice_score_threshold,
                splice_score_quantile,
                mode=splice_routing_mode,
                seed=splice_routing_seed,
            )
        else:
            routing_scores, resolved_threshold, semantic_threshold = scores, float("inf"), None
        train_dataset = ConceptAwareSSLSubset(
            score_subset,
            routing_scores,
            ConceptAwareTwoCropTransform(
                ssl_train_transform,
                strong_ssl_train_transform if uses_augmentation else ssl_train_transform,
                resolved_threshold,
            ),
            concept_weights=concept_weights if uses_regularizer else None,
            routing_mode=splice_routing_mode if uses_augmentation else "disabled",
            semantic_threshold=semantic_threshold,
        )
    else:
        train_dataset = full_dataset.get_subset("train", transform=TwoCropTransform(ssl_train_transform))
    return get_ssl_train_loader(
        "standard",
        train_dataset,
        batch_size=batch_size,
        uniform_over_groups=False,
        grouper=full_dataset._eval_grouper,
        drop_last=False,
        **loader_kwargs,
    )


def make_spur_cifar10_rank_loader(
    config: SpurCIFAR10Config,
    batch_size: int,
    num_workers: int | None = None,
    **loader_kwargs,
) -> torch.utils.data.DataLoader:
    """Build an ordered, non-augmented train loader for diagnostics only."""

    if num_workers is not None:
        loader_kwargs = {"num_workers": num_workers, "pin_memory": True, **loader_kwargs}
    _, _, _, eval_transform = spur_cifar10_transforms(config.image_size)
    full_dataset = SpurCIFAR10Dataset(
        config.root_dir,
        val_fraction=config.val_fraction,
        train_spurious_correlation=config.train_spurious_correlation,
        eval_spurious_correlation=config.eval_spurious_correlation,
        spurious_seed=config.spurious_seed,
        line_width=config.line_width,
        download=config.download,
    )
    rank_dataset = full_dataset.get_subset("train", transform=eval_transform)
    return get_eval_loader(
        "standard",
        rank_dataset,
        batch_size=batch_size,
        drop_last=False,
        **loader_kwargs,
    )


SPUR_CIFAR10_SPEC = {
    "dataset": SpurCIFAR10Dataset,
    "config": SpurCIFAR10Config,
    "ssl_loader": make_spur_cifar10_ssl_loader,
    "rank_loader": make_spur_cifar10_rank_loader,
    "probe_loaders": make_spur_cifar10_loaders,
    "num_classes": 10,
    "spurious_metadata_index": 0,
    "target_metadata_index": 1,
}
