from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

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


CELEBA_MEAN = (0.485, 0.456, 0.406)
CELEBA_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class CelebAConfig(StrongAugmentationConfig):
    root_dir: str = "./datasets"
    image_size: int = 224
    train_split: str = "ds_train"
    eval_split: str = "val"
    ssl_crop_min: float = 0.2


def celeba_transforms(
    image_size: int = 224,
    ssl_crop_min: float = 0.2,
    strong_config: CelebAConfig | None = None,
) -> tuple[transforms.Compose, transforms.Compose, transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=CELEBA_MEAN, std=CELEBA_STD)
    strong_config = strong_config or CelebAConfig(image_size=image_size, ssl_crop_min=ssl_crop_min)
    ssl_train_transform, strong_ssl_train_transform = build_standard_and_strong_ssl_transforms(
        image_size=image_size,
        ssl_crop_min=ssl_crop_min,
        normalize=normalize,
        strong_config=strong_config,
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


class CelebADataset(WILDSDataset):
    """CelebA with Blond_Hair as target and Male as the spurious attribute.

    This matches the CelebA protocol used by SpurSSL, LateTVG, and the common
    group-robustness benchmark: predict hair colour while measuring groups
    formed by gender and the target label.
    """

    _dataset_name = "celeba"

    def __init__(self, root_dir: str = "./datasets", split_scheme: str = "official") -> None:
        self.root_dir = Path(root_dir)
        self._data_dir = self._find_data_dir(self.root_dir)
        attrs_path = self._data_dir / "list_attr_celeba.csv"
        if not attrs_path.exists():
            raise FileNotFoundError(f"CelebA attributes not found at {attrs_path}")
        attrs = pd.read_csv(attrs_path)
        required_columns = {"image_id", "Male", "Blond_Hair"}
        if not required_columns.issubset(attrs.columns):
            if attrs.columns[0] != "image_id":
                attrs = attrs.rename(columns={attrs.columns[0]: "image_id"})
        missing = required_columns.difference(attrs.columns)
        if missing:
            raise ValueError(f"CelebA attributes are missing columns: {sorted(missing)}")

        self.attrs = attrs.reset_index(drop=True)
        self._input_array = self.attrs["image_id"].astype(str).values
        self._y_array = torch.LongTensor(
            (self.attrs["Blond_Hair"].astype(int).values == 1).astype(np.int64)
        )
        male = (self.attrs["Male"].astype(int).values == 1).astype(np.int64)
        self._y_size = 1
        self._n_classes = 2
        self._metadata_array = torch.stack((torch.LongTensor(male), self._y_array), dim=1)
        self._metadata_fields = ["gender", "y"]
        self._metadata_map = {
            "gender": ["female", "male"],
            "y": ["not_blond", "blond"],
        }
        self._split_scheme = split_scheme
        if self._split_scheme != "official":
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")
        self._split_array = self._load_split_array()
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=["gender", "y"])
        super().__init__(root_dir, split_scheme)

    @staticmethod
    def _find_data_dir(root_dir: Path) -> Path:
        return resolve_dataset_root(root_dir, "celeba", ["list_attr_celeba.csv"])

    def _load_split_array(self) -> np.ndarray:
        split_path = self._data_dir / "list_eval_partition.csv"
        if split_path.exists():
            splits = pd.read_csv(split_path)
            if "partition" not in splits.columns:
                splits = splits.rename(columns={splits.columns[-1]: "partition"})
            if "image_id" not in splits.columns:
                splits = splits.rename(columns={splits.columns[0]: "image_id"})
            split_lookup = dict(zip(splits["image_id"].astype(str), splits["partition"].astype(int)))
            return np.asarray([split_lookup[str(image_id)] for image_id in self._input_array], dtype=np.int64)

        rng = np.random.RandomState(0)
        permutation = rng.permutation(len(self._input_array))
        split_array = np.zeros(len(self._input_array), dtype=np.int64)
        val_start = int(round(0.8 * len(permutation)))
        test_start = int(round(0.9 * len(permutation)))
        split_array[permutation[val_start:test_start]] = 1
        split_array[permutation[test_start:]] = 2
        return split_array

    def get_input(self, idx: int):
        image_path = self._data_dir / "img_align_celeba" / self._input_array[idx]
        if not image_path.exists():
            raise FileNotFoundError(f"CelebA image not found at {image_path}")
        return Image.open(image_path).convert("RGB")

    def eval(self, y_pred: torch.Tensor, y_true: torch.Tensor, metadata: torch.Tensor):
        metrics = compute_group_metrics(y_pred, y_true, metadata)
        lines = [f"Average acc: {metrics.average:.3f}"]
        for idx, (acc, count) in enumerate(zip(metrics.group_accuracy, metrics.group_counts)):
            if count > 0:
                lines.append(f"  group {idx} [n = {count:6.0f}]:\tacc = {acc:5.3f}")
        lines.append(f"Worst-group acc: {metrics.worst_group:.3f}")
        lines.append(f"Best-group  acc: {metrics.best_group:.3f}")
        return metrics.as_spurssl_dict(), "\n".join(lines)


def make_celeba_loaders(
    config: CelebAConfig,
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
    _, _, linear_train_transform, eval_transform = celeba_transforms(config.image_size)
    full_dataset = CelebADataset(config.root_dir)
    train_dataset = full_dataset.get_subset(config.train_split, transform=linear_train_transform)
    eval_dataset = full_dataset.get_subset(config.eval_split, transform=eval_transform)
    train_loader = get_train_loader("standard", train_dataset, batch_size=batch_size, drop_last=False, **train_loader_kwargs)
    eval_loader = get_eval_loader("standard", eval_dataset, batch_size=batch_size, drop_last=False, **eval_loader_kwargs)
    return train_loader, eval_loader


def make_celeba_ssl_loader(
    config: CelebAConfig,
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
    ssl_train_transform, strong_ssl_train_transform, _, _ = celeba_transforms(
        config.image_size,
        ssl_crop_min=config.ssl_crop_min,
        strong_config=config,
    )
    full_dataset = CelebADataset(config.root_dir)
    if splice_mode in {"augment", "corr_reg", "augment_corr_reg", "counterfactual"}:
        if concept_scorer is None:
            raise ValueError("SpLiCE modes require a SpLiCE concept scorer.")
        score_subset = full_dataset.get_subset("train", transform=None)
        cache_key = dataset_score_cache_key("celeba", full_dataset, "train")
        if splice_mode == "counterfactual":
            concept_weights = concept_scorer.counterfactual_targets_dataset(
                score_subset,
                cache_key=cache_key,
                spurious_metadata_index=0,
            )
            scores = torch.zeros(len(score_subset))
        else:
            concept_weights = concept_scorer.concept_weights_dataset(score_subset, cache_key=cache_key)
            scores = concept_scorer.reduce_selected_weights(concept_weights)
        uses_augmentation = splice_mode in {"augment", "augment_corr_reg"}
        uses_regularizer = splice_mode in {"corr_reg", "augment_corr_reg", "counterfactual"}
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


def make_celeba_rank_loader(
    config: CelebAConfig,
    batch_size: int,
    num_workers: int | None = None,
    **loader_kwargs,
) -> torch.utils.data.DataLoader:
    """Build an ordered, non-augmented train loader for diagnostics only."""

    if num_workers is not None:
        loader_kwargs = {"num_workers": num_workers, "pin_memory": True, **loader_kwargs}
    _, _, _, eval_transform = celeba_transforms(config.image_size)
    full_dataset = CelebADataset(config.root_dir)
    rank_dataset = full_dataset.get_subset("train", transform=eval_transform)
    return get_eval_loader(
        "standard",
        rank_dataset,
        batch_size=batch_size,
        drop_last=False,
        **loader_kwargs,
    )


CELEBA_SPEC = {
    "dataset": CelebADataset,
    "config": CelebAConfig,
    "ssl_loader": make_celeba_ssl_loader,
    "rank_loader": make_celeba_rank_loader,
    "probe_loaders": make_celeba_loaders,
    "num_classes": 2,
    "spurious_metadata_index": 0,
    "target_metadata_index": 1,
}
