from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from experiments.spurious_eval.metrics import compute_group_metrics
from experiments.spurious_eval.datasets.transforms import ConceptAwareTwoCropTransform, TwoCropTransform
from experiments.spurious_eval.datasets.wilds_compat import (
    CombinatorialGrouper,
    WILDSDataset,
    get_eval_loader,
    get_ssl_train_loader,
    get_train_loader,
)


WATERBIRDS_MEAN = (0.485, 0.456, 0.406)
WATERBIRDS_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class WaterbirdsConfig:
    root_dir: str = "./datasets"
    image_size: int = 224
    train_split: str = "ds_train"
    eval_split: str = "val"
    ssl_crop_min: float = 0.2


def waterbirds_transforms(
    image_size: int = 224,
    ssl_crop_min: float = 0.2,
) -> tuple[transforms.Compose, transforms.Compose, transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=WATERBIRDS_MEAN, std=WATERBIRDS_STD)
    ssl_train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=image_size, scale=(ssl_crop_min, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    strong_ssl_train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                ],
                p=0.9,
            ),
            transforms.RandomGrayscale(p=0.3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
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


class ConceptAwareSSLSubset(torch.utils.data.Dataset):
    """Wrap a WILDS subset and attach precomputed SpLiCE scores for SSL training."""

    def __init__(self, subset: torch.utils.data.Dataset, scores: torch.Tensor, transform: ConceptAwareTwoCropTransform) -> None:
        if len(subset) != len(scores):
            raise ValueError(f"Expected one SpLiCE score per sample, got {len(scores)} scores for {len(subset)} samples.")
        self.subset = subset
        self.scores = scores.float()
        self.transform = transform
        self.og_group_counts = getattr(subset, "og_group_counts", None)

    def __len__(self) -> int:
        return len(self.subset)

    @property
    def collate(self):
        return getattr(self.subset, "collate", None)

    @property
    def metadata_array(self) -> torch.Tensor:
        return self.subset.metadata_array

    def __getitem__(self, idx: int):
        image, label, metadata = self.subset[idx]
        score = self.scores[idx]
        views = self.transform(image, float(score.item()))
        return views, label, metadata, score

    def eval(self, y_pred: torch.Tensor, y_true: torch.Tensor, metadata: torch.Tensor):
        return self.subset.eval(y_pred, y_true, metadata)


class WaterbirdsDataset(WILDSDataset):
    """Waterbirds with WILDS-style metadata and SpurSSL-compatible splits."""

    _dataset_name = "waterbirds"

    def __init__(self, root_dir: str = "./datasets", split_scheme: str = "official") -> None:
        self.root_dir = Path(root_dir)
        self._data_dir = self._find_data_dir(self.root_dir)
        metadata_path = Path(self.data_dir) / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Waterbirds metadata not found at {metadata_path}")

        metadata_df = pd.read_csv(metadata_path)
        required_columns = {"img_filename", "y", "place", "split"}
        missing = required_columns.difference(metadata_df.columns)
        if missing:
            raise ValueError(f"Waterbirds metadata is missing columns: {sorted(missing)}")

        self.metadata_df = metadata_df.reset_index(drop=True)
        self._y_array = torch.LongTensor(self.metadata_df["y"].values)
        self._y_size = 1
        self._n_classes = 2
        self._metadata_array = torch.stack(
            (
                torch.LongTensor(self.metadata_df["place"].values),
                self._y_array,
            ),
            dim=1,
        )
        self._metadata_fields = ["background", "y"]
        self._metadata_map = {
            "background": [" land", "water"],
            "y": [" landbird", "waterbird"],
        }
        self._input_array = self.metadata_df["img_filename"].values
        self._split_scheme = split_scheme
        if self._split_scheme != "official":
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")
        self._split_array = self.metadata_df["split"].values
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=["background", "y"])
        super().__init__(root_dir, split_scheme)

    @staticmethod
    def _find_data_dir(root_dir: Path) -> Path:
        candidates = [
            root_dir,
            root_dir / "waterbirds",
            root_dir / "waterbird_complete95_forest2water2",
        ]
        for candidate in candidates:
            if (candidate / "metadata.csv").exists():
                return candidate
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find Waterbirds metadata.csv. Searched: {searched}")

    def get_input(self, idx: int):
        image_path = Path(self.data_dir) / self._input_array[idx]
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


def make_waterbirds_loaders(
    config: WaterbirdsConfig,
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
    _, _, linear_train_transform, eval_transform = waterbirds_transforms(config.image_size)
    full_dataset = WaterbirdsDataset(config.root_dir)
    train_dataset = full_dataset.get_subset(config.train_split, transform=linear_train_transform)
    eval_dataset = full_dataset.get_subset(config.eval_split, transform=eval_transform)

    train_loader = get_train_loader(
        "standard",
        train_dataset,
        batch_size=batch_size,
        drop_last=False,
        **train_loader_kwargs,
    )
    eval_loader = get_eval_loader(
        "standard",
        eval_dataset,
        batch_size=batch_size,
        drop_last=False,
        **eval_loader_kwargs,
    )
    return train_loader, eval_loader


def make_waterbirds_ssl_loader(
    config: WaterbirdsConfig,
    batch_size: int,
    num_workers: int | None = None,
    concept_scorer=None,
    splice_mode: str = "none",
    splice_score_threshold: float = 0.0,
    **loader_kwargs,
) -> torch.utils.data.DataLoader:
    if num_workers is not None:
        loader_kwargs = {"num_workers": num_workers, "pin_memory": True, **loader_kwargs}
    ssl_train_transform, strong_ssl_train_transform, _, _ = waterbirds_transforms(
        config.image_size,
        ssl_crop_min=config.ssl_crop_min,
    )
    full_dataset = WaterbirdsDataset(config.root_dir)
    if splice_mode in {"augment", "augment_corr_reg"}:
        if concept_scorer is None:
            raise ValueError("Concept-aware augmentation requires a SpLiCE concept scorer.")
        score_subset = full_dataset.get_subset("train", transform=None)
        scores = concept_scorer.score_dataset(score_subset)
        train_dataset = ConceptAwareSSLSubset(
            score_subset,
            scores,
            ConceptAwareTwoCropTransform(ssl_train_transform, strong_ssl_train_transform, splice_score_threshold),
        )
    elif splice_mode == "corr_reg":
        if concept_scorer is None:
            raise ValueError("SpLiCE correlation regularization requires a SpLiCE concept scorer.")
        score_subset = full_dataset.get_subset("train", transform=None)
        scores = concept_scorer.score_dataset(score_subset)
        train_dataset = ConceptAwareSSLSubset(
            score_subset,
            scores,
            ConceptAwareTwoCropTransform(ssl_train_transform, ssl_train_transform, float("inf")),
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


WATERBIRDS_SPEC = {
    "dataset": WaterbirdsDataset,
    "config": WaterbirdsConfig,
    "ssl_loader": make_waterbirds_ssl_loader,
    "probe_loaders": make_waterbirds_loaders,
    "num_classes": 2,
    "spurious_metadata_index": 0,
    "target_metadata_index": 1,
}
