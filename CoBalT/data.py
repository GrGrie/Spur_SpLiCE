from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from CoBalT.augment import (
    RecordedViewTransform,
    TwoRecordedViews,
    classifier_train_transform,
    evaluation_transform,
)
from experiments.spurious_eval.datasets.celeba import CelebADataset
from experiments.spurious_eval.datasets.waterbirds import WaterbirdsDataset


def full_dataset(name: str, root: str):
    canonical = name.lower()
    if canonical == "waterbirds":
        return WaterbirdsDataset(root)
    if canonical == "celeba":
        return CelebADataset(root)
    raise ValueError(f"Unsupported CoBalT dataset {name!r}.")


class IndexedSubset(Dataset):
    def __init__(self, subset) -> None:
        self.subset = subset
        self.indices = torch.as_tensor(subset.indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        image, label, metadata = self.subset[index]
        return image, label, metadata, self.indices[index]


def discovery_loader(
    dataset_name: str,
    root: str,
    batch_size: int,
    workers: int,
    image_size: int = 224,
    crop_min: float = 0.2,
    shuffle: bool = True,
) -> DataLoader:
    dataset = full_dataset(dataset_name, root)
    transform = TwoRecordedViews(RecordedViewTransform(image_size, crop_min))
    subset = IndexedSubset(dataset.get_subset("train", transform=transform))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=workers > 0,
    )


def split_loader(
    dataset_name: str,
    root: str,
    split: str,
    batch_size: int,
    workers: int,
    image_size: int = 224,
    training: bool = False,
    sampler=None,
) -> DataLoader:
    dataset = full_dataset(dataset_name, root)
    transform = (
        classifier_train_transform(image_size) if training else evaluation_transform(image_size)
    )
    subset = IndexedSubset(dataset.get_subset(split, transform=transform))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=workers > 0,
    )


def split_ids(dataset_name: str, root: str, split: str) -> torch.Tensor:
    dataset = full_dataset(dataset_name, root)
    return torch.as_tensor(dataset.get_subset(split).indices, dtype=torch.long)


def validate_concept_artifact(
    artifact: dict, dataset_name: str, root: str, required_splits: tuple[str, ...]
) -> None:
    if artifact.get("artifact") != "cobalt_concepts_v1":
        raise ValueError("Not a CoBalT concept artifact.")
    if artifact.get("dataset") != dataset_name.lower():
        raise ValueError("Concept artifact belongs to a different dataset.")
    for split in required_splits:
        if split not in artifact.get("splits", {}):
            raise ValueError(f"Concept artifact is missing split {split!r}.")
        record = artifact["splits"][split]
        expected_ids = split_ids(dataset_name, root, split)
        ids = record["sample_ids"].long()
        concepts = record["concepts"].long()
        if not torch.equal(ids, expected_ids):
            raise ValueError(f"Concept sample order does not match current {split} split.")
        if concepts.ndim != 2 or concepts.shape[0] != ids.numel():
            raise ValueError(f"Invalid concept tensor for split {split!r}.")
        if "confidence" in record:
            confidence = torch.as_tensor(record["confidence"], dtype=torch.float32).view(-1)
            if confidence.shape != (ids.numel(),) or not torch.isfinite(confidence).all():
                raise ValueError(f"Invalid confidence tensor for split {split!r}.")
            if torch.any((confidence < 0) | (confidence > 1)):
                raise ValueError(f"CoBalT confidence must lie in [0, 1] for split {split!r}.")
