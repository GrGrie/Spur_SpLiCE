"""What every dataset in this project has in common.

A dataset here pairs a target label ``y`` with a spurious attribute ``a`` and splits its samples
into train, validation and test. Adapters differ only in where that metadata comes from and how one
image is produced. The transforms, the group evaluation, the image size a model must match and the
four loader roles are shared, so a new dataset is a metadata reader plus a handful of attributes.

Roles name what a loader is for, so no caller assembles a split, a transform and a sampler by hand:

``ssl``          two augmented crops of the train split, shuffled, for contrastive training
``rank``         the train split in order without augmentation, for observational diagnostics
``probe_train``  the configured probe training split, augmented and shuffled
``probe_eval``   the configured evaluation split in order without augmentation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torchvision import transforms

from experiments.spurious_eval.datasets.augmentation import build_ssl_transform
from experiments.spurious_eval.datasets.transforms import TwoCropTransform
from experiments.spurious_eval.datasets.wilds_compat import (
    WILDSDataset,
    get_eval_loader,
    get_ssl_train_loader,
    get_train_loader,
)
from experiments.spurious_eval.metrics import compute_group_metrics

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
#: The image size above which a model may use the 224-pixel ResNet stem.
LARGE_INPUT_SIZE = 224
LOADER_ROLES = ("ssl", "rank", "probe_train", "probe_eval")

#: Registered adapters, keyed by canonical name and by every historical spelling.
DATASETS: dict[str, type["SpuriousDataset"]] = {}
CANONICAL_DATASETS: dict[str, type["SpuriousDataset"]] = {}


@dataclass(frozen=True)
class DatasetConfig:
    """The options every adapter accepts. A dataset adds its own fields by subclassing."""

    root_dir: str = "./datasets"
    image_size: int = LARGE_INPUT_SIZE
    train_split: str = "ds_train"
    eval_split: str = "val"
    ssl_crop_min: float = 0.2


class SpuriousDataset(WILDSDataset):
    """A dataset whose metadata carries the spurious attribute next to the target label.

    A subclass reads its metadata in ``__init__``, returns an image from ``get_input`` and declares
    the attributes below. Everything else, including the group report and the loaders, is inherited.
    """

    #: Canonical name. Sample IDs embed it, so every producer and consumer must agree on it.
    name: ClassVar[str] = ""
    #: Historical spellings accepted at the command line and resolved to ``name``.
    aliases: ClassVar[tuple[str, ...]] = ()
    num_classes: ClassVar[int] = 0
    image_size: ClassVar[int] = LARGE_INPUT_SIZE
    mean: ClassVar[tuple[float, float, float]] = IMAGENET_MEAN
    std: ClassVar[tuple[float, float, float]] = IMAGENET_STD
    #: The configuration class this adapter is built from.
    Config: ClassVar[type[DatasetConfig]] = DatasetConfig
    #: Columns of ``metadata_array``: the spurious attribute first, the target second.
    spurious_metadata_index: ClassVar[int] = 0
    target_metadata_index: ClassVar[int] = 1

    @classmethod
    def from_config(cls, config: DatasetConfig) -> "SpuriousDataset":
        """Build the full dataset a configuration describes, before any split is taken."""

        return cls(config.root_dir)

    @classmethod
    def default_model(cls) -> str:
        """The encoder these images are meant for."""

        return "resnet18_large" if cls.image_size >= LARGE_INPUT_SIZE else "resnet18"

    @classmethod
    def model_error(cls, model: str) -> str | None:
        """Why a model cannot train on these images, or None when it can."""

        if cls.image_size >= LARGE_INPUT_SIZE:
            return None
        if not (model.endswith("_large") or model == "resnet50_pretrained"):
            return None
        return (
            f"{cls.name} uses {cls.image_size}x{cls.image_size} images; "
            "choose --model resnet18 or --model resnet50."
        )

    @property
    def grouper(self):
        """The grouper that turns metadata into the (attribute, label) groups results report on."""

        return self._eval_grouper

    def eval(self, y_pred: torch.Tensor, y_true: torch.Tensor, metadata: torch.Tensor):
        """Average, per-group, worst-group and best-group accuracy, with a printable report."""

        metrics = compute_group_metrics(y_pred, y_true, metadata)
        lines = [f"Average acc: {metrics.average:.3f}"]
        for idx, (acc, count) in enumerate(zip(metrics.group_accuracy, metrics.group_counts)):
            if count > 0:
                lines.append(f"  group {idx} [n = {count:6.0f}]:\tacc = {acc:5.3f}")
        lines.append(f"Worst-group acc: {metrics.worst_group:.3f}")
        lines.append(f"Best-group  acc: {metrics.best_group:.3f}")
        return metrics.as_spurssl_dict(), "\n".join(lines)


def register_dataset(cls: type[SpuriousDataset]) -> type[SpuriousDataset]:
    """Register an adapter under its canonical name and its historical spellings."""

    if not cls.name:
        raise ValueError(f"{cls.__name__} must declare a canonical name.")
    spellings = (cls.name, *cls.aliases)
    for spelling in spellings:
        registered = DATASETS.get(spelling)
        if registered is not None and registered is not cls:
            raise ValueError(f"Datasets {cls.name!r} and {registered.name!r} both claim {spelling!r}.")
    cls._dataset_name = cls.name
    DATASETS.update(dict.fromkeys(spellings, cls))
    CANONICAL_DATASETS[cls.name] = cls
    return cls


def dataset_transforms(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    ssl_crop_min: float = 0.2,
) -> tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """The SSL, probe-training and evaluation transforms, in that order."""

    normalize = transforms.Normalize(mean=mean, std=std)
    ssl_transform = build_ssl_transform(
        image_size=image_size, crop_min=ssl_crop_min,
        color_jitter=(0.4, 0.4, 0.4, 0.1), color_jitter_p=0.8,
        grayscale_p=0.2, normalize=normalize,
    )
    probe_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return ssl_transform, probe_train_transform, eval_transform


def build_loader(
    dataset_class: type[SpuriousDataset],
    role: str,
    config: DatasetConfig,
    batch_size: int,
    *,
    dataset: SpuriousDataset | None = None,
    **loader_kwargs,
):
    """The loader one role needs. Pass ``dataset`` to reuse an already built adapter."""

    if role not in LOADER_ROLES:
        raise ValueError(f"Unsupported loader role {role!r}. Roles: {list(LOADER_ROLES)}")
    ssl_transform, probe_train_transform, eval_transform = dataset_transforms(
        config.image_size, dataset_class.mean, dataset_class.std, ssl_crop_min=config.ssl_crop_min,
    )
    dataset = dataset_class.from_config(config) if dataset is None else dataset
    if role == "ssl":
        subset = dataset.get_subset("train", transform=TwoCropTransform(ssl_transform))
        return get_ssl_train_loader(
            "standard", subset, batch_size=batch_size, uniform_over_groups=False,
            grouper=dataset.grouper, drop_last=False, **loader_kwargs,
        )
    if role == "probe_train":
        subset = dataset.get_subset(config.train_split, transform=probe_train_transform)
        return get_train_loader("standard", subset, batch_size=batch_size, drop_last=False, **loader_kwargs)
    split = "train" if role == "rank" else config.eval_split
    subset = dataset.get_subset(split, transform=eval_transform)
    return get_eval_loader("standard", subset, batch_size=batch_size, drop_last=False, **loader_kwargs)


def build_probe_loaders(
    dataset_class: type[SpuriousDataset],
    config: DatasetConfig,
    batch_size: int,
    train_loader_kwargs: dict | None = None,
    eval_loader_kwargs: dict | None = None,
):
    """The probe's training and evaluation loaders over one instance of the dataset."""

    dataset = dataset_class.from_config(config)
    return (
        build_loader(dataset_class, "probe_train", config, batch_size, dataset=dataset,
                     **(train_loader_kwargs or {})),
        build_loader(dataset_class, "probe_eval", config, batch_size, dataset=dataset,
                     **(eval_loader_kwargs or {})),
    )
