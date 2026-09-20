"""The registered dataset adapters and how a name resolves to one.

Importing this module registers every adapter, so ``dataset_class`` and ``dataset_names`` see all
of them. Adding a dataset means adding a module here and importing it below; nothing else in the
project carries a list of dataset names. See `docs/ADDING_A_DATASET.md`.

Artifact sample IDs include the dataset name, so every producer and consumer must agree on one
spelling. Aliases stay at the command-line boundary: ``canonical_dataset_name`` resolves them
before any artifact is written.
"""

from __future__ import annotations

from experiments.spurious_eval.datasets.base import (  # noqa: F401  (re-exported)
    CANONICAL_DATASETS,
    DATASETS,
    DatasetConfig,
    SpuriousDataset,
    build_loader,
    build_probe_loaders,
)
from experiments.spurious_eval.datasets.celeba import CelebADataset  # noqa: F401  (registers celeba)
from experiments.spurious_eval.datasets.spur_cifar10 import SpurCIFAR10Dataset  # noqa: F401
from experiments.spurious_eval.datasets.waterbirds import WaterbirdsDataset  # noqa: F401


def dataset_names() -> list[str]:
    """Every canonical dataset name, sorted."""

    return sorted(CANONICAL_DATASETS)


def canonical_dataset_name(name: str) -> str:
    """The one spelling artifacts use, from any accepted spelling of ``name``."""

    for candidate in (str(name).strip(), str(name).strip().lower()):
        registered = DATASETS.get(candidate)
        if registered is not None:
            return registered.name
    raise ValueError(f"Unsupported dataset: {name}. Choices: {dataset_names()}")


def dataset_class(name: str) -> type[SpuriousDataset]:
    """The adapter a name selects, accepting historical spellings."""

    return CANONICAL_DATASETS[canonical_dataset_name(name)]
