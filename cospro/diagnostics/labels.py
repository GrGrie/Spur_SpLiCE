"""Sole reader of hidden labels for post-hoc diagnostics.

Artifacts identify images as ``<dataset>:<index>``, where the index is the row of the dataset
metadata. This module maps such sample IDs to the class label ``y`` and the spurious attribute ``a``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.spurious_eval.datasets.registry import canonical_dataset_name, get_dataset_spec


@dataclass(frozen=True)
class SampleLabels:
    """Class labels and spurious attributes indexed by dataset metadata row."""

    dataset: str
    y: np.ndarray
    a: np.ndarray
    class_names: list[str] = field(default_factory=list)
    attribute_names: list[str] = field(default_factory=list)

    @property
    def n_attributes(self) -> int:
        return int(self.a.max()) + 1

    def group_names(self) -> list[str]:
        """Names of the joint (y, a) groups in the order ``y * n_attributes + a``."""

        n_classes = int(self.y.max()) + 1
        names = []
        for label in range(n_classes):
            for attribute in range(self.n_attributes):
                class_name = self.class_names[label] if label < len(self.class_names) else f"y={label}"
                attribute_name = (
                    self.attribute_names[attribute] if attribute < len(self.attribute_names) else f"a={attribute}"
                )
                names.append(f"{class_name} / {attribute_name}")
        return names

    def for_ids(self, sample_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Labels of ``sample_ids`` in the given order."""

        rows = np.empty(len(sample_ids), dtype=np.int64)
        for position, sample_id in enumerate(sample_ids):
            dataset, _, index = str(sample_id).partition(":")
            if canonical_dataset_name(dataset) != self.dataset or not index.isdigit():
                raise ValueError(f"Sample ID {sample_id!r} does not belong to dataset {self.dataset!r}.")
            rows[position] = int(index)
        if rows.size and rows.max() >= len(self.y):
            raise ValueError("A sample ID points past the end of the dataset metadata.")
        return self.y[rows], self.a[rows]


def load_labels(dataset: str, data_folder: str | Path) -> SampleLabels:
    """Read ``y``, ``a`` and their display names for every metadata row of a registered dataset."""

    name = canonical_dataset_name(dataset)
    spec = get_dataset_spec(name)
    source = spec["dataset"](str(data_folder))
    metadata = np.asarray(source.metadata_array)
    fields = list(getattr(source, "_metadata_fields", []))
    name_map = getattr(source, "_metadata_map", None) or {}

    def names(index: int) -> list[str]:
        if index >= len(fields):
            return []
        return [str(value).strip() for value in name_map.get(fields[index], [])]

    return SampleLabels(
        dataset=name,
        y=metadata[:, spec["target_metadata_index"]].astype(np.int64),
        a=metadata[:, spec["spurious_metadata_index"]].astype(np.int64),
        class_names=names(spec["target_metadata_index"]),
        attribute_names=names(spec["spurious_metadata_index"]),
    )
