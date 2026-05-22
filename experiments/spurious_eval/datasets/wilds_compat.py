from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler


def get_counts(groups: torch.Tensor, n_groups: int) -> torch.Tensor:
    unique_groups, unique_counts = torch.unique(groups, sorted=False, return_counts=True)
    counts = torch.zeros(n_groups, device=groups.device)
    counts[unique_groups] = unique_counts.float()
    return counts


class WILDSDataset(Dataset):
    """Small local subset of WILDS dataset behavior used by SpurSSL-style training."""

    DEFAULT_SPLITS = {"train": 0, "val": 1, "test": 2}
    DEFAULT_SPLIT_NAMES = {"train": "Train", "val": "Validation", "test": "Test"}

    def __init__(self, root_dir: str | Path, split_scheme: str) -> None:
        if len(self._metadata_array.shape) == 1:
            self._metadata_array = self._metadata_array.unsqueeze(1)
        self.check_init()

    def __len__(self) -> int:
        return len(self.y_array)

    def __getitem__(self, idx: int):
        x = self.get_input(idx)
        y = self.y_array[idx]
        metadata = self.metadata_array[idx]
        return x, y, metadata

    def get_input(self, idx: int):
        raise NotImplementedError

    def eval(self, y_pred: torch.Tensor, y_true: torch.Tensor, metadata: torch.Tensor):
        raise NotImplementedError

    def get_subset(self, split: str, frac: float = 1.0, transform=None, ordering=None) -> "WILDSSubset":
        og_group_counts = None
        if "_train" in split:
            split_mask = self.split_array == self.split_dict["train"]
            groups, group_counts = self._eval_grouper.metadata_to_group(
                self.metadata_array[split_mask.ravel()],
                return_counts=True,
            )
            og_group_counts = group_counts
            n_groups = torch.count_nonzero(group_counts, dim=0).item()
            if split == "ds_train":
                num_to_retain = [min(int(torch.sum(groups == group)) for group in range(n_groups))] * n_groups
            elif split == "us_train":
                num_to_retain = [max(int(torch.sum(groups == group)) for group in range(n_groups))] * n_groups
            elif split == "balanced_train":
                split_mask_val = self.split_array == self.split_dict["val"]
                _, group_counts_val = self._eval_grouper.metadata_to_group(
                    self.metadata_array[split_mask_val.ravel()],
                    return_counts=True,
                )
                val_portions = np.array(group_counts_val) / min(group_counts_val)
                min_group = min(group_counts)
                num_to_retain = [int(value) for value in (val_portions * min_group).tolist()]
            else:
                raise ValueError(f"Split {split} not found in dataset's split_dict.")

            indices: list[int] = []
            split_idx = np.where(split_mask)[0]
            for group in range(n_groups):
                group_idx = split_idx[np.where(groups == group)[0]]
                if num_to_retain[group] <= len(group_idx):
                    indices.extend(group_idx.tolist()[: num_to_retain[group]])
                else:
                    indices.extend(np.random.choice(group_idx.tolist(), num_to_retain[group]).tolist())
            split_idx = np.sort(indices)
        elif split == "ds_test":
            split_idx, og_group_counts = self._downsample_split_by_group("test")
        else:
            if split not in self.split_dict:
                raise ValueError(f"Split {split} not found in dataset's split_dict.")
            split_mask = self.split_array == self.split_dict[split]
            if ordering is None:
                split_idx = np.where(split_mask)[0]
            else:
                groups, group_counts = self._eval_grouper.metadata_to_group(
                    self.metadata_array[split_mask.ravel()],
                    return_counts=True,
                )
                if len(group_counts) != len(ordering):
                    raise ValueError("ordering must include one entry per group")
                raw_split_idx = np.where(split_mask)[0]
                split_idx = np.concatenate([raw_split_idx[np.where(groups == group)[0]] for group in ordering])

        if frac < 1.0:
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])

        subset = WILDSSubset(self, split_idx, transform)
        subset.og_group_counts = og_group_counts
        return subset

    def _downsample_split_by_group(self, split: str) -> tuple[np.ndarray, torch.Tensor]:
        split_mask = self.split_array == self.split_dict[split]
        groups, group_counts = self._eval_grouper.metadata_to_group(
            self.metadata_array[split_mask.ravel()],
            return_counts=True,
        )
        n_groups = len(group_counts)
        num_to_retain = [min(int(torch.sum(groups == group)) for group in range(n_groups))] * n_groups
        indices: list[int] = []
        split_idx = np.where(split_mask)[0]
        for group in range(n_groups):
            group_idx = split_idx[np.where(groups == group)[0]]
            if num_to_retain[group] <= len(group_idx):
                indices.extend(group_idx.tolist()[: num_to_retain[group]])
            else:
                indices.extend(np.random.choice(group_idx.tolist(), num_to_retain[group]).tolist())
        return np.sort(indices), group_counts

    def check_init(self) -> None:
        required_attrs = [
            "_dataset_name",
            "_data_dir",
            "_split_scheme",
            "_split_array",
            "_y_array",
            "_y_size",
            "_metadata_fields",
            "_metadata_array",
        ]
        for attr_name in required_attrs:
            if not hasattr(self, attr_name):
                raise AssertionError(f"WILDSDataset is missing {attr_name}.")
        if not os.path.exists(self.data_dir):
            raise ValueError(f"{self.data_dir} does not exist yet.")
        if self.split_dict.keys() != self.split_names.keys():
            raise AssertionError("split_dict and split_names must have the same keys")
        if "train" not in self.split_dict or "val" not in self.split_dict:
            raise AssertionError("WILDSDataset requires train and val splits")
        if not isinstance(self.metadata_array, torch.Tensor):
            raise AssertionError("metadata_array must be a torch.Tensor")
        if len(self.y_array) != len(self.metadata_array):
            raise AssertionError("y_array and metadata_array lengths must match")
        if len(self.split_array) != len(self.metadata_array):
            raise AssertionError("split_array and metadata_array lengths must match")
        if len(self.metadata_array.shape) != 2:
            raise AssertionError("metadata_array must be two-dimensional")
        if len(self.metadata_fields) != self.metadata_array.shape[1]:
            raise AssertionError("metadata_fields must match metadata_array columns")
        if self.y_size == 1 and "y" not in self.metadata_fields:
            raise AssertionError("metadata_fields must include y when y_size is 1")

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def data_dir(self) -> str:
        return str(self._data_dir)

    @property
    def collate(self):
        return getattr(self, "_collate", None)

    @property
    def split_scheme(self) -> str:
        return self._split_scheme

    @property
    def split_dict(self) -> dict[str, int]:
        return getattr(self, "_split_dict", WILDSDataset.DEFAULT_SPLITS)

    @property
    def split_names(self) -> dict[str, str]:
        return getattr(self, "_split_names", WILDSDataset.DEFAULT_SPLIT_NAMES)

    @property
    def split_array(self):
        return self._split_array

    @property
    def y_array(self) -> torch.Tensor:
        return self._y_array

    @property
    def y_size(self) -> int:
        return self._y_size

    @property
    def n_classes(self) -> int | None:
        return getattr(self, "_n_classes", None)

    @property
    def is_classification(self) -> bool:
        return getattr(self, "_is_classification", self.n_classes is not None)

    @property
    def is_detection(self) -> bool:
        return getattr(self, "_is_detection", False)

    @property
    def metadata_fields(self) -> list[str]:
        return self._metadata_fields

    @property
    def metadata_array(self) -> torch.Tensor:
        return self._metadata_array

    @property
    def metadata_map(self) -> dict[str, list[str]] | None:
        return getattr(self, "_metadata_map", None)


class WILDSSubset(Dataset):
    def __init__(self, dataset: WILDSDataset, indices, transform=None, do_transform_y: bool = False) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices)
        inherited_attrs = [
            "_dataset_name",
            "_data_dir",
            "_collate",
            "_split_scheme",
            "_split_dict",
            "_split_names",
            "_y_size",
            "_n_classes",
            "_metadata_fields",
            "_metadata_map",
        ]
        for attr_name in inherited_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        self.transform = transform
        self.do_transform_y = do_transform_y
        self.og_group_counts = None

    def __getitem__(self, idx: int):
        x, y, metadata = self.dataset[int(self.indices[idx])]
        if self.transform is not None:
            if self.do_transform_y:
                x, y = self.transform(x, y)
            else:
                x = self.transform(x)
        return x, y, metadata

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def split_array(self):
        return self.dataset.split_array[self.indices]

    @property
    def y_array(self) -> torch.Tensor:
        return self.dataset.y_array[self.indices]

    @property
    def metadata_array(self) -> torch.Tensor:
        return self.dataset.metadata_array[self.indices]

    @property
    def collate(self):
        return getattr(self.dataset, "collate", None)

    def eval(self, y_pred: torch.Tensor, y_true: torch.Tensor, metadata: torch.Tensor):
        return self.dataset.eval(y_pred, y_true, metadata)


class CombinatorialGrouper:
    def __init__(self, dataset: WILDSDataset, groupby_fields: list[str] | None) -> None:
        if isinstance(dataset, WILDSSubset):
            raise ValueError("Grouper should be defined for the full dataset, not a subset")
        self.groupby_fields = groupby_fields
        if groupby_fields is None:
            self._n_groups = 1
            return

        self.groupby_field_indices = [
            idx for idx, field in enumerate(dataset.metadata_fields) if field in groupby_fields
        ]
        if len(self.groupby_field_indices) != len(groupby_fields):
            raise ValueError("At least one group field not found in dataset.metadata_fields")
        grouped_metadata = dataset.metadata_array[:, self.groupby_field_indices]
        grouped_metadata_long = grouped_metadata.long()
        if not torch.all(grouped_metadata == grouped_metadata_long):
            warnings.warn(f"CombinatorialGrouper: converting metadata with fields {groupby_fields} into long")
        grouped_metadata = grouped_metadata_long
        for idx, field in enumerate(groupby_fields):
            min_value = grouped_metadata[:, idx].min()
            if min_value < 0:
                raise ValueError(f"Metadata for CombinatorialGrouper cannot have values less than 0: {field}")
            if min_value > 0:
                warnings.warn(f"Minimum metadata value for CombinatorialGrouper is not 0 ({field}, {min_value})")

        self.cardinality = 1 + torch.max(grouped_metadata, dim=0)[0]
        cumprod = torch.cumprod(self.cardinality, dim=0)
        self._n_groups = cumprod[-1].item()
        self.factors_np = np.concatenate(([1], cumprod[:-1].numpy()))
        self.factors = torch.from_numpy(self.factors_np)
        self.metadata_map = dataset.metadata_map

    @property
    def n_groups(self) -> int:
        return self._n_groups

    def metadata_to_group(self, metadata: torch.Tensor, return_counts: bool = True):
        if self.groupby_fields is None:
            groups = torch.zeros(metadata.shape[0], dtype=torch.long)
        else:
            groups = metadata[:, self.groupby_field_indices].long() @ self.factors.to(metadata.device)
        if return_counts:
            return groups, get_counts(groups, self._n_groups)
        return groups

    def group_str(self, group: int) -> str:
        if self.groupby_fields is None:
            return "all"
        n_fields = len(self.factors_np)
        metadata = np.zeros(n_fields)
        for idx in range(n_fields - 1):
            metadata[idx] = (group % self.factors_np[idx + 1]) // self.factors_np[idx]
        metadata[n_fields - 1] = group // self.factors_np[n_fields - 1]

        parts = []
        for idx in reversed(range(n_fields)):
            meta_val = int(metadata[idx])
            field = self.groupby_fields[idx]
            if self.metadata_map is not None and field in self.metadata_map:
                meta_val = self.metadata_map[field][meta_val]
            parts.append(f"{field} = {meta_val}")
        return ", ".join(parts)

    def group_field_str(self, group: int) -> str:
        return self.group_str(group).replace("=", ":").replace(",", "_").replace(" ", "")


def get_ssl_train_loader(
    loader: str,
    dataset: WILDSDataset | WILDSSubset,
    batch_size: int,
    uniform_over_groups: bool | None = None,
    grouper: CombinatorialGrouper | None = None,
    **loader_kwargs,
) -> DataLoader:
    if loader != "standard":
        raise ValueError("Only standard SSL loaders are supported")
    if uniform_over_groups is None or not uniform_over_groups:
        return DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=dataset.collate, **loader_kwargs)
    if grouper is None:
        raise ValueError("grouper is required when uniform_over_groups=True")
    groups, group_counts = grouper.metadata_to_group(dataset.metadata_array, return_counts=True)
    weights = (1 / group_counts)[groups]
    sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
    return DataLoader(dataset, shuffle=False, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate, **loader_kwargs)


def get_train_loader(
    loader: str,
    dataset: WILDSDataset | WILDSSubset,
    batch_size: int,
    **loader_kwargs,
) -> DataLoader:
    if loader != "standard":
        raise ValueError("Only standard train loaders are supported")
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=dataset.collate, **loader_kwargs)


def get_eval_loader(
    loader: str,
    dataset: WILDSDataset | WILDSSubset,
    batch_size: int,
    **loader_kwargs,
) -> DataLoader:
    if loader != "standard":
        raise ValueError("Only standard eval loaders are supported")
    return DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=dataset.collate, **loader_kwargs)
