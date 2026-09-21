from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from experiments.spurious_eval.datasets.base import DatasetConfig, SpuriousDataset, register_dataset
from experiments.spurious_eval.datasets.paths import resolve_dataset_root
from third_party.wilds_compat import CombinatorialGrouper


@dataclass(frozen=True)
class CelebAConfig(DatasetConfig):
    pass


@register_dataset
class CelebADataset(SpuriousDataset):
    """CelebA with Blond_Hair as target and Male as the spurious attribute.

    This matches the CelebA protocol used by SpurSSL, LateTVG, and the common
    group-robustness benchmark: predict hair colour while measuring groups
    formed by gender and the target label.
    """

    name = "celeba"
    aliases = ("CelebA", "celebA")
    num_classes = 2
    Config = CelebAConfig

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
        if attrs["image_id"].astype(str).duplicated().any():
            raise ValueError("CelebA attributes contain duplicate image_id values.")

        self.attrs = attrs.reset_index(drop=True)
        self._input_array = self.attrs["image_id"].astype(str).values
        self._y_array = torch.LongTensor(
            (self.attrs["Blond_Hair"].astype(int).values == 1).astype(np.int64)
        )
        male = (self.attrs["Male"].astype(int).values == 1).astype(np.int64)
        self._y_size = 1
        self._n_classes = self.num_classes
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
        if not split_path.exists():
            raise FileNotFoundError(
                f"CelebA official split metadata not found at {split_path}. "
                "Refusing to substitute a random split."
            )
        splits = pd.read_csv(split_path)
        if "partition" not in splits.columns:
            splits = splits.rename(columns={splits.columns[-1]: "partition"})
        if "image_id" not in splits.columns:
            splits = splits.rename(columns={splits.columns[0]: "image_id"})
        split_ids = splits["image_id"].astype(str)
        if split_ids.duplicated().any():
            raise ValueError("CelebA split metadata contains duplicate image_id values.")
        partitions = splits["partition"].astype(int)
        unexpected = sorted(set(partitions).difference({0, 1, 2}))
        if unexpected:
            raise ValueError(f"CelebA split metadata contains invalid partitions: {unexpected}")
        split_lookup = dict(zip(split_ids, partitions))
        missing = sorted(set(map(str, self._input_array)).difference(split_lookup))
        if missing:
            preview = ", ".join(missing[:3])
            raise ValueError(
                f"CelebA split metadata is missing {len(missing)} attribute image(s), including {preview}."
            )
        return np.asarray(
            [split_lookup[str(image_id)] for image_id in self._input_array], dtype=np.int64
        )

    def get_input(self, idx: int):
        image_path = self._data_dir / "img_align_celeba" / self._input_array[idx]
        if not image_path.exists():
            raise FileNotFoundError(f"CelebA image not found at {image_path}")
        return Image.open(image_path).convert("RGB")
