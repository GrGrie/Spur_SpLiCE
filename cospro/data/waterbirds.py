from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from cospro.data.base import DatasetConfig, SpuriousDataset, register_dataset
from cospro.data.paths import resolve_dataset_root
from third_party.wilds_compat import CombinatorialGrouper


@dataclass(frozen=True)
class WaterbirdsConfig(DatasetConfig):
    pass


@register_dataset
class WaterbirdsDataset(SpuriousDataset):
    """Waterbirds with WILDS-style metadata and SpurSSL-compatible splits."""

    name = "waterbirds"
    num_classes = 2
    Config = WaterbirdsConfig

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
        self._n_classes = self.num_classes
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
        return resolve_dataset_root(
            root_dir,
            "waterbirds",
            ["metadata.csv"],
        )

    def get_input(self, idx: int):
        image_path = Path(self.data_dir) / self._input_array[idx]
        return Image.open(image_path).convert("RGB")
