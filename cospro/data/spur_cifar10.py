from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import ImageDraw
from torchvision import datasets

from cospro.data.base import DatasetConfig, SpuriousDataset, register_dataset
from cospro.data.paths import resolve_dataset_root
from third_party.wilds_compat import CombinatorialGrouper


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


@dataclass(frozen=True)
class SpurCIFAR10Config(DatasetConfig):
    image_size: int = 32
    val_fraction: float = 0.1
    train_spurious_correlation: float = 0.95
    eval_spurious_correlation: float = 0.1
    spurious_seed: int = 0
    line_width: int = 2
    download: bool = True


@register_dataset
class SpurCIFAR10Dataset(SpuriousDataset):
    """CIFAR-10 with one class-associated horizontal-line color per class."""

    name = "spur_cifar10"
    aliases = ("spur-cifar10",)
    num_classes = 10
    image_size = 32
    mean = CIFAR10_MEAN
    std = CIFAR10_STD
    Config = SpurCIFAR10Config

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
        self._n_classes = self.num_classes
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

    @classmethod
    def from_config(cls, config: SpurCIFAR10Config) -> "SpurCIFAR10Dataset":
        """The spurious correlation and the val split are drawn at construction, so they come
        from the configuration rather than from a file."""

        return cls(
            config.root_dir,
            val_fraction=config.val_fraction,
            train_spurious_correlation=config.train_spurious_correlation,
            eval_spurious_correlation=config.eval_spurious_correlation,
            spurious_seed=config.spurious_seed,
            line_width=config.line_width,
            download=config.download,
        )

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
