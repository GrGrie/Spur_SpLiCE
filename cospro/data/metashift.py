"""MetaShift Cats and Dogs: predict the animal while the indoor/outdoor context is the spurious one.

The released tree holds two directories of images, ``train`` and ``test``, each split by class and
by context::

    <root>/train/cat/cat(indoor)/<image>.jpg
    <root>/test/dog/dog(outdoor)/<image>.jpg

Training images carry the correlation the method has to survive: cats are mostly indoor and dogs
mostly outdoor. The released test images are balanced over the four groups, so this adapter uses a
deterministic half of them for validation and the other half for test, stratified per group. Both
halves are disjoint and the training images stay untouched, which keeps model selection away from
the images the final numbers come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cospro.data.base import DatasetConfig, SpuriousDataset, register_dataset
from cospro.data.paths import resolve_dataset_root
from third_party.wilds_compat import CombinatorialGrouper

#: Target label and spurious attribute, in the order their metadata columns use.
CLASSES = ("cat", "dog")
CONTEXTS = ("indoor", "outdoor")
#: Directory names the release ships, as <class>/<class>(<context>).
SPLIT_DIRECTORIES = ("train", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class MetaShiftConfig(DatasetConfig):
    #: Share of the balanced released test images that becomes the validation split.
    val_fraction: float = 0.5
    #: Seed of the per-group validation draw. Changing it changes which images validate.
    val_seed: int = 0


@register_dataset
class MetaShiftDataset(SpuriousDataset):
    """MetaShift Cats and Dogs with the indoor/outdoor context as the spurious attribute."""

    name = "metashift"
    aliases = ("MetaShift", "metashift_cats_dogs", "metashift-cats-dogs", "MetaShiftCatsDogs")
    num_classes = 2
    Config = MetaShiftConfig

    def __init__(
        self,
        root_dir: str = "./datasets",
        split_scheme: str = "official",
        val_fraction: float = 0.5,
        val_seed: int = 0,
    ) -> None:
        if not 0 < val_fraction < 1:
            raise ValueError(f"MetaShift val_fraction must lie in (0, 1); got {val_fraction}.")
        self.root_dir = Path(root_dir)
        self._data_dir = self._find_data_dir(self.root_dir)
        self.val_fraction = float(val_fraction)
        self.val_seed = int(val_seed)

        paths, targets, contexts, released = self._read_tree(self._data_dir)
        self._input_array = np.asarray([str(path) for path in paths])
        self._y_array = torch.LongTensor(targets)
        self._y_size = 1
        self._n_classes = self.num_classes
        self._metadata_array = torch.stack((torch.LongTensor(contexts), self._y_array), dim=1)
        self._metadata_fields = ["context", "y"]
        self._metadata_map = {"context": list(CONTEXTS), "y": list(CLASSES)}
        self._split_scheme = split_scheme
        if self._split_scheme != "official":
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")
        self._split_array = self._build_split_array(released, targets, contexts)
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=["context", "y"])
        counts = [int((self._split_array == split).sum()) for split in (0, 1, 2)]
        print(
            f"[INFO] MetaShift resolved root={self._data_dir} "
            f"train={counts[0]} val={counts[1]} test={counts[2]} val_seed={self.val_seed}",
            flush=True,
        )
        super().__init__(root_dir, split_scheme)

    @classmethod
    def from_config(cls, config: MetaShiftConfig) -> "MetaShiftDataset":
        """The validation split is drawn here, so it comes from the configuration."""

        return cls(config.root_dir, val_fraction=config.val_fraction, val_seed=config.val_seed)

    @staticmethod
    def _find_data_dir(root_dir: Path) -> Path:
        markers = [f"{split}/{name}" for split in SPLIT_DIRECTORIES for name in CLASSES]
        errors = []
        for alias in ("metashift", "metashift_cats_dogs", "MetaShiftCatsDogs"):
            try:
                return resolve_dataset_root(root_dir, alias, markers)
            except FileNotFoundError as error:
                errors.append(str(error))
        raise FileNotFoundError(
            "Could not find MetaShift Cats and Dogs. Expected <root>/train/<class>/<class>(<context>)/*.jpg "
            f"with classes {list(CLASSES)} and contexts {list(CONTEXTS)}.\n" + "\n".join(errors)
        )

    @staticmethod
    def _read_tree(data_dir: Path) -> tuple[list[Path], np.ndarray, np.ndarray, np.ndarray]:
        """Every image of the release in a fixed order, with its target, context and source split."""

        paths: list[Path] = []
        targets: list[int] = []
        contexts: list[int] = []
        released: list[int] = []
        for split_index, split in enumerate(SPLIT_DIRECTORIES):
            for target, class_name in enumerate(CLASSES):
                for context, context_name in enumerate(CONTEXTS):
                    directory = data_dir / split / class_name / f"{class_name}({context_name})"
                    if not directory.is_dir():
                        raise FileNotFoundError(f"MetaShift directory not found: {directory}")
                    images = sorted(
                        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
                    )
                    if not images:
                        raise ValueError(f"MetaShift directory holds no images: {directory}")
                    paths.extend(images)
                    targets.extend([target] * len(images))
                    contexts.extend([context] * len(images))
                    released.extend([split_index] * len(images))
        return (
            paths,
            np.asarray(targets, dtype=np.int64),
            np.asarray(contexts, dtype=np.int64),
            np.asarray(released, dtype=np.int64),
        )

    def _build_split_array(self, released: np.ndarray, targets: np.ndarray, contexts: np.ndarray) -> np.ndarray:
        """Train as released; the released test images split per group into validation and test."""

        splits = np.full(len(released), 2, dtype=np.int64)
        splits[released == 0] = 0
        generator = np.random.RandomState(self.val_seed)
        for target in range(len(CLASSES)):
            for context in range(len(CONTEXTS)):
                group = np.where((released == 1) & (targets == target) & (contexts == context))[0]
                validation = int(round(len(group) * self.val_fraction))
                if validation < 1 or validation >= len(group):
                    raise ValueError(
                        f"MetaShift group (y={CLASSES[target]}, a={CONTEXTS[context]}) holds {len(group)} "
                        f"released test images; val_fraction={self.val_fraction} leaves an empty split."
                    )
                splits[generator.permutation(group)[:validation]] = 1
        return splits

    def get_input(self, idx: int):
        image_path = Path(self._input_array[idx])
        if not image_path.exists():
            raise FileNotFoundError(f"MetaShift image not found at {image_path}")
        return Image.open(image_path).convert("RGB")
