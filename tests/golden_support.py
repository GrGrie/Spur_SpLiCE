"""Deterministic synthetic fixtures and snapshot helpers for the golden regression tests.

The golden tests freeze the observable behaviour of the pipeline before a refactor: runner
commands, teacher-graph structure and short training runs. A behaviour change shows up as a
snapshot diff. Regenerate the snapshots deliberately with ``SPUR_SPLICE_UPDATE_GOLDEN=1``.
"""

from __future__ import annotations

import json
import math
import os
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
UPDATE_ENV = "SPUR_SPLICE_UPDATE_GOLDEN"

# Group sizes per split in (y, place) order: (0,0), (0,1), (1,0), (1,1). The train split
# mirrors the Waterbirds imbalance, where the majority groups align y with the background.
SPLIT_GROUP_SIZES = {0: (18, 6, 6, 18), 1: (4, 4, 4, 4), 2: (4, 4, 4, 4)}
IMAGE_SIZE = 32
CLIP_DIM = 16
VOCABULARY = ["water", "lake", "ocean", "forest", "bamboo", "waterbird", "gull", "sparrow"]


def update_requested() -> bool:
    return os.environ.get(UPDATE_ENV, "") == "1"


def compare_or_update(name: str, actual: Any, compare) -> None:
    """Compare ``actual`` with the stored snapshot, or rewrite the snapshot on request."""

    path = GOLDEN_DIR / name
    if update_requested() or not path.is_file():
        created = not path.is_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if created and not update_requested():
            raise unittest.SkipTest(f"Created golden snapshot {path}; commit it to enable the comparison.")
        return
    expected = json.loads(path.read_text(encoding="utf-8"))
    compare(expected, actual)


def assert_close_tree(testcase, expected: Any, actual: Any, *, rel: float, abs_: float, path: str = "") -> None:
    """Recursively compare JSON trees with a numeric tolerance for floats."""

    if isinstance(expected, dict):
        testcase.assertIsInstance(actual, dict, path)
        testcase.assertEqual(sorted(expected), sorted(actual), f"keys differ at {path or '<root>'}")
        for key in expected:
            assert_close_tree(testcase, expected[key], actual[key], rel=rel, abs_=abs_, path=f"{path}/{key}")
    elif isinstance(expected, list):
        testcase.assertIsInstance(actual, list, path)
        testcase.assertEqual(len(expected), len(actual), f"length differs at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_close_tree(testcase, left, right, rel=rel, abs_=abs_, path=f"{path}[{index}]")
    elif isinstance(expected, float) and not isinstance(expected, bool):
        testcase.assertTrue(
            math.isclose(float(actual), expected, rel_tol=rel, abs_tol=abs_),
            f"{path}: expected {expected}, got {actual}",
        )
    else:
        testcase.assertEqual(expected, actual, path)


def split_metadata() -> pd.DataFrame:
    """Return Waterbirds-format metadata: img_filename, y, place and split."""

    rows = []
    for split, sizes in SPLIT_GROUP_SIZES.items():
        for group, count in enumerate(sizes):
            y, place = divmod(group, 2)
            for _ in range(count):
                rows.append({"y": y, "place": place, "split": split})
    frame = pd.DataFrame(rows)
    frame.insert(0, "img_filename", [f"images/{index:04d}.png" for index in range(len(frame))])
    return frame


def write_waterbirds_fixture(root: Path) -> Path:
    """Write a tiny Waterbirds-format dataset: background colour encodes place, a centred shape encodes y."""

    dataset_dir = root / "waterbirds"
    (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
    metadata = split_metadata()
    generator = np.random.default_rng(0)
    for row in metadata.itertuples():
        background = np.array([40, 90, 200] if row.place else [60, 150, 60], dtype=np.float32)
        pixels = np.tile(background, (IMAGE_SIZE, IMAGE_SIZE, 1))
        # A small low-contrast mark encodes y, so the background dominates as in Waterbirds.
        yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]
        mark = (np.abs(yy - 16) <= 2) & (np.abs(xx - 16) <= 2) if row.y else (yy - 16) ** 2 + (xx - 16) ** 2 <= 6
        pixels[mark] += 45.0
        pixels += generator.normal(0.0, 30.0, size=pixels.shape)
        image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
        image.save(dataset_dir / row.img_filename)
    metadata.to_csv(dataset_dir / "metadata.csv", index=False)
    return root


def train_sample_ids() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Sample IDs, labels and places of the train split in dataset order."""

    metadata = split_metadata()
    train = metadata[metadata["split"] == 0]
    sample_ids = [f"waterbirds:{index}" for index in train.index]
    return sample_ids, train["y"].to_numpy(), train["place"].to_numpy()


def synthetic_splice_cache(sample_ids: list[str], y: np.ndarray, place: np.ndarray, seed: int = 7) -> dict:
    """Build a structured SpLiCE dataset cache.

    Embeddings combine a class direction, a background direction and noise. The first three
    vocabulary entries are background synonyms, so a grouping and audit can recover them.
    Labels shape the synthetic signal and are absent from the returned cache.
    """

    generator = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(CLIP_DIM, CLIP_DIM, generator=generator))[0].T
    class_direction, background_direction = basis[0], basis[1]
    y_signed = torch.as_tensor(y, dtype=torch.float32) * 2 - 1
    place_signed = torch.as_tensor(place, dtype=torch.float32) * 2 - 1
    n_samples = len(sample_ids)
    noise = 0.35 * torch.randn(n_samples, CLIP_DIM, generator=generator)
    embeddings = (
        y_signed[:, None] * class_direction
        + 1.2 * place_signed[:, None] * background_direction
        + noise
        + 0.5
    )

    dictionary = torch.stack([
        background_direction + 0.10 * basis[2],
        background_direction + 0.10 * basis[3],
        background_direction + 0.10 * basis[4],
        -background_direction + 0.30 * basis[5],
        basis[6],
        class_direction + 0.10 * basis[7],
        class_direction + 0.20 * basis[8],
        -class_direction + 0.20 * basis[9],
    ])
    water = torch.as_tensor(place, dtype=torch.float32)
    bird = torch.as_tensor(y, dtype=torch.float32)
    jitter = 0.05 * torch.rand(n_samples, len(VOCABULARY), generator=generator)
    codes = torch.stack([
        0.9 * water, 0.7 * water, 0.5 * water,
        0.8 * (1 - water),
        (torch.rand(n_samples, generator=generator) > 0.7).float() * 0.4,
        0.8 * bird, 0.5 * bird,
        0.7 * (1 - bird),
    ], dim=1)
    codes = torch.where(codes > 0, codes + jitter, codes)
    return {
        "cache_version": 1,
        "provenance": {"fixture": "golden-synthetic-splice-cache", "seed": seed},
        "sample_ids": list(sample_ids),
        "clip_embeddings": F.normalize(embeddings, dim=1),
        "image_mean": embeddings.mean(dim=0),
        "splice_codes": codes,
        "dictionary": F.normalize(dictionary, dim=1),
        "vocabulary": list(VOCABULARY),
    }


def synthetic_target_bank(sample_ids: list[str], seed: int = 11) -> dict:
    """Frozen concept-transfer target bank with 512-dimensional targets."""

    generator = torch.Generator().manual_seed(seed)
    n_samples = len(sample_ids)
    raw = torch.randn(n_samples, 512, generator=generator)
    reconstruction = raw + 0.1 * torch.randn(n_samples, 512, generator=generator)
    permutation = torch.randperm(n_samples, generator=generator)
    return {
        "artifact": "splice_concept_transfer_targets_v1",
        "version": 1,
        "target_dim": 512,
        "sample_ids": list(sample_ids),
        "raw": raw,
        "reconstruction": reconstruction,
        "shuffled_reconstruction": reconstruction[permutation],
        "valid_mask": torch.ones(n_samples, dtype=torch.bool),
        "permutation": permutation,
        "cache_fingerprint": "golden-fixture",
    }
