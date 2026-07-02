import numpy as np
import torch

from experiments.spurious_eval.datasets.spur_cifar10 import CIFAR10_CLASSES, SpurCIFAR10Dataset
from experiments.spurious_eval.metrics import spurious_group_ids


def test_spurious_colors_match_class_at_full_correlation():
    labels = np.tile(np.arange(len(CIFAR10_CLASSES)), 100)
    colors = SpurCIFAR10Dataset._make_spurious_values(labels, correlation=1.0, seed=0)
    np.testing.assert_array_equal(colors, labels)


def test_spurious_colors_never_match_class_at_zero_correlation():
    labels = np.tile(np.arange(len(CIFAR10_CLASSES)), 100)
    colors = SpurCIFAR10Dataset._make_spurious_values(labels, correlation=0.0, seed=0)
    assert np.all(colors != labels)
    assert set(colors.tolist()) == set(range(len(CIFAR10_CLASSES)))


def test_spurious_colors_follow_requested_correlation():
    labels = np.tile(np.arange(len(CIFAR10_CLASSES)), 10_000)
    colors = SpurCIFAR10Dataset._make_spurious_values(labels, correlation=0.95, seed=0)
    observed_correlation = np.mean(colors == labels)
    assert abs(observed_correlation - 0.95) < 0.005


def test_group_ids_cover_ten_classes_by_ten_colors():
    metadata = torch.tensor(
        [[color, label] for label in range(10) for color in range(10)],
        dtype=torch.long,
    )
    groups = spurious_group_ids(metadata)
    assert torch.equal(torch.sort(groups).values, torch.arange(100))
