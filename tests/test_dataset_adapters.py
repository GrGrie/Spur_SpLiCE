"""The dataset seam: what registration guarantees and what each loader role reads.

Every adapter is reached through the registry and built through one ``build_loader``, so these
tests exercise the contract rather than any one dataset. They use a fake adapter for the parts that
would need the real images on disk.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from cospro.data.base import (
    CANONICAL_DATASETS,
    LOADER_ROLES,
    DatasetConfig,
    SpuriousDataset,
    build_loader,
    build_probe_loaders,
    register_dataset,
)
from cospro.data.registry import canonical_dataset_name, dataset_class, dataset_names
from cospro.data.transforms import TwoCropTransform


class FakeSubset(TensorDataset):
    """A split that remembers the transform its role asked for."""

    def __init__(self, transform) -> None:
        super().__init__(torch.arange(4))
        self.transform = transform
        self.collate = None


class FakeSplitDataset:
    """Stand in for a built adapter and record the split each loader role reads."""

    def __init__(self, requests: list) -> None:
        self.requests = requests
        self.grouper = None

    def get_subset(self, split, transform=None):
        subset = FakeSubset(transform)
        self.requests.append((split, subset))
        return subset


class RegistryTests(unittest.TestCase):
    def test_every_spelling_resolves_to_the_canonical_name(self):
        self.assertEqual(dataset_names(), ["celeba", "spur_cifar10", "waterbirds"])
        for spelling in ("CelebA", "celebA", "CELEBA", " celeba "):
            self.assertEqual(canonical_dataset_name(spelling), "celeba")
        self.assertEqual(canonical_dataset_name("spur-cifar10"), "spur_cifar10")
        self.assertIs(dataset_class("CelebA"), CANONICAL_DATASETS["celeba"])

    def test_an_unknown_dataset_names_the_choices(self):
        with self.assertRaisesRegex(ValueError, "Unsupported dataset: mnist"):
            canonical_dataset_name("mnist")

    def test_two_adapters_cannot_claim_one_spelling(self):
        with self.assertRaisesRegex(ValueError, "both claim"):

            @register_dataset
            class Impostor(SpuriousDataset):
                name = "impostor"
                aliases = ("waterbirds",)

        # A rejected registration leaves no half-registered spelling behind.
        self.assertEqual(dataset_names(), ["celeba", "spur_cifar10", "waterbirds"])
        with self.assertRaises(ValueError):
            canonical_dataset_name("impostor")

    def test_an_adapter_needs_a_canonical_name(self):
        with self.assertRaisesRegex(ValueError, "canonical name"):

            @register_dataset
            class Anonymous(SpuriousDataset):
                pass

    def test_image_size_decides_the_model_a_dataset_accepts(self):
        cifar = dataset_class("spur_cifar10")
        self.assertEqual(cifar.default_model(), "resnet18")
        self.assertIsNone(cifar.model_error("resnet18"))
        self.assertEqual(
            cifar.model_error("resnet18_large"),
            "spur_cifar10 uses 32x32 images; choose --model resnet18 or --model resnet50.",
        )
        self.assertIsNotNone(cifar.model_error("resnet50_pretrained"))
        for name in ("waterbirds", "celeba"):
            adapter = dataset_class(name)
            self.assertEqual(adapter.default_model(), "resnet18_large")
            self.assertIsNone(adapter.model_error("resnet18_large"))


class LoaderRoleTests(unittest.TestCase):
    def build_all_roles(self, adapter) -> list:
        requests: list = []
        with patch.object(adapter, "from_config", lambda config: FakeSplitDataset(requests)):
            for role in LOADER_ROLES:
                loader = build_loader(adapter, role, adapter.Config(), batch_size=2, num_workers=0)
                self.assertIsInstance(loader, DataLoader)
        return requests

    def test_every_dataset_serves_the_four_roles_from_the_documented_splits(self):
        for name, adapter in CANONICAL_DATASETS.items():
            with self.subTest(dataset=name):
                requests = self.build_all_roles(adapter)
                self.assertEqual([split for split, _ in requests], ["train", "train", "ds_train", "val"])
                self.assertIsInstance(requests[0][1].transform, TwoCropTransform, "ssl makes two crops")
                self.assertNotIsInstance(requests[1][1].transform, TwoCropTransform, "rank makes one")

    def test_the_ordered_roles_do_not_shuffle(self):
        adapter = dataset_class("waterbirds")
        requests: list = []
        with patch.object(adapter, "from_config", lambda config: FakeSplitDataset(requests)):
            shuffled = {
                role: isinstance(
                    build_loader(adapter, role, adapter.Config(), batch_size=2, num_workers=0).sampler,
                    torch.utils.data.RandomSampler,
                )
                for role in LOADER_ROLES
            }
        self.assertEqual(shuffled, {"ssl": True, "rank": False, "probe_train": True, "probe_eval": False})

    def test_build_loader_refuses_an_unknown_role(self):
        adapter = dataset_class("waterbirds")
        with self.assertRaisesRegex(ValueError, "Unsupported loader role"):
            build_loader(adapter, "validation", adapter.Config(), batch_size=2)

    def test_the_probe_reads_its_two_splits_from_one_built_dataset(self):
        adapter = dataset_class("celeba")
        requests: list = []
        builds = []

        def build_once(config):
            builds.append(config)
            return FakeSplitDataset(requests)

        with patch.object(adapter, "from_config", build_once):
            train_loader, eval_loader = build_probe_loaders(
                adapter, adapter.Config(train_split="us_train", eval_split="test"), batch_size=2,
                train_loader_kwargs={"num_workers": 0}, eval_loader_kwargs={"num_workers": 0},
            )
        self.assertEqual(len(builds), 1, "reading the metadata twice would double the probe setup")
        self.assertEqual([split for split, _ in requests], ["us_train", "test"])
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(eval_loader, DataLoader)

    def test_a_dataset_adds_its_own_configuration_fields(self):
        cifar = dataset_class("spur_cifar10").Config()
        self.assertEqual(cifar.image_size, 32)
        self.assertEqual(cifar.spurious_seed, 0)
        self.assertIsInstance(cifar, DatasetConfig)
        self.assertEqual(dataset_class("waterbirds").Config().image_size, 224)


if __name__ == "__main__":
    unittest.main()
