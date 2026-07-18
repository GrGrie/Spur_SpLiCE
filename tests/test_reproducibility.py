import argparse
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.spurious_eval.datasets.registry import DATASET_REGISTRY
from experiments.spurious_eval.training.checkpointing import load_checkpoint, save_checkpoint
from experiments.spurious_eval.training.ssl_loop import extract_normalized_train_features
from spur_splice import make_dataloader_kwargs, preserve_rng_state


def loader_order(loader: DataLoader) -> list[int]:
    return [int(value) for batch in loader for value in batch[0]]


class EncoderOnlyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Flatten()


class FakeScaler:
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = state["value"]


class ReproducibilityTests(unittest.TestCase):
    def test_every_dataset_has_a_dedicated_rank_loader(self):
        for dataset_name, spec in DATASET_REGISTRY.items():
            with self.subTest(dataset=dataset_name):
                self.assertIn("rank_loader", spec)

    def test_rank_loader_iteration_does_not_advance_training_sampler(self):
        args = argparse.Namespace(seed=17, num_workers=0)
        dataset = TensorDataset(torch.arange(24))

        train_with_rank = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            **make_dataloader_kwargs(args, shuffle=True),
        )
        rank_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            **make_dataloader_kwargs(args, shuffle=False, seed=args.seed + 1_000_000),
        )
        first_epoch_with_rank = loader_order(train_with_rank)
        loader_order(rank_loader)
        second_epoch_with_rank = loader_order(train_with_rank)

        uninterrupted_train = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            **make_dataloader_kwargs(args, shuffle=True),
        )
        first_epoch_uninterrupted = loader_order(uninterrupted_train)
        second_epoch_uninterrupted = loader_order(uninterrupted_train)

        self.assertEqual(first_epoch_with_rank, first_epoch_uninterrupted)
        self.assertEqual(second_epoch_with_rank, second_epoch_uninterrupted)

    def test_rng_context_restores_python_numpy_and_torch(self):
        torch.manual_seed(5)
        np.random.seed(5)
        random.seed(5)
        expected_torch_state = torch.get_rng_state().clone()
        expected_numpy_state = np.random.get_state()
        expected_python_state = random.getstate()

        with preserve_rng_state():
            torch.rand(8)
            np.random.rand(8)
            random.random()

        torch.testing.assert_close(torch.get_rng_state(), expected_torch_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], expected_numpy_state[1]))
        self.assertEqual(random.getstate(), expected_python_state)

    def test_rank_feature_extraction_uses_tensor_batches_and_restores_model_mode(self):
        images = torch.arange(24, dtype=torch.float32).reshape(3, 1, 2, 4)
        loader = DataLoader(TensorDataset(images, torch.zeros(3)), batch_size=2, shuffle=False)
        model = EncoderOnlyModel()
        model.train()
        args = argparse.Namespace(device="cpu", channels_last=False)

        features = extract_normalized_train_features(model, loader, args)

        self.assertEqual(tuple(features.shape), (3, 8))
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.linalg.vector_norm(features, dim=1), torch.ones(3))

    def test_checkpoint_restores_rng_scaler_and_loader_generator(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = FakeScaler(7)
        loader_generator = torch.Generator().manual_seed(101)
        torch.manual_seed(23)
        np.random.seed(23)
        random.seed(23)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = str(Path(temporary_directory) / "resume.pth")
            save_checkpoint(
                model,
                optimizer,
                argparse.Namespace(),
                12,
                checkpoint_path,
                scaler=scaler,
                loader_generator=loader_generator,
            )
            expected = (
                torch.rand(3),
                np.random.rand(3),
                random.random(),
                torch.randperm(10, generator=loader_generator),
            )

            torch.rand(10)
            np.random.rand(10)
            random.random()
            torch.randperm(10, generator=loader_generator)
            scaler.value = -1

            epoch = load_checkpoint(
                model,
                optimizer,
                checkpoint_path,
                torch.device("cpu"),
                scaler=scaler,
                loader_generator=loader_generator,
            )
            actual = (
                torch.rand(3),
                np.random.rand(3),
                random.random(),
                torch.randperm(10, generator=loader_generator),
            )

        self.assertEqual(epoch, 12)
        self.assertEqual(scaler.value, 7)
        torch.testing.assert_close(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2], expected[2])
        torch.testing.assert_close(actual[3], expected[3])


if __name__ == "__main__":
    unittest.main()
