import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from experiments.complete_projection_controls import MANIFESTS, validate_inputs
from experiments.runner import command_for, load_manifest
from experiments.spurious_eval.losses.contrastive import SimCLRLoss
from experiments.spurious_eval.training.checkpointing import load_checkpoint, save_checkpoint
from experiments.spurious_eval.training.la_ssl import build_la_ssl_loader, sampling_probabilities
from experiments.spurious_eval.training.ssl_loop import simclr_forward_loss
from splice.artifacts import PROJECT_ROOT
from splice.concept_distillation import ConceptDistillationRegularizer, FrozenConceptTransferSubset
from spur_splice import preserve_rng_state, seed_worker


class TwoViews(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        x = torch.tensor([index / 6, 1.0, 0.5])
        return [x, x + torch.rand(3) * 0.3], 0, torch.tensor([0, 0])


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
        self.head = torch.nn.Linear(4, 3)
        self.clip_distillation_head = torch.nn.Linear(4, 5)

    def forward(self, x):
        return torch.nn.functional.normalize(self.head(self.encoder(x)), dim=1)


class SubmissionControlsTests(unittest.TestCase):
    def test_completion_configs_match_both_historical_seeds(self):
        aliases = {"cospro_temperature": "crp_temperature", "cospro_start_epoch": "crp_start_epoch",
                   "cospro_warmup_epochs": "crp_warmup_epochs"}
        relocated = {"data_folder", "wandb_run_name", "concept_transfer_targets", "cospro_teacher_graph", "splice_mode"}
        for name in MANIFESTS:
            manifest = load_manifest(PROJECT_ROOT / "experiments/manifests" / name)
            arm = next(iter(manifest["arms"]))
            for seed in (1, 3):
                path = next((PROJECT_ROOT / "outputs/seeds" / manifest["name"] / f"seed_{seed:02d}" / arm).glob("*/command.json"))
                old = json.loads(path.read_text())["recovered_config"]
                for key, value in {**manifest["common"], **manifest["arms"][arm]["args"]}.items():
                    if key in relocated:
                        continue
                    expected = old[aliases.get(key, key)]
                    if isinstance(expected, list):
                        expected = ",".join(map(str, expected))
                    self.assertEqual(value, expected, (name, seed, key))
            command, output = command_for(manifest, 2, arm, "check")
            self.assertIn("seed2_check", command[command.index("--wandb_run_name") + 1])
            self.assertEqual(output.name, "check")
            if arm == "semantic_splice":
                validate_inputs(command)

    def test_frozen_subset_aligns_rows_by_id_without_labels(self):
        class Images:
            def get_input(self, index):
                return index
        subset = argparse.Namespace(indices=[7, 2], dataset=Images())
        wrapped = FrozenConceptTransferSubset(subset, lambda x: x, {"sample_ids": ["waterbirds:2", "waterbirds:7"]})
        self.assertEqual(wrapped[0], (7, 1))
        self.assertEqual(wrapped[1], (2, 0))
        with self.assertRaisesRegex(ValueError, "Missing frozen target"):
            FrozenConceptTransferSubset(subset, lambda x: x, {"sample_ids": ["waterbirds:2"]})

    def test_transfer_loss_uses_valid_rows_and_updates_encoder_and_head(self):
        torch.manual_seed(12)
        model = TinyModel()
        targets = {"reconstruction": torch.randn(3, 5), "valid_mask": torch.tensor([True, False, True])}
        regularizer = ConceptDistillationRegularizer(targets, "reconstruction", .1, 10, 10)
        regularizer.set_epoch(20)
        views = [torch.randn(3, 3), torch.randn(3, 3)]
        loss, parts, _ = simclr_forward_loss(model, SimCLRLoss(.05), views, regularizer, torch.arange(3))
        prediction = model.clip_distillation_head(parts["_embeddings"])
        valid = torch.tensor([True, False, True, True, False, True])
        expected = .1 * (1 - torch.nn.functional.cosine_similarity(prediction, targets["reconstruction"].repeat(2, 1)))[valid].mean()
        torch.testing.assert_close(parts["splice"], expected)
        parts["splice"].backward()
        self.assertGreater(float(model.encoder[0].weight.grad.norm()), 0)
        self.assertGreater(float(model.clip_distillation_head.weight.grad.norm()), 0)
        self.assertTrue(torch.isfinite(loss))
        regularizer.set_epoch(10)
        self.assertEqual(float(regularizer(prediction, targets["reconstruction"].repeat(2, 1), targets["valid_mask"]).detach()), 0)

    def test_la_ssl_upsamples_slower_examples(self):
        probabilities = sampling_probabilities(torch.tensor([1., 2., 3., 4.]), .5, 2.)
        torch.testing.assert_close(probabilities, torch.tensor([5.5, 3.5, 1.5, 0.], dtype=torch.float64) / 10.5)
        torch.testing.assert_close(sampling_probabilities(torch.ones(6), .1, 10), torch.ones(6, dtype=torch.float64) / 6)

    def test_la_ssl_scoring_is_observational_and_resume_restores_sampling(self):
        args = argparse.Namespace(seed=2, la_ssl_eta=.1, la_ssl_gamma=10., la_ssl_quantile=.1,
                                  la_ssl_warmup_epochs=0, la_ssl_update_freq=1)
        def make_loader():
            base = DataLoader(TwoViews(), batch_size=3, generator=torch.Generator().manual_seed(2))
            return build_la_ssl_loader(base, args, seed_worker)
        loader = make_loader()
        model = TinyModel().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with preserve_rng_state():
            metrics = loader.la_ssl.refresh(model, "cpu", .05, 1)
        torch.testing.assert_close(rng, torch.get_rng_state())
        self.assertTrue(model.training)
        self.assertEqual(metrics["la_ssl_score_examples"], 6)
        self.assertTrue(loader.sampler.replacement)
        for key, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[key])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "state.pth")
            save_checkpoint(model, optimizer, argparse.Namespace(), 1, path,
                            loader_generator=loader.generator, training_state=loader.la_ssl)
            expected = list(iter(loader.sampler))
            restored = make_loader()
            load_checkpoint(model, optimizer, path, torch.device("cpu"),
                            loader_generator=restored.generator, training_state=restored.la_ssl)
            self.assertEqual(expected, list(iter(restored.sampler)))
            torch.testing.assert_close(loader.la_ssl.ema, restored.la_ssl.ema)
            self.assertEqual(restored.la_ssl.epoch, 1)
            with self.assertRaisesRegex(ValueError, "without its state"):
                load_checkpoint(model, optimizer, path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
