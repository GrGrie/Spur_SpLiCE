"""LateTVG: the pruned second view, its options and its storage identity."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import torch

import spur_splice
from cospro.config.options import ConfigError
from cospro.models.simclr import SimCLRModel
from cospro.training.contrastive import SimCLRLoss
from cospro.training.late_pruning import LatePruning
from cospro.training.ssl_loop import simclr_forward_loss
from tests.test_storage_identity import storage_name


def normalized(argv: list[str]):
    with patch.object(sys, "argv", ["spur_splice.py"]), patch("torch.cuda.is_available", return_value=False):
        parser = spur_splice.build_training_parser()
        args = spur_splice.parse_training_arguments(parser, argv)
        return spur_splice.normalize_training_options(args)


class LatePruningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = SimCLRModel("resnet18")

    def test_masks_cover_the_last_convolutions_at_the_pruning_rate(self):
        pruning = LatePruning(0.7, 3)
        masks = pruning.masks(self.model.encoder)
        self.assertEqual(list(masks), ["layer4.0.shortcut.0.weight", "layer4.1.conv1.weight", "layer4.1.conv2.weight"])
        self.assertAlmostEqual(pruning.last_kept_fraction, 0.3, places=3)

    def test_pruned_view_trains_only_the_kept_weights_and_leaves_the_encoder_intact(self):
        pruning = LatePruning(0.9, 2)
        weight = self.model.encoder.layer4[1].conv2.weight
        before = weight.detach().clone()
        mask = pruning.masks(self.model.encoder)["layer4.1.conv2.weight"]
        views = [torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)]
        loss, _, _ = simclr_forward_loss(self.model, SimCLRLoss(0.1), views, late_pruning=pruning)
        loss.backward()
        self.assertTrue(torch.equal(weight.detach(), before))
        self.assertGreater(float(weight.grad[mask].abs().sum()), 0.0)
        # Pruned weights still learn from the full-encoder view 1.
        self.assertGreater(float(weight.grad[~mask].abs().sum()), 0.0)

    def test_pruned_view_leaves_batch_norm_statistics_untouched(self):
        before = {name: buffer.clone() for name, buffer in self.model.encoder.named_buffers()}
        LatePruning(0.9, 5)(self.model.encoder, torch.randn(4, 3, 32, 32))
        for name, buffer in self.model.encoder.named_buffers():
            self.assertTrue(torch.equal(buffer, before[name]), name)

    def test_pruned_view_differs_from_the_full_view(self):
        self.model.eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            full = self.model.encoder(images)
            pruned = LatePruning(0.9, 5)(self.model.encoder, images)
        self.assertFalse(torch.allclose(full, pruned))


class LateTVGOptionTests(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertEqual(normalized([]).latetvg_prune_rate, 0.0)

    def test_invalid_options_are_rejected(self):
        for argv in (
            ["--latetvg_prune_rate", "1.0"],
            ["--latetvg_prune_rate", "0.5", "--latetvg_layers", "0"],
            ["--latetvg_prune_rate", "0.5", "--la_ssl"],
        ):
            with self.subTest(argv=argv), self.assertRaises(ConfigError):
                normalized(argv)

    def test_enabled_latetvg_gets_its_own_storage_name(self):
        name = storage_name(["--splice_mode", "none", "--latetvg_prune_rate", "0.7"])
        self.assertTrue(name.startswith("waterbirds_s3_base-latetvg_e500_"))
        self.assertNotEqual(name, storage_name(["--splice_mode", "none", "--latetvg_prune_rate", "0.5"]))


if __name__ == "__main__":
    unittest.main()
