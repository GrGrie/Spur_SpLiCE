"""Partial fine-tuning trains only the last blocks and the head, and the held-out halves stay disjoint."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torchvision import models

from cospro.evaluation.logistic_probe import fit_logistic_probe
from cospro.methods.partial_finetune import (
    IJEPABackbone, PartialFineTune, ResNetBackbone, held_out_halves, logistic_logits, mean_and_spread,
    parameter_groups, trainable_state, warmup_cosine,
)
from cospro.models.ijepa import IJEPAEncoder, IJEPASpec

TINY = IJEPASpec("tiny", image_size=28, patch_size=14, embed_dim=16, depth=4, num_heads=2)


def tiny_model(trainable_blocks: int = 2, checkpointing: bool = False) -> PartialFineTune:
    torch.manual_seed(0)
    return PartialFineTune(IJEPABackbone(IJEPAEncoder(TINY)), num_classes=2, trainable_blocks=trainable_blocks,
                           gradient_checkpointing=checkpointing)


def resnet_model(trainable_blocks: int = 3) -> PartialFineTune:
    torch.manual_seed(0)
    resnet = models.resnet50(weights=None)
    resnet.fc = torch.nn.Identity()
    return PartialFineTune(ResNetBackbone(resnet, 2048), num_classes=2, trainable_blocks=trainable_blocks)


class PartialFineTuneTests(unittest.TestCase):
    def test_features_match_the_frozen_encoder_and_the_head_starts_at_zero(self):
        torch.manual_seed(0)
        encoder = IJEPAEncoder(TINY)
        model = PartialFineTune(IJEPABackbone(encoder), num_classes=2, trainable_blocks=2)
        images = torch.randn(3, 3, 28, 28)
        torch.testing.assert_close(model.features(images), encoder(images))
        self.assertTrue(torch.equal(model(images), torch.zeros(3, 2)))

    def test_only_the_last_blocks_the_norm_and_the_head_train(self):
        model = tiny_model(trainable_blocks=2)
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(all(name.startswith(("backbone.blocks.2.", "backbone.blocks.3.", "backbone.tail.", "head."))
                            for name in names))
        self.assertIn("backbone.blocks.3.attn.qkv.weight", names)
        self.assertEqual(set(trainable_state(model)), names)
        before = {name: parameter.clone() for name, parameter in model.named_parameters()}
        optimizer = torch.optim.AdamW(parameter_groups(model, 1e-2, 1e-2, 0.05))
        images, labels = torch.randn(4, 3, 28, 28), torch.tensor([0, 1, 0, 1])
        for _ in range(2):  # the zero head passes no gradient to the encoder on the first step
            optimizer.zero_grad()
            F.cross_entropy(model.train()(images), labels).backward()
            optimizer.step()
        for name, parameter in model.named_parameters():
            self.assertEqual(not torch.equal(parameter, before[name]), name in names, name)

    def test_gradient_checkpointing_gives_the_same_gradients(self):
        images, labels = torch.randn(4, 3, 28, 28), torch.tensor([0, 1, 1, 0])
        gradients = []
        for checkpointing in (False, True):
            model = tiny_model(checkpointing=checkpointing).train()
            torch.nn.init.normal_(model.head.weight, generator=torch.Generator().manual_seed(1))
            F.cross_entropy(model(images), labels).backward()
            gradients.append(model.backbone.blocks[2].mlp.fc1.weight.grad)
        torch.testing.assert_close(gradients[0], gradients[1])

    def test_the_frozen_prefix_stays_in_eval_mode(self):
        model = tiny_model().train()
        self.assertFalse(model.backbone.blocks[0].training)
        self.assertTrue(model.backbone.blocks[3].training)

    def test_resnet_trains_its_last_bottlenecks_and_keeps_frozen_batch_norm_statistics(self):
        model = resnet_model(trainable_blocks=3).train()
        self.assertEqual(len(model.backbone.blocks), 16)
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(all(name.startswith(("backbone.blocks.13.", "backbone.blocks.14.", "backbone.blocks.15.",
                                             "head.")) for name in names))
        statistics = model.backbone.bn1.running_mean.clone(), model.backbone.blocks[0].bn1.running_mean.clone()
        self.assertEqual(tuple(model(torch.randn(2, 3, 64, 64)).shape), (2, 2))
        self.assertTrue(torch.equal(model.backbone.bn1.running_mean, statistics[0]))
        self.assertTrue(torch.equal(model.backbone.blocks[0].bn1.running_mean, statistics[1]))
        self.assertTrue(model.backbone.blocks[15].training)

    def test_parameter_groups_decay_only_trainable_matrices(self):
        decay, no_decay, head = parameter_groups(tiny_model(), 1e-5, 1e-3, 0.05)
        self.assertTrue(all(parameter.ndim >= 2 and parameter.requires_grad for parameter in decay["params"]))
        self.assertTrue(all(parameter.ndim == 1 for parameter in no_decay["params"]))
        self.assertEqual((decay["weight_decay"], no_decay["weight_decay"], head["lr"]), (0.05, 0.0, 1e-3))

    def test_warmup_then_cosine(self):
        self.assertAlmostEqual(warmup_cosine(0, 100, 10), 0.1)
        self.assertAlmostEqual(warmup_cosine(9, 100, 10), 1.0)
        self.assertAlmostEqual(warmup_cosine(10, 100, 10), 1.0)
        self.assertAlmostEqual(warmup_cosine(55, 100, 10), 0.5)
        self.assertAlmostEqual(warmup_cosine(100, 100, 10), 0.0)

    def test_trainable_blocks_must_fit_the_depth(self):
        with self.assertRaises(ValueError):
            tiny_model(trainable_blocks=5)


class HeadRefitTests(unittest.TestCase):
    def test_held_out_halves_are_disjoint_balanced_and_fixed(self):
        groups = torch.tensor([0] * 467 + [1] * 466 + [2] * 133 + [3] * 133)[torch.randperm(1199)]
        fit, holdout = held_out_halves(groups, seed=0)
        self.assertEqual(torch.bincount(groups[fit]).tolist(), [66, 66, 66, 66])
        self.assertEqual(torch.bincount(groups[holdout]).tolist(), [234, 233, 67, 67])
        self.assertEqual(len(set(fit.tolist()) & set(holdout.tolist())), 0)
        self.assertTrue(torch.equal(fit, held_out_halves(groups, seed=0)[0]))

    def test_logistic_logits_reproduce_the_probe_predictions(self):
        torch.manual_seed(0)
        x, y = torch.randn(40, 5), torch.randint(2, (40,))
        y[:2] = torch.tensor([0, 1])
        parameters: dict = {}
        records, _ = fit_logistic_probe(x, y, x, y, num_classes=2, l2=0.1, parameters=parameters)
        self.assertTrue(torch.equal(logistic_logits(parameters, x).argmax(1), records[-1].eval_predictions))

    def test_mean_and_spread(self):
        self.assertEqual(mean_and_spread([1.0, 3.0]), {"mean": 2.0, "std": 1.41})
        self.assertEqual(mean_and_spread([5.0]), {"mean": 5.0, "std": 0.0})


if __name__ == "__main__":
    unittest.main()
