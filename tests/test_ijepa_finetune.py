"""Partial I-JEPA fine-tuning trains only the last blocks and the head and matches the frozen features."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from cospro.methods.ijepa_finetune import PartialFineTune, parameter_groups, trainable_state, warmup_cosine
from cospro.models.ijepa import IJEPAEncoder, IJEPASpec

TINY = IJEPASpec("tiny", image_size=28, patch_size=14, embed_dim=16, depth=4, num_heads=2)


def tiny_model(trainable_blocks: int = 2, checkpointing: bool = False) -> PartialFineTune:
    torch.manual_seed(0)
    return PartialFineTune(IJEPAEncoder(TINY), num_classes=2, trainable_blocks=trainable_blocks,
                           gradient_checkpointing=checkpointing)


class PartialFineTuneTests(unittest.TestCase):
    def test_features_match_the_frozen_encoder_and_the_head_starts_at_zero(self):
        model = tiny_model()
        images = torch.randn(3, 3, 28, 28)
        torch.testing.assert_close(model.features(images), model.encoder(images))
        self.assertTrue(torch.equal(model(images), torch.zeros(3, 2)))

    def test_only_the_last_blocks_the_norm_and_the_head_train(self):
        model = tiny_model(trainable_blocks=2)
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(all(name.startswith(("encoder.blocks.2.", "encoder.blocks.3.", "encoder.norm.", "head."))
                            for name in names))
        self.assertIn("encoder.blocks.3.attn.qkv.weight", names)
        self.assertEqual(set(trainable_state(model)), names)
        before = {name: parameter.clone() for name, parameter in model.named_parameters()}
        optimizer = torch.optim.AdamW(parameter_groups(model, 1e-2, 1e-2, 0.05))
        images, labels = torch.randn(4, 3, 28, 28), torch.tensor([0, 1, 0, 1])
        for _ in range(2):  # the zero head passes no gradient to the encoder on the first step
            optimizer.zero_grad()
            F.cross_entropy(model.train()(images), labels).backward()
            optimizer.step()
        for name, parameter in model.named_parameters():
            changed = not torch.equal(parameter, before[name])
            self.assertEqual(changed, name in names, name)

    def test_gradient_checkpointing_gives_the_same_gradients(self):
        images, labels = torch.randn(4, 3, 28, 28), torch.tensor([0, 1, 1, 0])
        gradients = []
        for checkpointing in (False, True):
            model = tiny_model(checkpointing=checkpointing).train()
            torch.nn.init.normal_(model.head.weight, generator=torch.Generator().manual_seed(1))
            F.cross_entropy(model(images), labels).backward()
            gradients.append(model.encoder.blocks[2].mlp.fc1.weight.grad)
        torch.testing.assert_close(gradients[0], gradients[1])

    def test_the_frozen_prefix_stays_in_eval_mode(self):
        model = tiny_model().train()
        self.assertFalse(model.encoder.blocks[0].training)
        self.assertTrue(model.encoder.blocks[3].training)

    def test_parameter_groups_decay_only_trainable_matrices(self):
        model = tiny_model()
        decay, no_decay, head = parameter_groups(model, 1e-5, 1e-3, 0.05)
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


if __name__ == "__main__":
    unittest.main()
