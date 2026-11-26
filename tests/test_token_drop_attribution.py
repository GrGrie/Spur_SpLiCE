"""Token removal is exact in the I-JEPA ViT, and the attribution helpers count and score correctly."""

from __future__ import annotations

import unittest

import torch

from cospro.evaluation.token_drop_attribution import (
    concept_units, drop_masks, group_accuracy, hard_regions, keep_pooled, paired_effect, presence,
    region_agreement, shortcut_index, shuffled_regions, spearman, unit_maps, unit_typing,
)
from cospro.models.ijepa import IJEPAEncoder, IJEPASpec


def tiny_encoder() -> IJEPAEncoder:
    torch.manual_seed(0)
    encoder = IJEPAEncoder(IJEPASpec("tiny", image_size=28, patch_size=7, embed_dim=16, depth=3, num_heads=2))
    with torch.no_grad():
        encoder.pos_embed.normal_()
    return encoder.eval()


class TokenRemovalTests(unittest.TestCase):
    def test_masked_tokens_equal_an_encoder_without_those_patches(self):
        encoder = tiny_encoder()
        images = torch.randn(2, 3, 28, 28)
        keep = torch.ones(2, 16, dtype=torch.bool)
        keep[0, [1, 5, 6]] = False
        keep[1, 8:] = False
        with torch.no_grad():
            masked = encoder.patch_tokens(images, keep)
            for row in range(2):
                x = (encoder.patch_embed(images[row:row + 1]) + encoder.pos_embed)[:, keep[row]]
                for block in encoder.blocks:
                    x = block(x)
                torch.testing.assert_close(masked[row:row + 1, keep[row]], encoder.norm(x), atol=1e-5, rtol=1e-5)

    def test_full_keep_matches_the_plain_forward(self):
        encoder = tiny_encoder()
        images = torch.randn(3, 3, 28, 28)
        with torch.no_grad():
            torch.testing.assert_close(encoder.patch_tokens(images, torch.ones(3, 16, dtype=torch.bool)),
                                       encoder.patch_tokens(images))

    def test_keep_pooled_averages_kept_patches(self):
        tokens = torch.arange(8.0).reshape(1, 4, 2)
        keep = torch.tensor([[True, False, True, False]])
        torch.testing.assert_close(keep_pooled(tokens, keep), torch.tensor([[2.0, 3.0]]))
        torch.testing.assert_close(keep_pooled(tokens, None), tokens.mean(dim=1))


class UnitTests(unittest.TestCase):
    def test_units_group_factors_that_share_cells(self):
        # Factors 0 and 1 cover the left half together, factors 2 and 3 the right half.
        maps = torch.zeros(10, 4, 4, 4)
        maps[:, 0, :, :2] = maps[:, 1, :, :2] = 0.5
        maps[:, 2, :, 2:] = maps[:, 3, :, 2:] = 0.5
        units = concept_units(maps, 2)
        self.assertEqual(units[0], units[1])
        self.assertEqual(units[2], units[3])
        self.assertNotEqual(units[0], units[2])
        self.assertTrue(torch.equal(concept_units(maps, 0), torch.arange(4)))

    def test_regions_partition_patches_and_presence_counts(self):
        mass = unit_maps(torch.rand(5, 196, 6).softmax(-1), torch.tensor([0, 0, 1, 1, 2, 2]))
        regions = hard_regions(mass)
        self.assertEqual(tuple(regions.shape), (5, 196, 3))
        self.assertTrue(torch.equal(regions.sum(-1), torch.ones(5, 196, dtype=torch.long)))
        small = torch.zeros(1, 196, 2, dtype=torch.bool)
        small[0, :2, 0], small[0, 2:, 1] = True, True
        self.assertEqual(presence(small).tolist(), [[False, True]])

    def test_drop_masks_and_shuffle(self):
        regions = torch.zeros(4, 10, 2, dtype=torch.bool)
        regions[:, :, 1] = True
        for row, cells in ((0, slice(0, 3)), (1, slice(0, 10)), (2, slice(5, 9))):
            regions[row, cells, 0], regions[row, cells, 1] = True, False
        keep, valid = drop_masks(regions[:, :, 0], presence(regions, 0.2)[:, 0])
        # Image 1 would lose every patch and image 3 shows no unit 0.
        self.assertEqual(valid.tolist(), [True, False, True, False])
        self.assertTrue(torch.equal(keep, ~regions[:, :, 0]))
        shuffled = shuffled_regions(regions, 0, seed=1)
        for row in torch.nonzero(presence(regions)[:, 0]).flatten().tolist():
            self.assertFalse(torch.equal(shuffled[row], regions[row, :, 0]))
        self.assertTrue(torch.equal(shuffled[3], regions[3, :, 0]))


class ScoringTests(unittest.TestCase):
    def test_group_accuracy_and_paired_effect(self):
        labels = torch.tensor([0, 0, 1, 1, 0, 1])
        groups = torch.tensor([0, 1, 2, 3, 1, 2])
        right = torch.stack([1 - labels, labels], dim=1).float()
        torch.testing.assert_close(group_accuracy(right.argmax(1), labels, groups),
                                   torch.full((4,), 100.0, dtype=torch.float64))
        occluded = right.clone()
        occluded[1] = right[1].flip(0)
        effect = paired_effect(right, occluded, labels, groups, resamples=50)
        self.assertEqual(effect["wga_before"], 100.0)
        self.assertEqual(effect["wga_after"], 50.0)
        self.assertEqual(effect["wga_effect"], -50.0)
        self.assertEqual(effect["changed_predictions"], 1)
        self.assertLess(effect["true_class_margin_change"][1], 0)
        self.assertEqual(effect["true_class_margin_change"][0], 0)

    def test_shortcut_index_rewards_freeing_conflict_groups(self):
        # Groups 0..3, ten images each; class is group // 2, conflict groups 1 and 2.
        groups = torch.arange(4).repeat_interleave(10)
        labels = groups // 2
        right = torch.stack([1 - labels, labels], dim=1).float() * 2 - 1
        intact = right.clone()
        intact[groups == 1] *= -1  # the shortcut fools the conflict images
        intact[groups == 2] *= -1
        removed = right.clone()  # removal frees them...
        removed[torch.tensor([0, 1, 30, 31])] *= -1  # ...and costs a few aligned ones
        control = intact.clone()
        result = shortcut_index(intact, removed, control, labels, groups, (1, 2), resamples=200)
        self.assertEqual(result["aligned_groups"], [0, 3])
        self.assertAlmostEqual(result["removed_accuracy"], 100.0 + 20.0)
        self.assertAlmostEqual(result["net_accuracy"], 120.0)
        self.assertGreater(result["net_accuracy_ci95"][0], 0)
        self.assertEqual(result["verdict"], "shortcut")
        self.assertEqual(result["verdict_from"], "net_accuracy")
        # A core concept is what the conflict images get right with: removing it costs them first.
        core = right.clone()
        core[groups == 1] *= -1
        core[groups == 2] *= -1
        needed = shortcut_index(right, core, None, labels, groups, (1, 2), resamples=50)
        self.assertLess(needed["removed_accuracy"], 0)
        self.assertEqual(needed["verdict"], "needed")
        self.assertNotIn("net_accuracy", needed)
        same = shortcut_index(intact, removed, removed, labels, groups, (1, 2), resamples=50)
        self.assertEqual(same["net_accuracy"], 0.0)
        self.assertEqual(same["verdict"], "neutral")

    def test_typing_spearman_and_agreement(self):
        labels = torch.tensor([0, 0, 1, 1] * 25)
        attributes = torch.tensor([0, 1, 0, 1] * 25)
        present = torch.stack([attributes.bool(), labels.bool(), torch.ones(100, dtype=torch.bool)], dim=1)
        kinds = [entry["type"] for entry in unit_typing(present, labels, attributes)]
        self.assertEqual(kinds, ["attribute", "class", "untyped"])
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        regions = hard_regions(torch.rand(3, 8, 2))
        self.assertEqual(region_agreement(regions, regions)["patch_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
