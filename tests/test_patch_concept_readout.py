"""The patch-concept readout recovers planted concepts, keeps probabilities summing to one and splits margins."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from cospro.evaluation.logistic_probe import fit_logistic_probe
from cospro.evaluation.patch_concept_readout import (
    Moments, concept_decision_shares, cross_fit_readout, image_folds, margin_direction, mean_pool,
    pooled_control, predict_concepts, resize_maps, shuffled_control,
)


def planted(images: int = 60, patches: int = 16, dim: int = 12, factors: int = 3, seed: int = 0):
    """Tokens whose patches point along one of ``factors`` directions, with soft one-hot maps."""

    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(factors, (images, patches), generator=generator)
    basis = F.normalize(torch.randn(factors, dim, generator=generator), dim=1)
    tokens = basis[labels] + 0.05 * torch.randn(images, patches, dim, generator=generator)
    targets = F.one_hot(labels, factors).float() * 0.9 + 0.1 / factors
    return tokens.half(), targets / targets.sum(-1, keepdim=True)


class PatchConceptReadoutTests(unittest.TestCase):
    def test_resize_keeps_probabilities(self):
        maps = torch.rand(2, 5, 14, 14).softmax(dim=1)
        resized = resize_maps(maps, 16)
        self.assertEqual(tuple(resized.shape), (2, 256, 5))
        torch.testing.assert_close(resized.sum(-1), torch.ones(2, 256))

    def test_folds_are_balanced_and_seeded(self):
        folds = image_folds(23, 5, seed=3)
        self.assertEqual(sorted(torch.bincount(folds).tolist()), [4, 4, 5, 5, 5])
        self.assertTrue(torch.equal(folds, image_folds(23, 5, seed=3)))

    def test_moments_solve_matches_least_squares(self):
        torch.manual_seed(0)
        x, y = torch.randn(50, 4), torch.randn(50, 2)
        weights, intercept = Moments.of(x, y).solve(1e-9)
        design = torch.cat([x, torch.ones(50, 1)], dim=1).double()
        reference = torch.linalg.lstsq(design, y.double()).solution
        torch.testing.assert_close(weights, reference[:4])
        torch.testing.assert_close(intercept, reference[4])

    def test_planted_concepts_are_read_and_controls_fall_below(self):
        tokens, targets = planted()
        folds = image_folds(len(tokens), 4)
        readout = cross_fit_readout(tokens, targets, folds, [0.01, 1.0], chunk=7)
        self.assertGreater(readout["explained_variance"], 0.9)
        self.assertEqual(tuple(readout["oof_concept_vectors"].shape), (60, 3))
        best = readout["best_ridge"]
        self.assertLess(pooled_control(tokens, targets, folds, best), 0.5)
        self.assertLess(shuffled_control(tokens, targets, folds, best), 0.1)

    def test_predicted_concepts_sum_to_one(self):
        tokens, targets = planted()
        readout = cross_fit_readout(tokens, targets, image_folds(len(tokens), 3), [1.0])
        predicted = predict_concepts(tokens, readout["full_fit"])
        torch.testing.assert_close(predicted.sum(-1), torch.ones(60, 16), atol=1e-4, rtol=0)

    def test_shares_add_up_to_the_margin(self):
        tokens, _ = planted()
        pooled = mean_pool(tokens)
        labels = (pooled[:, 0] > pooled[:, 0].median()).long()
        parameters: dict = {}
        fit_logistic_probe(pooled, labels, pooled, labels, num_classes=2, l2=0.1, parameters=parameters)
        direction, offset = margin_direction(parameters["weight"].float(), parameters["scale"].float(),
                                             parameters["mean"].float())
        concepts = torch.rand(60, 16, 3).softmax(dim=-1)
        shares = concept_decision_shares(tokens, concepts, direction, offset)
        weight = parameters["weight"].float()
        logits = ((pooled - parameters["mean"].float()) / parameters["scale"].float()) @ weight.T
        torch.testing.assert_close(shares.sum(-1), logits[:, 1] - logits[:, 0], atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
