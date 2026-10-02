"""Concept factors: discovery from SpLiCE codes, F1 conditioned batches and F2 factor distillation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from golden_support import synthetic_splice_cache, train_sample_ids, write_waterbirds_fixture
from test_golden_training import COMMON_ARGS, GRAPH_CONFIG
from cospro.methods.concept_factors import (
    ConceptBlockContrast,
    ConceptFactors,
    CrossFitDistillation,
    cross_fit_predictions,
    ConceptConditionedBatchSampler,
    FactorDistillationRegularizer,
)
from cospro.cli import build_meaning_groups, sweep_concept_factors
from cospro.cli.cache_splice_dataset import cache_config_name
from cospro.diagnostics.factor_validity import conditional_uncertainty, diagnose_factors
from cospro.pipeline import CoSpRoAuditConfig, build_concept_groups
from cospro.pipeline.grouping import group_by_meaning
from cospro.pipeline.concept_factors import (
    FactorConfig,
    build_concept_factors,
    factor_report,
    heldout_explained_variance,
    load_concept_factors,
    rows_for_subset,
    unseen_explained_variance,
    whitened,
)
from cospro.tracking.artifacts import PROJECT_ROOT
from unittest.mock import patch
import argparse
import numpy as np
from cospro.tracking.metrics import canonical_train_metrics
from tests.test_storage_identity import storage_name


def synthetic_inputs():
    sample_ids, y, place = train_sample_ids()
    cache = synthetic_splice_cache(sample_ids, y, place)
    return cache, build_concept_groups(cache, GRAPH_CONFIG)


class FactorDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.cache, self.groups = synthetic_inputs()
        self.factors = build_concept_factors(self.cache, self.groups, FactorConfig(min_correlation=0.3))

    def test_the_correlated_class_and_context_groups_form_the_top_pairs(self):
        pairs = factor_report(self.factors)["pairs"]
        self.assertEqual({tuple(pair["concepts"]) for pair in pairs}, {
            ("water | lake | ocean", "waterbird | gull"), ("forest", "sparrow"),
        })
        for pair in pairs:
            self.assertAlmostEqual(pair["phi"], 0.5, places=6)

    def test_whitening_decorrelates_the_entangled_factors(self):
        names = [" / ".join(factor["concepts"]) for factor in self.factors["factors"]]
        water, bird = names.index("water / lake / ocean"), names.index("waterbird / gull")
        raw = torch.corrcoef(self.factors["targets"]["standardized"].T)[water, bird]
        white = torch.corrcoef(self.factors["targets"]["whitened"].T)[water, bird]
        self.assertGreater(float(raw), 0.4)
        self.assertLess(abs(float(white)), 0.1)
        variance = self.factors["targets"]["whitened"].var(dim=0, unbiased=False)
        self.assertTrue(torch.allclose(variance, torch.ones_like(variance), atol=1e-4))

    def test_near_synonyms_never_pair(self):
        factors = build_concept_factors(
            self.cache, self.groups, FactorConfig(min_correlation=0.3, max_text_similarity=-1.0),
        )
        self.assertEqual(factors["pairs"], [])

    def test_groups_from_another_cache_are_rejected(self):
        other = {**self.groups, "sample_ids": list(reversed(self.groups["sample_ids"]))}
        with self.assertRaises(ValueError):
            build_concept_factors(self.cache, other, FactorConfig())

    def test_rows_follow_the_training_subset(self):
        rows = rows_for_subset(["waterbirds:4", "waterbirds:9"], "waterbirds", [9, 4])
        self.assertEqual(rows.tolist(), [1, 0])
        with self.assertRaises(ValueError):
            rows_for_subset(["waterbirds:4"], "waterbirds", [5])


class TargetKindTests(unittest.TestCase):
    def test_shuffled_targets_keep_the_statistics_and_lose_the_images(self):
        cache, groups = synthetic_inputs()
        targets = build_concept_factors(cache, groups, FactorConfig())["targets"]
        standard, shuffled = targets["standardized"], targets["shuffled"]
        self.assertTrue(torch.allclose(standard.sort(dim=0).values, shuffled.sort(dim=0).values))
        self.assertFalse(torch.allclose(standard, shuffled))
        presence = targets["presence"]
        self.assertEqual(presence.shape, standard.shape)
        self.assertTrue(all(len(torch.unique(column)) <= 2 for column in presence.T))


class StudySummaryTests(unittest.TestCase):
    def test_runs_of_a_study_are_summarized_per_arm(self):
        from cospro.cli import summarize_study

        def record(seed, wga):
            probes = [{"stage": "linear_probe", "values": {"ssl_epoch": epoch, "eval_worst_group_accuracy": wga + epoch / 100,
                                                        "eval_accuracy": 60.0}} for epoch in (25, 50, 75, 100, 125)]
            ssl = [{"stage": "ssl", "values": {"SSL relational_factor_explained_variance": 0.4}}]
            return {"status": "complete", "config": {"seed": seed, "dataset": "metashift", "factor_targets": "standardized"},
                    "final_metrics": {"Last linear val worst-group acc": wga + 1.25}, "metrics": probes + ssl}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, wga in ((1, 40.0), (2, 50.0)):
                path = root / "seeds" / "study" / f"seed_{seed:02d}" / "f2_std" / "123" / "run.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(record(seed, wga)), encoding="utf-8")
            with patch.dict(os.environ, {"SPUR_SPLICE_OUTPUT_ROOT": str(root)}):
                markdown = summarize_study.main(["--study", "study"])
            summary = json.loads((markdown.parent / "summary.json").read_text(encoding="utf-8"))
            arm = summary["arms"][0]
            self.assertEqual((arm["arm"], arm["seeds"]), ("f2_std", [1, 2]))
            # The last four probes are epochs 50 to 125: wga + 0.875 on average.
            self.assertAlmostEqual(arm["val_wga_last4"]["mean"], 45.875)
            self.assertAlmostEqual(arm["factor_explained_variance"]["mean"], 0.4)
            self.assertIn("| f2_std | 1,2 |", markdown.read_text(encoding="utf-8"))


class AtypicalityTests(unittest.TestCase):
    def test_label_free_weights_favour_the_minority_groups(self):
        from cospro.diagnostics.factor_validity import weight_by_group

        cache, groups = synthetic_inputs()
        factors = build_concept_factors(cache, groups, FactorConfig())
        weights = factors["atypicality_weights"]
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)
        _, y, place = train_sample_ids()
        self.assertGreater(weight_by_group(weights.numpy(), y, place)["minority_to_majority_ratio"], 1.5)

    def test_weighted_loss_counts_heavy_images_more(self):
        targets = torch.zeros(2, 1)
        predictions = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        plain = FactorDistillationRegularizer(targets, 1.0, 0, 0)
        weighted = FactorDistillationRegularizer(targets, 1.0, 0, 0, sample_weights=torch.tensor([3.0, 1.0]))
        for regularizer in (plain, weighted):
            regularizer.set_epoch(1)
        self.assertAlmostEqual(float(plain(predictions, torch.tensor([0, 1]))), 0.5)
        self.assertAlmostEqual(float(weighted(predictions, torch.tensor([0, 1]))), 1.5)
        self.assertAlmostEqual(weighted.last_diagnostics["factor_mse"], 0.5)


class CrossFitTests(unittest.TestCase):
    def test_the_kernel_form_equals_the_weighted_primal_ridge(self):
        torch.manual_seed(0)
        features, targets = torch.randn(10, 6), torch.randn(10, 2)
        weights = torch.rand(10) + 0.5
        fit = torch.arange(10) < 6
        predicted = cross_fit_predictions(features, targets, weights, fit, 0.5)
        unit = torch.nn.functional.normalize(features, dim=1)
        h, t, w = unit[fit], targets[fit], weights[fit] / weights[fit].mean()
        mean_h, mean_t = (w[:, None] * h).mean(0), (w[:, None] * t).mean(0)
        hc, tc = h - mean_h, t - mean_t
        primal = torch.linalg.solve(hc.T @ torch.diag(w) @ hc + 0.5 * torch.eye(6), hc.T @ torch.diag(w) @ tc)
        expected = (unit[~fit] - mean_h) @ primal + mean_t
        self.assertTrue(torch.allclose(predicted, expected, atol=1e-4))

    def test_shrinking_the_features_cannot_weaken_the_ridge(self):
        torch.manual_seed(1)
        features, targets = torch.randn(12, 8), torch.randn(12, 1)
        fit = torch.arange(12) < 6
        weights = torch.ones(12)
        large = cross_fit_predictions(features, targets, weights, fit, 1.0)
        small = cross_fit_predictions(features * 1e-4, targets, weights, fit, 1.0)
        self.assertTrue(torch.allclose(large, small, atol=1e-4))

    def test_memorized_image_identities_do_not_lower_the_held_out_loss(self):
        # Features that only identify images (one-hot per image) explain nothing across halves;
        # features that carry the concept do.
        torch.manual_seed(0)
        count = 64
        targets = torch.randn(count, 1)
        distillation = CrossFitDistillation(targets, 1.0, 0, 0, ridge=0.01)
        distillation.set_epoch(1)
        identity = torch.eye(count)
        # The concept rides on a constant component, so unit-norm rows keep its value.
        concept = torch.cat([0.2 * targets, torch.ones(count, 1), torch.randn(count, 2) * 0.001], dim=1)
        indices = torch.arange(count)
        identity_loss = float(distillation(torch.cat([identity, identity]), indices))
        concept_loss = float(distillation(torch.cat([concept, concept]), indices))
        self.assertGreater(identity_loss, 0.5)
        self.assertLess(concept_loss, 0.2)
        self.assertGreater(distillation.last_diagnostics["factor_heldout_explained_variance"], 0.8)

    def test_gradients_reach_the_backbone_through_the_solve(self):
        targets = torch.randn(8, 2)
        distillation = CrossFitDistillation(targets, 1.0, 0, 0, sample_weights=torch.rand(8) + 0.5)
        distillation.set_epoch(1)
        backbone = torch.nn.Linear(5, 4)
        embeddings = backbone(torch.randn(8, 5))
        distillation(torch.cat([embeddings, embeddings]), torch.arange(8)).backward()
        self.assertGreater(float(backbone.weight.grad.norm()), 0.0)


class BalancingWeightTests(unittest.TestCase):
    def test_weights_decorrelate_the_factors_and_favour_the_minority_groups(self):
        from cospro.diagnostics.factor_validity import weight_by_group
        from cospro.pipeline.concept_factors import balancing_weights

        cache, groups = synthetic_inputs()
        factors = build_concept_factors(cache, groups, FactorConfig())
        weights, diagnostics = balancing_weights(factors["active"])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)
        self.assertLess(diagnostics["mean_abs_correlation_after"], 0.7 * diagnostics["mean_abs_correlation_before"])
        _, y, place = train_sample_ids()
        self.assertGreater(weight_by_group(weights.numpy(), y, place)["minority_to_majority_ratio"], 2.0)


class ConceptBlockTests(unittest.TestCase):
    # Factors: cat, dog, sofa, street. Images: cat+sofa, cat+street, dog+sofa, dog+street.
    PRESENCE = torch.tensor([[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1]], dtype=torch.bool)

    def blocks(self, context_weight=1.0):
        blocks = ConceptBlockContrast(self.PRESENCE, 1.0, 0.1, context_weight, 0, 0)
        blocks.set_epoch(1)
        return blocks

    def test_a_pair_is_positive_only_in_the_blocks_of_the_concepts_it_shares(self):
        weights = self.blocks().pair_weights(torch.cat([self.PRESENCE, self.PRESENCE]))
        cat, street = 0, 3
        cat_sofa, cat_street, dog_street = 0, 1, 3
        self.assertGreater(float(weights[cat, cat_sofa, cat_street]), 0.0)
        self.assertEqual(float(weights[cat, cat_street, dog_street]), 0.0)
        self.assertGreater(float(weights[street, cat_street, dog_street]), 0.0)
        # The other view of the same image is a positive with the base weight; cross-context pairs weigh more.
        self.assertAlmostEqual(float(weights[cat, cat_sofa, cat_sofa + 4]), 1.0)
        self.assertGreater(float(weights[cat, cat_sofa, cat_street]), 1.0)

    def test_separate_concept_directions_beat_one_fused_direction(self):
        # Fused: one axis "cat + indoor" shared by every block. Decoupled: each block reads its own concept.
        cat = torch.tensor([1.0, 1.0, -1.0, -1.0])
        sofa = torch.tensor([1.0, -1.0, 1.0, -1.0])
        fused_axis = (cat + sofa) / 2
        fused = torch.stack([fused_axis, -fused_axis, fused_axis, -fused_axis], dim=1)
        decoupled = torch.stack([cat, -cat, sofa, -sofa], dim=1)

        def loss(values):
            embedded = torch.stack([values, torch.ones_like(values)], dim=-1)
            embedded = torch.cat([embedded, embedded])
            return float(self.blocks()(embedded, torch.arange(4)))

        self.assertLess(loss(decoupled), loss(fused))


class ResultsBookTests(unittest.TestCase):
    def test_one_page_per_dataset_and_method_with_a_simclr_reference(self):
        from cospro.cli import build_results_book

        def record(seed, mode, wga, **config):
            probes = [{"stage": "linear_probe", "values": {"ssl_epoch": epoch, "eval_worst_group_accuracy": wga,
                                                        "eval_accuracy": 60.0}} for epoch in (25, 50)]
            return {"status": "complete", "config": {"seed": seed, "dataset": "metashift", "splice_mode": mode,
                                                     "epochs": 500, **config},
                    "final_metrics": {"Last linear val worst-group acc": wga}, "metrics": probes}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {("factors", "simclr", 1): record(1, "none", 40.0), ("factors", "simclr", 2): record(2, "none", 42.0),
                    ("factors", "f2", 1): record(1, "concept_factors", 50.0, factor_distill_weight=3.0),
                    ("factors", "f2", 2): record(2, "concept_factors", 52.0, factor_distill_weight=3.0),
                    ("old", "cospro", 1): record(1, "crp_relational", 45.0, crp_teacher_graph="graphs/crp_graph.json")}
            for (study, arm, seed), payload in runs.items():
                path = root / "seeds" / study / f"seed_{seed:02d}" / arm / "1" / "run.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SPUR_SPLICE_OUTPUT_ROOT": str(root)}):
                index = build_results_book.main([])
            book = index.parent
            factors_page = (book / "metashift" / "concept_factors.md").read_text(encoding="utf-8")
            self.assertIn("| factors | f2 |", factors_page)
            self.assertIn("51.0 ± 1.4", factors_page)
            self.assertIn("SimCLR reference", factors_page)
            self.assertIn("crp_graph", (book / "metashift" / "cospro.md").read_text(encoding="utf-8"))
            self.assertIn("factors / f2", index.read_text(encoding="utf-8"))


class RedundancyMergeTests(unittest.TestCase):
    def test_synonym_groups_merge_by_image_similarity_and_the_pair_survives(self):
        cache, _ = synthetic_inputs()
        # A strict text threshold leaves the three water synonyms in separate groups.
        split = build_concept_groups(cache, CoSpRoAuditConfig(text_similarity_threshold=0.999, coactivation_threshold=0.35))
        self.assertGreater(len(split["groups"]), 5)
        unmerged = build_concept_factors(cache, split, FactorConfig(min_correlation=0.3))
        merged = build_concept_factors(cache, split, FactorConfig(min_correlation=0.3, merge_similarity=0.9))
        self.assertLess(len(merged["factors"]), len(unmerged["factors"]))
        names = [" / ".join(sorted(factor["concepts"])) for factor in merged["factors"]]
        self.assertIn("lake / ocean / water", names)
        pairs = {tuple(sorted(pair["concepts"])) for pair in factor_report(merged)["pairs"]}
        self.assertTrue(any("gull" in " ".join(pair) and "water" in " ".join(pair) for pair in pairs))


class FactorValidityTests(unittest.TestCase):
    def test_class_and_attribute_factors_are_told_apart_under_correlation(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 4000)
        a = np.where(rng.random(4000) < 0.9, y, 1 - y)
        active = np.stack([(y == 1) & (rng.random(4000) < 0.9), (a == 1) & (rng.random(4000) < 0.9),
                           rng.random(4000) < 0.3], axis=1)
        pairs = [{"factors": [0, 1], "phi": 0.7}, {"factors": [0, 2], "phi": 0.1}]
        diagnosis = diagnose_factors(active, pairs, ["bird", "water", "noise"], y, a)
        self.assertEqual([factor["type"] for factor in diagnosis["factors"]], ["class", "attribute", "neither"])
        self.assertEqual([pair["verdict"] for pair in diagnosis["pairs"]], ["cross", "noise"])
        self.assertEqual(diagnosis["summary"]["pair_precision"], 0.5)

    def test_a_single_column_gives_a_scalar(self):
        y = np.array([0, 0, 1, 1]); a = np.array([0, 1, 0, 1])
        self.assertAlmostEqual(float(conditional_uncertainty(np.array([0, 0, 1, 1], bool), y, a)), 1.0)


class FactorLearnabilityTests(unittest.TestCase):
    def test_encoded_factors_score_high_and_unrelated_ones_near_zero(self):
        generator = torch.Generator().manual_seed(0)
        features = torch.randn(600, 32, generator=generator)
        targets = torch.stack([features[:, 0] + 0.1 * torch.randn(600, generator=generator),
                               torch.randn(600, generator=generator)], dim=1)
        scores = heldout_explained_variance(features, targets, ridge=1.0)
        self.assertGreater(float(scores[0]), 0.8)
        self.assertLess(abs(float(scores[1])), 0.1)

    def test_stored_targets_score_on_seen_images_only(self):
        # An encoder that stores each seen image's target, and nothing of the unseen images.
        generator = torch.Generator().manual_seed(2)
        targets = torch.randn(600, 3, generator=generator)
        unseen = torch.zeros(600, dtype=torch.bool)
        unseen[::5] = True
        features = torch.cat([torch.where(unseen[:, None], torch.randn(600, 3, generator=generator), targets),
                              0.1 * torch.randn(600, 13, generator=generator)], dim=1)
        self.assertGreater(float(heldout_explained_variance(features[~unseen], targets[~unseen], ridge=1.0).mean()), 0.8)
        self.assertLess(float(unseen_explained_variance(features, targets, unseen, ridge=1.0).mean()), 0.1)
        # Features that encode the factor everywhere carry over.
        encoded = torch.cat([targets, 0.1 * torch.randn(600, 13, generator=generator)], dim=1)
        self.assertGreater(float(unseen_explained_variance(encoded, targets, unseen, ridge=1.0).mean()), 0.8)

    @staticmethod
    def prepared(method, targets, clip_features, unseen=None):
        count = len(targets)
        method._prepare_learnability(
            {"targets": {"standardized": targets, "whitened": whitened(targets, 0.01)},
             "clip_embeddings": clip_features,
             "factors": [{"concepts": [name], "frequency": 0.5} for name in ("cat", "couch")]},
            torch.arange(count), list(range(100, 100 + count)),
            torch.zeros(count, dtype=torch.bool) if unseen is None else unseen,
        )

    def test_the_method_matches_features_to_factors_by_source_index(self):
        method = ConceptFactors()
        generator = torch.Generator().manual_seed(1)
        features = torch.randn(400, 16, generator=generator)
        self.prepared(method, features[:, :2].clone(), features)
        reverse = list(range(499, 99, -1))
        result = method.factor_learnability(features.flip(0), reverse)
        self.assertGreater(result["student_mean"], 0.8)
        self.assertEqual(result["learned_fraction"], 1.0)
        shuffled = method.factor_learnability(features, reverse)
        self.assertLess(shuffled["student_mean"], 0.1)

    def test_a_fused_direction_fails_the_residual_and_the_minority_groups(self):
        # Cat and couch co-occur in 90 percent of the images; group = 2 * cat + couch.
        generator = torch.Generator().manual_seed(3)
        cat = (torch.rand(2000, generator=generator) < 0.5).float()
        couch = torch.where(torch.rand(2000, generator=generator) < 0.9, cat, 1 - cat)
        targets = torch.stack([cat, couch], dim=1)
        targets = (targets - targets.mean(dim=0)) / targets.std(dim=0)
        noise = 0.05 * torch.randn(2000, 14, generator=generator)
        separate = torch.cat([targets, noise], dim=1)
        fused = torch.cat([(targets[:, :1] + targets[:, 1:]) / 2, torch.zeros(2000, 1), noise], dim=1)
        groups = (2 * cat + couch).long()
        scores = {}
        for name, features in (("separate", separate), ("fused", fused)):
            method = ConceptFactors()
            self.prepared(method, targets, separate)
            scores[name] = method.factor_learnability(features, list(range(100, 2100)), groups, ["a", "b", "c", "d"])
        self.assertGreater(scores["separate"]["student_residual_mean"], 0.8)
        self.assertLess(scores["fused"]["student_residual_mean"], 0.2)
        self.assertGreater(scores["separate"]["student_worst_group_mean"], 0.8)
        self.assertLess(scores["fused"]["student_worst_group_mean"], scores["fused"]["student_mean"] - 0.3)
        self.assertEqual(sum(scores["fused"]["group_counts"].values()), 2000)


class MeaningGroupingTests(unittest.TestCase):
    def cache(self, codes):
        return {"splice_codes": torch.as_tensor(codes, dtype=torch.float32),
                "vocabulary": ["path", "walkway", "bamboo", "jungle"], "sample_ids": ["0", "1", "2", "3"]}

    def test_synonyms_that_never_coactivate_share_a_group(self):
        # path and walkway fire on different images, as SpLiCE's L1 penalty makes synonyms do.
        codes = torch.eye(4)
        text = torch.tensor([[1.0, 0.0, 0.0], [0.95, 0.31, 0.0], [0.0, 0.0, 1.0], [0.0, 0.8, 0.6]])
        groups = group_by_meaning(self.cache(codes), text, text_threshold=0.8, response_threshold=0.0,
                                  min_frequency=0.1, max_frequency=0.95)
        self.assertEqual(groups, [[0, 1], [2], [3]])

    def test_average_linkage_breaks_a_chain_of_pairwise_matches(self):
        # 0~1 and 1~2 pass the threshold, 0~2 does not: a union of pairs would join all three.
        text = torch.tensor([[1.0, 0.0], [0.906, 0.423], [0.643, 0.766], [-1.0, 0.0]])
        groups = group_by_meaning(self.cache(torch.eye(4)), text, text_threshold=0.9, response_threshold=0.0,
                                  min_frequency=0.1, max_frequency=0.95)
        self.assertEqual(len(groups), 3)
        self.assertNotIn([0, 1, 2], groups)


class MeaningGroupsCommandTests(unittest.TestCase):
    def test_the_written_groups_load_as_training_factors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, _ = synthetic_inputs()
            name = cache_config_name(argparse.Namespace(
                dataset="waterbirds", splice_model=sweep_concept_factors.SPLICE_MODEL,
                splice_pretrained=sweep_concept_factors.SPLICE_PRETRAINED, splice_vocab="laion",
                splice_vocab_size=10000, splice_l1_penalty=0.25, splice_vocab_file=None, splice_vocab_order=None,
            ))
            cache_path = root / "features" / "waterbirds" / "splice_dataset_cache" / name / "splice_dataset_cache.pt"
            cache_path.parent.mkdir(parents=True)
            torch.save(cache, cache_path)
            text = torch.as_tensor(cache["dictionary"]).float()
            with patch.dict(os.environ, {"SPUR_SPLICE_OUTPUT_ROOT": str(root / "outputs")}), \
                    patch.object(build_meaning_groups, "raw_text_embeddings", return_value=text):
                output = build_meaning_groups.main([
                    "--dataset", "waterbirds", "--vocab", "laion", "--feature-root", str(root / "features"),
                    "--text-threshold", "0.9", "--factor-min-frequency", "0.02",
                ])
            factors, _, loaded_cache = load_concept_factors(
                "waterbirds", FactorConfig(), concept_groups=str(output), splice_cache=str(cache_path),
            )
            self.assertEqual(loaded_cache, cache_path)
            self.assertGreaterEqual(len(factors["factors"]), 2)
            self.assertTrue((output.parent / "factor_report.json").is_file())


class SweepTests(unittest.TestCase):
    def test_the_sweep_writes_groups_factor_sets_and_a_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_waterbirds_fixture(root / "datasets")
            cache, _ = synthetic_inputs()
            name = cache_config_name(argparse.Namespace(
                dataset="waterbirds", splice_model=sweep_concept_factors.SPLICE_MODEL,
                splice_pretrained=sweep_concept_factors.SPLICE_PRETRAINED, splice_vocab="laion",
                splice_vocab_size=10000, splice_l1_penalty=0.25, splice_vocab_file=None, splice_vocab_order=None,
            ))
            cache_path = root / "features" / "waterbirds" / "splice_dataset_cache" / name / "splice_dataset_cache.pt"
            cache_path.parent.mkdir(parents=True)
            torch.save(cache, cache_path)
            with patch.dict(os.environ, {"SPUR_SPLICE_OUTPUT_ROOT": str(root / "outputs")}):
                summary = sweep_concept_factors.main([
                    "--data-folder", str(root / "datasets"), "--datasets", "waterbirds", "--vocabs", "laion",
                    "--feature-root", str(root / "features"), "--text-thresholds", "0.80",
                    "--coactivation-thresholds", "0.30", "--merge-similarities", "0", "0.9",
                ])
            rows = json.loads((summary.parent / "summary.json").read_text(encoding="utf-8"))["rows"]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["attribute_factors"] >= 1 for row in rows))
            folder = root / "outputs" / "shared" / "waterbirds" / "factor_sweep" / "laion" / "text_0p80_coactivation_0p30"
            self.assertTrue((folder / "groups.json").is_file())
            record = json.loads((folder / "factors_merge_0p00.json").read_text(encoding="utf-8"))
            # Both planted pairs join a class factor to an attribute factor; weaker pairs follow them.
            self.assertEqual([pair["verdict"] for pair in record["pairs"][:2]], ["cross", "cross"])

    def test_the_meaning_grouping_runs_with_raw_text_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_waterbirds_fixture(root / "datasets")
            cache, _ = synthetic_inputs()
            name = cache_config_name(argparse.Namespace(
                dataset="waterbirds", splice_model=sweep_concept_factors.SPLICE_MODEL,
                splice_pretrained=sweep_concept_factors.SPLICE_PRETRAINED, splice_vocab="laion",
                splice_vocab_size=10000, splice_l1_penalty=0.25, splice_vocab_file=None, splice_vocab_order=None,
            ))
            cache_path = root / "features" / "waterbirds" / "splice_dataset_cache" / name / "splice_dataset_cache.pt"
            cache_path.parent.mkdir(parents=True)
            torch.save(cache, cache_path)
            text = torch.as_tensor(cache["dictionary"]).float()
            with patch.dict(os.environ, {"SPUR_SPLICE_OUTPUT_ROOT": str(root / "outputs")}), \
                    patch.object(sweep_concept_factors, "raw_text_embeddings", return_value=text):
                summary = sweep_concept_factors.main([
                    "--data-folder", str(root / "datasets"), "--datasets", "waterbirds", "--vocabs", "laion",
                    "--feature-root", str(root / "features"), "--methods", "meaning",
                    "--meaning-text-thresholds", "0.9", "--response-thresholds", "0", "0.5",
                    "--factor-min-frequencies", "0.02", "0.01", "--merge-similarities", "0",
                ])
            rows = json.loads((summary.parent / "summary.json").read_text(encoding="utf-8"))["rows"]
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["method"] for row in rows}, {"meaning"})
            self.assertTrue(all("error" not in row for row in rows))
            folder = root / "outputs" / "shared" / "waterbirds" / "factor_sweep" / "laion"
            self.assertTrue((folder / "meaning_text_0p90_response_0p00_min_0.002" / "factors_merge_0p00_min_0.01.json").is_file())


class ConditionedSamplerTests(unittest.TestCase):
    def active(self):
        active = torch.zeros(100, 2, dtype=torch.bool)
        active[:40, 0] = True
        active[60:, 1] = True
        return active

    def test_every_image_occurs_once_per_epoch(self):
        sampler = ConceptConditionedBatchSampler(self.active(), 16, 0.5, torch.Generator().manual_seed(0))
        for _ in range(3):
            batches = list(sampler)
            self.assertEqual(sorted(index for batch in batches for index in batch), list(range(100)))
            self.assertEqual(len(batches), len(sampler))

    def test_a_conditioned_batch_shows_one_factor(self):
        sampler = ConceptConditionedBatchSampler(self.active(), 16, 1.0, torch.Generator().manual_seed(1))
        first = next(iter(sampler))
        shared = self.active()[first].all(dim=0)
        self.assertTrue(bool(shared.any()))

    def test_zero_fraction_is_plain_shuffling(self):
        sampler = ConceptConditionedBatchSampler(self.active(), 16, 0.0, torch.Generator().manual_seed(2))
        list(sampler)
        self.assertEqual(sampler.last_conditioned_fraction, 0.0)

    def test_the_same_generator_seed_gives_the_same_batches(self):
        first = list(ConceptConditionedBatchSampler(self.active(), 16, 0.5, torch.Generator().manual_seed(3)))
        second = list(ConceptConditionedBatchSampler(self.active(), 16, 0.5, torch.Generator().manual_seed(3)))
        self.assertEqual(first, second)


class FactorDistillationTests(unittest.TestCase):
    def test_loss_follows_the_schedule_and_trains_the_backbone(self):
        targets = torch.randn(6, 3)
        regularizer = FactorDistillationRegularizer(targets, weight=2.0, start_epoch=1, warmup_epochs=2)
        backbone, head = torch.nn.Linear(4, 5), torch.nn.Linear(5, 3)
        embeddings = backbone(torch.randn(8, 4))
        regularizer.set_epoch(1)
        self.assertEqual(float(regularizer(head(embeddings), torch.tensor([0, 1, 2, 3]))), 0.0)
        regularizer.set_epoch(2)
        loss = regularizer(head(embeddings), torch.tensor([0, 1, 2, 3]))
        self.assertAlmostEqual(regularizer.scheduled_weight, 1.0)
        loss.backward()
        self.assertGreater(float(backbone.weight.grad.norm()), 0.0)
        mse = regularizer.last_diagnostics["factor_mse"]
        self.assertAlmostEqual(regularizer.last_diagnostics["factor_explained_variance"], 1 - mse)


class ConceptFactorTrainingTests(unittest.TestCase):
    def test_both_mechanisms_train_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_waterbirds_fixture(root / "datasets")
            cache, groups = synthetic_inputs()
            torch.save(cache, root / "cache.pt")
            (root / "groups.json").write_text(json.dumps(groups), encoding="utf-8")
            run_dir = root / "run"
            env = {
                **os.environ, "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled",
                "SPUR_SPLICE_SCRATCH_ROOT": str(root / "scratch"), "SPUR_SPLICE_OUTPUT_ROOT": str(root / "outputs"),
                "PYTHONPATH": os.pathsep.join([str(PROJECT_ROOT), os.environ.get("PYTHONPATH", "")]),
            }
            command = [
                sys.executable, "-u", str(PROJECT_ROOT / "spur_splice.py"), *COMMON_ARGS,
                "--data_folder", str(root / "datasets"), "--arm", "factors",
                "--artifact_dir", str(run_dir / "training"), "--run_record", str(run_dir / "run.json"),
                "--splice_mode", "concept_factors",
                "--factor_concept_groups", str(root / "groups.json"), "--factor_splice_cache", str(root / "cache.pt"),
                "--factor_min_correlation", "0.3", "--factor_condition_fraction", "0.5",
                "--factor_distill_weight", "1.0", "--factor_start_epoch", "0", "--factor_warmup_epochs", "0",
                "--factor_sample_weighting", "balanced", "--factor_block_weight", "1.0", "--factor_block_count", "4",
            ]
            cross_fit = [*command, "--factor_cross_fit", "true", "--arm", "factors_cross_fit",
                         "--run_record", str(run_dir / "cross_fit.json")]
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-5000:])
            self.assertIn("water | lake | ocean  <->  waterbird | gull", result.stdout)
            record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "complete")
            ssl = [event["values"] for event in record["metrics"] if event.get("stage") == "ssl"]
            self.assertGreater(ssl[-1]["SSL splice loss"], 0.0)
            # The diagnostics reach W&B under method/.
            canonical = canonical_train_metrics(ssl[-1])
            self.assertIn("method/factor_mse", canonical)
            self.assertIn("method/factor_conditioned_batch_fraction", canonical)
            self.assertIn("method/factor_block_loss", canonical)
            self.assertIn("Concept blocks (real presence)", result.stdout)
            self.assertIn("method/factor_heldout_explained_variance", canonical)
            result = subprocess.run(cross_fit, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-5000:])
            record = json.loads((run_dir / "cross_fit.json").read_text(encoding="utf-8"))
            last = [event["values"] for event in record["metrics"] if event.get("stage") == "ssl"][-1]
            self.assertIn("method/factor_heldout_explained_variance", canonical_train_metrics(last))


class ConceptFactorOptionTests(unittest.TestCase):
    def test_factor_runs_get_their_own_storage_name(self):
        name = storage_name(["--splice_mode", "concept_factors", "--factor_distill_weight", "1"])
        self.assertTrue(name.startswith("waterbirds_s3_concept-factors_e500_"))
        self.assertNotEqual(
            name, storage_name(["--splice_mode", "concept_factors", "--factor_condition_fraction", "0.5"]),
        )


if __name__ == "__main__":
    unittest.main()
