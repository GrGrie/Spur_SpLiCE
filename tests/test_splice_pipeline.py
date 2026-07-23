import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import torch

from experiments.spurious_eval.datasets.celeba import CelebADataset
from experiments.spurious_eval.datasets.transforms import (
    ConceptAwareTwoCropTransform,
    build_augmentation_routing,
)
from experiments.spurious_eval.evaluation_protocol import resolve_evaluation_split, resolve_probe_mode
from experiments.spurious_eval.linear_probe import resolve_lr_decay_epochs, run_spurious_attribute_probe
from experiments.spurious_eval.splice_cbm import zero_sparse_columns
from splice.ssl_regularization import (
    CorrelationSpliceRegularizer,
    CounterfactualSpliceDistillation,
    SpliceConfig,
    edit_spurious_concept_weights,
    residual_preserving_intervention,
    score_cache_path,
)
from splice.model import SPLICE
from spur_splice import resolve_epoch_schedule
from scripts.tools.discover_splice_spurious_concepts import SparseConceptWeights, rank_concepts


class SplicePipelineTests(unittest.TestCase):
    def test_automatic_lr_schedules_scale_with_training_length(self):
        self.assertEqual(resolve_epoch_schedule("auto", 1000, (0.70, 0.80, 0.90)), [700, 800, 900])
        self.assertEqual(resolve_epoch_schedule("auto", 500, (0.70, 0.80, 0.90)), [350, 400, 450])
        self.assertEqual(resolve_epoch_schedule("auto", 100, (0.60, 0.75, 0.90)), [60, 75, 90])
        self.assertEqual(resolve_epoch_schedule("auto", 1, (0.70, 0.80, 0.90)), [])
        self.assertEqual(resolve_lr_decay_epochs("auto", 50), [30, 38, 45])
        with self.assertRaises(ValueError):
            resolve_epoch_schedule("350,350,450", 500, (0.70, 0.80, 0.90))

    def test_evaluation_protocol_requires_explicit_final_test(self):
        self.assertEqual(resolve_evaluation_split(None, final_test=False), "val")
        self.assertEqual(resolve_evaluation_split(None, final_test=True), "test")
        self.assertEqual(resolve_evaluation_split("test", final_test=True), "test")
        with self.assertRaises(ValueError):
            resolve_evaluation_split("test", final_test=False)
        with self.assertRaises(ValueError):
            resolve_evaluation_split("val", final_test=True)
        self.assertEqual(resolve_probe_mode(None, final_test=False), "periodic")
        self.assertEqual(resolve_probe_mode(None, final_test=True), "final")
        with self.assertRaises(ValueError):
            resolve_probe_mode("periodic", final_test=True)

    def test_celeba_matches_spurssl_target_and_confounder(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "celeba"
            root.mkdir()
            (root / "list_attr_celeba.csv").write_text(
                "image_id,Male,Blond_Hair\n"
                "1.jpg,1,-1\n"
                "2.jpg,-1,1\n"
                "3.jpg,1,1\n"
                "4.jpg,-1,-1\n",
                encoding="utf-8",
            )
            (root / "list_eval_partition.csv").write_text(
                "image_id,partition\n1.jpg,0\n2.jpg,0\n3.jpg,1\n4.jpg,2\n",
                encoding="utf-8",
            )
            dataset = CelebADataset(root)
            torch.testing.assert_close(dataset.y_array, torch.tensor([0, 1, 1, 0]))
            torch.testing.assert_close(dataset.metadata_array[:, 0], torch.tensor([1, 0, 1, 0]))
            self.assertEqual(dataset.metadata_fields, ["gender", "y"])

    def test_targeted_transform_keeps_one_standard_view(self):
        transform = ConceptAwareTwoCropTransform(lambda _: "standard", lambda _: "strong", threshold=0.5)
        self.assertEqual(transform(object(), 0.7), ["standard", "strong"])
        self.assertEqual(transform(object(), 0.2), ["standard", "standard"])

    def test_routing_controls_match_the_semantic_augmentation_budget(self):
        scores = torch.tensor([0.0, 1.0, 2.0, 3.0])
        semantic, semantic_threshold, _ = build_augmentation_routing(scores, None, 0.5, "semantic", seed=7)
        shuffled, shuffled_threshold, _ = build_augmentation_routing(scores, None, 0.5, "shuffled", seed=7)
        random, random_threshold, _ = build_augmentation_routing(scores, None, 0.5, "random", seed=7)
        all_scores, all_threshold, _ = build_augmentation_routing(scores, None, 0.5, "all", seed=7)

        semantic_count = int((semantic >= semantic_threshold).sum())
        self.assertEqual(int((shuffled >= shuffled_threshold).sum()), semantic_count)
        self.assertEqual(int((random >= random_threshold).sum()), semantic_count)
        self.assertEqual(int((all_scores >= all_threshold).sum()), len(scores))
        torch.testing.assert_close(torch.sort(shuffled).values, torch.sort(scores).values)
        repeated, _, _ = build_augmentation_routing(scores, None, 0.5, "random", seed=7)
        torch.testing.assert_close(random, repeated)

    def test_conditional_regularizer_ignores_target_only_signal(self):
        targets = torch.tensor([0, 0, 1, 1])
        concepts = targets.float().unsqueeze(1)
        embeddings = torch.stack([targets.float(), targets.float()], dim=1).requires_grad_()
        conditional_loss = CorrelationSpliceRegularizer(1.0, conditional_on_target=True)(
            embeddings, concepts, targets
        )
        unconditional_loss = CorrelationSpliceRegularizer(1.0, conditional_on_target=False)(
            embeddings, concepts, targets
        )
        self.assertEqual(float(conditional_loss), 0.0)
        self.assertGreater(float(unconditional_loss), 0.9)

    def test_conditional_regularizer_penalizes_within_target_concepts(self):
        targets = torch.tensor([0, 0, 0, 1, 1, 1])
        concepts = torch.tensor([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]])
        embeddings = torch.tensor(
            [[0.1, 0.0], [0.7, 1.2], [2.2, 1.8], [0.0, 0.2], [1.4, 0.8], [1.7, 2.4]],
            requires_grad=True,
        )
        loss = CorrelationSpliceRegularizer(0.25, conditional_on_target=True)(embeddings, concepts, targets)
        loss.backward()
        self.assertGreater(float(loss), 0.15)
        self.assertGreater(float(embeddings.grad.norm()), 0.0)

    def test_residual_preserving_intervention_changes_only_selected_direction_before_normalization(self):
        embeddings = torch.tensor([[0.6, 0.8, 0.0]])
        weights = torch.tensor([[0.2]])
        edited = torch.tensor([[0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])
        actual = residual_preserving_intervention(embeddings, weights, edited, directions, strength=0.5)
        expected = torch.nn.functional.normalize(torch.tensor([[0.5, 0.8, 0.0]]), dim=1)
        torch.testing.assert_close(actual, expected)

    def test_counterfactual_edits_include_median_and_matched_controls(self):
        weights = torch.tensor([[0.0], [2.0], [10.0], [12.0]])
        embeddings = torch.tensor(
            [[0.0, 1.0, 0.0], [2.0, 0.0, 1.0], [10.0, 0.9, 0.1], [12.0, 0.1, 0.9]]
        )
        directions = torch.tensor([[1.0, 0.0, 0.0]])
        targets = torch.zeros(4, dtype=torch.long)
        spurious = torch.tensor([0, 0, 1, 1])
        median = edit_spurious_concept_weights(
            "class_median", weights, embeddings, directions, targets, spurious
        )
        torch.testing.assert_close(median, torch.full_like(weights, 2.0))
        matched = edit_spurious_concept_weights(
            "matched_swap", weights, embeddings, directions, targets, spurious
        )
        torch.testing.assert_close(matched, torch.tensor([[10.0], [12.0], [0.0], [2.0]]))
        zeroed = edit_spurious_concept_weights(
            "zero_out", weights, embeddings, directions, targets, spurious
        )
        torch.testing.assert_close(zeroed, torch.zeros_like(weights))

    def test_counterfactual_distillation_stops_teacher_gradients(self):
        predictions = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        teacher = torch.tensor([[0.0, 1.0], [0.0, 1.0]], requires_grad=True)
        loss = CounterfactualSpliceDistillation(0.5)(predictions, teacher)
        loss.backward()
        self.assertGreater(float(predictions.grad.norm()), 0.0)
        self.assertIsNone(teacher.grad)

    def test_sparse_discovery_storage_selects_concepts_without_dense_vocab(self):
        weights = SparseConceptWeights(
            rows=torch.tensor([0, 0, 1, 2]),
            columns=torch.tensor([1, 5, 5, 9]),
            values=torch.tensor([0.1, 0.2, 0.3, 0.4]),
            n_rows=3,
            n_columns=10,
        )
        torch.testing.assert_close(
            weights.select_columns([5, 1]),
            torch.tensor([[0.2, 0.1], [0.3, 0.0], [0.0, 0.0]]),
        )

    def test_discovery_penalizes_target_specific_concept_at_full_scale(self):
        # concept 0 varies only with the spurious value; concept 1 varies only with target.
        group_means = {
            (0, 0): torch.tensor([0.0, 0.0]),
            (1, 0): torch.tensor([1.0, 0.0]),
            (0, 1): torch.tensor([0.0, 1.0]),
            (1, 1): torch.tensor([1.0, 1.0]),
        }
        args = argparse.Namespace(label_penalty=1.0, instability_penalty=1.0, use_abs_score=False, min_mean_weight=0.0, top_k=2)
        candidates = rank_concepts(
            ["spurious", "target"],
            group_means,
            {key: 1 for key in group_means},
            torch.tensor([0.5, 0.5]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            {"spurious": {0: "s0", 1: "s1"}, "target": {0: "y0", 1: "y1"}},
            args,
        )
        self.assertEqual([candidate["concept"] for candidate in candidates], ["spurious"])

    def test_discovery_requires_a_consistent_signed_spurious_effect(self):
        group_means = {
            (0, 0): torch.tensor([0.0, 0.0]),
            (1, 0): torch.tensor([1.0, 1.0]),
            (0, 1): torch.tensor([0.0, 1.0]),
            (1, 1): torch.tensor([1.0, 0.0]),
        }
        args = argparse.Namespace(
            label_penalty=0.0,
            instability_penalty=0.0,
            use_abs_score=False,
            min_mean_weight=0.0,
            top_k=2,
            require_consistent_spurious_direction=True,
            deduplicate_concepts=False,
        )
        candidates = rank_concepts(
            ["consistent", "reverses"],
            group_means,
            {key: 1 for key in group_means},
            torch.tensor([0.5, 0.5]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            {"spurious": {0: "s0", 1: "s1"}, "target": {0: "y0", 1: "y1"}},
            args,
        )
        self.assertEqual([candidate["concept"] for candidate in candidates], ["consistent"])

    def test_discovery_deduplicates_plural_concept_variants(self):
        group_means = {
            (0, 0): torch.tensor([0.0, 0.0, 0.0]),
            (1, 0): torch.tensor([3.0, 2.0, 1.0]),
            (0, 1): torch.tensor([0.0, 0.0, 0.0]),
            (1, 1): torch.tensor([3.0, 2.0, 1.0]),
        }
        args = argparse.Namespace(
            label_penalty=0.0,
            instability_penalty=0.0,
            use_abs_score=False,
            min_mean_weight=0.0,
            top_k=2,
            require_consistent_spurious_direction=True,
            deduplicate_concepts=True,
        )
        candidates = rank_concepts(
            ["forests", "forest", "lake"],
            group_means,
            {key: 1 for key in group_means},
            torch.tensor([1.5, 1.0, 0.5]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            {"spurious": {0: "s0", 1: "s1"}, "target": {0: "y0", 1: "y1"}},
            args,
        )
        self.assertEqual([candidate["concept"] for candidate in candidates], ["forests", "lake"])

    def test_cache_fingerprint_separates_vectors_and_scalar_reductions(self):
        config_mean = SpliceConfig(concepts="1,2", score_reduction="mean", pretrained="a")
        config_max = SpliceConfig(concepts="1,2", score_reduction="max", pretrained="a")
        vector_mean = score_cache_path(config_mean, 4, [1, 2], "dataset", artifact="concept_weights")
        vector_max = score_cache_path(config_max, 4, [1, 2], "dataset", artifact="concept_weights")
        score_mean = score_cache_path(config_mean, 4, [1, 2], "dataset", artifact="scores")
        score_max = score_cache_path(config_max, 4, [1, 2], "dataset", artifact="scores")
        self.assertEqual(vector_mean, vector_max)
        self.assertNotEqual(score_mean, score_max)

    def test_splice_cpu_solver_returns_nonnegative_sparse_weights(self):
        model = SPLICE(
            image_mean=torch.zeros(2),
            dictionary=torch.eye(2),
            clip=None,
            solver="skl",
            l1_penalty=0.01,
            return_weights=True,
            device="cpu",
        )
        weights = model.encode_image(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        self.assertEqual(tuple(weights.shape), (2, 2))
        self.assertTrue(torch.all(weights >= 0))
        with self.assertRaises(RuntimeError):
            SPLICE(torch.zeros(2), torch.eye(2), solver="unsupported")

    def test_sparse_cbm_intervention_zeroes_only_requested_columns(self):
        matrix = sparse.csr_matrix(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        intervened = zero_sparse_columns(matrix, [1]).toarray()
        np.testing.assert_array_equal(intervened, np.asarray([[1.0, 0.0, 3.0], [4.0, 0.0, 6.0]]))

    def test_spurious_leakage_probe_reports_last_ten_metrics(self):
        features = torch.tensor(
            [[-1.0, 0.0], [-0.8, 0.1], [1.0, 0.0], [0.8, -0.1]] * 2,
            dtype=torch.float32,
        )
        target = torch.tensor([0, 1, 0, 1] * 2)
        spurious = torch.tensor([0, 0, 1, 1] * 2)
        metadata = torch.stack((spurious, target), dim=1)
        dataset = torch.utils.data.TensorDataset(features, target, metadata)
        args = argparse.Namespace(
            batch_size=4,
            seed=0,
            epochs=2,
            learning_rate=0.1,
            momentum=0.0,
            weight_decay=0.0,
            cosine=False,
            lr_decay_rate=0.2,
            lr_decay_epochs=[],
        )
        metrics = run_spurious_attribute_probe(dataset, dataset, 2, args, torch.device("cpu"))
        self.assertIn("Spurious probe average over last 10 val acc", metrics)
        self.assertIn("Spurious probe average over last 10 val worst-group acc", metrics)


if __name__ == "__main__":
    unittest.main()
