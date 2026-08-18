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
from experiments.spurious_eval.losses.contrastive import SimCLRLoss
from experiments.spurious_eval.splice_cbm import zero_sparse_columns
from experiments.spurious_eval.training.ssl_loop import simclr_forward_loss
from splice.ssl_regularization import (
    CorrelationSpliceRegularizer,
    SpliceSynthesisDistillation,
    SpliceConfig,
    edit_spurious_concept_weights,
    random_dictionary_indices,
    residual_preserving_intervention,
    score_cache_path,
)
from splice.crp import (
    CrpAuditConfig,
    orthonormal_basis,
    project_out,
    run_frozen_audit,
    save_feature_cache,
    validate_feature_cache,
)
from splice.model import SPLICE
from spur_splice import resolve_epoch_schedule
from scripts.tools.discover_splice_spurious_concepts import (
    SparseConceptWeights,
    UtilityProbeFold,
    evaluate_concept_set_utility,
    evaluate_single_concept_utility,
    fit_cross_fitted_sparse_probes,
    rank_concepts,
)
from scripts.tools.cache_crp_features import IndexedImages


class SplicePipelineTests(unittest.TestCase):
    @staticmethod
    def _tiny_crp_cache():
        generator = torch.Generator().manual_seed(4)
        return {
            "cache_version": 1,
            "provenance": {"fixture": "tiny-crp-cache"},
            "sample_ids": [f"image-{index}" for index in range(8)],
            "clip_embeddings": torch.nn.functional.normalize(torch.randn(8, 4, generator=generator), dim=1),
            "image_mean": torch.zeros(4),
            "splice_codes": torch.tensor(
                [[1.0, 0.0], [1.0, 0.0], [0.8, 0.0], [0.8, 0.0],
                 [0.0, 1.0], [0.0, 1.0], [0.0, 0.8], [0.0, 0.8]]
            ),
            "dictionary": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            "vocabulary": ["forest", "water"],
            "dino_embeddings": torch.nn.functional.normalize(torch.randn(8, 3, generator=generator), dim=1),
        }

    def test_crp_projection_removes_the_full_group_subspace(self):
        basis = orthonormal_basis(torch.tensor([[1.0, 0.0, 0.0], [1.0, 1e-9, 0.0]]))
        self.assertEqual(tuple(basis.shape), (3, 1))
        embeddings = torch.tensor([[0.6, 0.0, 0.8], [0.0, 0.6, 0.8]])
        projected = project_out(embeddings, basis)
        torch.testing.assert_close(projected @ basis, torch.zeros(2, 1), atol=1e-6, rtol=0)

    def test_crp_cache_rejects_training_annotations(self):
        cache = self._tiny_crp_cache()
        cache["labels"] = torch.zeros(8)
        with self.assertRaisesRegex(ValueError, "forbidden annotation"):
            validate_feature_cache(cache)

    def test_crp_cache_does_not_require_manual_hashes(self):
        cache = self._tiny_crp_cache()
        cache.pop("provenance")
        validated = validate_feature_cache(cache)
        self.assertEqual(validated["sample_ids"], cache["sample_ids"])

    def test_crp_feature_cache_is_saved_atomically(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "cache.pt"
            save_feature_cache(self._tiny_crp_cache(), path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(loaded["cache_version"], 1)
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_crp_cache_builder_reads_images_without_labels(self):
        class ImagesOnlyDataset:
            def get_subset(self, split, transform=None):
                self.asserted_split = split
                return argparse.Namespace(indices=np.asarray([2, 0]))

            def get_input(self, index):
                return f"image-{index}"

            def __getitem__(self, index):
                raise AssertionError("The CRP cache builder must not read labels or metadata.")

        dataset = ImagesOnlyDataset()
        images = IndexedImages(dataset)
        self.assertEqual(images[0], (2, "image-2"))
        self.assertEqual(dataset.asserted_split, "train")

    def test_crp_audit_is_deterministic_and_exports_row_stochastic_graph(self):
        config = CrpAuditConfig(
            min_concept_frequency=0.1,
            max_concept_frequency=0.9,
            projected_neighbors=3,
            dino_neighbors=4,
            graph_top_k=2,
            null_trials=1,
            null_quantile=0.0,
            min_coverage=0.0,
            seed=7,
        )
        first = run_frozen_audit(self._tiny_crp_cache(), config)
        second = run_frozen_audit(self._tiny_crp_cache(), config)
        torch.testing.assert_close(first["neighbor_indices"], second["neighbor_indices"])
        torch.testing.assert_close(first["weights"], second["weights"])
        row_sums = first["weights"].sum(dim=1)
        supported = row_sums > 0
        torch.testing.assert_close(row_sums[supported], torch.ones_like(row_sums[supported]))
        self.assertTrue(torch.all(first["neighbor_indices"][first["weights"] == 0] == -1))
        self.assertTrue(all("activation_gain_alignment" in group for group in first["groups"]))

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

    def test_synthesis_edits_include_neutralize_swap_and_donor_controls(self):
        weights = torch.tensor([[0.0], [2.0], [10.0], [12.0]])
        embeddings = torch.tensor(
            [[0.0, 1.0, 0.0], [2.0, 0.0, 1.0], [10.0, 0.9, 0.1], [12.0, 0.1, 0.9]]
        )
        directions = torch.tensor([[1.0, 0.0, 0.0]])
        targets = torch.zeros(4, dtype=torch.long)
        spurious = torch.tensor([0, 0, 1, 1])
        neutralized = edit_spurious_concept_weights(
            "class_neutralize", weights, embeddings, directions, targets, spurious
        )
        torch.testing.assert_close(neutralized, torch.full_like(weights, 2.0))
        matched = edit_spurious_concept_weights(
            "core_matched_swap", weights, embeddings, directions, targets, spurious
        )
        torch.testing.assert_close(matched, torch.tensor([[10.0], [12.0], [0.0], [2.0]]))
        same_class = edit_spurious_concept_weights(
            "same_class_random_donor", weights, embeddings, directions, targets, spurious, seed=3
        )
        self.assertTrue(torch.all(same_class != weights))
        zeroed = edit_spurious_concept_weights(
            "zero_out", weights, embeddings, directions, targets, spurious
        )
        torch.testing.assert_close(zeroed, torch.zeros_like(weights))

    def test_random_coordinate_control_excludes_selected_coordinates(self):
        indices = random_dictionary_indices(5, [1, 3], count=2, seed=7)
        self.assertEqual(len(indices), 2)
        self.assertTrue(set(indices).isdisjoint({1, 3}))

    def test_synthesis_distillation_stops_teacher_gradients(self):
        predictions = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        teacher = torch.tensor([[0.0, 1.0], [0.0, 1.0]], requires_grad=True)
        loss = SpliceSynthesisDistillation(0.5)(predictions, teacher)
        loss.backward()
        self.assertGreater(float(predictions.grad.norm()), 0.0)
        self.assertIsNone(teacher.grad)

    def test_synthesis_mode_trains_simclr_and_clip_distillation_heads(self):
        class TinyTwoHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(2, 3, bias=False)
                self.head = torch.nn.Linear(3, 2, bias=False)
                self.clip_distillation_head = torch.nn.Linear(3, 2, bias=False)

        model = TinyTwoHead()
        images = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
        ]
        teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss, _, _ = simclr_forward_loss(
            model,
            SimCLRLoss(temperature=0.1),
            images,
            teacher,
            torch.tensor([0, 1]),
            SpliceSynthesisDistillation(0.5),
        )
        loss.backward()
        self.assertGreater(float(model.head.weight.grad.norm()), 0.0)
        self.assertGreater(float(model.clip_distillation_head.weight.grad.norm()), 0.0)

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

    def test_intervention_utility_rewards_repairs_and_penalizes_damage(self):
        features = sparse.csc_matrix(
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            )
        )
        logits = np.asarray(
            [
                [0.0, 1.0],  # wrong y=0; deleting concept 0 repairs it
                [0.0, 2.0],
                [1.0, 0.0],  # correct y=0; deleting concept 1 damages it
                [0.0, 2.0],
            ]
        )
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        fold = UtilityProbeFold(
            features=features,
            labels=np.asarray([0, 1, 0, 1]),
            logits=logits,
            probabilities=probabilities,
            predictions=logits.argmax(axis=1),
            coefficients=np.asarray([[0.0, 2.0], [2.0, 0.0]]),
        )
        error_support = np.asarray([1, 0])
        correct_support = np.asarray([1, 2])
        repair = evaluate_single_concept_utility([fold], 0, error_support, correct_support)
        damage = evaluate_single_concept_utility([fold], 1, error_support, correct_support)
        self.assertEqual(repair["repaired"], 1)
        self.assertEqual(repair["damaged"], 0)
        self.assertGreater(repair["score"], 0.0)
        self.assertEqual(damage["repaired"], 0)
        self.assertEqual(damage["damaged"], 1)
        self.assertLess(damage["score"], 0.0)

    def test_intervention_utility_cross_fits_binary_sparse_probe(self):
        labels = torch.tensor([index % 2 for index in range(40)], dtype=torch.long)
        weights = SparseConceptWeights(
            rows=torch.arange(40, dtype=torch.long),
            columns=labels.clone(),
            values=torch.ones(40),
            n_rows=40,
            n_columns=2,
        )
        folds, diagnostics = fit_cross_fitted_sparse_probes(
            weights,
            labels,
            argparse.Namespace(
                utility_max_samples=0,
                probe_cv_folds=4,
                probe_c=1.0,
                probe_max_iter=1000,
                seed=7,
            ),
        )
        self.assertEqual(len(folds), 4)
        self.assertEqual(sum(len(fold.labels) for fold in folds), 40)
        self.assertTrue(all(fold.logits.shape[1] == 2 for fold in folds))
        self.assertEqual(diagnostics["audit_sample_count"], 40)
        self.assertGreaterEqual(diagnostics["probe_cv_accuracy"], 0.95)

    def test_joint_intervention_utility_evaluates_selected_set_exactly(self):
        features = sparse.csc_matrix(np.asarray([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
        logits = np.asarray([[0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        fold = UtilityProbeFold(
            features=features,
            labels=np.asarray([0, 0, 1]),
            logits=logits,
            probabilities=probabilities,
            predictions=logits.argmax(axis=1),
            coefficients=np.asarray([[0.0, 0.0], [0.6, 0.6]]),
        )
        metrics = evaluate_concept_set_utility(
            [fold],
            [0, 1],
            np.asarray([1, 0]),
            np.asarray([1, 1]),
        )
        self.assertEqual(metrics["repaired"], 1)
        self.assertEqual(metrics["damaged"], 0)
        self.assertGreater(metrics["score"], 0.0)

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
        vector_reordered = score_cache_path(config_mean, 4, [2, 1], "dataset", artifact="concept_weights")
        score_mean = score_cache_path(config_mean, 4, [1, 2], "dataset", artifact="scores")
        score_max = score_cache_path(config_max, 4, [1, 2], "dataset", artifact="scores")
        self.assertEqual(vector_mean, vector_max)
        self.assertNotEqual(vector_mean, vector_reordered)
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
