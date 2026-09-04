import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
from experiments.spurious_eval.training.ssl_loop import simclr_forward_loss, train_one_epoch
from splice.ssl_regularization import (
    CorrelationSpliceRegularizer,
    OracleRelationalRegularizer,
    SpliceSynthesisDistillation,
    SpliceConfig,
    edit_spurious_concept_weights,
    random_dictionary_indices,
    residual_preserving_intervention,
    score_cache_path,
)
from splice.crp import (
    CrpAuditConfig,
    group_concepts,
    orthonormal_basis,
    project_out,
    run_frozen_audit,
    save_feature_cache,
    validate_feature_cache,
)
from splice.cqt import CQT_ARTIFACT, CqtAuditConfig, concept_quotient, run_cqt_audit
from splice.cobalt_check import concept_balanced_sample_weights, load_cobalt_train_concepts
from splice.crp_training import (
    CrpGraphBatchSampler,
    CrpRelationalRegularizer,
    IndexedCrpDataset,
    build_crp_concept_report,
    validate_teacher_graph,
)
from splice.graph_io import load_graph_json, save_graph_json
from splice.model import SPLICE
from splice.splice import (
    DEFAULT_VOCABULARY,
    DEFAULT_VOCABULARY_SIZE,
    _clean_openimages_class_names,
)
import spur_splice
from spur_splice import resolve_epoch_schedule
from scripts.tools.audit_cqt_graph_oracle import audit_graph
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
    def test_openimages_v7_is_the_default_vocabulary(self):
        self.assertEqual(DEFAULT_VOCABULARY, "openimages_v7")
        self.assertEqual(DEFAULT_VOCABULARY_SIZE, -1)
        self.assertEqual(SpliceConfig().vocab, DEFAULT_VOCABULARY)
        self.assertEqual(SpliceConfig().vocab_size, DEFAULT_VOCABULARY_SIZE)
        with patch("sys.argv", ["spur_splice.py"]):
            args = spur_splice.parse_args()
        self.assertEqual(args.splice_vocab, DEFAULT_VOCABULARY)
        self.assertEqual(args.splice_vocab_size, DEFAULT_VOCABULARY_SIZE)

    def test_openimages_class_names_are_cleaned_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "classes.csv"
            destination = Path(temporary_directory) / "vocab" / "openimages_v7.txt"
            source.write_text(
                "LabelName,DisplayName\n"
                "/m/1, Cat \n"
                "/m/2,cat\n"
                "/m/3,\n"
                "/m/4,Fire-truck\n",
                encoding="utf-8",
            )
            _clean_openimages_class_names(str(source), str(destination))
            self.assertEqual(
                destination.read_text(encoding="utf-8").splitlines(),
                ["Cat", "Fire-truck"],
            )

    @staticmethod
    def _tiny_teacher_graph():
        return {
            "artifact": "splice_crp_v2_teacher_graph",
            "graph_version": 2,
            "sample_ids": [f"waterbirds:{index}" for index in range(4)],
            "neighbor_indices": torch.tensor([[1, -1], [0, -1], [3, -1], [2, -1]]),
            "weights": torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
            "confidence": torch.ones(4),
            "anchor_confidence": torch.tensor([1.0, 0.8, 0.6, 0.4]),
        }

    @staticmethod
    def _tiny_crp_cache():
        generator = torch.Generator().manual_seed(4)
        return {
            "cache_version": 2,
            "provenance": {"fixture": "tiny-crp-cache"},
            "sample_ids": [f"image-{index}" for index in range(8)],
            "clip_embeddings": torch.nn.functional.normalize(torch.randn(8, 4, generator=generator), dim=1),
            "image_mean": torch.zeros(4),
            "splice_codes": torch.tensor(
                [[1.0, 0.0], [1.0, 0.0], [0.8, 0.0], [0.8, 0.0],
                 [0.0, 1.0], [0.0, 1.0], [0.0, 0.8], [0.0, 0.8]]
            ),
            "dictionary": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            "vocabulary": ["concept_a", "concept_b"],
        }

    @staticmethod
    def _tiny_cqt_cache():
        sample_count = 24
        codes = torch.zeros(sample_count, 5)
        clip = torch.zeros(sample_count, 6)
        for index in range(sample_count):
            state = int(index >= sample_count // 2)
            context = index % 2
            codes[index, state] = 1.0
            codes[index, 2] = 1.0
            codes[index, 3 + context] = 0.8
            clip[index, state] = 1.4
            clip[index, 2] = 1.0
            clip[index, 3 + context] = 0.8
            clip[index, 5] = 0.02 * (index % 3)
        dictionary = torch.eye(6)[:5]
        return {
            "cache_version": 2,
            "provenance": {"fixture": "tiny-cqt-cache"},
            "sample_ids": [f"waterbirds:{index}" for index in range(sample_count)],
            "clip_embeddings": torch.nn.functional.normalize(clip, dim=1),
            "image_mean": torch.zeros(6),
            "splice_codes": codes,
            "dictionary": dictionary,
            "vocabulary": ["state_a", "state_b", "shared_feature", "context_a", "context_b"],
        }

    def test_crp_projection_removes_the_full_group_subspace(self):
        basis = orthonormal_basis(torch.tensor([[1.0, 0.0, 0.0], [1.0, 1e-9, 0.0]]))
        self.assertEqual(tuple(basis.shape), (3, 1))
        embeddings = torch.tensor([[0.6, 0.0, 0.8], [0.0, 0.6, 0.8]])
        projected = project_out(embeddings, basis)
        torch.testing.assert_close(projected @ basis, torch.zeros(2, 1), atol=1e-6, rtol=0)

    def test_zero_coactivation_threshold_forms_semantic_families(self):
        codes = torch.eye(4)
        dictionary = torch.nn.functional.normalize(
            torch.tensor(
                [
                    [1.0, 0.00, 0.0, 0.0],
                    [1.0, 0.10, 0.0, 0.0],
                    [1.0, 0.20, 0.0, 0.0],
                    [1.0, 0.30, 0.0, 0.0],
                ]
            ),
            dim=1,
        )
        vocabulary = ["semantic_a", "semantic_b", "semantic_c", "semantic_d"]
        coactivation_gated = CrpAuditConfig(
            min_concept_frequency=0.2,
            text_similarity_threshold=0.9,
            coactivation_threshold=0.2,
        )
        semantic_only = CrpAuditConfig(
            min_concept_frequency=0.2,
            text_similarity_threshold=0.9,
            coactivation_threshold=0.0,
        )

        self.assertEqual(
            group_concepts(codes, dictionary, vocabulary, coactivation_gated),
            [[0], [1], [2], [3]],
        )
        self.assertEqual(
            group_concepts(codes, dictionary, vocabulary, semantic_only),
            [[0, 1, 2, 3]],
        )

    def test_crp_cache_rejects_training_annotations(self):
        cache = self._tiny_crp_cache()
        cache["labels"] = torch.zeros(8)
        with self.assertRaisesRegex(ValueError, "forbidden annotation"):
            validate_feature_cache(cache)

    def test_crp_cache_rejects_the_previous_schema(self):
        cache = self._tiny_crp_cache()
        cache["cache_version"] = 1
        with self.assertRaisesRegex(ValueError, "Unsupported CRP cache version"):
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
            self.assertEqual(loaded["cache_version"], 2)
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
        self.assertTrue(torch.all((first["anchor_confidence"] >= 0) & (first["anchor_confidence"] <= 1)))
        self.assertTrue(all("activation_gain_alignment" in group for group in first["groups"]))
        self.assertEqual(first["artifact"], "splice_crp_v3_teacher_graph")
        self.assertEqual(first["degree_stats"]["indegree_cap"], 10)
        self.assertEqual(first["degree_stats"]["indegree_rule"], "absolute")
        self.assertNotIn("cross_fold_summary", first)
        self.assertTrue(all("cross_fold" not in group for group in first["groups"]))

    def test_crp_audit_can_cap_null_passing_groups_without_labels(self):
        config = CrpAuditConfig(
            min_concept_frequency=0.1,
            max_concept_frequency=0.9,
            projected_neighbors=3,
            graph_top_k=2,
            null_trials=1,
            null_quantile=0.0,
            min_coverage=0.0,
            max_selected_groups=1,
            seed=7,
        )
        graph = run_frozen_audit(self._tiny_crp_cache(), config)
        self.assertLessEqual(len(graph["selected_group_ids"]), 1)
        self.assertEqual(graph["config"]["max_selected_groups"], 1)

    def test_crp_audit_can_use_projected_candidates_without_residual_gate(self):
        config = CrpAuditConfig(
            min_concept_frequency=0.1,
            max_concept_frequency=0.9,
            projected_neighbors=3,
            graph_top_k=2,
            null_trials=1,
            null_quantile=0.0,
            min_coverage=0.0,
            seed=7,
            use_residual_splice_gate=False,
        )
        graph = run_frozen_audit(self._tiny_crp_cache(), config)
        self.assertTrue(all(group["semantic_agreement"] == 1.0 for group in graph["groups"]))

    def test_crp_regularizer_does_not_modify_simclr_positives(self):
        regularizer = CrpRelationalRegularizer(
            validate_teacher_graph(self._tiny_teacher_graph()),
            weight=0.1,
            temperature=0.1,
            start_epoch=0,
            warmup_epochs=0,
        )
        self.assertFalse(regularizer.uses_graph_positives)

    def test_cobalt_memberships_are_aligned_and_concept_balanced_without_labels(self):
        artifact = {
            "artifact": "cobalt_concepts_v1",
            "dataset": "waterbirds",
            "seed": 3,
            "model_config": {"codebook_size": 2},
            "splits": {
                "train": {
                    "sample_ids": torch.tensor([2, 0, 1, 3]),
                    "concepts": torch.tensor([[0], [0], [0], [1]]),
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "concepts.pt"
            torch.save(artifact, path)
            aligned, provenance = load_cobalt_train_concepts(
                path,
                "waterbirds",
                ["waterbirds:0", "waterbirds:1", "waterbirds:2", "waterbirds:3"],
            )
        weights, summary = concept_balanced_sample_weights(aligned)
        torch.testing.assert_close(weights, torch.tensor([2 / 3, 2 / 3, 2 / 3, 2.0]))
        self.assertEqual(provenance["active_concept_count"], 2)
        self.assertEqual(summary["concept_sample_counts"], [3, 1])

    def test_crp_grouping_accepts_cobalt_concept_balancing(self):
        config = CrpAuditConfig(
            min_concept_frequency=0.1,
            max_concept_frequency=0.9,
            projected_neighbors=3,
            graph_top_k=2,
            null_trials=1,
            null_quantile=0.0,
            min_coverage=0.0,
            seed=7,
            cobalt=True,
        )
        concepts = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]])
        graph = run_frozen_audit(self._tiny_crp_cache(), config, cobalt_concepts=concepts)
        self.assertTrue(graph["config"]["cobalt"])
        self.assertEqual(graph["cobalt_check"]["concept_sample_counts"], [4, 4])

    def test_cobalt_confidence_downweights_uncertain_memberships(self):
        concepts = torch.tensor([[0], [0], [1], [1]])
        confidence = torch.tensor([0.1, 1.0, 1.0, 1.0])
        weights, summary = concept_balanced_sample_weights(concepts, confidence)
        self.assertTrue(summary["confidence_enabled"])
        self.assertAlmostEqual(float(summary["confidence_min"]), 0.1)
        self.assertLess(float(weights[0]), float(weights[1]))

    def test_cobalt_weights_rebalance_grouping_frequency_filter(self):
        codes = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        dictionary = torch.eye(2)
        config = CrpAuditConfig(
            min_concept_frequency=0.2,
            max_concept_frequency=0.6,
            text_similarity_threshold=0.99,
            coactivation_threshold=0.99,
        )
        self.assertEqual(group_concepts(codes, dictionary, ["concept_a", "concept_b"], config), [[1]])
        weights = torch.tensor([2 / 3, 2 / 3, 2 / 3, 2.0])
        self.assertEqual(
            group_concepts(codes, dictionary, ["concept_a", "concept_b"], config, weights),
            [[0], [1]],
        )

    def test_cqt_quotient_removes_only_the_rank_one_state_contrast(self):
        embeddings = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]), dim=1
        )
        contrast = torch.tensor([1.0, -1.0, 0.0])
        quotient = concept_quotient(embeddings, contrast)
        torch.testing.assert_close(
            quotient @ torch.nn.functional.normalize(contrast, dim=0),
            torch.zeros(2),
            atol=1e-6,
            rtol=0,
        )
        self.assertTrue(torch.all(quotient[:, 2] > 0))

    def test_cqt_audit_is_deterministic_interpretable_and_crp_compatible(self):
        config = CqtAuditConfig(
            min_concept_frequency=0.1,
            max_concept_frequency=0.9,
            text_similarity_threshold=0.99,
            coactivation_threshold=0.99,
            max_candidate_groups=8,
            max_factors=4,
            min_state_samples=2,
            min_context_similarity=0.0,
            min_state_balanced_accuracy=0.5,
            min_quotient_efficacy=0.0,
            transport_candidates=6,
            transport_mass=0.5,
            min_transport_pairs=2,
            min_word_similarity=0.0,
            min_coverage=0.0,
            null_trials=1,
            null_quantile=0.0,
            graph_top_k=2,
            seed=3,
        )
        first = run_cqt_audit(self._tiny_cqt_cache(), config)
        second = run_cqt_audit(self._tiny_cqt_cache(), config)
        self.assertEqual(first["artifact"], CQT_ARTIFACT)
        self.assertGreater(len(first["factors"]), 0)
        self.assertGreater(len(first["selected_factor_ids"]), 0)
        self.assertIn("concepts", first["factors"][0]["state_a"])
        self.assertIn("representative_pairs", first["factors"][0])
        json.dumps(
            {
                "config": first["config"],
                "factors": first["factors"],
                "degree_stats": first["degree_stats"],
            }
        )
        torch.testing.assert_close(first["neighbor_indices"], second["neighbor_indices"])
        torch.testing.assert_close(first["weights"], second["weights"])
        graph = validate_teacher_graph(first, first["sample_ids"])
        row_sums = graph["weights"].sum(dim=1)
        supported = row_sums > 0
        torch.testing.assert_close(row_sums[supported], torch.ones_like(row_sums[supported]))

    def test_crp_teacher_graph_is_bound_to_exact_training_order(self):
        graph = validate_teacher_graph(
            self._tiny_teacher_graph(),
            [f"waterbirds:{index}" for index in range(4)],
        )
        self.assertEqual(graph["neighbor_indices"].shape, (4, 2))
        with self.assertRaisesRegex(ValueError, "do not exactly match"):
            validate_teacher_graph(graph, ["waterbirds:1", "waterbirds:0", "waterbirds:2", "waterbirds:3"])

    def test_crp_teacher_graph_rejects_hidden_annotations(self):
        graph = self._tiny_teacher_graph()
        graph["provenance"] = {"labels": [0, 1, 0, 1]}
        with self.assertRaisesRegex(ValueError, "forbidden annotation"):
            validate_teacher_graph(graph)

    def test_crp_batch_sampler_visits_every_anchor_and_adds_graph_donors(self):
        graph = validate_teacher_graph(self._tiny_teacher_graph())
        sampler = CrpGraphBatchSampler(
            graph["neighbor_indices"],
            graph["weights"],
            batch_size=4,
            generator=torch.Generator().manual_seed(5),
        )
        batches = list(sampler)
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), [0, 1, 2, 3])
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(
            any(
                neighbor in batch
                for batch in batches
                for anchor in batch
                for neighbor in graph["neighbor_indices"][anchor].tolist()
                if neighbor >= 0
            )
        )

    def test_crp_training_dataset_does_not_read_labels_or_metadata(self):
        class ImagesOnlySource:
            def get_input(self, index):
                return f"image-{index}"

        class AnnotationReturningSubset:
            indices = np.asarray([4, 9])
            dataset = ImagesOnlySource()
            transform = staticmethod(lambda image: [f"view-a:{image}", f"view-b:{image}"])
            collate = None

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, index):
                raise AssertionError("CRP training must bypass annotation-returning __getitem__.")

        dataset = IndexedCrpDataset(AnnotationReturningSubset())
        self.assertEqual(dataset[1], (["view-a:image-9", "view-b:image-9"], 1))

    def test_crp_relational_loss_prefers_teacher_aligned_student_geometry(self):
        graph = validate_teacher_graph(self._tiny_teacher_graph())
        regularizer = CrpRelationalRegularizer(
            graph, weight=1.0, temperature=0.1, start_epoch=0, warmup_epochs=0
        )
        regularizer.set_epoch(1)
        aligned = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        misaligned = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        sample_indices = torch.arange(4)
        aligned_loss = regularizer(torch.cat([aligned, aligned]), sample_indices)
        misaligned_loss = regularizer(torch.cat([misaligned, misaligned]), sample_indices)
        self.assertTrue(torch.isfinite(aligned_loss))
        self.assertLess(float(aligned_loss), float(misaligned_loss))

    def test_crp_schedule_has_pure_simclr_start_and_linear_ramp(self):
        graph = validate_teacher_graph(self._tiny_teacher_graph())
        regularizer = CrpRelationalRegularizer(
            graph, weight=0.2, temperature=0.1, start_epoch=2, warmup_epochs=2
        )
        regularizer.set_epoch(2)
        self.assertEqual(regularizer.scheduled_weight, 0.0)
        regularizer.set_epoch(3)
        self.assertAlmostEqual(regularizer.scheduled_weight, 0.1)
        regularizer.set_epoch(4)
        self.assertAlmostEqual(regularizer.scheduled_weight, 0.2)

    def test_crp_schedule_decays_to_zero(self):
        graph = validate_teacher_graph(self._tiny_teacher_graph())
        regularizer = CrpRelationalRegularizer(
            graph,
            weight=0.2,
            temperature=0.1,
            start_epoch=0,
            warmup_epochs=0,
            decay_start_epoch=4,
            decay_end_epoch=8,
        )
        regularizer.set_epoch(4)
        self.assertAlmostEqual(regularizer.scheduled_weight, 0.2)
        regularizer.set_epoch(6)
        self.assertAlmostEqual(regularizer.scheduled_weight, 0.1)
        regularizer.set_epoch(8)
        self.assertEqual(regularizer.scheduled_weight, 0.0)

    def test_graph_neighbors_become_symmetric_extra_positives(self):
        graph = validate_teacher_graph(self._tiny_teacher_graph())
        regularizer = CrpRelationalRegularizer(
            graph, weight=0.1, temperature=0.1, start_epoch=0, warmup_epochs=0
        )
        regularizer.set_epoch(1)
        mask = regularizer.batch_positive_mask(torch.tensor([0, 2, 1, 3]))
        expected = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.6],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.6, 0.0, 0.0],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_teacher_graph_json_round_trip_and_oracle_audit(self):
        graph = self._tiny_teacher_graph()
        graph.update(
            {
                "artifact": CQT_ARTIFACT,
                "group_ids": torch.tensor([[0, -1], [0, -1], [0, -1], [0, -1]]),
                "intervention_gains": torch.tensor(
                    [[0.2, 0.0], [0.2, 0.0], [0.1, 0.0], [0.1, 0.0]]
                ),
                "factors": [
                    {
                        "factor_id": 0,
                        "selected": True,
                        "state_a": {"concepts": ["state_a"]},
                        "state_b": {"concepts": ["state_b"]},
                        "preserved_concepts": ["shared_feature"],
                    }
                ],
                "selected_factor_ids": [0],
            }
        )
        metadata = [
            {"y": "0", "place": "0"},
            {"y": "0", "place": "1"},
            {"y": "0", "place": "0"},
            {"y": "1", "place": "1"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "teacher_graph.json"
            save_graph_json(graph, graph_path)
            restored = load_graph_json(graph_path)
            self.assertTrue(torch.equal(restored["neighbor_indices"], graph["neighbor_indices"]))
            report = audit_graph(restored, metadata)
        self.assertEqual(report["removed_concepts"], ["state_a", "state_b"])
        self.assertAlmostEqual(report["graph_metrics"]["desired_relation_rate"], 0.5)
        self.assertAlmostEqual(report["graph_metrics"]["different_target_rate"], 0.5)

    def test_crp_concept_report_ranks_graph_usage(self):
        graph = self._tiny_teacher_graph()
        graph.update(
            {
                "group_ids": torch.tensor([[0, -1], [0, -1], [0, -1], [0, -1]]),
                "edge_confidences": torch.tensor(
                    [[0.5, 0.0], [0.4, 0.0], [0.3, 0.0], [0.2, 0.0]]
                ),
                "intervention_gains": torch.tensor(
                    [[0.2, 0.0], [0.2, 0.0], [0.1, 0.0], [0.1, 0.0]]
                ),
                "groups": [
                    {
                        "group_id": 0,
                        "concepts": ["concept_a", "concept_b"],
                        "selected": True,
                        "score": 0.3,
                        "null_threshold": 0.1,
                        "null_excess_score": 0.2,
                        "null_excess_ratio": 2 / 3,
                        "coverage": 1.0,
                        "robust_positive_gain": 0.15,
                        "semantic_agreement": 0.8,
                    }
                ],
            }
        )
        report = build_crp_concept_report(graph)
        self.assertEqual(report["teacher_projected_concepts"], ["concept_a", "concept_b"])
        self.assertEqual(report["important_concepts"][0]["concept"], "concept_a")
        self.assertAlmostEqual(report["groups"][0]["training_evidence_mass"], 1.4, places=6)

    def test_empty_crp_graph_regularizer_has_zero_loss(self):
        graph = self._tiny_teacher_graph()
        graph["neighbor_indices"] = torch.full((4, 2), -1)
        graph["weights"] = torch.zeros(4, 2)
        graph["confidence"] = torch.zeros(4)
        graph["anchor_confidence"] = torch.zeros(4)
        graph = validate_teacher_graph(graph)
        regularizer = CrpRelationalRegularizer(
            graph, weight=0.1, temperature=0.1, start_epoch=0, warmup_epochs=0
        )
        regularizer.set_epoch(1)
        loss = regularizer(torch.randn(8, 3), torch.arange(4))
        self.assertEqual(float(loss), 0.0)

    def test_empty_relational_graph_returns_original_simclr_loader(self):
        loader = SimpleNamespace(
            dataset=SimpleNamespace(indices=np.asarray([0, 1])),
            generator=torch.Generator().manual_seed(0),
        )
        graph = {
            "artifact": "splice_cqt_v1_teacher_graph",
            "config": {},
            "neighbor_indices": torch.full((2, 1), -1),
            "weights": torch.zeros(2, 1),
            "degree_stats": {"edge_count": 0, "coverage": 0.0},
        }
        args = argparse.Namespace(
            dataset="waterbirds",
            splice_mode="cqt_relational",
            batch_size=2,
            num_workers=0,
            seed=0,
            crp_teacher_graph="teacher_graph.json",
            crp_graph_fingerprint="digest",
            splice_score_threshold=None,
            splice_score_quantile=0.75,
            splice_routing_mode="semantic",
        )

        with (
            patch.object(
                spur_splice,
                "DATASET_REGISTRY",
                {"waterbirds": {"ssl_loader": lambda *args, **kwargs: loader}},
            ),
            patch.object(spur_splice, "build_dataset_config", return_value={}),
            patch.object(spur_splice, "make_dataloader_kwargs", return_value={}),
            patch.object(spur_splice, "load_teacher_graph", return_value=(graph, "digest")),
            patch.object(spur_splice, "build_crp_training_loader") as build_crp_loader,
        ):
            result = spur_splice.build_ssl_loader(args)

        self.assertIs(result, loader)
        self.assertTrue(args.relational_graph_empty)
        build_crp_loader.assert_not_called()

    def test_crp_relational_loss_reaches_simclr_encoder(self):
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(2, 3)
                self.head = torch.nn.Linear(3, 2)

        graph = validate_teacher_graph(self._tiny_teacher_graph())
        regularizer = CrpRelationalRegularizer(
            graph, weight=0.1, temperature=0.1, start_epoch=0, warmup_epochs=0
        )
        regularizer.set_epoch(1)
        model = TinyEncoder()
        images = [torch.randn(4, 2), torch.randn(4, 2)]
        loss, parts, _ = simclr_forward_loss(
            model,
            SimCLRLoss(temperature=0.1),
            images,
            splice_regularizer=regularizer,
            sample_indices=torch.arange(4),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.encoder.weight.grad.norm()), 0.0)
        self.assertGreaterEqual(float(parts["splice"]), 0.0)

    def test_zero_simclr_weight_uses_only_relational_kl(self):
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(2, 3)
                self.head = torch.nn.Linear(3, 2)

        graph = validate_teacher_graph(self._tiny_teacher_graph())
        regularizer = CrpRelationalRegularizer(
            graph, weight=1.0, temperature=0.1, start_epoch=0, warmup_epochs=0
        )
        regularizer.set_epoch(1)
        model = TinyEncoder()
        images = [torch.randn(4, 2), torch.randn(4, 2)]
        loss, parts, _ = simclr_forward_loss(
            model,
            SimCLRLoss(temperature=0.1),
            images,
            splice_regularizer=regularizer,
            sample_indices=torch.arange(4),
            simclr_weight=0.0,
        )
        self.assertEqual(float(parts["simclr"]), 0.0)
        self.assertAlmostEqual(float(loss), float(parts["splice"]), places=7)
        loss.backward()
        self.assertIsNone(model.head.weight.grad)
        self.assertGreater(float(model.encoder.weight.grad.norm()), 0.0)
        self.assertGreater(regularizer.last_diagnostics["supported_anchor_fraction"], 0.0)
        self.assertGreaterEqual(regularizer.last_diagnostics["unweighted_kl"], 0.0)

    def test_crp_label_free_batch_runs_through_training_loop(self):
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(2, 3)
                self.head = torch.nn.Linear(3, 2)

        class LabelFreeBatchDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                image = torch.tensor([float(index == 0), float(index != 0)])
                return [image, image + 0.01], index

        graph = validate_teacher_graph(self._tiny_teacher_graph())
        regularizer = CrpRelationalRegularizer(
            graph, weight=0.1, temperature=0.1, start_epoch=0, warmup_epochs=0
        )
        model = TinyEncoder()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        args = argparse.Namespace(
            device="cpu",
            channels_last=False,
            warm=False,
            optimizer="SGD",
            amp=False,
            print_freq=100,
        )
        metrics = train_one_epoch(
            torch.utils.data.DataLoader(LabelFreeBatchDataset(), batch_size=4),
            model,
            SimCLRLoss(temperature=0.1),
            optimizer,
            torch.amp.GradScaler("cuda", enabled=False),
            epoch=1,
            args=args,
            splice_regularizer=regularizer,
        )
        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertGreaterEqual(metrics["splice_loss"], 0.0)

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

    def test_oracle_relational_regularizer_uses_same_target_opposite_spurious_pairs(self):
        metadata = torch.tensor([[0, 0], [1, 0], [0, 1]])
        aligned = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        separated = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        regularizer = OracleRelationalRegularizer(1.0)
        aligned_loss = regularizer(aligned, targets=metadata)
        separated_loss = regularizer(separated, targets=metadata)
        self.assertLess(float(aligned_loss), float(separated_loss))

    def test_oracle_relational_metadata_reaches_ssl_loss(self):
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(2, 3)
                self.head = torch.nn.Linear(3, 2)

        model = TinyEncoder()
        images = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.8]]),
        ]
        metadata = torch.tensor([[0, 0], [1, 0], [0, 1]])
        loss, parts, _ = simclr_forward_loss(
            model,
            SimCLRLoss(temperature=0.1),
            images,
            None,
            metadata[:, 1],
            OracleRelationalRegularizer(0.1),
            metadata=metadata,
        )
        loss.backward()
        self.assertGreater(float(model.encoder.weight.grad.norm()), 0.0)
        self.assertGreater(float(parts["splice"]), 0.0)

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
            ["signals", "signal", "anchor"],
            group_means,
            {key: 1 for key in group_means},
            torch.tensor([1.5, 1.0, 0.5]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            {"spurious": {0: "s0", 1: "s1"}, "target": {0: "y0", 1: "y1"}},
            args,
        )
        self.assertEqual([candidate["concept"] for candidate in candidates], ["signals", "anchor"])

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
