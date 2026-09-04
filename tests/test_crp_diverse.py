import unittest

import torch
import torch.nn.functional as F

from splice.crp import CrpAuditConfig
from splice.crp_diverse import (
    DiverseSelectionConfig,
    preselect_diverse_candidates,
    run_diverse_frozen_audit,
    select_diverse_evidence,
)
from splice.crp_training import validate_teacher_graph
from splice.spatial_balance import SPATIAL_BALANCE_ARTIFACT


class DiverseCrpTests(unittest.TestCase):
    def test_preaudit_budget_spans_semantic_clusters_deterministically(self):
        dictionary = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.1, 0.9, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.1, 0.9],
                ]
            ),
            dim=1,
        )
        groups = [[index] for index in range(6)]
        codes = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )
        spatial = {
            "concept_indices": torch.tensor(
                [[0, 2, 4], [1, 3, 5], [0, 2, 4], [1, 3, 5]]
            )
        }
        config = DiverseSelectionConfig(
            semantic_cluster_count=3,
            candidates_per_cluster=1,
            candidate_budget=3,
        )
        first, first_stats, assignments = preselect_diverse_candidates(
            codes, dictionary, groups, spatial, config
        )
        second, _, _ = preselect_diverse_candidates(
            codes, dictionary, groups, spatial, config
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len({int(assignments[index]) for index in first}), 3)
        self.assertTrue(all("preaudit_score" in item for item in first_stats))

    def test_final_selector_enforces_cluster_cap_and_semantic_ceiling(self):
        centroids = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.99, 0.01, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            dim=1,
        )
        audited = [
            {
                "null_excess_score": score,
                "preaudit_score": 1.0,
                "spatial_support": 0.5,
                "semantic_cluster": cluster,
            }
            for score, cluster in [(4.0, 0), (3.0, 0), (2.0, 1), (1.0, 2)]
        ]
        passing = []
        for group_id in range(4):
            passing.append(
                (
                    group_id,
                    {
                        "rows": torch.tensor([group_id]),
                        "columns": torch.tensor([(group_id + 1) % 4]),
                    },
                )
            )
        selected, trace = select_diverse_evidence(
            passing,
            audited,
            centroids,
            DiverseSelectionConfig(
                semantic_cluster_count=3,
                candidates_per_cluster=2,
                candidate_budget=4,
                selected_group_count=3,
                max_selected_per_cluster=1,
                semantic_similarity_ceiling=0.8,
            ),
        )
        selected_ids = [group_id for group_id, _ in selected]
        self.assertEqual(selected_ids, [0, 2, 3])
        self.assertEqual(len(trace), 3)
        self.assertEqual(len({audited[index]["semantic_cluster"] for index in selected_ids}), 3)

    def test_isolated_audit_emits_a_trainer_compatible_crpv4_graph(self):
        generator = torch.Generator().manual_seed(7)
        sample_ids = [f"waterbirds:{index}" for index in range(12)]
        dictionary = torch.eye(4)
        codes = torch.zeros(12, 4)
        for index in range(12):
            codes[index, index % 4] = 1.0
            codes[index, (index + 1) % 4] = 0.5
        cache = {
            "cache_version": 1,
            "provenance": {
                "dataset": "waterbirds",
                "split": "train",
                "splice_model": "open_clip:ViT-B-32",
                "splice_pretrained": "fixture",
                "splice_vocab": "fixture",
                "splice_vocab_size": 4,
                "splice_l1_penalty": 0.25,
            },
            "sample_ids": sample_ids,
            "clip_embeddings": F.normalize(torch.randn(12, 4, generator=generator), dim=1),
            "image_mean": torch.zeros(4),
            "splice_codes": codes,
            "dictionary": dictionary,
            "vocabulary": ["alpha", "beta", "gamma", "delta"],
        }
        spatial = {
            "artifact": SPATIAL_BALANCE_ARTIFACT,
            "dataset": "waterbirds",
            "sample_ids": sample_ids,
            "vocabulary": cache["vocabulary"],
            "variant": "vanilla_patchwise",
            "concept_indices": torch.tensor(
                [[index % 4, (index + 1) % 4] for index in range(12)]
            ),
            "evidence": torch.ones(12, 2),
            "confidence": torch.ones(12),
            "config": {},
            "cache_provenance": cache["provenance"],
        }
        graph = run_diverse_frozen_audit(
            cache,
            CrpAuditConfig(
                min_concept_frequency=0.0,
                max_concept_frequency=1.0,
                text_similarity_threshold=0.99,
                coactivation_threshold=0.99,
                min_group_size=1,
                max_selected_groups=0,
                projected_neighbors=2,
                activation_difference_quantile=0.5,
                min_intervention_gain=0.0,
                min_coverage=0.0,
                graph_top_k=2,
                max_indegree=3,
                null_trials=1,
                null_quantile=0.5,
                use_residual_splice_gate=False,
                spatial_balance=True,
                spatial_balance_variant="vanilla_patchwise",
            ),
            DiverseSelectionConfig(
                semantic_cluster_count=2,
                candidates_per_cluster=2,
                candidate_budget=4,
                selected_group_count=2,
                max_selected_per_cluster=1,
            ),
            spatial,
        )
        self.assertEqual(graph["artifact"], "splice_crp_v4_teacher_graph")
        self.assertEqual(graph["diverse_selection"]["source_group_count"], 4)
        self.assertGreaterEqual(graph["diverse_selection"]["preselected_group_count"], 2)
        self.assertLessEqual(graph["diverse_selection"]["preselected_group_count"], 4)
        validate_teacher_graph(graph, sample_ids)


if __name__ == "__main__":
    unittest.main()
