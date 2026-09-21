"""Golden snapshot of concept grouping and teacher-graph construction on a synthetic cache."""

from __future__ import annotations

import unittest

import torch

from golden_support import assert_close_tree, compare_or_update, synthetic_splice_cache, train_sample_ids
from cospro.pipeline import CoSpRoAuditConfig, build_concept_groups, build_teacher_graph

GRAPH_CONFIG = CoSpRoAuditConfig(
    text_similarity_threshold=0.8,
    coactivation_threshold=0.35,
    projected_neighbors=6,
    graph_top_k=3,
    max_indegree=6,
    null_trials=8,
)


def build_snapshot() -> dict:
    sample_ids, y, place = train_sample_ids()
    cache = synthetic_splice_cache(sample_ids, y, place)
    concept_groups = build_concept_groups(cache, GRAPH_CONFIG)
    graph = build_teacher_graph(cache, concept_groups, GRAPH_CONFIG, device="cpu")
    return {
        "concept_groups": [group["concepts"] for group in concept_groups["groups"]],
        "grouping_diagnostics": {
            key: concept_groups["diagnostics"][key]
            for key in ("total_active_concepts", "total_groups", "singleton_count", "maximum_group_size")
        },
        "audit": [
            {
                "concepts": group["concepts"],
                "selected": bool(group["selected"]),
                "score": float(group["score"]),
                "null_threshold": float(group["null_threshold"]),
                "coverage": float(group["coverage"]),
                "accepted_edges": int(group["accepted_edges"]),
            }
            for group in graph["groups"]
        ],
        "selected_group_ids": list(graph["selected_group_ids"]),
        "degree_stats": {
            key: float(value) if isinstance(value, float) else value
            for key, value in graph["degree_stats"].items()
        },
        "neighbor_indices": torch.as_tensor(graph["neighbor_indices"]).tolist(),
        "weights": [[float(value) for value in row] for row in torch.as_tensor(graph["weights"]).tolist()],
        "anchor_confidence": [float(value) for value in torch.as_tensor(graph["anchor_confidence"]).tolist()],
    }


class GoldenGraphTests(unittest.TestCase):
    def test_grouping_and_teacher_graph_match_snapshot(self):
        compare_or_update(
            "teacher_graph.json",
            build_snapshot(),
            lambda expected, actual: assert_close_tree(self, expected, actual, rel=1e-4, abs_=1e-6),
        )


if __name__ == "__main__":
    unittest.main()
