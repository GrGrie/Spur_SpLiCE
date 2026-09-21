"""Diagnostics metrics on hand-checkable inputs plus an end-to-end evaluate and dashboard run."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cospro.diagnostics import graph_metrics, group_metrics
from cospro.diagnostics.dashboard import render_file
from cospro.diagnostics.evaluate import evaluate, write_record
from cospro.diagnostics.labels import load_labels
from golden_support import synthetic_splice_cache, train_sample_ids, write_waterbirds_fixture
from cospro.pipeline import CoSpRoAuditConfig, build_concept_groups, build_teacher_graph, save_splice_dataset_cache
from cospro.pipeline.graph_io import save_graph_json

Y = np.array([0, 0, 1, 1])
A = np.array([0, 1, 0, 1])


def tiny_graph(neighbours: list[int]) -> dict:
    """One edge per anchor with unit weight over four samples covering all (y, a) groups."""

    return {
        "sample_ids": [f"waterbirds:{index}" for index in range(4)],
        "neighbor_indices": torch.tensor([[value] for value in neighbours]),
        "weights": torch.ones(4, 1),
        "edge_confidences": torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
        "group_ids": torch.tensor([[0], [0], [1], [1]]),
        "groups": [
            {"group_id": 0, "concepts": ["water"], "selected": True, "score": 2.0, "null_threshold": 1.0},
            {"group_id": 1, "concepts": ["gull"], "selected": True, "score": 3.0, "null_threshold": 1.0},
        ],
    }


class GraphMetricTests(unittest.TestCase):
    def test_counterfactual_rates_match_a_hand_count(self):
        # 0 -> 1 and 1 -> 0 keep the class and flip the attribute; 2 -> 0 and 3 -> 1 change the class.
        metrics = graph_metrics.label_metrics(tiny_graph([1, 0, 0, 1]), Y, A, random_trials=2)
        self.assertAlmostEqual(metrics["counterfactual"], 0.5)
        self.assertAlmostEqual(metrics["same_class"], 0.5)
        self.assertAlmostEqual(metrics["group_balanced_counterfactual"], 0.5)
        self.assertAlmostEqual(metrics["group_balanced_same_class"], 0.5)
        np.testing.assert_allclose(np.array(metrics["transition"]).sum(axis=1), np.ones(4))

    def test_per_group_rates_split_edges_by_source_group(self):
        rows = {row["group_id"]: row for row in graph_metrics.per_concept_group_label_rates(tiny_graph([1, 0, 0, 1]), Y, A)}
        self.assertAlmostEqual(rows[0]["counterfactual"], 1.0)
        self.assertAlmostEqual(rows[1]["counterfactual"], 0.0)

    def test_structure_overlap_and_selection(self):
        graph = tiny_graph([1, 0, 0, 1])
        structure = graph_metrics.structure_metrics(graph)
        self.assertEqual((structure["edges"], structure["max_indegree"]), (4, 2))
        self.assertEqual(graph_metrics.edge_jaccard(graph, graph), 1.0)
        # Undirected edges {01, 02, 13} and {01, 23} share one of four.
        self.assertAlmostEqual(graph_metrics.edge_jaccard(graph, tiny_graph([1, 0, 3, 2])), 1 / 4)
        selection = graph_metrics.selection_metrics(graph)
        self.assertEqual(selection["selected_groups"], 2)
        self.assertAlmostEqual(selection["median_null_margin"], 2.5)


class GroupMetricTests(unittest.TestCase):
    groups = {"groups": [
        {"concept_indices": [0, 1], "concepts": ["water", "lake"]},
        {"concept_indices": [2], "concepts": ["forest"]},
    ]}

    def test_identical_partitions_agree_perfectly(self):
        agreement = group_metrics.partition_agreement(self.groups, self.groups)
        self.assertAlmostEqual(agreement["adjusted_rand_index"], 1.0)
        self.assertAlmostEqual(agreement["variation_of_information"], 0.0)

    def test_npmi_is_one_for_concepts_that_always_fire_together(self):
        codes = torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.7, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.0]])
        self.assertAlmostEqual(group_metrics.npmi_coherence(self.groups, codes)["mean"], 1.0)

    def test_selectivity_ranks_the_attribute_group_first(self):
        a = np.array([1, 1, 0, 0])
        y = np.array([1, 0, 1, 0])
        codes = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = group_metrics.spurious_selectivity(self.groups, codes, y, a)
        self.assertEqual(result["groups"][0]["concepts"], ["water", "lake"])
        self.assertAlmostEqual(result["groups"][0]["auc_a"], 1.0)
        self.assertEqual(result["fragmentation"], 1)


class EndToEndTests(unittest.TestCase):
    def test_evaluate_and_render_on_a_synthetic_sweep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_folder = write_waterbirds_fixture(root / "datasets")
            sample_ids, y, place = train_sample_ids()
            cache = synthetic_splice_cache(sample_ids, y, place)
            cache_path = root / "cache.pt"
            save_splice_dataset_cache(cache, cache_path)
            sweep = root / "concept_groups"
            for text_threshold in (0.8, 0.95):
                config = CoSpRoAuditConfig(text_similarity_threshold=text_threshold, coactivation_threshold=0.35,
                                           projected_neighbors=6, graph_top_k=3, max_indegree=6, null_trials=8)
                groups = build_concept_groups(cache, config)
                point = sweep / f"text_{text_threshold}"
                point.mkdir(parents=True)
                (point / "concept_groups.json").write_text(json.dumps(groups, default=str), encoding="utf-8")
                graph = build_teacher_graph(cache, groups, config, device="cpu")
                save_graph_json(graph, point / "teacher_graphs" / "audit" / "teacher_graph.json")
            main_graph = sweep / "text_0.8" / "teacher_graphs" / "audit" / "teacher_graph.json"

            record = evaluate(
                "waterbirds", sweep_dir=sweep, graphs={"cospro": main_graph}, reference_graph="cospro",
                cache_path=cache_path, data_folder=data_folder, bootstrap_trials=2,
            )
            self.assertEqual(record["tiers"], {"cache": True, "labels": True})
            self.assertEqual(len(record["sweep"]), 2)
            first = record["sweep"][0]
            self.assertEqual(first["grouping"]["structure"]["composite_groups"], 2)
            self.assertIn("npmi", first["grouping"])
            self.assertIn("bootstrap", first["grouping"])
            self.assertAlmostEqual(first["graphs"][0]["overlap_with_reference"], 1.0)
            audited = record["graphs"]["cospro"]["audited_groups"]
            top = max(audited, key=lambda row: row["selectivity"])
            self.assertEqual(top["concepts"], ["water", "lake", "ocean"])
            self.assertEqual(record["group_names"][1], "landbird / water")

            record_path = write_record(record, root / "diagnostics.json")
            json.loads(record_path.read_text(encoding="utf-8"))
            page = render_file(record_path, root / "dashboard.html", data_folder=data_folder).read_text(encoding="utf-8")
            for heading in ("Teacher graphs", "Concept groups", "Grouping sweep", "Edge gallery"):
                self.assertIn(heading, page)
            self.assertIn("data:image/jpeg;base64", page)

    def test_labels_follow_dataset_metadata_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = load_labels("waterbirds", write_waterbirds_fixture(Path(directory)))
            sample_ids, y, place = train_sample_ids()
            got_y, got_a = labels.for_ids(sample_ids)
            np.testing.assert_array_equal(got_y, y)
            np.testing.assert_array_equal(got_a, place)
            with self.assertRaises(ValueError):
                labels.for_ids(["celeba:0"])


if __name__ == "__main__":
    unittest.main()
