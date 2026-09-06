import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from scripts.tools.search_crp_graphs import candidate_grid, load_search_config, posthoc_only
from splice.crp_graph_selection import (
    evaluate_grouping_gates,
    group_partition_jaccard,
    grouping_metrics,
    lexicographic_candidate_key,
)
from splice.crp_safe_graph import SafeCrpGraphConfig, build_safe_crp_graph
from splice.graph_io import save_graph_json


def test_search_config_and_grid_are_fixed_and_label_free():
    config = load_search_config(Path("scripts/crp_graph_search.conf"))
    candidates = candidate_grid(config)
    assert len(candidates) == 9
    assert candidates[-1]["singleton_control"]
    assert config["graph_seeds"] == [0, 1, 2]
    assert config["primary_treatment_mass_budget"] == 0.05
    assert not any(key in config for key in {"labels", "targets", "contexts", "metadata"})


def test_grouping_metrics_and_hard_gates_are_pure():
    dictionary = F.normalize(torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]), dim=1)
    codes = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    metrics = grouping_metrics(
        codes, dictionary, [[0, 1], [2]],
        source_fidelity=torch.tensor([0.95, 0.96, 0.91]),
        fidelity_threshold=0.90,
        centered_embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )
    assert metrics["compression_gain"] == pytest.approx(1 / 3)
    assert metrics["within_group_text_cosine_p10"] > 0.9
    assert metrics["within_group_coactivation_cosine_p10"] == pytest.approx(1.0)
    assert evaluate_grouping_gates(metrics, None)["passed"] is False
    assert "mini_null_passing_group" in evaluate_grouping_gates(metrics, None)["failed_gates"]


def test_transitive_group_partition_stability_and_selection_key():
    assert group_partition_jaccard([{1, 2}, {3}], [{1, 2}, {3}]) == 1.0
    assert group_partition_jaccard([{1, 2}, {3}], [{1}, {2, 3}]) < 1.0
    left = {
        "candidate_id": "b", "median_training_mass_weighted_evidence_density": 0.2,
        "minimum_replacement_edge_jaccard": 0.9, "treatment_mass_hhi": 0.2,
    }
    right = {**left, "candidate_id": "a", "median_training_mass_weighted_evidence_density": 0.2}
    assert sorted([left, right], key=lexicographic_candidate_key)[0]["candidate_id"] == "a"


def _graph_fixture(n=20):
    generator = torch.Generator().manual_seed(3)
    ids = [f"waterbirds:{i}" for i in range(n)]
    clip = F.normalize(torch.randn(n, 30, generator=generator), dim=1)
    raw_indices = torch.tensor([[((i + 1) % n), ((i + 2) % n)] for i in range(n)])
    crp_indices = torch.tensor([[((i + 3) % n), ((i + 4) % n)] for i in range(n)])
    weights = torch.full((n, 2), 0.5)
    common = {
        "sample_ids": ids, "config": {}, "confidence": torch.ones(n),
        "anchor_confidence": torch.ones(n),
        "degree_stats": {"indegree_cap": 10, "maximum_indegree": 2},
    }
    raw = {**common, "artifact": "splice_raw_clip_matched_teacher_graph", "graph_version": 1,
           "neighbor_indices": raw_indices, "weights": weights}
    crp = {**common, "artifact": "splice_crp_v3_teacher_graph", "graph_version": 3,
           "neighbor_indices": crp_indices, "weights": weights,
           "edge_confidences": torch.full((n, 2), 0.8),
           "group_ids": torch.tensor([[i % 3, (i + 1) % 3] for i in range(n)]),
           "intervention_gains": torch.full((n, 2), 0.2),
           "selected_group_ids": [0, 1, 2]}
    cache = {
        "cache_version": 1, "sample_ids": ids, "clip_embeddings": clip,
        "centered_clip": clip, "image_mean": torch.zeros(30),
        "splice_codes": torch.ones(n, 3), "dictionary": torch.eye(3, 30),
        "vocabulary": ["a", "b", "c"],
    }
    return cache, crp, raw


def test_safe_training_mass_budget_and_invariants():
    cache, crp, raw = _graph_fixture()
    safe = build_safe_crp_graph(
        cache, crp, raw,
        SafeCrpGraphConfig(
            raw_guard_k=20, min_treated_anchor_fraction=0.0,
            min_crp_weight_mass_fraction=0.0,
            max_crp_training_mass_fraction=0.05,
            max_group_training_mass_fraction=0.35,
        ),
    )
    stats = safe["degree_stats"]
    assert stats["safe_crp_training_mass_fraction"] <= 0.05 + 1e-12
    assert max(stats["safe_group_training_mass_fraction"].values(), default=0.0) <= 0.35 + 1e-12
    assert torch.equal(safe["weights"], raw["weights"])
    assert torch.equal(safe["anchor_confidence"], raw["anchor_confidence"])
    assert torch.all((safe["edge_source"] == 2).sum(dim=1) <= 1)


def test_posthoc_labels_are_optional_and_cannot_change_selection(tmp_path):
    cache, crp, raw = _graph_fixture()
    safe = build_safe_crp_graph(
        cache, crp, raw,
        SafeCrpGraphConfig(
            raw_guard_k=20, min_treated_anchor_fraction=0.0,
            min_crp_weight_mass_fraction=0.0,
            max_crp_training_mass_fraction=0.05,
            max_group_training_mass_fraction=0.35,
        ),
    )
    graph_path = tmp_path / "selected_graph.json"
    save_graph_json(safe, graph_path)
    labels_path = tmp_path / "labels.json"
    contexts_path = tmp_path / "contexts.json"
    labels_path.write_text(json.dumps([i % 2 for i in range(20)]), encoding="utf-8")
    contexts_path.write_text(json.dumps([(i // 2) % 2 for i in range(20)]), encoding="utf-8")
    config = json.loads(Path("scripts/crp_graph_search.conf").read_text(encoding="utf-8"))
    config["output"] = str(tmp_path / "search-output")
    config_path = tmp_path / "search.conf"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    report_path = posthoc_only(config_path, graph_path, labels_path, contexts_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["annotation_source"] == "explicit_posthoc_vectors"
    assert report["selection_mutated"] is False
    assert not (tmp_path / "search-output" / "selection.json").exists()
