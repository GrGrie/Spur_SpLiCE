"""Build the matched label-free SpLiCE-semantic neighbour graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from splice.crp import validate_feature_cache
from splice.crp_training import validate_teacher_graph
from splice.graph_io import load_graph_json, save_graph_json


ARTIFACT = "splice_semantic_splice_matched_teacher_graph"


def build_matched_semantic_graph(cache: dict, reference: dict) -> dict:
    """Change only donor identities while inheriting CRP row budgets and weights."""

    reference = validate_teacher_graph(reference, cache["sample_ids"])
    reconstruction = cache["splice_codes"].float() @ cache["dictionary"].float()
    features = F.normalize(reconstruction, dim=1)
    n_samples = len(cache["sample_ids"])
    indices = torch.full_like(reference["neighbor_indices"], -1)
    weights = torch.zeros_like(reference["weights"])
    indegree = torch.zeros(n_samples, dtype=torch.long)
    cap = int(reference["degree_stats"]["indegree_cap"])
    seed = int(reference.get("config", {}).get("seed", 0))
    generator = torch.Generator().manual_seed(seed)

    for row in torch.randperm(n_samples, generator=generator).tolist():
        budget = int((reference["neighbor_indices"][row] >= 0).sum())
        if not budget:
            continue
        scores = features @ features[row]
        scores[indegree >= cap] = -torch.inf
        scores[row] = -torch.inf
        values, donors = scores.topk(budget)
        if not torch.isfinite(values).all():
            raise ValueError("Cannot satisfy the reference row degrees and indegree cap.")
        indices[row, :budget] = donors
        weights[row, :budget] = reference["weights"][row][
            reference["neighbor_indices"][row] >= 0
        ].sort(descending=True).values
        indegree[donors] += 1

    valid = indices >= 0
    graph = {
        "artifact": ARTIFACT,
        "graph_version": 1,
        "cache_version": int(cache.get("cache_version", 0)),
        "sample_ids": cache["sample_ids"],
        "config": dict(reference.get("config", {})),
        "provenance": dict(cache.get("provenance", {})),
        "neighbor_indices": indices,
        "weights": weights,
        "confidence": weights.sum(1),
        "anchor_confidence": reference["anchor_confidence"].clone(),
        "degree_stats": {
            "edge_count": int(valid.sum()),
            "supported_anchors": int((weights.sum(1) > 0).sum()),
            "coverage": float((weights.sum(1) > 0).float().mean()),
            "indegree_cap": cap,
            "maximum_indegree": int(indegree.max()) if indegree.numel() else 0,
        },
        "control": (
            "normalized SpLiCE reconstruction cosine; matched CRP anchor support, "
            "row degrees, weight profile, anchor confidence, and indegree cap"
        ),
    }
    return validate_teacher_graph(graph, cache["sample_ids"])


def edge_overlap(left: dict, right: dict) -> dict:
    left_indices = left["neighbor_indices"]
    right_indices = right["neighbor_indices"]
    left_valid = left_indices >= 0
    right_valid = right_indices >= 0
    shared = (
        (left_indices[:, :, None] == right_indices[:, None, :])
        & left_valid[:, :, None]
        & right_valid[:, None, :]
    ).any(dim=2)
    denominator = int(left_valid.sum())
    return {
        "shared_edge_count": int(shared.sum()),
        "left_edge_count": denominator,
        "right_edge_count": int(right_valid.sum()),
        "left_edges_found_in_right_fraction": float(shared.sum() / denominator) if denominator else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--reference-graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overlap-output", required=True, type=Path)
    args = parser.parse_args()

    cache = validate_feature_cache(torch.load(args.cache, map_location="cpu", weights_only=True))
    reference = load_graph_json(args.reference_graph)
    semantic = build_matched_semantic_graph(cache, reference)
    save_graph_json(semantic, args.output)
    report = edge_overlap(validate_teacher_graph(reference, cache["sample_ids"]), semantic)
    args.overlap_output.parent.mkdir(parents=True, exist_ok=True)
    args.overlap_output.write_text(
        json.dumps(
            {"artifact": "matched_graph_edge_overlap_v1", "reference": str(args.reference_graph),
             "semantic": str(args.output), **report}, indent=2, sort_keys=True
        ) + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Wrote semantic graph to {args.output}")
    print(f"[INFO] Edge overlap: {report}")


if __name__ == "__main__":
    main()
