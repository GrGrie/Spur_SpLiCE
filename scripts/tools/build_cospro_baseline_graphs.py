"""Build the raw-CLIP control matched to a frozen CoSpRo teacher graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from splice.cospro import validate_splice_dataset_cache
from splice.graph_io import load_graph_json, save_graph_json


def build_matched_raw_clip_graph(cache: dict, reference: dict) -> dict:
    """Replace CoSpRo neighbours by nearest CLIP neighbours with matched row budgets.

    Match supported rows, each row's outdegree, weight profile, and confidence
    exactly. Apply the same absolute indegree cap, without using annotations.
    The complete indegree distribution and identities of donors are not matched.
    """
    from splice.cospro_training import validate_teacher_graph
    reference = validate_teacher_graph(reference, cache["sample_ids"])
    n = len(cache["sample_ids"])
    indices = torch.full_like(reference["neighbor_indices"], -1)
    weights = torch.zeros_like(reference["weights"])
    indegree = torch.zeros(n, dtype=torch.long)
    cap = int(reference["degree_stats"]["indegree_cap"])
    features = cache["centered_clip"]
    # Seeded order prevents dataset ordering from systematically receiving priority.
    generator = torch.Generator().manual_seed(int(reference.get("config", {}).get("seed", 0)))
    for row in torch.randperm(n, generator=generator).tolist():
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
        weights[row, :budget] = reference["weights"][row][reference["neighbor_indices"][row] >= 0].sort(descending=True).values
        indegree[donors] += 1
    graph = {
        "artifact": "splice_raw_clip_matched_teacher_graph", "graph_version": 1,
        "sample_ids": cache["sample_ids"], "config": dict(reference.get("config", {})),
        "provenance": dict(cache.get("provenance", {})),
        "neighbor_indices": indices, "weights": weights, "confidence": weights.sum(1),
        "anchor_confidence": reference["anchor_confidence"].clone(),
        "degree_stats": {"edge_count": int((indices >= 0).sum()),
                         "coverage": float((weights.sum(1) > 0).float().mean()),
                         "indegree_cap": cap, "maximum_indegree": int(indegree.max())},
        "control": "raw centered CLIP; matched anchor support, row degree, weight profile and confidence; same indegree cap",
    }
    return validate_teacher_graph(graph, cache["sample_ids"])


def build_graph(splice_dataset_cache_path: Path, reference_path: Path, output_path: Path) -> Path:
    cache = torch.load(splice_dataset_cache_path, map_location="cpu", weights_only=True)
    cache = validate_splice_dataset_cache(cache)
    reference = load_graph_json(reference_path)
    graph = build_matched_raw_clip_graph(cache, reference)
    save_graph_json(graph, output_path)
    print(f"[INFO] Wrote matched raw-CLIP teacher graph to {output_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splice-dataset-cache", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path, help="Canonical CoSpRo teacher graph to match.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_graph(args.splice_dataset_cache, args.reference, args.output)


if __name__ == "__main__":
    main()
