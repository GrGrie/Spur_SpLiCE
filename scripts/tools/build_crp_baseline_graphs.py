"""Build a raw-CLIP kNN baseline graph from a frozen CRP cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from splice.crp import topk_neighbors, validate_feature_cache
from splice.graph_io import save_graph_json


def build_matched_raw_clip_graph(cache: dict, reference: dict) -> dict:
    """Replace CRP neighbours by nearest CLIP neighbours with matched row budgets.

    Match supported rows, each row's outdegree, weight profile, and confidence
    exactly. Apply the same absolute indegree cap, without using annotations.
    The complete indegree distribution and identities of donors are not matched.
    """
    from splice.crp_training import validate_teacher_graph
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


def _row_stochastic_knn(neighbours: torch.Tensor, similarities: torch.Tensor) -> dict[str, torch.Tensor]:
    weights = torch.full_like(similarities, 1.0 / similarities.shape[1])
    return {
        "neighbor_indices": neighbours,
        "weights": weights,
        "similarities": similarities,
    }


def build_graphs(cache_path: Path, output_dir: Path, top_k: int, chunk_size: int) -> None:
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    cache = validate_feature_cache(cache)
    output_dir.mkdir(parents=True, exist_ok=True)
    neighbours, similarities = topk_neighbors(cache["centered_clip"], top_k, chunk_size)
    graph = {
        "artifact": "splice_crp_baseline_knn_graph",
        "graph_version": 1,
        "baseline": "raw_clip",
        "sample_ids": cache["sample_ids"],
        "provenance": dict(cache.get("provenance", {})),
        **_row_stochastic_knn(neighbours, similarities),
    }
    output_path = output_dir / "raw_clip_graph.json"
    save_graph_json(graph, output_path)
    print(f"[INFO] Wrote raw_clip baseline graph to {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    if args.top_k <= 0 or args.chunk_size <= 0:
        raise ValueError("top-k and chunk-size must be positive")
    build_graphs(args.cache, args.output_dir, args.top_k, args.chunk_size)


if __name__ == "__main__":
    main()
