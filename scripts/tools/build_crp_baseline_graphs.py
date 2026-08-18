"""Build raw-CLIP and DINO-only kNN graphs from a frozen CRP cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from splice.crp import topk_neighbors, validate_feature_cache


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
    graph_specs = {
        "raw_clip": cache["centered_clip"],
        "dino": cache["dino_embeddings"],
    }
    for name, features in graph_specs.items():
        neighbours, similarities = topk_neighbors(features, top_k, chunk_size)
        graph = {
            "artifact": "splice_crp_baseline_knn_graph",
            "graph_version": 1,
            "baseline": name,
            "sample_ids": cache["sample_ids"],
            "provenance": dict(cache.get("provenance", {})),
            **_row_stochastic_knn(neighbours, similarities),
        }
        output_path = output_dir / f"{name}_graph.pt"
        torch.save(graph, output_path)
        print(f"[INFO] Wrote {name} baseline graph to {output_path}", flush=True)


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
