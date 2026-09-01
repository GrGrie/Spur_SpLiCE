"""Label-free frozen audit for SpLiCE-CRP v2.

The audit consumes representations that were cached in dataset order.  It does
not load target, spurious, or group annotations; those belong in a separate
post-hoc diagnostic step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from splice.graph_io import save_graph_json


CACHE_VERSION = 1
GRAPH_VERSION = 2
REQUIRED_CACHE_KEYS = {
    "cache_version",
    "sample_ids",
    "clip_embeddings",
    "image_mean",
    "splice_codes",
    "dictionary",
    "vocabulary",
}
FORBIDDEN_CACHE_KEYS = {
    "a",
    "attribute",
    "attributes",
    "group",
    "group_ids",
    "groups",
    "label",
    "labels",
    "metadata",
    "spurious",
    "target",
    "targets",
    "y",
}


@dataclass(frozen=True)
class CrpAuditConfig:
    min_concept_frequency: float = 0.01
    max_concept_frequency: float = 0.95
    text_similarity_threshold: float = 0.82
    coactivation_threshold: float = 0.35
    min_group_size: int = 1
    max_selected_groups: int = 0
    projected_neighbors: int = 20
    dino_neighbors: int = 50
    activation_difference_quantile: float = 0.75
    min_intervention_gain: float = 1e-4
    min_coverage: float = 0.01
    graph_top_k: int = 10
    indegree_factor: float = 3.0
    null_trials: int = 16
    null_quantile: float = 0.95
    seed: int = 0
    similarity_chunk_size: int = 512
    orthogonal_tolerance: float = 1e-6
    use_dino: bool = True
    cobalt: bool = False


def _validate_config(config: CrpAuditConfig) -> None:
    if not isinstance(config.use_dino, bool) or not isinstance(config.cobalt, bool):
        raise ValueError("use_dino and cobalt must be booleans.")
    probabilities = {
        "min_concept_frequency": config.min_concept_frequency,
        "max_concept_frequency": config.max_concept_frequency,
        "activation_difference_quantile": config.activation_difference_quantile,
        "min_coverage": config.min_coverage,
        "null_quantile": config.null_quantile,
    }
    for name, value in probabilities.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
    if config.min_concept_frequency > config.max_concept_frequency:
        raise ValueError("min_concept_frequency cannot exceed max_concept_frequency.")
    integer_fields = {
        "min_group_size": config.min_group_size,
        "projected_neighbors": config.projected_neighbors,
        "dino_neighbors": config.dino_neighbors,
        "graph_top_k": config.graph_top_k,
        "null_trials": config.null_trials,
        "similarity_chunk_size": config.similarity_chunk_size,
    }
    for name, value in integer_fields.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if config.max_selected_groups < 0:
        raise ValueError("max_selected_groups must be non-negative; 0 disables the cap.")


def _normalized_rows(values: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor.")
    values = values.detach().float().cpu()
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")
    norms = values.norm(dim=1)
    if torch.any(norms <= 1e-12):
        raise ValueError(f"{name} contains a zero vector.")
    return F.normalize(values, dim=1)


def validate_feature_cache(cache: dict, require_dino: bool = True) -> dict:
    """Validate and normalize the frozen cache without accepting hidden labels."""

    if not isinstance(cache, dict):
        raise ValueError("CRP cache must be a dictionary.")
    def collect_keys(value) -> set[str]:
        if not isinstance(value, dict):
            return set()
        keys = {str(key).lower() for key in value}
        for nested in value.values():
            keys.update(collect_keys(nested))
        return keys

    forbidden = FORBIDDEN_CACHE_KEYS.intersection(collect_keys(cache))
    if forbidden:
        raise ValueError(f"CRP discovery cache contains forbidden annotation keys: {sorted(forbidden)}")
    missing = REQUIRED_CACHE_KEYS.difference(cache)
    if require_dino and "dino_embeddings" not in cache:
        missing.add("dino_embeddings")
    if missing:
        raise ValueError(f"CRP cache is missing required keys: {sorted(missing)}")
    if cache["cache_version"] != CACHE_VERSION:
        raise ValueError(f"Unsupported CRP cache version {cache['cache_version']!r}; expected {CACHE_VERSION}.")
    if "provenance" in cache and not isinstance(cache["provenance"], dict):
        raise ValueError("Optional provenance must be a dictionary.")

    sample_ids = list(cache["sample_ids"])
    if not sample_ids or len(sample_ids) != len(set(map(str, sample_ids))):
        raise ValueError("sample_ids must be non-empty and unique.")
    n_samples = len(sample_ids)
    clip = _normalized_rows(cache["clip_embeddings"], "clip_embeddings")
    dino = (
        _normalized_rows(cache["dino_embeddings"], "dino_embeddings")
        if "dino_embeddings" in cache
        else None
    )
    codes = cache["splice_codes"]
    dictionary = cache["dictionary"]
    mean = cache["image_mean"]
    vocabulary = [str(word) for word in cache["vocabulary"]]

    if not isinstance(codes, torch.Tensor) or codes.ndim != 2 or codes.is_sparse:
        raise ValueError("splice_codes must be a dense rank-2 tensor.")
    codes = codes.detach().float().cpu()
    dictionary = _normalized_rows(dictionary, "dictionary")
    mean = torch.as_tensor(mean).detach().float().cpu().view(-1)
    if clip.shape[0] != n_samples or codes.shape[0] != n_samples:
        raise ValueError("All cached representations must have one row per sample_id in the same order.")
    if dino is not None and dino.shape[0] != n_samples:
        raise ValueError("DINO embeddings must have one row per sample_id in the same order.")
    if clip.shape[1] != dictionary.shape[1] or mean.numel() != clip.shape[1]:
        raise ValueError("CLIP embeddings, image_mean, and dictionary directions must share a dimension.")
    if codes.shape[1] != dictionary.shape[0] or len(vocabulary) != dictionary.shape[0]:
        raise ValueError("splice_codes, dictionary, and vocabulary must share a concept dimension.")
    if not torch.isfinite(codes).all() or torch.any(codes < 0):
        raise ValueError("splice_codes must contain finite non-negative activations.")

    centered_clip = clip - mean
    centered_norms = centered_clip.norm(dim=1)
    if torch.any(centered_norms <= 1e-12):
        raise ValueError("Centering produced a zero CLIP vector.")
    return {
        **cache,
        "sample_ids": sample_ids,
        "clip_embeddings": clip,
        "centered_clip": F.normalize(centered_clip, dim=1),
        "dino_embeddings": dino,
        "splice_codes": codes,
        "dictionary": dictionary,
        "image_mean": mean,
        "vocabulary": vocabulary,
    }


def _lexical_key(word: str) -> str:
    normalized = "".join(character for character in word.lower() if character.isalnum())
    if normalized.endswith("ies") and len(normalized) > 4:
        return normalized[:-3] + "y"
    if normalized.endswith("es") and len(normalized) > 4:
        return normalized[:-2]
    if normalized.endswith("s") and not normalized.endswith("ss") and len(normalized) > 3:
        return normalized[:-1]
    return normalized


class _DisjointSet:
    def __init__(self, values: Sequence[int]) -> None:
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def group_concepts(
    codes: torch.Tensor,
    dictionary: torch.Tensor,
    vocabulary: Sequence[str],
    config: CrpAuditConfig,
    sample_weights: torch.Tensor | None = None,
) -> list[list[int]]:
    """Group active concepts using text, coactivation, and lexical evidence."""

    sample_weights_tensor = None
    if sample_weights is not None:
        sample_weights_tensor = torch.as_tensor(sample_weights, dtype=torch.float32).cpu().view(-1)
        if sample_weights_tensor.shape != (codes.shape[0],):
            raise ValueError("sample_weights must contain one value per cached sample.")
        if not torch.isfinite(sample_weights_tensor).all() or torch.any(sample_weights_tensor < 0):
            raise ValueError("sample_weights must contain finite non-negative values.")
        if float(sample_weights_tensor.sum()) <= 0:
            raise ValueError("sample_weights must have positive total mass.")
        sample_weights_tensor = sample_weights_tensor / sample_weights_tensor.mean()

    occurrences = (codes > 0).float()
    frequency = (
        occurrences.mean(dim=0)
        if sample_weights_tensor is None
        else (occurrences * sample_weights_tensor.unsqueeze(1)).mean(dim=0)
    )
    active = torch.where(
        (frequency >= config.min_concept_frequency) & (frequency <= config.max_concept_frequency)
    )[0]
    active_indices = [int(index) for index in active.tolist()]
    if not active_indices:
        return []

    active_codes = codes[:, active].T
    if sample_weights_tensor is not None:
        active_codes = active_codes * sample_weights_tensor.sqrt().unsqueeze(0)
    active_codes = F.normalize(active_codes, dim=1)
    active_dictionary = F.normalize(dictionary[active], dim=1)
    families: dict[str, int] = {}
    groups = _DisjointSet(active_indices)
    for index in active_indices:
        family = _lexical_key(vocabulary[index])
        if family in families:
            groups.union(index, families[family])
        else:
            families[family] = index

    chunk_size = config.similarity_chunk_size
    for start in range(0, len(active_indices), chunk_size):
        stop = min(start + chunk_size, len(active_indices))
        text_similarity = active_dictionary[start:stop] @ active_dictionary.T
        code_similarity = active_codes[start:stop] @ active_codes.T
        matches = (text_similarity >= config.text_similarity_threshold) & (
            code_similarity >= config.coactivation_threshold
        )
        rows, columns = torch.where(matches)
        for row, column in zip(rows.tolist(), columns.tolist()):
            left_position = start + row
            if left_position < column:
                groups.union(active_indices[left_position], active_indices[column])

    result: dict[int, list[int]] = {}
    for index in active_indices:
        result.setdefault(groups.find(index), []).append(index)
    return sorted(
        (sorted(indices) for indices in result.values() if len(indices) >= config.min_group_size),
        key=lambda indices: (indices[0], len(indices)),
    )


def orthonormal_basis(directions: torch.Tensor, tolerance: float = 1e-6) -> torch.Tensor:
    """Return a column basis, dropping near-collinear dictionary directions."""

    if directions.ndim != 2 or directions.numel() == 0:
        raise ValueError("directions must be a non-empty rank-2 tensor.")
    _, singular_values, right_vectors = torch.linalg.svd(directions.float(), full_matrices=False)
    cutoff = max(float(singular_values.max()) * tolerance, tolerance)
    rank = int((singular_values > cutoff).sum().item())
    if rank == 0:
        raise ValueError("Concept directions do not span a numerically stable subspace.")
    return right_vectors[:rank].T.contiguous()


def project_out(centered_embeddings: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project full centered CLIP vectors away from a concept subspace."""

    residual = centered_embeddings - (centered_embeddings @ basis) @ basis.T
    norms = residual.norm(dim=1)
    if torch.any(norms <= 1e-12):
        raise ValueError("Projection removed an entire image embedding.")
    return F.normalize(residual, dim=1)


def topk_neighbors(features: torch.Tensor, k: int, chunk_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact cosine neighbours computed in chunks to bound peak memory."""

    n_samples = len(features)
    if n_samples < 2:
        raise ValueError("At least two samples are required to construct relations.")
    k = min(k, n_samples - 1)
    indices, similarities = [], []
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        similarity = features[start:stop] @ features.T
        local_rows = torch.arange(stop - start)
        similarity[local_rows, torch.arange(start, stop)] = -torch.inf
        values, neighbours = similarity.topk(k, dim=1)
        indices.append(neighbours)
        similarities.append(values)
    return torch.cat(indices), torch.cat(similarities)


def _edge_membership(
    reference_neighbours: torch.Tensor,
    query_rows: torch.Tensor,
    query_columns: torch.Tensor,
) -> torch.Tensor:
    """Test directed-edge membership without an O(n_samples^2) matrix."""

    n_samples = len(reference_neighbours)
    reference_rows = torch.arange(n_samples).view(-1, 1).expand_as(reference_neighbours)
    reference_keys = (reference_rows * n_samples + reference_neighbours).flatten().sort().values
    query_keys = (query_rows * n_samples + query_columns).flatten()
    positions = torch.searchsorted(reference_keys, query_keys)
    positions = positions.clamp_max(len(reference_keys) - 1)
    return (reference_keys[positions] == query_keys).view_as(query_rows)


def _gini(values: torch.Tensor) -> float:
    values = values.float().sort().values
    total = float(values.sum())
    if total == 0:
        return 0.0
    n = len(values)
    positions = torch.arange(1, n + 1, dtype=values.dtype)
    return float((2 * (positions * values).sum() / (n * values.sum())) - (n + 1) / n)


@dataclass(frozen=True)
class _AuditGeometry:
    centered_clip: torch.Tensor
    raw_neighbours: torch.Tensor
    dino_embeddings: torch.Tensor | None
    dino_neighbours: torch.Tensor | None


def _relation_geometry(
    audit: _AuditGeometry,
    basis: torch.Tensor,
    config: CrpAuditConfig,
) -> dict:
    projected = project_out(audit.centered_clip, basis)
    neighbours, projected_similarity = topk_neighbors(
        projected, config.projected_neighbors, config.similarity_chunk_size
    )
    anchors = torch.arange(len(projected)).view(-1, 1).expand_as(neighbours)
    raw_similarity = (audit.centered_clip[anchors] * audit.centered_clip[neighbours]).sum(dim=2)
    gain = projected_similarity - raw_similarity
    reciprocal = _edge_membership(neighbours, neighbours, anchors)
    if config.use_dino:
        if audit.dino_embeddings is None or audit.dino_neighbours is None:
            raise ValueError("use_dino=true requires DINO embeddings in the frozen cache.")
        dino_support = _edge_membership(audit.dino_neighbours, anchors, neighbours)
        dino_similarity = (
            audit.dino_embeddings[anchors] * audit.dino_embeddings[neighbours]
        ).sum(dim=2).clamp_min(0)
    else:
        dino_support = torch.ones_like(reciprocal)
        dino_similarity = torch.ones_like(projected_similarity)
    raw_overlap = (
        neighbours.unsqueeze(2) == audit.raw_neighbours.unsqueeze(1)
    ).any(dim=2).float().mean(dim=1)
    return {
        "anchors": anchors,
        "neighbours": neighbours,
        "projected_similarity": projected_similarity,
        "gain": gain,
        "semantic_similarity": dino_similarity,
        "supported": reciprocal & dino_support,
        "top1_neighbor_turnover": float(
            (neighbours[:, 0] != audit.raw_neighbours[:, 0]).float().mean()
        ),
        "mean_neighbor_turnover": float(1.0 - raw_overlap.mean()),
        "mean_jaccard_at_k": float((raw_overlap / (2.0 - raw_overlap)).mean()),
    }


def _score_relations(geometry: dict, activation: torch.Tensor, config: CrpAuditConfig) -> dict:
    anchors = geometry["anchors"]
    neighbours = geometry["neighbours"]
    gain = geometry["gain"]
    activation_difference = (activation[anchors] - activation[neighbours]).abs()
    positive_differences = activation_difference[gain > config.min_intervention_gain]
    difference_threshold = (
        float(torch.quantile(positive_differences, config.activation_difference_quantile))
        if positive_differences.numel()
        else math.inf
    )
    accepted = (
        (gain > config.min_intervention_gain)
        & (activation_difference >= difference_threshold)
        & geometry["supported"]
    )
    rows, positions = torch.where(accepted)
    columns = neighbours[rows, positions]
    edge_gain = gain[rows, positions]
    edge_dino = geometry["semantic_similarity"][rows, positions]
    confidence = edge_gain * (0.5 + 0.5 * edge_dino)
    positive_gain_values = gain[gain > config.min_intervention_gain]
    positive_activation_differences = activation_difference[gain > config.min_intervention_gain]
    if positive_gain_values.numel() >= 2:
        gain_centered = positive_gain_values - positive_gain_values.mean()
        activation_centered = (
            positive_activation_differences - positive_activation_differences.mean()
        )
        denominator = gain_centered.norm() * activation_centered.norm()
        activation_gain_alignment = (
            float(torch.dot(gain_centered, activation_centered) / denominator)
            if float(denominator) > 1e-12
            else 0.0
        )
    else:
        activation_gain_alignment = 0.0
    activation_gain_alignment = max(0.0, min(1.0, activation_gain_alignment))
    covered = torch.zeros(len(neighbours), dtype=torch.bool)
    if rows.numel():
        covered[rows] = True
    indegree = torch.bincount(columns, minlength=len(neighbours))
    positive_gain = float(edge_gain.median()) if edge_gain.numel() else 0.0
    semantic_agreement = float(edge_dino.mean()) if edge_dino.numel() else 0.0
    coverage = float(covered.float().mean())
    hubness_penalty = 1.0 + _gini(indegree) + float(indegree.max()) / max(1, len(neighbours))
    score = (
        positive_gain
        * coverage
        * max(semantic_agreement, 0.0)
        * activation_gain_alignment
        / hubness_penalty
    )
    return {
        "rows": rows,
        "columns": columns,
        "confidence": confidence,
        "gain": edge_gain,
        "projected_similarity": geometry["projected_similarity"][rows, positions],
        "coverage": coverage,
        "positive_gain": positive_gain,
        "semantic_agreement": semantic_agreement,
        "hubness_penalty": hubness_penalty,
        "activation_gain_alignment": activation_gain_alignment,
        "score": score,
        "top1_neighbor_turnover": geometry["top1_neighbor_turnover"],
        "mean_neighbor_turnover": geometry["mean_neighbor_turnover"],
        "mean_jaccard_at_k": geometry["mean_jaccard_at_k"],
        "activation_difference_threshold": difference_threshold,
    }


def _null_scores(
    group_geometry: dict,
    random_geometries: Sequence[dict],
    activation: torch.Tensor,
    config: CrpAuditConfig,
    generator: torch.Generator,
) -> tuple[list[float], list[float]]:
    random_scores = [
        _score_relations(geometry, activation, config)["score"]
        for geometry in random_geometries
    ]
    shuffled_scores = []
    for _ in random_geometries:
        shuffled = activation[torch.randperm(len(activation), generator=generator)]
        shuffled_scores.append(_score_relations(group_geometry, shuffled, config)["score"])
    return random_scores, shuffled_scores


def _build_teacher_graph(
    n_samples: int,
    selected: list[tuple[int, dict]],
    config: CrpAuditConfig,
) -> dict[str, torch.Tensor | dict]:
    candidates: dict[tuple[int, int], dict[str, float | int]] = {}
    for group_id, evidence in selected:
        for row, column, confidence, gain in zip(
            evidence["rows"].tolist(),
            evidence["columns"].tolist(),
            evidence["confidence"].tolist(),
            evidence["gain"].tolist(),
        ):
            key = (row, column)
            current = candidates.get(key)
            if current is None or confidence > current["confidence"]:
                candidates[key] = {"confidence": confidence, "gain": gain, "group_id": group_id}

    by_anchor: list[list[tuple[int, dict]]] = [[] for _ in range(n_samples)]
    for (row, column), evidence in candidates.items():
        by_anchor[row].append((column, evidence))
    for edges in by_anchor:
        edges.sort(key=lambda item: (-float(item[1]["confidence"]), item[0]))
        del edges[config.graph_top_k :]

    all_edges = [(row, column, evidence) for row, edges in enumerate(by_anchor) for column, evidence in edges]
    average_indegree = len(all_edges) / max(1, n_samples)
    indegree_cap = max(1, int(math.ceil(config.indegree_factor * average_indegree)))
    by_destination: dict[int, list[tuple[int, int, dict]]] = {}
    for edge in all_edges:
        by_destination.setdefault(edge[1], []).append(edge)
    retained = set()
    for edges in by_destination.values():
        edges.sort(key=lambda edge: (-float(edge[2]["confidence"]), edge[0]))
        retained.update((row, column) for row, column, _ in edges[:indegree_cap])

    indices = torch.full((n_samples, config.graph_top_k), -1, dtype=torch.long)
    weights = torch.zeros((n_samples, config.graph_top_k), dtype=torch.float32)
    edge_confidences = torch.zeros_like(weights)
    group_ids = torch.full_like(indices, -1)
    gains = torch.zeros_like(weights)
    for row, edges in enumerate(by_anchor):
        kept = [(column, evidence) for column, evidence in edges if (row, column) in retained]
        if not kept:
            continue
        raw_weights = torch.tensor([float(evidence["confidence"]) for _, evidence in kept]).clamp_min(0)
        if float(raw_weights.sum()) <= 0:
            continue
        raw_weights /= raw_weights.sum()
        for position, ((column, evidence), weight) in enumerate(zip(kept, raw_weights)):
            indices[row, position] = column
            weights[row, position] = weight
            edge_confidences[row, position] = float(evidence["confidence"])
            group_ids[row, position] = int(evidence["group_id"])
            gains[row, position] = float(evidence["gain"])

    valid = indices >= 0
    indegree = torch.bincount(indices[valid], minlength=n_samples)
    row_sums = weights.sum(dim=1)
    supported = row_sums > 0
    # Keep confidence on its absolute evidence scale. Per-graph normalization made
    # a uniformly weak graph exert the same pressure as a strong one.
    anchor_confidence = edge_confidences.max(dim=1).values.clamp(0.0, 1.0)
    return {
        "neighbor_indices": indices,
        "weights": weights,
        "edge_confidences": edge_confidences,
        "group_ids": group_ids,
        "intervention_gains": gains,
        "anchor_confidence": anchor_confidence,
        # Kept for backward compatibility with existing graph-v2 artifacts.
        "confidence": row_sums,
        "degree_stats": {
            "edge_count": int(valid.sum()),
            "supported_anchors": int(supported.sum()),
            "coverage": float(supported.float().mean()),
            "maximum_indegree": int(indegree.max()) if indegree.numel() else 0,
            "indegree_cap": indegree_cap,
            "indegree_gini": _gini(indegree),
            "effective_donor_count": float((indegree.sum() ** 2) / indegree.square().sum())
            if float(indegree.square().sum()) > 0
            else 0.0,
        },
    }


def run_frozen_audit(
    cache: dict,
    config: CrpAuditConfig,
    cobalt_concepts: torch.Tensor | None = None,
) -> dict:
    """Run the label-free audit and return a versioned sparse teacher graph."""

    _validate_config(config)
    cache = validate_feature_cache(cache, require_dino=config.use_dino)
    cobalt_check = None
    sample_weights = None
    if config.cobalt:
        if cobalt_concepts is None:
            raise ValueError("cobalt=true requires aligned CoBalT train concepts.")
        from splice.cobalt_check import concept_balanced_sample_weights

        sample_weights, cobalt_check = concept_balanced_sample_weights(cobalt_concepts)
    groups = group_concepts(
        cache["splice_codes"],
        cache["dictionary"],
        cache["vocabulary"],
        config,
        sample_weights=sample_weights,
    )
    raw_neighbours, _ = topk_neighbors(
        cache["centered_clip"], config.projected_neighbors, config.similarity_chunk_size
    )
    dino_neighbours = None
    if config.use_dino:
        dino_neighbours, _ = topk_neighbors(
            cache["dino_embeddings"], config.dino_neighbors, config.similarity_chunk_size
        )
    n_samples = len(cache["sample_ids"])
    generator = torch.Generator().manual_seed(config.seed)
    audit_geometry = _AuditGeometry(
        centered_clip=cache["centered_clip"],
        raw_neighbours=raw_neighbours,
        dino_embeddings=cache["dino_embeddings"],
        dino_neighbours=dino_neighbours,
    )
    random_geometries_by_rank: dict[int, list[dict]] = {}

    audited_groups, candidate_evidence = [], []
    print(f"[INFO] Auditing {len(groups)} concept groups over {n_samples} samples", flush=True)
    report_every = max(1, len(groups) // 20)
    for group_id, concept_indices in enumerate(groups):
        basis = orthonormal_basis(cache["dictionary"][concept_indices], config.orthogonal_tolerance)
        activation = cache["splice_codes"][:, concept_indices].sum(dim=1)
        group_geometry = _relation_geometry(audit_geometry, basis, config)
        evidence = _score_relations(group_geometry, activation, config)
        basis_rank = basis.shape[1]
        if basis_rank not in random_geometries_by_rank:
            random_geometries = []
            for _ in range(config.null_trials):
                random_directions = torch.randn(
                    basis_rank,
                    cache["centered_clip"].shape[1],
                    generator=generator,
                )
                random_basis = orthonormal_basis(
                    random_directions,
                    config.orthogonal_tolerance,
                )
                random_geometries.append(_relation_geometry(audit_geometry, random_basis, config))
            random_geometries_by_rank[basis_rank] = random_geometries
        random_scores, shuffled_scores = _null_scores(
            group_geometry,
            random_geometries_by_rank[basis_rank],
            activation,
            config,
            generator,
        )
        null_scores = torch.tensor(random_scores + shuffled_scores)
        threshold = float(torch.quantile(null_scores, config.null_quantile)) if null_scores.numel() else math.inf
        selected = evidence["coverage"] >= config.min_coverage and evidence["score"] > threshold
        null_excess_score = max(0.0, evidence["score"] - threshold)
        null_excess_ratio = min(
            1.0,
            null_excess_score / max(abs(evidence["score"]), 1e-12),
        )
        group_payload = {
            "group_id": group_id,
            "concept_indices": concept_indices,
            "concepts": [cache["vocabulary"][index] for index in concept_indices],
            "basis_rank": basis_rank,
            "selected": selected,
            "score": evidence["score"],
            "null_threshold": threshold,
            "null_excess_score": null_excess_score,
            "null_excess_ratio": null_excess_ratio,
            "coverage": evidence["coverage"],
            "robust_positive_gain": evidence["positive_gain"],
            "semantic_agreement": evidence["semantic_agreement"],
            "dino_guard_enabled": config.use_dino,
            "hubness_penalty": evidence["hubness_penalty"],
            "activation_gain_alignment": evidence["activation_gain_alignment"],
            "accepted_edges": len(evidence["rows"]),
            "top1_neighbor_turnover": evidence["top1_neighbor_turnover"],
            "mean_neighbor_turnover": evidence["mean_neighbor_turnover"],
            "mean_jaccard_at_k": evidence["mean_jaccard_at_k"],
            "activation_difference_threshold": evidence["activation_difference_threshold"],
            "random_subspace_scores": random_scores,
            "shuffled_code_scores": shuffled_scores,
        }
        audited_groups.append(group_payload)
        if selected:
            candidate_evidence.append(
                (
                    group_id,
                    {
                        **evidence,
                        "confidence": evidence["confidence"] * null_excess_ratio,
                    },
                )
            )
        if (group_id + 1) % report_every == 0 or group_id + 1 == len(groups):
            print(
                f"[INFO] Audited {group_id + 1}/{len(groups)} groups; "
                f"passing_null={len(candidate_evidence)}",
                flush=True,
            )

    candidate_evidence.sort(
        key=lambda item: (
            -float(audited_groups[item[0]]["null_excess_score"]),
            -float(audited_groups[item[0]]["score"]),
            item[0],
        )
    )
    selected_evidence = (
        candidate_evidence[: config.max_selected_groups]
        if config.max_selected_groups
        else candidate_evidence
    )
    retained_group_ids = {group_id for group_id, _ in selected_evidence}
    for group in audited_groups:
        if group["selected"] and group["group_id"] not in retained_group_ids:
            group["selected"] = False
            group["rejection_reason"] = "max_selected_groups_cap"

    graph = _build_teacher_graph(n_samples, selected_evidence, config)
    config_payload = asdict(config)
    return {
        "artifact": "splice_crp_v2_teacher_graph",
        "graph_version": GRAPH_VERSION,
        "cache_version": int(cache.get("cache_version", CACHE_VERSION)),
        "sample_ids": cache["sample_ids"],
        "config": config_payload,
        "provenance": dict(cache.get("provenance", {})),
        "cobalt_check": cobalt_check,
        "groups": audited_groups,
        "selected_group_ids": [group["group_id"] for group in audited_groups if group["selected"]],
        **graph,
    }


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_feature_cache(cache: dict, path: str | Path, require_dino: bool = True) -> None:
    """Validate and atomically save a CRP cache produced by frozen encoders."""

    validate_feature_cache(cache, require_dino=require_dino)
    _atomic_torch_save(cache, Path(path))


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the label-free SpLiCE-CRP v2 frozen audit.")
    parser.add_argument("--cache", required=True, help="Frozen feature cache (.pt).")
    parser.add_argument("--output", required=True, help="Complete teacher graph output (.json).")
    parser.add_argument("--config", help="Optional JSON object overriding CrpAuditConfig fields.")
    parser.add_argument("--seed", type=int, help="Override the null-control seed.")
    parser.add_argument("--use-dino", "--use_dino", type=_parse_bool, nargs="?", const=True)
    parser.add_argument("--cobalt", type=_parse_bool, nargs="?", const=True)
    parser.add_argument("--cobalt-concepts", default="", help="Fixed CoBalT Stage-1 concept artifact.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_values = json.loads(args.config) if args.config else {}
    unknown = set(config_values).difference(CrpAuditConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown CRP audit settings: {sorted(unknown)}")
    if args.seed is not None:
        config_values["seed"] = args.seed
    if args.use_dino is not None:
        config_values["use_dino"] = args.use_dino
    if args.cobalt is not None:
        config_values["cobalt"] = args.cobalt
    config = CrpAuditConfig(**config_values)
    cache_path, output_path = Path(args.cache), Path(args.output)
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    cobalt_concepts = None
    cobalt_provenance = None
    if config.cobalt:
        if not args.cobalt_concepts:
            raise ValueError("--cobalt-concepts is required when --cobalt true.")
        from splice.cobalt_check import load_cobalt_train_concepts

        cobalt_concepts, cobalt_provenance = load_cobalt_train_concepts(
            args.cobalt_concepts,
            str(cache.get("provenance", {}).get("dataset", "")),
            cache["sample_ids"],
        )
    artifact = run_frozen_audit(cache, config, cobalt_concepts=cobalt_concepts)
    if cobalt_provenance is not None:
        artifact["cobalt_check"].update(cobalt_provenance)
    save_graph_json(artifact, output_path)
    print(f"[INFO] Wrote CRP v2 teacher graph to {output_path}")
    print(f"[INFO] Selected {len(artifact['selected_group_ids'])}/{len(artifact['groups'])} groups")


if __name__ == "__main__":
    main()
