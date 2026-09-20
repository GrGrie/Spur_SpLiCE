"""The frozen intervention audit of one concept group.

For each group the audit projects the group's subspace out of the centered CLIP embeddings, looks
at how the neighbourhood changes and scores the relations that survive. Null controls repeat the
same measurement with random subspaces and shuffled codes, so a group is kept only when it beats
what chance produces. Nothing here reads a label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from cospro.pipeline.config import CoSpRoAuditConfig
from cospro.pipeline.neighbors import orthonormal_basis, project_out, topk_neighbors


def _gini(values: torch.Tensor) -> float:
    values = values.float().sort().values
    total = float(values.sum())
    if total == 0:
        return 0.0
    n = len(values)
    positions = torch.arange(1, n + 1, dtype=values.dtype, device=values.device)
    return float((2 * (positions * values).sum() / (n * values.sum())) - (n + 1) / n)


@dataclass(frozen=True)
class _AuditGeometry:
    centered_clip: torch.Tensor
    raw_neighbours: torch.Tensor
    splice_codes: torch.Tensor
    code_dot: torch.Tensor | None = None
    code_norm_squared: torch.Tensor | None = None
    sparse_codes: Any | None = None

    def __post_init__(self):
        # SpLiCE activations are sparse even though the frozen artifact is a
        # dense tensor. Keep one CSR view so pairwise residual gates touch only
        # non-zero concepts rather than materializing [pairs, vocabulary].
        if self.splice_codes.device.type == "cpu":
            from scipy.sparse import csr_matrix
            sparse = csr_matrix(self.splice_codes.numpy())
            object.__setattr__(self, "sparse_codes", sparse)
            object.__setattr__(self, "code_norm_squared", self.splice_codes.square().sum(1))
        # A dense product is faster for tiny test/research datasets only.
        if self.sparse_codes is not None and len(self.splice_codes) <= 10000:
            object.__setattr__(self, "code_dot", torch.from_numpy((sparse @ sparse.T).toarray()))


def _candidate_code_dot(
    audit: _AuditGeometry,
    anchors: torch.Tensor,
    neighbours: torch.Tensor,
) -> torch.Tensor:
    if audit.code_dot is not None:
        return audit.code_dot[anchors, neighbours]
    if audit.sparse_codes is None:
        raise RuntimeError("Residual SpLiCE gating requires CPU-resident sparse codes.")
    flat_anchors = anchors.reshape(-1).numpy()
    flat_neighbours = neighbours.reshape(-1).numpy()
    result = torch.empty(len(flat_anchors), dtype=torch.float32)
    pair_chunk_size = 262_144
    for start in range(0, len(flat_anchors), pair_chunk_size):
        stop = min(start + pair_chunk_size, len(flat_anchors))
        products = audit.sparse_codes[flat_anchors[start:stop]].multiply(
            audit.sparse_codes[flat_neighbours[start:stop]]
        )
        result[start:stop] = torch.from_numpy(products.sum(axis=1).A1).float()
    return result.view_as(neighbours)


def _residual_splice_similarity(
    audit: _AuditGeometry,
    geometry: dict,
    excluded_concept_indices: Sequence[int],
) -> torch.Tensor:
    """Compare remaining SpLiCE codes for cached candidate pairs."""

    anchors, neighbours = geometry["anchors"], geometry["neighbours"]
    excluded = torch.as_tensor(excluded_concept_indices, dtype=torch.long)
    numerator = geometry["code_dot"].clone()
    left_norm = audit.code_norm_squared[anchors].clone()
    right_norm = audit.code_norm_squared[neighbours].clone()
    flat_anchors, flat_neighbours = anchors.reshape(-1), neighbours.reshape(-1)
    numerator_flat, left_norm_flat, right_norm_flat = (
        numerator.reshape(-1), left_norm.reshape(-1), right_norm.reshape(-1)
    )
    pair_chunk_size = 262_144
    for start in range(0, len(flat_anchors), pair_chunk_size):
        stop = min(start + pair_chunk_size, len(flat_anchors))
        left = audit.splice_codes[
            flat_anchors[start:stop].view(-1, 1), excluded.view(1, -1)
        ]
        right = audit.splice_codes[
            flat_neighbours[start:stop].view(-1, 1), excluded.view(1, -1)
        ]
        numerator_flat[start:stop] -= (left * right).sum(dim=1)
        left_norm_flat[start:stop] -= left.square().sum(dim=1)
        right_norm_flat[start:stop] -= right.square().sum(dim=1)
    denominator = (left_norm.clamp_min(0) * right_norm.clamp_min(0)).sqrt()
    similarity = numerator / denominator.clamp_min(1e-12)
    return similarity.masked_fill(denominator <= 1e-12, 0.0).clamp(0.0, 1.0)


def _neighbor_geometry(
    audit: _AuditGeometry,
    basis: torch.Tensor,
    config: CoSpRoAuditConfig,
    *,
    search_seed: int,
) -> dict:
    projected = project_out(audit.centered_clip, basis)
    neighbours, projected_similarity = topk_neighbors(
        projected,
        config.projected_neighbors,
        config.similarity_chunk_size,
        backend=config.neighbor_backend,
        ann_threshold=config.ann_threshold,
        ann_tables=config.ann_tables,
        ann_bucket_size=config.ann_bucket_size,
        seed=search_seed,
    )
    anchors = torch.arange(len(projected), device=projected.device).view(-1, 1).expand_as(neighbours)
    raw_similarity = (audit.centered_clip[anchors] * audit.centered_clip[neighbours]).sum(dim=2)
    gain = projected_similarity - raw_similarity
    raw_overlap = (
        neighbours.unsqueeze(2) == audit.raw_neighbours.unsqueeze(1)
    ).any(dim=2).float().mean(dim=1)
    top1_neighbor_turnover = float(
        (neighbours[:, 0] != audit.raw_neighbours[:, 0]).float().mean()
    )
    mean_neighbor_turnover = float(1.0 - raw_overlap.mean())
    mean_jaccard_at_k = float((raw_overlap / (2.0 - raw_overlap)).mean())
    anchors, neighbours = anchors.cpu(), neighbours.cpu()
    geometry = {
        "anchors": anchors,
        "raw_neighbours": audit.raw_neighbours.cpu(),
        "neighbours": neighbours,
        "projected_similarity": projected_similarity.cpu(),
        "gain": gain.cpu(),
        "top1_neighbor_turnover": top1_neighbor_turnover,
        "mean_neighbor_turnover": mean_neighbor_turnover,
        "mean_jaccard_at_k": mean_jaccard_at_k,
    }
    if config.use_residual_splice_gate:
        geometry["code_dot"] = _candidate_code_dot(audit, anchors, neighbours)
    return geometry


def _relation_geometry(
    audit: _AuditGeometry,
    neighbour_geometry: dict,
    config: CoSpRoAuditConfig,
    excluded_concept_indices: Sequence[int],
) -> dict:
    projected_similarity = neighbour_geometry["projected_similarity"]
    residual_similarity = torch.ones_like(projected_similarity)
    residual_support = torch.ones_like(projected_similarity, dtype=torch.bool)
    if config.use_residual_splice_gate:
        residual_similarity = _residual_splice_similarity(
            audit, neighbour_geometry, excluded_concept_indices,
        )
        residual_support = residual_similarity >= config.residual_splice_similarity_threshold
    return {
        **{key: value for key, value in neighbour_geometry.items() if key != "code_dot"},
        "semantic_similarity": residual_similarity,
        "residual_splice_similarity": residual_similarity,
        "residual_splice_support": residual_support,
        "supported": residual_support,
    }


def _score_relations(geometry: dict, activation: torch.Tensor, config: CoSpRoAuditConfig) -> dict:
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
    edge_semantic = geometry["semantic_similarity"][rows, positions]
    confidence = edge_gain * (0.5 + 0.5 * edge_semantic)
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
    covered = torch.zeros(
        len(neighbours), dtype=torch.bool, device=neighbours.device
    )
    if rows.numel():
        covered[rows] = True
    indegree = torch.bincount(columns, minlength=len(neighbours))
    positive_gain = float(edge_gain.median()) if edge_gain.numel() else 0.0
    semantic_agreement = float(edge_semantic.mean()) if edge_semantic.numel() else 0.0
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
    config: CoSpRoAuditConfig,
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
