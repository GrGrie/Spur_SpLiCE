"""Label-free SpLiCE-CQT concept quotient transport audit.

CQT reuses the frozen CRP feature cache and emits the same sparse teacher-graph
interface consumed by relational SimCLR.  It changes only how graph edges are
discovered: mutually exclusive concept groups define two pseudo-states, a
rank-one concept contrast is quotiented out of CLIP, and a capacitated partial
transport plan links samples whose remaining SpLiCE semantics agree.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from splice.crp import (
    CACHE_VERSION,
    GRAPH_VERSION,
    CrpAuditConfig,
    _atomic_torch_save,
    _build_teacher_graph,
    group_concepts,
    topk_neighbors,
    validate_feature_cache,
)


CQT_ARTIFACT = "splice_cqt_v1_teacher_graph"


@dataclass(frozen=True)
class CqtAuditConfig:
    # The first five fields deliberately match CRP concept grouping.
    min_concept_frequency: float = 0.01
    max_concept_frequency: float = 0.95
    text_similarity_threshold: float = 0.80
    coactivation_threshold: float = 0.35
    min_group_size: int = 1
    max_candidate_groups: int = 128
    max_factors: int = 8
    min_state_samples: int = 16
    min_exclusivity: float = 0.80
    min_context_similarity: float = 0.20
    min_state_balanced_accuracy: float = 0.70
    min_quotient_efficacy: float = 0.50
    transport_candidates: int = 32
    transport_mass: float = 0.25
    min_transport_pairs: int = 8
    min_word_similarity: float = 0.20
    dino_neighbors: int = 10
    dino_damage_quantile: float = 0.50
    min_intervention_gain: float = 1e-4
    min_coverage: float = 0.01
    null_trials: int = 4
    null_quantile: float = 0.95
    graph_top_k: int = 10
    indegree_factor: float = 3.0
    similarity_chunk_size: int = 512
    seed: int = 0


def _validate_config(config: CqtAuditConfig) -> None:
    probabilities = {
        "min_concept_frequency": config.min_concept_frequency,
        "max_concept_frequency": config.max_concept_frequency,
        "min_exclusivity": config.min_exclusivity,
        "transport_mass": config.transport_mass,
        "dino_damage_quantile": config.dino_damage_quantile,
        "min_coverage": config.min_coverage,
        "null_quantile": config.null_quantile,
    }
    for name, value in probabilities.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
    if config.min_concept_frequency > config.max_concept_frequency:
        raise ValueError("min_concept_frequency cannot exceed max_concept_frequency.")
    positive_integers = {
        "min_group_size": config.min_group_size,
        "max_candidate_groups": config.max_candidate_groups,
        "max_factors": config.max_factors,
        "min_state_samples": config.min_state_samples,
        "transport_candidates": config.transport_candidates,
        "min_transport_pairs": config.min_transport_pairs,
        "dino_neighbors": config.dino_neighbors,
        "null_trials": config.null_trials,
        "graph_top_k": config.graph_top_k,
        "similarity_chunk_size": config.similarity_chunk_size,
    }
    for name, value in positive_integers.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if config.transport_mass <= 0:
        raise ValueError("transport_mass must be positive.")
    if config.indegree_factor <= 0:
        raise ValueError("indegree_factor must be positive.")


def concept_quotient(embeddings: torch.Tensor, contrast: torch.Tensor) -> torch.Tensor:
    """Remove one normalized concept-state contrast and preserve its complement."""

    if embeddings.ndim != 2 or contrast.ndim != 1 or embeddings.shape[1] != contrast.numel():
        raise ValueError("embeddings and contrast must have compatible dimensions.")
    contrast = contrast.detach().float().cpu()
    if not torch.isfinite(contrast).all() or float(contrast.norm()) <= 1e-12:
        raise ValueError("contrast must be a finite non-zero vector.")
    contrast = F.normalize(contrast, dim=0)
    embeddings = embeddings.detach().float().cpu()
    residual = embeddings - (embeddings @ contrast).unsqueeze(1) * contrast.unsqueeze(0)
    if torch.any(residual.norm(dim=1) <= 1e-12):
        raise ValueError("The concept quotient removed an entire image embedding.")
    return F.normalize(residual, dim=1)


def _grouping_config(config: CqtAuditConfig) -> CrpAuditConfig:
    return CrpAuditConfig(
        min_concept_frequency=config.min_concept_frequency,
        max_concept_frequency=config.max_concept_frequency,
        text_similarity_threshold=config.text_similarity_threshold,
        coactivation_threshold=config.coactivation_threshold,
        min_group_size=config.min_group_size,
        similarity_chunk_size=config.similarity_chunk_size,
    )


def _semantic_code_embeddings(codes: torch.Tensor, dictionary: torch.Tensor) -> torch.Tensor:
    # SpLiCE codes are stored densely for cache portability but are sparse in value.
    return torch.sparse.mm(codes.to_sparse(), dictionary)


def _group_activations(codes: torch.Tensor, groups: Sequence[Sequence[int]]) -> torch.Tensor:
    return torch.stack([codes[:, list(indices)].sum(dim=1) for indices in groups], dim=1)


def _crossfit_folds(n_samples: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_samples, generator=generator)
    folds = torch.zeros(n_samples, dtype=torch.long)
    folds[permutation[n_samples // 2 :]] = 1
    return folds


def _state_indices(
    activations: torch.Tensor,
    left: int,
    right: int,
    fold_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_active = activations[:, left] > 0
    right_active = activations[:, right] > 0
    return (
        torch.where(fold_mask & left_active & ~right_active)[0],
        torch.where(fold_mask & right_active & ~left_active)[0],
    )


def _context_centroid(
    semantic_codes: torch.Tensor,
    codes: torch.Tensor,
    dictionary: torch.Tensor,
    rows: torch.Tensor,
    excluded: Sequence[int],
) -> torch.Tensor | None:
    if rows.numel() == 0:
        return None
    excluded = list(excluded)
    centroid = semantic_codes[rows].mean(dim=0)
    centroid -= codes[rows][:, excluded].mean(dim=0) @ dictionary[excluded]
    if float(centroid.norm()) <= 1e-12:
        return None
    return F.normalize(centroid, dim=0)


def _candidate_groups(
    groups: list[list[int]],
    activations: torch.Tensor,
    config: CqtAuditConfig,
) -> list[list[int]]:
    frequencies = (activations > 0).float().mean(dim=0)
    ranked = sorted(
        range(len(groups)),
        key=lambda index: (-min(float(frequencies[index]), 1.0 - float(frequencies[index])), groups[index][0]),
    )
    return [groups[index] for index in ranked[: config.max_candidate_groups]]


def _propose_factors(
    groups: list[list[int]],
    activations: torch.Tensor,
    folds: torch.Tensor,
    semantic_codes: torch.Tensor,
    codes: torch.Tensor,
    dictionary: torch.Tensor,
    config: CqtAuditConfig,
) -> list[dict]:
    prototypes: dict[tuple[int, int], torch.Tensor | None] = {}
    for fold in (0, 1):
        fold_mask = folds == fold
        for group_id, indices in enumerate(groups):
            rows = torch.where(fold_mask & (activations[:, group_id] > 0))[0]
            prototypes[(fold, group_id)] = _context_centroid(
                semantic_codes, codes, dictionary, rows, indices
            )

    preliminary = []
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            fold_payloads = []
            viable = True
            for fold in (0, 1):
                fold_mask = folds == fold
                state_a, state_b = _state_indices(activations, left, right, fold_mask)
                left_count = int((fold_mask & (activations[:, left] > 0)).sum())
                right_count = int((fold_mask & (activations[:, right] > 0)).sum())
                overlap = int(
                    (fold_mask & (activations[:, left] > 0) & (activations[:, right] > 0)).sum()
                )
                denominator = max(1, min(left_count, right_count))
                exclusivity = 1.0 - overlap / denominator
                left_context, right_context = prototypes[(fold, left)], prototypes[(fold, right)]
                context_similarity = (
                    float(torch.dot(left_context, right_context))
                    if left_context is not None and right_context is not None
                    else -1.0
                )
                if (
                    len(state_a) < config.min_state_samples
                    or len(state_b) < config.min_state_samples
                    or exclusivity < config.min_exclusivity
                ):
                    viable = False
                    break
                fold_payloads.append(
                    {
                        "fold": fold,
                        "state_a_samples": len(state_a),
                        "state_b_samples": len(state_b),
                        "exclusivity": exclusivity,
                        "preliminary_context_similarity": context_similarity,
                    }
                )
            if viable:
                preliminary.append(
                    (
                        min(
                            payload["preliminary_context_similarity"]
                            for payload in fold_payloads
                        ),
                        left,
                        right,
                        fold_payloads,
                    )
                )

    preliminary.sort(key=lambda value: (-value[0], value[1], value[2]))
    proposal_pool = preliminary[: config.max_factors * 16]
    proposals = []
    for _, left, right, fold_payloads in proposal_pool:
        excluded = sorted(set(groups[left] + groups[right]))
        exact_contexts = []
        for payload in fold_payloads:
            fold_mask = folds == payload["fold"]
            state_a, state_b = _state_indices(activations, left, right, fold_mask)
            context_a = _context_centroid(
                semantic_codes, codes, dictionary, state_a, excluded
            )
            context_b = _context_centroid(
                semantic_codes, codes, dictionary, state_b, excluded
            )
            exact = (
                float(torch.dot(context_a, context_b))
                if context_a is not None and context_b is not None
                else -1.0
            )
            payload["context_similarity"] = exact
            exact_contexts.append(exact)
        minimum_context = min(exact_contexts)
        if minimum_context < config.min_context_similarity:
            continue
        state_balance = min(
            min(payload["state_a_samples"], payload["state_b_samples"])
            / max(1, payload["state_a_samples"] + payload["state_b_samples"])
            for payload in fold_payloads
        )
        proposals.append(
            {
                "left": left,
                "right": right,
                "folds": fold_payloads,
                "proposal_score": minimum_context
                * min(payload["exclusivity"] for payload in fold_payloads)
                * (2.0 * state_balance),
            }
        )
    proposals.sort(key=lambda item: (-item["proposal_score"], item["left"], item["right"]))
    return proposals[: config.max_factors]


def _balanced_centroid_accuracy(
    discovery_features: torch.Tensor,
    discovery_a: torch.Tensor,
    discovery_b: torch.Tensor,
    evaluation_features: torch.Tensor,
    evaluation_a: torch.Tensor,
    evaluation_b: torch.Tensor,
) -> float:
    centroid_a = F.normalize(discovery_features[discovery_a].mean(dim=0), dim=0)
    centroid_b = F.normalize(discovery_features[discovery_b].mean(dim=0), dim=0)
    score_a = evaluation_features[evaluation_a] @ centroid_a - evaluation_features[evaluation_a] @ centroid_b
    score_b = evaluation_features[evaluation_b] @ centroid_a - evaluation_features[evaluation_b] @ centroid_b
    return 0.5 * (float((score_a >= 0).float().mean()) + float((score_b < 0).float().mean()))


def _cross_topk(
    left: torch.Tensor,
    right: torch.Tensor,
    k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = min(k, len(right))
    rows, columns, values = [], [], []
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        similarity = left[start:stop] @ right.T
        chunk_values, chunk_columns = similarity.topk(k, dim=1)
        chunk_rows = torch.arange(start, stop).view(-1, 1).expand_as(chunk_columns)
        rows.append(chunk_rows.flatten())
        columns.append(chunk_columns.flatten())
        values.append(chunk_values.flatten())
    return torch.cat(rows), torch.cat(columns), torch.cat(values)


def _word_candidates(
    word_embeddings: torch.Tensor,
    state_a: torch.Tensor,
    state_b: torch.Tensor,
    config: CqtAuditConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_to_b = _cross_topk(
        word_embeddings[state_a],
        word_embeddings[state_b],
        config.transport_candidates,
        config.similarity_chunk_size,
    )
    b_to_a = _cross_topk(
        word_embeddings[state_b],
        word_embeddings[state_a],
        config.transport_candidates,
        config.similarity_chunk_size,
    )
    candidates: dict[tuple[int, int], float] = {}
    for source, destination, similarity in zip(*[values.tolist() for values in a_to_b]):
        candidates[(source, destination)] = max(candidates.get((source, destination), -math.inf), similarity)
    for destination, source, similarity in zip(*[values.tolist() for values in b_to_a]):
        candidates[(source, destination)] = max(candidates.get((source, destination), -math.inf), similarity)
    ordered = sorted((source, destination, similarity) for (source, destination), similarity in candidates.items())
    return (
        torch.tensor([item[0] for item in ordered], dtype=torch.long),
        torch.tensor([item[1] for item in ordered], dtype=torch.long),
        torch.tensor([item[2] for item in ordered], dtype=torch.float32),
    )


def _transport_cost(
    quotient_similarity: torch.Tensor,
    word_similarity: torch.Tensor,
    source: torch.Tensor,
    n_sources: int,
) -> torch.Tensor:
    ranks = torch.zeros_like(quotient_similarity)
    for row in range(n_sources):
        positions = torch.where(source == row)[0]
        if positions.numel() <= 1:
            continue
        order = torch.argsort(quotient_similarity[positions], descending=True, stable=True)
        ranks[positions[order]] = torch.arange(len(order), dtype=torch.float32) / (len(order) - 1)
    word_distance = (1.0 - word_similarity.clamp(-1, 1)) / 2.0
    return 0.5 * ranks + 0.5 * word_distance


def _solve_partial_transport(
    source: torch.Tensor,
    destination: torch.Tensor,
    cost: torch.Tensor,
    n_sources: int,
    n_destinations: int,
    matched_pairs: int,
) -> torch.Tensor | None:
    if source.numel() == 0:
        return None
    source_np = source.numpy()
    destination_np = destination.numpy()
    support = csr_matrix(
        (np.ones(len(source_np)), (source_np, destination_np)),
        shape=(n_sources, n_destinations),
    )
    maximum_mass = int((maximum_bipartite_matching(support, perm_type="column") >= 0).sum())
    if maximum_mass < matched_pairs:
        return None

    edge_count = len(source_np)
    columns = np.arange(edge_count)
    capacity = coo_matrix(
        (
            np.ones(2 * edge_count),
            (
                np.concatenate([source_np, n_sources + destination_np]),
                np.concatenate([columns, columns]),
            ),
        ),
        shape=(n_sources + n_destinations, edge_count),
    ).tocsr()
    result = linprog(
        cost.detach().double().numpy(),
        A_ub=capacity,
        b_ub=np.ones(n_sources + n_destinations),
        A_eq=np.ones((1, edge_count)),
        b_eq=np.asarray([matched_pairs], dtype=np.float64),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not result.success:
        return None
    flow = torch.from_numpy(result.x).float()
    flow[flow < 1e-6] = 0
    return flow


def _local_dino_signatures(
    dino: torch.Tensor,
    state_indices: torch.Tensor,
    config: CqtAuditConfig,
) -> torch.Tensor:
    _, similarities = topk_neighbors(
        dino[state_indices],
        min(config.dino_neighbors, len(state_indices) - 1),
        config.similarity_chunk_size,
    )
    return similarities


def _plan_for_states(
    centered_clip: torch.Tensor,
    quotient_clip: torch.Tensor,
    word_embeddings: torch.Tensor,
    dino: torch.Tensor,
    state_a: torch.Tensor,
    state_b: torch.Tensor,
    config: CqtAuditConfig,
    include_dino: bool = True,
) -> dict | None:
    source, destination, word_similarity = _word_candidates(
        word_embeddings, state_a, state_b, config
    )
    raw_similarity = (
        centered_clip[state_a[source]] * centered_clip[state_b[destination]]
    ).sum(dim=1)
    quotient_similarity = (
        quotient_clip[state_a[source]] * quotient_clip[state_b[destination]]
    ).sum(dim=1)
    cost = _transport_cost(quotient_similarity, word_similarity, source, len(state_a))
    requested_mass = min(
        min(len(state_a), len(state_b)),
        max(config.min_transport_pairs, int(math.ceil(config.transport_mass * min(len(state_a), len(state_b))))),
    )
    flow = _solve_partial_transport(
        source, destination, cost, len(state_a), len(state_b), requested_mass
    )
    if flow is None:
        return None
    selected = flow > 0
    mass = float(flow.sum())
    gain = quotient_similarity - raw_similarity
    payload = {
        "state_a": state_a,
        "state_b": state_b,
        "source": source,
        "destination": destination,
        "word_similarity_edges": word_similarity,
        "raw_similarity": raw_similarity,
        "quotient_similarity": quotient_similarity,
        "cost": cost,
        "flow": flow,
        "selected": selected,
        "matched_pairs": requested_mass,
        "intervention_gain": float((flow * gain).sum() / mass),
        "word_similarity": float((flow * word_similarity).sum() / mass),
        "coverage": 2.0 * mass / (len(state_a) + len(state_b)),
    }
    if include_dino:
        signature_a = _local_dino_signatures(dino, state_a, config)
        signature_b = _local_dino_signatures(dino, state_b, config)
        signature_width = min(signature_a.shape[1], signature_b.shape[1])
        dino_damage = (
            signature_a[source, :signature_width] - signature_b[destination, :signature_width]
        ).abs().mean(dim=1)
        payload["dino_damage"] = float((flow * dino_damage).sum() / mass)
        payload["dino_damage_threshold"] = float(
            torch.quantile(dino_damage, config.dino_damage_quantile)
        )
    return payload


def _random_contrast_scores(
    centered_clip: torch.Tensor,
    word_embeddings: torch.Tensor,
    dino: torch.Tensor,
    state_a: torch.Tensor,
    state_b: torch.Tensor,
    config: CqtAuditConfig,
    generator: torch.Generator,
) -> list[float]:
    scores = []
    for _ in range(config.null_trials):
        random_contrast = F.normalize(
            torch.randn(centered_clip.shape[1], generator=generator), dim=0
        )
        random_quotient = concept_quotient(centered_clip, random_contrast)
        plan = _plan_for_states(
            centered_clip,
            random_quotient,
            word_embeddings,
            dino,
            state_a,
            state_b,
            config,
            include_dino=False,
        )
        scores.append(plan["intervention_gain"] if plan is not None else 0.0)
    return scores


def _shuffled_state_scores(
    centered_clip: torch.Tensor,
    quotient_clip: torch.Tensor,
    word_embeddings: torch.Tensor,
    dino: torch.Tensor,
    state_a: torch.Tensor,
    state_b: torch.Tensor,
    config: CqtAuditConfig,
    generator: torch.Generator,
) -> list[float]:
    union = torch.cat([state_a, state_b])
    scores = []
    for _ in range(config.null_trials):
        shuffled = union[torch.randperm(len(union), generator=generator)]
        plan = _plan_for_states(
            centered_clip,
            quotient_clip,
            word_embeddings,
            dino,
            shuffled[: len(state_a)],
            shuffled[len(state_a) :],
            config,
            include_dino=False,
        )
        scores.append(plan["intervention_gain"] if plan is not None else 0.0)
    return scores


def _preserved_concepts(
    plans: Sequence[dict],
    codes: torch.Tensor,
    vocabulary: Sequence[str],
    excluded: Sequence[int],
    top_k: int = 5,
) -> list[str]:
    scores = torch.zeros(codes.shape[1])
    total_mass = 0.0
    for plan in plans:
        selected = plan["selected"]
        flow = plan["flow"][selected]
        rows = plan["state_a"][plan["source"][selected]]
        columns = plan["state_b"][plan["destination"][selected]]
        scores += (((codes[rows] + codes[columns]) / 2.0) * flow.unsqueeze(1)).sum(dim=0)
        total_mass += float(flow.sum())
    scores /= max(total_mass, 1e-12)
    scores[list(excluded)] = 0
    top_indices = torch.topk(scores, min(top_k, len(scores))).indices.tolist()
    return [vocabulary[index] for index in top_indices if scores[index] > 0]


def _representative_pairs(
    plans: Sequence[dict],
    sample_ids: Sequence[str],
    limit: int = 6,
) -> list[dict]:
    representatives = []
    for fold, plan in enumerate(plans):
        positions = torch.where(plan["selected"])[0]
        positions = positions[torch.argsort(plan["cost"][positions], stable=True)]
        for position in positions[:limit].tolist():
            row = int(plan["state_a"][plan["source"][position]])
            column = int(plan["state_b"][plan["destination"][position]])
            representatives.append(
                {
                    "fold": fold,
                    "state_a_sample": sample_ids[row],
                    "state_b_sample": sample_ids[column],
                    "mass": float(plan["flow"][position]),
                    "raw_distance": float((1.0 - plan["raw_similarity"][position]) / 2.0),
                    "quotient_distance": float((1.0 - plan["quotient_similarity"][position]) / 2.0),
                    "word_similarity": float(plan["word_similarity_edges"][position]),
                    "transport_cost": float(plan["cost"][position]),
                }
            )
    representatives.sort(key=lambda item: (item["transport_cost"], item["state_a_sample"], item["state_b_sample"]))
    return representatives[:limit]


def run_cqt_audit(cache: dict, config: CqtAuditConfig) -> dict:
    """Discover CQT factors and return a CRP-compatible sparse teacher graph."""

    _validate_config(config)
    cache = validate_feature_cache(cache)
    all_groups = group_concepts(
        cache["splice_codes"], cache["dictionary"], cache["vocabulary"], _grouping_config(config)
    )
    all_activations = (
        _group_activations(cache["splice_codes"], all_groups)
        if all_groups
        else torch.empty(len(cache["sample_ids"]), 0)
    )
    groups = _candidate_groups(all_groups, all_activations, config) if all_groups else []
    activations = (
        _group_activations(cache["splice_codes"], groups)
        if groups
        else torch.empty(len(cache["sample_ids"]), 0)
    )
    folds = _crossfit_folds(len(cache["sample_ids"]), config.seed)
    semantic_codes = _semantic_code_embeddings(cache["splice_codes"], cache["dictionary"])
    proposals = _propose_factors(
        groups,
        activations,
        folds,
        semantic_codes,
        cache["splice_codes"],
        cache["dictionary"],
        config,
    ) if groups else []

    generator = torch.Generator().manual_seed(config.seed + 1)
    audited_factors, selected_evidence = [], []
    print(
        f"[INFO] Auditing {len(proposals)} cross-reproduced CQT factor proposals "
        f"from {len(groups)} candidate concept groups",
        flush=True,
    )
    for factor_id, proposal in enumerate(proposals):
        left, right = proposal["left"], proposal["right"]
        excluded = sorted(set(groups[left] + groups[right]))
        factor_identity = {
            "factor_id": factor_id,
            "state_a": {
                "concept_indices": groups[left],
                "concepts": [cache["vocabulary"][index] for index in groups[left]],
            },
            "state_b": {
                "concept_indices": groups[right],
                "concepts": [cache["vocabulary"][index] for index in groups[right]],
            },
            "proposal_score": proposal["proposal_score"],
            "proposal_folds": proposal["folds"],
        }
        state_a_direction = F.normalize(cache["dictionary"][groups[left]].mean(dim=0), dim=0)
        state_b_direction = F.normalize(cache["dictionary"][groups[right]].mean(dim=0), dim=0)
        contrast = state_a_direction - state_b_direction
        if float(contrast.norm()) <= 1e-12:
            audited_factors.append(
                {
                    **factor_identity,
                    "selected": False,
                    "rejection_reason": "degenerate_state_contrast",
                }
            )
            continue
        quotient_clip = concept_quotient(cache["centered_clip"], contrast)
        factor_word_embeddings = semantic_codes - cache["splice_codes"][:, excluded] @ cache["dictionary"][excluded]
        factor_word_embeddings = F.normalize(factor_word_embeddings, dim=1)

        plans, fold_metrics, random_by_fold, shuffled_by_fold = [], [], [], []
        failed_evaluation_fold = None
        for evaluation_fold in (0, 1):
            discovery_fold = 1 - evaluation_fold
            discovery_a, discovery_b = _state_indices(
                activations, left, right, folds == discovery_fold
            )
            evaluation_a, evaluation_b = _state_indices(
                activations, left, right, folds == evaluation_fold
            )
            raw_accuracy = _balanced_centroid_accuracy(
                cache["centered_clip"],
                discovery_a,
                discovery_b,
                cache["centered_clip"],
                evaluation_a,
                evaluation_b,
            )
            quotient_accuracy = _balanced_centroid_accuracy(
                quotient_clip,
                discovery_a,
                discovery_b,
                quotient_clip,
                evaluation_a,
                evaluation_b,
            )
            quotient_efficacy = max(
                0.0,
                min(1.0, (raw_accuracy - quotient_accuracy) / max(raw_accuracy - 0.5, 1e-6)),
            )
            plan = _plan_for_states(
                cache["centered_clip"],
                quotient_clip,
                factor_word_embeddings,
                cache["dino_embeddings"],
                evaluation_a,
                evaluation_b,
                config,
            )
            if plan is None:
                failed_evaluation_fold = evaluation_fold
                break
            plans.append(plan)
            fold_metrics.append(
                {
                    "discovery_fold": discovery_fold,
                    "evaluation_fold": evaluation_fold,
                    "raw_state_balanced_accuracy": raw_accuracy,
                    "quotient_state_balanced_accuracy": quotient_accuracy,
                    "quotient_efficacy": quotient_efficacy,
                    "matched_pairs": plan["matched_pairs"],
                    "intervention_gain": plan["intervention_gain"],
                    "word_similarity": plan["word_similarity"],
                    "dino_local_damage": plan["dino_damage"],
                    "dino_damage_threshold": plan["dino_damage_threshold"],
                    "coverage": plan["coverage"],
                }
            )
            random_by_fold.append(
                _random_contrast_scores(
                    cache["centered_clip"],
                    factor_word_embeddings,
                    cache["dino_embeddings"],
                    evaluation_a,
                    evaluation_b,
                    config,
                    generator,
                )
            )
            shuffled_by_fold.append(
                _shuffled_state_scores(
                    cache["centered_clip"],
                    quotient_clip,
                    factor_word_embeddings,
                    cache["dino_embeddings"],
                    evaluation_a,
                    evaluation_b,
                    config,
                    generator,
                )
            )
        if failed_evaluation_fold is not None:
            audited_factors.append(
                {
                    **factor_identity,
                    "selected": False,
                    "rejection_reason": "infeasible_fixed_mass_transport",
                    "failed_evaluation_fold": failed_evaluation_fold,
                }
            )
            continue

        random_scores = [float(np.mean(values)) for values in zip(*random_by_fold)]
        shuffled_scores = [float(np.mean(values)) for values in zip(*shuffled_by_fold)]
        null_scores = torch.tensor(random_scores + shuffled_scores)
        null_threshold = float(torch.quantile(null_scores, config.null_quantile))
        gain = float(np.mean([metric["intervention_gain"] for metric in fold_metrics]))
        word_similarity = float(np.mean([metric["word_similarity"] for metric in fold_metrics]))
        coverage = float(np.mean([metric["coverage"] for metric in fold_metrics]))
        dino_safe = all(
            metric["dino_local_damage"] <= metric["dino_damage_threshold"]
            for metric in fold_metrics
        )
        gates = {
            "state_predictable": min(metric["raw_state_balanced_accuracy"] for metric in fold_metrics)
            >= config.min_state_balanced_accuracy,
            "quotient_effective": min(metric["quotient_efficacy"] for metric in fold_metrics)
            >= config.min_quotient_efficacy,
            "positive_gain": min(metric["intervention_gain"] for metric in fold_metrics)
            > config.min_intervention_gain,
            "beats_null": gain > null_threshold,
            "word_preserved": word_similarity >= config.min_word_similarity,
            "dino_locally_safe": dino_safe,
            "coverage": coverage >= config.min_coverage,
        }
        selected = all(gates.values())
        null_excess_gain = max(0.0, gain - null_threshold)
        factor_payload = {
            **factor_identity,
            "selected": selected,
            "evaluation_folds": fold_metrics,
            "intervention_gain": gain,
            "null_threshold": null_threshold,
            "null_excess_gain": null_excess_gain,
            "random_contrast_scores": random_scores,
            "shuffled_state_scores": shuffled_scores,
            "word_similarity": word_similarity,
            "coverage": coverage,
            "matched_pairs": sum(plan["matched_pairs"] for plan in plans),
            "gates": gates,
            "preserved_concepts": _preserved_concepts(
                plans, cache["splice_codes"], cache["vocabulary"], excluded
            ),
            "representative_pairs": _representative_pairs(plans, cache["sample_ids"]),
        }
        audited_factors.append(factor_payload)
        if selected:
            rows, columns, confidences, gains = [], [], [], []
            for plan in plans:
                selected_edges = plan["selected"]
                state_a_rows = plan["state_a"][plan["source"][selected_edges]]
                state_b_rows = plan["state_b"][plan["destination"][selected_edges]]
                flow = plan["flow"][selected_edges]
                edge_gain = (
                    plan["quotient_similarity"][selected_edges]
                    - plan["raw_similarity"][selected_edges]
                )
                rows.extend([state_a_rows, state_b_rows])
                columns.extend([state_b_rows, state_a_rows])
                confidences.extend([flow * null_excess_gain, flow * null_excess_gain])
                gains.extend([edge_gain, edge_gain])
            selected_evidence.append(
                (
                    factor_id,
                    {
                        "rows": torch.cat(rows),
                        "columns": torch.cat(columns),
                        "confidence": torch.cat(confidences),
                        "gain": torch.cat(gains),
                    },
                )
            )

    graph = _build_teacher_graph(len(cache["sample_ids"]), selected_evidence, config)
    return {
        "artifact": CQT_ARTIFACT,
        "graph_version": GRAPH_VERSION,
        "cache_version": int(cache.get("cache_version", CACHE_VERSION)),
        "sample_ids": cache["sample_ids"],
        "config": asdict(config),
        "provenance": dict(cache.get("provenance", {})),
        "factors": audited_factors,
        "selected_factor_ids": [factor["factor_id"] for factor in audited_factors if factor["selected"]],
        **graph,
    }


def _artifact_summary(artifact: dict) -> dict:
    return {
        "artifact": artifact["artifact"],
        "graph_version": artifact["graph_version"],
        "config": artifact["config"],
        "provenance": artifact["provenance"],
        "sample_count": len(artifact["sample_ids"]),
        "candidate_factor_count": len(artifact["factors"]),
        "selected_factor_ids": artifact["selected_factor_ids"],
        "degree_stats": artifact["degree_stats"],
        "factors": artifact["factors"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the label-free SpLiCE-CQT frozen audit.")
    parser.add_argument("--cache", required=True, help="Existing CRP frozen feature cache (.pt).")
    parser.add_argument("--output", required=True, help="CQT teacher graph output (.pt).")
    parser.add_argument("--config", help="Optional JSON object overriding CqtAuditConfig fields.")
    parser.add_argument("--seed", type=int, help="Override the proposal/null seed.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_values = json.loads(args.config) if args.config else {}
    unknown = set(config_values).difference(CqtAuditConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown CQT audit settings: {sorted(unknown)}")
    if args.seed is not None:
        config_values["seed"] = args.seed
    config = CqtAuditConfig(**config_values)
    cache_path, output_path = Path(args.cache), Path(args.output)
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    artifact = run_cqt_audit(cache, config)
    _atomic_torch_save(artifact, output_path)
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(_artifact_summary(artifact), indent=2), encoding="utf-8")
    print(f"[INFO] Wrote CQT teacher graph to {output_path}")
    print(f"[INFO] Wrote CQT concept cards to {summary_path}")
    print(
        f"[INFO] Selected {len(artifact['selected_factor_ids'])}/{len(artifact['factors'])} factors"
    )


if __name__ == "__main__":
    main()
