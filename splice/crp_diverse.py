"""Optional diversity-constrained CRPv4 audit.

This module is deliberately separate from :mod:`splice.crp`.  It reuses the
canonical projection, relation scoring, null controls, cross-fold validation,
and graph builder, while changing only candidate budgeting and final group
selection.  Removing this file restores the original behavior completely.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from splice import crp as canonical
from splice.graph_io import save_graph_json
from splice.spatial_balance import load_spatial_balance_artifact, spatially_balanced_codes


@dataclass(frozen=True)
class DiverseSelectionConfig:
    """Label-free candidate budget and MMR-style selection settings."""

    semantic_cluster_count: int = 12
    candidates_per_cluster: int = 4
    candidate_budget: int = 48
    selected_group_count: int = 8
    max_selected_per_cluster: int = 1
    semantic_similarity_ceiling: float = 0.75
    semantic_redundancy_penalty: float = 0.75
    edge_overlap_penalty: float = 0.25
    preaudit_quality_weight: float = 0.25
    spatial_support_weight: float = 0.15
    activation_entropy_weight: float = 0.65
    spatial_agreement_weight: float = 0.25
    preaudit_spatial_support_weight: float = 0.10
    kmeans_iterations: int = 12


def _validate_diversity_config(config: DiverseSelectionConfig) -> None:
    positive_integers = {
        "semantic_cluster_count": config.semantic_cluster_count,
        "candidates_per_cluster": config.candidates_per_cluster,
        "candidate_budget": config.candidate_budget,
        "selected_group_count": config.selected_group_count,
        "max_selected_per_cluster": config.max_selected_per_cluster,
        "kmeans_iterations": config.kmeans_iterations,
    }
    for name, value in positive_integers.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if not -1 <= config.semantic_similarity_ceiling <= 1:
        raise ValueError("semantic_similarity_ceiling must be in [-1, 1].")
    nonnegative = {
        "semantic_redundancy_penalty": config.semantic_redundancy_penalty,
        "edge_overlap_penalty": config.edge_overlap_penalty,
        "preaudit_quality_weight": config.preaudit_quality_weight,
        "spatial_support_weight": config.spatial_support_weight,
        "activation_entropy_weight": config.activation_entropy_weight,
        "spatial_agreement_weight": config.spatial_agreement_weight,
        "preaudit_spatial_support_weight": config.preaudit_spatial_support_weight,
    }
    for name, value in nonnegative.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")


def _group_centroids(dictionary: torch.Tensor, groups: Sequence[Sequence[int]]) -> torch.Tensor:
    normalized = F.normalize(dictionary.float().cpu(), dim=1)
    centroids = [F.normalize(normalized[list(group)].mean(dim=0), dim=0) for group in groups]
    return torch.stack(centroids) if centroids else torch.empty(0, dictionary.shape[1])


def _binary_entropy(probability: float) -> float:
    probability = min(max(probability, 0.0), 1.0)
    if probability in {0.0, 1.0}:
        return 0.0
    return -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log(1.0 - probability)
    ) / math.log(2.0)


def _preaudit_statistics(
    codes: torch.Tensor,
    groups: Sequence[Sequence[int]],
    spatial_artifact: dict,
    config: DiverseSelectionConfig,
) -> list[dict]:
    spatial_indices = torch.as_tensor(spatial_artifact["concept_indices"], dtype=torch.long).cpu()
    statistics = []
    for source_group_id, group in enumerate(groups):
        group_indices = torch.as_tensor(group, dtype=torch.long)
        active = codes[:, list(group)].sum(dim=1) > 0
        frequency = float(active.float().mean())
        spatial_hit = torch.isin(spatial_indices, group_indices).any(dim=1)
        spatial_support = float(spatial_hit.float().mean())
        union = int((active | spatial_hit).sum())
        spatial_agreement = float((active & spatial_hit).sum()) / max(1, union)
        activation_entropy = _binary_entropy(frequency)
        preaudit_score = (
            config.activation_entropy_weight * activation_entropy
            + config.spatial_agreement_weight * spatial_agreement
            + config.preaudit_spatial_support_weight * math.sqrt(spatial_support)
        )
        statistics.append(
            {
                "source_group_id": source_group_id,
                "activation_frequency": frequency,
                "activation_entropy": activation_entropy,
                "spatial_support": spatial_support,
                "spatial_agreement": spatial_agreement,
                "preaudit_score": preaudit_score,
            }
        )
    return statistics


def _semantic_clusters(
    centroids: torch.Tensor,
    quality: torch.Tensor,
    cluster_count: int,
    iterations: int,
) -> torch.Tensor:
    """Deterministic spherical clustering with quality-weighted farthest seeds."""

    if len(centroids) == 0:
        return torch.empty(0, dtype=torch.long)
    cluster_count = min(cluster_count, len(centroids))
    seeds = [int(quality.argmax())]
    for _ in range(1, cluster_count):
        similarity = centroids @ centroids[seeds].T
        distance = (1.0 - similarity.max(dim=1).values).clamp_min(0.0)
        seed_score = (0.25 + quality) * distance
        seed_score[seeds] = -1.0
        seeds.append(int(seed_score.argmax()))

    cluster_centroids = centroids[seeds].clone()
    assignments = torch.full((len(centroids),), -1, dtype=torch.long)
    for _ in range(iterations):
        updated_assignments = (centroids @ cluster_centroids.T).argmax(dim=1)
        if torch.equal(updated_assignments, assignments):
            break
        assignments = updated_assignments
        updated_centroids = []
        for cluster_id in range(cluster_count):
            members = centroids[assignments == cluster_id]
            updated_centroids.append(
                F.normalize(members.mean(dim=0), dim=0)
                if len(members)
                else cluster_centroids[cluster_id]
            )
        cluster_centroids = torch.stack(updated_centroids)
    return assignments


def preselect_diverse_candidates(
    codes: torch.Tensor,
    dictionary: torch.Tensor,
    groups: Sequence[Sequence[int]],
    spatial_artifact: dict,
    config: DiverseSelectionConfig,
) -> tuple[list[int], list[dict], torch.Tensor]:
    """Budget expensive audits across unlabeled semantic regions."""

    _validate_diversity_config(config)
    if not groups:
        return [], [], torch.empty(0, dtype=torch.long)
    centroids = _group_centroids(dictionary, groups)
    statistics = _preaudit_statistics(codes, groups, spatial_artifact, config)
    quality = torch.tensor([item["preaudit_score"] for item in statistics])
    assignments = _semantic_clusters(
        centroids,
        quality,
        config.semantic_cluster_count,
        config.kmeans_iterations,
    )
    for index, cluster_id in enumerate(assignments.tolist()):
        statistics[index]["semantic_cluster"] = int(cluster_id)

    chosen = []
    for cluster_id in sorted(set(assignments.tolist())):
        members = torch.where(assignments == cluster_id)[0].tolist()
        members.sort(key=lambda index: (-float(quality[index]), int(index)))
        chosen.extend(members[: config.candidates_per_cluster])
    chosen = sorted(
        set(chosen),
        key=lambda index: (-float(quality[index]), int(assignments[index]), int(index)),
    )[: config.candidate_budget]
    # Audit order is stable and independent of floating-point sorting ties.
    chosen.sort(key=lambda index: (int(assignments[index]), int(index)))
    return chosen, statistics, assignments


def _unit_interval(values: Sequence[float]) -> dict[int, float]:
    if not values:
        return {}
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return {index: 1.0 for index in range(len(values))}
    return {index: (value - low) / (high - low) for index, value in enumerate(values)}


def _edge_pairs(evidence: dict) -> set[tuple[int, int]]:
    return set(zip(evidence["rows"].tolist(), evidence["columns"].tolist()))


def _jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 0.0
    return len(left.intersection(right)) / max(1, len(left.union(right)))


def select_diverse_evidence(
    passing: list[tuple[int, dict]],
    audited_groups: list[dict],
    centroids: torch.Tensor,
    config: DiverseSelectionConfig,
) -> tuple[list[tuple[int, dict]], list[dict]]:
    """Apply a quality-aware, cluster-capped MMR selector after all gates."""

    _validate_diversity_config(config)
    if not passing:
        return [], []
    null_quality = _unit_interval(
        [math.log1p(max(0.0, float(audited_groups[group_id]["null_excess_score"]))) for group_id, _ in passing]
    )
    preaudit_quality = _unit_interval(
        [float(audited_groups[group_id]["preaudit_score"]) for group_id, _ in passing]
    )
    spatial_support = _unit_interval(
        [float(audited_groups[group_id]["spatial_support"]) for group_id, _ in passing]
    )
    pair_sets = [_edge_pairs(evidence) for _, evidence in passing]
    selected_positions: list[int] = []
    traces: list[dict] = []
    cluster_counts: dict[int, int] = {}

    while len(selected_positions) < config.selected_group_count:
        best_position = None
        best_trace = None
        for position, (group_id, _) in enumerate(passing):
            if position in selected_positions:
                continue
            cluster_id = int(audited_groups[group_id]["semantic_cluster"])
            if cluster_counts.get(cluster_id, 0) >= config.max_selected_per_cluster:
                continue
            if selected_positions:
                maximum_semantic_similarity = max(
                    float(centroids[group_id] @ centroids[passing[other][0]])
                    for other in selected_positions
                )
                maximum_edge_overlap = max(
                    _jaccard(pair_sets[position], pair_sets[other]) for other in selected_positions
                )
            else:
                maximum_semantic_similarity = 0.0
                maximum_edge_overlap = 0.0
            if maximum_semantic_similarity > config.semantic_similarity_ceiling:
                continue
            diverse_score = (
                null_quality[position]
                + config.preaudit_quality_weight * preaudit_quality[position]
                + config.spatial_support_weight * spatial_support[position]
                - config.semantic_redundancy_penalty * max(0.0, maximum_semantic_similarity)
                - config.edge_overlap_penalty * maximum_edge_overlap
            )
            trace = {
                "group_id": int(group_id),
                "selection_step": len(selected_positions) + 1,
                "diverse_score": diverse_score,
                "normalized_null_quality": null_quality[position],
                "normalized_preaudit_quality": preaudit_quality[position],
                "normalized_spatial_support": spatial_support[position],
                "maximum_semantic_similarity": maximum_semantic_similarity,
                "maximum_edge_overlap": maximum_edge_overlap,
                "semantic_cluster": cluster_id,
            }
            key = (diverse_score, null_quality[position], -group_id)
            if best_trace is None or key > (
                best_trace["diverse_score"],
                best_trace["normalized_null_quality"],
                -best_trace["group_id"],
            ):
                best_position, best_trace = position, trace
        if best_position is None:
            break
        selected_positions.append(best_position)
        cluster_id = int(best_trace["semantic_cluster"])
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        traces.append(best_trace)

    return [passing[position] for position in selected_positions], traces


def _validate_feature_cache(cache: dict, *, require_dino: bool) -> dict:
    """Validate raw or already validated caches with an optional DINO branch."""

    # ``validate_feature_cache`` materializes the optional DINO field as ``None``
    # for no-DINO caches. Such a cache can legitimately reach this isolated API
    # after an earlier validation pass, so omit only the absent optional value.
    cache_for_validation = cache
    if not require_dino and cache.get("dino_embeddings") is None:
        cache_for_validation = dict(cache)
        cache_for_validation.pop("dino_embeddings", None)
    return canonical.validate_feature_cache(
        cache_for_validation, require_dino=require_dino
    )


def run_diverse_frozen_audit(
    cache: dict,
    audit_config: canonical.CrpAuditConfig,
    diversity_config: DiverseSelectionConfig,
    spatial_balance_artifact: dict,
) -> dict:
    """Run an isolated diversity-constrained CRPv4 frozen audit."""

    canonical._validate_config(audit_config)
    _validate_diversity_config(diversity_config)
    if not audit_config.spatial_balance:
        raise ValueError("The isolated diverse audit requires spatial_balance=true.")
    if audit_config.cobalt:
        raise ValueError("The isolated diverse audit does not combine legacy CoBalT weighting.")
    if audit_config.max_selected_groups:
        raise ValueError("Use diversity selected_group_count; max_selected_groups must be zero.")
    cache = _validate_feature_cache(cache, require_dino=audit_config.use_dino)
    artifact_variant = str(spatial_balance_artifact.get("variant", ""))
    if artifact_variant != audit_config.spatial_balance_variant:
        raise ValueError("Spatial artifact variant does not match the audit configuration.")
    audit_codes, spatial_summary = spatially_balanced_codes(
        cache["splice_codes"],
        spatial_balance_artifact,
        floor=audit_config.spatial_balance_floor,
        frequency_power=audit_config.spatial_frequency_power,
    )
    source_groups = canonical.group_concepts(
        audit_codes,
        cache["dictionary"],
        cache["vocabulary"],
        audit_config,
    )
    chosen_source_ids, preaudit_stats, source_assignments = preselect_diverse_candidates(
        audit_codes,
        cache["dictionary"],
        source_groups,
        spatial_balance_artifact,
        diversity_config,
    )
    groups = [source_groups[index] for index in chosen_source_ids]
    candidate_centroids = _group_centroids(cache["dictionary"], groups)
    print(
        f"[INFO] Diverse preselection retained {len(groups)}/{len(source_groups)} groups "
        f"across {len(set(source_assignments[chosen_source_ids].tolist())) if chosen_source_ids else 0} clusters",
        flush=True,
    )

    raw_neighbours, _ = canonical.topk_neighbors(
        cache["centered_clip"], audit_config.projected_neighbors, audit_config.similarity_chunk_size
    )
    dino_neighbours = None
    if audit_config.use_dino:
        dino_neighbours, _ = canonical.topk_neighbors(
            cache["dino_embeddings"], audit_config.dino_neighbors, audit_config.similarity_chunk_size
        )
    n_samples = len(cache["sample_ids"])
    generator = torch.Generator().manual_seed(audit_config.seed)
    audit_geometry = canonical._AuditGeometry(
        centered_clip=cache["centered_clip"],
        raw_neighbours=raw_neighbours,
        splice_codes=audit_codes,
        dino_embeddings=cache["dino_embeddings"],
        dino_neighbours=dino_neighbours,
    )

    audited_groups: list[dict] = []
    passing: list[tuple[int, dict]] = []
    report_every = max(1, len(groups) // 20)
    for group_id, (source_group_id, concept_indices) in enumerate(zip(chosen_source_ids, groups)):
        basis = canonical.orthonormal_basis(
            cache["dictionary"][concept_indices], audit_config.orthogonal_tolerance
        )
        activation = audit_codes[:, concept_indices].sum(dim=1)
        group_geometry = canonical._relation_geometry(
            audit_geometry, basis, audit_config, concept_indices
        )
        evidence = canonical._score_relations(group_geometry, activation, audit_config)
        random_geometries = []
        for _ in range(audit_config.null_trials):
            random_directions = torch.randn(
                basis.shape[1], cache["centered_clip"].shape[1], generator=generator
            )
            random_basis = canonical.orthonormal_basis(
                random_directions, audit_config.orthogonal_tolerance
            )
            random_geometries.append(
                canonical._relation_geometry(
                    audit_geometry, random_basis, audit_config, concept_indices
                )
            )
        random_scores, shuffled_scores = canonical._null_scores(
            group_geometry, random_geometries, activation, audit_config, generator
        )
        null_scores = torch.tensor(random_scores + shuffled_scores)
        threshold = (
            float(torch.quantile(null_scores, audit_config.null_quantile))
            if null_scores.numel()
            else math.inf
        )
        cross_fold = canonical._cross_fold_validation(
            audit_geometry,
            basis,
            activation,
            evidence,
            concept_indices,
            audit_config,
        )
        quality_gate_passed = (
            evidence["coverage"] >= audit_config.min_coverage
            and evidence["score"] > threshold
            and cross_fold["passed"]
        )
        null_excess_score = max(0.0, evidence["score"] - threshold)
        null_excess_ratio = min(
            1.0, null_excess_score / max(abs(evidence["score"]), 1e-12)
        )
        preaudit = preaudit_stats[source_group_id]
        audited_groups.append(
            {
                "group_id": group_id,
                "source_group_id": source_group_id,
                "concept_indices": concept_indices,
                "concepts": [cache["vocabulary"][index] for index in concept_indices],
                "basis_rank": basis.shape[1],
                "semantic_cluster": int(preaudit["semantic_cluster"]),
                "preaudit_score": float(preaudit["preaudit_score"]),
                "activation_frequency": float(preaudit["activation_frequency"]),
                "activation_entropy": float(preaudit["activation_entropy"]),
                "spatial_support": float(preaudit["spatial_support"]),
                "spatial_agreement": float(preaudit["spatial_agreement"]),
                "quality_gate_passed": quality_gate_passed,
                "selected": False,
                "score": evidence["score"],
                "null_threshold": threshold,
                "null_excess_score": null_excess_score,
                "null_excess_ratio": null_excess_ratio,
                "coverage": evidence["coverage"],
                "robust_positive_gain": evidence["positive_gain"],
                "semantic_agreement": evidence["semantic_agreement"],
                "dino_guard_enabled": audit_config.use_dino,
                "residual_splice_gate_enabled": (
                    audit_config.use_residual_splice_gate and not audit_config.use_dino
                ),
                "residual_splice_similarity_threshold": (
                    audit_config.residual_splice_similarity_threshold
                    if audit_config.use_residual_splice_gate and not audit_config.use_dino
                    else None
                ),
                "cross_fold": cross_fold,
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
        )
        if quality_gate_passed:
            passing.append(
                (
                    group_id,
                    {
                        **evidence,
                        "confidence": evidence["confidence"]
                        * null_excess_ratio
                        * float(cross_fold["edge_persistence"] or 1.0),
                    },
                )
            )
        if (group_id + 1) % report_every == 0 or group_id + 1 == len(groups):
            print(
                f"[INFO] Diverse audit {group_id + 1}/{len(groups)}; "
                f"quality_gate={len(passing)}",
                flush=True,
            )

    selected_evidence, selection_trace = select_diverse_evidence(
        passing, audited_groups, candidate_centroids, diversity_config
    )
    selected_ids = {group_id for group_id, _ in selected_evidence}
    trace_by_group = {int(item["group_id"]): item for item in selection_trace}
    for group in audited_groups:
        group_id = int(group["group_id"])
        if group_id in selected_ids:
            group["selected"] = True
            group["diversity_selection"] = trace_by_group[group_id]
        elif group["quality_gate_passed"]:
            group["rejection_reason"] = "diversity_selector"
        else:
            group["rejection_reason"] = "quality_gate"

    graph = canonical._build_teacher_graph(n_samples, selected_evidence, audit_config)
    selected_centroids = candidate_centroids[sorted(selected_ids)] if selected_ids else None
    if selected_centroids is not None and len(selected_centroids) > 1:
        similarities = selected_centroids @ selected_centroids.T
        upper = similarities[torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)]
        mean_pairwise_similarity = float(upper.mean())
        maximum_pairwise_similarity = float(upper.max())
    else:
        mean_pairwise_similarity = 0.0
        maximum_pairwise_similarity = 0.0
    return {
        "artifact": "splice_crp_v4_teacher_graph",
        "graph_version": canonical.CRP_V4_GRAPH_VERSION,
        "cache_version": int(cache.get("cache_version", canonical.CACHE_VERSION)),
        "sample_count": n_samples,
        "sample_ids": cache["sample_ids"],
        "config": asdict(audit_config),
        "provenance": dict(cache.get("provenance", {})),
        "cobalt_check": None,
        "spatial_balance": spatial_summary,
        "diverse_selection": {
            "implementation": "isolated_crpv4_diverse_v1",
            "config": asdict(diversity_config),
            "source_group_count": len(source_groups),
            "preselected_group_count": len(groups),
            "quality_gate_count": len(passing),
            "selected_group_count": len(selected_evidence),
            "selected_semantic_clusters": [
                int(audited_groups[group_id]["semantic_cluster"])
                for group_id, _ in selected_evidence
            ],
            "mean_selected_pairwise_similarity": mean_pairwise_similarity,
            "maximum_selected_pairwise_similarity": maximum_pairwise_similarity,
            "selection_trace": selection_trace,
        },
        "groups": audited_groups,
        "selected_group_ids": [int(group_id) for group_id, _ in selected_evidence],
        "cross_fold_summary": {
            "enabled": audit_config.use_cross_fold_validation,
            "fold_count": audit_config.cross_fold_count
            if audit_config.use_cross_fold_validation
            else 0,
            "min_edge_persistence": audit_config.cross_fold_min_edge_persistence,
            "passed_groups": sum(
                bool(group.get("cross_fold", {}).get("passed")) for group in audited_groups
            ),
        },
        **graph,
    }


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the isolated diverse CRPv4 audit.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--spatial-balance-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True, help="CRP audit configuration JSON.")
    parser.add_argument("--diversity-config", default="{}", help="Diversity configuration JSON.")
    parser.add_argument("--use-dino", type=_parse_bool, nargs="?", const=True)
    args = parser.parse_args(argv)

    audit_values = json.loads(args.config)
    unknown_audit = set(audit_values).difference(canonical.CrpAuditConfig.__dataclass_fields__)
    if unknown_audit:
        raise ValueError(f"Unknown CRP audit settings: {sorted(unknown_audit)}")
    if args.use_dino is not None:
        audit_values["use_dino"] = args.use_dino
    audit_config = canonical.CrpAuditConfig(**audit_values)
    diversity_values = json.loads(args.diversity_config)
    unknown_diversity = set(diversity_values).difference(
        DiverseSelectionConfig.__dataclass_fields__
    )
    if unknown_diversity:
        raise ValueError(f"Unknown diversity settings: {sorted(unknown_diversity)}")
    diversity_config = DiverseSelectionConfig(**diversity_values)

    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    cache = _validate_feature_cache(cache, require_dino=audit_config.use_dino)
    spatial_artifact = load_spatial_balance_artifact(
        args.spatial_balance_artifact,
        str(cache.get("provenance", {}).get("dataset", "")),
        cache["sample_ids"],
        cache["vocabulary"],
        dict(cache.get("provenance", {})),
    )
    graph = run_diverse_frozen_audit(
        cache, audit_config, diversity_config, spatial_artifact
    )
    save_graph_json(graph, Path(args.output))
    print(f"[INFO] Wrote isolated diverse CRPv4 graph to {args.output}")
    print(
        f"[INFO] Selected {len(graph['selected_group_ids'])}/"
        f"{len(graph['groups'])} audited groups",
        flush=True,
    )


if __name__ == "__main__":
    main()
