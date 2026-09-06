"""Pure, label-free metrics and deterministic selection helpers for CRP search."""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from splice.crp import orthonormal_basis


def _quantiles(values: torch.Tensor) -> dict[str, float | None]:
    values = torch.as_tensor(values, dtype=torch.float32).flatten()
    if not values.numel():
        return {"p10": None, "median": None}
    return {"p10": float(torch.quantile(values, 0.10)), "median": float(torch.median(values))}


def _pairwise_cosines(vectors: torch.Tensor, groups: Sequence[Sequence[int]]) -> torch.Tensor:
    values: list[torch.Tensor] = []
    normalized = F.normalize(torch.as_tensor(vectors, dtype=torch.float32), dim=0)
    for group in groups:
        if len(group) < 2:
            continue
        members = normalized[:, list(group)].T
        pairwise = members @ members.T
        values.append(pairwise[torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)])
    return torch.cat(values) if values else torch.empty(0)


def grouping_coherence_metrics(
    codes: torch.Tensor,
    dictionary: torch.Tensor,
    groups: Sequence[Sequence[int]],
) -> dict[str, float | int | None]:
    """Measure final transitive-group coherence without annotations."""

    text = _pairwise_cosines(dictionary.T, groups)
    code = _pairwise_cosines(codes, groups)
    text_q = _quantiles(text)
    code_q = _quantiles(code)
    return {
        "within_group_text_cosine_p10": text_q["p10"],
        "within_group_text_cosine_median": text_q["median"],
        "within_group_coactivation_cosine_p10": code_q["p10"],
        "within_group_coactivation_cosine_median": code_q["median"],
        "non_singleton_group_count": sum(len(group) > 1 for group in groups),
    }


def projection_energy_metrics(
    centered_embeddings: torch.Tensor,
    dictionary: torch.Tensor,
    groups: Sequence[Sequence[int]],
    selected_group_ids: Sequence[int] | None = None,
    tolerance: float = 1e-6,
) -> dict[str, float | None]:
    """Measure removed CLIP energy for selected concept-group subspaces."""

    centered = torch.as_tensor(centered_embeddings, dtype=torch.float32)
    selected = list(range(len(groups))) if selected_group_ids is None else list(selected_group_ids)
    energies: list[torch.Tensor] = []
    denominator = centered.square().sum(dim=1).clamp_min(1e-12)
    for group_id in selected:
        group = groups[group_id]
        if not group:
            continue
        basis = orthonormal_basis(torch.as_tensor(dictionary)[list(group)], tolerance)
        projection = centered @ basis @ basis.T
        energies.append(projection.square().sum(dim=1) / denominator)
    values = torch.cat(energies) if energies else torch.empty(0)
    quantiles = _quantiles(values)
    return {
        "projection_removed_energy_p10": quantiles["p10"],
        "projection_removed_energy_median": quantiles["median"],
        "projection_removed_energy_max": float(values.max()) if values.numel() else None,
    }


def grouping_metrics(
    codes: torch.Tensor,
    dictionary: torch.Tensor,
    groups: Sequence[Sequence[int]],
    source_fidelity: torch.Tensor | None = None,
    fidelity_threshold: float = 0.90,
    selected_group_ids: Sequence[int] | None = None,
    centered_embeddings: torch.Tensor | None = None,
) -> dict:
    """Return the complete label-free Stage-B measurement payload."""

    active_concepts = sum(len(group) for group in groups)
    sizes = [len(group) for group in groups]
    metrics = {
        "active_concept_count": active_concepts,
        "group_count": len(groups),
        "group_size_distribution": {
            "min": min(sizes) if sizes else 0,
            "median": float(torch.median(torch.tensor(sizes, dtype=torch.float32))) if sizes else 0.0,
            "max": max(sizes) if sizes else 0,
        },
        "compression_gain": 1.0 - len(groups) / max(1, active_concepts),
        "largest_group_size": max(sizes) if sizes else 0,
        "largest_group_fraction": max(sizes) / max(1, active_concepts) if sizes else 0.0,
    }
    metrics.update(grouping_coherence_metrics(codes, dictionary, groups))
    if source_fidelity is not None:
        fidelity = torch.as_tensor(source_fidelity, dtype=torch.float32)
        metrics.update({
            "reconstruction_source_coverage": float((fidelity >= fidelity_threshold).float().mean()) if fidelity.numel() else 0.0,
            "reconstruction_fidelity_p10": _quantiles(fidelity)["p10"],
            "reconstruction_fidelity_median": _quantiles(fidelity)["median"],
        })
    else:
        metrics.update({
            "reconstruction_source_coverage": None,
            "reconstruction_fidelity_p10": None,
            "reconstruction_fidelity_median": None,
        })
    if centered_embeddings is not None:
        metrics.update(projection_energy_metrics(
            centered_embeddings, dictionary, groups, selected_group_ids
        ))
    else:
        metrics.update({
            "projection_removed_energy_p10": None,
            "projection_removed_energy_median": None,
            "projection_removed_energy_max": None,
        })
    return metrics


def evaluate_grouping_gates(metrics: dict, mini: dict | None, gates: dict | None = None) -> dict:
    """Apply fixed Stage-B gates while distinguishing unmeasured dependent gates."""

    gates = gates or {}
    checks: dict[str, bool | None] = {
        "reconstruction_source_coverage": float(metrics.get("reconstruction_source_coverage") or 0.0) >= gates.get("min_reconstruction_coverage", 0.99),
        "compression_gain": gates.get("min_compression_gain", 0.05) <= float(metrics.get("compression_gain", 0.0)) <= gates.get("max_compression_gain", 0.50),
        "largest_group": float(metrics.get("largest_group_size", 0)) <= gates.get("max_largest_group_size", 16)
        and float(metrics.get("largest_group_fraction", 1.0)) <= gates.get("max_largest_group_fraction", 0.05),
        "text_coherence": float(metrics.get("within_group_text_cosine_p10") or -1.0) >= gates.get("min_text_cosine_p10", 0.55),
        "coactivation_coherence": float(metrics.get("within_group_coactivation_cosine_p10") or -1.0) >= gates.get("min_coactivation_cosine_p10", 0.05),
        "mini_null_passing_group": (
            any(item.get("passed_null", False) for item in mini.get("groups", []))
            if mini is not None
            else None
        ),
        "geometry_change": (
            float(mini.get("median_top1_neighbor_turnover", 0.0)) >= gates.get("min_top1_turnover", 0.10)
            or float(mini.get("median_jaccard_at_k", 1.0)) <= gates.get("max_jaccard_for_change", 0.90)
            if mini is not None
            else None
        ),
        "destructive_change_guard": (
            float(mini.get("median_jaccard_at_k", 0.0)) >= gates.get("min_jaccard_guard", 0.50)
            if mini is not None
            else None
        ),
    }
    not_evaluated = [name for name, passed in checks.items() if passed is None]
    failed = [name for name, passed in checks.items() if passed is False]
    return {
        "passed": not not_evaluated and not failed,
        "checks": checks,
        "failed_gates": failed,
        "not_evaluated_gates": not_evaluated,
        "gate_status": {
            name: "NOT_EVALUATED" if passed is None else "PASS" if passed else "FAIL"
            for name, passed in checks.items()
        },
    }


def set_jaccard(left: Iterable[object], right: Iterable[object]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def minimum_pairwise_jaccard(collections: Sequence[Iterable[object]]) -> float:
    values = [set(item) for item in collections]
    if len(values) < 2:
        return 1.0
    return min(set_jaccard(left, right) for left, right in itertools.combinations(values, 2))


def selected_group_sets(graph: dict) -> list[frozenset[int]]:
    return [
        frozenset(int(value) for value in group.get("concept_indices", []))
        for group in graph.get("groups", [])
        if group.get("selected", False)
    ]


def group_partition_jaccard(left: Sequence[Iterable[int]], right: Sequence[Iterable[int]]) -> float:
    """Compare partitions by the minimum best-match Jaccard in both directions."""

    left_sets, right_sets = [set(item) for item in left], [set(item) for item in right]
    if not left_sets and not right_sets:
        return 1.0
    if not left_sets or not right_sets:
        return 0.0
    forward = [max(set_jaccard(group, other) for other in right_sets) for group in left_sets]
    backward = [max(set_jaccard(group, other) for other in left_sets) for group in right_sets]
    return min(forward + backward)


def replacement_edges(graph: dict) -> set[tuple[int, int]]:
    return {
        (int(item["row"]), int(item["crp_donor"]))
        for item in graph.get("safe_replacements", [])
    }


def lexicographic_candidate_key(report: dict) -> tuple[float, float, float, str]:
    """Sort admissible reports by the protocol's deterministic rule."""

    return (
        -float(report.get("median_training_mass_weighted_evidence_density", 0.0)),
        -float(report.get("minimum_replacement_edge_jaccard", 0.0)),
        float(report.get("treatment_mass_hhi", 1.0)),
        str(report.get("candidate_id", "")),
    )
