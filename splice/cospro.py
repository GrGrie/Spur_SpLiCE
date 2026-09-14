"""Label-free concept grouping and frozen audit for CoSpRo.

Concept grouping consumes a frozen SpLiCE dataset cache and produces a reusable JSON
artifact. The audit resumes from that artifact and the same cache. Neither
stage loads target, spurious, or group annotations; those belong in a separate
post-hoc diagnostic step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from splice.graph_io import save_graph_json


SPLICE_DATASET_CACHE_VERSION = 1
CONCEPT_GROUPS_VERSION = 1
# GRAPH_VERSION remains the legacy v2 format. CoSpRo v3 has its own version
# because its fixed-density and validation fields are method changes.
GRAPH_VERSION = 2
CRP_GRAPH_VERSION = 3
COSPRO_CONCEPT_GROUP_ARTIFACT = "cospro_concept_groups_v1"
COSPRO_TEACHER_GRAPH_ARTIFACT = "cospro_teacher_graph_v3"
LEGACY_CONCEPT_GROUP_ARTIFACTS = {"splice_crp_concept_groups"}
GROUPING_CONFIG_FIELDS = (
    "min_concept_frequency",
    "max_concept_frequency",
    "text_similarity_threshold",
    "coactivation_threshold",
    "min_group_size",
    "similarity_chunk_size",
)
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
    activation_difference_quantile: float = 0.75
    min_intervention_gain: float = 1e-4
    min_coverage: float = 0.01
    graph_top_k: int = 3
    max_indegree: int = 10
    # Retained as a legacy fallback for callers constructing a config without
    # max_indegree; CRPv3 uses the absolute cap above.
    indegree_factor: float = 3.0
    null_trials: int = 16
    null_quantile: float = 0.95
    seed: int = 0
    similarity_chunk_size: int = 512
    orthogonal_tolerance: float = 1e-6
    use_residual_splice_gate: bool = True
    residual_splice_similarity_threshold: float = 0.25
    # Exact all-pairs search is useful for small audits and regression tests.
    # Large datasets use deterministic random-hyperplane LSH followed by exact
    # cosine reranking inside the candidate buckets.
    neighbor_backend: str = "auto"
    ann_threshold: int = 20_000
    ann_tables: int = 8
    ann_bucket_size: int = 512


def _validate_config(config: CrpAuditConfig) -> None:
    boolean_fields = {
        "use_residual_splice_gate": config.use_residual_splice_gate,
    }
    if any(not isinstance(value, bool) for value in boolean_fields.values()):
        raise ValueError("CoSpRo boolean settings must be booleans.")
    probabilities = {
        "min_concept_frequency": config.min_concept_frequency,
        "max_concept_frequency": config.max_concept_frequency,
        "activation_difference_quantile": config.activation_difference_quantile,
        "min_coverage": config.min_coverage,
        "null_quantile": config.null_quantile,
        "residual_splice_similarity_threshold": config.residual_splice_similarity_threshold,
    }
    for name, value in probabilities.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
    if config.min_concept_frequency > config.max_concept_frequency:
        raise ValueError("min_concept_frequency cannot exceed max_concept_frequency.")
    integer_fields = {
        "min_group_size": config.min_group_size,
        "projected_neighbors": config.projected_neighbors,
        "graph_top_k": config.graph_top_k,
        "max_indegree": config.max_indegree,
        "null_trials": config.null_trials,
        "similarity_chunk_size": config.similarity_chunk_size,
        "ann_threshold": config.ann_threshold,
        "ann_tables": config.ann_tables,
        "ann_bucket_size": config.ann_bucket_size,
    }
    for name, value in integer_fields.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if config.max_selected_groups < 0:
        raise ValueError("max_selected_groups must be non-negative; 0 disables the cap.")
    if config.neighbor_backend not in {"auto", "exact", "lsh"}:
        raise ValueError("neighbor_backend must be one of: auto, exact, lsh.")


def validate_crp_config(config: CrpAuditConfig) -> CrpAuditConfig:
    """Validate a complete grouping/audit configuration and return it unchanged."""

    _validate_config(config)
    return config


# Canonical public spelling; the old name remains for callers and serialized
# option dictionaries created before the method was named CoSpRo consistently.
validate_cospro_config = validate_crp_config


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


def validate_splice_dataset_cache(cache: dict) -> dict:
    """Validate and normalize the frozen SpLiCE dataset cache."""

    if not isinstance(cache, dict):
        raise ValueError("SpLiCE dataset cache must be a dictionary.")
    def collect_keys(value) -> set[str]:
        if not isinstance(value, dict):
            return set()
        keys = {str(key).lower() for key in value}
        for nested in value.values():
            keys.update(collect_keys(nested))
        return keys

    forbidden = FORBIDDEN_CACHE_KEYS.intersection(collect_keys(cache))
    if forbidden:
        raise ValueError(
            f"SpLiCE dataset cache contains forbidden annotation keys: {sorted(forbidden)}"
        )
    unexpected = set(cache).difference(REQUIRED_CACHE_KEYS, {"provenance", "centered_clip"})
    if unexpected:
        raise ValueError(f"SpLiCE dataset cache contains unsupported keys: {sorted(unexpected)}")
    missing = REQUIRED_CACHE_KEYS.difference(cache)
    if missing:
        raise ValueError(f"SpLiCE dataset cache is missing required keys: {sorted(missing)}")
    if cache["cache_version"] != SPLICE_DATASET_CACHE_VERSION:
        raise ValueError(
            f"Unsupported SpLiCE dataset cache version {cache['cache_version']!r}; "
            f"expected {SPLICE_DATASET_CACHE_VERSION}."
        )
    if "provenance" in cache and not isinstance(cache["provenance"], dict):
        raise ValueError("Optional provenance must be a dictionary.")

    sample_ids = list(cache["sample_ids"])
    if not sample_ids or len(sample_ids) != len(set(map(str, sample_ids))):
        raise ValueError("sample_ids must be non-empty and unique.")
    n_samples = len(sample_ids)
    clip = _normalized_rows(cache["clip_embeddings"], "clip_embeddings")
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


def _active_concept_indices(
    codes: torch.Tensor,
    config: CrpAuditConfig,
    sample_weights: torch.Tensor | None = None,
) -> list[int]:
    occurrences = (codes > 0).float()
    frequency = (
        occurrences.mean(dim=0)
        if sample_weights is None
        else (occurrences * sample_weights.unsqueeze(1)).mean(dim=0)
    )
    active = torch.where(
        (frequency >= config.min_concept_frequency) & (frequency <= config.max_concept_frequency)
    )[0]
    return [int(index) for index in active.tolist()]


def _group_concepts(
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

    active_indices = _active_concept_indices(codes, config, sample_weights_tensor)
    if not active_indices:
        return []

    active = torch.as_tensor(active_indices, dtype=torch.long)
    # Keep only one concept-by-sample copy. F.normalize would allocate a
    # second copy of this matrix, which is prohibitively large for CelebA's
    # 162k samples and 20k-concept vocabulary.
    active_codes = codes.index_select(1, active).T.contiguous()
    if sample_weights_tensor is not None:
        active_codes.mul_(sample_weights_tensor.sqrt().unsqueeze(0))
    active_codes.div_(active_codes.norm(dim=1, keepdim=True).clamp_min(1e-12))
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


def _grouping_config(config: CrpAuditConfig) -> dict:
    values = asdict(config)
    return {name: values[name] for name in GROUPING_CONFIG_FIELDS}


def _concept_group_diagnostics(groups: Sequence[dict], active_count: int) -> dict:
    sizes = [int(group["size"]) for group in groups]
    composite = [group for group in groups if int(group["size"]) > 1]
    composite_concepts = sum(int(group["size"]) for group in composite)
    largest = sorted(composite, key=lambda group: (-int(group["size"]), int(group["group_id"])))[:20]
    return {
        "total_active_concepts": active_count,
        "total_groups": len(groups),
        "singleton_count": sum(size == 1 for size in sizes),
        "singleton_fraction": sum(size == 1 for size in sizes) / len(groups) if groups else 0.0,
        "composite_group_count": len(composite),
        "concepts_in_composite_groups": composite_concepts,
        "concepts_in_composite_groups_fraction": composite_concepts / active_count if active_count else 0.0,
        "groups_of_size_2": sum(size == 2 for size in sizes),
        "groups_of_size_3": sum(size == 3 for size in sizes),
        "groups_of_size_4": sum(size == 4 for size in sizes),
        "groups_of_size_5_plus": sum(size >= 5 for size in sizes),
        "mean_group_size": statistics.mean(sizes) if sizes else 0.0,
        "median_group_size": statistics.median(sizes) if sizes else 0.0,
        "maximum_group_size": max(sizes, default=0),
        "largest_composite_groups": [
            {
                "group_id": int(group["group_id"]),
                "size": int(group["size"]),
                "concepts": list(group["concepts"]),
                "listing": ", ".join(group["concepts"]),
            }
            for group in largest
        ],
    }


def _concept_group_report_diagnostics(
    cache: dict,
    groups: Sequence[dict],
    config: CrpAuditConfig,
    representative_count: int = 5,
) -> dict:
    """Collect grouping evidence for the human-facing report without changing grouping."""

    codes = cache["splice_codes"]
    frequencies = (codes > 0).float().mean(dim=0)
    text_directions = F.normalize(cache["dictionary"], dim=1)
    activation_norms = torch.linalg.vector_norm(codes, dim=0).clamp_min(1e-12)

    below_minimum = frequencies < config.min_concept_frequency
    above_maximum = frequencies > config.max_concept_frequency
    active_below_minimum = torch.where((frequencies > 0) & below_minimum)[0].tolist()
    above_maximum_indices = torch.where(above_maximum)[0].tolist()

    def filtered_concepts(indices: Sequence[int]) -> list[dict]:
        return [
            {
                "concept": cache["vocabulary"][index],
                "concept_index": int(index),
                "frequency": float(frequencies[index]),
            }
            for index in sorted(
                (int(value) for value in indices),
                key=lambda index: (-float(frequencies[index]), index),
            )
        ]

    composite_groups = []
    for group in groups:
        if int(group["size"]) <= 1:
            continue
        indices = [int(value) for value in group["concept_indices"]]
        group_activation = codes[:, indices].sum(dim=1)
        representative_positions = sorted(
            range(len(cache["sample_ids"])),
            key=lambda position: (-float(group_activation[position]), position),
        )[:representative_count]

        relations = []
        for left_offset, left_index in enumerate(indices):
            for right_index in indices[left_offset + 1:]:
                text_similarity = float(text_directions[left_index] @ text_directions[right_index])
                coactivation = float(
                    torch.dot(codes[:, left_index], codes[:, right_index])
                    / (activation_norms[left_index] * activation_norms[right_index])
                )
                text_passed = text_similarity >= config.text_similarity_threshold
                coactivation_passed = coactivation >= config.coactivation_threshold
                lexical_passed = _lexical_key(cache["vocabulary"][left_index]) == _lexical_key(
                    cache["vocabulary"][right_index]
                )
                if not lexical_passed and not (text_passed and coactivation_passed):
                    continue
                relations.append(
                    {
                        "concept_a": cache["vocabulary"][left_index],
                        "concept_b": cache["vocabulary"][right_index],
                        "text_similarity": text_similarity,
                        "coactivation": coactivation,
                        "text_threshold_passed": text_passed,
                        "coactivation_threshold_passed": coactivation_passed,
                        "lexical_family_passed": lexical_passed,
                    }
                )

        failed_text = sum(not relation["text_threshold_passed"] for relation in relations)
        failed_coactivation = sum(
            not relation["coactivation_threshold_passed"] for relation in relations
        )
        failed_both = sum(
            not relation["text_threshold_passed"]
            and not relation["coactivation_threshold_passed"]
            for relation in relations
        )
        if failed_both:
            verdict = "suspicious"
            reason = "lexical grouping relation has weak semantic similarity and low coactivation"
        elif failed_text or failed_coactivation:
            verdict = "borderline"
            weak = []
            if failed_text:
                weak.append("weak semantic similarity")
            if failed_coactivation:
                weak.append("low coactivation")
            reason = "one or more lexical grouping relations have " + " and ".join(weak)
        else:
            verdict = "good"
            reason = ""

        composite_groups.append(
            {
                "group_id": int(group["group_id"]),
                "verdict": verdict,
                "verdict_reason": reason,
                "relations": relations,
                "representative_samples": [
                    {
                        "sample_id": str(cache["sample_ids"][position]),
                        "activation": float(group_activation[position]),
                    }
                    for position in representative_positions
                ],
            }
        )

    return {
        "filtering_funnel": {
            "raw_vocabulary_count": len(cache["vocabulary"]),
            "dataset_active_count": int((frequencies > 0).sum().item()),
            "frequency_filtered_count": int(
                ((frequencies >= config.min_concept_frequency)
                 & (frequencies <= config.max_concept_frequency)).sum().item()
            ),
            "final_group_count": len(groups),
            "below_minimum_count": int(below_minimum.sum().item()),
            "above_maximum_count": int(above_maximum.sum().item()),
            "inactive_count": int((frequencies == 0).sum().item()),
        },
        "active_below_minimum": filtered_concepts(active_below_minimum),
        "above_maximum": filtered_concepts(above_maximum_indices),
        "composite_groups": composite_groups,
    }


def build_concept_groups(splice_dataset_cache: dict, config: CrpAuditConfig) -> dict:
    """Generate reusable concept groups from a frozen SpLiCE dataset cache."""

    _validate_config(config)
    cache = validate_splice_dataset_cache(splice_dataset_cache)
    active_indices = _active_concept_indices(cache["splice_codes"], config)
    concept_indices = _group_concepts(
        cache["splice_codes"], cache["dictionary"], cache["vocabulary"], config
    )
    groups = [
        {
            "group_id": group_id,
            "concept_indices": indices,
            "concepts": [cache["vocabulary"][index] for index in indices],
            "size": len(indices),
        }
        for group_id, indices in enumerate(concept_indices)
    ]
    diagnostics = _concept_group_diagnostics(groups, len(active_indices))
    report_diagnostics = _concept_group_report_diagnostics(cache, groups, config)
    return {
        "artifact": COSPRO_CONCEPT_GROUP_ARTIFACT,
        "concept_groups_version": CONCEPT_GROUPS_VERSION,
        "cache_version": int(cache.get("cache_version", SPLICE_DATASET_CACHE_VERSION)),
        "sample_ids": cache["sample_ids"],
        "provenance": dict(cache.get("provenance", {})),
        "config": _grouping_config(config),
        "vocabulary": cache["vocabulary"],
        "active_concept_count": len(active_indices),
        "active_concept_indices": active_indices,
        "groups": groups,
        "group_sizes": [group["size"] for group in groups],
        "diagnostics": diagnostics,
        "report_diagnostics": report_diagnostics,
    }


def validate_concept_groups(artifact: dict, cache: dict | None = None) -> dict:
    """Validate a reusable concept-group artifact and optionally bind it to a cache."""

    if not isinstance(artifact, dict):
        raise ValueError("CoSpRo concept groups must be a dictionary.")
    if artifact.get("artifact") not in {
        COSPRO_CONCEPT_GROUP_ARTIFACT,
        *LEGACY_CONCEPT_GROUP_ARTIFACTS,
    }:
        raise ValueError(f"Unexpected concept-group artifact type: {artifact.get('artifact')!r}.")
    if artifact.get("concept_groups_version") != CONCEPT_GROUPS_VERSION:
        raise ValueError(
            f"Unsupported concept-group version {artifact.get('concept_groups_version')!r}; "
            f"expected {CONCEPT_GROUPS_VERSION}."
        )
    required = {
        "sample_ids", "config", "vocabulary", "active_concept_count",
        "active_concept_indices", "groups", "group_sizes", "diagnostics",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Concept-group artifact is missing required keys: {sorted(missing)}")
    config = artifact["config"]
    if not isinstance(config, dict) or set(config) != set(GROUPING_CONFIG_FIELDS):
        raise ValueError("Concept-group config must contain exactly the CoSpRo grouping settings.")
    vocabulary = [str(value) for value in artifact["vocabulary"]]
    active_indices = [int(value) for value in artifact["active_concept_indices"]]
    if len(active_indices) != len(set(active_indices)) or any(
        index < 0 or index >= len(vocabulary) for index in active_indices
    ):
        raise ValueError("Active concept indices must be unique vocabulary indices.")
    if int(artifact["active_concept_count"]) != len(active_indices):
        raise ValueError("active_concept_count does not match active_concept_indices.")

    groups = list(artifact["groups"])
    seen_indices: set[int] = set()
    for expected_id, group in enumerate(groups):
        indices = [int(value) for value in group.get("concept_indices", [])]
        concepts = [str(value) for value in group.get("concepts", [])]
        if int(group.get("group_id", -1)) != expected_id:
            raise ValueError("Concept group IDs must be contiguous and ordered.")
        if not indices or len(indices) != int(group.get("size", -1)):
            raise ValueError(f"Concept group {expected_id} has an invalid size.")
        if indices != sorted(indices) or seen_indices.intersection(indices):
            raise ValueError("Concept groups must contain sorted, disjoint concept indices.")
        if any(index not in active_indices for index in indices):
            raise ValueError("Concept groups may contain only active concept indices.")
        if concepts != [vocabulary[index] for index in indices]:
            raise ValueError(f"Concept labels do not match indices in group {expected_id}.")
        seen_indices.update(indices)
    if [int(value) for value in artifact["group_sizes"]] != [int(group["size"]) for group in groups]:
        raise ValueError("group_sizes does not match the serialized groups.")
    expected_diagnostics = _concept_group_diagnostics(groups, len(active_indices))
    if artifact["diagnostics"] != expected_diagnostics:
        raise ValueError("Concept-group diagnostics do not match the serialized groups.")

    if cache is not None:
        cache = validate_splice_dataset_cache(cache)
        if [str(value) for value in artifact["sample_ids"]] != [str(value) for value in cache["sample_ids"]]:
            raise ValueError("Concept groups and SpLiCE dataset cache sample IDs do not exactly match.")
        if vocabulary != cache["vocabulary"]:
            raise ValueError("Concept groups and SpLiCE dataset cache vocabularies do not exactly match.")
        if int(artifact.get("cache_version", -1)) != int(cache["cache_version"]):
            raise ValueError("Concept groups and SpLiCE dataset cache versions do not match.")
    return artifact


def save_concept_groups_json(artifact: dict, path: str | Path) -> Path:
    """Atomically save a validated concept-group artifact."""

    validate_concept_groups(artifact)
    output_path = Path(path)
    if output_path.suffix.lower() != ".json":
        raise ValueError("Concept-group artifacts must use a .json file extension.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def load_concept_groups_json(path: str | Path) -> dict:
    """Load and validate a concept-group JSON artifact."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_concept_groups(artifact)


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


def _exact_topk_neighbors(
    features: torch.Tensor, k: int, chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices, similarities = [], []
    for start in range(0, len(features), chunk_size):
        stop = min(start + chunk_size, len(features))
        similarity = features[start:stop] @ features.T
        local_rows = torch.arange(stop - start, device=features.device)
        similarity[local_rows, torch.arange(start, stop, device=features.device)] = -torch.inf
        values, neighbours = similarity.topk(k, dim=1)
        indices.append(neighbours)
        similarities.append(values)
    return torch.cat(indices), torch.cat(similarities)


def _lsh_topk_neighbors(
    features: torch.Tensor,
    k: int,
    chunk_size: int,
    *,
    tables: int,
    bucket_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate cosine neighbours using deterministic SimHash buckets.

    Each random-hyperplane table supplies ``k`` candidates per row. Candidates
    are then reranked with their exact cosine similarities. Extending small
    buckets in hash-sorted order guarantees enough non-self candidates even
    for an unlucky hash partition.
    """

    n_samples, dimensions = features.shape
    bits = max(1, min(30, int(round(math.log2(max(2, n_samples / bucket_size))))))
    powers = (2 ** torch.arange(bits, dtype=torch.int64, device=features.device)).view(1, -1)
    table_indices = []
    for table in range(tables):
        generator = torch.Generator(device=features.device)
        generator.manual_seed(int(seed) + 1_000_003 * table)
        planes = torch.randn(
            dimensions, bits, generator=generator, device=features.device, dtype=features.dtype,
        )
        codes = (((features @ planes) >= 0).to(torch.int64) * powers).sum(dim=1)
        order = torch.argsort(codes, stable=True)
        sorted_codes = codes[order]
        boundaries = torch.cat((
            torch.zeros(1, dtype=torch.long, device=features.device),
            torch.where(sorted_codes[1:] != sorted_codes[:-1])[0] + 1,
            torch.tensor([n_samples], dtype=torch.long, device=features.device),
        )).cpu().tolist()
        candidates = torch.empty((n_samples, k), dtype=torch.long, device=features.device)
        for boundary_index in range(len(boundaries) - 1):
            start, stop = boundaries[boundary_index], boundaries[boundary_index + 1]
            # Small buckets borrow adjacent entries in deterministic hash order.
            needed = max(k + 1, bucket_size)
            extra = max(0, needed - (stop - start))
            pool_start = max(0, start - extra // 2)
            pool_stop = min(n_samples, stop + extra - (start - pool_start))
            pool_start = max(0, pool_start - max(0, needed - (pool_stop - pool_start)))
            query_rows = order[start:stop]
            pool_rows = order[pool_start:pool_stop]
            for offset in range(0, len(query_rows), chunk_size):
                rows = query_rows[offset:offset + chunk_size]
                similarity = features[rows] @ features[pool_rows].T
                similarity.masked_fill_(rows.view(-1, 1) == pool_rows.view(1, -1), -torch.inf)
                candidates[rows] = pool_rows[similarity.topk(k, dim=1).indices]
        table_indices.append(candidates)

    candidates = torch.cat(table_indices, dim=1).sort(dim=1).values
    final_indices, final_similarities = [], []
    row_ids = torch.arange(n_samples, device=features.device)
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        rows = row_ids[start:stop]
        candidate_rows = candidates[start:stop]
        similarity = (features[rows].unsqueeze(1) * features[candidate_rows]).sum(dim=2)
        similarity.masked_fill_(candidate_rows == rows.view(-1, 1), -torch.inf)
        duplicate = torch.zeros_like(candidate_rows, dtype=torch.bool)
        duplicate[:, 1:] = candidate_rows[:, 1:] == candidate_rows[:, :-1]
        similarity.masked_fill_(duplicate, -torch.inf)
        values, positions = similarity.topk(k, dim=1)
        final_indices.append(candidate_rows.gather(1, positions))
        final_similarities.append(values)
    return torch.cat(final_indices), torch.cat(final_similarities)


def topk_neighbors(
    features: torch.Tensor,
    k: int,
    chunk_size: int = 512,
    *,
    backend: str = "exact",
    ann_threshold: int = 20_000,
    ann_tables: int = 8,
    ann_bucket_size: int = 512,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine neighbours via exact chunks or a scalable PyTorch LSH index."""

    n_samples = len(features)
    if n_samples < 2:
        raise ValueError("At least two samples are required to construct relations.")
    k = min(k, n_samples - 1)
    resolved = "lsh" if backend == "auto" and n_samples >= ann_threshold else backend
    resolved = "exact" if resolved == "auto" else resolved
    if resolved == "exact":
        return _exact_topk_neighbors(features, k, chunk_size)
    if resolved == "lsh":
        return _lsh_topk_neighbors(
            features, k, chunk_size, tables=ann_tables, bucket_size=ann_bucket_size, seed=seed,
        )
    raise ValueError(f"Unknown neighbour backend: {backend!r}")


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
    config: CrpAuditConfig,
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
    config: CrpAuditConfig,
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
    configured_cap = getattr(config, "max_indegree", None)
    if configured_cap is None:
        indegree_cap = max(1, int(math.ceil(config.indegree_factor * average_indegree)))
        indegree_rule = "relative"
    else:
        indegree_cap = int(configured_cap)
        indegree_rule = "absolute"
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
            "indegree_rule": indegree_rule,
            "indegree_gini": _gini(indegree),
            "effective_donor_count": float((indegree.sum() ** 2) / indegree.square().sum())
            if float(indegree.square().sum()) > 0
            else 0.0,
        },
    }


def build_teacher_graph(
    splice_dataset_cache: dict,
    concept_groups: dict,
    config: CrpAuditConfig,
    concept_groups_source: dict | None = None,
    *,
    device: str | torch.device = "auto",
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
) -> dict:
    """Build a validated, label-free CoSpRo teacher graph.

    Grouping is an explicit input. Intervention geometry, null controls and
    sparse graph assembly remain implementation details.
    """

    _validate_config(config)
    cache = validate_splice_dataset_cache(splice_dataset_cache)
    concept_groups = validate_concept_groups(concept_groups)
    if [str(value) for value in concept_groups["sample_ids"]] != [
        str(value) for value in cache["sample_ids"]
    ]:
        raise ValueError("Concept groups and SpLiCE dataset cache sample IDs do not exactly match.")
    if [str(value) for value in concept_groups["vocabulary"]] != cache["vocabulary"]:
        raise ValueError("Concept groups and SpLiCE dataset cache vocabularies do not exactly match.")
    if int(concept_groups.get("cache_version", -1)) != int(cache["cache_version"]):
        raise ValueError("Concept groups and SpLiCE dataset cache versions do not match.")

    config_values = asdict(config)
    config_values.update(concept_groups["config"])
    config = CrpAuditConfig(**config_values)
    _validate_config(config)
    if str(device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA neighbour search was requested, but torch.cuda.is_available() is false.")

    audit_codes = cache["splice_codes"]
    groups = [list(group["concept_indices"]) for group in concept_groups["groups"]]
    search_features = cache["centered_clip"].to(device)
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    checkpoint_identity = hashlib.sha256(json.dumps({
        "schema": "cospro-group-checkpoint-v1",
        "config": asdict(config),
        "sample_ids": [str(value) for value in cache["sample_ids"]],
        "groups": groups,
        "concept_groups_source": concept_groups_source or {},
        "search_device": str(device),
        "cache_version": int(cache["cache_version"]),
        "cache_provenance": cache.get("provenance", {}),
        "representation_shapes": {
            "clip_embeddings": list(cache["clip_embeddings"].shape),
            "splice_codes": list(cache["splice_codes"].shape),
            "dictionary": list(cache["dictionary"].shape),
        },
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        manifest_path = checkpoint_root / "manifest.json"
        manifest = {
            "schema": "cospro-group-checkpoint-v1",
            "identity": checkpoint_identity,
            "group_count": len(groups),
            "config": asdict(config),
        }
        if resume and manifest_path.is_file():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing_manifest != manifest:
                raise RuntimeError(
                    f"Checkpoint identity does not match this graph audit: {checkpoint_root}"
                )
        else:
            temporary_manifest = manifest_path.with_suffix(f".{os.getpid()}.tmp")
            temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary_manifest, manifest_path)

    resolved_backend = (
        "lsh"
        if config.neighbor_backend == "auto" and len(search_features) >= config.ann_threshold
        else "exact" if config.neighbor_backend == "auto" else config.neighbor_backend
    )
    print(
        f"[INFO] Neighbour search backend={resolved_backend} device={device} "
        f"samples={len(search_features)}",
        flush=True,
    )
    raw_checkpoint = checkpoint_root / "raw_neighbours.pt" if checkpoint_root is not None else None
    if resume and raw_checkpoint is not None and raw_checkpoint.is_file():
        saved_raw = torch.load(raw_checkpoint, map_location="cpu", weights_only=True)
        if saved_raw.get("identity") != checkpoint_identity:
            raise RuntimeError(f"Invalid raw-neighbour checkpoint: {raw_checkpoint}")
        raw_neighbours = saved_raw["raw_neighbours"].to(device)
        if raw_neighbours.shape != (len(search_features), min(config.projected_neighbors, len(search_features) - 1)):
            raise RuntimeError(f"Raw-neighbour checkpoint has an invalid shape: {raw_checkpoint}")
        print(f"[INFO] Restored raw neighbours from {raw_checkpoint}", flush=True)
    else:
        raw_neighbours, _ = topk_neighbors(
            search_features,
            config.projected_neighbors,
            config.similarity_chunk_size,
            backend=config.neighbor_backend,
            ann_threshold=config.ann_threshold,
            ann_tables=config.ann_tables,
            ann_bucket_size=config.ann_bucket_size,
            seed=config.seed,
        )
        if raw_checkpoint is not None:
            _atomic_torch_save({
                "schema": "cospro-raw-neighbours-v1",
                "identity": checkpoint_identity,
                "raw_neighbours": raw_neighbours.cpu(),
            }, raw_checkpoint)
    n_samples = len(cache["sample_ids"])
    audit_geometry = _AuditGeometry(
        centered_clip=search_features,
        raw_neighbours=raw_neighbours,
        splice_codes=audit_codes,
    )

    audited_groups, candidate_evidence = [], []
    random_neighbour_cache: dict[int, list[dict]] = {}

    def random_neighbour_geometries(basis_rank: int) -> list[dict]:
        if basis_rank in random_neighbour_cache:
            return random_neighbour_cache[basis_rank]
        random_generator = torch.Generator().manual_seed(
            config.seed + 104_729 * basis_rank + 17
        )
        geometries = []
        for trial in range(config.null_trials):
            random_directions = torch.randn(
                basis_rank, search_features.shape[1], generator=random_generator,
            )
            random_basis = orthonormal_basis(
                random_directions, config.orthogonal_tolerance,
            ).to(device)
            geometries.append(_neighbor_geometry(
                audit_geometry,
                random_basis,
                config,
                search_seed=config.seed + 10_000_019 * basis_rank + trial + 1,
            ))
        random_neighbour_cache[basis_rank] = geometries
        return geometries

    print(f"[INFO] Auditing {len(groups)} concept groups over {n_samples} samples", flush=True)
    report_every = max(1, len(groups) // 20)
    restored_groups = 0
    for group_id, concept_indices in enumerate(groups):
        checkpoint_path = (
            checkpoint_root / f"group_{group_id:06d}.pt" if checkpoint_root is not None else None
        )
        if resume and checkpoint_path is not None and checkpoint_path.is_file():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if (
                saved.get("schema") != "cospro-group-checkpoint-v1"
                or saved.get("identity") != checkpoint_identity
                or int(saved.get("group_id", -1)) != group_id
            ):
                raise RuntimeError(f"Invalid or incompatible group checkpoint: {checkpoint_path}")
            group_payload = saved["group"]
            audited_groups.append(group_payload)
            if saved.get("candidate_evidence") is not None:
                candidate_evidence.append((group_id, saved["candidate_evidence"]))
            restored_groups += 1
            continue

        basis = orthonormal_basis(
            cache["dictionary"][concept_indices], config.orthogonal_tolerance,
        ).to(device)
        activation = audit_codes[:, concept_indices].sum(dim=1)
        neighbour_geometry = _neighbor_geometry(
            audit_geometry,
            basis,
            config,
            search_seed=config.seed + 1_000_003 * (group_id + 1),
        )
        group_geometry = _relation_geometry(
            audit_geometry, neighbour_geometry, config, concept_indices,
        )
        evidence = _score_relations(group_geometry, activation, config)
        basis_rank = basis.shape[1]
        random_geometries = [
            _relation_geometry(audit_geometry, geometry, config, concept_indices)
            for geometry in random_neighbour_geometries(basis_rank)
        ]
        group_generator = torch.Generator().manual_seed(
            config.seed + 2_000_003 * (group_id + 1)
        )
        random_scores, shuffled_scores = _null_scores(
            group_geometry,
            random_geometries,
            activation,
            config,
            group_generator,
        )
        null_scores = torch.tensor(random_scores + shuffled_scores)
        threshold = float(torch.quantile(null_scores, config.null_quantile)) if null_scores.numel() else math.inf
        selected = (
            evidence["coverage"] >= config.min_coverage
            and evidence["score"] > threshold
        )
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
            "residual_splice_gate_enabled": config.use_residual_splice_gate,
            "residual_splice_similarity_threshold": (
                config.residual_splice_similarity_threshold
                if config.use_residual_splice_gate
                else None
            ),
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
        saved_candidate = None
        if selected:
            saved_candidate = {
                key: evidence[key].cpu()
                for key in ("rows", "columns", "gain")
            }
            saved_candidate["confidence"] = evidence["confidence"].cpu() * null_excess_ratio
            candidate_evidence.append((group_id, saved_candidate))
        if checkpoint_path is not None:
            _atomic_torch_save({
                "schema": "cospro-group-checkpoint-v1",
                "identity": checkpoint_identity,
                "group_id": group_id,
                "group": group_payload,
                "candidate_evidence": saved_candidate,
            }, checkpoint_path)
        if (group_id + 1) % report_every == 0 or group_id + 1 == len(groups):
            print(
                f"[INFO] Audited {group_id + 1}/{len(groups)} groups; "
                f"passing_null={len(candidate_evidence)}",
                flush=True,
            )

    if restored_groups:
        print(
            f"[INFO] Restored {restored_groups}/{len(groups)} completed group audits from "
            f"{checkpoint_root}",
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
        "artifact": COSPRO_TEACHER_GRAPH_ARTIFACT,
        "graph_version": CRP_GRAPH_VERSION,
        "cache_version": int(cache.get("cache_version", SPLICE_DATASET_CACHE_VERSION)),
        "sample_ids": cache["sample_ids"],
        "config": config_payload,
        "grouping_config": dict(concept_groups["config"]),
        "concept_groups_source": concept_groups_source or {
            "path": "<in-memory>",
            "sha256": None,
            "artifact": concept_groups["artifact"],
            "concept_groups_version": concept_groups["concept_groups_version"],
        },
        "provenance": dict(cache.get("provenance", {})),
        "neighbor_search": {
            "requested_backend": config.neighbor_backend,
            "resolved_backend": resolved_backend,
            "approximate": resolved_backend == "lsh",
            "device": str(device),
            "ann_tables": config.ann_tables if resolved_backend == "lsh" else None,
            "ann_bucket_size": config.ann_bucket_size if resolved_backend == "lsh" else None,
            "shared_null_subspaces_by_rank": True,
        },
        "groups": audited_groups,
        "selected_group_ids": [group["group_id"] for group in audited_groups if group["selected"]],
        **graph,
    }


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_splice_dataset_cache(splice_dataset_cache: dict, path: str | Path) -> None:
    """Validate and atomically save a frozen SpLiCE dataset cache."""

    validate_splice_dataset_cache(splice_dataset_cache)
    _atomic_torch_save(splice_dataset_cache, Path(path))


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
    parser = argparse.ArgumentParser(description="Build a CoSpRo teacher graph from saved groups.")
    parser.add_argument(
        "--splice-dataset-cache", required=True, help="Frozen SpLiCE dataset cache (.pt)."
    )
    parser.add_argument("--concept-groups", required=True, help="Reusable concept_groups.json artifact.")
    parser.add_argument("--output", required=True, help="Complete teacher graph output (.json).")
    parser.add_argument("--html", help="HTML mechanism report (default: output with .html suffix).")
    parser.add_argument("--config", help="Optional JSON object overriding CrpAuditConfig fields.")
    parser.add_argument("--seed", type=int, help="Override the null-control seed.")
    parser.add_argument("--device", default="auto", help="Neighbour-search device.")
    parser.add_argument("--neighbor-backend", choices=("auto", "exact", "lsh"))
    parser.add_argument("--ann-threshold", type=int)
    parser.add_argument("--ann-tables", type=int)
    parser.add_argument("--ann-bucket-size", type=int)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--use-residual-splice-gate",
        type=_parse_bool,
        nargs="?",
        const=True,
        help="Enable the residual SpLiCE semantic gate.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_values = json.loads(args.config) if args.config else {}
    unknown = set(config_values).difference(CrpAuditConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown CoSpRo audit settings: {sorted(unknown)}")
    if args.seed is not None:
        config_values["seed"] = args.seed
    for name in ("neighbor_backend", "ann_threshold", "ann_tables", "ann_bucket_size"):
        value = getattr(args, name)
        if value is not None:
            config_values[name] = value
    if args.use_residual_splice_gate is not None:
        config_values["use_residual_splice_gate"] = args.use_residual_splice_gate
    config = CrpAuditConfig(**config_values)
    cache_path = Path(args.splice_dataset_cache)
    groups_path, output_path = Path(args.concept_groups), Path(args.output)
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    concept_groups = load_concept_groups_json(groups_path)
    source = {
        "path": str(groups_path.resolve()),
        "sha256": hashlib.sha256(groups_path.read_bytes()).hexdigest(),
        "artifact": concept_groups["artifact"],
        "concept_groups_version": concept_groups["concept_groups_version"],
    }
    artifact = build_teacher_graph(
        cache,
        concept_groups,
        config,
        source,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir or output_path.parent / "group_checkpoints",
        resume=not args.no_resume,
    )
    save_graph_json(artifact, output_path)
    from splice.cospro_reporting import render_teacher_graph_report
    render_teacher_graph_report(artifact, Path(args.html) if args.html else output_path.with_suffix(".html"))
    print(f"[INFO] Wrote {artifact['artifact']} to {output_path}")
    print(f"[INFO] Selected {len(artifact['selected_group_ids'])}/{len(artifact['groups'])} groups")


if __name__ == "__main__":
    main()
