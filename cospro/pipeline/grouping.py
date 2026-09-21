"""Concept grouping: from a frozen cache to a reusable set of concept groups.

Grouping merges vocabulary entries that mean the same thing for this dataset, by lexical key, text
similarity and co-activation. It consumes no label and produces a JSON artifact the audit resumes
from, so one grouping serves many graph builds.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from cospro.pipeline.cache import SPLICE_DATASET_CACHE_VERSION, validate_splice_dataset_cache
from cospro.pipeline.config import GROUPING_CONFIG_FIELDS, CoSpRoAuditConfig, _validate_config
from cospro.compat import LEGACY_CONCEPT_GROUP_ARTIFACTS

CONCEPT_GROUPS_VERSION = 1
COSPRO_CONCEPT_GROUP_ARTIFACT = "cospro_concept_groups_v1"


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
    config: CoSpRoAuditConfig,
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
    config: CoSpRoAuditConfig,
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


def _grouping_config(config: CoSpRoAuditConfig) -> dict:
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
    config: CoSpRoAuditConfig,
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


def build_concept_groups(splice_dataset_cache: dict, config: CoSpRoAuditConfig) -> dict:
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
