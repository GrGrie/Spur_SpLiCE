"""Experimental target-preserving replacements on a matched raw-CLIP graph.

This module is deliberately downstream of the ordinary CRP audit.  It never
selects concepts or computes CRP evidence; it only proposes bounded replacements
using edges already retained by an existing, label-free CRP graph.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch

from splice.crp import topk_neighbors, validate_feature_cache


SAFE_CRP_GRAPH_ARTIFACT = "splice_safe_crp_teacher_graph"
SAFE_CRP_GRAPH_VERSION = 1
SAFE_CONFIG_FIELDS = {
    "raw_guard_k",
    "max_replacements_per_row",
    "max_replacement_weight",
    "min_treated_anchor_fraction",
    "min_crp_weight_mass_fraction",
}


@dataclass(frozen=True)
class SafeCrpGraphConfig:
    raw_guard_k: int = 20
    max_replacements_per_row: int = 1
    max_replacement_weight: float = 0.34
    min_treated_anchor_fraction: float = 0.05
    min_crp_weight_mass_fraction: float = 0.01

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "SafeCrpGraphConfig":
        value = dict(value or {})
        unknown = set(value).difference(SAFE_CONFIG_FIELDS)
        if unknown:
            raise ValueError(f"Unknown safe graph settings: {sorted(unknown)}")
        config = cls(**value)
        config.validate()
        return config

    def validate(self) -> None:
        if self.raw_guard_k <= 0:
            raise ValueError("safe_graph.raw_guard_k must be positive")
        if self.max_replacements_per_row != 1:
            raise ValueError("safe_graph.max_replacements_per_row must be 1 for this protocol")
        if not 0 <= self.max_replacement_weight <= 1:
            raise ValueError("safe_graph.max_replacement_weight must be in [0, 1]")
        if not 0 <= self.min_treated_anchor_fraction <= 1:
            raise ValueError("safe_graph.min_treated_anchor_fraction must be in [0, 1]")
        if not 0 <= self.min_crp_weight_mass_fraction <= 1:
            raise ValueError("safe_graph.min_crp_weight_mass_fraction must be in [0, 1]")


def _json_ready(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _payload_fingerprint(value: dict) -> str:
    encoded = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _require_aligned_tensor(graph: dict, key: str, shape: torch.Size, dtype) -> torch.Tensor:
    if key not in graph:
        raise ValueError(f"Safe CRP graph source is missing {key!r}.")
    value = torch.as_tensor(graph[key], dtype=dtype).detach().cpu()
    if value.shape != shape:
        raise ValueError(f"Safe CRP graph field {key!r} must have shape {tuple(shape)}.")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"Safe CRP graph field {key!r} contains non-finite values.")
    return value


def _validate_source_graphs(crp_graph: dict, raw_graph: dict, sample_ids: Sequence[str]) -> tuple[dict, dict]:
    from splice.crp_training import validate_teacher_graph

    crp = validate_teacher_graph(crp_graph, sample_ids)
    raw = validate_teacher_graph(raw_graph, sample_ids)
    if crp["artifact"] not in {
        "splice_crp_v2_teacher_graph",
        "splice_crp_v3_teacher_graph",
        "splice_crp_v4_teacher_graph",
    }:
        raise ValueError("Safe CRP graph proposals must come from an ordinary CRP graph.")
    if raw["artifact"] != "splice_raw_clip_matched_teacher_graph":
        raise ValueError("Safe CRP graph requires a matched raw-CLIP graph.")
    return crp, raw


def build_safe_crp_graph(
    cache: dict,
    crp_graph: dict,
    raw_graph: dict,
    safe_config: Mapping[str, object] | SafeCrpGraphConfig | None = None,
    source_crp_fingerprint: str | None = None,
    source_raw_fingerprint: str | None = None,
) -> dict:
    """Build the deterministic safe graph from already-fixed source artifacts."""

    config = safe_config if isinstance(safe_config, SafeCrpGraphConfig) else SafeCrpGraphConfig.from_mapping(safe_config)
    cache = validate_feature_cache(cache)
    crp, raw = _validate_source_graphs(crp_graph, raw_graph, cache["sample_ids"])
    if int(raw["degree_stats"].get("indegree_cap", 0)) <= 0:
        raise ValueError("Matched raw graph must record a positive absolute indegree cap.")

    indices = raw["neighbor_indices"].clone()
    weights = raw["weights"].clone()
    shape = indices.shape
    crp_indices = crp["neighbor_indices"]
    crp_gains = _require_aligned_tensor(crp, "intervention_gains", crp_indices.shape, torch.float32)
    crp_groups = _require_aligned_tensor(crp, "group_ids", crp_indices.shape, torch.long)
    crp_confidence = _require_aligned_tensor(crp, "edge_confidences", crp_indices.shape, torch.float32)
    if torch.any(crp_gains < 0) or torch.any(crp_groups < -1):
        raise ValueError("CRP provenance fields contain invalid retained-edge values.")

    raw_guard_indices, _ = topk_neighbors(cache["centered_clip"], config.raw_guard_k)
    raw_sets = [set(row.tolist()) for row in raw_guard_indices]
    raw_rows = [set(row[row >= 0].tolist()) for row in raw["neighbor_indices"]]
    indegree = torch.bincount(
        raw["neighbor_indices"][raw["neighbor_indices"] >= 0],
        minlength=len(cache["sample_ids"]),
    ).long()
    indegree_cap = int(raw["degree_stats"]["indegree_cap"])
    selected_group_ids = set(int(value) for value in crp.get("selected_group_ids", []))
    declared_group_ids = {int(group["group_id"]) for group in crp.get("groups", []) if group.get("selected", False)}
    valid_group_ids = selected_group_ids or declared_group_ids

    proposals: list[tuple[float, int, int, int, int]] = []
    for row in range(shape[0]):
        for position, donor_tensor in enumerate(crp_indices[row]):
            donor = int(donor_tensor)
            if donor < 0:
                continue
            group_id = int(crp_groups[row, position])
            gain = float(crp_gains[row, position])
            if donor == row or donor in raw_rows[row] or donor not in raw_sets[row]:
                continue
            if gain <= 0 or group_id < 0 or (valid_group_ids and group_id not in valid_group_ids):
                continue
            proposals.append((float(crp_confidence[row, position]), row, donor, position, group_id))
    proposals.sort(key=lambda item: (-item[0], item[1], item[2]))

    edge_source = torch.zeros(shape, dtype=torch.long)
    edge_source[indices >= 0] = 1
    safe_group_ids = torch.full(shape, -1, dtype=torch.long)
    safe_gains = torch.zeros(shape, dtype=torch.float32)
    safe_confidences = torch.zeros(shape, dtype=torch.float32)
    replaced_rows: set[int] = set()
    replacements: list[dict[str, int | float]] = []
    for confidence, row, donor, crp_position, group_id in proposals:
        if row in replaced_rows or len(replaced_rows) >= shape[0]:
            continue
        raw_positions = torch.where(indices[row] >= 0)[0]
        if not len(raw_positions):
            continue
        replacement_position = min(
            (int(position) for position in raw_positions),
            key=lambda position: (float(weights[row, position]), position),
        )
        replacement_weight = float(weights[row, replacement_position])
        if replacement_weight > config.max_replacement_weight:
            continue
        if int(indegree[donor]) >= indegree_cap:
            continue
        removed_donor = int(indices[row, replacement_position])
        indices[row, replacement_position] = donor
        indegree[removed_donor] -= 1
        indegree[donor] += 1
        edge_source[row, replacement_position] = 2
        safe_group_ids[row, replacement_position] = group_id
        safe_gains[row, replacement_position] = crp_gains[row, crp_position]
        safe_confidences[row, replacement_position] = crp_confidence[row, crp_position]
        replaced_rows.add(row)
        replacements.append(
            {
                "row": row,
                "removed_donor": removed_donor,
                "crp_donor": donor,
                "weight": replacement_weight,
                "confidence": confidence,
                "group_id": group_id,
            }
        )

    total_weight = float(raw["weights"].sum())
    replacement_weight_mass = sum(float(item["weight"]) for item in replacements)
    safe_stats = dict(raw.get("degree_stats", {}))
    safe_stats.update(
        {
            "maximum_indegree": int(indegree.max()) if len(indegree) else 0,
            "safe_replaced_edges": len(replacements),
            "safe_treated_anchors": len(replaced_rows),
            "safe_treated_anchor_fraction": len(replaced_rows) / max(1, len(cache["sample_ids"])),
            "safe_crp_weight_mass_fraction": replacement_weight_mass / total_weight if total_weight else 0.0,
        }
    )
    result = {
        "artifact": SAFE_CRP_GRAPH_ARTIFACT,
        "graph_version": SAFE_CRP_GRAPH_VERSION,
        "sample_ids": list(cache["sample_ids"]),
        "config": dict(raw.get("config", {})),
        "safe_config": asdict(config),
        "provenance": dict(raw.get("provenance", {})),
        "source_crp_fingerprint": source_crp_fingerprint or _payload_fingerprint(crp),
        "source_raw_fingerprint": source_raw_fingerprint or _payload_fingerprint(raw),
        "neighbor_indices": indices,
        "weights": weights,
        "confidence": raw["confidence"].clone(),
        "anchor_confidence": raw["anchor_confidence"].clone(),
        "edge_source": edge_source,
        "group_ids": safe_group_ids,
        "intervention_gains": safe_gains,
        "edge_confidences": safe_confidences,
        "degree_stats": safe_stats,
        "safe_replacements": replacements,
        "control": "matched raw-CLIP graph with bounded, target-preserving CRP rerank proposals",
    }
    return validate_safe_crp_graph(result, cache["sample_ids"])


def validate_safe_crp_graph(graph: dict, expected_sample_ids: Sequence[str] | None = None) -> dict:
    """Validate safe provenance and structural invariants before training."""

    from splice.crp_training import validate_teacher_graph

    validated = validate_teacher_graph(graph, expected_sample_ids)
    if validated["artifact"] != SAFE_CRP_GRAPH_ARTIFACT or validated["graph_version"] != SAFE_CRP_GRAPH_VERSION:
        raise ValueError("Unsupported safe CRP graph artifact/version.")
    config = SafeCrpGraphConfig.from_mapping(validated.get("safe_config"))
    shape = validated["neighbor_indices"].shape
    edge_source = _require_aligned_tensor(validated, "edge_source", shape, torch.long)
    groups = _require_aligned_tensor(validated, "group_ids", shape, torch.long)
    gains = _require_aligned_tensor(validated, "intervention_gains", shape, torch.float32)
    confidences = _require_aligned_tensor(validated, "edge_confidences", shape, torch.float32)
    valid = validated["neighbor_indices"] >= 0
    if torch.any(edge_source[~valid] != 0) or torch.any(edge_source[valid] == 0):
        raise ValueError("Safe edge_source must be 0 for padding and nonzero for every edge.")
    if torch.any((edge_source == 1) & ((groups != -1) | (gains != 0) | (confidences != 0))):
        raise ValueError("Raw safe edges must have zero CRP provenance.")
    if torch.any((edge_source == 2) & ((groups < 0) | (gains <= 0) | (confidences <= 0))):
        raise ValueError("CRP replacement edges must have valid positive provenance.")
    if torch.any(edge_source > 2):
        raise ValueError("Unknown safe edge provenance.")
    stats = validated.get("degree_stats", {})
    if "safe_replaced_edges" not in stats or "safe_treated_anchors" not in stats:
        raise ValueError("Safe graph degree_stats must record replacement counts.")
    if int((edge_source == 2).sum()) != int(stats["safe_replaced_edges"]):
        raise ValueError("Safe replacement count does not match edge provenance.")
    if int((edge_source == 2).any(dim=1).sum()) != int(stats["safe_treated_anchors"]):
        raise ValueError("Safe treated-anchor count does not match edge provenance.")
    indegree = torch.bincount(
        validated["neighbor_indices"][validated["neighbor_indices"] >= 0],
        minlength=len(validated["sample_ids"]),
    )
    if int(indegree.max()) > int(stats.get("indegree_cap", 0)):
        raise ValueError("Safe CRP graph exceeds its absolute indegree cap.")
    if config.max_replacements_per_row == 1 and torch.any((edge_source == 2).sum(dim=1) > 1):
        raise ValueError("Safe graph permits at most one replacement per row.")
    return {**validated, "edge_source": edge_source, "group_ids": groups,
            "intervention_gains": gains, "edge_confidences": confidences}


def safe_training_gate(graph: dict) -> tuple[bool, str | None]:
    config = SafeCrpGraphConfig.from_mapping(graph.get("safe_config"))
    stats = graph.get("degree_stats", {})
    if float(stats.get("safe_treated_anchor_fraction", 0.0)) < config.min_treated_anchor_fraction:
        return False, "safe_treated_anchor_fraction"
    if float(stats.get("safe_crp_weight_mass_fraction", 0.0)) < config.min_crp_weight_mass_fraction:
        return False, "safe_crp_weight_mass_fraction"
    return True, None
