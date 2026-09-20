"""Teacher-graph assembly: audit every concept group, then build the sparse graph.

``build_teacher_graph`` is the stage that turns a cache plus concept groups into the artifact
training consumes. It audits each group, applies the selection rule, assembles a fixed-density
sparse graph from the surviving evidence and records the provenance a run is reproduced from. Long
audits checkpoint per group, so a Slurm job that runs out of time resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import torch

from cospro.pipeline.audit import (
    _AuditGeometry,
    _gini,
    _neighbor_geometry,
    _null_scores,
    _relation_geometry,
    _score_relations,
)
from cospro.pipeline.cache import SPLICE_DATASET_CACHE_VERSION, _atomic_torch_save, validate_splice_dataset_cache
from cospro.pipeline.config import CoSpRoAuditConfig, _validate_config
from cospro.pipeline.grouping import validate_concept_groups
from cospro.pipeline.neighbors import build_index, orthonormal_basis
from cospro.pipeline.selection import SelectionRule, selection_rule

# GRAPH_VERSION remains the legacy v2 format. CoSpRo v3 has its own version
# because its fixed-density and validation fields are method changes.
GRAPH_VERSION = 2
COSPRO_GRAPH_VERSION = 3
COSPRO_TEACHER_GRAPH_ARTIFACT = "cospro_teacher_graph_v3"


def _build_teacher_graph(
    n_samples: int,
    selected: list[tuple[int, dict]],
    config: CoSpRoAuditConfig,
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
    config: CoSpRoAuditConfig,
    concept_groups_source: dict | None = None,
    *,
    device: str | torch.device = "auto",
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
    selection: str | SelectionRule | None = None,
) -> dict:
    """Build a validated, label-free CoSpRo teacher graph.

    Grouping is an explicit input. Intervention geometry, null controls and
    sparse graph assembly remain implementation details.
    """

    _validate_config(config)
    rule = selection_rule(selection)
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
    config = CoSpRoAuditConfig(**config_values)
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

    neighbor_index = build_index(
        config.neighbor_backend,
        len(search_features),
        chunk_size=config.similarity_chunk_size,
        ann_threshold=config.ann_threshold,
        ann_tables=config.ann_tables,
        ann_bucket_size=config.ann_bucket_size,
        seed=config.seed,
    )
    resolved_backend = neighbor_index.name
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
        raw_neighbours, _ = neighbor_index.search(
            search_features, min(config.projected_neighbors, len(search_features) - 1),
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
        selected = rule.accepts(
            {"coverage": evidence["coverage"], "score": evidence["score"], "null_threshold": threshold},
            config,
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

    selected_evidence = rule.retain(candidate_evidence, audited_groups, config)
    retained_group_ids = {group_id for group_id, _ in selected_evidence}
    for group in audited_groups:
        if group["selected"] and group["group_id"] not in retained_group_ids:
            group["selected"] = False
            group["rejection_reason"] = "max_selected_groups_cap"

    graph = _build_teacher_graph(n_samples, selected_evidence, config)
    config_payload = asdict(config)
    return {
        "artifact": COSPRO_TEACHER_GRAPH_ARTIFACT,
        "graph_version": COSPRO_GRAPH_VERSION,
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
        "selection": rule.provenance(config),
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
