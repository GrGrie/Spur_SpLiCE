"""Fast CRPv4 concept-group reconstruction screen and mini intervention audit.

The screen is deliberately smaller than a teacher-graph audit. It builds the
same flat CRP concept groups, collapses every group to one text-space prototype,
measures how faithfully those prototypes reproduce the SpLiCE representation,
and optionally audits a small deterministic image subset. It never reads target
or spurious annotations; those are added only by the HTML renderer.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import Sequence

import torch
import torch.nn.functional as F

from splice import crp as canonical
from splice.spatial_balance import load_spatial_balance_artifact, spatially_balanced_codes


GROUP_SCREEN_ARTIFACT = "splice_crpv4_group_screen_v1"


@dataclass(frozen=True)
class ReconstructionScreenConfig:
    fidelity_threshold: float = 0.95
    target_image_coverage: float = 0.99
    curve_points: int = 80
    top_groups_per_image: int = 3
    top_images_per_group: int = 8
    coherence_warning_threshold: float = 0.55


@dataclass(frozen=True)
class MiniInterventionConfig:
    enabled: bool = True
    sample_count: int = 1024
    max_groups: int = 24
    projected_neighbors: int = 10
    null_trials: int = 4
    null_quantile: float = 0.95
    activation_difference_quantile: float = 0.85
    min_intervention_gain: float = 5e-4
    min_coverage: float = 0.01
    residual_splice_similarity_threshold: float = 0.25
    example_edges_per_group: int = 3
    seed: int = 0
    device: str = "auto"


def _validate_screen_config(config: ReconstructionScreenConfig) -> None:
    if not 0 < config.fidelity_threshold <= 1:
        raise ValueError("fidelity_threshold must lie in (0, 1].")
    if not 0 < config.target_image_coverage <= 1:
        raise ValueError("target_image_coverage must lie in (0, 1].")
    if config.curve_points <= 0:
        raise ValueError("curve_points must be positive.")
    if config.top_groups_per_image <= 0 or config.top_images_per_group <= 0:
        raise ValueError("top-group and top-image counts must be positive.")
    if not -1 <= config.coherence_warning_threshold <= 1:
        raise ValueError("coherence_warning_threshold must lie in [-1, 1].")


def _validate_mini_config(config: MiniInterventionConfig) -> None:
    positive = {
        "sample_count": config.sample_count,
        "max_groups": config.max_groups,
        "projected_neighbors": config.projected_neighbors,
        "null_trials": config.null_trials,
        "example_edges_per_group": config.example_edges_per_group,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    for name, value in {
        "null_quantile": config.null_quantile,
        "activation_difference_quantile": config.activation_difference_quantile,
        "min_coverage": config.min_coverage,
        "residual_splice_similarity_threshold": config.residual_splice_similarity_threshold,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0, 1].")


def _safe_normalize(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values.float(), dim=1, eps=1e-12)


def _cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (_safe_normalize(left) * _safe_normalize(right)).sum(dim=1).clamp(-1.0, 1.0)


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float(), q)) if values.numel() else 0.0


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _group_representations(
    codes: torch.Tensor,
    dictionary: torch.Tensor,
    vocabulary: Sequence[str],
    groups: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    normalized_dictionary = F.normalize(dictionary.float(), dim=1)
    activations: list[torch.Tensor] = []
    prototypes: list[torch.Tensor] = []
    summaries: list[dict] = []
    for group_id, concept_indices in enumerate(groups):
        indices = torch.as_tensor(concept_indices, dtype=torch.long)
        directions = normalized_dictionary.index_select(0, indices)
        prototype = F.normalize(directions.mean(dim=0), dim=0)
        activation = codes.index_select(1, indices).sum(dim=1)
        pairwise = directions @ directions.T
        if len(indices) > 1:
            mask = ~torch.eye(len(indices), dtype=torch.bool)
            similarities = pairwise[mask]
            mean_similarity = float(similarities.mean())
            minimum_similarity = float(similarities.min())
        else:
            mean_similarity = 1.0
            minimum_similarity = 1.0
        nearest_index = int((normalized_dictionary @ prototype).argmax())
        summaries.append(
            {
                "group_id": group_id,
                "name": str(vocabulary[nearest_index]),
                "concept_indices": [int(value) for value in concept_indices],
                "concepts": [str(vocabulary[index]) for index in concept_indices],
                "concept_count": len(concept_indices),
                "mean_text_similarity": mean_similarity,
                "minimum_text_similarity": minimum_similarity,
                "activation_frequency": float((activation > 0).float().mean()),
                "activation_mass": float(activation.sum()),
            }
        )
        activations.append(activation)
        prototypes.append(prototype)
    if not activations:
        return (
            torch.empty(codes.shape[0], 0),
            torch.empty(0, dictionary.shape[1]),
            [],
        )
    return torch.stack(activations, dim=1), torch.stack(prototypes), summaries


def _rank_groups(activations: torch.Tensor, summaries: list[dict]) -> list[int]:
    return sorted(
        range(activations.shape[1]),
        key=lambda group_id: (
            -float(summaries[group_id]["activation_mass"]),
            -float(summaries[group_id]["activation_frequency"]),
            group_id,
        ),
    )


def _coverage_curve(
    activations: torch.Tensor,
    prototypes: torch.Tensor,
    order: Sequence[int],
    source_reconstruction: torch.Tensor,
    balanced_reconstruction: torch.Tensor,
    config: ReconstructionScreenConfig,
) -> tuple[list[dict], int, torch.Tensor, torch.Tensor]:
    n_groups = len(order)
    if n_groups == 0:
        zeros = torch.zeros(source_reconstruction.shape[0])
        return [], 0, zeros, zeros
    ordered_activations = activations[:, list(order)]
    ordered_prototypes = prototypes[list(order)]
    step = max(1, math.ceil(n_groups / config.curve_points))
    reconstruction = torch.zeros_like(source_reconstruction)
    curve: list[dict] = []
    chosen_count: int | None = None
    chosen_source_fidelity: torch.Tensor | None = None
    chosen_internal_fidelity: torch.Tensor | None = None

    start = 0
    while start < n_groups:
        stop = min(start + step, n_groups)
        before = reconstruction.clone()
        reconstruction.add_(
            ordered_activations[:, start:stop] @ ordered_prototypes[start:stop]
        )
        source_fidelity = _cosine_rows(reconstruction, source_reconstruction)
        internal_fidelity = _cosine_rows(reconstruction, balanced_reconstruction)
        coverage = float((source_fidelity >= config.fidelity_threshold).float().mean())
        curve.append(
            {
                "group_count": stop,
                "source_coverage": coverage,
                "median_source_fidelity": float(source_fidelity.median()),
                "median_internal_fidelity": float(internal_fidelity.median()),
            }
        )
        if chosen_count is None and coverage >= config.target_image_coverage:
            reconstruction = before
            for position in range(start, stop):
                reconstruction.add_(
                    ordered_activations[:, position : position + 1]
                    @ ordered_prototypes[position : position + 1]
                )
                exact_source = _cosine_rows(reconstruction, source_reconstruction)
                exact_coverage = float(
                    (exact_source >= config.fidelity_threshold).float().mean()
                )
                if exact_coverage >= config.target_image_coverage:
                    chosen_count = position + 1
                    chosen_source_fidelity = exact_source.clone()
                    chosen_internal_fidelity = _cosine_rows(
                        reconstruction, balanced_reconstruction
                    )
                    break
            reconstruction = before + (
                ordered_activations[:, start:stop] @ ordered_prototypes[start:stop]
            )
        start = stop

    if chosen_count is None:
        chosen_count = n_groups
        chosen_reconstruction = ordered_activations @ ordered_prototypes
        chosen_source_fidelity = _cosine_rows(
            chosen_reconstruction, source_reconstruction
        )
        chosen_internal_fidelity = _cosine_rows(
            chosen_reconstruction, balanced_reconstruction
        )
    return curve, chosen_count, chosen_source_fidelity, chosen_internal_fidelity


def _top_image_ids(
    sample_ids: Sequence[str], activations: torch.Tensor, count: int
) -> list[str]:
    if not len(sample_ids):
        return []
    count = min(count, len(sample_ids))
    indices = torch.topk(activations, count).indices.tolist()
    return [str(sample_ids[index]) for index in indices]


def _mini_sample_indices(n_samples: int, requested: int) -> torch.Tensor:
    count = min(n_samples, requested)
    if count == n_samples:
        return torch.arange(n_samples)
    return torch.linspace(0, n_samples - 1, steps=count).round().long().unique()


def _rank_spaced_group_ids(
    selected_group_ids: Sequence[int], requested: int
) -> list[int]:
    """Sample the activation ranking from head to tail, including both ends."""

    count = min(len(selected_group_ids), requested)
    if count == len(selected_group_ids):
        return [int(group_id) for group_id in selected_group_ids]
    positions = torch.linspace(0, len(selected_group_ids) - 1, steps=count)
    return [int(selected_group_ids[index]) for index in positions.round().long().unique()]


def _neighbor_triplets(
    geometry: dict,
    centered: torch.Tensor,
    subset_indices: torch.Tensor,
    sample_ids: Sequence[str],
    count: int,
) -> list[dict]:
    raw_top1 = geometry["raw_neighbours"][:, 0]
    projected_top1 = geometry["neighbours"][:, 0]
    changed = torch.where(raw_top1 != projected_top1)[0]
    candidate_rows = changed if len(changed) else torch.arange(len(raw_top1), device=raw_top1.device)
    if not len(candidate_rows):
        return []
    positions = (
        [len(candidate_rows) // 2]
        if count == 1
        else [
            round(index * (len(candidate_rows) - 1) / (count - 1))
            for index in range(count)
        ]
    )
    examples = []
    for position in dict.fromkeys(positions):
        row = int(candidate_rows[position])
        raw_column = int(raw_top1[row])
        projected_column = int(projected_top1[row])
        global_row = int(subset_indices[row])
        global_raw_column = int(subset_indices[raw_column])
        global_projected_column = int(subset_indices[projected_column])
        raw_top1_similarity = float(
            torch.dot(centered[row], centered[raw_column])
        )
        projected_neighbor_raw_similarity = float(
            geometry["projected_similarity"][row, 0] - geometry["gain"][row, 0]
        )
        projected_neighbor_similarity = float(geometry["projected_similarity"][row, 0])
        examples.append(
            {
                "anchor_sample_id": str(sample_ids[global_row]),
                "raw_neighbor_sample_id": str(sample_ids[global_raw_column]),
                "projected_neighbor_sample_id": str(
                    sample_ids[global_projected_column]
                ),
                "raw_neighbor_similarity": raw_top1_similarity,
                "projected_neighbor_raw_similarity": projected_neighbor_raw_similarity,
                "projected_neighbor_similarity": projected_neighbor_similarity,
                "projected_neighbor_gain": (
                    projected_neighbor_similarity - projected_neighbor_raw_similarity
                ),
                "top1_changed": raw_column != projected_column,
            }
        )
    return examples


def run_mini_intervention_audit(
    cache: dict,
    audit_codes: torch.Tensor,
    groups: Sequence[Sequence[int]],
    selected_group_ids: Sequence[int],
    base_config: canonical.CrpAuditConfig,
    config: MiniInterventionConfig,
) -> dict:
    _validate_mini_config(config)
    subset_indices = _mini_sample_indices(len(cache["sample_ids"]), config.sample_count)
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    centered = cache["centered_clip"].index_select(0, subset_indices).to(device)
    subset_codes = audit_codes.index_select(0, subset_indices).to(device)
    dictionary = cache["dictionary"].to(device)
    neighbors = min(config.projected_neighbors, len(subset_indices) - 1)
    if neighbors <= 0:
        raise ValueError("Mini audit requires at least two cached samples.")
    audit_config = replace(
        base_config,
        projected_neighbors=neighbors,
        activation_difference_quantile=config.activation_difference_quantile,
        min_intervention_gain=config.min_intervention_gain,
        min_coverage=config.min_coverage,
        null_trials=config.null_trials,
        null_quantile=config.null_quantile,
        seed=config.seed,
        residual_splice_similarity_threshold=config.residual_splice_similarity_threshold,
    )
    raw_neighbors, _ = canonical.topk_neighbors(
        centered, neighbors, audit_config.similarity_chunk_size
    )
    geometry_input = canonical._AuditGeometry(centered, raw_neighbors, subset_codes)
    generator = torch.Generator(device=device.type).manual_seed(config.seed)
    audited_ids = _rank_spaced_group_ids(selected_group_ids, config.max_groups)
    results = []
    for group_id in audited_ids:
        concept_indices = list(groups[group_id])
        basis = canonical.orthonormal_basis(
            dictionary[concept_indices], audit_config.orthogonal_tolerance
        )
        activation = subset_codes[:, concept_indices].sum(dim=1)
        geometry = canonical._relation_geometry(
            geometry_input, basis, audit_config, concept_indices
        )
        evidence = canonical._score_relations(geometry, activation, audit_config)
        random_geometries = []
        for _ in range(config.null_trials):
            random_basis = canonical.orthonormal_basis(
                torch.randn(
                    basis.shape[1], centered.shape[1], generator=generator, device=device
                ),
                audit_config.orthogonal_tolerance,
            )
            random_geometries.append(
                canonical._relation_geometry(
                    geometry_input, random_basis, audit_config, concept_indices
                )
            )
        random_scores = [
            canonical._score_relations(random_geometry, activation, audit_config)["score"]
            for random_geometry in random_geometries
        ]
        shuffled_scores = []
        for _ in random_geometries:
            permutation = torch.randperm(
                len(activation), generator=generator, device=device
            )
            shuffled_scores.append(
                canonical._score_relations(
                    geometry, activation[permutation], audit_config
                )["score"]
            )
        null_values = torch.tensor(random_scores + shuffled_scores)
        threshold = (
            float(torch.quantile(null_values, config.null_quantile))
            if null_values.numel()
            else math.inf
        )
        passed = evidence["coverage"] >= config.min_coverage and evidence["score"] > threshold
        results.append(
            {
                "group_id": int(group_id),
                "passed_null": bool(passed),
                "score": float(evidence["score"]),
                "null_threshold": _finite_or_none(threshold),
                "null_margin": _finite_or_none(float(evidence["score"]) - threshold),
                "coverage": float(evidence["coverage"]),
                "accepted_edges": int(len(evidence["rows"])),
                "robust_positive_gain": float(evidence["positive_gain"]),
                "semantic_agreement": float(evidence["semantic_agreement"]),
                "activation_gain_alignment": float(evidence["activation_gain_alignment"]),
                "top1_neighbor_turnover": float(evidence["top1_neighbor_turnover"]),
                "mean_neighbor_turnover": float(evidence["mean_neighbor_turnover"]),
                "mean_jaccard_at_k": float(evidence["mean_jaccard_at_k"]),
                "neighbor_triplets": _neighbor_triplets(
                    geometry,
                    centered,
                    subset_indices,
                    cache["sample_ids"],
                    config.example_edges_per_group,
                ),
            }
        )
    pass_fraction = (
        sum(bool(item["passed_null"]) for item in results) / len(results)
        if results
        else 0.0
    )
    median_top1 = median(
        [float(item["top1_neighbor_turnover"]) for item in results]
    ) if results else 0.0
    median_jaccard = median(
        [float(item["mean_jaccard_at_k"]) for item in results]
    ) if results else 1.0
    geometry_changed = median_top1 >= 0.10 or median_jaccard <= 0.90
    passed = pass_fraction >= 0.5 and geometry_changed
    return {
        "config": asdict(config),
        "device": str(device),
        "sample_count": int(len(subset_indices)),
        "requested_group_count": len(selected_group_ids),
        "audited_group_count": len(results),
        "sampling_strategy": "rank_spaced_including_head_and_tail",
        "truncated": len(selected_group_ids) > len(results),
        "null_pass_fraction": pass_fraction,
        "median_top1_neighbor_turnover": median_top1,
        "median_jaccard_at_k": median_jaccard,
        "geometry_changed": geometry_changed,
        "passed": passed,
        "groups": results,
    }


def build_group_screen(
    raw_cache: dict,
    group_config: canonical.CrpAuditConfig,
    screen_config: ReconstructionScreenConfig,
    mini_config: MiniInterventionConfig,
    spatial_artifact: dict | None = None,
) -> dict:
    _validate_screen_config(screen_config)
    canonical._validate_config(group_config)
    cache = canonical.validate_feature_cache(raw_cache)
    original_codes = cache["splice_codes"]
    audit_codes = original_codes
    spatial_summary = None
    if group_config.spatial_balance:
        if spatial_artifact is None:
            raise ValueError("spatial_balance requires an aligned spatial artifact.")
        artifact_variant = str(spatial_artifact.get("variant", ""))
        if artifact_variant != group_config.spatial_balance_variant:
            raise ValueError(
                "Spatial balance variant does not match the group-screen configuration: "
                f"{artifact_variant!r} != {group_config.spatial_balance_variant!r}."
            )
        audit_codes, spatial_summary = spatially_balanced_codes(
            original_codes,
            spatial_artifact,
            floor=group_config.spatial_balance_floor,
            frequency_power=group_config.spatial_frequency_power,
        )
    groups = canonical.group_concepts(
        audit_codes,
        cache["dictionary"],
        cache["vocabulary"],
        group_config,
    )
    activations, prototypes, summaries = _group_representations(
        audit_codes, cache["dictionary"], cache["vocabulary"], groups
    )
    order = _rank_groups(activations, summaries)
    source_reconstruction = original_codes @ cache["dictionary"]
    balanced_reconstruction = audit_codes @ cache["dictionary"]
    curve, selected_count, source_fidelity, internal_fidelity = _coverage_curve(
        activations,
        prototypes,
        order,
        source_reconstruction,
        balanced_reconstruction,
        screen_config,
    )
    selected_group_ids = list(order[:selected_count])
    selected_set = set(selected_group_ids)
    selected_concepts = sum(len(groups[group_id]) for group_id in selected_group_ids)
    source_coverage = float(
        (source_fidelity >= screen_config.fidelity_threshold).float().mean()
    ) if source_fidelity.numel() else 0.0
    reconstruction_passed = source_coverage >= screen_config.target_image_coverage

    for rank, group_id in enumerate(order, start=1):
        summary = summaries[group_id]
        summary["activation_rank"] = rank
        summary["selected_for_reconstruction"] = group_id in selected_set
        summary["coherence_warning"] = (
            len(groups[group_id]) > 1
            and float(summary["mean_text_similarity"])
            < screen_config.coherence_warning_threshold
        )
        summary["top_sample_ids"] = _top_image_ids(
            cache["sample_ids"],
            activations[:, group_id],
            screen_config.top_images_per_group,
        )

    image_rows = []
    if selected_group_ids:
        selected_activations = activations[:, selected_group_ids]
        top_count = min(screen_config.top_groups_per_image, selected_activations.shape[1])
        top_values, top_positions = torch.topk(selected_activations, top_count, dim=1)
        for index, sample_id in enumerate(cache["sample_ids"]):
            assignments = []
            for value, position in zip(top_values[index].tolist(), top_positions[index].tolist()):
                if value <= 0:
                    continue
                group_id = selected_group_ids[position]
                assignments.append(
                    {
                        "group_id": int(group_id),
                        "name": str(summaries[group_id]["name"]),
                        "activation": float(value),
                    }
                )
            image_rows.append(
                {
                    "sample_id": str(sample_id),
                    "source_fidelity": float(source_fidelity[index]),
                    "internal_fidelity": float(internal_fidelity[index]),
                    "top_groups": assignments,
                }
            )
    else:
        for index, sample_id in enumerate(cache["sample_ids"]):
            image_rows.append(
                {
                    "sample_id": str(sample_id),
                    "source_fidelity": float(source_fidelity[index]),
                    "internal_fidelity": float(internal_fidelity[index]),
                    "top_groups": [],
                }
            )

    report = {
        "artifact": GROUP_SCREEN_ARTIFACT,
        "sample_count": len(cache["sample_ids"]),
        "sample_ids": [str(value) for value in cache["sample_ids"]],
        "provenance": dict(cache.get("provenance", {})),
        "group_config": asdict(group_config),
        "screen_config": asdict(screen_config),
        "spatial_balance": spatial_summary,
        "metrics": {
            "candidate_group_count": len(groups),
            "selected_group_count": selected_count,
            "selected_concept_count": selected_concepts,
            "compression_ratio": (
                selected_concepts / selected_count if selected_count else 0.0
            ),
            "source_coverage": source_coverage,
            "median_source_fidelity": float(source_fidelity.median()) if source_fidelity.numel() else 0.0,
            "p01_source_fidelity": _quantile(source_fidelity, 0.01),
            "median_internal_fidelity": float(internal_fidelity.median()) if internal_fidelity.numel() else 0.0,
            "reconstruction_passed": reconstruction_passed,
            "coherence_warning_count": sum(
                bool(item["coherence_warning"]) for item in summaries
                if item["selected_for_reconstruction"]
            ),
        },
        "coverage_curve": curve,
        "selected_group_ids": selected_group_ids,
        "groups": summaries,
        "images": image_rows,
    }
    if mini_config.enabled and reconstruction_passed and selected_group_ids:
        report["mini_intervention"] = run_mini_intervention_audit(
            cache,
            audit_codes,
            groups,
            selected_group_ids,
            group_config,
            mini_config,
        )
    else:
        report["mini_intervention"] = None

    mini = report["mini_intervention"]
    if not reconstruction_passed:
        decision = "FAIL_RECONSTRUCTION"
        reason = "The selected group prototypes do not reach the requested image coverage."
    elif mini is None:
        decision = "REVIEW_GROUPS"
        reason = "Reconstruction passed; inspect group coherence before running the mini audit."
    elif not mini["passed"]:
        decision = "FAIL_INTERVENTION"
        reason = (
            "Reconstruction passed, but the rank-spaced projection audit is too weak "
            f"({mini['audited_group_count']}/{mini['requested_group_count']} selected groups checked)."
        )
    else:
        decision = "PROVISIONAL_GO"
        reason = (
            "Reconstruction and the rank-spaced intervention sample passed "
            f"({mini['audited_group_count']}/{mini['requested_group_count']} selected groups checked); "
            "inspect the report before a full audit."
        )
    report["decision"] = {"status": decision, "reason": reason}
    return report


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("waterbirds", "celeba"), required=True)
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--spatial-balance-artifact", default="")
    parser.add_argument("--spatial-variant", default="")
    parser.add_argument("--spatial-floor", type=float, default=0.25)
    parser.add_argument("--spatial-frequency-power", type=float, default=0.0)
    parser.add_argument("--min-concept-frequency", type=float, default=0.01)
    parser.add_argument("--max-concept-frequency", type=float, default=0.95)
    parser.add_argument("--text-similarity-threshold", type=float, default=0.70)
    parser.add_argument("--coactivation-threshold", type=float, default=0.0)
    parser.add_argument("--fidelity-threshold", type=float, default=0.95)
    parser.add_argument("--target-image-coverage", type=float, default=0.99)
    parser.add_argument("--coherence-warning-threshold", type=float, default=0.55)
    parser.add_argument("--mini-audit", type=_parse_bool, default=True)
    parser.add_argument("--mini-samples", type=int, default=1024)
    parser.add_argument("--mini-max-groups", type=int, default=24)
    parser.add_argument("--mini-neighbors", type=int, default=10)
    parser.add_argument("--mini-null-trials", type=int, default=4)
    parser.add_argument("--mini-null-quantile", type=float, default=0.95)
    parser.add_argument("--mini-min-coverage", type=float, default=0.01)
    parser.add_argument("--mini-min-intervention-gain", type=float, default=5e-4)
    parser.add_argument("--mini-activation-difference-quantile", type=float, default=0.85)
    parser.add_argument("--mini-residual-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-max-groups", type=int, default=24)
    parser.add_argument("--report-images-per-band", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raw_cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    cache = canonical.validate_feature_cache(raw_cache)
    cache_dataset = str(cache.get("provenance", {}).get("dataset", "")).lower()
    if cache_dataset and cache_dataset != args.dataset:
        raise ValueError(
            f"CRP cache belongs to {cache_dataset!r}, not {args.dataset!r}."
        )
    spatial_artifact = None
    spatial_enabled = bool(args.spatial_balance_artifact)
    if spatial_enabled:
        if not args.spatial_variant:
            raise ValueError("--spatial-variant is required with a spatial artifact.")
        spatial_artifact = load_spatial_balance_artifact(
            args.spatial_balance_artifact,
            args.dataset,
            cache["sample_ids"],
            cache["vocabulary"],
            cache.get("provenance"),
        )
    group_config = canonical.CrpAuditConfig(
        min_concept_frequency=args.min_concept_frequency,
        max_concept_frequency=args.max_concept_frequency,
        text_similarity_threshold=args.text_similarity_threshold,
        coactivation_threshold=args.coactivation_threshold,
        min_group_size=1,
        max_selected_groups=0,
        spatial_balance=spatial_enabled,
        spatial_balance_variant=args.spatial_variant if spatial_enabled else "",
        spatial_balance_floor=args.spatial_floor,
        spatial_frequency_power=args.spatial_frequency_power,
        seed=args.seed,
    )
    screen_config = ReconstructionScreenConfig(
        fidelity_threshold=args.fidelity_threshold,
        target_image_coverage=args.target_image_coverage,
        coherence_warning_threshold=args.coherence_warning_threshold,
    )
    mini_config = MiniInterventionConfig(
        enabled=args.mini_audit,
        sample_count=args.mini_samples,
        max_groups=args.mini_max_groups,
        projected_neighbors=args.mini_neighbors,
        null_trials=args.mini_null_trials,
        null_quantile=args.mini_null_quantile,
        activation_difference_quantile=args.mini_activation_difference_quantile,
        min_intervention_gain=args.mini_min_intervention_gain,
        min_coverage=args.mini_min_coverage,
        residual_splice_similarity_threshold=args.mini_residual_threshold,
        seed=args.seed,
    )
    report = build_group_screen(
        raw_cache,
        group_config,
        screen_config,
        mini_config,
        spatial_artifact,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    from scripts.tools.render_crp_group_screen import render_group_screen

    render_group_screen(
        report,
        Path(args.data_folder),
        Path(args.output_html),
        max_groups=args.report_max_groups,
        images_per_band=args.report_images_per_band,
    )
    print(f"[INFO] Group screen JSON: {output_json}")
    print(f"[INFO] Group screen HTML: {args.output_html}")
    print(f"[INFO] Decision: {report['decision']['status']} — {report['decision']['reason']}")


if __name__ == "__main__":
    main()
