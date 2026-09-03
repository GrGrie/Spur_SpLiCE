"""Validate and apply CRPv4 image-specific spatial concept balancing."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch


SPATIAL_BALANCE_ARTIFACT = "splice_spatial_balance_v1"
SPATIAL_VARIANTS = {
    "vanilla_patchwise",
    "vanilla_slots",
    "sclip_patchwise",
    "sclip_slots",
}
FORBIDDEN_ANNOTATION_KEYS = {
    "a",
    "attribute",
    "attributes",
    "group",
    "group_ids",
    "label",
    "labels",
    "metadata",
    "spurious",
    "target",
    "targets",
    "y",
}


def _collect_keys(value) -> set[str]:
    if not isinstance(value, dict):
        return set()
    keys = {str(key).lower() for key in value}
    for nested in value.values():
        keys.update(_collect_keys(nested))
    return keys


def validate_spatial_balance_artifact(
    artifact: dict,
    dataset: str,
    sample_ids: Sequence[str],
    vocabulary: Sequence[str],
    cache_provenance: dict | None = None,
) -> dict:
    """Validate exact cache alignment without accepting downstream annotations."""

    if not isinstance(artifact, dict) or artifact.get("artifact") != SPATIAL_BALANCE_ARTIFACT:
        raise ValueError(f"Expected a {SPATIAL_BALANCE_ARTIFACT} artifact.")
    forbidden = FORBIDDEN_ANNOTATION_KEYS.intersection(_collect_keys(artifact))
    if forbidden:
        raise ValueError(
            f"Spatial balance artifact contains forbidden annotation keys: {sorted(forbidden)}"
        )
    if str(artifact.get("dataset", "")).lower() != dataset.lower():
        raise ValueError("Spatial balance artifact belongs to a different dataset.")
    if artifact.get("variant") not in SPATIAL_VARIANTS:
        raise ValueError(f"Unknown CRPv4 spatial variant {artifact.get('variant')!r}.")
    source_ids = [str(value) for value in artifact.get("sample_ids", [])]
    expected_ids = [str(value) for value in sample_ids]
    if source_ids != expected_ids:
        raise ValueError("Spatial balance sample IDs do not exactly match the CRP cache.")
    source_vocabulary = [str(value) for value in artifact.get("vocabulary", [])]
    expected_vocabulary = [str(value) for value in vocabulary]
    if source_vocabulary != expected_vocabulary:
        raise ValueError("Spatial balance vocabulary does not exactly match the CRP cache.")
    if cache_provenance is not None:
        source_provenance = artifact.get("cache_provenance")
        if not isinstance(source_provenance, dict):
            raise ValueError("Spatial balance artifact is missing CRP cache provenance.")
        identity_keys = (
            "dataset",
            "split",
            "splice_model",
            "splice_pretrained",
            "splice_vocab",
            "splice_vocab_size",
            "splice_l1_penalty",
        )
        mismatched = [
            key
            for key in identity_keys
            if source_provenance.get(key) != cache_provenance.get(key)
        ]
        if mismatched:
            raise ValueError(
                f"Spatial balance artifact does not match CRP cache provenance: {mismatched}."
            )

    indices = torch.as_tensor(artifact.get("concept_indices"), dtype=torch.long).cpu()
    evidence = torch.as_tensor(artifact.get("evidence"), dtype=torch.float32).cpu()
    confidence = torch.as_tensor(artifact.get("confidence"), dtype=torch.float32).view(-1).cpu()
    if indices.ndim != 2 or evidence.shape != indices.shape:
        raise ValueError("Spatial concept indices and evidence must be aligned rank-2 tensors.")
    if indices.shape[0] != len(expected_ids) or confidence.shape != (len(expected_ids),):
        raise ValueError("Spatial balance tensors must contain one row per cached sample.")
    if torch.any(indices < 0) or torch.any(indices >= len(expected_vocabulary)):
        raise ValueError("Spatial concept index is outside the SpLiCE vocabulary.")
    sorted_indices = indices.sort(dim=1).values
    if indices.shape[1] > 1 and torch.any(sorted_indices[:, 1:] == sorted_indices[:, :-1]):
        raise ValueError("Spatial concept indices must be unique within each image.")
    if not torch.isfinite(evidence).all() or torch.any(evidence < 0):
        raise ValueError("Spatial evidence must be finite and non-negative.")
    if not torch.isfinite(confidence).all() or torch.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Spatial confidence must be finite and lie in [0, 1].")
    return {
        **artifact,
        "sample_ids": source_ids,
        "vocabulary": source_vocabulary,
        "concept_indices": indices,
        "evidence": evidence,
        "confidence": confidence,
    }


def load_spatial_balance_artifact(
    path: str | Path,
    dataset: str,
    sample_ids: Sequence[str],
    vocabulary: Sequence[str],
    cache_provenance: dict | None = None,
) -> dict:
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    return validate_spatial_balance_artifact(
        artifact, dataset, sample_ids, vocabulary, cache_provenance
    )


def spatially_balanced_codes(
    splice_codes: torch.Tensor,
    spatial_artifact: dict,
    floor: float = 0.25,
    frequency_power: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Return ``w * B`` while preserving every original row's total SpLiCE mass.

    Unreported concepts receive ``floor``.  Reported evidence is scaled within
    each image, and spatial confidence interpolates the factor back toward one.
    The original tensor is never modified.
    """

    if not 0 <= floor <= 1:
        raise ValueError("spatial balance floor must lie in [0, 1].")
    if frequency_power < 0:
        raise ValueError("spatial frequency power must be non-negative.")
    codes = torch.as_tensor(splice_codes, dtype=torch.float32).cpu()
    indices = torch.as_tensor(spatial_artifact["concept_indices"], dtype=torch.long).cpu()
    evidence = torch.as_tensor(spatial_artifact["evidence"], dtype=torch.float32).cpu()
    confidence = torch.as_tensor(spatial_artifact["confidence"], dtype=torch.float32).view(-1).cpu()
    if codes.ndim != 2 or indices.shape[0] != codes.shape[0]:
        raise ValueError("Spatial evidence and SpLiCE codes must share the sample dimension.")

    adjusted = evidence.clone()
    concept_mass = torch.zeros(codes.shape[1], dtype=torch.float32)
    concept_mass.scatter_add_(0, indices.flatten(), evidence.flatten())
    if frequency_power:
        positive_mass = concept_mass[concept_mass > 0]
        reference_mass = positive_mass.median() if positive_mass.numel() else torch.tensor(1.0)
        correction = (reference_mass / concept_mass[indices].clamp_min(1e-12)).pow(frequency_power)
        adjusted.mul_(correction)
    row_max = adjusted.max(dim=1, keepdim=True).values
    normalized = adjusted / row_max.clamp_min(1e-12)

    factors = torch.full_like(codes, floor)
    factors.scatter_(1, indices, floor + (1.0 - floor) * normalized)
    factors = 1.0 + confidence.unsqueeze(1) * (factors - 1.0)
    balanced = codes * factors
    original_mass = codes.sum(dim=1, keepdim=True)
    balanced_mass = balanced.sum(dim=1, keepdim=True)
    rescalable = (original_mass > 0) & (balanced_mass > 0)
    scale = torch.ones_like(original_mass)
    scale[rescalable] = original_mass[rescalable] / balanced_mass[rescalable]
    balanced = balanced * scale

    changed = (balanced - codes).abs().sum(dim=1) > 1e-8
    return balanced, {
        "artifact": SPATIAL_BALANCE_ARTIFACT,
        "variant": str(spatial_artifact.get("variant", "unknown")),
        "sample_count": int(codes.shape[0]),
        "concept_count": int(codes.shape[1]),
        "concepts_per_image": int(indices.shape[1]),
        "floor": float(floor),
        "frequency_power": float(frequency_power),
        "confidence_min": float(confidence.min()),
        "confidence_max": float(confidence.max()),
        "confidence_mean": float(confidence.mean()),
        "changed_sample_count": int(changed.sum()),
        "original_mass_preserved": bool(
            torch.allclose(balanced.sum(dim=1), codes.sum(dim=1), atol=1e-5, rtol=1e-5)
        ),
        "source_config": dict(spatial_artifact.get("config", {})),
    }
