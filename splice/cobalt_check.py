"""Label-free CoBalT concept balancing for SpLiCE group construction."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch


def load_cobalt_train_concepts(
    path: str | Path,
    dataset: str,
    cache_sample_ids: Sequence[str],
    include_confidence: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Load and align fixed CoBalT Stage-1 concepts without reading labels."""

    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("artifact") != "cobalt_concepts_v1":
        raise ValueError("Expected a cobalt_concepts_v1 artifact.")
    if str(artifact.get("dataset", "")).lower() != dataset.lower():
        raise ValueError("CoBalT concepts belong to a different dataset.")
    train = artifact.get("splits", {}).get("train")
    if not isinstance(train, dict):
        raise ValueError("CoBalT concept artifact is missing the train split.")
    source_ids = torch.as_tensor(train.get("sample_ids"), dtype=torch.long).view(-1)
    concepts = torch.as_tensor(train.get("concepts"), dtype=torch.long)
    if concepts.ndim != 2 or concepts.shape[0] != source_ids.numel():
        raise ValueError("CoBalT train concepts must be shaped [samples, slots].")
    if source_ids.numel() != torch.unique(source_ids).numel():
        raise ValueError("CoBalT train sample IDs must be unique.")
    if torch.any(concepts < -1):
        raise ValueError("CoBalT concepts may use only -1 for inactive slots.")

    positions = {int(source_id): position for position, source_id in enumerate(source_ids.tolist())}
    aligned_rows = []
    for sample_id in cache_sample_ids:
        prefix, separator, source_index = str(sample_id).rpartition(":")
        if not separator or prefix.lower() != dataset.lower():
            raise ValueError(f"Unexpected cache sample ID for {dataset}: {sample_id!r}.")
        index = int(source_index)
        if index not in positions:
            raise ValueError(f"CoBalT concepts do not contain cache sample {sample_id!r}.")
        aligned_rows.append(positions[index])
    aligned = concepts[torch.tensor(aligned_rows, dtype=torch.long)]
    aligned_confidence = None
    confidence_record = train.get("confidence")
    if confidence_record is not None:
        confidence = torch.as_tensor(confidence_record, dtype=torch.float32).view(-1)
        if confidence.shape != (source_ids.numel(),):
            raise ValueError("CoBalT confidence must contain one value per train sample.")
        if not torch.isfinite(confidence).all() or torch.any((confidence < 0) | (confidence > 1)):
            raise ValueError("CoBalT confidence must be finite and lie in [0, 1].")
        aligned_confidence = confidence[torch.tensor(aligned_rows, dtype=torch.long)]
    active = aligned[aligned >= 0]
    if active.numel() == 0:
        raise ValueError("CoBalT train concepts contain no active assignments.")
    provenance = {
        "artifact": "cobalt_concepts_v1",
        "dataset": dataset.lower(),
        "seed": int(artifact.get("seed", 0)),
        "model_config": dict(artifact.get("model_config", {})),
        "sample_count": int(aligned.shape[0]),
        "slot_count": int(aligned.shape[1]),
        "active_concept_count": int(torch.unique(active).numel()),
        "confidence_available": aligned_confidence is not None,
    }
    if include_confidence:
        provenance["confidence"] = aligned_confidence
    return aligned, provenance


def concept_balanced_sample_weights(
    concepts: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Approximate uniform-concept sampling using only CoBalT memberships.

    CoBalT Stage 2 first samples a discovered concept uniformly.  Without target
    labels, the label-free part of that marginal gives sample weight
    ``sum_c 1[i contains c] / count(c)``.  Mean normalization keeps the scale of
    the existing coactivation cosine unchanged.
    """

    concepts = torch.as_tensor(concepts, dtype=torch.long).cpu()
    if concepts.ndim != 2:
        raise ValueError("CoBalT concepts must be shaped [samples, slots].")
    active_ids = torch.unique(concepts[concepts >= 0], sorted=True)
    if active_ids.numel() == 0:
        raise ValueError("CoBalT concepts contain no active assignments.")
    membership = torch.stack(
        [concepts.eq(int(concept_id)).any(dim=1) for concept_id in active_ids.tolist()],
        dim=1,
    ).float()
    if confidence is None:
        confidence_tensor = torch.ones(concepts.shape[0], dtype=torch.float32)
    else:
        confidence_tensor = torch.as_tensor(confidence, dtype=torch.float32).cpu().view(-1)
        if confidence_tensor.shape != (concepts.shape[0],):
            raise ValueError("CoBalT confidence must contain one value per sample.")
        if not torch.isfinite(confidence_tensor).all() or torch.any(
            (confidence_tensor < 0) | (confidence_tensor > 1)
        ):
            raise ValueError("CoBalT confidence must be finite and lie in [0, 1].")
    raw_counts = membership.sum(dim=0)
    counts = (membership * confidence_tensor.unsqueeze(1)).sum(dim=0)
    if torch.any(counts <= 0):
        raise ValueError("CoBalT concept membership contains an empty active concept.")
    weights = confidence_tensor * (membership / counts.unsqueeze(0)).sum(dim=1)
    if float(weights.sum()) <= 0:
        raise ValueError("CoBalT balancing produced zero total sample weight.")
    weights = weights / weights.mean()
    return weights, {
        "active_concept_ids": [int(value) for value in active_ids.tolist()],
        "concept_sample_counts": [int(value) for value in raw_counts.tolist()],
        "concept_sample_masses": [float(value) for value in counts.tolist()],
        "zero_weight_samples": int(weights.eq(0).sum()),
        "min_sample_weight": float(weights.min()),
        "max_sample_weight": float(weights.max()),
        "mean_sample_weight": float(weights.mean()),
        "confidence_enabled": confidence is not None,
        "confidence_min": float(confidence_tensor.min()),
        "confidence_max": float(confidence_tensor.max()),
        "confidence_mean": float(confidence_tensor.mean()),
        "zero_confidence_samples": int(confidence_tensor.eq(0).sum()),
    }
