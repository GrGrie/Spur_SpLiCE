"""Label-free, image-specific frozen-teacher transfer for the next-test screen.

This module deliberately has no dependency on dataset annotations.  The only
alignment key is the explicit sample id stored in the frozen feature cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from splice.crp import validate_feature_cache


TARGET_ARTIFACT = "splice_concept_transfer_targets_v1"


def _normalise_nonzero(values: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(values).detach().float().cpu()
    if values.ndim != 2:
        raise ValueError(f"{name} must be rank-2.")
    finite = torch.isfinite(values).all(dim=1)
    norms = values.norm(dim=1)
    valid = finite & torch.isfinite(norms) & (norms > 1e-12)
    normalised = torch.zeros_like(values)
    if valid.any():
        normalised[valid] = F.normalize(values[valid], dim=1)
    return normalised, valid


def _derangement(indices: torch.Tensor, seed: int) -> torch.Tensor:
    if indices.numel() < 2:
        raise ValueError("The shuffled reconstruction control needs at least two valid sample ids.")
    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(1000):
        permutation = indices[torch.randperm(indices.numel(), generator=generator)]
        if torch.all(permutation != indices):
            return permutation
    raise RuntimeError("Could not construct a derangement for the valid sample ids.")


def prepare_target_artifact(
    cache_path: str | Path,
    output_path: str | Path,
    *,
    shuffle_seed: int = 101,
) -> Path:
    """Prepare raw/reconstruction/shuffled targets without reading labels."""

    cache_path = Path(cache_path)
    cache = validate_feature_cache(torch.load(cache_path, map_location="cpu", weights_only=False))
    raw, raw_valid = _normalise_nonzero(cache["centered_clip"], "centered_clip")
    reconstruction = cache["splice_codes"].float() @ cache["dictionary"].float()
    reconstruction, reconstruction_valid = _normalise_nonzero(reconstruction, "reconstruction")
    valid = raw_valid & reconstruction_valid
    if not valid.any():
        raise ValueError("No finite non-zero common target rows remain after validation.")
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    permutation = torch.arange(len(cache["sample_ids"]), dtype=torch.long)
    shuffled_source = _derangement(valid_indices, shuffle_seed)
    permutation[valid_indices] = shuffled_source

    payload = {
        "artifact": TARGET_ARTIFACT,
        "version": 1,
        "sample_ids": [str(value) for value in cache["sample_ids"]],
        "raw": raw,
        "reconstruction": reconstruction,
        "shuffled_reconstruction": reconstruction.index_select(0, permutation),
        "valid_mask": valid,
        "permutation": permutation,
        "shuffle_seed": int(shuffle_seed),
        "cache_path": str(cache_path),
        "cache_fingerprint": hashlib.md5(cache_path.read_bytes()).hexdigest(),
        "target_dim": int(raw.shape[1]),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    metadata = {
        "artifact": TARGET_ARTIFACT,
        "version": 1,
        "sample_count": len(payload["sample_ids"]),
        "valid_count": int(valid.sum()),
        "target_dim": payload["target_dim"],
        "sample_ids": payload["sample_ids"],
        "permutation": permutation.tolist(),
        "cache_fingerprint": payload["cache_fingerprint"],
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def load_target_artifact(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("artifact") != TARGET_ARTIFACT or payload.get("version") != 1:
        raise ValueError("Unsupported concept-transfer target artifact.")
    sample_ids = [str(value) for value in payload.get("sample_ids", [])]
    tensors = {name: payload.get(name) for name in ("raw", "reconstruction", "shuffled_reconstruction")}
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Target artifact sample_ids must be unique and non-empty.")
    shape = None
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"Target artifact field {name} must be rank-2.")
        shape = shape or tensor.shape
        if tensor.shape != shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Target artifact field {name} is malformed.")
    valid = torch.as_tensor(payload.get("valid_mask"), dtype=torch.bool).view(-1)
    permutation = torch.as_tensor(payload.get("permutation"), dtype=torch.long).view(-1)
    if valid.numel() != len(sample_ids) or permutation.numel() != len(sample_ids):
        raise ValueError("Target artifact alignment fields have the wrong length.")
    if sorted(permutation.tolist()) != list(range(len(sample_ids))):
        raise ValueError("Target artifact permutation is not a bijection.")
    return {**payload, "sample_ids": sample_ids, "valid_mask": valid, "permutation": permutation}


class FrozenConceptTransferSubset(Dataset):
    """Two-crop images plus target-bank row ids; never returns y or metadata."""

    def __init__(self, dataset, transform, targets: dict, dataset_name: str = "waterbirds") -> None:
        self.source_dataset = dataset
        self.transform = transform
        self.source_indices = getattr(dataset, "indices", None)
        if self.source_indices is None or not hasattr(dataset, "dataset"):
            raise ValueError("Frozen transfer needs a WILDS indexed subset.")
        self.targets = targets
        row_by_id = {sample_id: row for row, sample_id in enumerate(targets["sample_ids"])}
        self.row_indices = []
        for source_index in self.source_indices:
            sample_id = f"{dataset_name}:{int(source_index)}"
            if sample_id not in row_by_id:
                raise ValueError(f"Missing frozen target row for sample id {sample_id}.")
            self.row_indices.append(row_by_id[sample_id])

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int):
        source_index = int(self.source_indices[index])
        image = self.source_dataset.dataset.get_input(source_index)
        return self.transform(image), int(self.row_indices[index])

    @property
    def collate(self):
        return None


class ConceptDistillationRegularizer:
    """Scheduled cosine loss against an immutable target bank."""

    enabled = True
    requires_concept_transfer = True
    requires_crp_indices = False
    requires_clip_distillation = True

    def __init__(self, targets: dict, target_kind: str, weight: float, start_epoch: int, warmup_epochs: int) -> None:
        if target_kind not in {"raw", "reconstruction", "shuffled_reconstruction"}:
            raise ValueError(f"Unknown concept-transfer target kind: {target_kind}")
        if weight < 0 or start_epoch < 0 or warmup_epochs < 0:
            raise ValueError("Concept-transfer schedule values must be non-negative.")
        self.targets = targets[target_kind].detach().float().cpu()
        self.valid_mask = targets["valid_mask"].detach().bool().cpu()
        self.target_kind = target_kind
        self.weight = float(weight)
        self.start_epoch = int(start_epoch)
        self.warmup_epochs = int(warmup_epochs)
        self.epoch = 0
        self.last_diagnostics: dict[str, float] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def scheduled_weight(self) -> float:
        if self.epoch <= self.start_epoch:
            return 0.0
        if self.warmup_epochs == 0:
            return self.weight
        return self.weight * min(1.0, max(0.0, (self.epoch - self.start_epoch) / self.warmup_epochs))

    def targets_for_indices(self, indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.as_tensor(indices, dtype=torch.long).detach().cpu().view(-1)
        return self.targets.index_select(0, rows).to(device), self.valid_mask.index_select(0, rows).to(device)

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid = valid.bool()
        if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[0] % 2:
            raise ValueError("Concept distillation expects aligned two-view predictions and targets.")
        valid = torch.cat([valid, valid], dim=0)
        if self.scheduled_weight <= 0 or not valid.any():
            self.last_diagnostics = {"scheduled_weight": self.scheduled_weight, "valid_fraction": float(valid.float().mean())}
            return predictions.sum() * 0.0
        cosine = (F.normalize(predictions.float(), dim=1) * F.normalize(targets.float(), dim=1)).sum(dim=1)
        loss = (1.0 - cosine[valid]).mean()
        self.last_diagnostics = {
            "scheduled_weight": self.scheduled_weight,
            "valid_fraction": float(valid.float().mean()),
            "cosine_loss": float(loss.detach()),
        }
        return (self.scheduled_weight * loss).to(dtype=predictions.dtype)
