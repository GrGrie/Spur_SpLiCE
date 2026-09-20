"""The frozen SpLiCE dataset cache: what it must contain and what it must not.

The cache holds the CLIP embeddings and sparse concept codes of one train split. Every later stage
reads it and none of them reads a label, so the required and forbidden key sets are checked here
once. Sample IDs are ``<dataset>:<metadata row>``, which is how groups, graphs and post-hoc labels
line up with the dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

SPLICE_DATASET_CACHE_VERSION = 1
REQUIRED_CACHE_KEYS = {
    "cache_version",
    "sample_ids",
    "clip_embeddings",
    "image_mean",
    "splice_codes",
    "dictionary",
    "vocabulary",
}
# A label of any kind in the cache would make the pipeline supervised without anyone noticing.
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


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_splice_dataset_cache(splice_dataset_cache: dict, path: str | Path) -> None:
    """Validate and atomically save a frozen SpLiCE dataset cache."""

    validate_splice_dataset_cache(splice_dataset_cache)
    _atomic_torch_save(splice_dataset_cache, Path(path))
