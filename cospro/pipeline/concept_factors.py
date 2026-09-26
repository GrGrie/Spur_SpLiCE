"""Concept factors: the dataset-level structure of the frozen SpLiCE codes, read without labels.

A factor is one concept group of the grouping stage; its activation on an image is the summed
SpLiCE code of the group's concepts. Two properties of the factors drive the concept-factor methods:

* Entangled pairs. Two factors whose presence is strongly correlated across the training images,
  such as a class concept and the context it usually appears in, form a pair. Neither side is
  labelled spurious: the student is asked to keep both factors apart and the balanced linear probe
  chooses between them.
* Decorrelated targets. ZCA whitening of the factor activations turns each factor into its part
  that the other factors leave unexplained. On the images that break a correlation, those targets
  are large; on typical images, small.

Nothing here reads a label, a group annotation or a property of the evaluation splits.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cospro.tracking.artifacts import scratch_root, shared

CONCEPT_FACTORS_ARTIFACT = "cospro_concept_factors_v1"
TARGET_KINDS = ("whitened", "standardized")


@dataclass(frozen=True)
class FactorConfig:
    """How factors, entangled pairs and targets are chosen from the codes."""

    min_frequency: float = 0.02
    max_frequency: float = 0.9
    max_count: int = 64
    condition_pairs: int = 8
    min_correlation: float = 0.2
    max_text_similarity: float = 0.75
    whitening_eps: float = 0.1

    def __post_init__(self) -> None:
        if not 0 <= self.min_frequency < self.max_frequency <= 1:
            raise ValueError("Factor frequencies need 0 <= min < max <= 1.")
        if self.max_count < 2 or self.condition_pairs < 1:
            raise ValueError("Concept factors need at least two factors and one pair.")
        if not -1 <= self.min_correlation <= 1 or not -1 <= self.max_text_similarity <= 1:
            raise ValueError("Correlation and text-similarity bounds lie in [-1, 1].")
        if self.whitening_eps <= 0:
            raise ValueError("The whitening ridge must be positive.")


def group_activations(codes: torch.Tensor, groups: list[dict]) -> torch.Tensor:
    """Summed SpLiCE code of each group's concepts: one column per group."""

    columns = [codes[:, list(group["concept_indices"])].sum(dim=1) for group in groups]
    return torch.stack(columns, dim=1).float()


def group_directions(dictionary: torch.Tensor, groups: list[dict]) -> torch.Tensor:
    """Unit text direction of each group: the normalized mean of its concept directions."""

    rows = [F.normalize(dictionary[list(group["concept_indices"])].float(), dim=1).mean(dim=0) for group in groups]
    return F.normalize(torch.stack(rows), dim=1)


def select_factors(active: torch.Tensor, config: FactorConfig) -> list[int]:
    """Columns within the frequency band, the most balanced first, at most ``max_count``."""

    frequency = active.float().mean(dim=0)
    band = (frequency >= config.min_frequency) & (frequency <= config.max_frequency)
    candidates = torch.nonzero(band).view(-1).tolist()
    balance = (frequency * (1 - frequency)).tolist()
    candidates.sort(key=lambda column: (-balance[column], column))
    return sorted(candidates[: config.max_count])


def presence_correlation(active: torch.Tensor) -> torch.Tensor:
    """Phi coefficient between the presence indicators of every pair of columns."""

    values = active.float()
    centered = values - values.mean(dim=0)
    covariance = centered.T @ centered / values.shape[0]
    scale = covariance.diagonal().clamp_min(1e-12).sqrt()
    return covariance / scale[:, None] / scale[None, :]


def entangled_pairs(active: torch.Tensor, directions: torch.Tensor, config: FactorConfig) -> list[dict]:
    """The most correlated factor pairs whose concepts name different things."""

    correlation = presence_correlation(active)
    similarity = directions @ directions.T
    pairs = []
    for first in range(correlation.shape[0]):
        for second in range(first + 1, correlation.shape[0]):
            phi = float(correlation[first, second])
            text_similarity = float(similarity[first, second])
            if phi >= config.min_correlation and text_similarity <= config.max_text_similarity:
                pairs.append({"factors": [first, second], "phi": phi, "text_similarity": text_similarity})
    pairs.sort(key=lambda pair: (-pair["phi"], pair["factors"]))
    return pairs[: config.condition_pairs]


def standardized(activations: torch.Tensor) -> torch.Tensor:
    centered = activations - activations.mean(dim=0)
    return centered / centered.std(dim=0, unbiased=False).clamp_min(1e-6)


def whitened(activations: torch.Tensor, eps: float) -> torch.Tensor:
    """ZCA-whitened factor activations with a ridge, rescaled to unit variance per column.

    ZCA keeps each output column closest to its own factor, so column k stays interpretable as
    "factor k with the other factors partialled out". The ridge ``eps`` limits the amplification of
    near-duplicate factors.
    """

    values = standardized(activations).double()
    covariance = values.T @ values / values.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    transform = eigenvectors @ torch.diag((eigenvalues.clamp_min(0) + eps).rsqrt()) @ eigenvectors.T
    return standardized((values @ transform).float())


def build_concept_factors(cache: dict, concept_groups: dict, config: FactorConfig) -> dict[str, Any]:
    """Factors, entangled pairs and both target kinds for every cached training image."""

    sample_ids = [str(value) for value in cache["sample_ids"]]
    if [str(value) for value in concept_groups["sample_ids"]] != sample_ids:
        raise ValueError("Concept groups and SpLiCE cache describe different training images.")
    vocabulary = [str(word) for word in cache["vocabulary"]]
    groups = list(concept_groups["groups"])
    for group in groups:
        names = [vocabulary[index] for index in group["concept_indices"]]
        if names != [str(concept) for concept in group["concepts"]]:
            raise ValueError(f"Concept group {group['group_id']} does not match the cache vocabulary.")
    codes = torch.as_tensor(cache["splice_codes"]).float()
    all_activations = group_activations(codes, groups)
    columns = select_factors(all_activations > 0, config)
    if len(columns) < 2:
        raise ValueError("Fewer than two concept groups fall inside the factor frequency band.")
    activations = all_activations[:, columns]
    active = activations > 0
    directions = group_directions(torch.as_tensor(cache["dictionary"]), [groups[column] for column in columns])
    pairs = entangled_pairs(active, directions, config)
    factors = [
        {
            "factor": position,
            "group_id": int(groups[column]["group_id"]),
            "concepts": [str(concept) for concept in groups[column]["concepts"]],
            "frequency": float(active[:, position].float().mean()),
        }
        for position, column in enumerate(columns)
    ]
    return {
        "artifact": CONCEPT_FACTORS_ARTIFACT,
        "config": asdict(config),
        "sample_ids": sample_ids,
        "factors": factors,
        "pairs": pairs,
        "condition_factors": sorted({factor for pair in pairs for factor in pair["factors"]}),
        "active": active,
        "targets": {
            "whitened": whitened(activations, config.whitening_eps),
            "standardized": standardized(activations),
        },
    }


def factor_report(factors: dict[str, Any]) -> dict[str, Any]:
    """The JSON-safe summary of a factor set: which concepts, which pairs."""

    names = [" / ".join(factor["concepts"][:3]) for factor in factors["factors"]]
    return {
        "artifact": factors["artifact"],
        "config": factors["config"],
        "sample_count": len(factors["sample_ids"]),
        "factors": factors["factors"],
        "pairs": [
            {**pair, "concepts": [names[pair["factors"][0]], names[pair["factors"][1]]]}
            for pair in factors["pairs"]
        ],
        "condition_factors": [names[factor] for factor in factors["condition_factors"]],
    }


def format_factor_report(report: dict[str, Any]) -> str:
    lines = [f"{len(report['factors'])} concept factors over {report['sample_count']} training images."]
    lines.append("Entangled pairs (phi, text similarity):")
    for pair in report["pairs"]:
        first, second = pair["concepts"]
        lines.append(f"  {pair['phi']:.3f}  {pair['text_similarity']:.3f}  {first}  <->  {second}")
    if not report["pairs"]:
        lines.append("  none: lower --factor_min_correlation or widen the frequency band.")
    return "\n".join(lines)


def resolve_concept_groups(dataset: str, explicit: str = "") -> Path:
    """The explicit concept-groups file, or the single one the grouping stage stored for ``dataset``."""

    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Concept groups not found: {path}")
        return path
    root = shared(dataset, "graphs", "concept_groups")
    candidates = sorted(root.glob("*/concept_groups.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one concept_groups.json under {root}, found {len(candidates)}; "
            "pass --factor_concept_groups PATH."
        )
    return candidates[0]


def resolve_splice_cache(dataset: str, concept_groups: dict, explicit: str = "") -> Path:
    """The explicit cache, or the one the cache stage built with the groups' provenance."""

    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"SpLiCE dataset cache not found: {path}")
        return path
    from cospro.cli.cache_splice_dataset import cache_config_name

    provenance = concept_groups.get("provenance", {})
    root = scratch_root() / "features" / "Spur_SpLiCE" / dataset / "splice_dataset_cache"
    try:
        name = cache_config_name(argparse.Namespace(
            dataset=dataset,
            splice_model=provenance["splice_model"],
            splice_pretrained=provenance["splice_pretrained"],
            splice_vocab=provenance["splice_vocab"],
            splice_vocab_size=int(provenance["splice_vocab_size"]),
            splice_l1_penalty=float(provenance["splice_l1_penalty"]),
            splice_vocab_file=None,
            splice_vocab_order=None,
        ))
        path = root / name / "splice_dataset_cache.pt"
    except (KeyError, ValueError, TypeError):
        path = None
    if path is not None and path.is_file():
        return path
    candidates = sorted(root.glob("*/splice_dataset_cache.pt"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Cannot find the SpLiCE cache of the concept groups under {root} ({len(candidates)} candidates); "
        "pass --factor_splice_cache PATH."
    )


def load_concept_factors(dataset: str, config: FactorConfig, *, concept_groups: str = "",
                         splice_cache: str = "") -> tuple[dict[str, Any], Path, Path]:
    """Build the factors of ``dataset`` from its stored concept groups and SpLiCE cache."""

    groups_path = resolve_concept_groups(dataset, concept_groups)
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    cache_path = resolve_splice_cache(dataset, groups, splice_cache)
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    expected = groups.get("provenance")
    if expected is not None and cache.get("provenance") is not None:
        mismatched = {key for key, value in expected.items() if cache["provenance"].get(key) != value}
        if mismatched:
            raise ValueError(f"SpLiCE cache {cache_path} differs from the concept groups in {sorted(mismatched)}.")
    factors = build_concept_factors(cache, groups, config)
    return factors, groups_path, cache_path


def rows_for_subset(sample_ids: list[str], dataset: str, source_indices) -> torch.Tensor:
    """Factor rows in the order of the training subset, matched by stable sample id."""

    row_by_id = {sample_id: row for row, sample_id in enumerate(sample_ids)}
    missing = [index for index in source_indices if f"{dataset}:{int(index)}" not in row_by_id]
    if missing:
        raise ValueError(f"{len(missing)} training images have no SpLiCE codes, e.g. {dataset}:{int(missing[0])}.")
    return torch.tensor([row_by_id[f"{dataset}:{int(index)}"] for index in source_indices], dtype=torch.long)


def expected_condition_pool(batch_size: int) -> int:
    """Smallest number of unused images a factor needs to fill a conditioned batch."""

    return max(2, math.ceil(batch_size / 2))
