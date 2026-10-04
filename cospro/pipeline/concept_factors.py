"""Concept factors: the dataset-level structure of the frozen SpLiCE codes, read without labels.

A factor starts as one concept group of the grouping stage; its activation on an image is the summed
SpLiCE code of the group's concepts. Three steps turn groups into what the concept-factor methods use:

* Redundancy merging. Two groups whose concept directions rise and fall together across the
  dataset's CLIP image embeddings describe the same visual content, whatever their words: breeds of
  one cat, "delta" and "aircraft". Groups above ``merge_similarity`` join one factor.
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

CONCEPT_FACTORS_ARTIFACT = "cospro_concept_factors_v2"
TARGET_KINDS = ("whitened", "standardized", "presence", "shuffled", "residual", "residual_shuffled", "clip_pca",
                "response", "response_residual")
#: Seed of the fixed image permutation behind the ``shuffled`` control targets.
SHUFFLE_SEED = 0


@dataclass(frozen=True)
class FactorConfig:
    """How factors, entangled pairs and targets are chosen from the codes."""

    min_frequency: float = 0.02
    max_frequency: float = 0.9
    max_count: int = 0
    merge_similarity: float = 0.0
    condition_pairs: int = 8
    min_correlation: float = 0.2
    max_text_similarity: float = 0.75
    whitening_eps: float = 0.1
    pca_components: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.min_frequency < self.max_frequency <= 1:
            raise ValueError("Factor frequencies need 0 <= min < max <= 1.")
        if self.max_count == 1 or self.max_count < 0 or self.condition_pairs < 1:
            raise ValueError("Concept factors need max_count 0 (all) or at least 2, and at least one pair.")
        if not 0 <= self.merge_similarity <= 1:
            raise ValueError("The merge similarity lies in [0, 1]; 0 disables merging.")
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
    """Columns within the frequency band; with ``max_count`` only that many, the most balanced first."""

    frequency = active.float().mean(dim=0)
    band = (frequency >= config.min_frequency) & (frequency <= config.max_frequency)
    candidates = torch.nonzero(band).view(-1).tolist()
    if config.max_count:
        balance = (frequency * (1 - frequency)).tolist()
        candidates.sort(key=lambda column: (-balance[column], column))
        candidates = candidates[: config.max_count]
    return sorted(candidates)


def image_similarity(directions: torch.Tensor, cache: dict) -> torch.Tensor:
    """Correlation, across the dataset's images, of their alignment with each pair of directions.

    With centered image embeddings E, the alignment with direction d is E d, and the correlation of
    E d_A with E d_B is d_A' S d_B normalized by the two variances, where S is the image covariance.
    Two directions correlate when the images that match one also match the other, which is what
    two names for the same visual content do.
    """

    embeddings = F.normalize(torch.as_tensor(cache["clip_embeddings"]).float(), dim=1)
    centered = embeddings - torch.as_tensor(cache["image_mean"]).float().view(1, -1)
    centered = centered - centered.mean(dim=0)
    covariance = centered.T @ centered / centered.shape[0]
    cross = directions.float() @ covariance @ directions.float().T
    scale = cross.diagonal().clamp_min(1e-12).sqrt()
    return cross / scale[:, None] / scale[None, :]


def merge_redundant(similarity: torch.Tensor, threshold: float) -> list[list[int]]:
    """Connected components of the columns whose image similarity reaches ``threshold``."""

    count = similarity.shape[0]
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    if threshold > 0:
        rows, columns = torch.where(torch.triu(similarity >= threshold, diagonal=1))
        for row, column in zip(rows.tolist(), columns.tolist()):
            parent[find(row)] = find(column)
    components: dict[int, list[int]] = {}
    for column in range(count):
        components.setdefault(find(column), []).append(column)
    return sorted(components.values(), key=lambda members: members[0])


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


#: Ridge of the whole-dataset learnability fit. On MetaShift's 1,700 images it keeps shuffled targets
#: and random features at an explained variance of -0.01 while CLIP features reach +0.26.
LEARNABILITY_RIDGE = 10.0


def partial_residuals(targets: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Each column minus its linear regression on the other columns, rescaled to unit variance.

    With P the inverse covariance, the residual of column k is (T P)_k / P_kk. A direction shared by
    two correlated factors predicts their common part and little of these residuals: for two factors
    of correlation rho, a fused direction explains (1 - rho) / 2 of each residual.
    """

    values = torch.as_tensor(targets).double()
    values = values - values.mean(dim=0)
    covariance = values.T @ values / values.shape[0]
    precision = torch.linalg.inv(covariance + eps * torch.eye(covariance.shape[0], dtype=values.dtype))
    residuals = values @ precision / precision.diagonal()
    return standardized(residuals.float())


def _ridge_predictions(features: torch.Tensor, targets: torch.Tensor, fit: torch.Tensor, predict: torch.Tensor,
                       ridge: float) -> torch.Tensor:
    """Ridge predictions for the ``predict`` rows, fitted on the ``fit`` rows with centred features and targets."""

    feature_mean, target_mean = features[fit].mean(dim=0), targets[fit].mean(dim=0)
    centred = features[fit] - feature_mean
    identity = torch.eye(features.shape[1], dtype=features.dtype)
    weights = torch.linalg.solve(centred.T @ centred + ridge * identity, centred.T @ (targets[fit] - target_mean))
    return (features[predict] - feature_mean) @ weights + target_mean


def out_of_fold_predictions(features: torch.Tensor, targets: torch.Tensor, *, ridge: float = LEARNABILITY_RIDGE,
                            folds: int = 5, seed: int = 0) -> torch.Tensor:
    """Every row predicted by a ridge regression fitted on the other folds; rows of ``features`` at unit norm."""

    features = F.normalize(torch.as_tensor(features).double(), dim=1)
    targets = torch.as_tensor(targets).double()
    count = features.shape[0]
    if count < folds * 2:
        raise ValueError("The learnability fit needs at least two images per fold.")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    predictions = torch.zeros_like(targets)
    for fold in range(folds):
        held_out = torch.zeros(count, dtype=torch.bool)
        held_out[order[fold::folds]] = True
        predictions[held_out] = _ridge_predictions(features, targets, ~held_out, held_out, ridge)
    return predictions


def unseen_predictions(features: torch.Tensor, targets: torch.Tensor, unseen: torch.Tensor, *,
                       ridge: float = LEARNABILITY_RIDGE) -> torch.Tensor:
    """Predictions for the ``unseen`` rows of a ridge regression fitted on the other rows."""

    features = F.normalize(torch.as_tensor(features).double(), dim=1)
    targets = torch.as_tensor(targets).double()
    unseen = torch.as_tensor(unseen, dtype=torch.bool)
    return _ridge_predictions(features, targets, ~unseen, unseen, ridge)


def explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-column explained variance of ``predictions`` against ``targets`` over the same rows."""

    targets = torch.as_tensor(targets).double()
    variance = (targets - targets.mean(dim=0)).pow(2).mean(dim=0).clamp_min(1e-12)
    return (1.0 - (torch.as_tensor(predictions).double() - targets).pow(2).mean(dim=0) / variance).float()


def group_explained_variance(predictions: torch.Tensor, targets: torch.Tensor,
                             groups: torch.Tensor) -> dict[int, torch.Tensor]:
    """Per-column explained variance inside every group, against the variance over all rows.

    The shared denominator makes the groups comparable: a direction that fuses a concept with its
    usual context predicts the concept well where the two co-occur and badly in the groups that break
    the co-occurrence, so its score drops there while its score over all rows stays high.
    """

    predictions = torch.as_tensor(predictions).double()
    targets = torch.as_tensor(targets).double()
    groups = torch.as_tensor(groups)
    variance = (targets - targets.mean(dim=0)).pow(2).mean(dim=0).clamp_min(1e-12)
    return {
        int(group): (1.0 - (predictions[groups == group] - targets[groups == group]).pow(2).mean(dim=0) / variance).float()
        for group in torch.unique(groups).tolist()
    }


def heldout_explained_variance(features: torch.Tensor, targets: torch.Tensor, *, ridge: float = LEARNABILITY_RIDGE,
                               folds: int = 5, seed: int = 0) -> torch.Tensor:
    """Per-column explained variance of a ridge regression predicting each fold from the others.

    A column the features encode across images scores high; a fixed linear map gains nothing from
    single images, because every prediction is for images outside its fit.
    """

    predictions = out_of_fold_predictions(features, targets, ridge=ridge, folds=folds, seed=seed)
    return explained_variance(predictions, targets)


def unseen_explained_variance(features: torch.Tensor, targets: torch.Tensor, unseen: torch.Tensor, *,
                              ridge: float = LEARNABILITY_RIDGE) -> torch.Tensor:
    """Per-column explained variance on the ``unseen`` rows of a ridge regression fitted on the other rows.

    With ``unseen`` marking the images a concept loss never touched, this measures whether the
    encoder's concept directions carry over to new images. The whole-dataset score of
    ``heldout_explained_variance`` holds images out of the ridge only, so an encoder that stores the
    target of each training image still scores high there.
    """

    unseen = torch.as_tensor(unseen, dtype=torch.bool)
    predictions = unseen_predictions(features, targets, unseen, ridge=ridge)
    return explained_variance(predictions, torch.as_tensor(targets)[unseen])


def atypicality_weights(white: torch.Tensor, cap: float = 10.0) -> torch.Tensor:
    """Per-image weights, mean 1, that grow with how far an image's factors break the dataset's correlations.

    The atypicality of an image is the mean square of its whitened factor values: small for images
    whose concepts co-occur as usual, large for images that show one concept without the partner it
    usually comes with. No label enters; ``cospro.diagnostics.factor_validity.weight_by_group``
    checks post hoc whether the weight lands on the minority groups.
    """

    atypicality = white.pow(2).mean(dim=1)
    weights = (atypicality / atypicality.mean().clamp_min(1e-12)).clamp(max=cap)
    return weights / weights.mean()


def balancing_weights(active: torch.Tensor, *, steps: int = 300, learning_rate: float = 0.05,
                      entropy: float = 0.05, cap: float = 20.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-image weights, mean 1, under which the factors' presences are as uncorrelated as possible.

    Minimizes the mean squared off-diagonal correlation of the weighted presence indicators plus
    ``entropy`` times the mean of w log w, which keeps the weights close to uniform. Under such weights
    a concept that usually comes with another one no longer predicts it, so a loss that relies on the
    co-occurrence gains nothing from it. This is sample reweighting for independence, as in stable
    learning, applied to the concept presences; no label enters.
    """

    values = active.float()
    values = (values - values.mean(dim=0)) / values.std(dim=0).clamp_min(1e-6)
    count, width = values.shape

    def correlation(weights: torch.Tensor) -> torch.Tensor:
        centered = values - (weights[:, None] * values).mean(dim=0)
        covariance = (centered * weights[:, None]).T @ centered / count
        scale = covariance.diagonal().clamp_min(1e-8).sqrt()
        return covariance / scale[:, None] / scale[None, :]

    off_diagonal = ~torch.eye(width, dtype=torch.bool)
    logits = torch.zeros(count, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    for _ in range(steps):
        weights = torch.softmax(logits, dim=0) * count
        loss = correlation(weights)[off_diagonal].pow(2).mean() + entropy * (weights * weights.log()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        weights = (torch.softmax(logits, dim=0) * count).clamp(max=cap)
        weights = weights / weights.mean()
        diagnostics = {
            "mean_abs_correlation_before": float(correlation(torch.ones(count))[off_diagonal].abs().mean()),
            "mean_abs_correlation_after": float(correlation(weights)[off_diagonal].abs().mean()),
            "effective_sample_fraction": float(weights.sum() ** 2 / weights.pow(2).sum() / count),
            "max_weight": float(weights.max()),
        }
    return weights.detach(), diagnostics


def block_factor_columns(active: torch.Tensor, count: int) -> list[int]:
    """The ``count`` factors whose presence is most balanced, which give a block both positives and negatives."""

    frequency = active.float().mean(dim=0)
    balance = (frequency * (1 - frequency)).tolist()
    ranked = sorted(range(len(balance)), key=lambda column: (-balance[column], column))
    return sorted(ranked[:count])


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
    selected = [groups[column] for column in columns]
    group_values = all_activations[:, columns]
    dictionary = torch.as_tensor(cache["dictionary"])
    components = merge_redundant(
        image_similarity(group_directions(dictionary, selected), cache), config.merge_similarity,
    )
    if len(components) < 2:
        raise ValueError("Redundancy merging left fewer than two factors; raise --factor_merge_similarity.")
    activations = torch.stack([group_values[:, members].sum(dim=1) for members in components], dim=1)
    active = activations > 0
    directions = F.normalize(torch.stack([
        group_directions(dictionary, [selected[member] for member in members]).mean(dim=0)
        for members in components
    ]), dim=1)
    pairs = entangled_pairs(active, directions, config)
    factors = []
    for position, members in enumerate(components):
        # Members in decreasing frequency, so the name of a factor leads with its commonest concept.
        members = sorted(members, key=lambda member: (-float((group_values[:, member] > 0).float().mean()), member))
        factors.append({
            "factor": position,
            "group_ids": [int(selected[member]["group_id"]) for member in members],
            "concepts": [str(concept) for member in members for concept in selected[member]["concepts"]],
            "frequency": float(active[:, position].float().mean()),
        })
    shuffle = torch.randperm(activations.shape[0], generator=torch.Generator().manual_seed(SHUFFLE_SEED))
    residuals = partial_residuals(standardized(activations))
    # Control: as many principal components of the CLIP image embeddings as there are factors. It
    # distils the same frozen model at the same width with no concept, vocabulary or grouping.
    clip = F.normalize(torch.as_tensor(cache["clip_embeddings"]).float(), dim=1)
    clip = clip - clip.mean(dim=0)
    left, values, _ = torch.linalg.svd(clip, full_matrices=False)
    width = min(config.pca_components or activations.shape[1], values.shape[0])
    clip_components = standardized(left[:, :width] * values[:width])
    # Dense alternative to the sparse codes: every image's CLIP alignment with each factor's text
    # direction. SpLiCE's L1 penalty keeps a handful of concepts per image; the alignment exists for all
    # of them, and as a linear function of CLIP each residual is a direction CLIP features carry.
    centred = F.normalize(torch.as_tensor(cache["clip_embeddings"]).float(), dim=1)
    centred = centred - torch.as_tensor(cache["image_mean"]).float().view(1, -1)
    responses = standardized(centred @ directions.float().T)
    return {
        "artifact": CONCEPT_FACTORS_ARTIFACT,
        "config": asdict(config),
        "sample_ids": sample_ids,
        "group_count": len(columns),
        "factors": factors,
        "pairs": pairs,
        "condition_factors": sorted({factor for pair in pairs for factor in pair["factors"]}),
        "active": active,
        "atypicality_weights": atypicality_weights(whitened(activations, config.whitening_eps)),
        "targets": {
            "whitened": whitened(activations, config.whitening_eps),
            "standardized": standardized(activations),
            # Presence only: whether each factor fires, without how strongly.
            "presence": standardized(active.float()),
            # Control: the standardized targets of other images. Same statistics, no image content.
            "shuffled": standardized(activations)[shuffle],
            # Each factor minus its regression on the others: only the part a factor does not share with
            # the factors it co-occurs with, which a fused direction cannot predict.
            "residual": residuals,
            "residual_shuffled": residuals[shuffle],
            "clip_pca": clip_components,
            "response": responses,
            "response_residual": partial_residuals(responses),
        },
    }


def factor_name(factor: dict) -> str:
    # " | " rather than a slash: run records read any string with a slash as a path.
    concepts = factor["concepts"]
    return " | ".join(concepts[:3]) + (f" (+{len(concepts) - 3})" if len(concepts) > 3 else "")


def factor_report(factors: dict[str, Any]) -> dict[str, Any]:
    """The JSON-safe summary of a factor set: which concepts, which pairs."""

    names = [factor_name(factor) for factor in factors["factors"]]
    return {
        "artifact": factors["artifact"],
        "config": factors["config"],
        "sample_count": len(factors["sample_ids"]),
        "group_count": factors["group_count"],
        "factors": factors["factors"],
        "pairs": [
            {**pair, "concepts": [names[pair["factors"][0]], names[pair["factors"][1]]]}
            for pair in factors["pairs"]
        ],
        "condition_factors": [names[factor] for factor in factors["condition_factors"]],
    }


def format_factor_report(report: dict[str, Any]) -> str:
    lines = [
        f"{len(report['factors'])} concept factors from {report['group_count']} concept groups "
        f"over {report['sample_count']} training images."
    ]
    lines.append("Entangled pairs (phi, text similarity):")
    for pair in report["pairs"]:
        first, second = pair["concepts"]
        lines.append(f"  {pair['phi']:.3f}  {pair['text_similarity']:.3f}  {first}  <->  {second}")
    if not report["pairs"]:
        lines.append("  none: lower --factor_min_correlation or widen the frequency band.")
    return "\n".join(lines)


def default_grouping_directory() -> str:
    """Directory name the grouping stage gives the default thresholds, e.g. ``text_0p8_coactivation_0p3``."""

    from cospro.pipeline.config import CoSpRoAuditConfig

    defaults = CoSpRoAuditConfig()

    def name(value: float) -> str:
        return f"{value:.12g}".replace("-", "neg").replace(".", "p")

    return f"text_{name(defaults.text_similarity_threshold)}_coactivation_{name(defaults.coactivation_threshold)}"


def resolve_concept_groups(dataset: str, explicit: str = "") -> Path:
    """The explicit concept-groups file, else the one at the default thresholds, else the only one."""

    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Concept groups not found: {path}")
        return path
    root = shared(dataset, "graphs", "concept_groups")
    preferred = root / default_grouping_directory() / "concept_groups.json"
    if preferred.is_file():
        return preferred
    candidates = sorted(root.glob("*/concept_groups.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected {preferred} or exactly one concept_groups.json under {root}, found {len(candidates)}; "
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
    # The CLIP embeddings the factors come from bound what a student can learn of them.
    factors["clip_embeddings"] = torch.as_tensor(cache["clip_embeddings"]).float()
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
