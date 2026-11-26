"""Concept shortcut attribution at inference: remove the tokens of one concept unit and read the probe.

The inference counterpart of Concept Shortcut Attribution (CSA, ``docs/PROJECT_CSA.md`` on the
concept-factors branch, later work step 1). A frozen ViT and a linear probe stay fixed. For every concept
unit, the patches the unit wins in an evaluation image are removed from the encoder's input, the image is
encoded again and the probe classifies it. The change of worst-group accuracy is the unit's WGA effect
``E_k``: it rises when the probe relied on the unit as a shortcut and falls when the probe needed it.

Units follow CSA exactly: concept factors grouped by the cells they share in the CLIP/SpLiCE maps of the
training images (spectral clustering, ``concept_units``), so unit ``k`` names the same concept as the
CSA arm ``curves_unit_<k>`` built from the same maps. A unit's hard region in an image is the set of
patches where its map is the largest. Removing tokens is exact for a ViT: the removed patches receive no
attention, so no fill value enters the image.

Every function is pure tensor code and reads no label, except ``unit_typing`` and the group metrics,
which score the attribution after the fact.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

#: CSA counts a unit as present with 3 of 196 cells; the same share of a 16 x 16 grid is 4 patches.
MIN_SHARE = 3 / 196
#: A unit is attributed when it is present in at least this share of the evaluation images, as in CSA.
MIN_PRESENCE = 0.02
#: Every occluded image keeps at least this share of its patches.
MIN_KEPT_SHARE = 0.1


def _kmeans(rows: torch.Tensor, count: int, seed: int, iterations: int = 100) -> torch.Tensor:
    """Cluster labels of ``rows`` [n, d] by k-means with k-means++ seeding, deterministic for a seed."""

    generator = torch.Generator().manual_seed(seed)
    centers = rows[torch.randint(len(rows), (1,), generator=generator)]
    while len(centers) < count:
        distance = torch.cdist(rows, centers).min(dim=1).values.pow(2)
        centers = torch.cat([centers, rows[torch.multinomial(distance / distance.sum().clamp_min(1e-12), 1,
                                                               generator=generator)]])
    labels = torch.cdist(rows, centers).argmin(dim=1)
    for _ in range(iterations):
        centers = torch.stack([rows[labels == group].mean(dim=0) if bool((labels == group).any()) else centers[group]
                               for group in range(count)])
        updated = torch.cdist(rows, centers).argmin(dim=1)
        if torch.equal(updated, labels):
            break
        labels = updated
    return labels


def concept_units(maps: torch.Tensor, count: int, seed: int = 0) -> torch.Tensor:
    """Unit id per factor: ``count`` groups of factors that share cells, or one unit per factor for 0.

    ``maps`` is [images, factors, h, w]. The same spectral clustering as ``coactivation_groups`` and
    ``build_units`` of the concept-factors branch, ids numbered in order of first appearance.
    """

    images, factors = maps.shape[:2]
    if count <= 0 or count >= factors:
        return torch.arange(factors)
    gram = torch.zeros(factors, factors, dtype=torch.float64)
    for start in range(0, images, 256):
        block = maps[start:start + 256].double().transpose(0, 1).reshape(factors, -1)
        gram += block @ block.T
    norm = gram.diag().clamp_min(1e-12).sqrt()
    similarity = gram / norm[:, None] / norm[None, :]
    degree = similarity.sum(dim=1)
    laplacian = torch.eye(factors, dtype=torch.float64) - similarity / (degree[:, None] * degree[None, :]).sqrt()
    _, vectors = torch.linalg.eigh(laplacian)
    labels = _kmeans(F.normalize(vectors[:, :count], dim=1).float(), count, seed)
    order: dict[int, int] = {}
    return torch.tensor([order.setdefault(int(label), len(order)) for label in labels], dtype=torch.long)


def unit_maps(factor_maps: torch.Tensor, unit_of_factor: torch.Tensor) -> torch.Tensor:
    """Per-patch unit mass [images, patches, units] from per-patch factor mass [images, patches, factors]."""

    units = int(unit_of_factor.max()) + 1
    output = torch.zeros(*factor_maps.shape[:2], units, dtype=torch.float32, device=factor_maps.device)
    return output.index_add_(2, unit_of_factor.to(factor_maps.device), factor_maps.float())


def hard_regions(unit_mass: torch.Tensor) -> torch.Tensor:
    """Boolean [images, patches, units]: each patch belongs to the unit with the largest mass there."""

    return F.one_hot(unit_mass.argmax(dim=-1), unit_mass.shape[-1]).bool()


def presence(regions: torch.Tensor, min_share: float = MIN_SHARE) -> torch.Tensor:
    """Boolean [images, units]: the unit wins at least ``min_share`` of the image's patches."""

    patches = regions.shape[1]
    return regions.sum(dim=1) >= max(1, math.ceil(min_share * patches))


def drop_masks(region: torch.Tensor, shown: torch.Tensor,
               min_kept_share: float = MIN_KEPT_SHARE) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep masks [images, patches] without ``region`` [images, patches] and the images where removal is valid.

    Removal is valid in the images that show the unit (``shown`` [images]) and keep at least
    ``min_kept_share`` of their patches.
    """

    keep = ~region
    valid = shown & region.any(dim=1) & (keep.float().mean(dim=1) >= min_kept_share)
    return keep, valid


def shuffled_regions(regions: torch.Tensor, unit: int, seed: int = 0) -> torch.Tensor:
    """The control: each image that shows ``unit`` receives the unit's region of another such image.

    Area and frequency of the removed patches stay those of the unit; their location loses its link to
    the image content. Images without the unit keep their own (empty or small) region.
    """

    region = regions[:, :, unit].clone()
    shown = torch.nonzero(presence(regions)[:, unit]).flatten()
    if len(shown) < 2:
        return region
    generator = torch.Generator().manual_seed(seed + 7919 * unit)
    order = torch.randperm(len(shown), generator=generator)
    # A cyclic shift of a random order gives every image a partner other than itself.
    partner = shown[order.roll(1)]
    region[shown[order]] = regions[partner, :, unit]
    return region


def keep_pooled(tokens: torch.Tensor, keep: torch.Tensor | None) -> torch.Tensor:
    """Mean over the kept patches of ``tokens`` [batch, patches, dim], in float32."""

    tokens = tokens.float()
    if keep is None:
        return tokens.mean(dim=1)
    weights = keep.to(tokens.device, tokens.dtype)[..., None]
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def group_accuracy(predictions: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor, count: int = 4) -> torch.Tensor:
    """Accuracy in percent of every group, NaN for an empty group."""

    correct = (predictions == labels).double()
    hits = torch.bincount(groups, weights=correct, minlength=count)
    sizes = torch.bincount(groups, minlength=count).double()
    return torch.where(sizes > 0, 100 * hits / sizes.clamp_min(1), torch.full_like(sizes, float("nan")))


def _worst(accuracy: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(accuracy, nan=float("inf")).min(dim=-1).values


def paired_effect(intact_logits: torch.Tensor, occluded_logits: torch.Tensor, labels: torch.Tensor,
                  groups: torch.Tensor, resamples: int = 1000, seed: int = 0) -> dict:
    """WGA, mean accuracy and per-group accuracy before and after removal, with a paired bootstrap of the WGA change.

    ``occluded_logits`` equal ``intact_logits`` for images the removal leaves untouched. The bootstrap
    resamples images within every group, so group sizes stay those of the split.
    """

    before, after = intact_logits.argmax(dim=1), occluded_logits.argmax(dim=1)
    groups_before, groups_after = group_accuracy(before, labels, groups), group_accuracy(after, labels, groups)
    effect = float(_worst(groups_after) - _worst(groups_before))
    generator = torch.Generator().manual_seed(seed)
    count = int(groups.max()) + 1
    correct_before, correct_after = (before == labels).double(), (after == labels).double()
    members = [torch.nonzero(groups == group).flatten() for group in range(count)]
    draws_before = torch.full((resamples, count), float("nan"), dtype=torch.float64)
    draws_after = draws_before.clone()
    for group, rows in enumerate(members):
        if len(rows) == 0:
            continue
        picks = rows[torch.randint(len(rows), (resamples, len(rows)), generator=generator)]
        draws_before[:, group] = 100 * correct_before[picks].mean(dim=1)
        draws_after[:, group] = 100 * correct_after[picks].mean(dim=1)
    changes = _worst(draws_after) - _worst(draws_before)
    low, high = torch.quantile(changes, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    sign = 2.0 * labels.double() - 1.0
    margin_before = (intact_logits[:, 1] - intact_logits[:, 0]).double() * sign
    margin_after = (occluded_logits[:, 1] - occluded_logits[:, 0]).double() * sign
    margin_change = torch.bincount(groups, weights=margin_after - margin_before, minlength=count) / \
        torch.bincount(groups, minlength=count).double().clamp_min(1)
    return {
        "wga_before": round(float(_worst(groups_before)), 2),
        "wga_after": round(float(_worst(groups_after)), 2),
        "wga_effect": round(effect, 2),
        "wga_effect_ci95": [round(low, 2), round(high, 2)],
        "accuracy_effect": round(100 * float(correct_after.mean() - correct_before.mean()), 2),
        "group_accuracy_effect": [round(float(value), 2) for value in groups_after - groups_before],
        "true_class_margin_change": [round(float(value), 4) for value in margin_change],
        "changed_predictions": int((before != after).sum()),
    }


def _entropy(counts: torch.Tensor) -> float:
    probabilities = counts[counts > 0].double() / counts.sum().clamp_min(1)
    return float(-(probabilities * probabilities.log()).sum())


def _mutual_information(first: torch.Tensor, second: torch.Tensor) -> float:
    if len(first) == 0:
        return 0.0
    first, second = first.long(), second.long()
    width = int(second.max()) + 1
    joint = torch.bincount(first * width + second, minlength=(int(first.max()) + 1) * width)
    return _entropy(torch.bincount(first)) + _entropy(torch.bincount(second)) - _entropy(joint)


def _conditional_information(first: torch.Tensor, second: torch.Tensor, given: torch.Tensor) -> float:
    return sum(float((given == value).float().mean()) * _mutual_information(first[given == value], second[given == value])
               for value in given.unique().tolist())


def unit_typing(present: torch.Tensor, labels: torch.Tensor, attributes: torch.Tensor,
                threshold: float = 0.05, ratio: float = 2.0) -> list[dict]:
    """Type every unit against the hidden labels as CSA does: attribute, class or untyped. Evaluation only.

    ``u_attribute = I(P; a | y) / H(P)`` and ``u_class = I(P; y | a) / H(P)`` for the presence indicator
    ``P`` of a unit.
    """

    typing = []
    for unit in range(present.shape[1]):
        indicator = present[:, unit].long()
        entropy = _entropy(torch.bincount(indicator, minlength=2))
        if entropy <= 0:
            typing.append({"unit": unit, "u_attribute": 0.0, "u_class": 0.0, "type": "untyped"})
            continue
        u_attribute = _conditional_information(indicator, attributes, labels) / entropy
        u_class = _conditional_information(indicator, labels, attributes) / entropy
        kind = "untyped"
        if u_attribute >= threshold and u_attribute >= ratio * u_class:
            kind = "attribute"
        elif u_class >= threshold and u_class >= ratio * u_attribute:
            kind = "class"
        typing.append({"unit": unit, "u_attribute": round(u_attribute, 4), "u_class": round(u_class, 4), "type": kind})
    return typing


def spearman(first: list[float], second: list[float]) -> float:
    """Spearman rank correlation with average ranks for ties; NaN below three pairs."""

    if len(first) < 3:
        return math.nan

    def ranks(values: list[float]) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.float64)
        order = tensor.argsort()
        result = torch.empty_like(tensor)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and tensor[order[end + 1]] == tensor[order[position]]:
                end += 1
            result[order[position:end + 1]] = (position + end) / 2.0
            position = end + 1
        return result

    a, b = ranks(first), ranks(second)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(a.norm() * b.norm())
    return float(a @ b) / denominator if denominator > 0 else math.nan


def region_agreement(predicted: torch.Tensor, reference: torch.Tensor) -> dict:
    """How well predicted hard regions match reference ones, both boolean [images, patches, units].

    ``patch_accuracy`` is the share of patches assigned to the same unit; ``unit_iou`` the intersection
    over union of every unit's regions summed over images.
    """

    same = (predicted.float().argmax(dim=-1) == reference.float().argmax(dim=-1)).float().mean()
    intersection = (predicted & reference).sum(dim=(0, 1)).double()
    union = (predicted | reference).sum(dim=(0, 1)).double()
    iou = torch.where(union > 0, intersection / union.clamp_min(1), torch.full_like(union, float("nan")))
    return {"patch_accuracy": round(float(same), 4), "unit_iou": [round(float(value), 4) for value in iou]}
