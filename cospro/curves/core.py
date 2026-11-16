"""The audit: which named concepts a model relies on, epoch by epoch, and which of them get flagged.

Patch-wise concept maps say where each concept factor sits in each training image. Factors that share
pixels form a concept unit, and the cells a unit wins in an image are its hard region. Occluding that
region and measuring the drop of the correct-class margin gives the model's reliance on the unit. The
reliance of every unit over the first epochs is its learning curve; the units that lead the curves are
flagged as spurious. Sections 4.2 to 4.5 of docs/PROJECT_PLAN.md define every quantity here.

The functions are pure tensor code. Group labels enter only ``unit_typing``, which scores the audit and
never feeds training, except for the ``oracle`` rule that is an upper-bound control by design.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from cospro.pipeline.concept_maps import coactivation_groups, inpaint, sharpen

#: A unit is present in an image when its hard region holds at least this many map cells.
MIN_CELLS = 3
#: A unit is audited when it is present in at least this share of the training images.
MIN_PRESENCE = 0.02
FLAG_RULES = ("top", "bottom", "random", "units", "oracle")


def load_unit_maps(path: str, dataset: str, source_indices) -> tuple[torch.Tensor, list[str]]:
    """Concept maps of the training subset in its order, [n, factors, 14, 14], and the factor names."""

    from cospro.pipeline.concept_factors import rows_for_subset

    stored = torch.load(path, map_location="cpu", weights_only=False)
    if stored.get("kind", "concepts") != "concepts":
        raise ValueError(f"{path} holds principal-component maps; the audit needs concept maps.")
    rows = rows_for_subset([str(sample_id) for sample_id in stored["sample_ids"]], dataset, source_indices)
    return stored["maps"].index_select(0, rows), [str(name) for name in stored["factor_names"]]


def build_units(maps: torch.Tensor, count: int, seed: int = 0) -> torch.Tensor:
    """Unit id per factor: ``count`` groups of factors that share pixels, or one unit per factor for 0."""

    factors = maps.shape[1]
    if count <= 0 or count >= factors:
        return torch.arange(factors)
    labels = coactivation_groups(maps, count, seed)
    # Consecutive ids in order of first appearance, so an empty cluster leaves no gap.
    order: dict[int, int] = {}
    return torch.tensor([order.setdefault(int(label), len(order)) for label in labels], dtype=torch.long)


def hard_regions(maps: torch.Tensor, unit_of_factor: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """The cells every unit wins in every image, a boolean [n, units, h, w] that partitions each image."""

    units = int(unit_of_factor.max()) + 1
    images, _, height, width = maps.shape
    regions = torch.zeros(images, units, height, width, dtype=torch.bool)
    for start in range(0, images, chunk):
        block = maps[start:start + chunk].float()
        unit_maps = torch.zeros(len(block), units, height, width).index_add_(1, unit_of_factor, block)
        regions[start:start + chunk].scatter_(1, unit_maps.argmax(dim=1, keepdim=True), True)
    return regions


def unit_presence(regions: torch.Tensor) -> torch.Tensor:
    """Boolean [n, units]: the unit's hard region holds at least ``MIN_CELLS`` cells."""

    return regions.flatten(2).sum(dim=2) >= MIN_CELLS


def describe_units(maps: torch.Tensor, unit_of_factor: torch.Tensor, regions: torch.Tensor,
                   factor_names: list[str]) -> list[dict]:
    """Name, factors, presence rate, mean area where present and the audited flag of every unit."""

    mass = torch.zeros(maps.shape[1], dtype=torch.float64)
    for start in range(0, len(maps), 512):
        mass += maps[start:start + 512].double().sum(dim=(0, 2, 3))
    present = unit_presence(regions)
    cells = regions.flatten(2).sum(dim=2).float() / (regions.shape[2] * regions.shape[3])
    units = []
    for unit in range(int(unit_of_factor.max()) + 1):
        members = (unit_of_factor == unit).nonzero().view(-1).tolist()
        members.sort(key=lambda factor: -float(mass[factor]))
        rate = float(present[:, unit].float().mean())
        area = float(cells[present[:, unit], unit].mean()) if bool(present[:, unit].any()) else 0.0
        units.append({
            "unit": unit,
            "name": " / ".join(factor_names[factor] for factor in members[:3]),
            "factors": [factor_names[factor] for factor in members],
            "presence": round(rate, 4),
            "area": round(area, 4),
            "audited": rate >= MIN_PRESENCE,
        })
    return units


def pixel_masks(regions: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Hard regions [b, 1, h, w] as pixel masks [b, 1, H, W] with a narrow soft border."""

    return sharpen(F.interpolate(regions.float(), size=size, mode="bilinear", align_corners=False))


def occlude(images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Images with the masked pixels filled from their surroundings (``inpaint``)."""

    return (1.0 - masks) * images + masks * inpaint(images, masks)


def margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Correct-class logit minus the largest other logit, one value per row."""

    logits = logits.float()
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    others = logits.scatter(1, labels[:, None], float("-inf")).max(dim=1).values
    return correct - others


@torch.no_grad()
def reliance_audit(model, loader, regions: torch.Tensor, audited: list[int], device, amp: bool = False) -> dict:
    """Mean and mean squared margin drop per unit when its region is occluded, over the images that show it.

    ``loader`` yields (images, rows, labels) of unaugmented training images; ``rows`` index ``regions``.
    Returns ``reliance`` and ``squared`` (one value per unit, 0 for units outside ``audited``) and the
    number of images behind each value.
    """

    units = regions.shape[1]
    present = unit_presence(regions)
    total = torch.zeros(units, dtype=torch.float64)
    total_squared = torch.zeros(units, dtype=torch.float64)
    counts = torch.zeros(units, dtype=torch.float64)
    was_training = model.training
    model.eval()
    try:
        for images, rows, labels in loader:
            images, labels = images.to(device), labels.to(device)
            rows = torch.as_tensor(rows, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=amp and str(device).startswith("cuda")):
                base = margins(model(images), labels)
                for unit in audited:
                    shown = present[rows, unit]
                    if not bool(shown.any()):
                        continue
                    chosen = shown.to(device)
                    masks = pixel_masks(regions[rows[shown], unit:unit + 1].to(device), tuple(images.shape[-2:]))
                    occluded = occlude(images[chosen].float(), masks).to(images.dtype)
                    drop = (base[chosen] - margins(model(occluded), labels[chosen])).double().cpu()
                    total[unit] += drop.sum()
                    total_squared[unit] += drop.pow(2).sum()
                    counts[unit] += len(drop)
    finally:
        model.train(was_training)
    safe = counts.clamp_min(1.0)
    return {"reliance": (total / safe).tolist(), "squared": (total_squared / safe).tolist(),
            "images": counts.tolist()}


def concept_shares(reliance: list[float], audited: list[int]) -> list[float]:
    """Positive part of the reliance, normalized over the audited units; all zero when none is positive."""

    positive = [max(float(reliance[unit]), 0.0) if unit in set(audited) else 0.0 for unit in range(len(reliance))]
    scale = sum(positive)
    return [value / scale if scale > 0 else 0.0 for value in positive]


def onset_scores(epochs: list[int], shares: list[list[float]], early_epochs: int) -> list[float]:
    """Mean share of every unit over the audits of epochs 1 to ``early_epochs``."""

    early = [row for epoch, row in zip(epochs, shares) if 1 <= epoch <= early_epochs]
    if not early:
        raise ValueError(f"No audit lies in epochs 1 to {early_epochs}.")
    return [sum(row[unit] for row in early) / len(early) for unit in range(len(early[0]))]


def flag_units(onset: list[float], units: list[dict], rule: str = "top", tau: float = 0.5,
               max_area: float = 0.75, seed: int = 0, explicit: list[int] | None = None) -> list[int]:
    """The flagged unit ids under one rule (Section 4.5 of the plan).

    ``top`` takes the shortest prefix of the audited units by decreasing onset score whose scores reach
    the share ``tau`` of the total. A unit that lifts the expected flagged area of an image (presence
    times area, summed) above ``max_area`` ends the prefix; one unit is always flagged and one audited
    unit always stays. ``bottom`` and ``random`` flag as many units as ``top``, from the end of the
    ranking and uniformly at random. ``units`` returns ``explicit``.
    """

    if rule == "units":
        if not explicit:
            raise ValueError("The rule 'units' needs an explicit list of unit ids.")
        return sorted({int(unit) for unit in explicit})
    if rule not in ("top", "bottom", "random"):
        raise ValueError(f"Unknown flag rule {rule!r}.")
    audited = [entry["unit"] for entry in units if entry["audited"]]
    if len(audited) < 2:
        raise ValueError("Flagging needs at least two audited units.")
    ranked = sorted(audited, key=lambda unit: (-onset[unit], unit))
    total = sum(onset[unit] for unit in audited)
    if total <= 0:
        raise ValueError("Every onset score is zero; the model shows no positive reliance in the early window.")
    top: list[int] = []
    reached, area = 0.0, 0.0
    for unit in ranked[:-1]:
        footprint = units[unit]["presence"] * units[unit]["area"]
        if top and area + footprint > max_area:
            break
        top.append(unit)
        reached += onset[unit] / total
        area += footprint
        if reached >= tau:
            break
    if rule == "top":
        return sorted(top)
    if rule == "bottom":
        return sorted(ranked[-len(top):])
    order = torch.randperm(len(audited), generator=torch.Generator().manual_seed(seed)).tolist()
    return sorted(audited[position] for position in order[:len(top)])


def flagged_area(flagged: list[int], units: list[dict]) -> float:
    """Expected share of an image that the flagged regions cover."""

    return sum(units[unit]["presence"] * units[unit]["area"] for unit in flagged)


def _entropy(counts: torch.Tensor) -> float:
    probabilities = counts[counts > 0].double() / counts.sum().clamp_min(1)
    return float(-(probabilities * probabilities.log()).sum())


def _mutual_information(first: torch.Tensor, second: torch.Tensor) -> float:
    """Mutual information in nats of two discrete vectors."""

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
    """Type every unit against the hidden labels: attribute, class or untyped. Evaluation only.

    ``u_attribute = I(P; a | y) / H(P)`` and ``u_class = I(P; y | a) / H(P)`` for the presence indicator
    ``P`` of a unit. A unit is an attribute unit when ``u_attribute`` reaches ``threshold`` and ``ratio``
    times ``u_class``, and a class unit in the mirrored case.
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


def auroc(scores: list[float], positive: list[bool]) -> float:
    """Probability that a positive scores above a negative, ties counted half; NaN without both kinds."""

    positives = [score for score, flag in zip(scores, positive) if flag]
    negatives = [score for score, flag in zip(scores, positive) if not flag]
    if not positives or not negatives:
        return math.nan
    wins = sum(1.0 if high > low else 0.5 if high == low else 0.0 for high in positives for low in negatives)
    return wins / (len(positives) * len(negatives))


def detection_auroc(onset: list[float], units: list[dict], typing: list[dict]) -> float:
    """AUROC of the onset score for attribute units against class units, over the audited units."""

    kept = [entry["unit"] for entry in units if entry["audited"] and typing[entry["unit"]]["type"] != "untyped"]
    return auroc([onset[unit] for unit in kept], [typing[unit]["type"] == "attribute" for unit in kept])


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda position: values[position])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in order[start:end + 1]:
            ranks[position] = (start + end) / 2.0
        start = end + 1
    return ranks


def spearman(first: list[float], second: list[float]) -> float:
    """Rank correlation of two score lists; NaN when one of them is constant."""

    a, b = _ranks(first), _ranks(second)
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    spread = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return covariance / spread if spread > 0 else math.nan
