"""Post-hoc validity of concept factors: do the entangled pairs join a class factor to an attribute factor?

This module reads the hidden labels, so it evaluates a factor set and never shapes one. In the
training images the class ``y`` and the spurious attribute ``a`` are correlated, so plain association
with ``a`` would also flag every class concept. Each factor is scored by conditional information
instead, as an uncertainty coefficient in [0, 1]:

* ``u_attribute = I(F; a | y) / H(F)``: how much the factor's presence tells about the attribute
  among images of the same class. A line colour or a background concept scores high.
* ``u_class = I(F; y | a) / H(F)``: how much it tells about the class among images with the same
  attribute. A breed, a vehicle or an animal concept scores high.

A factor is ``attribute`` or ``class`` when its score reaches ``min_signal`` and exceeds the other
score ``dominance`` times; ``mixed`` when both reach ``min_signal`` otherwise; ``neither`` when none
does. A pair is ``cross`` when it joins a class factor to an attribute factor: the kind of pair the
concept-factor methods are meant to find. Pair precision is the share of cross pairs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FACTOR_TYPES = ("class", "attribute", "mixed", "neither")


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    values = -(probability * np.log2(probability) + (1 - probability) * np.log2(1 - probability))
    return np.where((probability <= 1e-12) | (probability >= 1 - 1e-12), 0.0, values)


def conditional_uncertainty(active: np.ndarray, target: np.ndarray, condition: np.ndarray) -> np.ndarray:
    """``I(F; target | condition) / H(F)`` for every binary presence column ``F`` of ``active``."""

    active = np.asarray(active, dtype=bool)
    if active.ndim == 1:
        return conditional_uncertainty(active[:, None], target, condition)[0]
    _, condition_ids = np.unique(condition, return_inverse=True)
    _, target_ids = np.unique(target, return_inverse=True)
    n_targets = target_ids.max() + 1
    cells = condition_ids * n_targets + target_ids
    n_cells = (condition_ids.max() + 1) * n_targets
    cell_totals = np.bincount(cells, minlength=n_cells).astype(np.float64)
    one_hot = np.zeros((len(cells), n_cells), dtype=np.float32)
    one_hot[np.arange(len(cells)), cells] = 1.0
    present = (one_hot.T @ active.astype(np.float32)).astype(np.float64)
    shape = (condition_ids.max() + 1, n_targets)
    present = present.reshape(*shape, -1)
    totals = cell_totals.reshape(shape)
    condition_totals = totals.sum(axis=1)
    condition_present = present.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        within_condition = binary_entropy(condition_present / np.maximum(condition_totals, 1)[:, None])
        within_cell = binary_entropy(present / np.maximum(totals, 1)[:, :, None])
        cell_weights = totals / np.maximum(condition_totals, 1)[:, None]
    remaining = (cell_weights[:, :, None] * within_cell).sum(axis=1)
    information = ((condition_totals / len(cells))[:, None] * (within_condition - remaining)).sum(axis=0)
    marginal = binary_entropy(active.mean(axis=0))
    return np.where(marginal > 0, np.maximum(information, 0.0) / np.maximum(marginal, 1e-12), 0.0)


def factor_type(u_class: float, u_attribute: float, min_signal: float, dominance: float) -> str:
    if u_attribute >= min_signal and u_attribute >= dominance * u_class:
        return "attribute"
    if u_class >= min_signal and u_class >= dominance * u_attribute:
        return "class"
    if u_class >= min_signal or u_attribute >= min_signal:
        return "mixed"
    return "neither"


def pair_verdict(first: str, second: str) -> str:
    kinds = {first, second}
    if kinds == {"class", "attribute"}:
        return "cross"
    if "neither" in kinds:
        return "noise"
    if "mixed" in kinds:
        return "mixed"
    return f"both {first}"


def diagnose_factors(active: np.ndarray, pairs: list[dict], names: list[str], y: np.ndarray, a: np.ndarray, *,
                     min_signal: float = 0.05, dominance: float = 2.0) -> dict[str, Any]:
    """Type every factor and pair of a factor set against the hidden ``y`` and ``a``."""

    active = np.asarray(active, dtype=bool)
    y, a = np.asarray(y), np.asarray(a)
    class_scores = conditional_uncertainty(active, y, a)
    attribute_scores = conditional_uncertainty(active, a, y)
    factors = []
    for column, name in enumerate(names):
        u_class, u_attribute = float(class_scores[column]), float(attribute_scores[column])
        factors.append({
            "factor": column,
            "name": name,
            "u_class": round(u_class, 4),
            "u_attribute": round(u_attribute, 4),
            "type": factor_type(u_class, u_attribute, min_signal, dominance),
        })
    typed_pairs = []
    for pair in pairs:
        first, second = pair["factors"]
        typed_pairs.append({
            "concepts": [names[first], names[second]],
            "phi": round(float(pair["phi"]), 4),
            "types": [factors[first]["type"], factors[second]["type"]],
            "verdict": pair_verdict(factors[first]["type"], factors[second]["type"]),
        })
    counts = {kind: sum(factor["type"] == kind for factor in factors) for kind in FACTOR_TYPES}
    attribute_ranked = sorted(factors, key=lambda factor: -factor["u_attribute"])
    cross = sum(pair["verdict"] == "cross" for pair in typed_pairs)
    return {
        "thresholds": {"min_signal": min_signal, "dominance": dominance},
        "summary": {
            "factor_count": len(factors),
            "type_counts": counts,
            "pair_count": len(typed_pairs),
            "cross_pairs": cross,
            "pair_precision": cross / len(typed_pairs) if typed_pairs else None,
            "best_attribute_signal": attribute_ranked[0]["u_attribute"] if attribute_ranked else 0.0,
            "top_attribute_factors": [
                {"name": factor["name"], "u_attribute": factor["u_attribute"], "u_class": factor["u_class"]}
                for factor in attribute_ranked[:5]
            ],
        },
        "pairs": typed_pairs,
        "factors": factors,
    }


def format_diagnosis(diagnosis: dict[str, Any]) -> str:
    summary = diagnosis["summary"]
    precision = summary["pair_precision"]
    lines = [
        "Post-hoc validity (hidden labels, evaluation only):",
        f"  factor types: {summary['type_counts']}",
        f"  cross pairs: {summary['cross_pairs']} of {summary['pair_count']}"
        + ("" if precision is None else f" (precision {precision:.2f})"),
    ]
    for pair in diagnosis["pairs"]:
        lines.append(f"    {pair['verdict']:>16}  {pair['phi']:.3f}  {pair['concepts'][0]}  <->  {pair['concepts'][1]}")
    lines.append("  strongest attribute factors (u_attribute, u_class):")
    for factor in summary["top_attribute_factors"]:
        lines.append(f"    {factor['u_attribute']:.3f}  {factor['u_class']:.3f}  {factor['name']}")
    return "\n".join(lines)
