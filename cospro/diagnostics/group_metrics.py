"""Concept-group quality metrics.

Structure and cross-configuration agreement need only the concept-group artifact. Coherence,
coverage and stability also read the frozen SpLiCE cache (activation codes and text directions).
Spurious selectivity and fragmentation are post-hoc and read ``y`` and ``a``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, mutual_info_score, roc_auc_score, silhouette_score

from splice.cospro import CoSpRoAuditConfig, build_concept_groups


def _concept_lists(groups_artifact: dict) -> list[list[int]]:
    return [[int(index) for index in group["concept_indices"]] for group in groups_artifact["groups"]]


def structure_metrics(groups_artifact: dict) -> dict[str, Any]:
    sizes = np.array([len(group) for group in _concept_lists(groups_artifact)], dtype=np.float64)
    if sizes.size == 0:
        return {"groups": 0, "active_concepts": 0}
    shares = sizes / sizes.sum()
    return {
        "groups": int(sizes.size),
        "active_concepts": int(sizes.sum()),
        "composite_groups": int((sizes > 1).sum()),
        "singleton_fraction": float((sizes == 1).mean()),
        "concepts_in_composite_groups": int(sizes[sizes > 1].sum()),
        "max_group_size": int(sizes.max()),
        "size_entropy": float(-(shares * np.log(shares)).sum()),
    }


def partition_agreement(groups_a: dict, groups_b: dict) -> dict[str, float]:
    """Adjusted Rand index and variation of information over the concepts both groupings keep."""

    def labels(artifact: dict) -> dict[int, int]:
        return {concept: group for group, concepts in enumerate(_concept_lists(artifact)) for concept in concepts}

    left, right = labels(groups_a), labels(groups_b)
    shared = sorted(left.keys() & right.keys())
    if len(shared) < 2:
        return {"shared_concepts": len(shared), "adjusted_rand_index": float("nan"), "variation_of_information": float("nan")}
    left_labels = [left[concept] for concept in shared]
    right_labels = [right[concept] for concept in shared]

    def entropy(values: Sequence[int]) -> float:
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts / counts.sum()
        return float(-(probabilities * np.log(probabilities)).sum())

    mutual_information = mutual_info_score(left_labels, right_labels)
    return {
        "shared_concepts": len(shared),
        "adjusted_rand_index": float(adjusted_rand_score(left_labels, right_labels)),
        "variation_of_information": entropy(left_labels) + entropy(right_labels) - 2 * mutual_information,
    }


def _composite(groups_artifact: dict) -> list[list[int]]:
    return [group for group in _concept_lists(groups_artifact) if len(group) > 1]


def text_coherence(groups_artifact: dict, dictionary: torch.Tensor) -> dict[str, Any]:
    """Mean pairwise cosine of concept text directions inside each composite group."""

    directions = torch.nn.functional.normalize(dictionary.float(), dim=1)
    per_group = []
    for group in _composite(groups_artifact):
        similarity = directions[group] @ directions[group].T
        upper = similarity[torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)]
        per_group.append(float(upper.mean()))
    return {"mean": float(np.mean(per_group)) if per_group else None, "per_group": per_group}


def npmi_coherence(groups_artifact: dict, codes: torch.Tensor) -> dict[str, Any]:
    """Mean pairwise normalized PMI of binary concept activations inside each composite group."""

    active = codes > 0
    n_images = active.shape[0]
    per_group = []
    for group in _composite(groups_artifact):
        columns = active[:, group].float()
        joint = (columns.T @ columns) / n_images
        marginal = columns.mean(dim=0)
        values = []
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                p_joint = float(joint[left, right])
                if p_joint <= 0:
                    values.append(-1.0)
                elif p_joint >= 1:
                    values.append(1.0)
                else:
                    pmi = math.log(p_joint / (float(marginal[left]) * float(marginal[right])))
                    values.append(pmi / -math.log(p_joint))
        per_group.append(float(np.mean(values)))
    return {"mean": float(np.mean(per_group)) if per_group else None, "per_group": per_group}


def text_separation(groups_artifact: dict, dictionary: torch.Tensor) -> float | None:
    """Silhouette of composite groups on text directions with cosine distance."""

    composite = _composite(groups_artifact)
    if len(composite) < 2:
        return None
    concepts = [concept for group in composite for concept in group]
    labels = [index for index, group in enumerate(composite) for _ in group]
    directions = torch.nn.functional.normalize(dictionary[concepts].float(), dim=1).numpy()
    return float(silhouette_score(directions, labels, metric="cosine"))


def image_coverage(groups_artifact: dict, codes: torch.Tensor) -> dict[str, float]:
    """Share of images where a composite group, or any group, is active."""

    def covered(groups: list[list[int]]) -> float:
        if not groups:
            return 0.0
        concepts = sorted({concept for group in groups for concept in group})
        return float((codes[:, concepts] > 0).any(dim=1).float().mean())

    return {"composite_groups": covered(_composite(groups_artifact)), "all_groups": covered(_concept_lists(groups_artifact))}


def bootstrap_stability(
    cache: dict, config: CoSpRoAuditConfig, reference: dict, *, trials: int = 5, fraction: float = 0.8, seed: int = 0,
) -> dict[str, Any]:
    """Adjusted Rand index between the full grouping and groupings of random image subsets."""

    generator = torch.Generator().manual_seed(seed)
    n_images = len(cache["sample_ids"])
    scores = []
    for _ in range(trials):
        subset = torch.randperm(n_images, generator=generator)[: int(round(fraction * n_images))].sort().values
        sub_cache = {
            **cache,
            "sample_ids": [cache["sample_ids"][index] for index in subset.tolist()],
            "clip_embeddings": cache["clip_embeddings"][subset],
            "splice_codes": cache["splice_codes"][subset],
        }
        sub_cache.pop("centered_clip", None)
        scores.append(partition_agreement(reference, build_concept_groups(sub_cache, config))["adjusted_rand_index"])
    return {"mean_adjusted_rand_index": float(np.nanmean(scores)), "trials": scores, "fraction": fraction}


def spurious_selectivity(
    groups_artifact: dict, codes: torch.Tensor, y: np.ndarray, a: np.ndarray,
) -> dict[str, Any]:
    """Post-hoc: how well each group's activation predicts ``a`` compared with ``y``.

    Selectivity is ``|AUC_a - 0.5| - |AUC_y - 0.5|``. Fragmentation counts the groups needed to
    reach 80% of the total positive selectivity; a low count means the spurious signal sits in few groups.
    """

    rows = []
    for group_id, group in enumerate(_concept_lists(groups_artifact)):
        activation = codes[:, group].sum(dim=1).numpy()
        if activation.max() <= activation.min():
            continue
        auc_y = float(roc_auc_score(y, activation)) if len(np.unique(y)) == 2 else float("nan")
        auc_a = float(roc_auc_score(a, activation)) if len(np.unique(a)) == 2 else float("nan")
        rows.append({
            "group_id": group_id,
            "concepts": list(groups_artifact["groups"][group_id]["concepts"]),
            "size": len(group),
            "frequency": float((activation > 0).mean()),
            "auc_y": auc_y,
            "auc_a": auc_a,
            "selectivity": abs(auc_a - 0.5) - abs(auc_y - 0.5),
        })
    rows.sort(key=lambda row: -row["selectivity"])
    positive = np.array([row["selectivity"] for row in rows if row["selectivity"] > 0])
    fragmentation = None
    if positive.size:
        cumulative = np.cumsum(np.sort(positive)[::-1]) / positive.sum()
        fragmentation = int(np.searchsorted(cumulative, 0.8) + 1)
    return {"fragmentation": fragmentation, "positive_groups": int(positive.size), "groups": rows}


def describe_grouping(
    groups_artifact: dict,
    cache: dict | None = None,
    labels: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    bootstrap_trials: int = 0,
) -> dict[str, Any]:
    """All grouping metrics the available inputs allow, in one JSON-ready record."""

    record: dict[str, Any] = {"config": dict(groups_artifact.get("config", {})), "structure": structure_metrics(groups_artifact)}
    if cache is None:
        return record
    codes, dictionary = cache["splice_codes"], cache["dictionary"]
    record.update({
        "text_coherence": text_coherence(groups_artifact, dictionary)["mean"],
        "npmi": npmi_coherence(groups_artifact, codes)["mean"],
        "text_separation": text_separation(groups_artifact, dictionary),
        "image_coverage": image_coverage(groups_artifact, codes),
    })
    if bootstrap_trials:
        config = replace(CoSpRoAuditConfig(), **{
            key: value for key, value in groups_artifact["config"].items() if key in CoSpRoAuditConfig.__dataclass_fields__
        })
        record["bootstrap"] = bootstrap_stability(cache, config, groups_artifact, trials=bootstrap_trials)
    if labels is not None:
        record["spurious"] = spurious_selectivity(groups_artifact, codes, *labels)
    return record
