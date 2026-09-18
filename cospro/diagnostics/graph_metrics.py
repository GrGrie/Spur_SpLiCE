"""Teacher-graph quality metrics.

Label-free metrics describe graph shape, overlap with other graphs and the null-test selection.
Post-hoc metrics use ``y`` and ``a`` to measure how often an edge keeps the class while it flips
the spurious attribute: the counterfactual edge that relational distillation is meant to teach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class EdgeTable:
    """Directed teacher edges; ``anchors`` and ``neighbours`` index the graph's sample IDs."""

    anchors: np.ndarray
    neighbours: np.ndarray
    weights: np.ndarray
    confidences: np.ndarray
    group_ids: np.ndarray
    n_samples: int


def edge_table(graph: dict) -> EdgeTable:
    indices = torch.as_tensor(graph["neighbor_indices"]).long().numpy()
    weights = torch.as_tensor(graph["weights"]).float().numpy()
    confidences = torch.as_tensor(graph.get("edge_confidences", graph["weights"])).float().numpy()
    group_ids = torch.as_tensor(graph.get("group_ids", np.full(indices.shape, -1))).long().numpy()
    valid = indices >= 0
    anchors = np.broadcast_to(np.arange(indices.shape[0])[:, None], indices.shape)
    return EdgeTable(
        anchors=anchors[valid],
        neighbours=indices[valid],
        weights=weights[valid],
        confidences=confidences[valid],
        group_ids=group_ids[valid],
        n_samples=indices.shape[0],
    )


def _gini(values: np.ndarray) -> float:
    values = np.sort(values.astype(np.float64))
    if values.sum() <= 0:
        return 0.0
    ranks = np.arange(1, len(values) + 1)
    return float((2 * (ranks * values).sum()) / (len(values) * values.sum()) - (len(values) + 1) / len(values))


def structure_metrics(graph: dict) -> dict[str, float]:
    """Edge count, anchor coverage and in-degree concentration, recomputed for any graph type."""

    edges = edge_table(graph)
    indegree = np.bincount(edges.neighbours, minlength=edges.n_samples)
    covered = np.zeros(edges.n_samples, dtype=bool)
    covered[edges.anchors] = True
    squared = float((indegree.astype(np.float64) ** 2).sum())
    return {
        "samples": edges.n_samples,
        "edges": int(len(edges.anchors)),
        "coverage": float(covered.mean()),
        "max_indegree": int(indegree.max(initial=0)),
        "indegree_gini": _gini(indegree),
        "effective_donor_count": float(indegree.sum() ** 2 / squared) if squared else 0.0,
    }


def edge_jaccard(graph_a: dict, graph_b: dict) -> float:
    """Jaccard overlap of the undirected edge sets of two graphs over the same samples."""

    if list(graph_a["sample_ids"]) != list(graph_b["sample_ids"]):
        raise ValueError("Edge overlap needs two graphs over identical sample IDs.")

    def edge_set(graph: dict) -> set[tuple[int, int]]:
        edges = edge_table(graph)
        return {tuple(sorted(pair)) for pair in zip(edges.anchors.tolist(), edges.neighbours.tolist())}

    left, right = edge_set(graph_a), edge_set(graph_b)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def selection_metrics(graph: dict) -> dict[str, Any]:
    """How many audited concept groups pass the null test and by what margin."""

    groups = list(graph.get("groups", []))
    selected = [group for group in groups if group.get("selected")]
    margins = [
        float(group["score"]) / float(group["null_threshold"])
        for group in selected
        if float(group.get("null_threshold", 0.0)) > 0
    ]
    return {
        "audited_groups": len(groups),
        "selected_groups": len(selected),
        "selected_fraction": len(selected) / len(groups) if groups else 0.0,
        "selected_composite_groups": sum(len(group.get("concepts", [])) > 1 for group in selected),
        "median_null_margin": float(np.median(margins)) if margins else None,
    }


def _label_rates(
    anchors: np.ndarray, neighbours: np.ndarray, weights: np.ndarray, y: np.ndarray, a: np.ndarray,
) -> dict[str, float]:
    total = weights.sum()
    if total <= 0:
        return {"same_class": 0.0, "flipped_attribute": 0.0, "counterfactual": 0.0}
    same_class = y[anchors] == y[neighbours]
    flipped = a[anchors] != a[neighbours]
    return {
        "same_class": float(weights[same_class].sum() / total),
        "flipped_attribute": float(weights[flipped].sum() / total),
        "counterfactual": float(weights[same_class & flipped].sum() / total),
    }


def _transition(
    anchors: np.ndarray, neighbours: np.ndarray, weights: np.ndarray, groups: np.ndarray, n_groups: int,
) -> np.ndarray:
    matrix = np.zeros((n_groups, n_groups))
    np.add.at(matrix, (groups[anchors], groups[neighbours]), weights)
    return matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-12, None)


def _group_balanced_counterfactual(transition: np.ndarray, n_attributes: int, *, flip: bool = True) -> float:
    """Mean over anchor groups of the edge mass that keeps the class and, with ``flip``, flips the attribute."""

    rates = []
    for source in range(transition.shape[0]):
        if transition[source].sum() <= 0:
            continue
        label, attribute = divmod(source, n_attributes)
        targets = [
            label * n_attributes + other for other in range(n_attributes) if other != attribute or not flip
        ]
        rates.append(transition[source, targets].sum())
    return float(np.mean(rates)) if rates else 0.0


def label_metrics(
    graph: dict,
    y: np.ndarray,
    a: np.ndarray,
    *,
    random_trials: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Post-hoc edge semantics. ``y`` and ``a`` are aligned with the graph's sample IDs."""

    edges = edge_table(graph)
    n_attributes = int(a.max()) + 1
    groups = y * n_attributes + a
    n_groups = (int(y.max()) + 1) * n_attributes
    weights = edges.weights.astype(np.float64)
    transition = _transition(edges.anchors, edges.neighbours, weights, groups, n_groups)
    rates = _label_rates(edges.anchors, edges.neighbours, weights, y, a)
    balanced = _group_balanced_counterfactual(transition, n_attributes)

    # Degree-matched random baseline: every anchor keeps its edge weights, neighbours are redrawn.
    # Random neighbours of a minority anchor mostly come from the majority group of the same class,
    # so the random counterfactual rate is high while its class consistency is low. Read both
    # together and compare graphs against the raw-CLIP kNN baseline built at the same degree.
    generator = np.random.default_rng(seed)
    random_balanced, random_class, random_rate = [], [], []
    for _ in range(random_trials):
        shuffled = generator.integers(0, edges.n_samples, size=len(edges.neighbours))
        random_rate.append(_label_rates(edges.anchors, shuffled, weights, y, a)["counterfactual"])
        random_transition = _transition(edges.anchors, shuffled, weights, groups, n_groups)
        random_balanced.append(_group_balanced_counterfactual(random_transition, n_attributes))
        random_class.append(_group_balanced_counterfactual(random_transition, n_attributes, flip=False))

    counts = np.bincount(groups, minlength=n_groups)
    # Minority groups hold less than half of an even share of the samples.
    minority = (counts > 0) & (counts < 0.5 * counts[counts > 0].mean())
    is_counterfactual = (y[edges.anchors] == y[edges.neighbours]) & (a[edges.anchors] != a[edges.neighbours])
    reached = np.zeros(edges.n_samples, dtype=bool)
    reached[edges.anchors[is_counterfactual]] = True
    minority_anchor = minority[groups]

    return {
        **rates,
        "group_balanced_counterfactual": balanced,
        "group_balanced_same_class": _group_balanced_counterfactual(transition, n_attributes, flip=False),
        "random_group_balanced_counterfactual": float(np.mean(random_balanced)),
        "random_group_balanced_same_class": float(np.mean(random_class)),
        "random_counterfactual": float(np.mean(random_rate)),
        "minority_groups": [int(index) for index in np.flatnonzero(minority)],
        "minority_reach": float(reached[minority_anchor].mean()) if minority_anchor.any() else 0.0,
        "transition": transition.tolist(),
        "group_counts": counts.tolist(),
        "calibration": confidence_calibration(edges, is_counterfactual),
    }


def confidence_calibration(edges: EdgeTable, is_counterfactual: np.ndarray, bins: int = 10) -> list[dict]:
    """Counterfactual edge rate per edge-confidence decile."""

    if len(edges.confidences) == 0:
        return []
    edges_per_bin = np.array_split(np.argsort(edges.confidences, kind="stable"), bins)
    return [
        {
            "bin": index,
            "confidence_mean": float(edges.confidences[members].mean()),
            "counterfactual_rate": float(is_counterfactual[members].mean()),
            "edges": int(len(members)),
        }
        for index, members in enumerate(edges_per_bin)
        if len(members)
    ]


def per_concept_group_label_rates(graph: dict, y: np.ndarray, a: np.ndarray) -> list[dict[str, Any]]:
    """Post-hoc edge semantics split by the concept group that produced each edge."""

    edges = edge_table(graph)
    names = {int(group["group_id"]): list(group.get("concepts", [])) for group in graph.get("groups", [])}
    rows = []
    for group_id in sorted(set(edges.group_ids.tolist())):
        members = edges.group_ids == group_id
        rows.append({
            "group_id": int(group_id),
            "concepts": names.get(int(group_id), []),
            "edges": int(members.sum()),
            **_label_rates(edges.anchors[members], edges.neighbours[members],
                           edges.weights[members].astype(np.float64), y, a),
        })
    return sorted(rows, key=lambda row: -row["edges"])


def edge_samples(
    graph: dict, y: np.ndarray | None, a: np.ndarray | None, *, per_group: int = 4, max_groups: int = 12, seed: int = 0,
) -> list[dict[str, Any]]:
    """A reproducible edge sample per concept group for the dashboard gallery.

    Half of each group's slots show its highest-confidence edges, the edges the relational loss
    weights most. With labels, the other half shows edges from minority-group anchors, where a
    counterfactual edge matters most for worst-group accuracy; without labels it is a random draw.
    """

    edges = edge_table(graph)
    names = {int(group["group_id"]): list(group.get("concepts", [])) for group in graph.get("groups", [])}
    sample_ids = [str(value) for value in graph["sample_ids"]]
    generator = np.random.default_rng(seed)
    minority_anchor = np.zeros(edges.n_samples, dtype=bool)
    if y is not None and a is not None:
        joint = y * (int(a.max()) + 1) + a
        counts = np.bincount(joint)
        minority_anchor = ((counts > 0) & (counts < 0.5 * counts[counts > 0].mean()))[joint]
    group_sizes = sorted(
        ((int(group_id), int((edges.group_ids == group_id).sum())) for group_id in set(edges.group_ids.tolist())),
        key=lambda item: -item[1],
    )[:max_groups]
    samples = []
    for group_id, _ in group_sizes:
        members = np.flatnonzero(edges.group_ids == group_id)
        top = members[np.argsort(-edges.confidences[members], kind="stable")][: per_group // 2]
        rest = np.setdiff1d(members, top)
        pool = rest[minority_anchor[edges.anchors[rest]]] if minority_anchor.any() else rest
        pool = pool if len(pool) else rest
        extra = generator.choice(pool, size=min(per_group - len(top), len(pool)), replace=False) if len(pool) else []
        second_reason = "minority anchor" if minority_anchor.any() else "random"
        chosen = [("highest confidence", position) for position in top]
        chosen += [(second_reason, position) for position in extra]
        for reason, position in chosen:
            anchor, neighbour = int(edges.anchors[position]), int(edges.neighbours[position])
            row = {
                "group_id": group_id,
                "concepts": names.get(group_id, []),
                "anchor": sample_ids[anchor],
                "neighbour": sample_ids[neighbour],
                "confidence": float(edges.confidences[position]),
                "reason": reason,
            }
            if y is not None and a is not None:
                row.update({
                    "anchor_labels": [int(y[anchor]), int(a[anchor])],
                    "neighbour_labels": [int(y[neighbour]), int(a[neighbour])],
                    "counterfactual": bool(y[anchor] == y[neighbour] and a[anchor] != a[neighbour]),
                })
            samples.append(row)
    return samples


def describe_graph(
    graph: dict,
    labels: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    references: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """All graph metrics in one JSON-ready record."""

    record: dict[str, Any] = {
        "artifact": graph.get("artifact"),
        "structure": structure_metrics(graph),
        "selection": selection_metrics(graph),
        "overlap": {name: edge_jaccard(graph, other) for name, other in (references or {}).items()},
    }
    if labels is not None:
        y, a = labels
        record["labels"] = label_metrics(graph, y, a)
        record["concept_groups"] = per_concept_group_label_rates(graph, y, a)
    record["edge_samples"] = edge_samples(graph, *(labels or (None, None)))
    return record


def aligned_labels(graph: dict, sample_labels) -> tuple[np.ndarray, np.ndarray]:
    return sample_labels.for_ids(list(graph["sample_ids"]))


__all__: Sequence[str] = (
    "EdgeTable", "edge_table", "structure_metrics", "edge_jaccard", "selection_metrics", "label_metrics",
    "confidence_calibration", "per_concept_group_label_rates", "edge_samples", "describe_graph", "aligned_labels",
)
