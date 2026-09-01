"""Post-hoc oracle audit for a label-free CQT teacher graph.

This diagnostic is intentionally separate from graph discovery and SSL training.
It may read target and spurious annotations, but never writes them back into the
teacher graph.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch

from splice.crp_training import validate_teacher_graph
from splice.graph_io import load_graph_json


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _factor_concepts(graph: dict) -> dict[int, dict]:
    cards = {}
    for factor in graph.get("factors", []):
        factor_id = int(factor["factor_id"])
        state_a = [str(value) for value in factor.get("state_a", {}).get("concepts", [])]
        state_b = [str(value) for value in factor.get("state_b", {}).get("concepts", [])]
        cards[factor_id] = {
            "factor_id": factor_id,
            "state_a_concepts": state_a,
            "state_b_concepts": state_b,
            "removed_concepts": sorted(set(state_a + state_b)),
            "preserved_concepts": [str(value) for value in factor.get("preserved_concepts", [])],
        }
    return cards


def _edge_metrics(edges: Sequence[dict]) -> dict:
    total_weight = sum(float(edge["weight"]) for edge in edges)

    def count(key: str) -> int:
        return sum(bool(edge[key]) for edge in edges)

    def weighted(key: str) -> float:
        return sum(float(edge["weight"]) for edge in edges if edge[key])

    edge_count = len(edges)
    same_target = count("same_target")
    opposite_spurious = count("opposite_spurious")
    desired = count("desired_relation")
    harmful = count("harmful_relation")
    return {
        "edge_count": edge_count,
        "edge_weight": total_weight,
        "same_target_rate": _safe_ratio(same_target, edge_count),
        "different_target_rate": _safe_ratio(harmful, edge_count),
        "opposite_spurious_rate": _safe_ratio(opposite_spurious, edge_count),
        "desired_relation_rate": _safe_ratio(desired, edge_count),
        "desired_given_same_target_rate": _safe_ratio(desired, same_target),
        "weighted_same_target_rate": _safe_ratio(weighted("same_target"), total_weight),
        "weighted_different_target_rate": _safe_ratio(weighted("harmful_relation"), total_weight),
        "weighted_desired_relation_rate": _safe_ratio(weighted("desired_relation"), total_weight),
    }


def audit_graph(
    graph: dict,
    metadata_rows: Sequence[dict],
    target_column: str = "y",
    spurious_column: str = "place",
) -> dict:
    """Return edge-, factor-, and subgroup-level oracle diagnostics."""

    graph = validate_teacher_graph(graph)
    sample_ids = [str(value) for value in graph["sample_ids"]]
    source_indices = [int(sample_id.rsplit(":", 1)[1]) for sample_id in sample_ids]
    if source_indices and max(source_indices) >= len(metadata_rows):
        raise ValueError("Metadata does not contain every source index referenced by the graph.")
    for column in (target_column, spurious_column):
        if metadata_rows and column not in metadata_rows[0]:
            raise ValueError(f"Metadata is missing required column {column!r}.")

    targets = [int(row[target_column]) for row in metadata_rows]
    spurious = [int(row[spurious_column]) for row in metadata_rows]
    neighbor_indices = graph["neighbor_indices"]
    weights = graph["weights"]
    group_ids = torch.as_tensor(
        graph.get("group_ids", torch.full_like(neighbor_indices, -1)), dtype=torch.long
    )
    gains = torch.as_tensor(
        graph.get("intervention_gains", torch.zeros_like(weights)), dtype=torch.float32
    )
    factor_cards = _factor_concepts(graph)

    edge_rows, edge_positions = torch.where(weights > 0)
    edges = []
    edges_by_factor: dict[int, list[dict]] = defaultdict(list)
    for row, position in zip(edge_rows.tolist(), edge_positions.tolist()):
        destination_row = int(neighbor_indices[row, position])
        source_index = source_indices[row]
        destination_index = source_indices[destination_row]
        same_target = targets[source_index] == targets[destination_index]
        opposite_spurious = spurious[source_index] != spurious[destination_index]
        factor_id = int(group_ids[row, position])
        edge = {
            "source_sample_id": sample_ids[row],
            "destination_sample_id": sample_ids[destination_row],
            "factor_id": factor_id,
            "weight": float(weights[row, position]),
            "intervention_gain": float(gains[row, position]),
            "same_target": same_target,
            "opposite_spurious": opposite_spurious,
            "desired_relation": same_target and opposite_spurious,
            "harmful_relation": not same_target,
        }
        edges.append(edge)
        edges_by_factor[factor_id].append(edge)

    supported = weights.sum(dim=1) > 0
    subgroup_support: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    for graph_row, source_index in enumerate(source_indices):
        key = (targets[source_index], spurious[source_index])
        subgroup_support[key][1] += 1
        subgroup_support[key][0] += int(supported[graph_row])

    factors = []
    for factor_id in sorted(edges_by_factor):
        factors.append(
            {
                **factor_cards.get(
                    factor_id,
                    {
                        "factor_id": factor_id,
                        "state_a_concepts": [],
                        "state_b_concepts": [],
                        "removed_concepts": [],
                        "preserved_concepts": [],
                    },
                ),
                **_edge_metrics(edges_by_factor[factor_id]),
            }
        )

    return {
        "artifact": "splice_cqt_posthoc_oracle_audit_v1",
        "teacher_artifact": graph["artifact"],
        "sample_count": len(sample_ids),
        "graph_metrics": {
            **_edge_metrics(edges),
            "supported_anchor_count": int(supported.sum()),
            "anchor_coverage": float(supported.float().mean()),
        },
        "removed_concepts": sorted(
            {concept for card in factor_cards.values() for concept in card["removed_concepts"]}
        ),
        "factors": factors,
        "subgroup_coverage": [
            {
                "target": target,
                "spurious": attribute,
                "supported": counts[0],
                "total": counts[1],
                "coverage": _safe_ratio(counts[0], counts[1]),
            }
            for (target, attribute), counts in sorted(subgroup_support.items())
        ],
        "edges": edges,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--spurious-column", default="place")
    args = parser.parse_args()

    metadata_rows = list(csv.DictReader(args.metadata_csv.open(newline="", encoding="utf-8")))
    payload = audit_graph(
        load_graph_json(args.graph),
        metadata_rows,
        target_column=args.target_column,
        spurious_column=args.spurious_column,
    )
    output_path = args.output or args.graph.with_name(f"{args.graph.stem}.oracle_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(output_path)
    print(
        f"[INFO] Oracle audit: desired={payload['graph_metrics']['desired_relation_rate']:.4f}, "
        f"harmful={payload['graph_metrics']['different_target_rate']:.4f}, output={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
