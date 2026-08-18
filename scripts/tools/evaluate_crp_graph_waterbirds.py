"""Post-hoc Waterbirds check for a label-free CRP teacher graph.

The graph is built without annotations.  This script is intentionally separate
and may read ``y`` and ``place`` only to measure whether the frozen graph found
the intended relations after the fact.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.metadata_csv.open(newline="", encoding="utf-8")))
    graph = torch.load(args.graph, map_location="cpu", weights_only=True)
    sample_ids = [str(value) for value in graph["sample_ids"]]
    indices = graph["neighbor_indices"]
    weights = graph["weights"]
    if len(rows) <= max(int(sample_id.rsplit(":", 1)[1]) for sample_id in sample_ids):
        raise ValueError("metadata.csv does not contain all cached source indices")

    y = {int(index): int(rows[index]["y"]) for index in range(len(rows))}
    a = {int(index): int(rows[index]["place"]) for index in range(len(rows))}
    source_indices = [int(sample_id.rsplit(":", 1)[1]) for sample_id in sample_ids]
    valid = weights > 0
    edge_rows, edge_positions = torch.where(valid)
    same_target = 0
    same_target_opposite_spurious = 0
    support_by_group = defaultdict(lambda: [0, 0])
    for row, position in zip(edge_rows.tolist(), edge_positions.tolist()):
        source = source_indices[row]
        destination = source_indices[int(indices[row, position])]
        same_y = y[source] == y[destination]
        same_target += int(same_y)
        same_target_opposite_spurious += int(same_y and a[source] != a[destination])
    supported = valid.any(dim=1)
    for row, source in enumerate(source_indices):
        key = (y[source], a[source])
        support_by_group[key][1] += 1
        support_by_group[key][0] += int(supported[row])

    edge_count = len(edge_rows)
    precision = same_target / edge_count if edge_count else 0.0
    opposite_rate = (
        same_target_opposite_spurious / same_target if same_target else 0.0
    )
    print(f"graph={args.graph} edges={edge_count}")
    print(f"same_target_precision={precision:.4f}")
    print(f"same_target_opposite_spurious_rate={opposite_rate:.4f}")
    for key in sorted(support_by_group):
        supported_count, total_count = support_by_group[key]
        print(f"coverage_y{key[0]}_a{key[1]}={supported_count / total_count:.4f} ({supported_count}/{total_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
