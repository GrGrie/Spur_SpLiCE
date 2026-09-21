"""Collect grouping and teacher-graph metrics into one compact JSON record.

The record synchronizes through Git (outputs/reports/cospro_diagnostics/) and feeds the
dashboard on any machine. Metrics that need the SpLiCE cache or hidden labels are included
only when those inputs are given.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cospro.diagnostics import graph_metrics, group_metrics
from cospro.diagnostics.labels import SampleLabels, load_labels
from splice.cospro import validate_splice_dataset_cache
from cospro.pipeline.graph_io import load_graph_json

SCHEMA = "cospro-diagnostics-v1"
SWEEP_SELECTIVITY_ROWS = 25


def discover_sweep(sweep_dir: Path) -> list[dict[str, Any]]:
    """Grouping configurations below ``sweep_dir`` with the teacher graphs built from each."""

    points = []
    for groups_path in sorted(sweep_dir.glob("*/concept_groups.json")):
        points.append({
            "name": groups_path.parent.name,
            "groups": groups_path,
            "graphs": sorted(groups_path.parent.glob("teacher_graphs/*/teacher_graph.json")),
        })
    return points


def load_cache(path: Path) -> dict:
    return validate_splice_dataset_cache(torch.load(path, map_location="cpu", weights_only=True))


def _check_alignment(sample_ids: list[str], cache: dict | None, what: str) -> None:
    if cache is not None and [str(value) for value in cache["sample_ids"]] != sample_ids:
        raise ValueError(f"{what} and the SpLiCE cache cover different samples or orders.")


def audited_group_selectivity(graph: dict, cache: dict, labels: tuple[np.ndarray, np.ndarray]) -> list[dict]:
    """Post-hoc AUCs of every audited group together with its selection outcome."""

    groups_like = {"groups": [
        {"concept_indices": group["concept_indices"], "concepts": group.get("concepts", [])}
        for group in graph.get("groups", [])
    ]}
    rows = group_metrics.spurious_selectivity(groups_like, cache["splice_codes"], *labels)["groups"]
    audited = graph.get("groups", [])
    for row in rows:
        group = audited[row["group_id"]]
        row.update({
            "selected": bool(group.get("selected")),
            "null_margin": float(group["score"]) / float(group["null_threshold"])
            if float(group.get("null_threshold", 0.0)) > 0 else None,
            "null_excess_score": float(group.get("null_excess_score", 0.0)),
        })
    return rows


def evaluate(
    dataset: str,
    *,
    sweep_dir: Path | None = None,
    graphs: dict[str, Path] | None = None,
    reference_graph: str | None = None,
    cache_path: Path | None = None,
    data_folder: Path | None = None,
    bootstrap_trials: int = 0,
) -> dict[str, Any]:
    cache = load_cache(cache_path) if cache_path else None
    sample_labels: SampleLabels | None = load_labels(dataset, data_folder) if data_folder else None
    named = {name: load_graph_json(path) for name, path in (graphs or {}).items()}
    reference = named.get(reference_graph) if reference_graph else None

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "dataset": dataset,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "sweep_dir": str(sweep_dir) if sweep_dir else None,
            "graphs": {name: str(path) for name, path in (graphs or {}).items()},
            "reference_graph": reference_graph,
            "splice_dataset_cache": str(cache_path) if cache_path else None,
        },
        "tiers": {"cache": cache is not None, "labels": sample_labels is not None},
        "group_names": sample_labels.group_names() if sample_labels else None,
        "attribute_count": sample_labels.n_attributes if sample_labels else None,
        "graphs": {},
        "sweep": [],
    }

    for name, graph in named.items():
        sample_ids = [str(value) for value in graph["sample_ids"]]
        labels = sample_labels.for_ids(sample_ids) if sample_labels else None
        others = {other: graph_b for other, graph_b in named.items() if other != name}
        entry = graph_metrics.describe_graph(graph, labels, references=others)
        if cache is not None and labels is not None and graph.get("groups"):
            _check_alignment(sample_ids, cache, f"Graph {name}")
            entry["audited_groups"] = audited_group_selectivity(graph, cache, labels)
        record["graphs"][name] = entry

    for point in discover_sweep(sweep_dir) if sweep_dir else []:
        groups = json.loads(point["groups"].read_text(encoding="utf-8"))
        sample_ids = [str(value) for value in groups["sample_ids"]]
        _check_alignment(sample_ids, cache, f"Concept groups {point['name']}")
        labels = sample_labels.for_ids(sample_ids) if sample_labels else None
        grouping = group_metrics.describe_grouping(groups, cache, labels, bootstrap_trials=bootstrap_trials)
        if "spurious" in grouping:
            grouping["spurious"]["groups"] = grouping["spurious"]["groups"][:SWEEP_SELECTIVITY_ROWS]
        point_graphs = []
        for graph_path in point["graphs"]:
            graph = load_graph_json(graph_path)
            graph_labels = sample_labels.for_ids([str(v) for v in graph["sample_ids"]]) if sample_labels else None
            entry = {
                "audit": graph_path.parent.name,
                "structure": graph_metrics.structure_metrics(graph),
                "selection": graph_metrics.selection_metrics(graph),
            }
            if reference is not None:
                entry["overlap_with_reference"] = graph_metrics.edge_jaccard(graph, reference)
            if graph_labels is not None:
                entry["labels"] = graph_metrics.label_metrics(graph, *graph_labels)
            point_graphs.append(entry)
        record["sweep"].append({"name": point["name"], "grouping": grouping, "graphs": point_graphs})
    return record


def _json_ready(value: Any) -> Any:
    """Replace non-finite floats with None so the record stays strict JSON."""

    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_record(record: dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload = json.dumps(_json_ready(record), indent=1, sort_keys=True, allow_nan=False)
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(output)
    return output
