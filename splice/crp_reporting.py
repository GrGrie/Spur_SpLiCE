"""Human-inspectable reports for the two frozen CRP construction stages."""

from __future__ import annotations

import statistics
from pathlib import Path

from splice.crp import validate_concept_groups
from splice.reporting import render_report


def _number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _percent(value: object) -> str:
    return f"{100.0 * float(value):.1f}%"


def render_concept_groups_report(artifact: dict, output: str | Path) -> Path:
    """Render grouping-only census information without CRP audit metrics."""

    artifact = validate_concept_groups(artifact)
    diagnostics = artifact["diagnostics"]
    composite_rows = [
        {
            "group_id": group["group_id"],
            "size": group["size"],
            "concepts": ", ".join(group["concepts"]),
            "concept_indices": ", ".join(str(value) for value in group["concept_indices"]),
        }
        for group in artifact["groups"]
        if group["size"] > 1
    ]
    largest_rows = [
        {
            "group_id": group["group_id"],
            "size": group["size"],
            "concepts": group["listing"],
        }
        for group in diagnostics["largest_composite_groups"]
    ]
    summary = (
        f"Active concepts: {diagnostics['total_active_concepts']}\n"
        f"Groups: {diagnostics['total_groups']}\n"
        f"Singletons: {diagnostics['singleton_count']} "
        f"({_percent(diagnostics['singleton_fraction'])})\n"
        f"Composite groups: {diagnostics['composite_group_count']}\n"
        f"Concepts in composites: {diagnostics['concepts_in_composite_groups']} "
        f"({_percent(diagnostics['concepts_in_composite_groups_fraction'])})"
    )
    columns = [
        {"key": "group_id", "label": "Group"},
        {"key": "size", "label": "Size"},
        {"key": "concepts", "label": "Concepts"},
    ]
    return render_report(
        "CRP concept-group census",
        [
            {"title": "Summary", "text": summary},
            {"title": "Grouping configuration", "data": artifact["config"]},
            {
                "title": "Group-size distribution",
                "data": {
                    "size_1": diagnostics["singleton_count"],
                    "size_2": diagnostics["groups_of_size_2"],
                    "size_3": diagnostics["groups_of_size_3"],
                    "size_4": diagnostics["groups_of_size_4"],
                    "size_5_plus": diagnostics["groups_of_size_5_plus"],
                    "mean": diagnostics["mean_group_size"],
                    "median": diagnostics["median_group_size"],
                    "maximum": diagnostics["maximum_group_size"],
                },
            },
            {"title": "Largest composite groups", "table": {"columns": columns, "rows": largest_rows}},
            {
                "title": "All composite groups",
                "table": {
                    "columns": columns + [{"key": "concept_indices", "label": "Vocabulary indices"}],
                    "rows": composite_rows,
                },
            },
        ],
        output,
    )


def _decision(group: dict, config: dict) -> str:
    if group.get("selected"):
        return "selected"
    if group.get("rejection_reason"):
        return f"rejected: {group['rejection_reason']}"
    if float(group.get("coverage", 0.0)) < float(config.get("min_coverage", 0.0)):
        return "rejected: coverage below minimum"
    return "rejected: audit score did not exceed null threshold"


def _edge_evidence_summary(graph: dict) -> dict:
    indices = graph["neighbor_indices"].tolist() if hasattr(graph["neighbor_indices"], "tolist") else graph["neighbor_indices"]
    confidences = graph.get("edge_confidences", [])
    confidences = confidences.tolist() if hasattr(confidences, "tolist") else confidences
    gains = graph.get("intervention_gains", [])
    gains = gains.tolist() if hasattr(gains, "tolist") else gains
    accepted_confidences = [
        float(confidences[row][column])
        for row in range(len(indices))
        for column in range(len(indices[row]))
        if indices[row][column] >= 0
    ] if confidences else []
    accepted_gains = [
        float(gains[row][column])
        for row in range(len(indices))
        for column in range(len(indices[row]))
        if indices[row][column] >= 0
    ] if gains else []

    def distribution(values: list[float]) -> dict:
        return {
            "count": len(values),
            "minimum": min(values, default=0.0),
            "mean": statistics.mean(values) if values else 0.0,
            "median": statistics.median(values) if values else 0.0,
            "maximum": max(values, default=0.0),
        }

    return {
        "retained_edge_confidence": distribution(accepted_confidences),
        "retained_intervention_gain": distribution(accepted_gains),
        "confidence_definition": "intervention gain × semantic evidence × null-excess ratio; row weights are normalized after edge selection",
    }


def render_teacher_graph_report(graph: dict, output: str | Path) -> Path:
    """Render the complete CRP audit mechanism and final graph diagnostics."""

    config = graph["config"]
    rows = []
    for group in graph.get("groups", []):
        rows.append(
            {
                "group_id": group["group_id"],
                "decision": _decision(group, config),
                "concepts": ", ".join(group.get("concepts", [])),
                "score": _number(group.get("score", 0.0)),
                "null_threshold": _number(group.get("null_threshold", 0.0)),
                "null_excess": _number(group.get("null_excess_score", 0.0)),
                "coverage": _percent(group.get("coverage", 0.0)),
                "gain": _number(group.get("robust_positive_gain", 0.0)),
                "semantic": _number(group.get("semantic_agreement", 0.0)),
                "accepted_edges": group.get("accepted_edges", 0),
                "hubness": _number(group.get("hubness_penalty", 0.0)),
                "evidence": (
                    f"null ratio={_number(group.get('null_excess_ratio', 0.0))}; "
                    f"activation/gain={_number(group.get('activation_gain_alignment', 0.0))}"
                ),
            }
        )
    selected = len(graph.get("selected_group_ids", []))
    summary = (
        f"Audited groups: {len(rows)}\nSelected groups: {selected}\n"
        f"Final accepted edges: {graph.get('degree_stats', {}).get('edge_count', 0)}\n"
        f"Final graph coverage: {_percent(graph.get('degree_stats', {}).get('coverage', 0.0))}"
    )
    columns = [
        {"key": "group_id", "label": "Group"},
        {"key": "decision", "label": "Decision"},
        {"key": "concepts", "label": "Concepts"},
        {"key": "score", "label": "Audit score"},
        {"key": "null_threshold", "label": "Null threshold"},
        {"key": "null_excess", "label": "Null excess"},
        {"key": "coverage", "label": "Coverage"},
        {"key": "gain", "label": "Robust gain"},
        {"key": "semantic", "label": "Semantic agreement"},
        {"key": "accepted_edges", "label": "Accepted edges"},
        {"key": "hubness", "label": "Hubness penalty"},
        {"key": "evidence", "label": "Confidence evidence"},
    ]
    return render_report(
        "CRP teacher-graph mechanism report",
        [
            {"title": "Summary", "text": summary},
            {"title": "Concept-group source", "data": graph.get("concept_groups_source", {})},
            {"title": "Grouping configuration", "data": graph.get("grouping_config", {})},
            {"title": "Audit and graph configuration", "data": config},
            {"title": "All group decisions", "table": {"columns": columns, "rows": rows}},
            {"title": "Confidence and intervention evidence", "data": _edge_evidence_summary(graph)},
            {"title": "Final graph statistics", "data": graph.get("degree_stats", {})},
        ],
        output,
    )
