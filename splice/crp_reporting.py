"""Human-inspectable reports for the two frozen CRP construction stages."""

from __future__ import annotations

import html
import statistics
from pathlib import Path
from typing import Callable

from splice.crp import validate_concept_groups
from splice.reporting import render_report


def _number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _percent(value: object) -> str:
    return f"{100.0 * float(value):.1f}%"


def render_concept_groups_report(
    artifact: dict,
    output: str | Path,
    image_resolver: Callable[[str], str | None] | None = None,
) -> Path:
    """Render the grouping census with compact, grouping-only diagnostics."""

    artifact = validate_concept_groups(artifact)
    diagnostics = artifact["diagnostics"]
    report = artifact.get("report_diagnostics", {})
    funnel = report.get("filtering_funnel", {})
    detail_by_id = {
        int(group["group_id"]): group for group in report.get("composite_groups", [])
    }

    def escape(value: object) -> str:
        return html.escape(str(value))

    def badge(value: str, kind: str | None = None) -> str:
        style = kind or value.lower()
        return f'<span class="badge badge-{escape(style)}">{escape(value)}</span>'

    def table(headers: list[str], rows: list[list[object]]) -> str:
        heading = "".join(f"<th>{escape(value)}</th>" for value in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
            for row in rows
        )
        return (
            '<div class="table-wrap"><table><thead><tr>' + heading
            + "</tr></thead><tbody>" + body + "</tbody></table></div>"
        )

    funnel_values = [
        ("Raw vocabulary", funnel.get("raw_vocabulary_count", len(artifact["vocabulary"]))),
        ("Active on dataset", funnel.get("dataset_active_count", "—")),
        ("Survived frequency filter", funnel.get("frequency_filtered_count", artifact["active_concept_count"])),
        ("Final concept groups", funnel.get("final_group_count", len(artifact["groups"]))),
    ]
    funnel_html = "".join(
        f'<div class="funnel-step"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
        for label, value in funnel_values
    )
    removal_html = (
        '<div class="filter-removals">'
        f'<span><strong>{escape(funnel.get("below_minimum_count", "—"))}</strong> below minimum</span>'
        f'<span><strong>{escape(funnel.get("above_maximum_count", "—"))}</strong> above maximum</span>'
        f'<span><strong>{escape(funnel.get("inactive_count", "—"))}</strong> inactive</span>'
        "</div>"
    )

    filtered_sections = []
    for key, label in (
        ("active_below_minimum", "Active concepts below the minimum"),
        ("above_maximum", "Concepts above the maximum"),
    ):
        values = report.get(key, [])
        if values:
            rows = [
                [
                    escape(item["concept"]),
                    escape(item["concept_index"]),
                    escape(_percent(item["frequency"])),
                ]
                for item in values
            ]
            filtered_sections.append(
                f'<details><summary>{escape(label)} <span class="muted">({len(values)})</span></summary>'
                + table(["Concept", "Vocabulary index", "Frequency"], rows)
                + "</details>"
            )

    singleton_fraction = float(diagnostics["singleton_fraction"])
    composite_fraction = max(0.0, 1.0 - singleton_fraction)
    ratio_html = f"""
<div class="ratio-labels"><span><i class="dot singleton"></i>Singletons <strong>{diagnostics['singleton_count']}</strong></span>
<span><i class="dot composite"></i>Composite <strong>{diagnostics['composite_group_count']}</strong></span></div>
<div class="ratio" role="img" aria-label="{_percent(singleton_fraction)} singleton groups and {_percent(composite_fraction)} composite groups">
<span class="ratio-singleton" style="width:{100 * singleton_fraction:.3f}%"></span>
<span class="ratio-composite" style="width:{100 * composite_fraction:.3f}%"></span></div>
<p class="ratio-caption"><strong>{_percent(singleton_fraction)}</strong> of final groups are singletons.</p>"""

    config_html = "".join(
        f"<div><dt>{escape(key)}</dt><dd>{escape(_number(value))}</dd></div>"
        for key, value in artifact["config"].items()
    )

    resolved_images: dict[str, str | None] = {}

    def thumbnail(sample: dict) -> str:
        sample_id = str(sample["sample_id"])
        if sample_id not in resolved_images:
            try:
                resolved_images[sample_id] = image_resolver(sample_id) if image_resolver else None
            except (FileNotFoundError, OSError, ValueError):
                resolved_images[sample_id] = None
        source = resolved_images[sample_id]
        visual = (
            f'<img src="{escape(source)}" alt="Representative dataset sample {escape(sample_id)}" loading="lazy">'
            if source
            else '<div class="thumb-placeholder" aria-hidden="true">image<br>unavailable</div>'
        )
        return (
            f'<figure title="Group activation: {escape(_number(sample.get("activation", 0.0)))}">'
            f"{visual}<figcaption>{escape(sample_id)}</figcaption></figure>"
        )

    group_cards = []
    composite_rows = []
    for group in artifact["groups"]:
        if int(group["size"]) <= 1:
            continue
        detail = detail_by_id.get(int(group["group_id"]), {})
        verdict = str(detail.get("verdict", "unavailable"))
        reason = str(detail.get("verdict_reason", ""))
        concepts = "".join(f'<span class="concept">{escape(value)}</span>' for value in group["concepts"])
        samples = "".join(thumbnail(sample) for sample in detail.get("representative_samples", []))
        if not samples:
            samples = '<p class="muted compact">Regenerate with dataset access to embed representative images.</p>'

        relation_rows = []
        for relation in detail.get("relations", []):
            if relation["lexical_family_passed"] and relation["text_threshold_passed"] and relation["coactivation_threshold_passed"]:
                link = "lexical + thresholds"
            elif relation["lexical_family_passed"]:
                link = "lexical"
            else:
                link = "thresholds"
            relation_rows.append(
                [
                    escape(relation["concept_a"]),
                    escape(relation["concept_b"]),
                    escape(_number(relation["text_similarity"])),
                    badge("pass" if relation["text_threshold_passed"] else "fail"),
                    escape(_number(relation["coactivation"])),
                    badge("pass" if relation["coactivation_threshold_passed"] else "fail"),
                    escape(link),
                ]
            )
        relation_table = (
            table(
                ["Concept A", "Concept B", "Text sim.", "Text threshold", "Coactivation", "Coact. threshold", "Grouping link"],
                relation_rows,
            )
            if relation_rows
            else '<p class="muted compact">Relation evidence is unavailable in this artifact.</p>'
        )
        reason_html = f'<p class="verdict-reason">{escape(reason)}.</p>' if reason else ""
        group_cards.append(
            f"""<article class="group-card">
<div class="group-heading"><div><span class="eyebrow">Group {escape(group['group_id'])}</span>
<h3>{escape(group['size'])} related concepts</h3></div>{badge(verdict)}</div>
<div class="concepts">{concepts}</div>{reason_html}
<div class="examples"><span class="subhead">Top-activating samples</span><div class="thumb-row">{samples}</div></div>
<details><summary>Why grouped <span class="muted">· {len(relation_rows)} causal relations</span></summary>{relation_table}</details>
</article>"""
        )
        composite_rows.append(
            [
                escape(group["group_id"]),
                badge(verdict),
                escape(group["size"]),
                escape(", ".join(group["concepts"])),
                escape(", ".join(str(value) for value in group["concept_indices"])),
            ]
        )

    largest_rows = [
        [escape(group["group_id"]), escape(group["size"]), escape(group["listing"])]
        for group in diagnostics["largest_composite_groups"]
    ]
    verdict_counts = {
        verdict: sum(detail.get("verdict") == verdict for detail in detail_by_id.values())
        for verdict in ("good", "borderline", "suspicious")
    }
    verdict_summary = " ".join(
        badge(f"{verdict_counts[verdict]} {verdict}", verdict)
        for verdict in ("good", "borderline", "suspicious")
    )

    title = "CRP concept-group census"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title}</title><style>
:root{{--ink:#20231f;--muted:#687067;--line:#dde1d9;--panel:#fff;--wash:#f5f6f3;--soft:#f0f2ed;
--blue:#54798f;--green:#287a55;--green-soft:#e7f4ec;--amber:#966719;--amber-soft:#fff2d7;--red:#a6413c;--red-soft:#fbe9e7}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1060px;margin:auto;padding:38px 20px 64px}}h1{{font-size:30px;letter-spacing:-.03em;margin:0 0 6px}}
h2{{font-size:19px;margin:0 0 16px}}h3{{font-size:17px;margin:3px 0 0}}.intro{{color:var(--muted);margin:0 0 24px}}
section{{background:var(--panel);margin:16px 0;padding:22px;border:1px solid var(--line);border-radius:12px}}
.funnel{{display:grid;grid-template-columns:repeat(4,1fr);gap:28px}}.funnel-step{{position:relative;min-width:0}}
.funnel-step:not(:last-child)::after{{content:'→';position:absolute;right:-20px;top:18px;color:#a4aaa1;font-size:18px}}
.funnel-step span{{display:block;color:var(--muted);font-size:12px;line-height:1.25}}.funnel-step strong{{font-size:26px}}
.filter-removals{{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px;padding-top:14px;border-top:1px solid var(--line)}}
.filter-removals span{{background:var(--soft);padding:5px 9px;border-radius:6px;font-size:12px}}
.stats-grid{{display:grid;grid-template-columns:1.2fr 1fr;gap:16px}}.ratio-labels{{display:flex;justify-content:space-between;gap:16px;font-size:13px}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}.dot.singleton{{background:#bdc8bf}}.dot.composite{{background:var(--blue)}}
.ratio{{display:flex;height:12px;overflow:hidden;border-radius:99px;background:var(--soft);margin-top:12px}}.ratio span{{display:block;height:100%}}
.ratio-singleton{{background:#bdc8bf}}.ratio-composite{{background:var(--blue)}}.ratio-caption{{font-size:13px;color:var(--muted);margin:9px 0 0}}
.quick-stats{{display:grid;grid-template-columns:repeat(2,1fr);gap:9px}}.quick-stats div{{background:var(--soft);border-radius:8px;padding:10px}}
.quick-stats span{{display:block;color:var(--muted);font-size:11px}}.quick-stats strong{{font-size:18px}}
.config-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:0;border:1px solid var(--line);border-radius:9px;overflow:hidden;margin:0}}
.config-grid div{{padding:10px 12px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}}
.config-grid dt{{color:var(--muted);font-size:11px;overflow-wrap:anywhere}}.config-grid dd{{margin:2px 0 0;font-weight:650}}
.section-heading{{display:flex;justify-content:space-between;gap:16px;align-items:start;flex-wrap:wrap}}.verdict-counts{{display:flex;gap:6px;flex-wrap:wrap}}
.group-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.group-card{{border:1px solid var(--line);border-radius:10px;padding:16px;min-width:0}}
.group-heading{{display:flex;justify-content:space-between;gap:12px;align-items:start}}.eyebrow,.subhead{{color:var(--muted);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em}}
.badge{{display:inline-flex;align-items:center;border-radius:99px;padding:2px 7px;font-size:11px;font-weight:700;white-space:nowrap}}
.badge-good,.badge-pass{{color:var(--green);background:var(--green-soft)}}.badge-borderline{{color:var(--amber);background:var(--amber-soft)}}
.badge-suspicious,.badge-fail{{color:var(--red);background:var(--red-soft)}}.badge-unavailable{{color:var(--muted);background:var(--soft)}}
.concepts{{display:flex;flex-wrap:wrap;gap:5px;margin:12px 0}}.concept{{border:1px solid var(--line);background:var(--soft);border-radius:5px;padding:2px 7px;font-size:12px}}
.verdict-reason{{color:var(--red);font-size:12px;margin:8px 0}}.examples{{margin-top:14px}}.thumb-row{{display:flex;gap:7px;overflow-x:auto;padding-top:7px}}
figure{{margin:0;min-width:68px;width:68px}}figure img,.thumb-placeholder{{display:block;width:68px;height:58px;object-fit:cover;border-radius:6px;border:1px solid var(--line);background:var(--soft)}}
.thumb-placeholder{{display:grid;place-items:center;text-align:center;color:#90978f;font-size:9px;line-height:1.15}}figcaption{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);font-size:9px;margin-top:3px}}
details{{margin-top:13px;border-top:1px solid var(--line);padding-top:10px}}summary{{cursor:pointer;font-weight:650;font-size:12px}}details .table-wrap{{margin-top:10px}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{background:var(--soft);position:sticky;top:0;color:#50574f;white-space:nowrap}}.muted{{color:var(--muted);font-weight:400}}.compact{{margin:7px 0 0;font-size:12px}}
.note{{font-size:12px;color:var(--muted);margin:8px 0 0}}
@media(max-width:760px){{.funnel{{grid-template-columns:repeat(2,1fr)}}.funnel-step:nth-child(2)::after{{display:none}}.stats-grid,.group-grid{{grid-template-columns:1fr}}.config-grid{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:440px){{main{{padding:24px 12px 48px}}section{{padding:16px}}.funnel{{gap:18px}}.funnel-step:not(:last-child)::after{{display:none}}.config-grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>{title}</h1>
<p class="intro">A compact audit of frequency filtering, connected concept groups, and the evidence behind each composite.</p>
<section><h2>Filtering funnel</h2><div class="funnel">{funnel_html}</div>{removal_html}{''.join(filtered_sections)}
<p class="note">Below-minimum counts include inactive vocabulary entries; expand the active filtered lists to inspect named concepts.</p></section>
<div class="stats-grid"><section><h2>Singleton vs composite ratio</h2>{ratio_html}</section>
<section><h2>Group summary</h2><div class="quick-stats">
<div><span>Surviving concepts</span><strong>{diagnostics['total_active_concepts']}</strong></div>
<div><span>Final groups</span><strong>{diagnostics['total_groups']}</strong></div>
<div><span>Concepts in composites</span><strong>{diagnostics['concepts_in_composite_groups']}</strong></div>
<div><span>Largest group</span><strong>{diagnostics['maximum_group_size']}</strong></div></div></section></div>
<section><h2>Grouping configuration</h2><dl class="config-grid">{config_html}</dl></section>
<section><div class="section-heading"><div><h2>Composite group review</h2>
<p class="note">Verdicts use only causal relations: good passes both thresholds; borderline has a lexical relation missing one threshold; suspicious has one missing both.</p></div>
<div class="verdict-counts">{verdict_summary}</div></div><div class="group-grid">{''.join(group_cards)}</div></section>
<section><h2>Largest composite groups</h2>{table(['Group', 'Size', 'Concepts'], largest_rows)}</section>
<section><h2>All composite groups</h2>{table(['Group', 'Verdict', 'Size', 'Concepts', 'Vocabulary indices'], composite_rows)}</section>
</main></body></html>"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)
    return path


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
