"""Render the self-contained CRPv4 group-screen research dashboard.

THESIS: Turn one frozen experiment into an immediate go/no-go reading, without
making the researcher hunt through logs or decorative cards.
OWN-WORLD: A restrained laboratory report: white paper, graphite text, cobalt
status ink, amber warnings, ruled tables, and image contact sheets.
STORY: Read the decision, verify reconstruction, inspect class slices, review
groups, then inspect sampled interventions.
FIRST VIEWPORT: Decision band, experiment identity, and one compact metric rail.
FORM: Operate/read dashboard extending the repository's existing audit reports.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from PIL import Image

from experiments.spurious_eval.datasets.paths import resolve_dataset_root
from splice.crp_group_screen import GROUP_SCREEN_ARTIFACT


def _source_index(sample_id: str) -> int:
    prefix, separator, index = str(sample_id).rpartition(":")
    if not separator or not index.isdigit():
        raise ValueError(f"Expected a dataset:index sample ID, got {sample_id!r}.")
    return int(index)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _dataset_records(report: dict, data_folder: Path) -> tuple[dict[str, dict], str | None]:
    dataset = str(report.get("provenance", {}).get("dataset", "")).lower()
    sample_ids = [str(value) for value in report.get("sample_ids", [])]
    try:
        if dataset == "waterbirds":
            root = resolve_dataset_root(data_folder, "waterbirds", ["metadata.csv"])
            rows = _read_csv(root / "metadata.csv")
            records = {}
            for sample_id in sample_ids:
                row = rows[_source_index(sample_id)]
                target = "waterbird" if int(row["y"]) == 1 else "landbird"
                context = "water background" if int(row["place"]) == 1 else "land background"
                records[sample_id] = {
                    "target": target,
                    "context": context,
                    "subgroup": f"{target} · {context}",
                    "image_path": root / row["img_filename"],
                }
            return records, None
        if dataset == "celeba":
            root = resolve_dataset_root(data_folder, "celeba", ["list_attr_celeba.csv"])
            rows = _read_csv(root / "list_attr_celeba.csv")
            records = {}
            for sample_id in sample_ids:
                row = rows[_source_index(sample_id)]
                target = "blond" if int(row["Blond_Hair"]) == 1 else "not blond"
                context = "male" if int(row["Male"]) == 1 else "female"
                image_name = row.get("image_id") or next(iter(row.values()))
                records[sample_id] = {
                    "target": target,
                    "context": context,
                    "subgroup": f"{target} · {context}",
                    "image_path": root / "img_align_celeba" / str(image_name),
                }
            return records, None
        return {}, f"Post-hoc class view is not implemented for dataset {dataset!r}."
    except (FileNotFoundError, KeyError, IndexError, ValueError) as error:
        return {}, f"Post-hoc class view is unavailable: {error}"


def _image_data_uri(path: Path, max_width: int = 440, max_height: int = 300) -> str | None:
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            image.thumbnail((max_width, max_height), resampling)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=84, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    except (FileNotFoundError, OSError):
        return None


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    if not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _percent(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{100.0 * float(value):.{digits}f}%"


def _metric(label: str, value: str, note: str = "") -> str:
    return (
        '<div class="metric"><span class="metric-label">'
        f"{html.escape(label)}</span><strong>{html.escape(value)}</strong>"
        f'<small>{html.escape(note)}</small></div>'
    )


def _curve_svg(points: Sequence[dict], target: float) -> str:
    if not points:
        return '<p class="empty">No coverage curve was produced.</p>'
    width, height = 920, 260
    left, top, right, bottom = 58, 20, 18, 42
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum_x = max(1, max(int(point["group_count"]) for point in points))

    def xy(point: dict) -> tuple[float, float]:
        x = left + int(point["group_count"]) / maximum_x * plot_width
        y = top + (1.0 - float(point["source_coverage"])) * plot_height
        return x, y

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, points))
    target_y = top + (1.0 - target) * plot_height
    ticks = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + (1.0 - fraction) * plot_height
        ticks.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end">{fraction:.0%}</text>'
        )
    return f"""
    <svg class="curve" viewBox="0 0 {width} {height}" role="img" aria-label="Image coverage by number of groups">
      {''.join(ticks)}
      <line x1="{left}" y1="{target_y:.1f}" x2="{width-right}" y2="{target_y:.1f}" class="target"/>
      <polyline points="{polyline}" class="coverage-line"/>
      <text x="{left}" y="{height-10}">1 group</text>
      <text x="{width-right}" y="{height-10}" text-anchor="end">{maximum_x} groups</text>
      <text x="{width-right-6}" y="{target_y-7:.1f}" text-anchor="end" class="target-label">target {target:.0%}</text>
    </svg>
    """


def _image_figure(image_row: dict, records: dict[str, dict], compact: bool = False) -> str:
    sample_id = str(image_row["sample_id"])
    record = records.get(sample_id, {})
    uri = _image_data_uri(Path(record["image_path"])) if record.get("image_path") else None
    media = (
        f'<img src="{uri}" alt="{html.escape(record.get("subgroup", sample_id))}">'
        if uri
        else '<div class="missing-image">Image unavailable</div>'
    )
    groups = image_row.get("top_groups", [])
    group_text = ", ".join(
        f"{item['name']} ({float(item['activation']):.2g})" for item in groups
    ) or "no active selected group"
    class_text = record.get("subgroup", "class metadata unavailable")
    return f"""
      <figure class="image-sample{' compact' if compact else ''}">
        {media}
        <figcaption>
          <strong>fidelity {_fmt(float(image_row['source_fidelity']), 4)}</strong>
          <span>{html.escape(class_text)}</span>
          <span>{html.escape(group_text)}</span>
          <code>{html.escape(sample_id)}</code>
        </figcaption>
      </figure>
    """


def _reconstruction_bands(images: list[dict], count: int) -> dict[str, list[dict]]:
    ordered = sorted(images, key=lambda item: (float(item["source_fidelity"]), item["sample_id"]))
    if not ordered:
        return {"Poor reconstruction": [], "Median reconstruction": [], "Good reconstruction": []}
    count = min(count, len(ordered))
    midpoint = len(ordered) // 2
    median_start = max(0, min(len(ordered) - count, midpoint - count // 2))
    return {
        "Poor reconstruction": ordered[:count],
        "Median reconstruction": ordered[median_start : median_start + count],
        "Good reconstruction": ordered[-count:][::-1],
    }


def _class_rows(images: list[dict], records: dict[str, dict], threshold: float) -> str:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in images:
        record = records.get(str(item["sample_id"]))
        if record:
            grouped[str(record["subgroup"])].append(float(item["source_fidelity"]))
    if not grouped:
        return ""
    rows = []
    for name in sorted(grouped):
        values = sorted(grouped[name])
        rows.append(
            "<tr>"
            f"<th>{html.escape(name)}</th>"
            f"<td>{len(values):,}</td>"
            f"<td>{statistics.median(values):.4f}</td>"
            f"<td>{values[max(0, math.ceil(0.01 * len(values)) - 1)]:.4f}</td>"
            f"<td>{sum(value >= threshold for value in values) / len(values):.1%}</td>"
            "</tr>"
        )
    return "".join(rows)


def _config_table(report: dict) -> str:
    values = {
        "dataset": report.get("provenance", {}).get("dataset", "unknown"),
        "spatial variant": (report.get("spatial_balance") or {}).get("variant", "none"),
        "minimum concept frequency": report.get("group_config", {}).get("min_concept_frequency"),
        "maximum concept frequency": report.get("group_config", {}).get("max_concept_frequency"),
        "text similarity": report.get("group_config", {}).get("text_similarity_threshold"),
        "coactivation": report.get("group_config", {}).get("coactivation_threshold"),
        "spatial floor": report.get("group_config", {}).get("spatial_balance_floor"),
        "spatial frequency power": report.get("group_config", {}).get("spatial_frequency_power"),
        "fidelity gate": report.get("screen_config", {}).get("fidelity_threshold"),
        "coverage target": report.get("screen_config", {}).get("target_image_coverage"),
    }
    return "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in values.items()
    )


def _group_table(groups: list[dict], mini_lookup: dict[int, dict]) -> str:
    rows = []
    for group in sorted(groups, key=lambda item: int(item.get("activation_rank", 10**9))):
        group_id = int(group["group_id"])
        mini = mini_lookup.get(group_id)
        search = " ".join([str(group["name"]), *map(str, group.get("concepts", []))]).lower()
        state = "selected" if group.get("selected_for_reconstruction") else "candidate"
        if group.get("coherence_warning"):
            state += " warning"
        rows.append(
            f'<tr data-search="{html.escape(search)}" data-state="{state}">'
            f"<td>{int(group.get('activation_rank', 0))}</td>"
            f"<th>{html.escape(str(group['name']))}</th>"
            f"<td>{html.escape(', '.join(map(str, group.get('concepts', []))))}</td>"
            f"<td>{int(group.get('concept_count', 0))}</td>"
            f"<td>{float(group.get('mean_text_similarity', 0.0)):.3f}</td>"
            f"<td>{float(group.get('activation_frequency', 0.0)):.1%}</td>"
            f"<td>{'yes' if group.get('selected_for_reconstruction') else 'no'}</td>"
            f"<td>{'pass' if mini and mini.get('passed_null') else 'fail' if mini else '—'}</td>"
            "</tr>"
        )
    return "".join(rows)


def _group_details(
    groups: list[dict],
    image_lookup: dict[str, dict],
    records: dict[str, dict],
    mini_lookup: dict[int, dict],
    max_groups: int,
) -> str:
    selected = [group for group in groups if group.get("selected_for_reconstruction")]
    selected.sort(key=lambda item: int(item.get("activation_rank", 10**9)))
    if mini_lookup:
        visible_ids = set(mini_lookup)
        visible = [group for group in selected if int(group["group_id"]) in visible_ids]
    else:
        visible = selected[:max_groups]
    sections = []
    for group in visible:
        group_id = int(group["group_id"])
        mini = mini_lookup.get(group_id)
        images = [
            image_lookup[sample_id]
            for sample_id in group.get("top_sample_ids", [])
            if sample_id in image_lookup
        ]
        mini_text = (
            f"null={'pass' if mini.get('passed_null') else 'fail'} · "
            f"coverage={float(mini.get('coverage', 0.0)):.1%} · "
            f"top-1 turnover={float(mini.get('top1_neighbor_turnover', 0.0)):.1%} · "
            f"Jaccard@K={float(mini.get('mean_jaccard_at_k', 0.0)):.3f}"
            if mini
            else "not included in the mini audit"
        )
        edge_html = ""
        if mini and mini.get("neighbor_triplets"):
            triplets = []
            for edge in mini["neighbor_triplets"]:
                anchor = image_lookup.get(edge["anchor_sample_id"])
                raw = image_lookup.get(edge["raw_neighbor_sample_id"])
                projected = image_lookup.get(edge["projected_neighbor_sample_id"])
                if not anchor or not raw or not projected:
                    continue
                changed = "changed" if edge.get("top1_changed") else "unchanged"
                triplets.append(
                    '<div class="neighbor-triplet">'
                    '<div><span class="triplet-role">Anchor</span>'
                    f'{_image_figure(anchor, records, compact=True)}</div>'
                    '<div><span class="triplet-role">Raw top-1</span>'
                    f'{_image_figure(raw, records, compact=True)}'
                    f'<p class="similarity">raw similarity {float(edge["raw_neighbor_similarity"]):.3f}</p></div>'
                    '<div><span class="triplet-role">Projected top-1</span>'
                    f'{_image_figure(projected, records, compact=True)}'
                    f'<p class="similarity">{float(edge["projected_neighbor_raw_similarity"]):.3f} → '
                    f'{float(edge["projected_neighbor_similarity"]):.3f} · gain '
                    f'{float(edge["projected_neighbor_gain"]):+.4f} · {changed}</p></div>'
                    "</div>"
                )
            if triplets:
                edge_html = (
                    '<h4>Raw vs projected nearest neighbours</h4>'
                    '<p class="section-note">Each row shows the same anchor, its original nearest neighbour, and the nearest neighbour after removing this concept-group subspace.</p>'
                    + "".join(triplets)
                )
        sections.append(
            f"""
            <details class="group-detail">
              <summary>
                <span><strong>{html.escape(str(group['name']))}</strong><small>rank {int(group['activation_rank'])}</small></span>
                <span>{int(group['concept_count'])} concepts · {float(group['activation_frequency']):.1%} images</span>
              </summary>
              <div class="detail-body">
                <p class="concept-list">{html.escape(' · '.join(map(str, group.get('concepts', []))))}</p>
                <p class="audit-line">Mean text similarity {float(group['mean_text_similarity']):.3f} · {html.escape(mini_text)}</p>
                <div class="contact-sheet">{''.join(_image_figure(item, records, compact=True) for item in images)}</div>
                {edge_html}
              </div>
            </details>
            """
        )
    return "".join(sections) or '<p class="empty">No reconstruction groups are available.</p>'


def render_group_screen(
    report: dict,
    data_folder: Path,
    output_path: Path,
    max_groups: int = 24,
    images_per_band: int = 4,
) -> Path:
    if report.get("artifact") != GROUP_SCREEN_ARTIFACT:
        raise ValueError(f"Expected a {GROUP_SCREEN_ARTIFACT} report.")
    if max_groups <= 0 or images_per_band <= 0:
        raise ValueError("Report group and image limits must be positive.")
    records, class_warning = _dataset_records(report, data_folder)
    images = list(report.get("images", []))
    image_lookup = {str(item["sample_id"]): item for item in images}
    groups = list(report.get("groups", []))
    mini = report.get("mini_intervention")
    mini_groups = list(mini.get("groups", [])) if isinstance(mini, dict) else []
    mini_lookup = {int(item["group_id"]): item for item in mini_groups}
    metrics = report.get("metrics", {})
    decision = report.get("decision", {})
    status = str(decision.get("status", "UNKNOWN"))
    status_class = "go" if status == "PROVISIONAL_GO" else "review" if status == "REVIEW_GROUPS" else "fail"
    bands = _reconstruction_bands(images, images_per_band)
    band_html = "".join(
        f'<section class="band"><h3>{html.escape(name)}</h3><div class="contact-sheet">'
        + "".join(_image_figure(item, records) for item in items)
        + "</div></section>"
        for name, items in bands.items()
    )
    fidelity_threshold = float(report.get("screen_config", {}).get("fidelity_threshold", 0.95))
    class_table = _class_rows(images, records, fidelity_threshold)
    class_section = (
        f'<p class="notice">{html.escape(class_warning)}</p>'
        if class_warning
        else f"""
        <p class="section-note">Post-hoc only. These annotations do not enter grouping, reconstruction selection, or the mini audit.</p>
        <div class="table-wrap"><table><thead><tr><th>Class / context</th><th>Images</th><th>Median fidelity</th><th>p01 fidelity</th><th>Coverage@{fidelity_threshold:.2f}</th></tr></thead><tbody>{class_table}</tbody></table></div>
        """
    )
    mini_metrics = (
        _metric(
            "Mini groups",
            f"{int(mini['audited_group_count'])}/{int(mini['requested_group_count'])}",
            "rank-spaced head → tail",
        )
        + _metric("Null pass", _percent(float(mini["null_pass_fraction"])), "audited groups")
        + _metric("Top-1 turnover", _percent(float(mini["median_top1_neighbor_turnover"])), "median")
        + _metric("Jaccard@K", _fmt(float(mini["median_jaccard_at_k"])), "median")
        if isinstance(mini, dict)
        else _metric("Mini audit", "not run", "reconstruction-only report")
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>CRPv4 concept-group screen</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#5d687c; --line:#d8dee9; --paper:#ffffff; --wash:#f4f7fb; --blue:#174ea6; --blue-soft:#eaf1fd; --green:#176b4d; --green-soft:#e7f5ef; --amber:#8a4b08; --amber-soft:#fff3dc; --red:#a22b2b; --red-soft:#fdecec; --shadow:0 12px 34px rgba(24,39,66,.09); }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--wash); color:var(--ink); font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
    main {{ width:min(1460px,calc(100% - 32px)); margin:24px auto 64px; }}
    header, .panel {{ background:var(--paper); border:1px solid var(--line); border-radius:14px; box-shadow:var(--shadow); }}
    header {{ overflow:hidden; }}
    .decision {{ display:grid; grid-template-columns:minmax(230px,.55fr) 1.45fr; gap:28px; padding:28px 32px; align-items:center; background:var(--blue-soft); }}
    .decision.go {{ background:var(--green-soft); }} .decision.fail {{ background:var(--red-soft); }} .decision.review {{ background:var(--amber-soft); }}
    h1 {{ margin:0; font-size:2.15rem; letter-spacing:-.025em; line-height:1.06; }}
    h2 {{ margin:0 0 8px; font-size:1.35rem; letter-spacing:-.015em; }} h3 {{ margin:0 0 12px; font-size:1rem; }} h4 {{ margin:24px 0 10px; }}
    .status {{ display:inline-flex; width:max-content; margin-bottom:12px; padding:5px 9px; border:1px solid currentColor; border-radius:7px; color:var(--blue); font-weight:750; font-size:.78rem; }}
    .decision.go .status {{ color:var(--green); }} .decision.fail .status {{ color:var(--red); }} .decision.review .status {{ color:var(--amber); }}
    .decision p {{ margin:0; max-width:72ch; font-size:1.03rem; }}
    .metric-rail {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(135px,1fr)); border-top:1px solid var(--line); }}
    .metric {{ min-height:94px; padding:16px 18px; border-right:1px solid var(--line); }} .metric:last-child {{ border-right:0; }}
    .metric-label,.metric small {{ display:block; color:var(--muted); }} .metric-label {{ font-size:.78rem; }} .metric strong {{ display:block; margin:3px 0; font-size:1.35rem; font-variant-numeric:tabular-nums; }} .metric small {{ font-size:.73rem; }}
    .panel {{ margin-top:22px; padding:26px 28px; }} .section-note,.audit-line {{ color:var(--muted); max-width:76ch; }}
    .split {{ display:grid; grid-template-columns:1.55fr .75fr; gap:30px; align-items:start; }}
    .curve {{ display:block; width:100%; height:auto; overflow:visible; }} .curve text {{ fill:var(--muted); font-size:12px; }} .grid {{ stroke:#e7ebf1; }} .target {{ stroke:var(--amber); stroke-dasharray:6 5; }} .target-label {{ fill:var(--amber)!important; font-weight:700; }} .coverage-line {{ fill:none; stroke:var(--blue); stroke-width:3.5; stroke-linecap:round; stroke-linejoin:round; }}
    table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }} th,td {{ padding:9px 11px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }} thead th {{ position:sticky; top:0; background:#eef2f8; z-index:1; font-size:.78rem; }} tbody th {{ font-weight:680; }}
    .table-wrap {{ overflow:auto; max-height:620px; border:1px solid var(--line); border-radius:10px; }}
    .controls {{ display:flex; gap:10px; margin:16px 0 12px; }} input,select {{ min-height:40px; padding:8px 11px; border:1px solid #aeb8c9; border-radius:8px; background:white; color:var(--ink); font:inherit; }} input {{ flex:1; }} input:focus,select:focus,summary:focus-visible {{ outline:3px solid rgba(23,78,166,.24); outline-offset:2px; }}
    .band {{ margin-top:26px; }} .contact-sheet {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .image-sample {{ min-width:0; margin:0; background:#f7f9fc; border-radius:12px; overflow:hidden; }} .image-sample img,.missing-image {{ display:block; width:100%; height:220px; object-fit:contain; background:#e9eef5; }} .missing-image {{ display:grid; place-items:center; color:var(--muted); }}
    .image-sample figcaption {{ display:grid; gap:4px; padding:10px 12px; font-size:.78rem; }} .image-sample figcaption span,.image-sample code {{ color:var(--muted); overflow-wrap:anywhere; }} .image-sample.compact img,.image-sample.compact .missing-image {{ height:150px; }}
    .group-detail {{ margin-top:10px; border-bottom:1px solid var(--line); }} .group-detail summary {{ display:flex; justify-content:space-between; gap:20px; padding:15px 4px; cursor:pointer; }} .group-detail summary span:first-child {{ display:grid; }} .group-detail summary small {{ color:var(--muted); }} .detail-body {{ padding:0 4px 24px; }} .concept-list {{ max-width:95ch; }}
    .neighbor-triplet {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; align-items:start; margin-top:16px; padding:14px; border:1px solid var(--line); border-radius:12px; background:#fbfcfe; }} .triplet-role {{ display:block; margin-bottom:7px; color:var(--blue); font-size:.76rem; font-weight:760; text-transform:uppercase; letter-spacing:.04em; }} .similarity {{ margin:7px 2px 0; color:var(--muted); font-size:.78rem; font-variant-numeric:tabular-nums; }}
    .notice,.empty {{ padding:14px 16px; background:var(--amber-soft); border-radius:9px; color:var(--amber); }} code {{ font-family:ui-monospace,"Cascadia Mono",Consolas,monospace; }}
    footer {{ padding:26px 6px; color:var(--muted); max-width:78ch; }}
    @media (max-width:1080px) {{ .metric-rail {{ grid-template-columns:repeat(4,1fr); }} .split {{ grid-template-columns:1fr; }} .contact-sheet {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media (max-width:680px) {{ main {{ width:min(100% - 18px,1460px); margin-top:9px; }} .decision {{ grid-template-columns:1fr; padding:22px; }} .metric-rail {{ grid-template-columns:repeat(2,1fr); }} .panel {{ padding:20px 16px; }} .contact-sheet {{ grid-template-columns:1fr; }} .neighbor-triplet {{ grid-template-columns:1fr; }} .group-detail summary {{ display:grid; }} }}
  </style>
</head>
<body><main>
  <header>
    <div class="decision {status_class}">
      <div><span class="status">{html.escape(status)}</span><h1>CRPv4 group screen</h1></div>
      <p>{html.escape(str(decision.get('reason', 'No decision explanation was recorded.')))}</p>
    </div>
    <div class="metric-rail">
      {_metric('Candidate groups', _fmt(int(metrics.get('candidate_group_count', 0))), 'after frequency filtering')}
      {_metric('Groups required', _fmt(int(metrics.get('selected_group_count', 0))), 'activation-ranked prefix')}
      {_metric('Compression', _fmt(float(metrics.get('compression_ratio', 0.0)), 2) + '×', 'concepts per group')}
      {_metric('Coverage', _percent(float(metrics.get('source_coverage', 0.0))), 'at the fidelity gate')}
      {_metric('Median fidelity', _fmt(float(metrics.get('median_source_fidelity', 0.0)), 4), 'against original SpLiCE')}
      {_metric('p01 fidelity', _fmt(float(metrics.get('p01_source_fidelity', 0.0)), 4), 'weakest percentile')}
      {mini_metrics}
    </div>
  </header>

  <section class="panel split">
    <div><h2>Reconstruction coverage</h2><p class="section-note">Groups are ordered by total activation mass. The first prefix reaching the target becomes the reconstruction set.</p>{_curve_svg(report.get('coverage_curve', []), float(report.get('screen_config', {}).get('target_image_coverage', .99)))}</div>
    <div><h2>Resolved experiment</h2><table><tbody>{_config_table(report)}</tbody></table></div>
  </section>

  <section class="panel"><h2>Reconstruction examples</h2><p class="section-note">Poor, median, and good examples are selected only from label-free reconstruction fidelity. Class text is attached afterward for diagnosis.</p>{band_html}</section>

  <section class="panel"><h2>Class and context slices</h2>{class_section}</section>

  <section class="panel">
    <h2>All discovered groups</h2>
    <p class="section-note">Search by prototype or member concept. “Selected” means the group belongs to the smallest evaluated activation-ranked prefix that reaches the reconstruction target.</p>
    <div class="controls"><input id="group-search" type="search" placeholder="Search groups and concepts" aria-label="Search groups"><select id="group-state" aria-label="Filter group state"><option value="all">All groups</option><option value="selected">Selected only</option><option value="warning">Coherence warnings</option></select></div>
    <div class="table-wrap"><table id="group-table"><thead><tr><th>Rank</th><th>Prototype</th><th>Members</th><th>Size</th><th>Text sim.</th><th>Image support</th><th>Selected</th><th>Mini null</th></tr></thead><tbody>{_group_table(groups, mini_lookup)}</tbody></table></div>
  </section>

  <section class="panel"><h2>Selected group contact sheets</h2><p class="section-note">When the mini audit runs, this section expands its rank-spaced sample ({int(mini.get('audited_group_count', 0)) if isinstance(mini, dict) else min(max_groups, int(metrics.get('selected_group_count', 0)))} of {int(metrics.get('selected_group_count', 0))} selected groups), including the head and tail. Otherwise it shows the first {max_groups}. Every group remains available in the table above.</p>{_group_details(groups, image_lookup, records, mini_lookup, max_groups)}</section>

  <footer>This is a diagnostic artifact, not a training result. Group discovery and mini-intervention use no target or context labels. The class section is post-hoc and must not be used to rewrite the tested configuration.</footer>
  <script>
    const search = document.getElementById('group-search');
    const state = document.getElementById('group-state');
    const rows = [...document.querySelectorAll('#group-table tbody tr')];
    function filterRows() {{
      const query = search.value.trim().toLowerCase();
      const wanted = state.value;
      rows.forEach(row => {{
        const textMatch = !query || row.dataset.search.includes(query);
        const stateMatch = wanted === 'all' || row.dataset.state.split(' ').includes(wanted);
        row.hidden = !(textMatch && stateMatch);
      }});
    }}
    search.addEventListener('input', filterRows);
    state.addEventListener('change', filterRows);
  </script>
</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--data-folder", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-groups", type=int, default=24)
    parser.add_argument("--images-per-band", type=int, default=4)
    args = parser.parse_args(argv)
    report = json.loads(args.screen.read_text(encoding="utf-8"))
    render_group_screen(
        report,
        args.data_folder,
        args.output,
        max_groups=args.max_groups,
        images_per_band=args.images_per_band,
    )
    print(f"[INFO] Group screen HTML: {args.output}")


if __name__ == "__main__":
    main()
