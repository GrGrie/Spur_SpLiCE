"""Render a self-contained post-hoc Waterbirds concept-ablation report."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from experiments.spurious_eval.datasets.waterbirds import WaterbirdsDataset
from splice.cqt import CQT_ARTIFACT, concept_quotient
from splice.crp import orthonormal_basis, project_out, validate_feature_cache
from splice.graph_io import load_graph_json


CRP_ARTIFACTS = {
    "splice_crp_v2_teacher_graph",
    "splice_crp_v3_teacher_graph",
    "splice_crp_v4_teacher_graph",
}
LABEL_NAMES = {0: "landbird", 1: "waterbird"}
BACKGROUND_NAMES = {0: "land background", 1: "water background"}


@dataclass(frozen=True)
class Intervention:
    identifier: str
    title: str
    concepts: str
    kind: str
    direction: torch.Tensor
    concept_indices: tuple[int, ...]
    selected: bool
    evidence: str


@dataclass(frozen=True)
class PairSpec:
    title: str
    explanation: str
    left_label: int
    left_background: int
    right_label: int
    right_background: int
    same_pool: bool = False


PAIR_SPECS = (
    PairSpec(
        title="Pair 1: same target, different spurious attribute",
        explanation="Both images contain waterbirds; one is on land and the other is on water.",
        left_label=1,
        left_background=0,
        right_label=1,
        right_background=1,
    ),
    PairSpec(
        title="Pair 2: different targets, same spurious attribute",
        explanation="A waterbird and a landbird; both birds are on land.",
        left_label=1,
        left_background=0,
        right_label=0,
        right_background=0,
    ),
    PairSpec(
        title="Pair 3: same target and spurious attribute",
        explanation="Two different landbirds on land: a control for ordinary within-group similarity.",
        left_label=0,
        left_background=0,
        right_label=0,
        right_background=0,
        same_pool=True,
    ),
    PairSpec(
        title="Pair 4: different target and spurious attribute",
        explanation="A waterbird on water and a landbird on land: an expected negative control pair.",
        left_label=1,
        left_background=1,
        right_label=0,
        right_background=0,
    ),
)


def _apply(values: torch.Tensor, intervention: Intervention) -> torch.Tensor:
    if intervention.kind == "crp_projection":
        return project_out(values, intervention.direction)
    if intervention.kind == "cqt_quotient":
        return concept_quotient(values, intervention.direction)
    raise ValueError(f"Unknown intervention kind: {intervention.kind}")


def _interventions(
    graph: dict,
    cache: dict,
    scope: str,
    max_interventions: int = 0,
) -> list[Intervention]:
    selected_only = scope == "selected"
    interventions: list[Intervention] = []
    group_ids = torch.as_tensor(graph.get("group_ids", []), dtype=torch.long)
    weights = torch.as_tensor(graph.get("weights", []), dtype=torch.float32)

    def retained_edge_count(identifier: int) -> int:
        if group_ids.shape != weights.shape or group_ids.numel() == 0:
            return 0
        return int(((group_ids == identifier) & (weights > 0)).sum())

    if graph.get("artifact") in CRP_ARTIFACTS:
        tolerance = float(graph.get("config", {}).get("orthogonal_tolerance", 1e-6))
        groups = sorted(
            graph.get("groups", []),
            key=lambda group: (
                -int(bool(group.get("selected", False))),
                -retained_edge_count(int(group["group_id"])),
                -float(group.get("null_excess_score", 0.0)),
                -float(group.get("score", 0.0)),
                int(group["group_id"]),
            ),
        )
        for group in groups:
            selected = bool(group.get("selected", False))
            if selected_only and not selected:
                continue
            indices = [int(index) for index in group["concept_indices"]]
            concepts = [str(word) for word in group.get("concepts", [])]
            semantic_label = (
                "residual SpLiCE agreement"
                if not bool(graph.get("config", {}).get("use_dino", True))
                and bool(group.get("residual_splice_gate_enabled", False))
                else "DINO agreement"
            )
            interventions.append(
                Intervention(
                    identifier=f"G{int(group['group_id'])}",
                    title=f"Concept group G{int(group['group_id'])}",
                    concepts=", ".join(concepts),
                    kind="crp_projection",
                    direction=orthonormal_basis(cache["dictionary"][indices], tolerance),
                    concept_indices=tuple(indices),
                    selected=selected,
                    evidence=(
                        f"rank={int(group.get('basis_rank', len(indices)))}, "
                        f"score={float(group.get('score', 0.0)):.6g}, "
                        f"null threshold={float(group.get('null_threshold', 0.0)):.6g}, "
                        f"null excess={float(group.get('null_excess_score', 0.0)):.6g}, "
                        f"null ratio={float(group.get('null_excess_ratio', 0.0)):.2%}, "
                        f"coverage={float(group.get('coverage', 0.0)):.2%}, "
                        f"{semantic_label}={float(group.get('semantic_agreement', 0.0)):.4f}, "
                        f"alignment={float(group.get('activation_gain_alignment', 0.0)):.4f}, "
                        f"turnover={float(group.get('mean_neighbor_turnover', 0.0)):.2%}, "
                        f"retained edges={retained_edge_count(int(group['group_id']))}"
                    ),
                )
            )
    elif graph.get("artifact") == CQT_ARTIFACT:
        factors = sorted(
            graph.get("factors", []),
            key=lambda factor: (
                -int(bool(factor.get("selected", False))),
                -retained_edge_count(int(factor["factor_id"])),
                -float(factor.get("null_excess_gain", 0.0)),
                -float(factor.get("intervention_gain", 0.0)),
                int(factor["factor_id"]),
            ),
        )
        for factor in factors:
            selected = bool(factor.get("selected", False))
            if selected_only and not selected:
                continue
            state_a = factor["state_a"]
            state_b = factor["state_b"]
            indices_a = [int(index) for index in state_a["concept_indices"]]
            indices_b = [int(index) for index in state_b["concept_indices"]]
            direction_a = torch.nn.functional.normalize(cache["dictionary"][indices_a].mean(0), dim=0)
            direction_b = torch.nn.functional.normalize(cache["dictionary"][indices_b].mean(0), dim=0)
            factor_id = int(factor["factor_id"])
            concepts_a = ", ".join(str(word) for word in state_a.get("concepts", []))
            concepts_b = ", ".join(str(word) for word in state_b.get("concepts", []))
            interventions.append(
                Intervention(
                    identifier=f"F{factor_id}",
                    title=f"CQT factor F{factor_id}",
                    concepts=f"A: {concepts_a} ↔ B: {concepts_b}",
                    kind="cqt_quotient",
                    direction=direction_a - direction_b,
                    concept_indices=tuple(indices_a + indices_b),
                    selected=selected,
                    evidence=(
                        f"gain={float(factor.get('intervention_gain', 0.0)):.6g}, "
                        f"null threshold={float(factor.get('null_threshold', 0.0)):.6g}, "
                        f"coverage={float(factor.get('coverage', 0.0)):.2%}, "
                        f"retained edges={retained_edge_count(factor_id)}"
                    ),
                )
            )
    else:
        raise ValueError(f"Unsupported teacher graph artifact: {graph.get('artifact')!r}")
    return interventions[:max_interventions] if max_interventions else interventions


def _source_index(sample_id: str) -> int:
    prefix, separator, index = str(sample_id).rpartition(":")
    if not separator or prefix != "waterbirds":
        raise ValueError(f"Expected a Waterbirds sample ID, got {sample_id!r}.")
    return int(index)


def _candidate_positions(
    sample_ids: Sequence[str], metadata_rows: Sequence[dict], label: int, background: int
) -> torch.Tensor:
    positions = []
    for position, sample_id in enumerate(sample_ids):
        row = metadata_rows[_source_index(sample_id)]
        if int(row["y"]) == label and int(row["place"]) == background:
            positions.append(position)
    if not positions:
        raise ValueError(
            f"No cached samples found for label={LABEL_NAMES[label]}, "
            f"background={BACKGROUND_NAMES[background]}."
        )
    return torch.tensor(positions, dtype=torch.long)


def _typical_pair(
    centered: torch.Tensor,
    left_positions: torch.Tensor,
    right_positions: torch.Tensor,
    chunk_size: int,
    same_pool: bool = False,
) -> tuple[int, int, float]:
    """Choose the pair nearest the median raw cosine, independently of interventions."""

    right = centered[right_positions]
    values = []
    for start in range(0, len(left_positions), chunk_size):
        rows = left_positions[start : start + chunk_size]
        similarities = centered[rows] @ right.T
        if same_pool:
            valid = rows.view(-1, 1) < right_positions.view(1, -1)
            similarities = similarities[valid]
        else:
            similarities = similarities.flatten()
        if similarities.numel():
            values.append(similarities)
    if not values:
        raise ValueError("No distinct candidate pairs are available for this diagnostic case.")
    median = float(torch.cat(values).median())

    best_distance = float("inf")
    best_pair = (int(left_positions[0]), int(right_positions[-1]))
    for start in range(0, len(left_positions), chunk_size):
        rows = left_positions[start : start + chunk_size]
        similarities = centered[rows] @ right.T
        valid = (
            rows.view(-1, 1) < right_positions.view(1, -1)
            if same_pool
            else torch.ones_like(similarities, dtype=torch.bool)
        )
        distances = (similarities - median).abs().masked_fill(~valid, torch.inf)
        flat = int(distances.argmin())
        distance = float(distances.flatten()[flat])
        candidate = (int(rows[flat // distances.shape[1]]), int(right_positions[flat % distances.shape[1]]))
        if distance < best_distance or (distance == best_distance and candidate < best_pair):
            best_distance, best_pair = distance, candidate
    return best_pair[0], best_pair[1], median


def _image_data_uri(path: Path, max_size: int = 640) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image.thumbnail((max_size, max_size), resampling)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _image_card(
    position: int,
    cache: dict,
    metadata_rows: Sequence[dict],
    image_root: Path,
    compact: bool = False,
) -> str:
    sample_id = str(cache["sample_ids"][position])
    row = metadata_rows[_source_index(sample_id)]
    label = LABEL_NAMES[int(row["y"])]
    background = BACKGROUND_NAMES[int(row["place"])]
    image_path = image_root / str(row["img_filename"])
    if not image_path.is_file():
        raise FileNotFoundError(f"Waterbirds image not found: {image_path}")
    return f"""
      <figure class="card{' compact' if compact else ''}">
        <img src="{_image_data_uri(image_path)}" alt="{html.escape(label)}, {html.escape(background)}">
        <figcaption>
          <strong>{html.escape(label)}</strong><br>
          spurious attribute: {html.escape(background)}<br>
          <code>{html.escape(sample_id)}</code><br>
          <span class="muted">{html.escape(str(row['img_filename']))}</span>
        </figcaption>
      </figure>
    """


def _group_table(interventions: Sequence[Intervention]) -> str:
    if not interventions:
        return '<p class="warning">The teacher graph contains no selected concept groups or factors.</p>'
    rows = []
    for item in interventions:
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(item.identifier)}</strong></td>"
            f"<td>{html.escape(item.concepts)}</td>"
            f"<td>{html.escape(item.evidence)}</td>"
            f"<td>{'yes' if item.selected else 'no'}</td>"
            "</tr>"
        )
    return (
        '<table><thead><tr><th>ID</th><th>Concepts</th><th>Graph evidence</th>'
        f'<th>Selected</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _similarity_table(
    centered: torch.Tensor,
    left: int,
    right: int,
    interventions: Sequence[Intervention],
) -> str:
    before = float(torch.dot(centered[left], centered[right]))
    if not interventions:
        return (
            '<table><thead><tr><th>Intervention</th><th>Before</th><th>After</th><th>Δ</th>'
            f'</tr></thead><tbody><tr><td>none selected</td><td>{before:.6f}</td>'
            '<td>—</td><td>—</td></tr></tbody></table>'
        )
    rows = []
    pair = centered[[left, right]]
    for item in interventions:
        edited = _apply(pair, item)
        after = float(torch.dot(edited[0], edited[1]))
        delta = after - before
        delta_class = "positive" if delta > 0 else "negative" if delta < 0 else "neutral"
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(item.identifier)}</strong><br><span class=\"muted\">"
            f"{html.escape(item.concepts)}</span></td>"
            f"<td>{before:.6f}</td><td>{after:.6f}</td>"
            f"<td class=\"{delta_class}\">{delta:+.6f}</td></tr>"
        )
    return (
        '<table><thead><tr><th>Removed group / factor</th><th>Cosine before</th>'
        f'<th>Cosine after</th><th>Δ = after − before</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _graph_summary(graph: dict, cache: dict, metadata_rows: Sequence[dict]) -> str:
    indices = torch.as_tensor(graph.get("neighbor_indices", []), dtype=torch.long)
    weights = torch.as_tensor(graph.get("weights", []), dtype=torch.float32)
    valid = (indices >= 0) & (weights > 0)
    rows, slots = torch.where(valid) if valid.numel() else (torch.tensor([]), torch.tensor([]))
    columns = indices[rows, slots] if rows.numel() else torch.tensor([], dtype=torch.long)
    labels = torch.tensor(
        [int(metadata_rows[_source_index(sample_id)]["y"]) for sample_id in cache["sample_ids"]]
    )
    backgrounds = torch.tensor(
        [int(metadata_rows[_source_index(sample_id)]["place"]) for sample_id in cache["sample_ids"]]
    )
    same_target = labels[rows] == labels[columns] if rows.numel() else torch.tensor([], dtype=torch.bool)
    opposite_background = (
        backgrounds[rows] != backgrounds[columns] if rows.numel() else torch.tensor([], dtype=torch.bool)
    )
    same_target_precision = float(same_target.float().mean()) if same_target.numel() else 0.0
    cross_background_given_same = (
        float(opposite_background[same_target].float().mean()) if bool(same_target.any()) else 0.0
    )
    desired_edges = int((same_target & opposite_background).sum()) if same_target.numel() else 0
    confidence = torch.as_tensor(graph.get("anchor_confidence", []), dtype=torch.float32)
    supported_confidence = confidence[confidence > 0]
    stats = graph.get("degree_stats", {})
    config = graph.get("config", {})
    rows_html = [
        ("Selected groups / factors", str(len(graph.get("selected_group_ids", graph.get("selected_factor_ids", []))))),
        ("Supported anchors", f"{int(stats.get('supported_anchors', 0))} ({float(stats.get('coverage', 0.0)):.2%})"),
        ("Retained edges", str(int(stats.get("edge_count", int(valid.sum()))))),
        ("Same-target precision", f"{same_target_precision:.2%}"),
        ("Same-target + opposite-background edges", f"{desired_edges} ({cross_background_given_same:.2%} of same-target edges)"),
        ("Maximum indegree", str(int(stats.get("maximum_indegree", 0)))),
        ("Indegree Gini", f"{float(stats.get('indegree_gini', 0.0)):.4f}"),
        ("Effective donor count", f"{float(stats.get('effective_donor_count', 0.0)):.2f}"),
        ("Anchor confidence", (
            f"median={float(supported_confidence.median()):.6g}, mean={float(supported_confidence.mean()):.6g}, "
            f"max={float(supported_confidence.max()):.6g}" if supported_confidence.numel() else "no supported anchors"
        )),
        ("DINO / CoBalT", f"{bool(config.get('use_dino', True))} / {bool(config.get('cobalt', False))}"),
    ]
    return '<table><tbody>' + ''.join(
        f'<tr><th>{html.escape(name)}</th><td>{html.escape(value)}</td></tr>' for name, value in rows_html
    ) + '</tbody></table>'


def _typical_edge_slots(mask: torch.Tensor, confidences: torch.Tensor, count: int) -> list[tuple[int, int]]:
    rows, slots = torch.where(mask)
    if not rows.numel() or count <= 0:
        return []
    order = torch.argsort(confidences[rows, slots])
    if count == 1:
        chosen = [int(order[len(order) // 2])]
    else:
        chosen = [int(order[round(position * (len(order) - 1) / (count - 1))]) for position in range(count)]
    return [(int(rows[index]), int(slots[index])) for index in dict.fromkeys(chosen)]


def _retained_edge_examples(
    graph: dict,
    cache: dict,
    metadata_rows: Sequence[dict],
    image_root: Path,
    interventions: Sequence[Intervention],
    edges_per_group: int,
) -> str:
    if edges_per_group == 0 or not interventions:
        return ""
    indices = torch.as_tensor(graph.get("neighbor_indices", []), dtype=torch.long)
    weights = torch.as_tensor(graph.get("weights", []), dtype=torch.float32)
    group_ids = torch.as_tensor(graph.get("group_ids", []), dtype=torch.long)
    edge_confidences = torch.as_tensor(graph.get("edge_confidences", []), dtype=torch.float32)
    gains = torch.as_tensor(graph.get("intervention_gains", []), dtype=torch.float32)
    anchor_confidence = torch.as_tensor(graph.get("anchor_confidence", []), dtype=torch.float32)
    if not indices.numel() or indices.shape != weights.shape or group_ids.shape != weights.shape:
        return '<p class="warning">Graph artifact does not contain retained-edge tensors.</p>'

    sections = []
    for intervention in interventions:
        numeric_id = int(intervention.identifier[1:])
        edge_slots = _typical_edge_slots(
            (group_ids == numeric_id) & (indices >= 0) & (weights > 0),
            edge_confidences,
            edges_per_group,
        )
        if not edge_slots:
            continue
        cards = []
        for row, slot in edge_slots:
            column = int(indices[row, slot])
            pair = cache["centered_clip"][[row, column]]
            edited = _apply(pair, intervention)
            raw_cosine = float(torch.dot(pair[0], pair[1]))
            projected_cosine = float(torch.dot(edited[0], edited[1]))
            dino_similarity = "disabled"
            if bool(graph.get("config", {}).get("use_dino", True)) and cache.get("dino_embeddings") is not None:
                dino_similarity = f"{float(torch.dot(cache['dino_embeddings'][row], cache['dino_embeddings'][column])):.6f}"
            left_meta = metadata_rows[_source_index(cache["sample_ids"][row])]
            right_meta = metadata_rows[_source_index(cache["sample_ids"][column])]
            same_target = int(left_meta["y"]) == int(right_meta["y"])
            opposite_background = int(left_meta["place"]) != int(right_meta["place"])
            activation_text = "n/a"
            if intervention.kind == "crp_projection" and intervention.concept_indices:
                activation = cache["splice_codes"][:, list(intervention.concept_indices)].sum(dim=1)
                activation_text = f"{float(activation[row]):.6g} / {float(activation[column]):.6g}"
            cards.append(
                f'<article class="edge-example"><div class="images compact-grid">'
                f'{_image_card(row, cache, metadata_rows, image_root, compact=True)}'
                f'{_image_card(column, cache, metadata_rows, image_root, compact=True)}</div>'
                f'<table><tbody>'
                f'<tr><th>Raw / projected cosine</th><td>{raw_cosine:.6f} / {projected_cosine:.6f}</td></tr>'
                f'<tr><th>Stored gain</th><td>{float(gains[row, slot]):+.6f}</td></tr>'
                f'<tr><th>Group activation A / B</th><td>{activation_text}</td></tr>'
                f'<tr><th>DINO cosine</th><td>{dino_similarity}</td></tr>'
                f'<tr><th>Edge confidence / teacher weight</th><td>{float(edge_confidences[row, slot]):.6g} / {float(weights[row, slot]):.6g}</td></tr>'
                f'<tr><th>Anchor confidence</th><td>{float(anchor_confidence[row]):.6g}</td></tr>'
                f'<tr><th>Post-hoc relation</th><td>same target={same_target}; opposite background={opposite_background}</td></tr>'
                f'</tbody></table></article>'
            )
        sections.append(
            f'<section><h3>{html.escape(intervention.identifier)}: {html.escape(intervention.concepts)}</h3>'
            f'<p class="muted">Typical retained edges are selected by median edge confidence, not maximum effect.</p>'
            f'{"".join(cards)}</section>'
        )
    return ''.join(sections)


def generate_report(
    cache_path: Path,
    graph_path: Path,
    data_folder: Path,
    output_path: Path,
    scope: str = "selected",
    selection_chunk_size: int = 512,
    max_interventions: int = 0,
    edges_per_group: int = 1,
) -> Path:
    graph = load_graph_json(graph_path)
    cache = validate_feature_cache(
        torch.load(cache_path, map_location="cpu", weights_only=True),
        require_dino=False,
    )
    if list(map(str, graph.get("sample_ids", []))) != list(map(str, cache["sample_ids"])):
        raise ValueError("Teacher graph and feature cache sample IDs are not aligned.")
    if scope not in {"selected", "all"}:
        raise ValueError("scope must be selected or all.")
    if selection_chunk_size <= 0:
        raise ValueError("selection_chunk_size must be positive.")
    if max_interventions < 0:
        raise ValueError("max_interventions must be non-negative; 0 shows every eligible item.")
    if edges_per_group < 0:
        raise ValueError("edges_per_group must be non-negative.")

    dataset = WaterbirdsDataset(str(data_folder))
    metadata_rows = dataset.metadata_df.to_dict(orient="records")
    image_root = Path(dataset.data_dir)
    interventions = _interventions(graph, cache, scope, max_interventions=max_interventions)
    pair_sections = []
    for spec in PAIR_SPECS:
        left_candidates = _candidate_positions(
            cache["sample_ids"], metadata_rows, spec.left_label, spec.left_background
        )
        right_candidates = _candidate_positions(
            cache["sample_ids"], metadata_rows, spec.right_label, spec.right_background
        )
        left, right, median_cosine = _typical_pair(
            cache["centered_clip"],
            left_candidates,
            right_candidates,
            selection_chunk_size,
            same_pool=spec.same_pool,
        )
        selection_note = (
            "The pair was selected deterministically: its raw cosine is closest to the "
            f"median of all eligible pairs of this type (median={median_cosine:.6f})."
        )
        pair_sections.append(
            f"""
            <section>
              <h2>{html.escape(spec.title)}</h2>
              <p>{html.escape(spec.explanation)}</p>
              <p class="muted">{html.escape(selection_note)} Pair selection does not use concept-intervention results.</p>
              <div class="images">
                {_image_card(left, cache, metadata_rows, image_root)}
                {_image_card(right, cache, metadata_rows, image_root)}
              </div>
              <h3>Selected concept groups / factors</h3>
              {_group_table(interventions)}
              <h3>Cosine similarity before and after removal</h3>
              {_similarity_table(cache['centered_clip'], left, right, interventions)}
            </section>
            """
        )

    mode_name = "CRP full-subspace projection" if graph["artifact"] in CRP_ARTIFACTS else "CQT rank-one quotient"
    audited_count = len(graph.get("groups", graph.get("factors", [])))
    all_items = graph.get("groups", graph.get("factors", []))
    selected_count = sum(bool(item.get("selected", False)) for item in all_items)
    retained_edges = _retained_edge_examples(
        graph,
        cache,
        metadata_rows,
        image_root,
        interventions,
        edges_per_group,
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spur SpLiCE concept-ablation sanity-check</title>
  <style>
    :root {{ color-scheme: light; --ink:#18202b; --muted:#667085; --line:#d8dee8; --paper:#fff; --bg:#f4f6f9; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }}
    main {{ max-width:1180px; margin:32px auto; padding:0 20px 48px; }}
    header, section {{ background:var(--paper); border:1px solid var(--line); border-radius:14px; padding:24px; margin-bottom:22px; box-shadow:0 4px 18px #18202b0a; }}
    h1,h2,h3 {{ line-height:1.2; }} h1 {{ margin-top:0; }} h2 {{ margin-top:0; }}
    .images {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    .card {{ margin:0; border:1px solid var(--line); border-radius:12px; overflow:hidden; background:#fafbfc; }}
    .card img {{ display:block; width:100%; height:360px; object-fit:contain; background:#e9edf2; }}
    .card.compact img {{ height:180px; }} .compact-grid {{ max-width:720px; }}
    figcaption {{ padding:12px 14px; }} .muted {{ color:var(--muted); }}
    table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
    th,td {{ padding:10px 12px; border:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ background:#f0f3f7; }} .positive {{ color:#067647; font-weight:700; }}
    .negative {{ color:#b42318; font-weight:700; }} .neutral {{ color:var(--muted); }}
    .warning {{ padding:12px; border-left:4px solid #f79009; background:#fffaeb; }}
    .edge-example {{ padding:14px; margin:14px 0; border:1px solid var(--line); border-radius:12px; }}
    code {{ overflow-wrap:anywhere; }}
    @media (max-width:760px) {{ .images {{ grid-template-columns:1fr; }} .card img {{ height:280px; }} table {{ font-size:13px; }} }}
  </style>
</head>
<body><main>
  <header>
    <h1>Spur SpLiCE: concept-ablation sanity-check</h1>
    <p><strong>Intervention:</strong> {html.escape(mode_name)}.</p>
    <p><strong>Graph:</strong> <code>{html.escape(str(graph_path))}</code></p>
    <p><strong>Scope:</strong> {html.escape(scope)}; audited={audited_count}, selected={selected_count}, shown={len(interventions)}.</p>
    <p class="muted">Displayed interventions are sorted by the number of retained teacher edges, then by the label-free null margin. The display limit does not change the teacher graph.</p>
    <p>All values use the full centered CLIP embeddings from the frozen cache. Waterbirds metadata is used only after graph discovery to select four diagnostic pairs, annotate images, and compute post-hoc edge metrics.</p>
  </header>
  <section>
    <h2>Aggregate graph summary</h2>
    {_graph_summary(graph, cache, metadata_rows)}
    <details><summary>Complete graph configuration</summary><pre><code>{html.escape(json.dumps(graph.get('config', {}), indent=2, sort_keys=True))}</code></pre></details>
    <h3>Selected concept groups / factors</h3>
    {_group_table(interventions)}
  </section>
  {''.join(pair_sections)}
  <section><h2>Typical retained teacher edges</h2><p>These pairs are actual edges in the teacher graph.</p></section>
  {retained_edges}
</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--data-folder", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scope", choices=("selected", "all"), default="selected")
    parser.add_argument("--selection-chunk-size", type=int, default=512)
    parser.add_argument("--max-interventions", type=int, default=0)
    parser.add_argument("--edges-per-group", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = generate_report(
        args.cache,
        args.graph,
        args.data_folder,
        args.output,
        scope=args.scope,
        selection_chunk_size=args.selection_chunk_size,
        max_interventions=args.max_interventions,
        edges_per_group=args.edges_per_group,
    )
    print(f"[INFO] Wrote self-contained concept-ablation report to {path}")


if __name__ == "__main__":
    main()
