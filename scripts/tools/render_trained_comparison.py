"""Render a self-contained trained SimCLR vs CoSpRo Waterbirds comparison.

The report keeps pair selection fixed by the baseline encoder, then evaluates the
same images with both trained encoders.  Teacher-graph fields are shown separately
so that a missing retained edge is not confused with a zero-weight edge.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from experiments.spurious_eval.datasets.waterbirds import WaterbirdsDataset, waterbirds_transforms
from experiments.spurious_eval.models.resnet import build_resnet_encoder
from experiments.spurious_eval.training.checkpointing import load_encoder_checkpoint
from splice.crp import orthonormal_basis, project_out, validate_feature_cache
from splice.graph_io import load_graph_json


LABEL_NAMES = {0: "landbird", 1: "waterbird"}
BACKGROUND_NAMES = {0: "land background", 1: "water background"}


PAIR_CASES = (
    {
        "key": "same_target_cross_background",
        "title": "Same target · different background",
        "description": "Two birds of the same target class, one on land and one on water.",
        "predicate": lambda y1, a1, y2, a2: y1 == y2 and a1 != a2,
    },
    {
        "key": "cross_target_same_background",
        "title": "Different target · same background",
        "description": "A landbird and a waterbird with the same background.",
        "predicate": lambda y1, a1, y2, a2: y1 != y2 and a1 == a2,
    },
    {
        "key": "same_target_same_background",
        "title": "Same target · same background",
        "description": "A within-subgroup control: same target and same background.",
        "predicate": lambda y1, a1, y2, a2: y1 == y2 and a1 == a2,
    },
    {
        "key": "cross_target_cross_background",
        "title": "Different target · different background",
        "description": "A negative-control pair with both target and background different.",
        "predicate": lambda y1, a1, y2, a2: y1 != y2 and a1 != a2,
    },
)


class CachedImageDataset(Dataset):
    def __init__(self, dataset: WaterbirdsDataset, source_indices: Sequence[int], transform) -> None:
        self.dataset = dataset
        self.source_indices = list(source_indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, position: int):
        image = self.dataset.get_input(self.source_indices[position])
        return self.transform(image), position


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _source_index(sample_id: str) -> int:
    prefix, separator, index = str(sample_id).rpartition(":")
    if not separator or prefix != "waterbirds" or not index.isdigit():
        raise ValueError(f"Expected a waterbirds sample ID, got {sample_id!r}.")
    return int(index)


def _image_uri(path: Path, max_size: int = 520) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image.thumbnail((max_size, max_size), resampling)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=86, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    value = float(value)
    if not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{100.0 * float(value):.{digits}f}%"


def _metadata(dataset: WaterbirdsDataset, sample_ids: Sequence[str]) -> list[dict]:
    records = []
    metadata = dataset.metadata_df
    for sample_id in sample_ids:
        row = metadata.iloc[_source_index(sample_id)]
        y, background = int(row["y"]), int(row["place"])
        records.append(
            {
                "target_id": y,
                "background_id": background,
                "target": LABEL_NAMES[y],
                "background": BACKGROUND_NAMES[background],
                "subgroup": f"{LABEL_NAMES[y]} · {BACKGROUND_NAMES[background]}",
                "sample_id": str(sample_id),
                "image_path": str(Path(dataset.data_dir) / str(row["img_filename"])),
            }
        )
    return records


def _load_encoder(checkpoint: Path, model_name: str, device: torch.device) -> torch.nn.Module:
    encoder, _ = build_resnet_encoder(model_name, load_pretrained_weights=False)
    load_encoder_checkpoint(encoder, str(checkpoint))
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def _encode(
    encoder: torch.nn.Module,
    dataset: WaterbirdsDataset,
    source_indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> torch.Tensor:
    _, _, transform = waterbirds_transforms(image_size=224)
    loader = DataLoader(
        CachedImageDataset(dataset, source_indices, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    features = []
    with torch.inference_mode():
        for images, positions in loader:
            values = F.normalize(encoder(images.to(device, non_blocking=True)), dim=1)
            block = torch.zeros((len(positions), values.shape[1]), dtype=torch.float32)
            block[positions] = values.cpu()
            features.append(block)
    return torch.cat(features, dim=0)


def _extreme_pair(
    features: torch.Tensor,
    records: Sequence[dict],
    predicate,
    want_high: bool,
    block_size: int = 512,
) -> tuple[int, int, float]:
    """Find a deterministic min/max cosine pair without materialising N² values."""

    normalized = F.normalize(features.float(), dim=1)
    best_value = -float("inf") if want_high else float("inf")
    best_pair = None
    n = len(records)
    for left_start in range(0, n, block_size):
        left_end = min(n, left_start + block_size)
        for right_start in range(left_start, n, block_size):
            right_end = min(n, right_start + block_size)
            similarities = normalized[left_start:left_end] @ normalized[right_start:right_end].T
            valid = torch.ones_like(similarities, dtype=torch.bool)
            if left_start == right_start:
                valid = torch.triu(valid, diagonal=1)
            for i in range(left_end - left_start):
                left = left_start + i
                for j in torch.where(valid[i])[0].tolist():
                    right = right_start + j
                    a, b = records[left], records[right]
                    if not predicate(a["target_id"], a["background_id"], b["target_id"], b["background_id"]):
                        continue
                    value = float(similarities[i, j])
                    pair = (left, right)
                    better = value > best_value if want_high else value < best_value
                    if better or (value == best_value and (best_pair is None or pair < best_pair)):
                        best_value, best_pair = value, pair
    if best_pair is None:
        raise ValueError("No eligible pair found for one of the requested (target, background) cases.")
    return best_pair[0], best_pair[1], best_value


def _pair_edge(graph: dict, left: int, right: int) -> dict | None:
    indices = graph.get("neighbor_indices")
    if indices is None:
        return None
    for source, donor in ((left, right), (right, left)):
        slots = torch.where(indices[source] == donor)[0]
        for slot in slots.tolist():
            if float(graph["weights"][source, slot]) <= 0:
                continue
            group_id = int(graph["group_ids"][source, slot])
            groups = {int(group["group_id"]): group for group in graph.get("groups", [])}
            return {
                "direction": f"{source} → {donor}",
                "group_id": group_id,
                "group": groups.get(group_id, {}),
                "weight": float(graph["weights"][source, slot]),
                "edge_confidence": float(graph["edge_confidences"][source, slot]),
                "anchor_confidence": float(graph["anchor_confidence"][source]),
                "gain": float(graph["intervention_gains"][source, slot]),
            }
    return None


def _projected_cosine(
    raw_features: torch.Tensor,
    dictionary: torch.Tensor,
    left: int,
    right: int,
    group: dict,
) -> float | None:
    """Cosine after removing the selected group's dictionary subspace."""

    members = [int(index) for index in group.get("concept_indices", [])]
    if not members:
        return None
    basis = orthonormal_basis(dictionary[members])
    projected = project_out(raw_features[[left, right]], basis)
    norms = projected.norm(dim=1)
    if bool((norms <= 1e-12).any()):
        return None
    projected = F.normalize(projected, dim=1)
    return float(projected[0] @ projected[1])


def _edge_examples(
    graph: dict,
    raw_features: torch.Tensor,
    records: Sequence[dict],
    max_groups: int,
    per_group: int,
) -> list[dict]:
    indices = graph.get("neighbor_indices")
    weights = graph.get("weights")
    group_ids = graph.get("group_ids")
    if indices is None or weights is None or group_ids is None:
        return []
    valid = (indices >= 0) & (weights > 0)
    counts = {}
    for group_id in torch.unique(group_ids[valid]).tolist() if bool(valid.any()) else []:
        counts[int(group_id)] = int(((group_ids == group_id) & valid).sum())
    selected = [group for group in graph.get("groups", []) if group.get("selected")]
    selected.sort(key=lambda group: (-counts.get(int(group["group_id"]), 0), -float(group.get("score", 0.0)), int(group["group_id"])))
    selected = selected[:max_groups]
    output = []
    for group in selected:
        group_id = int(group["group_id"])
        rows, slots = torch.where((group_ids == group_id) & valid)
        candidates = []
        for row, slot in zip(rows.tolist(), slots.tolist()):
            donor = int(indices[row, slot])
            cosine = float(raw_features[row] @ raw_features[donor])
            candidates.append((cosine, row, slot, donor))
        candidates.sort(key=lambda item: (item[0], item[1], item[3]))
        low_count = per_group // 2
        high_count = per_group - low_count
        chosen = candidates[:low_count] + candidates[-high_count:]
        seen = set()
        for cosine, row, slot, donor in chosen:
            key = (row, slot)
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "group": group,
                    "left": row,
                    "right": donor,
                    "raw_clip_cosine": cosine,
                    "weight": float(weights[row, slot]),
                    "edge_confidence": float(graph["edge_confidences"][row, slot]),
                    "anchor_confidence": float(graph["anchor_confidence"][row]),
                    "gain": float(graph["intervention_gains"][row, slot]),
                }
            )
    return output


def _image_card(record: dict, uri: str, extra: str = "") -> str:
    return (
        '<figure class="image-card">'
        f'<img src="{uri}" alt="{html.escape(record["subgroup"])}">'
        f'<figcaption><strong>{html.escape(record["target"])}</strong> · '
        f'{html.escape(record["background"])}<small>{html.escape(record["sample_id"])}</small>'
        f'{extra}</figcaption></figure>'
    )


def _metric(label: str, value: str, note: str) -> str:
    return f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong><small>{html.escape(note)}</small></div>'


def _pair_section(
    pair: dict,
    records: Sequence[dict],
    uris: Sequence[str],
    raw_features: torch.Tensor,
    dictionary: torch.Tensor,
    baseline_features: torch.Tensor,
    method_features: torch.Tensor,
    graph: dict,
) -> str:
    left, right = pair["left"], pair["right"]
    raw_clip = float(raw_features[left] @ raw_features[right])
    baseline = float(baseline_features[left] @ baseline_features[right])
    method = float(method_features[left] @ method_features[right])
    edge = _pair_edge(graph, left, right)
    projected = _projected_cosine(raw_features, dictionary, left, right, edge["group"]) if edge else None
    relation = f"{records[left]['subgroup']} ↔ {records[right]['subgroup']}"
    if edge is None:
        edge_html = '<div class="edge-missing">This exact pair is not a retained teacher edge.</div>'
    else:
        concepts = ", ".join(edge["group"].get("concepts", [])) or f"G{edge['group_id']}"
        edge_html = (
            '<div class="edge-found"><span class="eyebrow">Retained teacher edge</span>'
            f'<strong>G{edge["group_id"]} · {html.escape(concepts)}</strong>'
            f'<span>{html.escape(edge["direction"])} · weight {_fmt(edge["weight"], 6)} · '
            f'edge confidence {_fmt(edge["edge_confidence"], 6)} · anchor confidence {_fmt(edge["anchor_confidence"], 6)} · '
            f'gain {edge["gain"]:+.6f}</span></div>'
        )
    return f"""
    <section class="pair-section">
      <div class="section-head"><div><span class="eyebrow">{html.escape(pair['level'])} initial cosine</span>
        <h2>{html.escape(pair['title'])}</h2><p>{html.escape(pair['description'])}</p></div>
        <span class="case-chip">{html.escape(pair['level'])}</span></div>
      <div class="pair-images">
        {_image_card(records[left], uris[left])}
        {_image_card(records[right], uris[right])}
      </div>
      <div class="comparison-grid">
        <div><span class="eyebrow">Same pair · encoder cosine</span>
          <div class="score-row raw"><span>Raw CLIP</span><strong>{raw_clip:.6f}</strong></div>
          <div class="score-row"><span>Baseline SimCLR</span><strong>{baseline:.6f}</strong></div>
          <div class="score-row method"><span>CoSpRo</span><strong>{method:.6f}</strong></div>
          <div class="score-row projected"><span>Raw CLIP after graph projection</span><strong>{_fmt(projected, 6)}</strong></div>
          <div class="delta">Δ CoSpRo − baseline <b>{method - baseline:+.6f}</b></div>
        </div>
        <div><span class="eyebrow">Post-hoc relation</span><p class="relation">{html.escape(relation)}</p>{edge_html}</div>
      </div>
    </section>
    """


def _group_section(item: dict, records: Sequence[dict], uris: Sequence[str]) -> str:
    group = item["group"]
    concepts = ", ".join(group.get("concepts", []))
    activation = []
    for position in range(len(records)):
        # The graph's cache rows and the report rows are aligned by construction.
        activation.append((float(item.get("group_activation", {}).get(str(position), 0.0)), position))
    activation.sort(key=lambda x: (-x[0], x[1]))
    top = activation[:4]
    images = "".join(
        _image_card(records[position], uris[position], f'<small>group activation {_fmt(value, 4)}</small>')
        for value, position in top
    )
    return f"""
    <details class="group-card" open>
      <summary><span><b>G{int(group['group_id'])}</b> · {html.escape(concepts)}</span>
        <small>{int(item['edge_count'])} retained edges · {html.escape(item['cosine_range'])}</small></summary>
      <div class="group-body">
        <div class="group-evidence"><span><b>selection</b> {html.escape(str(group.get('selected')))}</span>
          <span><b>score</b> {_fmt(group.get('score'), 6)}</span>
          <span><b>null excess</b> {_fmt(group.get('null_excess_score'), 6)}</span>
          <span><b>null ratio</b> {_pct(group.get('null_excess_ratio'))}</span>
          <span><b>null threshold</b> {_fmt(group.get('null_threshold'), 6)}</span>
          <span><b>rank</b> {int(group.get('basis_rank', len(group.get('concept_indices', []))))}</span>
          <span><b>coverage</b> {_pct(group.get('coverage'))}</span>
          <span><b>semantic agreement</b> {_fmt(group.get('semantic_agreement'), 4)}</span>
          <span><b>turnover</b> {_pct(group.get('mean_neighbor_turnover'))}</span></div>
        <div class="contact-sheet">{images}</div>
      </div>
    </details>
    """


def _edge_section(
    item: dict,
    records: Sequence[dict],
    uris: Sequence[str],
    raw_features: torch.Tensor,
    dictionary: torch.Tensor,
    baseline_features: torch.Tensor,
    method_features: torch.Tensor,
) -> str:
    left, right = item["left"], item["right"]
    raw_clip = item["raw_clip_cosine"]
    baseline = float(baseline_features[left] @ baseline_features[right])
    group = item["group"]
    projected = _projected_cosine(raw_features, dictionary, left, right, group)
    method = float(method_features[left] @ method_features[right])
    concepts = ", ".join(group.get("concepts", []))
    return f"""
    <article class="edge-card">
      <div class="edge-title"><span><b>G{int(group['group_id'])}</b> · {html.escape(concepts)}</span>
        <span class="case-chip">raw clip {raw_clip:.3f}</span></div>
      <div class="pair-images">
        {_image_card(records[left], uris[left], f'<small>activation {_fmt(item.get("left_activation"), 4)}</small>')}
        {_image_card(records[right], uris[right], f'<small>activation {_fmt(item.get("right_activation"), 4)}</small>')}
      </div>
      <table class="data-table"><tbody>
        <tr><th>Raw CLIP cosine</th><td>{raw_clip:.6f} → graph projection {_fmt(projected, 6)} ({'' if projected is None else f'{projected - raw_clip:+.6f}'})</td></tr>
        <tr><th>Trained encoder cosine</th><td>SimCLR {baseline:.6f} → CoSpRo {method:.6f} ({method - baseline:+.6f})</td></tr>
        <tr><th>Teacher weight</th><td>{item['weight']:.6f} · row-normalized over retained outgoing edges</td></tr>
        <tr><th>Edge confidence</th><td>{item['edge_confidence']:.6f}</td></tr>
        <tr><th>Anchor confidence</th><td>{item['anchor_confidence']:.6f}</td></tr>
        <tr><th>Intervention gain</th><td>{item['gain']:+.6f}</td></tr>
        <tr><th>Post-hoc relation</th><td>{html.escape(records[left]['subgroup'])} ↔ {html.escape(records[right]['subgroup'])}</td></tr>
      </tbody></table>
    </article>
    """


def _build_html(
    output: Path,
    records: Sequence[dict],
    uris: Sequence[str],
    raw_features: torch.Tensor,
    dictionary: torch.Tensor,
    baseline_features: torch.Tensor,
    method_features: torch.Tensor,
    graph: dict,
    pairs: Sequence[dict],
    groups: Sequence[dict],
    edges: Sequence[dict],
    baseline_checkpoint: Path,
    method_checkpoint: Path,
    method_name: str,
    seed: int,
    graph_path: Path,
) -> None:
    valid = (graph.get("neighbor_indices", torch.empty(0)) >= 0) & (graph.get("weights", torch.empty(0)) > 0)
    stats = graph.get("degree_stats", {})
    group_html = "".join(_group_section(item, records, uris) for item in groups)
    edge_html = "".join(_edge_section(item, records, uris, raw_features, dictionary, baseline_features, method_features) for item in edges)
    pair_html = "".join(_pair_section(pair, records, uris, raw_features, dictionary, baseline_features, method_features, graph) for pair in pairs)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trained representation comparison · seed {seed}</title>
<style>
:root {{ --ink:#172033; --muted:#647084; --line:#dbe1ea; --paper:#fff; --wash:#eef2f6; --navy:#17233b; --blue:#2864d7; --orange:#c56a27; --green:#18734d; --red:#a52b2b; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--wash); color:var(--ink); font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ width:min(1420px,calc(100% - 34px)); margin:24px auto 60px; }} header,.panel,.pair-section,.edge-card {{ background:var(--paper); border:1px solid var(--line); border-radius:16px; box-shadow:0 10px 30px rgba(31,47,74,.07); }}
header {{ overflow:hidden; }} .hero {{ padding:32px 36px; background:linear-gradient(120deg,#17233b 0%,#243b62 68%,#325f91 100%); color:white; }}
h1 {{ max-width:860px; margin:4px 0 12px; font-size:clamp(2rem,4vw,3.5rem); line-height:1.02; letter-spacing:-.045em; }} h2 {{ margin:0 0 6px; line-height:1.1; letter-spacing:-.02em; }} h3 {{ margin:0 0 10px; }} p {{ margin:8px 0; }}
.eyebrow {{ display:block; color:var(--muted); text-transform:uppercase; letter-spacing:.1em; font-size:.7rem; font-weight:800; }} .hero .eyebrow {{ color:#b8c9e4; }} .hero p {{ color:#d7e1f0; max-width:90ch; }}
.metric-rail {{ display:grid; grid-template-columns:repeat(6,1fr); gap:1px; background:var(--line); }} .metric {{ min-height:104px; padding:16px 18px; background:white; }} .metric span,.metric small {{ display:block; color:var(--muted); }} .metric strong {{ display:block; margin:7px 0 3px; font-size:1.4rem; font-variant-numeric:tabular-nums; }}
.panel,.pair-section {{ margin-top:22px; padding:26px 30px; }} .section-head {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-start; margin-bottom:18px; }} .section-head p {{ color:var(--muted); }}
.case-chip {{ display:inline-flex; padding:5px 9px; border-radius:999px; background:#edf2fb; color:#2458b5; font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.06em; white-space:nowrap; }}
.pair-section {{ border-top:4px solid var(--blue); }} .pair-images,.contact-sheet {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }} .image-card {{ margin:0; overflow:hidden; border:1px solid var(--line); border-radius:12px; background:#f7f9fc; }} .image-card img {{ display:block; width:100%; height:280px; object-fit:contain; background:#e7edf4; }} .image-card figcaption {{ display:grid; gap:3px; padding:10px 12px; }} .image-card small {{ color:var(--muted); overflow-wrap:anywhere; }}
.comparison-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; margin-top:18px; padding-top:18px; border-top:1px solid var(--line); }} .score-row {{ display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }} .score-row strong {{ color:var(--navy); }} .score-row.method strong {{ color:var(--orange); }} .delta {{ margin-top:10px; color:var(--muted); }} .delta b {{ color:var(--orange); font-variant-numeric:tabular-nums; }} .relation {{ font-weight:700; }}
.edge-found,.edge-missing {{ display:grid; gap:4px; margin-top:12px; padding:12px 14px; border-radius:10px; }} .edge-found {{ background:#eaf6ef; color:#14573b; }} .edge-missing {{ background:#fff4e8; color:#8a4e1e; }} .edge-found .eyebrow {{ color:#377e5c; }}
.group-card {{ margin-top:10px; border-bottom:1px solid var(--line); }} .group-card summary {{ display:flex; justify-content:space-between; gap:20px; padding:15px 2px; cursor:pointer; font-size:1rem; }} .group-card summary small {{ color:var(--muted); }} .group-body {{ padding:4px 2px 22px; }} .group-evidence {{ display:flex; flex-wrap:wrap; gap:8px 18px; margin-bottom:14px; color:var(--muted); font-variant-numeric:tabular-nums; }} .group-evidence b {{ color:var(--ink); }} .group-body .contact-sheet {{ grid-template-columns:repeat(4,minmax(0,1fr)); }} .group-body .image-card img,.edge-card .image-card img {{ height:150px; }}
.edge-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }} .edge-card {{ padding:18px; }} .edge-title {{ display:flex; justify-content:space-between; gap:10px; margin-bottom:12px; }} .data-table {{ width:100%; margin-top:12px; border-collapse:collapse; font-variant-numeric:tabular-nums; }} .data-table th,.data-table td {{ padding:8px; border-top:1px solid var(--line); text-align:left; vertical-align:top; }} .data-table th {{ width:34%; color:var(--muted); font-weight:600; }}
.note {{ color:var(--muted); max-width:95ch; }} footer {{ padding:24px 4px; color:var(--muted); font-size:.9rem; }} code {{ overflow-wrap:anywhere; }}
@media(max-width:900px) {{ .metric-rail {{ grid-template-columns:repeat(3,1fr); }} .comparison-grid,.edge-grid {{ grid-template-columns:1fr; }} .group-body .contact-sheet {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
@media(max-width:620px) {{ main {{ width:min(100% - 16px,1420px); margin-top:8px; }} .hero,.panel,.pair-section {{ padding:21px 18px; }} .metric-rail {{ grid-template-columns:repeat(2,1fr); }} .pair-images,.contact-sheet,.group-body .contact-sheet {{ grid-template-columns:1fr; }} .section-head,.group-card summary,.edge-title {{ display:grid; }} .image-card img {{ height:230px; }} }}
</style></head><body><main>
<header><div class="hero"><span class="eyebrow">Frozen pair selection · trained encoders · seed {seed}</span><h1>What changed after CoSpRo training?</h1>
<p>Exactly the same Waterbirds images are evaluated by raw frozen CLIP, a trained baseline encoder, and a trained {html.escape(method_name)} encoder. Pairs are selected by raw CLIP cosine, so “high” and “low” remain fixed during comparison.</p></div>
<div class="metric-rail">
  {_metric('Training baseline', 'SimCLR', 'checkpoint encoder')}
  {_metric('Method', method_name, 'checkpoint encoder')}
  {_metric('Teacher edges', f"{int(stats.get('edge_count', int(valid.sum()) if valid.numel() else 0)):,}", 'retained edges')}
  {_metric('Supported anchors', _pct(float(stats.get('coverage', 0.0))), 'graph coverage')}
  {_metric('Selected groups', str(len(graph.get('selected_group_ids', []))), 'teacher graph')}
  {_metric('Report scope', f"{len(pairs)} pairs", 'high + low per relation')}
</div></header>
<section class="panel"><span class="eyebrow">How to read this</span><h2>Two views of the same method</h2>
<p class="note"><b>Pair view:</b> high/low initial cosine is measured with raw frozen CLIP and then held fixed; the same pair is evaluated in raw CLIP, after graph projection, trained SimCLR, and trained {html.escape(method_name)}. <b>Edge view:</b> examples are actual retained teacher edges, selected from the lowest and highest raw-CLIP cosine cases within each displayed concept group.</p>
<p class="note"><b>Teacher weight</b> is row-normalized over retained outgoing edges. <b>Edge confidence</b> is the unnormalized evidence score; <b>anchor confidence</b> is the strongest outgoing edge confidence for the anchor. A pair can be visually similar yet absent from the graph.</p></section>
<section><span class="eyebrow">1 · Fixed relation cases</span><h2>High and low initial cosine</h2>{pair_html}</section>
<section class="panel"><span class="eyebrow">2 · Frozen teacher graph</span><h2>Which concept groups survived selection?</h2><p class="note">Groups and images are label-free during construction; target/background labels are attached only for this post-hoc visualization.</p>{group_html or '<p class="note">No selected groups were found in the supplied graph.</p>'}</section>
<section class="panel"><span class="eyebrow">3 · Actual retained edges</span><h2>Why did these pairs receive these weights?</h2><p class="note">Each card is a real teacher edge. The group, activations, confidence, weight and intervention gain are shown together with the trained-encoder cosine comparison.</p><div class="edge-grid">{edge_html or '<p class="note">No retained edge examples were available.</p>'}</div></section>
<footer>Baseline checkpoint: <code>{html.escape(str(baseline_checkpoint))}</code><br>Method checkpoint: <code>{html.escape(str(method_checkpoint))}</code><br>Teacher graph: <code>{html.escape(str(graph_path))}</code></footer>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def _find_checkpoint(root: Path) -> Path:
    candidates = sorted(root.glob("training/*/last.pth"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one training/*/last.pth below {root}, found {len(candidates)}")
    return candidates[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--data-folder", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path, help="Teacher graph used by the method")
    parser.add_argument("--baseline-ckpt", type=Path)
    parser.add_argument("--method-ckpt", type=Path)
    parser.add_argument("--baseline-root", type=Path, help="Run directory containing training/*/last.pth")
    parser.add_argument("--method-root", type=Path, help="Run directory containing training/*/last.pth")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="resnet18_large")
    parser.add_argument("--method-name", default="CoSpRo")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-groups", type=int, default=6)
    parser.add_argument("--edges-per-group", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.baseline_ckpt is None:
        if args.baseline_root is None:
            raise ValueError("Provide --baseline-ckpt or --baseline-root")
        args.baseline_ckpt = _find_checkpoint(args.baseline_root)
    if args.method_ckpt is None:
        if args.method_root is None:
            raise ValueError("Provide --method-ckpt or --method-root")
        args.method_ckpt = _find_checkpoint(args.method_root)
    for path in (args.cache, args.graph, args.baseline_ckpt, args.method_ckpt):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.max_groups <= 0 or args.edges_per_group <= 0:
        raise ValueError("--max-groups and --edges-per-group must be positive")
    if args.workers < 0 or args.batch_size <= 0:
        raise ValueError("--workers must be non-negative and --batch-size must be positive")

    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    cache = validate_feature_cache(torch.load(args.cache, map_location="cpu", weights_only=True))
    graph = load_graph_json(args.graph)
    if list(map(str, graph.get("sample_ids", []))) != list(map(str, cache["sample_ids"])):
        raise ValueError("Teacher graph and feature cache sample IDs are not aligned")
    dataset = WaterbirdsDataset(str(args.data_folder))
    records = _metadata(dataset, cache["sample_ids"])
    source_indices = [_source_index(sample_id) for sample_id in cache["sample_ids"]]
    baseline_encoder = _load_encoder(args.baseline_ckpt, args.model, device)
    method_encoder = _load_encoder(args.method_ckpt, args.model, device)
    raw_features = F.normalize(cache["centered_clip"].float(), dim=1)
    baseline_features = _encode(baseline_encoder, dataset, source_indices, device, args.batch_size, args.workers)
    method_features = _encode(method_encoder, dataset, source_indices, device, args.batch_size, args.workers)
    if baseline_features.shape != method_features.shape:
        raise ValueError("Baseline and method feature shapes differ")

    pairs = []
    for case in PAIR_CASES:
        for want_high, level in ((True, "high"), (False, "low")):
            left, right, value = _extreme_pair(raw_features, records, case["predicate"], want_high)
            pairs.append({**case, "left": left, "right": right, "initial_cosine": value, "level": level})

    edge_items = _edge_examples(graph, raw_features, records, args.max_groups, args.edges_per_group)
    group_items = []
    for edge in edge_items:
        group_id = int(edge["group"]["group_id"])
        existing = next((item for item in group_items if int(item["group"]["group_id"]) == group_id), None)
        if existing is None:
            existing = {
                "group": edge["group"],
                "edge_count": int(((graph["group_ids"] == group_id) & (graph["weights"] > 0)).sum()),
                "cosines": [],
                "group_activation": {
                    str(position): float(cache["splice_codes"][position, edge["group"]["concept_indices"]].sum())
                    for position in range(len(records))
                },
            }
            group_items.append(existing)
        existing["cosines"].append(edge["raw_clip_cosine"])
        edge["baseline_cosine"] = float(baseline_features[edge["left"]] @ baseline_features[edge["right"]])
        edge["left_activation"] = float(cache["splice_codes"][edge["left"], edge["group"]["concept_indices"]].sum())
        edge["right_activation"] = float(cache["splice_codes"][edge["right"], edge["group"]["concept_indices"]].sum())
    for item in group_items:
        values = item["cosines"]
        item["cosine_range"] = f"cosine {_fmt(min(values), 4)} … {_fmt(max(values), 4)}"

    # Embed only the deliberately limited report examples, not the whole cache.
    needed_positions = {position for pair in pairs for position in (pair["left"], pair["right"])}
    needed_positions.update(position for edge in edge_items for position in (edge["left"], edge["right"]))
    for item in group_items:
        top_positions = sorted(
            item["group_activation"].items(),
            key=lambda value: (-float(value[1]), int(value[0])),
        )[:4]
        needed_positions.update(int(position) for position, _ in top_positions)
    uris = [""] * len(records)
    for position in sorted(needed_positions):
        uris[position] = _image_uri(Path(records[position]["image_path"]))

    _build_html(
        args.output,
        records,
        uris,
        raw_features,
        cache["dictionary"],
        baseline_features,
        method_features,
        graph,
        pairs,
        group_items,
        edge_items,
        args.baseline_ckpt,
        args.method_ckpt,
        args.method_name,
        args.seed,
        args.graph,
    )
    print(f"[INFO] Wrote trained comparison report to {args.output}")


if __name__ == "__main__":
    main()
