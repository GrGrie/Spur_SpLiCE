"""Render one self-contained Waterbirds concept-group similarity sweep report.

The report is deliberately post-hoc: target and background annotations are used
only to choose and describe diagnostic examples.  Every intervention removes a
CRP group's full text subspace from the frozen, centered CLIP embeddings.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image

from experiments.spurious_eval.datasets.waterbirds import WaterbirdsDataset
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
class Group:
    group_id: int
    concepts: tuple[str, ...]
    concept_indices: tuple[int, ...]
    basis: torch.Tensor
    selected: bool
    rank: int
    score: float
    coverage: float


@dataclass(frozen=True)
class Example:
    key: str
    kicker: str
    title: str
    description: str
    letters: tuple[str, ...]
    positions: tuple[int, ...]
    comparisons: tuple[tuple[int, int, str], ...]


def _source_index(sample_id: str) -> int:
    prefix, separator, index = str(sample_id).rpartition(":")
    if not separator or prefix != "waterbirds" or not index.isdigit():
        raise ValueError(f"Expected a Waterbirds sample ID, got {sample_id!r}.")
    return int(index)


def _metadata(dataset: WaterbirdsDataset, sample_ids: Sequence[str]) -> list[dict]:
    records = []
    for sample_id in sample_ids:
        row = dataset.metadata_df.iloc[_source_index(sample_id)]
        label_id, background_id = int(row["y"]), int(row["place"])
        records.append(
            {
                "sample_id": str(sample_id),
                "label_id": label_id,
                "background_id": background_id,
                "label": LABEL_NAMES.get(label_id, f"label {label_id}"),
                "background": BACKGROUND_NAMES.get(background_id, f"spurious {background_id}"),
                "image_path": Path(dataset.data_dir) / str(row["img_filename"]),
            }
        )
    return records


def _groups(graph: dict, cache: dict, scope: str) -> list[Group]:
    if graph.get("artifact") not in CRP_ARTIFACTS:
        raise ValueError(f"Unsupported teacher graph artifact: {graph.get('artifact')!r}")
    if scope not in {"selected", "all"}:
        raise ValueError("scope must be 'selected' or 'all'.")
    tolerance = float(graph.get("config", {}).get("orthogonal_tolerance", 1e-6))
    output = []
    for item in sorted(graph.get("groups", []), key=lambda value: int(value["group_id"])):
        indices = tuple(int(index) for index in item.get("concept_indices", []))
        if not indices:
            continue
        output.append(
            Group(
                group_id=int(item["group_id"]),
                concepts=tuple(str(value) for value in item.get("concepts", [])),
                concept_indices=indices,
                basis=orthonormal_basis(cache["dictionary"][list(indices)], tolerance),
                selected=bool(item.get("selected", False)),
                rank=int(item.get("basis_rank", len(indices))),
                score=float(item.get("score", 0.0)),
                coverage=float(item.get("coverage", 0.0)),
            )
        )
    output = [group for group in output if scope == "all" or group.selected]
    if not output:
        raise ValueError(f"The graph contains no {scope} concept groups with concept indices.")
    return output


def _shared_anchor_extremes(
    features: torch.Tensor,
    records: Sequence[dict],
    predicate: Callable[[dict, dict], bool],
) -> tuple[int, int, int]:
    """Choose one anchor with maximally separated high/low eligible partners."""

    best = None
    for anchor, left in enumerate(records):
        candidates = [right for right, record in enumerate(records) if right != anchor and predicate(left, record)]
        if len(candidates) < 2:
            continue
        values = features[candidates] @ features[anchor]
        high_order = sorted(range(len(candidates)), key=lambda index: (-float(values[index]), candidates[index]))
        low_order = sorted(range(len(candidates)), key=lambda index: (float(values[index]), candidates[index]))
        high = candidates[high_order[0]]
        low = next((candidates[index] for index in low_order if candidates[index] != high), None)
        if low is None:
            continue
        spread = float(features[anchor] @ features[high] - features[anchor] @ features[low])
        candidate = (-spread, anchor, high, low)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No anchor with two eligible comparison images was found.")
    return best[1], best[2], best[3]


def _examples(features: torch.Tensor, records: Sequence[dict]) -> list[Example]:
    a, b, c = _shared_anchor_extremes(
        features,
        records,
        lambda left, right: left["label_id"] == right["label_id"] and left["background_id"] != right["background_id"],
    )
    d, e, f = _shared_anchor_extremes(
        features,
        records,
        lambda left, right: left["label_id"] != right["label_id"] and left["background_id"] == right["background_id"],
    )
    k, n, l = _shared_anchor_extremes(
        features,
        records,
        lambda left, right: (
            left["label_id"] == 1
            and left["background_id"] == 1
            and right["label_id"] == 1
            and right["background_id"] == 1
        ),
    )
    x, y, z = _shared_anchor_extremes(
        features,
        records,
        lambda left, right: (
            left["label_id"] == 1
            and left["background_id"] == 0
            and right["label_id"] == 0
            and right["background_id"] == 1
        ),
    )
    return [
        Example("ab", "PAIR 01 · HIGH INITIAL COSINE", "A + B · same label, different spurious data", "B is the most similar eligible partner of the shared anchor A.", ("A", "B"), (a, b), ((0, 1, "A + B · high cosine"),)),
        Example("ac", "PAIR 02 · LOW INITIAL COSINE", "A + C · same label, different spurious data", "C is the least similar eligible partner of the same anchor A.", ("A", "C"), (a, c), ((0, 1, "A + C · low cosine"),)),
        Example("de", "PAIR 03 · HIGH INITIAL COSINE", "D + E · same spurious data, different labels", "E is the most similar eligible partner of the shared anchor D.", ("D", "E"), (d, e), ((0, 1, "D + E · high cosine"),)),
        Example("df", "PAIR 04 · LOW INITIAL COSINE", "D + F · same spurious data, different labels", "F is the least similar eligible partner of the same anchor D.", ("D", "F"), (d, f), ((0, 1, "D + F · low cosine"),)),
        Example("knl", "TRIPLET 01 · TWO PAIR SWEEPS", "K + N / K + L · all metadata (1,1)", "K is shared. N is the most similar and L the least similar distinct (1,1) partner.", ("K", "N", "L"), (k, n, l), ((0, 1, "K + N · high cosine"), (0, 2, "K + L · low cosine"))),
        Example("xyz", "TRIPLET 02 · COMPLETE NEGATIVES", "X (1,0) + Y/Z (0,1)", "X is shared. Y is the most similar and Z the least similar eligible complete-negative partner.", ("X", "Y", "Z"), (x, y, z), ((0, 1, "X + Y · high cosine"), (0, 2, "X + Z · low cosine"))),
    ]


def _mean_pairwise_cosine(features: torch.Tensor, positions: Sequence[int]) -> float:
    pairs = list(combinations(positions, 2))
    return sum(float(features[left] @ features[right]) for left, right in pairs) / len(pairs)


def _projected_value(features: torch.Tensor, positions: Sequence[int], group: Group) -> float | None:
    try:
        return _mean_pairwise_cosine(project_out(features[list(positions)], group.basis), range(len(positions)))
    except ValueError:
        return None


def _image_uri(path: Path, max_size: int = 560) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Waterbirds image not found: {path}")
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size), getattr(Image, "Resampling", Image).LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=86, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _image_card(letter: str, position: int, record: dict, uri: str) -> str:
    return f"""
    <figure class="image-card">
      <div class="letter">{html.escape(letter)}</div>
      <img src="{uri}" alt="{html.escape(record['label'])}, {html.escape(record['background'])}">
      <figcaption><strong>{html.escape(record['label'])}</strong><span>{html.escape(record['background'])}</span><code>{html.escape(record['sample_id'])}</code></figcaption>
    </figure>"""


def _group_index(groups: Sequence[Group]) -> str:
    cards = []
    for group in groups:
        concepts = ", ".join(group.concepts) or "unnamed group"
        cards.append(
            f'<article class="group-chip"><div><b>G{group.group_id}</b><span class="status {"kept" if group.selected else "audited"}">{"selected" if group.selected else "audited"}</span></div>'
            f'<p>{html.escape(concepts)}</p><small>rank {group.rank} · score {group.score:.5g} · coverage {group.coverage:.1%}</small></article>'
        )
    return "".join(cards)


def _sweep_table(
    features: torch.Tensor,
    positions: tuple[int, int],
    title: str,
    groups: Sequence[Group],
) -> str:
    before = _mean_pairwise_cosine(features, positions)
    rows = []
    gains = []
    for group in groups:
        after = _projected_value(features, positions, group)
        gain = None if after is None else after - before
        if gain is not None:
            gains.append(gain)
        gain_class = "na" if gain is None else "up" if gain > 0 else "down" if gain < 0 else "flat"
        width = 0.0 if gain is None else min(100.0, abs(gain) * 250.0)
        rows.append(
            f'<tr><td><b>G{group.group_id}</b><span>{html.escape(", ".join(group.concepts))}</span></td>'
            f'<td>{_fmt(before)}</td><td>{_fmt(after)}</td><td class="gain {gain_class}">{_fmt(gain, signed=True)}</td>'
            f'<td><i class="bar {gain_class}" style="--w:{width:.2f}%"></i></td></tr>'
        )
    joint_basis = orthonormal_basis(torch.cat([group.basis.T for group in groups], dim=0))
    joint_after = _projected_value(
        features,
        positions,
        Group(-1, (), (), joint_basis, True, joint_basis.shape[1], 0.0, 0.0),
    )
    joint_gain = None if joint_after is None else joint_after - before
    mean_gain = sum(gains) / len(gains) if gains else None
    summary = (
        '<div class="aggregate-grid">'
        f'<div><span>Σ individual gains</span><strong>{_fmt(sum(gains), signed=True)}</strong><small>descriptive only</small></div>'
        f'<div><span>Mean individual gain</span><strong>{_fmt(mean_gain, signed=True)}</strong><small>{len(gains)} groups</small></div>'
        f'<div><span>Joint removal gain</span><strong>{_fmt(joint_gain, signed=True)}</strong><small>union subspace; cosine {_fmt(joint_after)}</small></div>'
        '</div>'
    )
    return (
        f'<article class="sub-sweep"><div class="sub-sweep-head"><h3>{html.escape(title)}</h3><div class="baseline"><span>INITIAL COSINE</span><strong>{before:.4f}</strong></div></div>'
        f'{summary}'
        '<div class="table-wrap"><table><thead><tr><th>Removed group</th><th>Initial cosine</th><th>Cosine after removal</th><th>Gain</th><th>Magnitude</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></article>'
    )


def generate_report(
    cache_path: Path,
    graph_path: Path,
    data_folder: Path,
    output_path: Path,
    scope: str = "selected",
) -> Path:
    cache = validate_feature_cache(torch.load(cache_path, map_location="cpu", weights_only=True))
    graph = load_graph_json(graph_path)
    if list(map(str, graph.get("sample_ids", []))) != list(map(str, cache["sample_ids"])):
        raise ValueError("Teacher graph and feature cache sample IDs are not aligned.")
    dataset = WaterbirdsDataset(str(data_folder))
    records = _metadata(dataset, cache["sample_ids"])
    features = cache["centered_clip"]
    groups = _groups(graph, cache, scope)
    examples = _examples(features, records)
    needed = sorted({position for example in examples for position in example.positions})
    uris = {position: _image_uri(records[position]["image_path"]) for position in needed}

    sections = []
    for example in examples:
        images = "".join(_image_card(letter, position, records[position], uris[position]) for letter, position in zip(example.letters, example.positions))
        sweeps = "".join(
            _sweep_table(
                features,
                (example.positions[left], example.positions[right]),
                title,
                groups,
            )
            for left, right, title in example.comparisons
        )
        sections.append(f"""
        <section class="case" id="{html.escape(example.key)}">
          <div class="case-head"><div><span class="eyebrow">{html.escape(example.kicker)}</span><h2>{html.escape(example.title)}</h2><p>{html.escape(example.description)}</p></div></div>
          <div class="images cols-{len(example.positions)}">{images}</div>
          <div class="formula"><b>gain(G)</b> = cosine(after removing G) − initial cosine. Every table is one pair; triplets contain two tables with a shared anchor.</div>
          {sweeps}
        </section>""")

    selected_count = sum(group.selected for group in groups)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CRP concept-group cosine sweep</title>
<style>
:root{{--ink:#15201d;--muted:#6d7772;--paper:#fffdf8;--wash:#ecebe4;--line:#d8d7cd;--forest:#153e32;--lime:#d8ff75;--coral:#f06c4f;--blue:#2d66d4}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}main{{width:min(1440px,calc(100% - 34px));margin:24px auto 64px}}header,.panel,.case{{background:var(--paper);border:1px solid var(--line);border-radius:20px;box-shadow:0 14px 34px #18231d12}}header{{overflow:hidden}}.hero{{display:grid;grid-template-columns:1.5fr .7fr;gap:32px;padding:46px;background:var(--forest);color:white}}.eyebrow{{display:block;font-size:.7rem;font-weight:850;letter-spacing:.12em;text-transform:uppercase;color:#6d7a74}}.hero .eyebrow{{color:var(--lime)}}h1{{max-width:850px;margin:8px 0 14px;font:700 clamp(2.5rem,5vw,5rem)/.96 Georgia,serif;letter-spacing:-.045em}}h2{{margin:5px 0 8px;font:700 clamp(1.65rem,2.5vw,2.5rem)/1.03 Georgia,serif;letter-spacing:-.025em}}h3{{margin:0;font-size:1.1rem}}p{{margin:7px 0}}.hero p{{max-width:80ch;color:#d8e3de}}.hero-stat{{align-self:end;border-left:1px solid #ffffff35;padding-left:24px}}.hero-stat strong{{display:block;font-size:4rem;line-height:1;color:var(--lime)}}.hero-stat span{{color:#c7d3ce}}.panel,.case{{margin-top:22px;padding:28px 32px}}.group-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:18px}}.group-chip{{padding:13px;border:1px solid var(--line);border-radius:12px;background:#fff}}.group-chip>div{{display:flex;align-items:center;justify-content:space-between;gap:8px}}.group-chip b{{font-size:1.1rem}}.group-chip p{{min-height:42px;font-weight:650}}.group-chip small{{color:var(--muted)}}.status{{padding:3px 7px;border-radius:99px;font-size:.63rem;font-weight:800;text-transform:uppercase;letter-spacing:.06em}}.status.kept{{background:#dff5e9;color:#176040}}.status.audited{{background:#eee;color:#68706c}}.case{{border-top:5px solid var(--forest)}}.case-head p{{color:var(--muted);max-width:80ch}}.baseline{{display:grid;gap:3px;padding:10px 15px;border-radius:12px;background:#edf4f0;text-align:right}}.baseline span{{font-size:.66rem;font-weight:850;letter-spacing:.1em;color:#557167}}.baseline strong{{font-size:2rem;font-variant-numeric:tabular-nums}}.images{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:20px 0}}.images.cols-3{{grid-template-columns:repeat(3,minmax(0,1fr))}}.image-card{{position:relative;margin:0;overflow:hidden;border:1px solid var(--line);border-radius:14px;background:#f1f1eb}}.image-card img{{display:block;width:100%;height:285px;object-fit:contain}}.image-card figcaption{{display:grid;gap:2px;padding:11px 13px;background:white}}.image-card figcaption span,.image-card code{{color:var(--muted)}}.image-card code{{font-size:.72rem;overflow-wrap:anywhere}}.letter{{position:absolute;z-index:1;top:12px;left:12px;width:42px;height:42px;display:grid;place-items:center;border-radius:50%;background:var(--lime);color:var(--forest);font:800 1.1rem/1 Georgia,serif;box-shadow:0 5px 14px #0003}}.formula{{margin:12px 0;padding:12px 15px;border-left:4px solid var(--blue);background:#edf2ff;color:#40506f}}.sub-sweep{{margin-top:17px;padding:15px;border:1px solid var(--line);border-radius:15px;background:#fff}}.sub-sweep-head{{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:12px}}.aggregate-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:0 0 12px}}.aggregate-grid>div{{display:grid;gap:2px;padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:#f7f8f4}}.aggregate-grid span,.aggregate-grid small{{color:var(--muted)}}.aggregate-grid strong{{font-size:1.15rem;font-variant-numeric:tabular-nums}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}th{{position:sticky;top:0;background:#e9ebe5;color:#505b56;font-size:.72rem;text-transform:uppercase;letter-spacing:.055em}}tbody tr:last-child td{{border-bottom:0}}td:first-child{{min-width:220px}}td:first-child span{{display:block;color:var(--muted);font-size:.78rem}}.gain{{font-weight:800}}.gain.up{{color:#147049}}.gain.down{{color:#b84431}}.gain.na{{color:var(--muted)}}.bar{{display:block;width:var(--w);min-width:2px;max-width:150px;height:7px;border-radius:99px;background:#77817c}}.bar.up{{background:#31a16f}}.bar.down{{background:var(--coral)}}footer{{padding:22px 4px;color:var(--muted);font-size:.82rem}}code{{overflow-wrap:anywhere}}@media(max-width:980px){{.group-grid{{grid-template-columns:repeat(2,1fr)}}.hero{{grid-template-columns:1fr}}.hero-stat{{border-left:0;padding-left:0}}.images.cols-3{{grid-template-columns:1fr 1fr}}}}@media(max-width:650px){{main{{width:calc(100% - 14px);margin-top:7px}}.hero,.panel,.case{{padding:22px 17px}}.group-grid,.case-head,.images,.images.cols-3,.aggregate-grid{{grid-template-columns:1fr}}.sub-sweep-head{{display:grid}}.baseline{{text-align:left}}.image-card img{{height:240px}}}}
</style></head><body><main>
<header><div class="hero"><div><span class="eyebrow">FROZEN CLIP · FULL-SUBSPACE ABLATION</span><h1>Which concept group moves similarity?</h1><p>One deterministic report over four pair cases and two triplet controls. Every row removes exactly one CRP concept group from the same centered CLIP embeddings; no encoder training or test split is used.</p></div><div class="hero-stat"><strong>{len(groups)}</strong><span>{html.escape(scope)} groups swept in every case<br>{selected_count} selected by the teacher graph</span></div></div></header>
<section class="panel"><span class="eyebrow">GROUP INDEX · ORIGINAL GRAPH IDS</span><h2>{'Selected' if scope == 'selected' else 'All audited'} concept groups</h2><p>Displayed IDs are the original graph IDs. Use <code>--scope all</code> only when rejected audited groups are intentionally required.</p><div class="group-grid">{_group_index(groups)}</div></section>
{"".join(sections)}
<footer>Frozen feature cache: <code>{html.escape(str(cache_path))}</code><br>Teacher graph: <code>{html.escape(str(graph_path))}</code><br>Selection metadata: labels/backgrounds are post-hoc annotations only. Configuration: <code>{html.escape(json.dumps(graph.get('config', {}), sort_keys=True))}</code></footer>
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = generate_report(args.cache, args.graph, args.data_folder, args.output, scope=args.scope)
    print(f"[INFO] Wrote concept-group similarity sweep to {output}")


if __name__ == "__main__":
    main()
