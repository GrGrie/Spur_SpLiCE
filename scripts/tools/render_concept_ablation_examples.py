"""Render a self-contained post-hoc Waterbirds concept-ablation report."""

from __future__ import annotations

import argparse
import base64
import html
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from experiments.spurious_eval.datasets.waterbirds import WaterbirdsDataset
from splice.cqt import CQT_ARTIFACT, concept_quotient
from splice.crp import orthonormal_basis, project_out, validate_feature_cache
from splice.graph_io import load_graph_json


CRP_ARTIFACT = "splice_crp_v2_teacher_graph"
LABEL_NAMES = {0: "landbird", 1: "waterbird"}
BACKGROUND_NAMES = {0: "land background", 1: "water background"}


@dataclass(frozen=True)
class Intervention:
    identifier: str
    title: str
    concepts: str
    kind: str
    direction: torch.Tensor
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


PAIR_SPECS = (
    PairSpec(
        title="Пара 1: одинаковый label, разный spurious attribute",
        explanation="Обе птицы — waterbird; одна на суше, другая на воде.",
        left_label=1,
        left_background=0,
        right_label=1,
        right_background=1,
    ),
    PairSpec(
        title="Пара 2: разные labels, одинаковый spurious attribute",
        explanation="Waterbird и landbird; обе птицы на суше.",
        left_label=1,
        left_background=0,
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


def _interventions(graph: dict, cache: dict, scope: str) -> list[Intervention]:
    selected_only = scope == "selected"
    interventions: list[Intervention] = []
    if graph.get("artifact") == CRP_ARTIFACT:
        tolerance = float(graph.get("config", {}).get("orthogonal_tolerance", 1e-6))
        for group in graph.get("groups", []):
            selected = bool(group.get("selected", False))
            if selected_only and not selected:
                continue
            indices = [int(index) for index in group["concept_indices"]]
            concepts = [str(word) for word in group.get("concepts", [])]
            interventions.append(
                Intervention(
                    identifier=f"G{int(group['group_id'])}",
                    title=f"Concept group G{int(group['group_id'])}",
                    concepts=", ".join(concepts),
                    kind="crp_projection",
                    direction=orthonormal_basis(cache["dictionary"][indices], tolerance),
                    selected=selected,
                    evidence=(
                        f"rank={int(group.get('basis_rank', len(indices)))}, "
                        f"score={float(group.get('score', 0.0)):.6g}, "
                        f"null threshold={float(group.get('null_threshold', 0.0)):.6g}, "
                        f"coverage={float(group.get('coverage', 0.0)):.2%}"
                    ),
                )
            )
    elif graph.get("artifact") == CQT_ARTIFACT:
        for factor in graph.get("factors", []):
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
                    selected=selected,
                    evidence=(
                        f"gain={float(factor.get('intervention_gain', 0.0)):.6g}, "
                        f"null threshold={float(factor.get('null_threshold', 0.0)):.6g}, "
                        f"coverage={float(factor.get('coverage', 0.0)):.2%}"
                    ),
                )
            )
    else:
        raise ValueError(f"Unsupported teacher graph artifact: {graph.get('artifact')!r}")
    return interventions


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


def _most_changed_pair(
    centered: torch.Tensor,
    left_positions: torch.Tensor,
    right_positions: torch.Tensor,
    interventions: Sequence[Intervention],
    chunk_size: int,
) -> tuple[int, int, str | None, float]:
    if not interventions:
        return int(left_positions[0]), int(right_positions[0]), None, 0.0

    raw_right = centered[right_positions]
    best_score = -1.0
    best = (int(left_positions[0]), int(right_positions[0]), None, 0.0)
    for intervention in interventions:
        edited_right = _apply(raw_right, intervention)
        for start in range(0, len(left_positions), chunk_size):
            rows = left_positions[start : start + chunk_size]
            raw_left = centered[rows]
            edited_left = _apply(raw_left, intervention)
            delta = edited_left @ edited_right.T - raw_left @ raw_right.T
            scores = delta.abs()
            flat_position = int(scores.argmax())
            score = float(scores.flatten()[flat_position])
            if score > best_score:
                local_row = flat_position // scores.shape[1]
                local_column = flat_position % scores.shape[1]
                best_score = score
                best = (
                    int(rows[local_row]),
                    int(right_positions[local_column]),
                    intervention.identifier,
                    float(delta[local_row, local_column]),
                )
    return best


def _image_data_uri(path: Path, max_size: int = 640) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image.thumbnail((max_size, max_size), resampling)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _image_card(position: int, cache: dict, metadata_rows: Sequence[dict], image_root: Path) -> str:
    sample_id = str(cache["sample_ids"][position])
    row = metadata_rows[_source_index(sample_id)]
    label = LABEL_NAMES[int(row["y"])]
    background = BACKGROUND_NAMES[int(row["place"])]
    image_path = image_root / str(row["img_filename"])
    if not image_path.is_file():
        raise FileNotFoundError(f"Waterbirds image not found: {image_path}")
    return f"""
      <figure class="card">
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
        return '<p class="warning">Teacher graph не содержит выбранных concept groups/factors.</p>'
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
        '<table><thead><tr><th>ID</th><th>Концепты</th><th>Graph evidence</th>'
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
        '<table><thead><tr><th>Удалённая группа / factor</th><th>Cosine before</th>'
        f'<th>Cosine after</th><th>Δ = after − before</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def generate_report(
    cache_path: Path,
    graph_path: Path,
    data_folder: Path,
    output_path: Path,
    scope: str = "selected",
    selection_chunk_size: int = 512,
) -> Path:
    cache = validate_feature_cache(torch.load(cache_path, map_location="cpu", weights_only=True))
    graph = load_graph_json(graph_path)
    if list(map(str, graph.get("sample_ids", []))) != list(map(str, cache["sample_ids"])):
        raise ValueError("Teacher graph and feature cache sample IDs are not aligned.")
    if scope not in {"selected", "all"}:
        raise ValueError("scope must be selected or all.")
    if selection_chunk_size <= 0:
        raise ValueError("selection_chunk_size must be positive.")

    dataset = WaterbirdsDataset(str(data_folder))
    metadata_rows = dataset.metadata_df.to_dict(orient="records")
    image_root = Path(dataset.data_dir)
    interventions = _interventions(graph, cache, scope)
    pair_sections = []
    for spec in PAIR_SPECS:
        left_candidates = _candidate_positions(
            cache["sample_ids"], metadata_rows, spec.left_label, spec.left_background
        )
        right_candidates = _candidate_positions(
            cache["sample_ids"], metadata_rows, spec.right_label, spec.right_background
        )
        left, right, selection_group, selection_delta = _most_changed_pair(
            cache["centered_clip"],
            left_candidates,
            right_candidates,
            interventions,
            selection_chunk_size,
        )
        selection_note = (
            f"Выбрана допустимая пара с максимальным |Δ| среди показанных interventions: "
            f"{selection_group}, Δ={selection_delta:+.6f}."
            if selection_group is not None
            else "Выбрана первая допустимая пара: выбранных interventions нет."
        )
        pair_sections.append(
            f"""
            <section>
              <h2>{html.escape(spec.title)}</h2>
              <p>{html.escape(spec.explanation)}</p>
              <p class="muted">{html.escape(selection_note)} Это иллюстративный sanity-check, не aggregate metric.</p>
              <div class="images">
                {_image_card(left, cache, metadata_rows, image_root)}
                {_image_card(right, cache, metadata_rows, image_root)}
              </div>
              <h3>Найденные concept groups / factors</h3>
              {_group_table(interventions)}
              <h3>Cosine similarity до и после удаления</h3>
              {_similarity_table(cache['centered_clip'], left, right, interventions)}
            </section>
            """
        )

    mode_name = "CRP full-subspace projection" if graph["artifact"] == CRP_ARTIFACT else "CQT rank-one quotient"
    audited_count = len(graph.get("groups", graph.get("factors", [])))
    selected_count = len(interventions) if scope == "selected" else sum(item.selected for item in interventions)
    document = f"""<!doctype html>
<html lang="ru">
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
    figcaption {{ padding:12px 14px; }} .muted {{ color:var(--muted); }}
    table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
    th,td {{ padding:10px 12px; border:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ background:#f0f3f7; }} .positive {{ color:#067647; font-weight:700; }}
    .negative {{ color:#b42318; font-weight:700; }} .neutral {{ color:var(--muted); }}
    .warning {{ padding:12px; border-left:4px solid #f79009; background:#fffaeb; }}
    code {{ overflow-wrap:anywhere; }}
    @media (max-width:760px) {{ .images {{ grid-template-columns:1fr; }} .card img {{ height:280px; }} table {{ font-size:13px; }} }}
  </style>
</head>
<body><main>
  <header>
    <h1>Spur SpLiCE: concept-ablation sanity-check</h1>
    <p><strong>Intervention:</strong> {html.escape(mode_name)}.</p>
    <p><strong>Scope:</strong> {html.escape(scope)}; audited={audited_count}, selected={selected_count}, shown={len(interventions)}.</p>
    <p>Все числа вычислены по полным centered CLIP embeddings из frozen cache. Метаданные Waterbirds используются только после graph discovery, чтобы выбрать две диагностические пары и подписать изображения.</p>
  </header>
  {''.join(pair_sections)}
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
    )
    print(f"[INFO] Wrote self-contained concept-ablation report to {path}")


if __name__ == "__main__":
    main()
