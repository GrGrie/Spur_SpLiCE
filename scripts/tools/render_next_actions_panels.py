"""Render deterministic, post-hoc CRP concept panels without training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from experiments.spurious_eval.datasets.waterbirds import WaterbirdsDataset
from splice.crp import (
    CrpAuditConfig,
    _AuditGeometry,
    _relation_geometry,
    orthonormal_basis,
    topk_neighbors,
    validate_feature_cache,
)
from splice.graph_io import load_graph_json


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(torch.quantile(torch.tensor(values), q))


def _source_index(sample_id: str) -> int:
    prefix, _, value = str(sample_id).rpartition(":")
    if prefix != "waterbirds":
        raise ValueError(f"Expected a Waterbirds sample id, got {sample_id!r}")
    return int(value)


def _pair_candidates(graph: dict, cache: dict, group: dict, config: CrpAuditConfig) -> list[dict]:
    concept_indices = [int(value) for value in group["concept_indices"]]
    basis = orthonormal_basis(cache["dictionary"][concept_indices], config.orthogonal_tolerance)
    raw_neighbours, _ = topk_neighbors(
        cache["centered_clip"], config.projected_neighbors, config.similarity_chunk_size
    )
    geometry = _relation_geometry(
        _AuditGeometry(cache["centered_clip"], raw_neighbours, cache["splice_codes"]),
        basis,
        config,
        concept_indices,
    )
    activation = cache["splice_codes"][:, concept_indices].sum(dim=1)
    contrast = (activation[geometry["anchors"]] - activation[geometry["neighbours"]]).abs()
    group_ratio = float(group.get("null_excess_ratio", 0.0))
    records = []
    for row in range(geometry["neighbours"].shape[0]):
        for position in range(geometry["neighbours"].shape[1]):
            column = int(geometry["neighbours"][row, position])
            gain = float(geometry["gain"][row, position])
            residual = float(geometry["residual_splice_similarity"][row, position])
            confidence = gain * (0.5 + 0.5 * residual)
            records.append({
                "row": row, "column": column, "position": position,
                "gain": gain,
                "raw_similarity": float((cache["centered_clip"][row] * cache["centered_clip"][column]).sum()),
                "projected_similarity": float(geometry["projected_similarity"][row, position]),
                "activation_contrast": float(contrast[row, position]),
                "residual_similarity": residual,
                "supported": bool(geometry["supported"][row, position]),
                "null_calibrated_confidence": confidence * group_ratio,
            })
    return records


def _final_edge(graph: dict, row: int, column: int, group_id: int) -> tuple[float, bool]:
    indices = graph["neighbor_indices"]
    mask = (indices[row] == column) & (graph.get("group_ids", torch.full_like(indices, -1))[row] == group_id)
    positions = torch.where(mask)[0]
    if not len(positions):
        return 0.0, False
    weight = float(graph["weights"][row, positions[0]])
    return weight, weight > 0


def _choose(records: list[dict], seed: int = 0) -> tuple[list[dict], list[dict], dict]:
    gains = [record["gain"] for record in records]
    high_cut, low_cut = _quantile(gains, 0.9), _quantile(gains, 0.1)
    high = [record for record in records if record["gain"] >= high_cut]
    low = [record for record in records if record["gain"] <= low_cut]
    high_by_row = {record["row"]: record for record in high}
    low_by_row = {record["row"]: record for record in low}
    common = sorted(set(high_by_row) & set(low_by_row))
    rng = random.Random(seed)
    rng.shuffle(common)
    if len(common) >= 2:
        rows = common[:2]
        high_choice = [high_by_row[row] for row in rows]
        low_choice = [low_by_row[row] for row in rows]
    else:
        high = sorted(high, key=lambda item: (abs(item["raw_similarity"] - 0.5), item["row"], item["column"]))
        low = sorted(low, key=lambda item: (abs(item["raw_similarity"] - 0.5), item["row"], item["column"]))
        high_choice, low_choice = high[:2], low[:2]
    return high_choice, low_choice, {
        "high_candidate_count": len(high), "low_candidate_count": len(low),
        "common_anchor_count": len(common), "seed": seed,
    }


def _load_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render(cache_path: Path, graph_path: Path, data_folder: Path, output_png: Path, output_pdf: Path, output_json: Path) -> None:
    cache = validate_feature_cache(torch.load(cache_path, map_location="cpu", weights_only=True))
    graph = load_graph_json(graph_path)
    config = CrpAuditConfig(**graph["config"])
    group_ids = graph["group_ids"]
    weights = graph["weights"]
    selected_groups = [group for group in graph.get("groups", []) if group.get("selected", False)]
    selected_groups.sort(key=lambda group: (-int(((group_ids == int(group["group_id"])) & (weights > 0)).sum()), int(group["group_id"])))
    selected_groups = selected_groups[:5]
    dataset = WaterbirdsDataset(str(data_folder))
    pages, report_groups = [], []
    title_font, body_font = _load_font(28), _load_font(16)
    for group in selected_groups:
        records = _pair_candidates(graph, cache, group, config)
        high, low, choice = _choose(records, seed=0)
        cases = [("high", item) for item in high] + [("low", item) for item in low]
        payload_cases = []
        page = Image.new("RGB", (1500, 1100), "white")
        draw = ImageDraw.Draw(page)
        concepts = ", ".join(str(value) for value in group.get("concepts", []))
        draw.text((30, 25), f"G{int(group['group_id'])}: {concepts}", fill="black", font=title_font)
        draw.text((30, 65), "High/low gain candidates; selection seed=0; confidence is a score, not a probability.", fill="black", font=body_font)
        draw.text((30, 90), "Review caption: does the visible difference match the concept group while overall semantics remain? Record counterexamples.", fill="black", font=body_font)
        for index, (band, record) in enumerate(cases):
            x = 30 + (index % 2) * 735
            y = 110 + (index // 2) * 490
            left_id, right_id = cache["sample_ids"][record["row"]], cache["sample_ids"][record["column"]]
            images = []
            for sample_id in (left_id, right_id):
                try:
                    image = dataset.get_input(_source_index(sample_id)).convert("RGB")
                    image.thumbnail((330, 300))
                except (FileNotFoundError, OSError, AttributeError):
                    image = Image.new("RGB", (330, 300), "#dddddd")
                images.append(image)
            page.paste(images[0], (x, y))
            page.paste(images[1], (x + 350, y))
            weight, retained = _final_edge(graph, record["row"], record["column"], int(group["group_id"]))
            status = "retained" if retained else "rejected"
            text_y = y + 315
            details = (
                f"{band.upper()} | {left_id} -> {right_id}\n"
                f"gain={record['gain']:+.6f}  activation contrast={record['activation_contrast']:.6f}\n"
                f"residual similarity={record['residual_similarity']:.6f}  "
                f"null-calibrated confidence={record['null_calibrated_confidence']:.6f}\n"
                f"final edge weight={weight:.6f}  {status}"
            )
            draw.multiline_text((x, text_y), details, fill="black", font=body_font, spacing=5)
            payload_cases.append({**record, "left_id": left_id, "right_id": right_id,
                                  "final_edge_weight": weight, "retained": retained,
                                  "confidence_interpretation": "score, not probability"})
        pages.append(page)
        report_groups.append({"group_id": int(group["group_id"]), "concepts": group.get("concepts", []),
                              "retained_edge_count": int(((group_ids == int(group["group_id"])) & (weights > 0)).sum()),
                              "selection": choice, "pairs": payload_cases,
                              "review_caption": "Does the visible difference match the group while overall semantics remain? Record counterexamples."})

    if not pages:
        pages = [Image.new("RGB", (1500, 1100), "white")]
        ImageDraw.Draw(pages[0]).text((30, 30), "No selected CRP groups with retained edges.", fill="black", font=title_font)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(output_png)
    pages[0].save(output_pdf, save_all=True, append_images=pages[1:])
    output_json.write_text(json.dumps({
        "artifact": "next_actions_visual_panels_v1",
        "graph": str(graph_path), "groups": report_groups,
        "selection_rule": "top five groups by retained-edge count, group_id tie-break; gain deciles; seed 0",
        "selection_is_label_free": True,
        "post_hoc_annotation": "labels are not used for selection; displayed panels contain IDs and concepts only",
        "interpretation_caveat": "visualization explains the mechanism but does not prove WGA growth or causal background removal",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote visual panels to {output_png}, {output_pdf}, and {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--data-folder", required=True, type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--output-pdf", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    render(args.cache, args.graph, args.data_folder, args.output_png, args.output_pdf, args.output_json)


if __name__ == "__main__":
    main()
