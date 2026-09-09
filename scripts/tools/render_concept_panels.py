"""Render annotated Waterbirds concept-pair panels from frozen panel selections."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

TARGET_NAMES = {0: "landbird", 1: "waterbird"}
BACKGROUND_NAMES = {0: "land", 1: "water"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _artifact_root(value: Path | None) -> Path:
    configured = value or os.environ.get("SPUR_SPLICE_ARTIFACT_ROOT")
    if configured is None or not str(configured).strip():
        return PROJECT_ROOT / "outputs"
    path = Path(configured).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def decision_reason(pair: dict, group: dict, config: dict) -> str:
    """Explain the exact candidate gates and final sparse-graph pruning decision."""

    gain_floor = float(config["min_intervention_gain"])
    activation_floor = float(group["activation_difference_threshold"])
    residual_floor = float(config["residual_splice_similarity_threshold"])
    failures = []
    if float(pair["gain"]) <= gain_floor:
        failures.append(f"gain {pair['gain']:+.5f} <= {gain_floor:.5f}")
    if float(pair["activation_contrast"]) < activation_floor:
        failures.append(
            f"activation contrast {pair['activation_contrast']:.5f} < {activation_floor:.5f}"
        )
    if config.get("use_residual_splice_gate", False) and float(pair["residual_similarity"]) < residual_floor:
        failures.append(
            f"residual similarity {pair['residual_similarity']:.3f} < {residual_floor:.3f}"
        )
    if failures:
        return "failed candidate gate(s): " + "; ".join(failures)
    if not pair["retained"]:
        return (
            "passed candidate gates; pruned by row top-k / donor indegree competition "
            f"(k={config['graph_top_k']}, max indegree={config['max_indegree']})"
        )
    return "passed gain, activation and residual gates; survived sparse-graph pruning"


def _metadata(dataset_root: Path) -> list[dict[str, str]]:
    path = dataset_root / "metadata.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Waterbirds metadata not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _annotate(sample_id: str, metadata: list[dict[str, str]], dataset_root: Path) -> dict[str, Any]:
    prefix, raw_index = sample_id.split(":", 1)
    if prefix != "waterbirds":
        raise ValueError(f"Unexpected sample ID: {sample_id}")
    index = int(raw_index)
    row = metadata[index]
    target = int(row["y"])
    background = int(row["place"])
    return {
        "sample_id": sample_id,
        "target": target,
        "target_name": TARGET_NAMES[target],
        "spurious_attribute": background,
        "background_name": BACKGROUND_NAMES[background],
        "group": [target, background],
        "image": str(dataset_root / row["img_filename"]),
    }


def _draw_image(page: canvas.Canvas, path: Path, x: float, y: float, width: float, height: float) -> None:
    reader = ImageReader(str(path))
    image_width, image_height = reader.getSize()
    scale = min(width / image_width, height / image_height)
    drawn_width = image_width * scale
    drawn_height = image_height * scale
    page.setFillColor(HexColor("#F2F4F7"))
    page.rect(x, y, width, height, stroke=0, fill=1)
    page.drawImage(
        reader,
        x + (width - drawn_width) / 2,
        y + (height - drawn_height) / 2,
        drawn_width,
        drawn_height,
        preserveAspectRatio=True,
        mask="auto",
    )


def _draw_wrapped(
    page: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    font: str,
    size: float,
    leading: float,
) -> float:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if current and stringWidth(candidate, font, size) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    page.setFont(font, size)
    for line in lines:
        page.drawString(x, y, line)
        y -= leading
    return y


def _draw_annotation(page: canvas.Canvas, annotation: dict, x: float, y: float) -> None:
    page.setFillColor(black)
    page.setFont("Helvetica-Bold", 8.5)
    page.drawString(x, y, annotation["sample_id"])
    page.setFont("Helvetica", 8.5)
    page.drawString(x, y - 12, f"label={annotation['target_name']} (y={annotation['target']})")
    page.drawString(
        x,
        y - 24,
        f"spurious background={annotation['background_name']} "
        f"(a={annotation['spurious_attribute']}); group=({annotation['target']},{annotation['spurious_attribute']})",
    )


def _draw_pair(page: canvas.Canvas, pair: dict, x: float, y: float, width: float, height: float) -> None:
    gap = 14
    image_width = (width - gap) / 2
    image_height = 245
    image_y = y + height - image_height
    left = pair["left_annotation"]
    right = pair["right_annotation"]
    _draw_image(page, Path(left["image"]), x, image_y, image_width, image_height)
    _draw_image(page, Path(right["image"]), x + image_width + gap, image_y, image_width, image_height)
    _draw_annotation(page, left, x, image_y - 14)
    _draw_annotation(page, right, x + image_width + gap, image_y - 14)

    status = "RETAINED" if pair["retained"] else "REJECTED"
    status_color = HexColor("#137A45") if pair["retained"] else HexColor("#B42318")
    text_y = image_y - 68
    page.setFillColor(status_color)
    page.roundRect(x, text_y - 3, 68, 16, 4, stroke=0, fill=1)
    page.setFillColor(white)
    page.setFont("Helvetica-Bold", 8)
    page.drawCentredString(x + 34, text_y + 1, status)
    page.setFillColor(black)
    page.setFont("Helvetica-Bold", 9)
    page.drawString(x + 78, text_y + 1, f"{pair['left_id']} -> {pair['right_id']}")
    page.setFont("Helvetica", 8.5)
    page.drawString(x, text_y - 18, f"raw={pair['raw_similarity']:.4f}  projected={pair['projected_similarity']:.4f}  gain={pair['gain']:+.5f}")
    page.drawString(
        x,
        text_y - 32,
        f"activation contrast={pair['activation_contrast']:.5f}  residual={pair['residual_similarity']:.4f}",
    )
    page.drawString(
        x,
        text_y - 46,
        f"confidence={pair['null_calibrated_confidence']:+.6f}  final edge weight={pair['final_edge_weight']:.6f}",
    )
    page.setFillColor(status_color)
    _draw_wrapped(
        page,
        "Decision: " + pair["decision_reason"],
        x,
        text_y - 62,
        width,
        "Helvetica-Bold",
        8.5,
        11,
    )


def render(dataset_root: Path, artifact_root: Path, output_dir: Path) -> tuple[Path, Path]:
    source_path = output_dir / "concept_panels.json"
    panels = json.loads(source_path.read_text(encoding="utf-8"))
    graph = json.loads(
        (artifact_root / "shared" / "waterbirds" / "graphs" / "crp_graph.json").read_text(
            encoding="utf-8"
        )
    )
    groups_by_id = {int(group["group_id"]): group for group in graph["groups"]}
    metadata = _metadata(dataset_root)
    for panel_group in panels["groups"]:
        graph_group = groups_by_id[int(panel_group["group_id"])]
        panel_group["decision_thresholds"] = {
            "minimum_gain": graph["config"]["min_intervention_gain"],
            "minimum_activation_contrast": graph_group["activation_difference_threshold"],
            "minimum_residual_similarity": graph["config"]["residual_splice_similarity_threshold"],
            "row_top_k": graph["config"]["graph_top_k"],
            "maximum_indegree": graph["config"]["max_indegree"],
        }
        for pair in panel_group["pairs"]:
            pair["left_annotation"] = _annotate(pair["left_id"], metadata, dataset_root)
            pair["right_annotation"] = _annotate(pair["right_id"], metadata, dataset_root)
            pair["decision_reason"] = decision_reason(pair, graph_group, graph["config"])

    pdf_path = output_dir / "concept_panels.pdf"
    page = canvas.Canvas(str(pdf_path), pagesize=(1500, 1100), pageCompression=1)
    page.setTitle("CoSpRo annotated concept panels")
    for panel_group in panels["groups"]:
        page.setFillColor(black)
        page.setFont("Helvetica-Bold", 23)
        page.drawString(30, 1055, f"G{panel_group['group_id']}: {', '.join(panel_group['concepts'])}")
        page.setFont("Helvetica", 10)
        page.drawString(
            30,
            1032,
            "Annotations are post-hoc only: target y (landbird/waterbird), spurious attribute a "
            "(land/water background), and group (y,a).",
        )
        thresholds = panel_group["decision_thresholds"]
        page.drawString(
            30,
            1016,
            f"Candidate gates: gain > {thresholds['minimum_gain']:.5f}; activation contrast >= "
            f"{thresholds['minimum_activation_contrast']:.5f}; residual similarity >= "
            f"{thresholds['minimum_residual_similarity']:.2f}. Final graph: row top-{thresholds['row_top_k']}, "
            f"max donor indegree {thresholds['maximum_indegree']}.",
        )
        positions = ((30, 535), (770, 535), (30, 45), (770, 45))
        for pair, (x, y) in zip(panel_group["pairs"], positions):
            _draw_pair(page, pair, x, y, 700, 450)
        page.showPage()
    page.save()

    json_path = output_dir / "concept_panels.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(panels, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(json_path)
    return pdf_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    artifact_root = _artifact_root(args.artifact_root)
    output_dir = args.output_dir or artifact_root / "reports" / "followups_corrected" / "visual"
    pdf_path, json_path = render(args.dataset_root, artifact_root, output_dir)
    print(f"[INFO] Wrote {pdf_path}")
    print(f"[INFO] Wrote {json_path}")


if __name__ == "__main__":
    main()
