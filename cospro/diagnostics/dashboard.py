"""Render a self-contained HTML dashboard from a diagnostics record.

Rendering runs on the local machine from the Git-synchronized JSON record. Edge thumbnails
appear when ``data_folder`` points at a local copy of the dataset.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PALETTE = {"primary": "#3b6ea5", "accent": "#c8553d", "muted": "#8a8f98", "good": "#2e8b57"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _figure_html(figure, alt: str) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    _plt().close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{encoded}">'


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


# ---------------------------------------------------------------- sweep


def _sweep_value(point: dict, key: str) -> float | None:
    grouping, graphs = point["grouping"], point["graphs"]
    graph = graphs[0] if graphs else None
    lookup = {
        "composite groups": lambda: grouping["structure"].get("composite_groups"),
        "max group size": lambda: grouping["structure"].get("max_group_size"),
        "NPMI": lambda: grouping.get("npmi"),
        "text coherence": lambda: grouping.get("text_coherence"),
        "bootstrap ARI": lambda: (grouping.get("bootstrap") or {}).get("mean_adjusted_rand_index"),
        "spurious fragmentation": lambda: (grouping.get("spurious") or {}).get("fragmentation"),
        "selected groups": lambda: graph and graph["selection"]["selected_groups"],
        "edge overlap with reference": lambda: graph and graph.get("overlap_with_reference"),
        "balanced counterfactual": lambda: graph and (graph.get("labels") or {}).get("group_balanced_counterfactual"),
        "balanced same class": lambda: graph and (graph.get("labels") or {}).get("group_balanced_same_class"),
        "minority reach": lambda: graph and (graph.get("labels") or {}).get("minority_reach"),
    }
    value = lookup[key]()
    return None if value is None else float(value)


NAMED_GRAPH_KEYS = {
    "selected groups": ("selection", "selected_groups"),
    "balanced counterfactual": ("labels", "group_balanced_counterfactual"),
    "balanced same class": ("labels", "group_balanced_same_class"),
    "minority reach": ("labels", "minority_reach"),
}


def _named_graph_values(record: dict, key: str) -> str:
    """The same metric for the fully described graphs, as a reading aid next to the sweep."""

    if key not in NAMED_GRAPH_KEYS:
        return ""
    section, field = NAMED_GRAPH_KEYS[key]
    values = []
    for name, entry in record.get("graphs", {}).items():
        value = (entry.get(section) or {}).get(field)
        if value is not None:
            values.append(f"{name} {value:.3g}" if isinstance(value, float) else f"{name} {value}")
    return ", ".join(values)


def sweep_heatmaps(record: dict) -> str:
    points = record.get("sweep", [])
    if not points:
        return ""
    configs = [point["grouping"]["config"] for point in points]
    text = sorted({config.get("text_similarity_threshold") for config in configs})
    coactivation = sorted({config.get("coactivation_threshold") for config in configs})
    keys = [
        key for key in (
            "composite groups", "max group size", "selected groups", "edge overlap with reference",
            "balanced counterfactual", "balanced same class", "minority reach",
            "NPMI", "text coherence", "bootstrap ARI", "spurious fragmentation",
        )
        if any(_sweep_value(point, key) is not None for point in points)
    ]
    plt = _plt()
    columns = 3
    rows = int(np.ceil(len(keys) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.2 * rows), squeeze=False)
    for axis, key in zip(axes.flat, keys):
        grid = np.full((len(text), len(coactivation)), np.nan)
        for point, config in zip(points, configs):
            value = _sweep_value(point, key)
            if value is not None:
                grid[text.index(config.get("text_similarity_threshold")),
                     coactivation.index(config.get("coactivation_threshold"))] = value
        image = axis.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
        axis.set_xticks(range(len(coactivation)), [f"{v:g}" for v in coactivation])
        axis.set_yticks(range(len(text)), [f"{v:g}" for v in text])
        axis.set_xlabel("coactivation threshold")
        axis.set_ylabel("text similarity threshold")
        reference_values = _named_graph_values(record, key)
        axis.set_title(key + (f"\n{reference_values}" if reference_values else ""), fontsize=9)
        for (row, column), value in np.ndenumerate(grid):
            if np.isfinite(value):
                label = f"{value:.0f}" if abs(value) >= 10 or float(value).is_integer() else f"{value:.2f}"
                axis.text(column, row, label, ha="center", va="center", fontsize=7, color="white")
        figure.colorbar(image, ax=axis, shrink=0.8)
    for axis in list(axes.flat)[len(keys):]:
        axis.axis("off")
    figure.tight_layout()
    reference = record["inputs"].get("reference_graph")
    notes = [
        "Each cell is one grouping configuration and the teacher graph built from it. Every panel has its "
        "own colour range; the numbers in the cells and the named-graph values under each title set the scale.",
        f"Edge overlap is the Jaccard index with the {html.escape(reference)} graph." if reference else "",
        "Balanced counterfactual averages, over the four (y, a) anchor groups, the edge mass that keeps the class "
        "and flips the spurious attribute. Read it together with balanced same class.",
    ]
    return (
        "<section><h2>Grouping sweep</h2>"
        + "".join(f"<p class='note'>{note}</p>" for note in notes if note)
        + _figure_html(figure, "sweep heatmaps")
        + "</section>"
    )


# ---------------------------------------------------------------- named graphs


def graph_comparison(record: dict) -> str:
    graphs = record.get("graphs", {})
    if not graphs:
        return ""
    headers = ["graph", "edges", "coverage", "max in-degree", "in-degree Gini", "selected groups"]
    has_labels = any("labels" in entry for entry in graphs.values())
    if has_labels:
        headers += ["balanced counterfactual", "balanced same class", "minority reach", "counterfactual (all edges)"]
    rows = []
    for name, entry in graphs.items():
        row = [name, entry["structure"]["edges"], entry["structure"]["coverage"], entry["structure"]["max_indegree"],
               entry["structure"]["indegree_gini"], entry["selection"]["selected_groups"]]
        if has_labels:
            labels = entry.get("labels", {})
            row += [labels.get("group_balanced_counterfactual"), labels.get("group_balanced_same_class"),
                    labels.get("minority_reach"), labels.get("counterfactual")]
        rows.append(row)
    random_note = ""
    if has_labels:
        first = next(entry["labels"] for entry in graphs.values() if "labels" in entry)
        random_note = (
            f"<p class='note'>Degree-matched random neighbours reach balanced counterfactual "
            f"{first['random_group_balanced_counterfactual']:.3f} at balanced same class "
            f"{first['random_group_balanced_same_class']:.3f}: random edges flip the attribute often and lose the class.</p>"
        )
    overlap = "".join(
        f"<p>{html.escape(name)}: "
        + ", ".join(f"{html.escape(other)} {value:.3f}" for other, value in entry["overlap"].items())
        + "</p>"
        for name, entry in graphs.items() if entry["overlap"]
    )
    return (
        "<section><h2>Teacher graphs</h2>" + _table(headers, rows) + random_note
        + ("<h3>Edge overlap (Jaccard)</h3>" + overlap if overlap else "")
        + transition_matrices(record) + calibration_curves(record) + "</section>"
    )


def transition_matrices(record: dict) -> str:
    graphs = {name: entry for name, entry in record.get("graphs", {}).items() if "labels" in entry}
    if not graphs:
        return ""
    names = record.get("group_names") or []
    plt = _plt()
    figure, axes = plt.subplots(1, len(graphs), figsize=(3.6 * len(graphs), 3.4), squeeze=False)
    for axis, (name, entry) in zip(axes.flat, graphs.items()):
        matrix = np.array(entry["labels"]["transition"])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        axis.set_title(name)
        axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
        axis.set_yticks(range(len(names)), names)
        axis.set_xlabel("neighbour group")
        axis.set_ylabel("anchor group")
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                      color="white" if value > 0.5 else "black")
    figure.tight_layout()
    return (
        "<h3>Where edges lead, by (y, a) group</h3>"
        "<p class='note'>Rows sum to one. Counterfactual mass sits in the cells with the same y and a different a.</p>"
        + _figure_html(figure, "transition matrices")
    )


def calibration_curves(record: dict) -> str:
    graphs = {name: entry for name, entry in record.get("graphs", {}).items() if entry.get("labels", {}).get("calibration")}
    if not graphs:
        return ""
    plt = _plt()
    figure, axis = plt.subplots(figsize=(5.2, 3.2))
    for name, entry in graphs.items():
        bins = entry["labels"]["calibration"]
        axis.plot([row["bin"] + 1 for row in bins], [row["counterfactual_rate"] for row in bins], marker="o", label=name)
    axis.set_xlabel("edge confidence decile (low to high)")
    axis.set_ylabel("counterfactual edge rate")
    axis.legend(frameon=False)
    return (
        "<h3>Confidence calibration</h3>"
        "<p class='note'>A rising curve means the relational loss puts more weight on counterfactual edges.</p>"
        + _figure_html(figure, "confidence calibration")
    )


# ---------------------------------------------------------------- concept groups


def concept_group_sections(record: dict) -> str:
    parts = []
    for name, entry in record.get("graphs", {}).items():
        audited = entry.get("audited_groups")
        if audited:
            parts.append(selectivity_scatter(name, audited))
        rows = entry.get("concept_groups")
        if rows:
            parts.append(
                f"<h3>{html.escape(name)}: edges per concept group</h3>"
                + _table(
                    ["group", "concepts", "edges", "same class", "flipped attribute", "counterfactual"],
                    [[row["group_id"], ", ".join(row["concepts"]), row["edges"], row["same_class"],
                      row["flipped_attribute"], row["counterfactual"]] for row in rows[:30]],
                )
            )
    return "<section><h2>Concept groups</h2>" + "".join(parts) + "</section>" if parts else ""


def selectivity_scatter(name: str, audited: list[dict]) -> str:
    plt = _plt()
    figure, axis = plt.subplots(figsize=(5.6, 4.6))
    selected = [row for row in audited if row["selected"]]
    rejected = [row for row in audited if not row["selected"]]
    axis.scatter([row["auc_y"] for row in rejected], [row["auc_a"] for row in rejected], s=10,
                 color=PALETTE["muted"], alpha=0.5, label="not selected")
    axis.scatter([row["auc_y"] for row in selected], [row["auc_a"] for row in selected], s=26,
                 color=PALETTE["accent"], label="selected")
    for row in sorted(selected, key=lambda item: -item["selectivity"])[:15]:
        axis.annotate(", ".join(row["concepts"])[:24], (row["auc_y"], row["auc_a"]), fontsize=6,
                      xytext=(3, 2), textcoords="offset points")
    axis.axhline(0.5, color=PALETTE["muted"], linewidth=0.6)
    axis.axvline(0.5, color=PALETTE["muted"], linewidth=0.6)
    axis.set_xlabel("AUC of group activation for the class y")
    axis.set_ylabel("AUC of group activation for the spurious attribute a")
    axis.set_title(f"{name}: audited concept groups")
    axis.legend(frameon=False)
    top = sorted(audited, key=lambda row: -row["selectivity"])[:15]
    return (
        f"<h3>{html.escape(name)}: which groups carry the spurious attribute</h3>"
        "<p class='note'>Groups far from 0.5 on the vertical axis and near 0.5 on the horizontal axis "
        "encode the spurious attribute and leave the class intact.</p>"
        + _figure_html(figure, "selectivity scatter")
        + _table(["group", "concepts", "selected", "AUC y", "AUC a", "selectivity", "null margin"],
                 [[row["group_id"], ", ".join(row["concepts"]), row["selected"], row["auc_y"], row["auc_a"],
                   row["selectivity"], row["null_margin"]] for row in top])
    )


# ---------------------------------------------------------------- edge gallery


def _thumbnail(image, size: int = 112) -> str:
    image = image.convert("RGB")
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def edge_gallery(record: dict, data_folder: Path | None, names: list[str] | None) -> str:
    if data_folder is None:
        return ""
    from experiments.spurious_eval.datasets.registry import get_dataset_spec

    dataset = get_dataset_spec(record["dataset"])["dataset"](str(data_folder))
    group_names = record.get("group_names") or []
    attribute_count = record.get("attribute_count")
    sections = []
    for name, entry in record.get("graphs", {}).items():
        if names and name not in names:
            continue
        cards = []
        for sample in entry.get("edge_samples", []):
            images = []
            for key in ("anchor", "neighbour"):
                index = int(sample[key].split(":")[1])
                labels = sample.get(f"{key}_labels")
                caption = sample[key]
                if labels and group_names and attribute_count:
                    caption = group_names[labels[0] * attribute_count + labels[1]]
                images.append(f"<figure><img src='{_thumbnail(dataset.get_input(index))}'>"
                              f"<figcaption>{html.escape(caption)}</figcaption></figure>")
            verdict = ""
            if "counterfactual" in sample:
                verdict = "counterfactual" if sample["counterfactual"] else "other"
            cards.append(
                f"<div class='edge {verdict}'>{images[0]}<span class='arrow'>→</span>{images[1]}"
                f"<p>{html.escape(', '.join(sample['concepts']))} · confidence {sample['confidence']:.3f}"
                f" · {html.escape(sample.get('reason', ''))}</p></div>"
            )
        if cards:
            sections.append(f"<h3>{html.escape(name)}</h3><div class='gallery'>{''.join(cards)}</div>")
    if not sections:
        return ""
    return (
        "<section><h2>Edge gallery</h2><p class='note'>Per concept group: its two highest-confidence edges "
        "and two edges from minority-group anchors. A green frame marks an edge that keeps the class and flips "
        "the spurious attribute.</p>"
        + "".join(sections) + "</section>"
    )


# ---------------------------------------------------------------- page

STYLE = """
:root{--bg:#fbfbfa;--fg:#1f2328;--muted:#5b6470;--line:#d8dde3;--panel:#ffffff}
@media (prefers-color-scheme: dark){:root{--bg:#15181c;--fg:#e6e8eb;--muted:#9aa4af;--line:#2d333b;--panel:#1c2026}
img{background:#fff}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
main{max-width:1280px;margin:0 auto;padding:24px 16px 64px}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;margin:18px 0;overflow-x:auto}
h1{margin:0 0 4px}h2{margin-top:0}.note{color:var(--muted);margin:4px 0}
table{border-collapse:collapse;font-size:12px;margin:8px 0}th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}
img{max-width:100%}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:10px}
.edge{border:2px solid var(--line);border-radius:8px;padding:6px;display:flex;flex-wrap:wrap;align-items:center;gap:4px}
.edge.counterfactual{border-color:#2e8b57}.edge p{width:100%;margin:2px 0;font-size:11px;color:var(--muted)}
figure{margin:0;text-align:center}figcaption{font-size:10px;color:var(--muted)}.arrow{font-size:20px}
"""


def render(record: dict, output: Path, *, data_folder: Path | None = None, gallery_graphs: list[str] | None = None) -> Path:
    tiers = record.get("tiers", {})
    header = (
        f"<h1>CoSpRo diagnostics · {html.escape(record['dataset'])}</h1>"
        f"<p class='note'>Record {html.escape(record.get('created', ''))} · SpLiCE cache metrics "
        f"{'included' if tiers.get('cache') else 'absent'} · post-hoc label metrics "
        f"{'included' if tiers.get('labels') else 'absent'}</p>"
    )
    body = (
        header + graph_comparison(record) + concept_group_sections(record) + sweep_heatmaps(record)
        + edge_gallery(record, data_folder, gallery_graphs)
    )
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>CoSpRo diagnostics</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


def render_file(record_path: Path, output: Path, **kwargs) -> Path:
    return render(json.loads(record_path.read_text(encoding="utf-8")), output, **kwargs)
