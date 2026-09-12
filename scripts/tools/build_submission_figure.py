"""Render a compact, post-hoc Waterbirds relation audit from existing graph artifacts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from splice.artifacts import atomic_write_json, resolve_output_root, sha256_file
from scripts.tools.render_concept_panels import _metadata, _annotate, decision_reason, render


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def annotation(sample_id, metadata, dataset_root):
    prefix, index = sample_id.split(":")
    if prefix != "waterbirds" or not 0 <= int(index) < len(metadata):
        raise ValueError(f"Invalid sample ID: {sample_id}")
    if int(metadata[int(index)]["split"]) != 0:
        raise ValueError(f"Graph must contain training images only: {sample_id}")
    return _annotate(sample_id, metadata, dataset_root)


def graph_mass(graph, metadata, dataset_root):
    """Use the actual objective's q_i * p_T(j|i), grouped by source (y,a)."""
    ids = graph["sample_ids"]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate graph sample IDs")
    labels = [annotation(i, metadata, dataset_root) for i in ids]
    totals = {f"{y}{a}": dict(total=0., cross_background=0., wrong_target=0., same_target_same_background=0.)
              for y in range(2) for a in range(2)}
    if any(len(graph[k]) != len(ids) for k in ("weights", "neighbor_indices", "anchor_confidence")):
        raise ValueError("Misaligned graph rows")
    for i, (neighbors, weights, q) in enumerate(zip(graph["neighbor_indices"], graph["weights"], graph["anchor_confidence"])):
        if len(neighbors) != len(weights) or not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("Invalid graph row/confidence")
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Invalid edge weights")
        if sum(weights) and not math.isclose(sum(weights), 1., abs_tol=1e-5):
            raise ValueError("Teacher preferences must sum to one")
        source = labels[i]
        row = totals[f"{source['target']}{source['spurious_attribute']}"]
        for j, w in zip(neighbors, weights):
            if not w:
                continue
            if not 0 <= j < len(ids) or j == i:
                raise ValueError("Invalid graph destination")
            donor = labels[j]
            category = ("wrong_target" if source["target"] != donor["target"] else
                        "cross_background" if source["spurious_attribute"] != donor["spurious_attribute"] else
                        "same_target_same_background")
            row[category] += q * w
            row["total"] += q * w
    return {key: {**row, "percent": {k: 100 * row[k] / row["total"] if row["total"] else None
                                     for k in ("cross_background", "wrong_target", "same_target_same_background")}}
            for key, row in totals.items()}


def enrich(panels, graph, metadata, dataset_root):
    groups = {g["group_id"]: g for g in graph["groups"]}
    pairs = []
    for group in panels["groups"]:
        actual = groups[group["group_id"]]
        if group["concepts"] != actual["concepts"]:
            raise ValueError("Panel concept names do not match the graph")
        for pair in group["pairs"]:
            i, j = pair["row"], pair["column"]
            if not 0 <= i < len(graph["sample_ids"]) or not 0 <= j < len(graph["sample_ids"]):
                raise ValueError("Panel row/column outside graph")
            if graph["sample_ids"][i] != pair["left_id"] or graph["sample_ids"][j] != pair["right_id"]:
                raise ValueError("Panel IDs do not match graph row order")
            retained = any(d == j and w > 0 for d, w in zip(graph["neighbor_indices"][i], graph["weights"][i]))
            if retained != pair["retained"]:
                raise ValueError("Stale panel retention status")
            left = annotation(pair["left_id"], metadata, dataset_root)
            right = annotation(pair["right_id"], metadata, dataset_root)
            same = left["target"] == right["target"]
            relation = "wrong_target" if not same else ("cross_background" if left["spurious_attribute"] != right["spurious_attribute"] else "same_background")
            pairs.append({**pair, "group_id": group["group_id"], "concepts": group["concepts"],
                          "left_annotation": left, "right_annotation": right, "relation": relation,
                          "decision_reason": decision_reason(pair, actual, graph["config"])})
    return pairs


def select_examples(pairs):
    """Prespecified illustrative categories; never claim these are prevalence estimates."""
    definitions = [("Retained: same target, different background", True, "cross_background"),
                   ("Retained: incorrect target relation", True, "wrong_target"),
                   ("Rejected: same target, different background", False, "cross_background"),
                   ("Rejected: different targets", False, "wrong_target")]
    selected = []
    for title, retained, relation in definitions:
        candidates = [p for p in pairs if p["retained"] == retained and p["relation"] == relation]
        candidates.sort(key=lambda p: (-p["gain"], p["group_id"], p["left_id"], p["right_id"]))
        selected.append(dict(title=title, candidate_count=len(candidates), pair=candidates[0] if candidates else None))
    return selected


def draw_figure(selected, mass, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    figure = plt.figure(figsize=(12, 10), layout="constrained")
    grid = figure.add_gridspec(3, 2, height_ratios=[1.05, 1.05, .85])
    for n, selection in enumerate(selected):
        slot = grid[n // 2, n % 2].subgridspec(2, 2, height_ratios=[3.3, 1.5])
        pair = selection["pair"]
        title = figure.add_subplot(slot[1, :]); title.axis("off")
        if pair is None:
            title.text(0, .95, selection["title"] + "\nNo example in the frozen 20-pair selection.", va="top", wrap=True)
            continue
        for side, column in (("left_annotation", 0), ("right_annotation", 1)):
            ax = figure.add_subplot(slot[0, column]); ax.axis("off")
            item = pair[side]
            with Image.open(item["image"]) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{item['sample_id']}\n{item['target_name']} / {item['background_name']}", fontsize=9)
        text = (f"{selection['title']}\nConcept: {', '.join(pair['concepts'])}\n"
                f"cosine {pair['raw_similarity']:.3f} → {pair['projected_similarity']:.3f}; "
                f"gain {pair['gain']:+.4f}; residual {pair['residual_similarity']:.3f}\n"
                f"activation contrast {pair['activation_contrast']:.4f}; edge weight {pair['final_edge_weight']:.3f}")
        title.text(0, .98, text, va="top", fontsize=9)
    colors = ("#2878a4", "#ba5a32")
    for col, (metric, heading) in enumerate((("cross_background", "Same target, different background"),
                                           ("wrong_target", "Wrong target (either background)"))):
        ax = figure.add_subplot(grid[2, col]); x = np.arange(4)
        for offset, (arm, color) in enumerate(zip(("raw_clip", "cospro"), colors)):
            values = [mass[arm][group]["percent"][metric] for group in ("00", "01", "10", "11")]
            bars = ax.bar(x + (offset - .5) * .36, [v if v is not None else 0 for v in values], .36,
                          label="Raw CLIP" if arm == "raw_clip" else "CoSpRo", color=color)
            ax.bar_label(bars, labels=[f"{v:.1f}" if v is not None else "N/A" for v in values], fontsize=8, padding=3)
        ax.set(title=heading, ylabel="Within-source graph mass (%)", xlabel="Source group (target, background)",
               xticks=x, xticklabels=["(0,0)", "(0,1)", "(1,0)", "(1,1)"], ylim=(0, 100))
        ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    figure.savefig(output / "graph_evidence.pdf", bbox_inches="tight")
    figure.savefig(output / "graph_evidence.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def build(args):
    root = args.artifact_root
    source = args.panels or root / "reports/paper_evidence/visual/concept_panels.json"
    crp_path = root / "shared/waterbirds/graphs/crp_graph.json"
    raw_path = root / "shared/waterbirds/graphs/raw_clip_graph.json"
    crp, raw = read(crp_path), read(raw_path)
    if crp["sample_ids"] != raw["sample_ids"]:
        raise ValueError("Raw and CoSpRo graph sample IDs differ")
    metadata = _metadata(args.dataset_root)
    pairs = enrich(read(source), crp, metadata, args.dataset_root)
    for pair in pairs:
        for side in ("left_annotation", "right_annotation"):
            if not Path(pair[side]["image"]).is_file():
                raise FileNotFoundError(pair[side]["image"])
    mass = {arm: graph_mass(graph, metadata, args.dataset_root) for arm, graph in (("raw_clip", raw), ("cospro", crp))}
    selected = select_examples(pairs)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "evidence.json", dict(
        posthoc_only=True, selection_rule="Within each declared category in the frozen pool: descending gain, then group/sample IDs. Missing categories remain empty.",
        interpretation="Illustrative label-stratified examples, not unbiased samples. Retained does not imply correct target semantics. Bars use all graph edges, q_i * p_T(j|i).",
        sources={str(p): sha256_file(p) for p in (source, crp_path, raw_path, args.dataset_root / "metadata.csv")},
        examples=selected, graph_mass=mass, all_pairs=pairs))
    draw_figure(selected, mass, output)
    # Reuse the detailed renderer on a COPY, preserving the frozen selection manifest.
    details = output / "details"
    details.mkdir()
    atomic_write_json(details / "concept_panels.json", read(source))
    render(args.dataset_root, root, details)
    caption = ("Post-hoc audit of CoSpRo relations. Image pairs illustrate prespecified retained/rejected and target/context categories "
               "within the frozen 20-pair pool; the highest-gain pair in each category is shown, and absent categories are explicitly marked. "
               "These examples are illustrative, not prevalence estimates. Labels are used only for this audit. "
               "Bars aggregate all edges using anchor confidence times normalized teacher probability, separately within each source group. "
               "Wrong-target relations include both backgrounds. Projection can reveal cross-background relations while also linking different targets.")
    (output / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    tex = ("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{\\detokenize{" +
           (output / "graph_evidence.pdf").as_posix() + "}}\n\\caption{" + caption +
           "}\n\\label{fig:graph-evidence}\n\\end{figure}\n")
    (output / "figure.tex").write_text(tex, encoding="utf-8")
    print(f"Wrote compact PDF/PNG, full panels, audit JSON and LaTeX: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory containing Waterbirds metadata.csv and images")
    parser.add_argument("--artifact-root", type=Path, default=resolve_output_root())
    parser.add_argument("--panels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing reports are never overwritten")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
