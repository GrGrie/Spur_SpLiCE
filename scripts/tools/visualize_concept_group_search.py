"""Post-hoc visual evidence and blinded pair rating; never read by selection."""
from __future__ import annotations

import argparse
import csv
import html
import random
from pathlib import Path

import torch

from scripts.tools.run_concept_group_search import OUT, POLICIES, REFERENCE, inputs, read, write
from splice.crp import orthonormal_basis, project_out, topk_neighbors
from splice.graph_io import load_graph_json


def sample_pairs(graph, seed=912, per_band=12):
    """Stratify by effective edge mass, q_i*p(j|i), not row probability alone."""
    neighbors = graph["neighbor_indices"]
    mass = graph["weights"] * graph["anchor_confidence"][:, None]
    entries = [(float(mass[i, j]), i, int(neighbors[i, j]), j)
               for i, j in torch.nonzero((neighbors >= 0) & (mass > 0)).tolist()]
    entries.sort()
    rng = random.Random(seed)
    selected = []
    for band, quantile in [("low", 0), ("middle", .45), ("high", .9)]:
        start = int(len(entries) * quantile)
        pool = entries[start:min(len(entries), start + max(1, len(entries)//10))]
        chosen = rng.sample(pool, min(per_band, len(pool)))
        for value, i, j, slot in chosen:
            selected.append({"band": band, "row": i, "donor": j, "mass": value,
                             "weight": float(graph["weights"][i, slot]),
                             "confidence": float(graph["anchor_confidence"][i]),
                             "degree": int((neighbors[i] >= 0).sum())})
    # Matched-anchor random donors; random pairs can genuinely share class/context.
    for item in [p for p in selected if p['band'] == 'high']:
        i = item["row"]
        j = rng.randrange(len(neighbors)-1)
        j += j >= i
        selected.append({"band": "random", "row": i, "donor": j,
                         "mass": None, "weight": None, "confidence": None, "degree": None})
    rng.shuffle(selected)
    return selected


def relation_curve(graph, y, context):
    neighbors = graph["neighbor_indices"]
    mask = (neighbors >= 0) & (graph["weights"] > 0)
    rows, slots = torch.where(mask)
    donors = neighbors[rows, slots]
    mass = (graph["weights"] * graph["anchor_confidence"][:, None])[rows, slots]
    order = torch.argsort(mass, stable=True)
    bins = []
    generator = torch.Generator().manual_seed(913)
    permutations = [torch.randperm(len(y), generator=generator) for _ in range(100)]
    for ids in torch.tensor_split(order, 5):
        if not len(ids):
            continue
        r, d = rows[ids], donors[ids]
        same = y[r] == y[d]
        chance = torch.tensor([float((y[p[r]] == y[p[d]]).float().mean()) for p in permutations])
        bins.append({"edges": len(ids), "mass_min": float(mass[ids].min()), "mass_max": float(mass[ids].max()),
                     "same_target": float(same.float().mean()),
                     "same_target_cross_context": float((same & (context[r] != context[d])).float().mean()),
                     "same_context": float((context[r] == context[d]).float().mean()),
                     "chance_same_target_mean": float(chance.mean()),
                     "chance_same_target_p025": float(torch.quantile(chance, .025)),
                     "chance_same_target_p975": float(torch.quantile(chance, .975))})
    return bins


def summarize_ratings(task):
    """Report blinded ratings, with matched-anchor uncertainty, without selecting graphs."""
    dest = OUT / 'visual' / POLICIES[task]
    key = {item['pair_id']: item for item in read(dest / 'pair_key.json')}
    with (dest / 'ratings.csv').open(encoding='utf-8-sig', newline='') as stream:
        ratings = list(csv.DictReader(stream))
    if len(ratings) != len(key) or {r['pair_id'] for r in ratings} != set(key):
        raise ValueError('Ratings must contain each blinded pair exactly once.')
    output = {}
    for field in ['shared_object_0_2', 'shared_background_0_2']:
        bands = {}
        for row in ratings:
            if row[field] not in {'0', '1', '2'}:
                raise ValueError(f'Complete 0/1/2 ratings first: {row["pair_id"]}, {field}')
            item = key[row['pair_id']]
            bands.setdefault(item['band'], {}).setdefault(item['row'], []).append(int(row[field]))
        means = {band: {anchor: sum(values)/len(values) for anchor, values in anchors.items()}
                 for band, anchors in bands.items()}
        common = sorted(set(means.get('high', {})) & set(means.get('random', {})))
        deltas = [means['high'][anchor] - means['random'][anchor] for anchor in common]
        rng = random.Random(914)
        boots = sorted(sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(2000)) if deltas else []
        output[field] = {'mean_by_band_equal_anchor_weight': {
            band: sum(values.values())/len(values) for band, values in means.items()},
            'matched_anchor_count': len(common),
            'high_minus_random': sum(deltas)/len(deltas) if deltas else None,
            'paired_anchor_bootstrap_95': [boots[49], boots[1949]] if boots else None}
    write(dest / 'human_ratings_summary.json', {'ratings': output,
          'usage': 'posthoc only, never graph selection; ordinal scores, small sample, descriptive uncertainty'})


def render(task, data_folder):
    from experiments.spurious_eval.datasets.waterbirds import WaterbirdsDataset
    from scripts.tools.crp_posthoc_diagnostics import diagnose_fixed_graphs
    cache, _ = inputs()
    policy = POLICIES[task]
    # Failed full-graph screens still deserve a visual diagnosis.
    available = {"baseline": REFERENCE}
    available.update({name: OUT / "graphs" / f"{name}.json" for name in POLICIES[1:]
                      if (OUT / "graphs" / f"{name}.json").exists()})
    if policy not in available:
        print(f"No eligible {policy} graph; no fabricated visual examples.")
        return
    graph = load_graph_json(available[policy])
    if policy != 'baseline':
        from splice.graph_io import graph_fingerprint
        if graph['group_search']['selection_fingerprint'] != graph_fingerprint(OUT / 'selection.json'):
            raise RuntimeError('Visual graph belongs to a different frozen selection.')
    dest = OUT / "visual" / policy
    dest.mkdir(parents=True, exist_ok=True)
    dataset = WaterbirdsDataset(str(data_folder))
    source_ids = [int(str(i).rsplit(":", 1)[1]) for i in cache["sample_ids"]]
    y = dataset.y_array[source_ids]
    context = dataset.metadata_array[source_ids, 0]

    def picture(row, caption=""):
        name = f"image_{row}.jpg"
        if not (dest / name).exists():
            image = dataset.get_input(source_ids[row]).convert("RGB")
            image.thumbnail((260, 210))
            image.save(dest / name, quality=90)
        return f'<figure><img src="{name}" alt="image"><figcaption>{html.escape(caption)}</figcaption></figure>'

    style = '<meta charset="utf-8"><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;background:#f5f4ef;color:#222}section{background:white;padding:20px;margin:20px 0;border:1px solid #ddd} .row{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;max-width:270px}img{max-width:260px;max-height:210px}figcaption{font-size:13px;white-space:pre-wrap}h1,h2{line-height:1.3}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:7px}</style>'
    pairs = sample_pairs(graph)
    blind = [style, '<h1>Blinded image-pair assessment</h1><p>Rate shared object/category and shared background separately (0=no, 1=uncertain, 2=yes). Use ratings.csv. No method score is shown.</p>']
    for index, pair in enumerate(pairs):
        pair["pair_id"] = f"pair_{index:03d}"
        pair["anchor_id"] = cache["sample_ids"][pair["row"]]
        pair["donor_id"] = cache["sample_ids"][pair["donor"]]
        blind += [f'<section><h2>{pair["pair_id"]}</h2><div class="row">', picture(pair["row"]), picture(pair["donor"]), '</div></section>']
    (dest / "blind_pairs.html").write_text("\n".join(blind), encoding="utf-8")
    write(dest / "pair_key.json", pairs)
    # Never overwrite filled-in ratings on a repeat render.
    rating_path = dest / "ratings.csv"
    if not rating_path.exists():
        with rating_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["pair_id", "shared_object_0_2", "shared_background_0_2", "notes"])
            writer.writerows([[p["pair_id"], "", "", ""] for p in pairs])

    report = [style, f'<h1>Concept and relation audit: {policy}</h1>',
              '<p>Scores are not probabilities of semantic correctness. Selection is deterministic; failures remain visible. Only training-image annotations are used post hoc.</p>']
    selected = [g for g in graph["groups"] if g["selected"]]
    selected.sort(key=lambda g: (-len(g["concept_indices"]), -g["null_excess_score"], g["group_id"]))
    raw_neighbors, _ = topk_neighbors(cache["centered_clip"], 1, 512)
    examples = []
    rejected = sorted([g for g in graph['groups'] if not g['selected']],
                      key=lambda g: (-g['score'], g['group_id']))
    for group in selected[:5] + rejected[:1]:
        members = group["concept_indices"]
        activation = cache["splice_codes"][:, members].sum(1)
        top = torch.argsort(activation, descending=True, stable=True)[:4].tolist()
        basis = orthonormal_basis(cache["dictionary"][members])
        projected = project_out(cache["centered_clip"], basis)
        projected_neighbors, _ = topk_neighbors(projected, 1, 512)
        report += [f'<section><h2>{html.escape(", ".join(group["concepts"]))}</h2>',
                   f'<p>selected={group["selected"]}; rank={basis.shape[1]}; score={group["score"]:.5g}; null threshold={group["null_threshold"]:.5g}</p><div class="row">']
        for row in top:
            contributions = [f'{cache["vocabulary"][i]}={float(cache["splice_codes"][row,i]):.3g}' for i in members]
            report.append(picture(row, f'{cache["sample_ids"][row]}\n' + "; ".join(contributions)))
        report.append('</div><h3>Anchor → raw neighbour → projected neighbour</h3>')
        # Top activator and a fixed low-activation anchor, without inspecting class labels.
        anchors = [top[0], int(torch.argsort(activation, stable=True)[len(activation)//4])]
        for row in anchors:
            raw, donor = int(raw_neighbors[row, 0]), int(projected_neighbors[row, 0])
            before = float(cache["centered_clip"][row] @ cache["centered_clip"][donor])
            after = float(projected[row] @ projected[donor])
            retained = bool(((graph["neighbor_indices"][row] == donor) & (graph["weights"][row] > 0)).any())
            record = {"group": group["concepts"], "row": row, "raw_neighbor": raw, "projected_neighbor": donor,
                      "raw_cosine_to_projected_neighbor": before, "projected_cosine": after, "retained": retained}
            examples.append(record)
            report += ['<div class="row">', picture(row, "anchor: " + str(cache["sample_ids"][row])),
                       picture(raw, "raw nearest neighbour"), picture(donor, f"projected nearest neighbour\ncos {before:.3f} → {after:.3f}; retained={retained}"), '</div>']
        report.append('</section>')
    # Explicit counterexample, selected only for post-hoc display after graph freeze.
    neighbors = graph['neighbor_indices']
    valid = (neighbors >= 0) & (graph['weights'] > 0)
    wrong = valid & (y[:, None] != y[neighbors.clamp_min(0)])
    if wrong.any():
        mass = graph['weights'] * graph['anchor_confidence'][:, None]
        position = int(mass.masked_fill(~wrong, -1).flatten().argmax())
        row, slot = divmod(position, neighbors.shape[1])
        donor = int(neighbors[row, slot])
        report += ['<section><h2>Failure case: largest-mass retained wrong-class edge</h2><p>Post-hoc class labels select this counterexample; it is not part of blind-pair sampling.</p><div class="row">',
                   picture(row, str(cache['sample_ids'][row])), picture(donor, f'mass={float(mass[row,slot]):.5g}'), '</div></section>']
    curve = relation_curve(graph, y, context)
    report += ['<section><h2>All retained edges, five equal-count mass bins</h2><p>Random reference: 100 joint image-annotation permutations, seed 913; class agreement is only a proxy for visible semantic agreement.</p>',
               '<table><tr><th>Bin</th><th>Edges</th><th>Same class</th><th>Same class / different background</th><th>Chance same class</th></tr>']
    for i, row in enumerate(curve):
        report.append(f'<tr><td>{i+1}</td><td>{row["edges"]}</td><td>{row["same_target"]:.3f}</td><td>{row["same_target_cross_context"]:.3f}</td><td>{row["chance_same_target_mean"]:.3f}</td></tr>')
    report += ['</table></section>']
    (dest / "index.html").write_text("\n".join(report), encoding="utf-8")
    write(dest / "diagnostics.json", {"curve": curve, "examples": examples,
          "posthoc": diagnose_fixed_graphs({policy: graph}, "waterbirds", str(data_folder)),
          "selection_use": "forbidden; descriptive posthoc only"})
    print(dest / "index.html")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--task", type=int, choices=range(3), required=True)
    parser.add_argument("--data-folder", type=Path, default=Path("/home/xar68reb/Datasets"))
    parser.add_argument('--summarize-ratings', action='store_true')
    args = parser.parse_args()
    if args.summarize_ratings:
        summarize_ratings(args.task)
    else:
        render(args.task, args.data_folder)


if __name__ == "__main__":
    main()
