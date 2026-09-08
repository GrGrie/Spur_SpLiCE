"""Finite composite-group search, held-apart frozen check, and six-run SSL screen."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import torch

from splice.concept_group_search import group_metrics, propose, select_groups, split_rows, subset
from splice.crp import CrpAuditConfig, run_frozen_audit, validate_feature_cache
from splice.crp_training import validate_teacher_graph
from splice.graph_io import graph_fingerprint, load_graph_json, save_graph_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/concept_group_search_v1"
BASE = ROOT / "outputs/crp_controls_logistic_v1_cluster_seeds12"
CACHE = BASE / "waterbirds_train_features.pt"
REFERENCE = BASE / "graphs/crp_graph.json"
POLICIES = ["baseline", "semantic", "compact"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def inputs():
    if not CACHE.is_file() or not REFERENCE.is_file():
        raise FileNotFoundError(f"Expected cache and reference graph: {CACHE}, {REFERENCE}")
    cache = validate_feature_cache(torch.load(CACHE, map_location="cpu", weights_only=True))
    graph = validate_teacher_graph(load_graph_json(REFERENCE), cache["sample_ids"])
    return cache, graph


def prepare():
    if (ROOT / "outputs/paper_completion_2026-09-08/final_test_lock.json").exists():
        raise RuntimeError("A main-paper test lock exists. Keep this new validation search in a separate research checkout/output; do not tune the locked paper.")
    cache, graph = inputs()
    left, right = split_rows(cache["sample_ids"])
    discovery = subset(cache, left)
    composites = propose(discovery)
    # Constituent controls distinguish joint removal from one useful member.
    groups = composites + [[i] for i in sorted({i for group in composites for i in group})]
    for group in graph["groups"]:
        if group["selected"] and sorted(group["concept_indices"]) not in groups:
            groups.append(sorted(group["concept_indices"]))
    vocabulary_probe = []
    for word in ("water", "sea", "lake", "pond"):
        matches = [i for i, name in enumerate(cache["vocabulary"]) if name.casefold().strip() == word]
        vocabulary_probe.append({"word": word, "matches": [
            {"index": i, "frequency_discovery": float((discovery["splice_codes"][:, i] > 0).float().mean())}
            for i in matches]})
    payload = {"groups": groups, "composite_count": len(composites),
               "concepts": [[cache["vocabulary"][i] for i in g] for g in groups],
               "discovery_rows": left, "confirmation_rows": right,
               "cache_fingerprint": graph_fingerprint(CACHE), "reference_fingerprint": graph_fingerprint(REFERENCE),
               "water_words_diagnostic_only": vocabulary_probe,
               "budget": "<=24 composites + constituent singletons + original selected groups; <=6 SSL runs",
               "selection": "discovery only; confirmation is a fixed pass/fail gate, never reranking",
               "shared_preprocessing": "original frozen dictionary, image_mean and SpLiCE cache; not an independent dataset"}
    payload['source_fingerprints'] = {str(p.relative_to(ROOT)): graph_fingerprint(p) for p in
        [Path(__file__), ROOT / 'splice/concept_group_search.py', ROOT / 'splice/crp.py']}
    path = OUT / "candidates.json"
    if path.exists() and read(path) != payload:
        raise RuntimeError("Candidate pool changed; use a new explicitly versioned study.")
    write(path, payload)
    print(f"Prepared {len(composites)} composites, {len(groups)} total groups: {path}")


def audit(task):
    cache, reference = inputs()
    manifest = read(OUT / "candidates.json")
    name = ["discovery", "confirmation"][task]
    path = OUT / f"{name}.json"
    if path.exists():
        if read(path)['candidate_fingerprint'] != graph_fingerprint(OUT / 'candidates.json'):
            raise RuntimeError('Existing audit has a different candidate identity.')
        print(f"Reuse {path}")
        return
    data = subset(cache, manifest[f"{name}_rows"])
    cfg = replace(CrpAuditConfig(**reference["config"]), max_selected_groups=0, null_trials=32,
                  seed=101 + task)
    graph = run_frozen_audit(data, cfg, candidate_groups=manifest["groups"])
    write(path, {"groups": graph["groups"], "config": graph["config"],
                 "metrics": [group_metrics(data, group) for group in manifest["groups"]],
                 "cache_fingerprint": graph_fingerprint(CACHE),
                 "candidate_fingerprint": graph_fingerprint(OUT / "candidates.json")})


def select():
    manifest = read(OUT / "candidates.json")
    discovery, confirmation = [read(OUT / f"{name}.json") for name in ("discovery", "confirmation")]
    for report in (discovery, confirmation):
        if report["candidate_fingerprint"] != graph_fingerprint(OUT / "candidates.json"):
            raise RuntimeError("Candidate pool changed after audit.")
    selections = {}
    for policy in POLICIES[1:]:
        chosen = select_groups(discovery["groups"], discovery["metrics"], policy)
        multi = [i for i in chosen if len(manifest["groups"][i]) > 1]
        survivors = [i for i in chosen if confirmation["groups"][i]["selected"]
                     and confirmation["metrics"][i]["removed_energy_p95"] <= .5]
        # Predeclared screen: >=half the frozen shortlist survives; at least one composite survives.
        passed = bool(multi) and len(survivors) >= max(1, (len(chosen)+1)//2) and bool(set(multi) & set(survivors))
        selections[policy] = {"chosen": chosen, "composites": multi, "confirmation_survivors": survivors,
                              "passed": passed, "groups": [manifest["groups"][i] for i in chosen]}
    path = OUT / "selection.json"
    payload = {"policies": selections, "candidate_fingerprint": graph_fingerprint(OUT / "candidates.json"),
               "endpoint": "500 epochs; val only; seeds 1/3; fixed lambda=.5; no automatic expansion"}
    if path.exists() and read(path) != payload:
        raise RuntimeError("Selection already frozen and differs.")
    write(path, payload)
    print(json.dumps(payload, indent=2))


def build(task):
    policy = POLICIES[task+1]
    selection = read(OUT / "selection.json")["policies"][policy]
    if not selection["passed"]:
        print(f"{policy}: negative frozen screen; no full graph or SSL.")
        return
    cache, reference = inputs()
    path = OUT / "graphs" / f"{policy}.json"
    if path.exists():
        validate_teacher_graph(load_graph_json(path), cache["sample_ids"])
        return
    cfg = replace(CrpAuditConfig(**reference["config"]), max_selected_groups=12, null_trials=32, seed=0)
    graph = run_frozen_audit(cache, cfg, candidate_groups=selection["groups"])
    graph["group_search"] = {"policy": policy, "selection_fingerprint": graph_fingerprint(OUT / "selection.json")}
    validate_teacher_graph(graph, cache["sample_ids"])
    save_graph_json(graph, path)


def ready_graphs():
    """Do not mistake an absent build for a negative result."""
    selection = read(OUT / "selection.json")
    graphs = {"baseline": REFERENCE}
    for policy, record in selection["policies"].items():
        if not record["passed"]:
            continue
        path = OUT / "graphs" / f"{policy}.json"
        graph = load_graph_json(path)
        if graph["group_search"]["selection_fingerprint"] != graph_fingerprint(OUT / "selection.json"):
            raise RuntimeError("Full graph has stale selection identity.")
        has_composite = any(g["selected"] and len(g["concept_indices"]) > 1 for g in graph["groups"])
        coverage = float((graph["weights"].sum(1) > 0).float().mean())
        # Prevent an effectively empty teacher from consuming a 500-epoch run.
        if has_composite and coverage >= .5:
            graphs[policy] = path
    return graphs


def train(task):
    from scripts.tools.run_crp_controls import training_command
    inputs()  # exact reference identity
    seed = [1, 3][task // 3]
    policy = POLICIES[task % 3]
    graphs = ready_graphs()
    if len(graphs) == 1 or policy not in graphs:
        print(f"Skip {policy}: no eligible composite treatment; report negative screen.")
        return
    output = OUT / "ssl" / f"seed{seed}" / policy
    fingerprint = graph_fingerprint(graphs[policy])
    status = output / "completed.json"
    if status.exists():
        record = read(status)
        if record["graph_fingerprint"] != fingerprint:
            raise RuntimeError("Completed run graph changed.")
        if not Path(record["result"]).exists():
            raise RuntimeError("Completed record has no final probe.")
        print(f"Reuse {status}")
        return
    if list(output.glob("training/*/args.json")):
        raise RuntimeError(f"Incomplete run exists at {output}; recover/archive before retrying.")
    cfg = read(ROOT / "scripts/run_crp_controls_cluster.conf")
    cfg.update(relational_weight=.5, delete_checkpoints_after_training=False,
               wandb_group="concept_group_search_v1", num_workers=4)
    command = training_command(cfg, seed, "splice_crp_kl", graphs[policy], output)
    command[command.index("--wandb_run_name")+1] = f"group_search_{policy}_seed{seed}"
    command += ["--save_freq", "50", "--checkpoint_keep_count", "2",
                "--delete_epoch_checkpoints_after_training", "true"]
    write(output / "command.json", {"command": command, "graph_fingerprint": fingerprint})
    subprocess.run(command, check=True, cwd=ROOT)
    results = list(output.glob("training/*/probe_features_epoch_500_ds_train_val.json"))
    if len(results) != 1 or not read(results[0]).get("convergence", {}).get("converged"):
        raise RuntimeError("Expected one converged epoch-500 validation probe.")
    write(status, {"result": str(results[0]), "graph_fingerprint": fingerprint, "seed": seed, "policy": policy})


def summary():
    rows, decisions = [], {}
    eligible = ready_graphs()
    if len(eligible) > 1:
        for seed in [1, 3]:
            for policy in eligible:
                status = read(OUT / "ssl" / f"seed{seed}" / policy / "completed.json")
                result = read(status["result"])
                metrics = result["metrics"]
                rows.append({"seed": seed, "policy": policy,
                             "avg": metrics["Average over last 10 linear val acc"],
                             "wga": metrics["Average over last 10 linear val worst-group acc"],
                             "groups": result["group_metrics"], "result": status["result"]})
        for policy in list(eligible)[1:]:
            deltas = []
            for seed in [1, 3]:
                by_policy = {r["policy"]: r for r in rows if r["seed"] == seed}
                deltas.append({"seed": seed, **{m: by_policy[policy][m]-by_policy["baseline"][m] for m in ["avg", "wga"]}})
            passed = all(d[m] > 0 for d in deltas for m in ["avg", "wga"])
            passed &= sum(d["avg"] for d in deltas)/2 >= 1 and sum(d["wga"] for d in deltas)/2 >= 2
            decisions[policy] = {"paired_deltas": deltas, "screen_passed": passed}
    write(OUT / "ssl_summary.json", {"rows": rows, "decisions": decisions,
          "frozen_selection": read(OUT / "selection.json"), "eligible": list(eligible),
          "interpretation": "two-seed validation screen; graph support/mass and batches differ; no causal or confirmed superiority claim",
          "automatic_expansion": False})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["prepare", "audit", "select", "build", "train", "summary"])
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    if args.stage != 'prepare':
        manifest = read(OUT / 'candidates.json')
        for path, expected in manifest['source_fingerprints'].items():
            if graph_fingerprint(ROOT / path) != expected:
                raise RuntimeError(f'Search implementation changed after preparation: {path}')
    if not 0 <= args.task < (6 if args.stage == "train" else 2):
        parser.error("Invalid task ID")
    if args.stage in {"audit", "build", "train"}:
        globals()[args.stage](args.task)
    else:
        globals()[args.stage]()


if __name__ == "__main__":
    main()
