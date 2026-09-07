"""Staged paper runs. Existing training/probe entry points remain authoritative."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/paper_completion_2026-09-08"
BASE = ROOT / "outputs/crp_controls_logistic_v1_cluster_seeds12"
ARMS = ["simclr", "crp_sampler_only", "raw_clip_sampler_only", "raw_clip_kl", "splice_crp_kl"]
CONFIGS = {
    "direct": ROOT / "scripts/next_actions_direct_transfer_2026-09-07.conf",
    "graph": ROOT / "scripts/next_actions_graph_ablation_2026-09-07.conf",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(module, *args):
    subprocess.run([sys.executable, "-u", "-m", module, *map(str, args)], check=True, cwd=ROOT)


def source_root(seed, arm):
    if arm.endswith("_kl"):
        name = "crp_signal_checks_v1/transfer/lambda_0.5" if seed < 3 else "crp_lambda05_replication_s34"
    elif seed >= 3:
        name = "crp_followup_controls_v1_seeds34"
    elif arm == "raw_clip_sampler_only":
        name = "crp_followup_raw_sampler_only_v1_seeds12"
    else:
        name = BASE.name
    return ROOT / "outputs" / name / f"seed{seed}" / arm


def matrix():
    return [{"task_id": i, "seed": seed, "arm": arm,
             "source": str(source_root(seed, arm))}
            for i, (seed, arm) in enumerate((s, a) for s in [1, 2, 3, 4] for a in ARMS)]


def checkpoint(root):
    """Accept only actual epoch-500 encoder checkpoints, never a filename alone."""
    import torch
    found = []
    for path in sorted(Path(root).rglob("*.pth")):
        if path.name not in {"last.pth", "epoch_500.pth"}:
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("epoch") == 500 and any(k.startswith("encoder.") for k in payload.get("model", {})):
            found.append(path)
        del payload
    parents = {p.parent for p in found}
    if len(parents) > 1:
        raise RuntimeError(f"Multiple final models under {root}; resolve before continuing: {found}")
    return next((p for p in found if p.name == "last.pth"), found[0] if found else None)


def inventory():
    rows = matrix()
    for row in rows:
        fresh = OUT / "core" / f"seed{row['seed']}" / row["arm"]
        model = checkpoint(row["source"]) or checkpoint(fresh)
        row["checkpoint"] = str(model) if model else None
        row["status"] = "available" if model else "missing_locally_check_wandb"
        status_files = list(Path(row["source"]).rglob("run_status.json"))
        row["wandb"] = [read(p).get("wandb") for p in status_files]
    write(OUT / "inventory.json", rows)
    missing = [str(r["task_id"]) for r in rows if not r["checkpoint"]]
    (OUT / "missing_core_tasks.txt").write_text(",".join(missing) + "\n", encoding="utf-8")
    print(f"Core models: {20-len(missing)}/20 available. Inventory: {OUT / 'inventory.json'}")
    print("Missing locally (check W&B model artifacts before reproducing): " + ",".join(missing))


def check_graphs():
    from splice.graph_io import graph_fingerprint
    expected = read(ROOT / "scripts/crp_signal_checks.conf")["expected_fingerprints"]
    paths = {"cache": BASE / "waterbirds_train_features.pt",
             "crp": BASE / "graphs/crp_graph.json", "raw_clip": BASE / "graphs/raw_clip_graph.json"}
    for name, path in paths.items():
        if not path.is_file() or graph_fingerprint(path) != expected[name]:
            raise RuntimeError(f"Restore the original frozen {name} artifact: {path}; do not rebuild a different graph.")
    return paths


def prepare():
    inventory()
    paths = check_graphs()
    # Reuse existing manifests; graph preparation otherwise rewrites the semantic graph.
    for kind, cfg_path in CONFIGS.items():
        cfg = read(cfg_path)
        if not (ROOT / cfg["output"] / "prepared.json").exists():
            module = "run_concept_transfer" if kind == "direct" else "run_graph_ablation"
            run("scripts.tools." + module, "--config", cfg_path, "--stage", "prepare")
    run("scripts.tools.run_concept_transfer", "--config", CONFIGS["direct"], "--stage", "smoke")
    run("scripts.tools.run_next_actions_smoke", "--cache", paths["cache"],
        "--targets", ROOT / read(CONFIGS["direct"])["output"] / "targets_v1.pt",
        "--graph", paths["crp"], "--graph",
        ROOT / read(CONFIGS["graph"])["output"] / "graphs/semantic_splice_graph.json",
        "--output", OUT / "preflight.json")


def meeting(kind, task):
    cfg_path = CONFIGS[kind]
    cfg = read(cfg_path)
    seed_index, arm_index = divmod(task, len(cfg["arms"]))
    seed, arm = cfg["seeds"][seed_index], cfg["arms"][arm_index]["name"]
    root = ROOT / cfg["output"] / f"seed{seed}" / arm
    status = root / "completed.json"
    results = list(root.glob("training/*/probe_features_epoch_500_*_val.json"))
    if status.exists() and read(status).get("status") == "complete" and len(results) == 1:
        if not read(results[0]).get("convergence", {}).get("converged"):
            raise RuntimeError(f"Non-converged saved probe: {results[0]}")
        saved = read(root / "command.json")["command"]
        for flag, expected in [("--epochs", "500"), ("--temp", "0.05"), ("--seed", str(seed))]:
            if flag not in saved or saved[saved.index(flag)+1] != expected:
                raise RuntimeError(f"Saved command mismatch for {root}: {flag}")
        print(f"Reuse completed {kind}: seed{seed}/{arm}")
        return
    # Never overlap an existing/incomplete run or silently replace its outputs.
    if list(root.glob("training/*/args.json")):
        raise RuntimeError(f"Incomplete existing run at {root}. Check squeue/W&B; recover or move this run before retrying.")
    if not (OUT / "preflight.json").is_file():
        raise RuntimeError("Run stage 01 preparation first.")
    module = "run_concept_transfer" if kind == "direct" else "run_graph_ablation"
    run("scripts.tools." + module, "--config", cfg_path, "--stage", "run", "--array-task-id", task)


def core(task):
    from scripts.tools.run_crp_controls import training_command
    row = matrix()[task]
    seed, arm = row["seed"], row["arm"]
    target = OUT / "core" / f"seed{seed}" / arm
    existing = checkpoint(row["source"]) or checkpoint(target)
    if existing:
        print(f"Reuse {existing}")
        return
    if not (OUT / "inventory.json").is_file():
        raise RuntimeError("Run inventory before selecting missing core task IDs.")
    paths = check_graphs()
    cfg = read(ROOT / "scripts/run_crp_controls_cluster.conf")
    cfg.update(relational_weight=0.5, delete_checkpoints_after_training=False,
               wandb_group="paper_completion_core_reproduction_2026-09-08")
    cfg["data_folder"] = os.environ.get("DATA_FOLDER", cfg["data_folder"])
    graph = None if arm == "simclr" else paths["raw_clip" if arm.startswith("raw_clip") else "crp"]
    command = training_command(cfg, seed, arm, graph, target)
    command += ["--save_freq", "50", "--checkpoint_keep_count", "2",
                "--delete_epoch_checkpoints_after_training", "true"]
    if list(target.glob("training/*/args.json")):
        raise RuntimeError(f"Incomplete reproduction at {target}; recover/move it before retrying.")
    write(target / "command.json", {"command": command, "source": row["source"],
          "execution": "fresh reproduction, not the historical execution", "config": cfg})
    subprocess.run(command, cwd=ROOT, check=True)
    if not checkpoint(target):
        raise RuntimeError("Training returned without a retained epoch-500 model.")


def lock():
    inventory()
    rows = read(OUT / "inventory.json")
    if any(not row["checkpoint"] for row in rows):
        raise RuntimeError("Recover/reproduce all 20 core models before locking final test.")
    paths = check_graphs()
    for row in rows:
        row["sha256"] = digest(row["checkpoint"])
        args_path = Path(row["checkpoint"]).parent / "args.json"
        row["args"] = read(args_path)
        args = row["args"]
        expected = {"seed": row["seed"], "epochs": 500, "dataset": "waterbirds",
                    "model": "resnet18_large", "train_set_linear_layer": "ds_train",
                    "linear_probe_solver": "logistic", "linear_probe_l2": 0.001,
                    "temp": 0.05, "batch_size": 128, "learning_rate": 0.01,
                    "optimizer": "SGD", "ssl_crop_min": 0.2,
                    "splice_mode": "none" if row["arm"] == "simclr" else "crp_relational",
                    "splice_weight": 0.5 if row["arm"].endswith("_kl") else 0}
        if row["arm"] != "simclr":
            from splice.graph_io import graph_fingerprint
            graph_name = "raw_clip" if row["arm"].startswith("raw_clip") else "crp"
            expected.update(crp_graph_fingerprint=graph_fingerprint(paths[graph_name]),
                            crp_temperature=0.25, crp_start_epoch=10, crp_warmup_epochs=10)
        for key, value in expected.items():
            if args.get(key) != value:
                raise RuntimeError(f"Core protocol mismatch: task {row['task_id']} {key}")
        row["args_sha256"] = digest(args_path)
    manifest = {"rows": rows, "graphs_sha256": {k: digest(p) for k, p in paths.items()},
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "probe": {"split": "test", "train": "ds_train", "l2": 0.001, "tolerance": 1e-6,
                          "max_epochs": 200, "ssl_epoch": 500, "batch_size": 128, "workers": 4},
                "rule": "Report all 20 endpoints, group metrics, mean/SD and paired deltas; no test-driven tuning.",
                "meeting_branch": "Excluded from the five-arm ordinary-CoSpRo held-out matrix."}
    sources = [ROOT / "spur_splice.py", Path(__file__)]
    sources += sorted((ROOT / "experiments/spurious_eval").rglob("*.py"))
    sources += sorted((ROOT / "splice").glob("*.py"))
    manifest["source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    path = OUT / "final_test_lock.json"
    if path.exists():
        if read(path) != manifest:
            raise RuntimeError("Final-test lock already exists and differs; do not overwrite after test.")
    else:
        write(path, manifest)
    print(path)


def test(task):
    manifest = read(OUT / "final_test_lock.json")
    for path, expected in manifest["source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Code changed after final-test lock: {path}")
    row = manifest["rows"][task]
    if digest(row["checkpoint"]) != row["sha256"]:
        raise RuntimeError("Locked checkpoint changed.")
    if digest(Path(row["checkpoint"]).parent / "args.json") != row["args_sha256"]:
        raise RuntimeError("Locked checkpoint arguments changed.")
    dest = OUT / "test" / f"seed{row['seed']}" / row["arm"]
    result = dest / "probe_features_epoch_500_ds_train_test.json"
    if result.exists():
        if not read(result).get("convergence", {}).get("converged"):
            raise RuntimeError(f"Saved test probe did not converge: {result}")
        print(f"Reuse {result}")
        return
    run("experiments.spurious_eval.linear_probe", "--ckpt", row["checkpoint"],
        "--artifact_dir", dest, "--dataset", "waterbirds", "--model", "resnet18_large",
        "--data_folder", os.environ.get("DATA_FOLDER", row["args"]["data_folder"]),
        "--seed", row["seed"], "--ssl_epoch", 500, "--train_set_linear_layer", "ds_train",
        "--eval_split", "test", "--final_test", "--probe_solver", "logistic",
        "--probe_l2", 0.001, "--probe_tolerance", 1e-6, "--probe_max_epochs", 200,
        "--batch_size", 128, "--num_workers", 4)
    if not result.exists() or not read(result).get("convergence", {}).get("converged"):
        raise RuntimeError(f"Missing/non-converged test output: {result}")


def test_summary():
    rows = []
    for row in read(OUT / "final_test_lock.json")["rows"]:
        path = OUT / "test" / f"seed{row['seed']}" / row["arm"] / "probe_features_epoch_500_ds_train_test.json"
        result = read(path)
        if result.get("eval_split") != "test" or not result.get("convergence", {}).get("converged"):
            raise RuntimeError(f"Invalid test result: {path}")
        # Existing probe JSON uses legacy 'val' metric keys even for eval_split=test.
        metrics = result["metrics"]
        rows.append({"seed": row["seed"], "arm": row["arm"],
                     "avg": metrics["Average over last 10 linear val acc"],
                     "wga": metrics["Average over last 10 linear val worst-group acc"],
                     "groups": result["group_metrics"]["val"], "result": str(path)})
    write(OUT / "test_results.json", rows)
    with (OUT / "test_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "arm", "avg", "wga", "result"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        summary[arm] = {metric: {"mean": statistics.mean(r[metric] for r in selected),
                               "sd": statistics.stdev(r[metric] for r in selected)}
                        for metric in ["avg", "wga"]}
    paired = []
    for seed in [1, 2, 3, 4]:
        by_arm = {r["arm"]: r for r in rows if r["seed"] == seed}
        for comparator in ARMS[:-1]:
            paired.append({"seed": seed, "comparator": comparator,
                           **{m: by_arm["splice_crp_kl"][m] - by_arm[comparator][m] for m in ["avg", "wga"]}})
    write(OUT / "test_summary.json", {"summary": summary, "paired_deltas": paired})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["prepare", "inventory", "direct", "graph", "core", "lock", "test", "summary", "test_summary"])
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    limit = 8 if args.stage == "direct" else 4 if args.stage == "graph" else 20
    if not 0 <= args.task < limit:
        parser.error(f"Task ID must be in 0..{limit-1}")
    os.chdir(ROOT)
    if args.stage in {"direct", "graph"}:
        meeting(args.stage, args.task)
    elif args.stage == "core":
        core(args.task)
    elif args.stage == "test":
        test(args.task)
    elif args.stage == "summary":
        for kind, cfg in CONFIGS.items():
            module = "run_concept_transfer" if kind == "direct" else "run_graph_ablation"
            run("scripts.tools." + module, "--config", cfg, "--stage", "summary")
    else:
        globals()[args.stage]()


if __name__ == "__main__":
    main()
