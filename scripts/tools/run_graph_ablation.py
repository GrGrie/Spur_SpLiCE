"""Prepare, run, and summarize the two-arm CRP/semantic-graph ablation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

from scripts.tools.build_semantic_splice_graph import build_matched_semantic_graph, edge_overlap
from splice.crp import validate_feature_cache
from splice.crp_training import validate_teacher_graph
from splice.graph_io import graph_fingerprint, load_graph_json, save_graph_json


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(path: Path) -> Path:
    cfg = read_config(path)
    output = Path(cfg["output"])
    cache_path = Path(cfg["cache"])
    crp_path = Path(cfg["crp_graph"])
    cache = validate_feature_cache(torch.load(cache_path, map_location="cpu", weights_only=True))
    crp = validate_teacher_graph(load_graph_json(crp_path), cache["sample_ids"])
    graph_dir = output / "graphs"
    semantic_path = graph_dir / "semantic_splice_graph.json"
    semantic = build_matched_semantic_graph(cache, crp)
    save_graph_json(semantic, semantic_path)
    overlap = edge_overlap(crp, semantic)
    manifest = {
        "artifact": "graph_ablation_v1_manifest",
        "config": str(path.resolve()),
        "cache_fingerprint": graph_fingerprint(cache_path),
        "crp_graph": str(crp_path.resolve()),
        "crp_fingerprint": graph_fingerprint(crp_path),
        "semantic_graph": str(semantic_path.resolve()),
        "semantic_fingerprint": graph_fingerprint(semantic_path),
        "seeds": cfg["seeds"],
        "arms": cfg["arms"],
        "epochs": cfg["epochs"],
        "edge_overlap": overlap,
    }
    _write(output / "prepared.json", manifest)
    _write(output / "edge_overlap.json", {"artifact": "matched_graph_edge_overlap_v1", **overlap})
    return output / "prepared.json"


def _prepared(path: Path) -> tuple[dict, dict, Path]:
    cfg = read_config(path)
    output = Path(cfg["output"])
    marker = json.loads((output / "prepared.json").read_text(encoding="utf-8"))
    semantic = Path(marker["semantic_graph"])
    if graph_fingerprint(Path(cfg["cache"])) != marker["cache_fingerprint"]:
        raise RuntimeError("Graph-ablation cache fingerprint changed after preparation.")
    if graph_fingerprint(Path(cfg["crp_graph"])) != marker["crp_fingerprint"]:
        raise RuntimeError("CRP graph fingerprint changed after preparation.")
    if graph_fingerprint(semantic) != marker["semantic_fingerprint"]:
        raise RuntimeError("Semantic graph fingerprint changed after preparation.")
    return cfg, marker, semantic


def resolve_task(cfg: dict, task_id: int) -> tuple[int, dict]:
    arms = cfg["arms"]
    if task_id < 0 or task_id >= len(cfg["seeds"]) * len(arms):
        raise ValueError(f"Array task id {task_id} is outside the configured matrix.")
    seed_index, arm_index = divmod(int(task_id), len(arms))
    return int(cfg["seeds"][seed_index]), arms[arm_index]


def run_one(path: Path, task_id: int) -> Path:
    cfg, marker, semantic_path = _prepared(path)
    seed, arm = resolve_task(cfg, task_id)
    output = Path(cfg["output"])
    arm_root = output / f"seed{seed}" / arm["name"]
    training_root = arm_root / "training"
    arm_root.mkdir(parents=True, exist_ok=True)
    graph = Path(cfg["crp_graph"] if arm["name"] == "crp" else semantic_path)
    command = [
        sys.executable, "-u", "spur_splice.py",
        "--dataset", cfg["dataset"], "--data_folder", cfg["data_folder"],
        "--model", cfg["model"], "--seed", str(seed), "--epochs", str(cfg["epochs"]),
        "--batch_size", str(cfg["batch_size"]), "--num_workers", str(cfg["num_workers"]),
        "--optimizer", "SGD", "--learning_rate", str(cfg["learning_rate"]),
        "--lr_decay_epochs", cfg["lr_decay_epochs"], "--lr_decay_rate", str(cfg["lr_decay_rate"]),
        "--weight_decay", str(cfg["weight_decay"]), "--temp", str(cfg["temperature"]),
        "--simclr_weight", "1.0", "--splice_mode", "crp_relational",
        "--splice_weight", str(cfg["relational_weight"]), "--crp_teacher_graph", str(graph),
        "--crp_temperature", str(cfg["crp_temperature"]), "--crp_start_epoch", str(cfg["start_epoch"]),
        "--crp_warmup_epochs", str(cfg["warmup_epochs"]), "--linear_probe_mode", "periodic",
        "--linear_probe_freq", str(cfg["probe_frequency"]), "--linear_probe_solver", "logistic",
        "--linear_probe_l2", str(cfg["probe_l2"]), "--linear_probe_tolerance", str(cfg["probe_tolerance"]),
        "--linear_probe_max_epochs", str(cfg["probe_max_epochs"]), "--train_set_linear_layer", "ds_train",
        "--linear_eval_split", "val", "--rank_eval_freq", "0", "--checkpoint_dir", str(training_root),
        "--keep_checkpoints", "--save_freq", str(cfg["save_frequency"]),
        "--checkpoint_keep_count", str(cfg["checkpoint_keep_count"]),
        "--delete_checkpoints_after_training", "false",
        "--delete_epoch_checkpoints_after_training", str(cfg.get("delete_epoch_checkpoints_after_training", True)).lower(),
        "--retain_probe_artifacts_every", str(cfg["retain_probe_artifacts_every"]), "--use_wandb",
        "--wandb_name", cfg["wandb_project"], "--entity", cfg["wandb_entity"],
        "--wandb_group", cfg["wandb_group"], "--wandb_run_name",
        f"graph_ablation_{arm['name']}_seed{seed}", "--wandb_tags",
        f"graph_ablation,{arm['name']},seed{seed}", "--amp", "true",
        "--channels_last", "true", "--cudnn_enabled", "true", "--gradient_diagnostics",
        "--gradient_diagnostics_batches", str(cfg["gradient_diagnostic_batches"]),
        "--gradient_diagnostics_epochs", ",".join(str(value) for value in cfg["gradient_diagnostic_epochs"]),
        "--gradient_diagnostics_output", str(arm_root / "gradient_diagnostics.json"),
    ]
    _write(arm_root / "command.json", {"command": command, "arm": arm, "seed": seed})
    status_path = arm_root / "completed.json"
    try:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=True)
        results = sorted(training_root.glob(f"*/probe_features_epoch_{cfg['epochs']}_*_val.json"))
        if len(results) != 1:
            raise RuntimeError(f"Expected one final probe under {arm_root}, found {len(results)}")
        diagnostic_path = arm_root / "gradient_diagnostics.json"
        if not diagnostic_path.is_file():
            raise RuntimeError(f"Missing gradient diagnostics: {diagnostic_path}")
        diagnostic_payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        actual = {
            (int(item["epoch"]), int(item["batch"]))
            for item in diagnostic_payload.get("gradient_diagnostics", [])
        }
        expected = {
            (int(epoch), batch)
            for epoch in cfg["gradient_diagnostic_epochs"]
            for batch in range(int(cfg["gradient_diagnostic_batches"]))
        }
        if not expected.issubset(actual):
            raise RuntimeError(f"Incomplete graph gradient diagnostics: missing={sorted(expected - actual)}")
        result = json.loads(results[0].read_text(encoding="utf-8"))
        if not result.get("convergence", {}).get("converged"):
            raise RuntimeError("Final graph-ablation probe did not converge.")
        metrics = result["metrics"]
        payload = {
            "status": "complete", "seed": seed, "arm": arm["name"], "graph": str(graph),
            "result": str(results[0]),
            "avg_acc_last10": metrics["Average over last 10 linear val acc"],
            "wga_last10": metrics["Average over last 10 linear val worst-group acc"],
            "best_group_last10": metrics["Average over last 10 linear val best-group acc"],
            "gradient_diagnostics": str(arm_root / "gradient_diagnostics.json"),
        }
    except Exception as exc:
        payload = {"status": "failed", "seed": seed, "arm": arm["name"], "error": repr(exc)}
        _write(status_path, payload)
        raise
    _write(status_path, payload)
    return status_path


def summarize(path: Path) -> Path:
    cfg, _, _ = _prepared(path)
    output = Path(cfg["output"])
    rows, failures = [], []
    for seed in cfg["seeds"]:
        for arm in cfg["arms"]:
            status_path = output / f"seed{seed}" / arm["name"] / "completed.json"
            if not status_path.is_file():
                failures.append(f"seed{seed}/{arm['name']}: missing completed.json")
                continue
            record = json.loads(status_path.read_text(encoding="utf-8"))
            if record.get("status") != "complete":
                failures.append(f"seed{seed}/{arm['name']}: {record.get('error', 'failed')}")
                continue
            rows.append(record)
    paired = {}
    for seed in cfg["seeds"]:
        by_arm = {row["arm"]: row for row in rows if row["seed"] == seed}
        if set(by_arm) == {arm["name"] for arm in cfg["arms"]}:
            paired[str(seed)] = {
                metric: by_arm["crp"][metric] - by_arm["semantic_splice"][metric]
                for metric in ("avg_acc_last10", "wga_last10", "best_group_last10")
            }
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "arm", "result", "avg_acc_last10", "wga_last10", "best_group_last10"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    summary = {"artifact": "graph_ablation_v1_summary", "rows": rows, "failures": failures,
               "passed": not failures and len(rows) == len(cfg["seeds"]) * len(cfg["arms"]),
               "comparison": "CRP versus semantic SpLiCE graph; report both seeds; no automatic selection",
               "paired_deltas_crp_minus_semantic": paired}
    _write(output / "summary.json", summary)
    _write(output / "paired_deltas_crp_minus_semantic.json", {"paired": paired})
    if failures:
        raise RuntimeError(f"Graph-ablation summary failed; see {output / 'summary.json'}")
    return output / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=("prepare", "run", "summary"), required=True)
    parser.add_argument("--array-task-id", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.config)
    elif args.stage == "run":
        if args.array_task_id is None:
            raise SystemExit("--array-task-id is required for --stage run")
        run_one(args.config, args.array_task_id)
    else:
        summarize(args.config)


if __name__ == "__main__":
    main()
