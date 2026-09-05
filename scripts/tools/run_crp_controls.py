"""One sequential, resumable current-CRP control experiment; no legacy training."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from splice.crp import CrpAuditConfig, run_frozen_audit, validate_feature_cache
from splice.crp_training import validate_teacher_graph
from splice.graph_io import load_graph_json, save_graph_json, graph_fingerprint
from scripts.tools.build_crp_baseline_graphs import build_matched_raw_clip_graph

ARMS = ("simclr", "crp_sampler_only", "raw_clip_kl", "splice_crp_kl")


def run_command(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        if process.wait():
            raise RuntimeError(f"Command failed; inspect {log}")


def training_command(config, seed, arm, graph, output):
    values = {
        "dataset": config["dataset"], "data_folder": config["data_folder"],
        "model": config["model"], "epochs": config["epochs"], "seed": seed,
        "batch_size": config["batch_size"], "num_workers": config["num_workers"],
        "learning_rate": config["learning_rate"], "optimizer": "SGD",
        "temp": 0.05, "weight_decay": 1e-4, "momentum": 0.9,
        "ssl_crop_min": 0.2, "splice_mode": "none" if arm == "simclr" else "crp_relational",
        "splice_weight": 0 if arm in {"simclr", "crp_sampler_only"} else config["relational_weight"],
        "crp_temperature": 0.25, "crp_start_epoch": 10, "crp_warmup_epochs": 10,
        "crp_decay_start_epoch": 0, "crp_decay_end_epoch": 0,
        "linear_probe_solver": "logistic", "linear_probe_l2": config["probe_l2"],
        "linear_probe_tolerance": config["probe_tolerance"],
        "linear_probe_max_epochs": config["probe_max_epochs"],
        "train_set_linear_layer": config["train_set_linear_layer"],
        "linear_eval_split": "val", "linear_probe_mode": "final",
        "linear_spurious_probe": "true", "checkpoint_dir": str(output / "training"),
        "delete_checkpoints_after_training": "false",
        "wandb_name": config["wandb_project"], "entity": config["wandb_entity"],
        "wandb_run_name": f"{config['wandb_group']}_{arm}_seed{seed}",
        "wandb_group": config["wandb_group"],
        "wandb_tags": f"controls,current_crp,logistic,seed{seed},{arm}",
    }
    if graph is not None:
        values["crp_teacher_graph"] = str(graph)
    command = [sys.executable, "-u", "spur_splice.py", "--use_wandb", "--keep_checkpoints"]
    for key, value in values.items():
        command.extend([f"--{key}", str(value)])
    return command


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Provide unique seeds.")
    if config["train_set_linear_layer"] not in {"train", "ds_train"}:
        raise ValueError("Controls support train or ds_train downstream splits.")
    if config["graph"].get("cobalt") or config["graph"].get("spatial_balance"):
        raise ValueError("The first control experiment excludes legacy CoBalT and spatial branches.")
    if not config["wandb_project"] or not config["wandb_entity"] or not config["wandb_group"]:
        raise ValueError("W&B project, entity and group must be configured.")
    CrpAuditConfig(**config["graph"])
    if args.validate_only:
        for seed in config["seeds"]:
            for arm in ARMS:
                print(f"seed={seed}: {arm}")
        print(f"Output: {config['output']}; probe split: {config['train_set_linear_layer']}; W&B: enabled")
        return

    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "experiment.json"
    resolved = {"protocol": "current_crp_controls_logistic_v1", **config}
    if manifest.exists() and json.loads(manifest.read_text()) != resolved:
        raise ValueError("Output belongs to another configuration. Choose a new output directory.")
    manifest.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    cache_path = Path(config["cache"])
    if not cache_path.exists():
        run_command([sys.executable, "-u", "-m", "scripts.tools.cache_crp_features",
                     "--dataset", config["dataset"], "--data-folder", config["data_folder"],
                     "--output", str(cache_path)], output / "cache.log")
    # Never silently strip unsupported/annotation fields from a historical cache.
    cache = validate_feature_cache(torch.load(cache_path, map_location="cpu", weights_only=True))
    cache_identity = {"path": str(cache_path.resolve()), "content_id": graph_fingerprint(cache_path)}
    cache_record = output / "cache_identity.json"
    if cache_record.exists() and json.loads(cache_record.read_text()) != cache_identity:
        raise ValueError("Frozen cache changed. Choose a new output directory.")
    cache_record.write_text(json.dumps(cache_identity, indent=2), encoding="utf-8")
    if cache.get("provenance", {}).get("dataset") != config["dataset"]:
        raise ValueError("Cache dataset differs from experiment dataset.")
    rows = []
    for seed in config["seeds"]:
        seed_root = output / f"seed{seed}"
        seed_root.mkdir(exist_ok=True)
        graph_root = output / "graphs"
        graph_root.mkdir(exist_ok=True)
        crp_path, raw_path = graph_root / "crp_graph.json", graph_root / "raw_clip_graph.json"
        audit_config = CrpAuditConfig(**{**config["graph"], "seed": config["graph_seed"]})
        if crp_path.exists():
            crp = validate_teacher_graph(load_graph_json(crp_path), cache["sample_ids"])
            if crp["config"] != asdict(audit_config):
                raise ValueError("Saved CRP graph configuration differs from this experiment.")
        else:
            crp = run_frozen_audit(cache, audit_config)
            save_graph_json(crp, crp_path)
        if not (crp["weights"].sum(1) > 0).any():
            raise RuntimeError("CRP graph is empty: the sampler/KL controls would be identical to SimCLR. Inspect graph diagnostics before training.")
        # Cheap deterministic rebuild ensures the raw control matches this graph.
        raw = build_matched_raw_clip_graph(cache, crp)
        save_graph_json(raw, raw_path)
        graph_identity = {"crp": graph_fingerprint(crp_path), "raw_clip": graph_fingerprint(raw_path)}
        graph_record = graph_root / "graph_identity.json"
        if graph_record.exists() and json.loads(graph_record.read_text()) != graph_identity:
            raise ValueError("Prepared graphs changed. Choose a new output directory.")
        graph_record.write_text(json.dumps(graph_identity, indent=2), encoding="utf-8")
        overlap = ((crp["neighbor_indices"][:, :, None] == raw["neighbor_indices"][:, None, :])
                   & (crp["neighbor_indices"][:, :, None] >= 0)).any(2)
        (seed_root / "graph_diagnostics.json").write_text(json.dumps({
            "crp": crp["degree_stats"], "raw_clip": raw["degree_stats"],
            "crp_edge_overlap_with_raw_clip": float(overlap.sum() / (crp["neighbor_indices"] >= 0).sum()),
            "selected_groups": crp["selected_group_ids"],
            "mean_supported_confidence": float(crp["anchor_confidence"][crp["anchor_confidence"] > 0].mean()),
        }, indent=2), encoding="utf-8")
        for arm in ARMS:
            arm_root = seed_root / arm
            arm_root.mkdir(exist_ok=True)
            completed = arm_root / "completed.json"
            if completed.exists():
                row = json.loads(completed.read_text())
            else:
                graph = None if arm == "simclr" else raw_path if arm == "raw_clip_kl" else crp_path
                command = training_command(config, seed, arm, graph, arm_root)
                (arm_root / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
                run_command(command, arm_root / "training.log")
                results = list((arm_root / "training").glob(f"*/probe_features_epoch_{config['epochs']}_*_val.json"))
                if len(results) != 1:
                    raise RuntimeError(f"Expected exactly one final converged probe result under {arm_root}.")
                result = json.loads(results[0].read_text())
                if not result["convergence"].get("converged"):
                    raise RuntimeError("Probe did not converge.")
                metrics = result["metrics"]
                row = {"seed": seed, "arm": arm,
                       "avg_acc_last10": metrics["Average over last 10 linear val acc"],
                       "wga_last10": metrics["Average over last 10 linear val worst-group acc"],
                       "probe_epochs": result["convergence"]["epochs"], "result": str(results[0])}
                completed.write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
    import statistics
    baseline = {r["seed"]: r for r in rows if r["arm"] == "simclr"}
    summary = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        summary[arm] = {key: {"mean": statistics.mean(r[key] for r in selected),
                             "std_across_seeds": statistics.stdev(r[key] for r in selected) if len(selected)>1 else None,
                             "paired_deltas_vs_simclr": [r[key]-baseline[r['seed']][key] for r in selected]}
                        for key in ("avg_acc_last10", "wga_last10")}
    (output / "paired_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Completed control experiment: {output / 'results.csv'}")


if __name__ == "__main__":
    main()
