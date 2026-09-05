"""One sequential, resumable current-CRP control experiment; no legacy training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from splice.crp import CrpAuditConfig, run_frozen_audit, validate_feature_cache
from splice.crp_training import validate_teacher_graph
from splice.graph_io import load_graph_json, save_graph_json, graph_fingerprint
from scripts.tools.build_crp_baseline_graphs import build_matched_raw_clip_graph
from scripts.tools.crp_posthoc_diagnostics import diagnose_fixed_graphs
from splice.crp_safe_graph import (
    SafeCrpGraphConfig,
    build_safe_crp_graph,
    safe_training_gate,
    validate_safe_crp_graph,
)

LEGACY_ARMS = ("simclr", "crp_sampler_only", "raw_clip_kl", "splice_crp_kl")
# Backward-compatible import used by existing tests and small local tools.
ARMS = LEGACY_ARMS
SUPPORTED_ARMS = LEGACY_ARMS + (
    "raw_clip_sampler_only",
    "safe_crp_sampler_only",
    "safe_crp_kl",
)


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
        "splice_weight": 0 if arm in {"simclr", "crp_sampler_only", "raw_clip_sampler_only", "safe_crp_sampler_only"} else config["relational_weight"],
        "crp_temperature": 0.25, "crp_start_epoch": 10, "crp_warmup_epochs": 10,
        "crp_decay_start_epoch": 0, "crp_decay_end_epoch": 0,
        "linear_probe_solver": "logistic", "linear_probe_l2": config["probe_l2"],
        "linear_probe_tolerance": config["probe_tolerance"],
        "linear_probe_max_epochs": config["probe_max_epochs"],
        "train_set_linear_layer": config["train_set_linear_layer"],
        "linear_eval_split": "val",
        "linear_spurious_probe": "true", "checkpoint_dir": str(output / "training"),
        "delete_checkpoints_after_training": "false",
        "wandb_name": config["wandb_project"], "entity": config["wandb_entity"],
        "wandb_run_name": f"{config['wandb_group']}_{arm}_seed{seed}",
        "wandb_group": config["wandb_group"],
        "wandb_tags": f"controls,current_crp,logistic,seed{seed},{arm}",
    }
    probe_mode = config.get("linear_probe_mode", "final")
    values["linear_probe_mode"] = probe_mode
    if probe_mode == "periodic":
        values["linear_probe_freq"] = config["linear_probe_freq"]
    if graph is not None:
        values["crp_teacher_graph"] = str(graph)
    command = [sys.executable, "-u", "spur_splice.py", "--use_wandb", "--keep_checkpoints"]
    for key, value in values.items():
        command.extend([f"--{key}", str(value)])
    return command


def load_and_validate_config(config_path: Path) -> tuple[dict, tuple[str, ...]]:
    """Load one immutable experiment definition for sequential or cluster use."""

    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    arms = tuple(config.get("arms", LEGACY_ARMS))
    if not arms or len(set(arms)) != len(arms) or any(arm not in SUPPORTED_ARMS for arm in arms):
        raise ValueError(f"arms must be a unique non-empty subset of {SUPPORTED_ARMS}.")
    protocol = config.get("protocol", "current_crp_controls_logistic_v1")
    if protocol not in {"current_crp_controls_logistic_v1", "safe_crp_controls_v1", "crp_controls_cluster_v1"}:
        raise ValueError(f"Unsupported control protocol: {protocol!r}")
    if protocol == "crp_controls_cluster_v1":
        if arms != LEGACY_ARMS:
            raise ValueError("The cluster protocol must contain exactly the original four arms in order.")
        if config.get("epochs") != 500:
            raise ValueError("The cluster protocol is fixed at 500 SSL epochs.")
        if config.get("linear_probe_mode") != "periodic" or config.get("linear_probe_freq") != 25:
            raise ValueError("The cluster protocol requires periodic linear probing every 25 epochs.")
    safe_arms = {arm for arm in arms if arm.startswith("safe_crp_")}
    if safe_arms:
        if "safe_graph" not in config:
            raise ValueError("safe arms require a safe_graph configuration.")
        SafeCrpGraphConfig.from_mapping(config["safe_graph"])
    if not config.get("seeds") or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Provide unique seeds.")
    if config["train_set_linear_layer"] not in {"train", "ds_train"}:
        raise ValueError("Controls support train or ds_train downstream splits.")
    if config["graph"].get("cobalt") or config["graph"].get("spatial_balance"):
        raise ValueError("The control protocols exclude legacy CoBalT and spatial branches.")
    if not config["wandb_project"] or not config["wandb_entity"] or not config["wandb_group"]:
        raise ValueError("W&B project, entity and group must be configured.")
    CrpAuditConfig(**config["graph"])
    return config, arms


def resolve_array_task(task_id: int, seeds_or_config, arms: Sequence[str] | None = None) -> tuple[int, str]:
    """Map a Slurm array element to one row-major ``(seed, arm)`` pair."""

    if isinstance(seeds_or_config, dict):
        config = seeds_or_config
        seeds = config["seeds"]
        arms = config["arms"]
    else:
        seeds = seeds_or_config
    if arms is None:
        raise ValueError("arms are required for array-task mapping")
    task_id = int(task_id)
    if task_id < 0 or task_id >= len(seeds) * len(arms):
        raise ValueError(f"Array task id {task_id} is outside 0..{len(seeds) * len(arms) - 1}.")
    seed_index, arm_index = divmod(task_id, len(arms))
    return int(seeds[seed_index]), str(arms[arm_index])


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _config_fingerprint(config: dict, protocol: str) -> str:
    payload = {"protocol": protocol, **config}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _cluster_graph_paths(output: Path) -> dict[str, Path]:
    graph_root = output / "graphs"
    return {"crp": graph_root / "crp_graph.json", "raw_clip": graph_root / "raw_clip_graph.json"}


def _prepare_cache(config: dict, output: Path) -> tuple[dict, str]:
    cache_path = Path(config["cache"])
    if not cache_path.exists():
        command = [sys.executable, "-u", "-m", "scripts.tools.cache_crp_features",
                   "--dataset", config["dataset"], "--data-folder", config["data_folder"],
                   "--output", str(cache_path), "--device", config.get("prepare_device", "cpu"),
                   "--batch-size", str(config.get("cache_batch_size", 64)),
                   "--num-workers", str(config.get("cache_num_workers", config.get("num_workers", 4)))]
        run_command(command, output / "cache.log")
    cache = validate_feature_cache(torch.load(cache_path, map_location="cpu", weights_only=True))
    if cache.get("provenance", {}).get("dataset") != config["dataset"]:
        raise ValueError("Cache dataset differs from experiment dataset.")
    return cache, graph_fingerprint(cache_path)


def prepare_experiment(config_path: Path) -> dict:
    """Prepare one immutable cache/graph set without launching SSL."""

    config, arms = load_and_validate_config(config_path)
    if config.get("protocol") != "crp_controls_cluster_v1":
        raise ValueError("--prepare-only is reserved for crp_controls_cluster_v1.")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    protocol = config["protocol"]
    resolved = {"protocol": protocol, **config}
    manifest = output / "experiment.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != resolved:
        raise ValueError("Output belongs to another configuration. Choose a new output directory.")
    _atomic_write_json(manifest, resolved)

    cache, cache_fingerprint = _prepare_cache(config, output)
    cache_identity = {"path": str(Path(config["cache"]).resolve()), "content_id": cache_fingerprint}
    cache_record = output / "cache_identity.json"
    if cache_record.exists() and json.loads(cache_record.read_text(encoding="utf-8")) != cache_identity:
        raise ValueError("Frozen cache changed. Choose a new output directory.")
    _atomic_write_json(cache_record, cache_identity)

    paths = _cluster_graph_paths(output)
    paths["crp"].parent.mkdir(parents=True, exist_ok=True)
    audit_config = CrpAuditConfig(**{**config["graph"], "seed": config["graph_seed"]})
    if paths["crp"].exists():
        crp = validate_teacher_graph(load_graph_json(paths["crp"]), cache["sample_ids"])
        if crp["config"] != asdict(audit_config):
            raise ValueError("Saved CRP graph configuration differs from this experiment.")
    else:
        crp = run_frozen_audit(cache, audit_config)
        save_graph_json(crp, paths["crp"])
    if not (crp["weights"].sum(1) > 0).any():
        raise RuntimeError("CRP graph is empty; the intended controls would be indistinguishable from SimCLR.")
    if paths["raw_clip"].exists():
        raw = validate_teacher_graph(load_graph_json(paths["raw_clip"]), cache["sample_ids"])
    else:
        raw = build_matched_raw_clip_graph(cache, crp)
        save_graph_json(raw, paths["raw_clip"])
    graph_fingerprints = {name: graph_fingerprint(path) for name, path in paths.items()}
    graph_record = output / "graph_identity.json"
    if graph_record.exists() and json.loads(graph_record.read_text(encoding="utf-8")) != graph_fingerprints:
        raise ValueError("Prepared graphs changed. Choose a new output directory.")
    _atomic_write_json(graph_record, graph_fingerprints)
    prepared = {
        "artifact": "crp_controls_cluster_prepared_v1",
        "protocol": protocol,
        "config_fingerprint": _config_fingerprint(config, protocol),
        "cache_fingerprint": cache_fingerprint,
        "graph_fingerprints": graph_fingerprints,
        "seeds": list(config["seeds"]),
        "arms": list(arms),
    }
    marker = output / "prepared.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != prepared:
        raise ValueError("Prepared marker differs; choose a new output directory.")
    _atomic_write_json(marker, prepared)
    print(f"Prepared immutable CRP controls at {output}")
    return prepared


def _load_prepared(config_path: Path) -> tuple[dict, tuple[str, ...], Path, dict]:
    config, arms = load_and_validate_config(config_path)
    if config.get("protocol") != "crp_controls_cluster_v1":
        raise ValueError("Cluster modes require protocol=crp_controls_cluster_v1.")
    output = Path(config["output"])
    marker = output / "prepared.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Prepared marker not found: {marker}; run --prepare-only first.")
    prepared = json.loads(marker.read_text(encoding="utf-8"))
    expected_config = _config_fingerprint(config, config["protocol"])
    if prepared.get("config_fingerprint") != expected_config:
        raise ValueError("Current configuration does not match prepared.json.")
    cache_path = Path(config["cache"])
    if not cache_path.is_file() or graph_fingerprint(cache_path) != prepared.get("cache_fingerprint"):
        raise ValueError("Frozen cache does not match prepared.json.")
    paths = _cluster_graph_paths(output)
    actual_graphs = {name: graph_fingerprint(path) for name, path in paths.items() if path.is_file()}
    if actual_graphs != prepared.get("graph_fingerprints"):
        raise ValueError("Prepared graph fingerprints do not match current files.")
    return config, arms, output, prepared


def _graph_for_arm(arm: str, output: Path) -> Path | None:
    if arm == "simclr":
        return None
    return _cluster_graph_paths(output)["raw_clip" if arm == "raw_clip_kl" else "crp"]


def _run_identity(config: dict, prepared: dict, seed: int, arm: str) -> dict:
    return {
        "config_fingerprint": prepared["config_fingerprint"],
        "cache_fingerprint": prepared["cache_fingerprint"],
        "graph_fingerprints": prepared["graph_fingerprints"],
        "seed": seed,
        "arm": arm,
    }


def _validate_completed(path: Path, identity: dict) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "complete" or record.get("run_identity") != identity:
        raise ValueError(f"Completed record does not match this experiment: {path}")
    return record


def run_one_arm(config_path: Path, task_id: int) -> dict:
    """Run exactly one independently retryable cluster array element."""

    config, arms, output, prepared = _load_prepared(config_path)
    seed, arm = resolve_array_task(task_id, config["seeds"], arms)
    arm_root = output / f"seed{seed}" / arm
    arm_root.mkdir(parents=True, exist_ok=True)
    identity = _run_identity(config, prepared, seed, arm)
    completed = arm_root / "completed.json"
    if completed.exists():
        record = _validate_completed(completed, identity)
        print(f"Reusing completed seed={seed}, arm={arm}")
        return record
    graph = _graph_for_arm(arm, output)
    command = training_command(config, seed, arm, graph, arm_root)
    _atomic_write_json(arm_root / "command.json", {"command": command, "run_identity": identity})
    run_command(command, arm_root / "training.log")
    results = list((arm_root / "training").glob(f"*/probe_features_epoch_{config['epochs']}_*_val.json"))
    if len(results) != 1:
        raise RuntimeError(f"Expected exactly one final converged probe result under {arm_root}.")
    result = json.loads(results[0].read_text(encoding="utf-8"))
    if not result.get("convergence", {}).get("converged"):
        raise RuntimeError("Final linear probe did not converge.")
    metrics = result["metrics"]
    record = {
        "status": "complete", "seed": seed, "arm": arm,
        "avg_acc_last10": metrics["Average over last 10 linear val acc"],
        "wga_last10": metrics["Average over last 10 linear val worst-group acc"],
        "best_group_last10": metrics["Average over last 10 linear val best-group acc"],
        "probe_epochs": result["convergence"]["epochs"], "result": str(results[0]),
        "run_identity": identity,
    }
    _atomic_write_json(completed, record)
    return record


def _probe_group_rows(record: dict, config: dict) -> list[dict]:
    result_path = Path(record["result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    group_metrics = result.get("group_metrics", {}).get("val", {})
    accuracies = group_metrics.get("accuracy", result.get("metrics", {}).get("Linear val group accuracies", []))
    counts = group_metrics.get("count", result.get("metrics", {}).get("Linear val group counts", []))
    if len(accuracies) != len(counts):
        raise ValueError(f"Probe group metrics are misaligned: {result_path}")
    return [{"seed": record["seed"], "arm": record["arm"], "ssl_epoch": config["epochs"],
             "group_id": index, "group_count": int(count), "group_accuracy": float(accuracy)}
            for index, (accuracy, count) in enumerate(zip(accuracies, counts))]


def summarize_experiment(config_path: Path) -> Path:
    """Aggregate only completed, identity-matched array arms on the CPU."""

    config, arms, output, prepared = _load_prepared(config_path)
    rows_by_seed: dict[int, dict[str, dict]] = {}
    for seed in config["seeds"]:
        rows_by_seed[seed] = {}
        for arm in arms:
            completed = output / f"seed{seed}" / arm / "completed.json"
            if completed.exists():
                rows_by_seed[seed][arm] = _validate_completed(completed, _run_identity(config, prepared, seed, arm))
    paired_seeds = [seed for seed in config["seeds"] if set(rows_by_seed[seed]) == set(arms)]
    rows = [rows_by_seed[seed][arm] for seed in paired_seeds for arm in arms]
    group_rows = [group for record in rows for group in _probe_group_rows(record, config)]
    summary_rows = [{"seed": row["seed"], "arm": row["arm"], "ssl_epoch": config["epochs"],
                     "avg_acc_last10": row["avg_acc_last10"], "wga_last10": row["wga_last10"],
                     "best_group_last10": row["best_group_last10"], "probe_epochs": row["probe_epochs"],
                     "result": row["result"]} for row in rows]
    summary = {"paired_seeds": paired_seeds}
    baseline = {(row["seed"], row["arm"]): row for row in summary_rows}
    import statistics
    for arm in arms:
        selected = [row for row in summary_rows if row["arm"] == arm]
        summary[arm] = {}
        for metric in ("avg_acc_last10", "wga_last10", "best_group_last10"):
            values = [row[metric] for row in selected]
            deltas = [row[metric] - baseline[(row["seed"], "simclr")][metric]
                      for row in selected if (row["seed"], "simclr") in baseline]
            summary[arm][metric] = {
                "mean": statistics.mean(values) if values else None,
                "std_across_seeds": statistics.stdev(values) if len(values) > 1 else None,
                "paired_deltas_vs_simclr": deltas,
            }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "arm", "ssl_epoch", "avg_acc_last10", "wga_last10", "best_group_last10", "probe_epochs", "result"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary_rows)
    with (output / "group_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "arm", "ssl_epoch", "group_id", "group_count", "group_accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(group_rows)
    _atomic_write_json(output / "paired_summary.json", summary)
    _atomic_write_json(output / "summary.json", {"paired_seeds": paired_seeds,
                                                   "missing": {str(seed): [arm for arm in arms if arm not in rows_by_seed[seed]]
                                                               for seed in config["seeds"] if seed not in paired_seeds}})
    print(f"Summarized paired seeds {paired_seeds} at {output}")
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--array-task-id", type=int)
    modes.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only or args.array_task_id is not None or args.summarize_only:
        if args.validate_only:
            raise ValueError("--validate-only cannot be combined with a cluster execution mode.")
        if args.prepare_only:
            prepare_experiment(args.config)
        elif args.array_task_id is not None:
            run_one_arm(args.config, args.array_task_id)
        else:
            summarize_experiment(args.config)
        return
    # Keep validation useful for the cluster manifest without touching any
    # cache, graph, or training artifact.
    if args.validate_only:
        config, arms = load_and_validate_config(args.config)
        protocol = config.get("protocol", "current_crp_controls_logistic_v1")
        for seed in config["seeds"]:
            for arm in arms:
                print(f"seed={seed}: {arm}")
        print(f"Protocol: {protocol}; output: {config['output']}; probe split: {config['train_set_linear_layer']}; W&B: enabled")
        return
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    arms = tuple(config.get("arms", LEGACY_ARMS))
    if not arms or len(set(arms)) != len(arms) or any(arm not in SUPPORTED_ARMS for arm in arms):
        raise ValueError(f"arms must be a unique non-empty subset of {SUPPORTED_ARMS}.")
    explicit_protocol = "arms" in config
    protocol = config.get("protocol", "current_crp_controls_logistic_v1")
    if explicit_protocol and protocol not in {"safe_crp_controls_v1", "current_crp_controls_logistic_v1"}:
        raise ValueError(f"Unsupported control protocol: {protocol!r}")
    safe_arms = {arm for arm in arms if arm.startswith("safe_crp_")}
    if safe_arms and "safe_graph" not in config:
        raise ValueError("safe arms require a safe_graph configuration.")
    safe_config = SafeCrpGraphConfig.from_mapping(config.get("safe_graph")) if safe_arms else None
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
            for arm in arms:
                print(f"seed={seed}: {arm}")
        print(f"Protocol: {protocol}; output: {config['output']}; probe split: {config['train_set_linear_layer']}; W&B: enabled")
        if safe_config:
            print(f"Safe graph: {json.dumps(asdict(safe_config), sort_keys=True)}")
        return

    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "experiment.json"
    resolved = {"protocol": protocol, **config}
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
    group_rows = []
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
        safe_path = graph_root / "safe_crp_graph.json"
        safe = None
        if safe_config:
            safe = build_safe_crp_graph(
                cache,
                crp,
                raw,
                safe_config,
                source_crp_fingerprint=graph_fingerprint(crp_path),
                source_raw_fingerprint=graph_fingerprint(raw_path),
            )
            save_graph_json(safe, safe_path)
            safe = validate_safe_crp_graph(load_graph_json(safe_path), cache["sample_ids"])
        graph_identity = {"crp": graph_fingerprint(crp_path), "raw_clip": graph_fingerprint(raw_path)}
        if safe is not None:
            graph_identity["safe_crp"] = graph_fingerprint(safe_path)
        graph_record = graph_root / "graph_identity.json"
        if graph_record.exists() and json.loads(graph_record.read_text()) != graph_identity:
            raise ValueError("Prepared graphs changed. Choose a new output directory.")
        graph_record.write_text(json.dumps(graph_identity, indent=2), encoding="utf-8")
        if seed == config["seeds"][0]:
            posthoc_graphs = {"crp": crp, "raw_clip": raw}
            if safe is not None:
                posthoc_graphs["safe_crp"] = safe
            posthoc = diagnose_fixed_graphs(posthoc_graphs,
                                           config["dataset"], config["data_folder"])
            (graph_root / "posthoc_group_diagnostics.json").write_text(
                json.dumps(posthoc, indent=2), encoding="utf-8")
        overlap = ((crp["neighbor_indices"][:, :, None] == raw["neighbor_indices"][:, None, :])
                   & (crp["neighbor_indices"][:, :, None] >= 0)).any(2)
        (seed_root / "graph_diagnostics.json").write_text(json.dumps({
            "crp": crp["degree_stats"], "raw_clip": raw["degree_stats"],
            "crp_edge_overlap_with_raw_clip": float(overlap.sum() / (crp["neighbor_indices"] >= 0).sum()),
            "selected_groups": crp["selected_group_ids"],
            "mean_supported_confidence": float(crp["anchor_confidence"][crp["anchor_confidence"] > 0].mean()),
        }, indent=2), encoding="utf-8")
        safe_allowed, safe_gate_reason = (safe_training_gate(safe) if safe is not None else (True, None))
        for arm in arms:
            arm_root = seed_root / arm
            arm_root.mkdir(exist_ok=True)
            completed = arm_root / "completed.json"
            if completed.exists():
                row = json.loads(completed.read_text())
                if row.get("avg_acc_last10") is not None and "best_group_last10" not in row:
                    result_path = Path(str(row.get("result", "")))
                    if result_path.is_file():
                        result_metrics = json.loads(result_path.read_text(encoding="utf-8"))["metrics"]
                        row["best_group_last10"] = result_metrics["Average over last 10 linear val best-group acc"]
            else:
                if arm == "simclr":
                    graph = None
                elif arm.startswith("raw_clip_"):
                    graph = raw_path
                elif arm.startswith("safe_crp_"):
                    graph = safe_path
                else:
                    graph = crp_path
                if arm.startswith("safe_crp_") and not safe_allowed:
                    row = {"seed": seed, "arm": arm, "avg_acc_last10": None,
                           "wga_last10": None, "best_group_last10": None,
                           "probe_epochs": 0, "result": "FAIL_NO_SAFE_TREATMENT",
                           "failure_reason": safe_gate_reason}
                    completed.write_text(json.dumps(row, indent=2), encoding="utf-8")
                    rows.append(row)
                    continue
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
                       "best_group_last10": metrics["Average over last 10 linear val best-group acc"],
                       "probe_epochs": result["convergence"]["epochs"], "result": str(results[0])}
                completed.write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            result_path = Path(str(row.get("result", "")))
            if result_path.is_file():
                result_payload = json.loads(result_path.read_text(encoding="utf-8"))
                group_metrics = result_payload.get("group_metrics", {}).get("val", {})
                accuracies = group_metrics.get("accuracy", result_payload.get("metrics", {}).get("Linear val group accuracies", []))
                counts = group_metrics.get("count", result_payload.get("metrics", {}).get("Linear val group counts", []))
                if len(accuracies) != len(counts):
                    raise ValueError(f"Probe group metrics are misaligned: {result_path}")
                group_rows.extend(
                    {"seed": seed, "arm": arm, "ssl_epoch": config["epochs"], "group_id": group_id,
                     "group_count": int(count), "group_accuracy": float(accuracy)}
                    for group_id, (accuracy, count) in enumerate(zip(accuracies, counts))
                )
            with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                fieldnames = ["seed", "arm", "avg_acc_last10", "wga_last10", "best_group_last10", "probe_epochs", "result", "failure_reason"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
    import statistics
    baseline = {r["seed"]: r for r in rows if r["arm"] == "simclr" and r.get("avg_acc_last10") is not None}
    summary = {}
    for arm in arms:
        selected = [r for r in rows if r["arm"] == arm and r.get("avg_acc_last10") is not None]
        if not selected:
            summary[arm] = {key: {"mean": None, "std_across_seeds": None, "paired_deltas_vs_simclr": []}
                            for key in ("avg_acc_last10", "wga_last10", "best_group_last10")}
            continue
        summary[arm] = {key: {"mean": statistics.mean(r[key] for r in selected),
                             "std_across_seeds": statistics.stdev(r[key] for r in selected) if len(selected)>1 else None,
                             "paired_deltas_vs_simclr": [r[key]-baseline[r['seed']][key] for r in selected if r["seed"] in baseline]}
                        for key in ("avg_acc_last10", "wga_last10", "best_group_last10")}
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["seed", "arm", "avg_acc_last10", "wga_last10", "best_group_last10", "probe_epochs", "result", "failure_reason"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "group_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "arm", "ssl_epoch", "group_id", "group_count", "group_accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(group_rows)
    (output / "paired_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Completed control experiment: {output / 'results.csv'}")


if __name__ == "__main__":
    main()
