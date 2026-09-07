"""Standalone prepare/array/summary runner for the eight direct-transfer runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from splice.concept_distillation import ConceptDistillationRegularizer, load_target_artifact, prepare_target_artifact


def config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(path: Path) -> Path:
    cfg = config(path)
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    target_path = output / "targets_v1.pt"
    if target_path.exists() and not cfg.get("allow_replace_targets", False):
        targets = load_target_artifact(target_path)
        expected_fingerprint = hashlib.md5(Path(cfg["cache"]).read_bytes()).hexdigest()
        if targets.get("cache_fingerprint") != expected_fingerprint:
            raise RuntimeError("Existing target artifact was prepared from a different frozen cache.")
    else:
        prepare_target_artifact(cfg["cache"], target_path, shuffle_seed=cfg["shuffle_seed"])
        targets = load_target_artifact(target_path)
    manifest = {
        "artifact": "concept_transfer_v1_manifest",
        "target_artifact": str(target_path),
        "cache": cfg["cache"],
        "sample_count": len(targets["sample_ids"]),
        "valid_count": int(targets["valid_mask"].sum()),
        "arms": cfg["arms"],
        "seeds": cfg["seeds"],
        "epochs": cfg["epochs"],
        "retention": cfg["retention"],
    }
    manifest_path = output / "prepared.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def smoke(path: Path) -> Path:
    """Deterministic 2-step toy E2E check; it never touches dataset labels."""

    torch.manual_seed(101)
    targets = {
        "raw": torch.nn.functional.normalize(torch.randn(4, 512), dim=1),
        "reconstruction": torch.nn.functional.normalize(torch.randn(4, 512), dim=1),
        "shuffled_reconstruction": torch.nn.functional.normalize(torch.randn(4, 512), dim=1),
        "valid_mask": torch.ones(4, dtype=torch.bool),
    }
    regularizer = ConceptDistillationRegularizer(targets, "reconstruction", 0.1, 0, 1)
    regularizer.set_epoch(1)
    predictions = torch.randn(8, 512, requires_grad=True)
    rows = torch.tensor([0, 1, 2, 3])
    bank, valid = regularizer.targets_for_indices(rows, predictions.device)
    loss = regularizer(predictions, torch.cat([bank, bank]), valid)
    loss.backward()
    if not torch.isfinite(loss) or predictions.grad is None or not torch.isfinite(predictions.grad).all():
        raise RuntimeError("Concept-transfer smoke produced a non-finite or disconnected gradient.")
    output = Path(config(path)["output"])
    output.mkdir(parents=True, exist_ok=True)
    smoke_path = output / "smoke.json"
    smoke_path.write_text(json.dumps({"artifact": "concept_transfer_smoke_v1", "passed": True, "alpha_active": regularizer.scheduled_weight}, indent=2) + "\n", encoding="utf-8")
    return smoke_path


def run_one(path: Path, task_id: int) -> Path:
    cfg = config(path)
    if task_id < 0 or task_id >= len(cfg["seeds"]) * len(cfg["arms"]):
        raise ValueError(f"array task id {task_id} is outside the configured 8-run matrix")
    arm_index = int(task_id) % len(cfg["arms"])
    seed_index = int(task_id) // len(cfg["arms"])
    arm = cfg["arms"][arm_index]
    seed = int(cfg["seeds"][seed_index])
    output = Path(cfg["output"])
    run_root = output / f"seed{seed}" / arm["name"]
    training_root = run_root / "training"
    training_root.mkdir(parents=True, exist_ok=True)
    target_path = output / "targets_v1.pt"
    command = [
        sys.executable, "-u", "spur_splice.py",
        "--dataset", cfg["dataset"], "--data_folder", cfg["data_folder"], "--model", cfg["model"],
        "--seed", str(seed), "--epochs", str(cfg["epochs"]), "--batch_size", str(cfg["batch_size"]),
        "--num_workers", str(cfg["num_workers"]), "--optimizer", "SGD", "--learning_rate", str(cfg["learning_rate"]),
        "--lr_decay_epochs", cfg["lr_decay_epochs"], "--lr_decay_rate", str(cfg["lr_decay_rate"]),
        "--weight_decay", str(cfg["weight_decay"]), "--temp", str(cfg["temperature"]), "--simclr_weight", "1.0",
        "--splice_mode", "frozen_concept_distill", "--concept_transfer_targets", str(target_path),
        "--concept_transfer_target_kind", arm["target_kind"], "--concept_transfer_alpha_max", str(arm["alpha_max"]),
        "--concept_transfer_start_epoch", str(cfg["start_epoch"]), "--concept_transfer_warmup_epochs", str(cfg["warmup_epochs"]),
        "--linear_probe_mode", "periodic", "--linear_probe_freq", str(cfg["probe_frequency"]),
        "--linear_probe_solver", "logistic", "--linear_probe_l2", str(cfg["probe_l2"]),
        "--linear_probe_tolerance", str(cfg["probe_tolerance"]), "--linear_probe_max_epochs", str(cfg["probe_max_epochs"]),
        "--train_set_linear_layer", "ds_train", "--linear_eval_split", "val", "--rank_eval_freq", "0",
        "--checkpoint_dir", str(training_root), "--keep_checkpoints", "--save_freq", str(cfg["save_frequency"]),
        "--checkpoint_keep_count", str(cfg["checkpoint_keep_count"]), "--delete_checkpoints_after_training", "false",
        "--retain_probe_artifacts_every", str(cfg["retain_probe_artifacts_every"]), "--use_wandb",
        "--wandb_name", cfg["wandb_project"], "--entity", cfg["wandb_entity"], "--wandb_group", cfg["wandb_group"],
        "--wandb_run_name", f"concept_transfer_v1_{arm['name']}_seed{seed}", "--amp", "true", "--channels_last", "true", "--cudnn_enabled", "true",
    ]
    (run_root / "command.json").write_text(json.dumps({"command": command, "arm": arm, "seed": seed}, indent=2) + "\n", encoding="utf-8")
    status_path = run_root / "completed.json"
    try:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=True)
    except Exception as exc:
        status_path.write_text(json.dumps({"status": "failed", "error": repr(exc), "command": command}, indent=2) + "\n", encoding="utf-8")
        raise
    status_path.write_text(json.dumps({"status": "complete", "arm": arm, "seed": seed, "command": command, "retention": cfg["retention"]}, indent=2) + "\n", encoding="utf-8")
    return status_path


def summary(path: Path) -> Path:
    cfg = config(path)
    output = Path(cfg["output"])
    rows = []
    failures = []
    for seed in cfg["seeds"]:
        for arm in cfg["arms"]:
            root = output / f"seed{seed}" / arm["name"]
            status = root / "completed.json"
            results = sorted(root.glob("training/**/probe_features_epoch_500_ds_train_val.json"))
            if not status.is_file() or json.loads(status.read_text()).get("status") != "complete" or len(results) != 1:
                failures.append(f"seed{seed}/{arm['name']}: missing completion or unique epoch500 probe")
                continue
            metrics = json.loads(results[0].read_text(encoding="utf-8"))["metrics"]
            rows.append({
                "seed": int(seed), "arm": arm["name"], "result": str(results[0]),
                "avg_acc_last10": metrics["Average over last 10 linear val acc"],
                "wga_last10": metrics["Average over last 10 linear val worst-group acc"],
                "group_accuracy": metrics["Linear val group accuracies"],
                "group_count": metrics["Linear val group counts"],
            })
    report = {
        "artifact": "concept_transfer_v1_summary",
        "rows": rows,
        "failures": failures,
        "passed": not failures and len(rows) == len(cfg["seeds"]) * len(cfg["arms"]),
        "winner_selection": "disabled; report all predeclared arms and seeds",
        "screen_gate": cfg["screen_gate"],
    }
    report_path = output / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"Concept-transfer summary failed; see {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=["prepare", "smoke", "run", "summary"], required=True)
    parser.add_argument("--array-task-id", type=int)
    args = parser.parse_args()
    if args.stage == "prepare": prepare(args.config)
    elif args.stage == "smoke": smoke(args.config)
    elif args.stage == "run": run_one(args.config, args.array_task_id if args.array_task_id is not None else -1)
    else: summary(args.config)


if __name__ == "__main__":
    main()
