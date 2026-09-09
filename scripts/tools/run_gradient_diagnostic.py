"""Prepare, run, and summarize the three bounded CRP gradient diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(config_path: Path) -> Path:
    config = read_config(config_path)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    for required in (config["cache"], *[arm["graph"] for arm in config["arms"]]):
        if not Path(required).is_file():
            raise FileNotFoundError(required)
    manifest = {
        "artifact": "crp_transfer_gradient_diagnostic_manifest_v1",
        "config": str(config_path.resolve()),
        "protocol": config["protocol"],
        "arms": config["arms"],
        "seed": config["seed"],
        "epochs": config["epochs"],
        "prepared": True,
    }
    path = output / "prepared.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_one(config_path: Path, task_id: int) -> Path:
    config = read_config(config_path)
    if task_id < 0 or task_id >= len(config["arms"]):
        raise ValueError(f"array task id {task_id} is outside the configured 3-run matrix")
    arm = config["arms"][int(task_id)]
    output = Path(config["output"])
    run_root = output / f"seed{config['seed']}" / arm["name"]
    training_root = run_root / "training"
    training_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", "spur_splice.py",
        "--dataset", config["dataset"], "--data_folder", config["data_folder"],
        "--model", config["model"], "--seed", str(config["seed"]),
        "--epochs", str(config["epochs"]), "--batch_size", str(config["batch_size"]),
        "--num_workers", str(config["num_workers"]), "--optimizer", "SGD",
        "--learning_rate", str(config["learning_rate"]), "--lr_decay_epochs", config["lr_decay_epochs"],
        "--lr_decay_rate", str(config["lr_decay_rate"]), "--weight_decay", str(config["weight_decay"]),
        "--temp", str(config["temperature"]), "--simclr_weight", "1.0",
        "--splice_mode", "crp_relational", "--splice_weight", str(arm["weight"]),
        "--crp_teacher_graph", arm["graph"], "--crp_temperature", str(config["crp_temperature"]),
        "--crp_start_epoch", str(config["start_epoch"]), "--crp_warmup_epochs", str(config["warmup_epochs"]),
        "--linear_probe_mode", "none", "--rank_eval_freq", "0", "--checkpoint_dir", str(training_root),
        "--gradient_diagnostics", "--gradient_diagnostics_batches", str(config["diagnostic_batches"]),
        "--gradient_diagnostics_epochs", ",".join(str(epoch) for epoch in config["diagnostic_epochs"]),
        "--gradient_diagnostics_output", str(run_root / "gradient_diagnostics.json"),
        "--use_wandb", "--wandb_name", config["wandb_project"], "--entity", config["wandb_entity"],
        "--wandb_group", config["wandb_group"], "--wandb_run_name", f"{config['protocol']}_{arm['name']}_seed{config['seed']}",
        "--amp", "true", "--channels_last", "true", "--cudnn_enabled", "true",
    ]
    (run_root / "command.json").write_text(json.dumps({"command": command}, indent=2) + "\n", encoding="utf-8")
    status_path = run_root / "completed.json"
    try:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=True)
    except Exception as exc:
        status_path.write_text(json.dumps({"status": "failed", "error": repr(exc), "command": command}, indent=2) + "\n", encoding="utf-8")
        raise
    status_path.write_text(json.dumps({"status": "complete", "command": command, "gradient_diagnostics": str(run_root / "gradient_diagnostics.json")}, indent=2) + "\n", encoding="utf-8")
    return status_path


def summarize(config_path: Path) -> Path:
    config = read_config(config_path)
    output = Path(config["output"])
    records = []
    failures = []
    for arm in config["arms"]:
        root = output / f"seed{config['seed']}" / arm["name"]
        diagnostic_path = root / "gradient_diagnostics.json"
        status_path = root / "completed.json"
        if not status_path.is_file() or json.loads(status_path.read_text()).get("status") != "complete":
            failures.append(f"{arm['name']}: missing/failed completed.json")
            continue
        if not diagnostic_path.is_file():
            failures.append(f"{arm['name']}: missing gradient diagnostics")
            continue
        payload = json.loads(diagnostic_path.read_text())
        diagnostics = payload.get("gradient_diagnostics", [])
        expected = {(epoch, batch) for epoch in config["diagnostic_epochs"] for batch in range(config["diagnostic_batches"])}
        actual = {(int(item["epoch"]), int(item["batch"])) for item in diagnostics}
        if not expected.issubset(actual):
            failures.append(f"{arm['name']}: incomplete diagnostic grid")
        if any(not item.get("finite", False) for item in diagnostics):
            failures.append(f"{arm['name']}: non-finite diagnostic")
        records.append({"arm": arm["name"], "diagnostics": diagnostics, "diagnostic_count": len(diagnostics)})
    report = {"artifact": "crp_transfer_gradient_diagnostic_report_v1", "records": records, "failures": failures, "passed": not failures}
    path = output / "gradient_diagnostic_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"Gradient diagnostic failed; see {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=["prepare", "run", "summary"], required=True)
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
