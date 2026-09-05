"""Post-hoc converged logistic evaluation of saved CRP-control checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

LEGACY_ARMS = ("simclr", "crp_sampler_only", "raw_clip_kl", "splice_crp_kl")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def load_experiment_manifest(config_path: Path) -> tuple[Path, dict, Path]:
    config = _read_json(config_path)
    output_root = Path(config["output"])
    manifest_path = output_root / "experiment.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")
    return output_root, _read_json(manifest_path), manifest_path


def resolve_selection(manifest: dict, seeds: list[int] | None, arms: list[str] | None) -> tuple[list[int], list[str]]:
    selected_seeds = list(manifest.get("seeds", [])) if seeds is None else list(seeds)
    selected_arms = list(manifest.get("arms", LEGACY_ARMS)) if arms is None else list(arms)
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Seeds must be a non-empty list of unique integers.")
    if not selected_arms or len(set(selected_arms)) != len(selected_arms):
        raise ValueError("Arms must be a non-empty list of unique names.")
    manifest_arms = set(manifest.get("arms", LEGACY_ARMS))
    unknown = set(selected_arms).difference(manifest_arms)
    if unknown:
        raise ValueError(f"Requested arms are not present in the experiment manifest: {sorted(unknown)}")
    return selected_seeds, selected_arms


def _assert_validation_split(manifest: dict) -> None:
    if bool(manifest.get("final_test", False)):
        raise ValueError("Checkpoint evaluator refuses a final-test experiment.")
    for key in ("eval_split", "linear_eval_split", "probe_eval_split"):
        if str(manifest.get(key, "val")).lower() != "val":
            raise ValueError("Checkpoint evaluator is validation-only; test split is forbidden.")


def discover_checkpoint(output_root: Path, seed: int, arm: str, epoch: int) -> Path | None:
    root = output_root / f"seed{seed}" / arm
    matches = sorted(root.rglob(f"epoch_{epoch}.pth")) if root.is_dir() else []
    if len(matches) > 1:
        raise ValueError(f"Ambiguous epoch-{epoch} checkpoints for seed={seed}, arm={arm}: {matches}")
    return matches[0] if matches else None


def discover_blocks(output_root: Path, seeds: list[int], arms: list[str], epochs: list[int]) -> list[dict]:
    blocks = []
    for epoch in epochs:
        for seed in seeds:
            checkpoints = {
                arm: discover_checkpoint(output_root, seed, arm, epoch)
                for arm in arms
            }
            missing = [arm for arm, path in checkpoints.items() if path is None]
            blocks.append({
                "seed": seed,
                "ssl_epoch": epoch,
                "status": "incomplete" if missing else "complete",
                "missing_arms": missing,
                "checkpoints": {arm: str(path) for arm, path in checkpoints.items() if path is not None},
            })
    return blocks


def _probe_args(checkpoint: Path, epoch: int, manifest: dict) -> SimpleNamespace:
    run_args_path = checkpoint.parent / "args.json"
    run_args = _read_json(run_args_path) if run_args_path.is_file() else {}
    eval_split = run_args.get("linear_eval_split", manifest.get("linear_eval_split", "val"))
    if eval_split != "val" or run_args.get("final_test", False):
        raise ValueError(f"Checkpoint {checkpoint} is not configured for validation evaluation.")
    values = {
        "dataset": manifest["dataset"],
        "data_folder": manifest["data_folder"],
        "train_set_linear_layer": run_args.get("train_set_linear_layer", manifest.get("train_set_linear_layer", "ds_train")),
        "eval_split": "val",
        "model": run_args.get("model", manifest.get("model", "resnet18")),
        "ckpt": str(checkpoint),
        "method": "SimCLR", "head": "mlp", "kappa": 1.0, "trial": "0",
        "augmented_features": False, "plot_path": "", "energy_threshold": 0.9,
        "rank_threshold": 0.1, "spur_str": 0.0, "num_zero_high": 0, "num_zero_low": 0,
        "batch_size": run_args.get("batch_size", manifest.get("batch_size", 256)),
        "num_workers": run_args.get("num_workers", manifest.get("num_workers", 0)),
        "epochs": run_args.get("linear_probe_epochs", 100),
        "ssl_epoch": epoch,
        "probe_solver": run_args.get("linear_probe_solver", "logistic"),
        "probe_l2": run_args.get("linear_probe_l2", manifest.get("probe_l2", 1e-3)),
        "probe_tolerance": run_args.get("linear_probe_tolerance", manifest.get("probe_tolerance", 1e-6)),
        "probe_max_epochs": run_args.get("linear_probe_max_epochs", manifest.get("probe_max_epochs", 200)),
        "learning_rate": run_args.get("linear_learning_rate", 1.0),
        "lr_decay_epochs": run_args.get("linear_lr_decay_epochs", "auto"),
        "lr_decay_rate": run_args.get("linear_lr_decay_rate", 0.2),
        "weight_decay": run_args.get("linear_weight_decay", 0.0),
        "momentum": 0.9, "cosine": False, "seed": int(run_args.get("seed", 0)),
        "device": run_args.get("device", "cuda"), "use_wandb": False,
        "wandb_name": "posthoc_checkpoint_evaluation", "entity": "",
        "spurious_probe": bool(run_args.get("linear_spurious_probe", True)),
    }
    return SimpleNamespace(**values)


def evaluate_checkpoint(checkpoint: Path, epoch: int, manifest: dict, log_path: Path,
                        force: bool = False) -> tuple[dict, Path]:
    from experiments.spurious_eval import linear_probe

    args = _probe_args(checkpoint, epoch, manifest)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result_path = checkpoint.parent / (
        f"probe_features_epoch_{epoch}_{args.train_set_linear_layer}_{args.eval_split}.json"
    )
    if result_path.is_file() and not force:
        log_path.write_text(
            f"Using existing probe result without overwrite: {result_path}\n", encoding="utf-8"
        )
    else:
        with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            linear_probe.main(args, supcon_epoch=epoch)
    if not result_path.is_file():
        raise RuntimeError(f"Linear probe did not produce its result JSON: {result_path}")
    return _read_json(result_path), result_path


def _group_rows(result: dict) -> list[dict]:
    group_metrics = result.get("group_metrics", {}).get("val", {})
    accuracies = group_metrics.get("accuracy")
    counts = group_metrics.get("count")
    if accuracies is None or counts is None:
        metrics = result.get("metrics", {})
        accuracies = metrics.get("Linear val group accuracies", [])
        counts = metrics.get("Linear val group counts", [])
    if len(accuracies) != len(counts):
        raise ValueError("Probe result has misaligned validation group accuracy/count lists.")
    return [
        {"group_id": group_id, "group_count": int(count), "group_accuracy": float(accuracy)}
        for group_id, (accuracy, count) in enumerate(zip(accuracies, counts))
    ]


def _result_row(seed: int, arm: str, epoch: int, checkpoint: Path, result: dict, result_path: Path) -> dict:
    metrics = result["metrics"]
    return {
        "seed": seed, "arm": arm, "ssl_epoch": epoch,
        "avg_acc_last10": metrics["Average over last 10 linear val acc"],
        "wga_last10": metrics["Average over last 10 linear val worst-group acc"],
        "best_group_last10": metrics["Average over last 10 linear val best-group acc"],
        "probe_epochs": result["convergence"]["epochs"],
        "checkpoint": str(checkpoint),
        "checkpoint_fingerprint": _fingerprint(checkpoint),
        "result": str(result_path),
    }


def _fingerprint(path: Path) -> str:
    import hashlib
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_paired_summary(rows: list[dict]) -> dict:
    baseline = {(row["seed"], row["ssl_epoch"]): row for row in rows if row["arm"] == "simclr"}
    summary = {}
    for arm in sorted({row["arm"] for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        arm_summary = {"count": len(selected)}
        for metric in ("avg_acc_last10", "wga_last10", "best_group_last10"):
            paired = [
                {"seed": row["seed"], "ssl_epoch": row["ssl_epoch"],
                 "delta": row[metric] - baseline[(row["seed"], row["ssl_epoch"])][metric]}
                for row in selected if (row["seed"], row["ssl_epoch"]) in baseline
            ]
            values = [row[metric] for row in selected]
            arm_summary[metric] = {
                "mean": statistics.mean(values) if values else None,
                "std_across_blocks": statistics.stdev(values) if len(values) > 1 else None,
                "paired_deltas_vs_simclr": paired,
            }
        summary[arm] = arm_summary
    return summary


def run(config_path: Path, epochs: list[int], seeds: list[int] | None, arms: list[str] | None,
        validate_only: bool = False, force: bool = False) -> Path:
    output_root, manifest, manifest_path = load_experiment_manifest(config_path)
    _assert_validation_split(manifest)
    selected_seeds, selected_arms = resolve_selection(manifest, seeds, arms)
    epochs = sorted(set(int(epoch) for epoch in epochs))
    if not epochs or any(epoch <= 0 for epoch in epochs):
        raise ValueError("Epochs must be positive integers.")
    blocks = discover_blocks(output_root, selected_seeds, selected_arms, epochs)
    complete = [block for block in blocks if block["status"] == "complete"]
    print(f"Manifest: {manifest_path}")
    for block in blocks:
        print(f"seed={block['seed']} epoch={block['ssl_epoch']}: {block['status']}" +
              (f" missing={block['missing_arms']}" if block["missing_arms"] else ""))
    evaluation_root = output_root / "checkpoint_evaluation"
    outputs = [evaluation_root / name for name in ("evaluation.json", "results.csv", "group_results.csv", "paired_summary.json")]
    if validate_only:
        return evaluation_root
    if not force and any(path.exists() for path in outputs):
        raise FileExistsError("Checkpoint evaluation outputs already exist; pass -Force to replace them.")

    rows, group_rows = [], []
    for block in complete:
        seed, epoch = block["seed"], block["ssl_epoch"]
        for arm in selected_arms:
            checkpoint = Path(block["checkpoints"][arm])
            log_path = evaluation_root / f"seed{seed}" / arm / f"epoch{epoch}" / "probe.log"
            result, result_path = evaluate_checkpoint(checkpoint, epoch, manifest, log_path, force=force)
            if not result.get("convergence", {}).get("converged", False):
                raise RuntimeError(f"Probe did not converge for seed={seed}, arm={arm}, epoch={epoch}.")
            rows.append(_result_row(seed, arm, epoch, checkpoint, result, result_path))
            for group in _group_rows(result):
                group_rows.append({"seed": seed, "arm": arm, "ssl_epoch": epoch, **group})

    evaluation_root.mkdir(parents=True, exist_ok=True)
    (evaluation_root / "evaluation.json").write_text(json.dumps({
        "artifact": "crp_control_checkpoint_evaluation_v1",
        "manifest": str(manifest_path), "epochs": epochs,
        "seeds": selected_seeds, "arms": selected_arms,
        "blocks": blocks, "complete_evaluated_blocks": len(complete),
    }, indent=2), encoding="utf-8")
    result_fields = ["seed", "arm", "ssl_epoch", "avg_acc_last10", "wga_last10", "best_group_last10",
                     "probe_epochs", "checkpoint", "checkpoint_fingerprint", "result"]
    with (evaluation_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields)
        writer.writeheader(); writer.writerows(rows)
    group_fields = ["seed", "arm", "ssl_epoch", "group_id", "group_count", "group_accuracy"]
    with (evaluation_root / "group_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_fields)
        writer.writeheader(); writer.writerows(group_rows)
    (evaluation_root / "paired_summary.json").write_text(
        json.dumps(build_paired_summary(rows), indent=2), encoding="utf-8"
    )
    return evaluation_root


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=Path("scripts/run_crp_controls.conf"))
    parser.add_argument("--epochs", type=int, nargs="+", default=[50])
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = run(args.config, args.epochs, args.seeds, args.arms, args.validate_only, args.force)
    print(f"Checkpoint evaluation output: {output}")


if __name__ == "__main__":
    main()
