"""Run label-using downstream diagnostics without changing SSL or graph artifacts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.spurious_eval import linear_probe
from experiments.spurious_eval.datasets.registry import DATASET_REGISTRY
from experiments.spurious_eval.metrics import compute_group_metrics
from experiments.spurious_eval.models.resnet import build_resnet_encoder
from experiments.spurious_eval.training.checkpointing import load_encoder_checkpoint
from experiments.spurious_eval.training.logistic_probe import fit_logistic_probe


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_probe_artifacts(run_root: Path, epoch: int) -> tuple[Path, Path]:
    json_paths = sorted(run_root.glob(f"*/probe_features_epoch_{epoch}_*_val.json"))
    pt_paths = sorted(run_root.glob(f"*/probe_features_epoch_{epoch}_*_val.pt"))
    if len(json_paths) != 1 or len(pt_paths) != 1:
        raise FileNotFoundError(
            f"Expected one final probe JSON and tensor under {run_root}; "
            f"found json={len(json_paths)}, pt={len(pt_paths)}"
        )
    return json_paths[0], pt_paths[0]


def _validate_ids_and_transforms(config: dict, payload: dict) -> dict:
    spec = DATASET_REGISTRY[config["dataset"]]
    dataset_config = spec["config"](
        root_dir=config["data_folder"],
        train_split=config["train_set_linear_layer"],
        eval_split=config["eval_split"],
    )
    full_dataset = spec["dataset"](config["data_folder"])
    train_subset = full_dataset.get_subset(config["train_set_linear_layer"], transform=None)
    eval_subset = full_dataset.get_subset(config["eval_split"], transform=None)
    train_tensors = payload["train"]
    eval_tensors = payload["evaluation"]
    checks = {
        "train_count": len(train_subset) == len(train_tensors[1]),
        "eval_count": len(eval_subset) == len(eval_tensors[1]),
        "train_labels_aligned": torch.equal(train_subset.y_array.cpu(), train_tensors[1].cpu()),
        "eval_labels_aligned": torch.equal(eval_subset.y_array.cpu(), eval_tensors[1].cpu()),
        "train_metadata_aligned": torch.equal(train_subset.metadata_array.cpu(), train_tensors[2].cpu()),
        "eval_metadata_aligned": torch.equal(eval_subset.metadata_array.cpu(), eval_tensors[2].cpu()),
        "target_metadata_index": int(getattr(spec, "target_metadata_index", 1)) == 1,
        "label_cardinality": int(torch.unique(train_tensors[1]).numel()) == int(spec["num_classes"]),
    }
    train_loader, eval_loader = spec["probe_loaders"](
        dataset_config,
        config["batch_size"],
        train_loader_kwargs={"num_workers": 0},
        eval_loader_kwargs={"num_workers": 0},
    )
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "train_transform": type(train_loader.dataset.transform).__name__,
        "eval_transform": type(eval_loader.dataset.transform).__name__,
        "metadata_fields": list(full_dataset.metadata_fields),
        "train_count": len(train_subset),
        "eval_count": len(eval_subset),
    }


def _reproduce_saved_probe(result_path: Path, feature_path: Path, config: dict) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    payload = torch.load(feature_path, map_location="cpu", weights_only=True)
    alignment = _validate_ids_and_transforms(config, payload)
    train_x, train_y, train_metadata = payload["train"]
    eval_x, eval_y, eval_metadata = payload["evaluation"]
    records, convergence = fit_logistic_probe(
        train_x,
        train_y,
        eval_x,
        eval_y,
        num_classes=config["num_classes"],
        l2=config["probe_l2"],
        tolerance=config["probe_tolerance"],
        max_epochs=config["probe_max_epochs"],
    )
    eval_metrics = [compute_group_metrics(record.eval_predictions, eval_y, eval_metadata) for record in records]
    reproduced = {
        "avg_acc_last10": float(np.mean([metric.average for metric in eval_metrics[-10:]]) * 100),
        "wga_last10": float(np.mean([metric.worst_group for metric in eval_metrics[-10:]]) * 100),
        "group_accuracy": (eval_metrics[-1].group_accuracy * 100).tolist(),
        "group_count": eval_metrics[-1].group_counts.long().tolist(),
    }
    expected_metrics = result["metrics"]
    comparisons = {
        "avg_acc_last10": abs(reproduced["avg_acc_last10"] - expected_metrics["Average over last 10 linear val acc"]) <= config["metric_tolerance"],
        "wga_last10": abs(reproduced["wga_last10"] - expected_metrics["Average over last 10 linear val worst-group acc"]) <= config["metric_tolerance"],
        "group_accuracy": np.allclose(
            np.asarray(reproduced["group_accuracy"]),
            np.asarray(expected_metrics["Linear val group accuracies"]),
            atol=config["metric_tolerance"],
            rtol=0,
        ),
        "group_count": reproduced["group_count"] == expected_metrics["Linear val group counts"],
    }
    return {
        "result": str(result_path),
        "features": str(feature_path),
        "alignment": alignment,
        "convergence": convergence,
        "reproduced": reproduced,
        "expected": {
            "avg_acc_last10": expected_metrics["Average over last 10 linear val acc"],
            "wga_last10": expected_metrics["Average over last 10 linear val worst-group acc"],
        },
        "matches_existing": all(comparisons.values()),
        "comparisons": comparisons,
    }


def _check_checkpoint_loading(run_root: Path, config: dict) -> dict:
    checkpoint_paths = sorted(run_root.glob("*/*.pth"))
    if not checkpoint_paths:
        return {"status": "unavailable_after_successful_cleanup", "checked": []}
    checked = []
    run_args_path = checkpoint_paths[0].parent / "args.json"
    run_args = json.loads(run_args_path.read_text(encoding="utf-8")) if run_args_path.is_file() else {}
    model_name = run_args.get("model", "resnet18_large")
    for checkpoint_path in checkpoint_paths:
        encoder, _ = build_resnet_encoder(model_name, load_pretrained_weights=False)
        load_encoder_checkpoint(encoder, str(checkpoint_path))
        checked.append(str(checkpoint_path))
    return {"status": "passed", "checked": checked, "model": model_name}


def _run_frozen_encoder_probe(config: dict, condition: str, model: str, seed: int) -> dict:
    _set_seed(seed)
    artifact_dir = Path(config["output"]) / condition / f"seed{seed}"
    args = SimpleNamespace(
        dataset=config["dataset"],
        data_folder=config["data_folder"],
        train_set_linear_layer=config["train_set_linear_layer"],
        eval_split=config["eval_split"],
        final_test=False,
        model=model,
        ckpt="",
        artifact_dir=str(artifact_dir),
        method="SimCLR",
        head="mlp",
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        probe_solver="logistic",
        probe_l2=config["probe_l2"],
        probe_tolerance=config["probe_tolerance"],
        probe_max_epochs=config["probe_max_epochs"],
        seed=seed,
        device=config["device"],
        use_wandb=False,
        spurious_probe=True,
        ssl_epoch=0,
    )
    metrics = linear_probe.main(args, supcon_epoch=0)
    return {
        "condition": condition,
        "model": model,
        "seed": seed,
        "artifact_dir": str(artifact_dir),
        "metrics": metrics,
    }


def run_diagnostic(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = Path(config["source_output"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "artifact": "downstream_evaluator_diagnostic_v1",
        "protocol": "posthoc_downstream_only",
        "source_output": str(source),
        "saved_probe_reproduction": {},
        "checkpoint_loading": {},
        "frozen_encoder_baselines": [],
    }
    failures: list[str] = []
    for seed in config["seeds"]:
        for arm in config["arms"]:
            run_root = source / f"seed{seed}" / arm / "training"
            result_path, feature_path = _find_probe_artifacts(run_root, config["ssl_epoch"])
            payload = torch.load(feature_path, map_location="cpu", weights_only=True)
            probe_config = {**config, "num_classes": 2}
            reproduction = _reproduce_saved_probe(result_path, feature_path, probe_config)
            report["saved_probe_reproduction"][f"seed{seed}/{arm}"] = reproduction
            report["checkpoint_loading"][f"seed{seed}/{arm}"] = _check_checkpoint_loading(run_root, config)
            if not reproduction["matches_existing"] or not reproduction["alignment"]["passed"]:
                failures.append(f"saved probe check failed for seed{seed}/{arm}")

    for condition, model in (("random_resnet50", "resnet50_large"), ("imagenet_resnet50", "resnet50_pretrained")):
        for seed in config["seeds"]:
            report["frozen_encoder_baselines"].append(
                _run_frozen_encoder_probe(config, condition, model, int(seed))
            )
    report["passed"] = not failures
    report["failures"] = failures
    report_path = output / "diagnostic_report.json"
    _write_json(report_path, report)
    if failures:
        raise RuntimeError("; ".join(failures))
    print(f"Downstream diagnostic passed: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_diagnostic(args.config)


if __name__ == "__main__":
    main()
