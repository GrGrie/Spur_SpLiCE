"""Run the complete CoSpRo cache-to-results training pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

import torch

import splice
from experiments.runner import manifest_fingerprint
from experiments.spurious_eval.datasets.registry import (
    CANONICAL_DATASET_REGISTRY,
    canonical_dataset_name,
)
from experiments.spurious_eval.models.resnet import SSL_RESNET_MODEL_NAMES
from scripts.tools.build_cospro_teacher_graphs import teacher_graph_path
from scripts.tools.cache_splice_dataset import resolve_cache_path
from scripts.tools.generate_cospro_concept_groups import (
    _dataset_image_resolver,
    concept_group_directory,
)
from splice.artifacts import PROJECT_ROOT, atomic_write_json, run_directory
from splice.cospro import (
    GROUPING_CONFIG_FIELDS,
    CrpAuditConfig,
    load_concept_groups_json,
    validate_cospro_config,
    validate_splice_dataset_cache,
)
from splice.cospro_training import validate_teacher_graph
from splice.cospro_reporting import render_concept_groups_report
from splice.graph_io import load_graph_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _run(command: list[str], env: dict[str, str]) -> None:
    print("[PIPELINE] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _stage(
    name: str,
    output: Path,
    command: list[str],
    validator: Callable[[], None],
    *,
    rebuild: bool,
    dry_run: bool,
    env: dict[str, str],
) -> None:
    if dry_run:
        if output.is_file() and not rebuild:
            print(f"[PIPELINE] Would validate and reuse {name}: {output}", flush=True)
        else:
            print(f"[PIPELINE] Would build {name}: {output}", flush=True)
            print("[PIPELINE] " + " ".join(command), flush=True)
        return
    if output.is_file() and not rebuild:
        validator()
        print(f"[PIPELINE] Reusing validated {name}: {output}", flush=True)
        return
    _run(command, env)
    if not output.is_file():
        raise RuntimeError(f"{name} completed without producing {output}")
    validator()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    paths = parser.add_argument_group("pipeline and paths")
    paths.add_argument(
        "--dataset", type=canonical_dataset_name, choices=sorted(CANONICAL_DATASET_REGISTRY),
        default="celeba",
    )
    paths.add_argument("--data-folder", type=Path, required=True)
    paths.add_argument("--feature-root", type=Path, required=True)
    paths.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    paths.add_argument("--python", default=sys.executable, help="Python interpreter used by every stage.")
    paths.add_argument("--rebuild-preprocessing", action="store_true")
    paths.add_argument("--dry-run", action="store_true")

    cache = parser.add_argument_group("SpLiCE dataset cache")
    cache.add_argument("--cache-batch-size", type=int, default=64)
    cache.add_argument("--cache-num-workers", type=int, default=4)
    cache.add_argument("--cache-device", default="cuda" if torch.cuda.is_available() else "cpu")
    cache.add_argument("--splice-model", default="open_clip:ViT-B-32")
    cache.add_argument("--splice-pretrained", default="laion2b_s34b_b79k")
    cache.add_argument("--splice-vocab", default=splice.DEFAULT_VOCABULARY)
    cache.add_argument("--splice-vocab-size", type=int, default=splice.DEFAULT_VOCABULARY_SIZE)
    cache.add_argument("--splice-l1-penalty", type=float, default=0.25)

    grouping = parser.add_argument_group("concept grouping")
    grouping.add_argument("--min-concept-frequency", type=float, default=0.01)
    grouping.add_argument("--max-concept-frequency", type=float, default=0.95)
    grouping.add_argument("--text-similarity-threshold", type=float, default=0.82)
    grouping.add_argument("--coactivation-threshold", type=float, default=0.35)
    grouping.add_argument("--min-group-size", type=int, default=1)
    grouping.add_argument("--similarity-chunk-size", type=int, default=512)

    audit = parser.add_argument_group("teacher graph audit")
    audit.add_argument("--max-selected-groups", type=int, default=0)
    audit.add_argument("--projected-neighbors", type=int, default=20)
    audit.add_argument("--activation-difference-quantile", type=float, default=0.75)
    audit.add_argument("--min-intervention-gain", type=float, default=1e-4)
    audit.add_argument("--min-coverage", type=float, default=0.01)
    audit.add_argument("--graph-top-k", type=int, default=3)
    audit.add_argument("--max-indegree", type=int, default=10)
    audit.add_argument("--indegree-factor", type=float, default=3.0)
    audit.add_argument("--null-trials", type=int, default=16)
    audit.add_argument("--null-quantile", type=float, default=0.95)
    audit.add_argument("--audit-seed", type=int, default=0)
    audit.add_argument("--orthogonal-tolerance", type=float, default=1e-6)
    audit.add_argument(
        "--use-residual-splice-gate", action=argparse.BooleanOptionalAction, default=True,
    )
    audit.add_argument("--residual-splice-similarity-threshold", type=float, default=0.25)

    student = parser.add_argument_group("student training")
    student.add_argument("--study", default="", help="Defaults to <dataset>_cospro_pipeline.")
    student.add_argument("--seed", type=int, default=1)
    student.add_argument("--student-existing", choices=("error", "reuse", "resume", "new-attempt"), default="error")
    student.add_argument("--attempt-id")
    student.add_argument("--student-device", default="cuda" if torch.cuda.is_available() else "cpu")
    student.add_argument("--model", choices=SSL_RESNET_MODEL_NAMES, default=None)
    student.add_argument("--head", choices=("linear", "mlp", "identity"), default="mlp")
    student.add_argument("--feat-dim", type=int, default=128)
    student.add_argument("--epochs", type=int, default=500)
    student.add_argument("--batch-size", type=int, default=128)
    student.add_argument("--num-workers", type=int, default=4)
    student.add_argument("--learning-rate", type=float, default=0.01)
    student.add_argument("--lr-decay-epochs", default="auto")
    student.add_argument("--lr-decay-rate", type=float, default=0.1)
    student.add_argument("--weight-decay", type=float, default=1e-4)
    student.add_argument("--momentum", type=float, default=0.9)
    student.add_argument("--optimizer", choices=("SGD", "AdamW"), default="SGD")
    student.add_argument("--temp", type=float, default=0.05)
    student.add_argument("--simclr-weight", type=float, default=1.0)
    student.add_argument("--splice-weight", type=float, default=0.5)
    student.add_argument("--cospro-temperature", "--crp-temperature", dest="crp_temperature", type=float, default=0.25)
    student.add_argument("--cospro-start-epoch", "--crp-start-epoch", dest="crp_start_epoch", type=int, default=10)
    student.add_argument("--cospro-warmup-epochs", "--crp-warmup-epochs", dest="crp_warmup_epochs", type=int, default=10)
    student.add_argument("--cospro-decay-start-epoch", "--crp-decay-start-epoch", dest="crp_decay_start_epoch", type=int, default=0)
    student.add_argument("--cospro-decay-end-epoch", "--crp-decay-end-epoch", dest="crp_decay_end_epoch", type=int, default=0)
    student.add_argument("--ssl-crop-min", type=float, default=0.2)
    student.add_argument("--rank-eval-freq", type=int, default=100)
    student.add_argument("--print-freq", type=int, default=10)
    student.add_argument("--save-freq", type=int, default=50)
    student.add_argument("--checkpoint-keep-count", type=int, default=2)
    student.add_argument("--keep-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    student.add_argument("--delete-checkpoints-after-training", action=argparse.BooleanOptionalAction, default=True)
    student.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    student.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    student.add_argument("--cudnn-enabled", action=argparse.BooleanOptionalAction, default=True)
    student.add_argument("--cosine", action=argparse.BooleanOptionalAction, default=False)

    probe = parser.add_argument_group("linear evaluation")
    probe.add_argument(
        "--linear-train-split",
        choices=("train", "ds_train", "us_train", "balanced_train", "val"),
        default="ds_train",
    )
    probe.add_argument("--linear-eval-split", choices=("val", "test"), default="val")
    probe.add_argument("--linear-probe-mode", choices=("final", "periodic", "none"), default="periodic")
    probe.add_argument("--linear-probe-freq", type=int, default=25)
    probe.add_argument("--linear-probe-solver", choices=("logistic", "sgd"), default="logistic")
    probe.add_argument("--linear-probe-epochs", type=int, default=100)
    probe.add_argument("--linear-probe-l2", type=float, default=1e-3)
    probe.add_argument("--linear-probe-tolerance", type=float, default=1e-6)
    probe.add_argument("--linear-probe-max-epochs", type=int, default=200)
    probe.add_argument("--linear-spurious-probe", action=argparse.BooleanOptionalAction, default=True)

    tracking = parser.add_argument_group("tracking and collection")
    tracking.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=False)
    tracking.add_argument("--wandb-name", default="CoSpRo")
    tracking.add_argument("--wandb-group", default="")
    tracking.add_argument("--wandb-tags", default="")
    tracking.add_argument("--entity", default="gsgrechkin-rptu")
    tracking.add_argument("--collect-results", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.model is None:
        args.model = "resnet18" if args.dataset == "spur_cifar10" else "resnet18_large"
    if args.dataset == "spur_cifar10" and (
        args.model.endswith("_large") or args.model == "resnet50_pretrained"
    ):
        parser.error("spur_cifar10 uses 32x32 images; choose --model resnet18 or --model resnet50.")
    return args


def _configs(args: argparse.Namespace) -> tuple[CrpAuditConfig, CrpAuditConfig]:
    grouping = {
        "min_concept_frequency": args.min_concept_frequency,
        "max_concept_frequency": args.max_concept_frequency,
        "text_similarity_threshold": args.text_similarity_threshold,
        "coactivation_threshold": args.coactivation_threshold,
        "min_group_size": args.min_group_size,
        "similarity_chunk_size": args.similarity_chunk_size,
    }
    audit = {
        **grouping,
        "max_selected_groups": args.max_selected_groups,
        "projected_neighbors": args.projected_neighbors,
        "activation_difference_quantile": args.activation_difference_quantile,
        "min_intervention_gain": args.min_intervention_gain,
        "min_coverage": args.min_coverage,
        "graph_top_k": args.graph_top_k,
        "max_indegree": args.max_indegree,
        "indegree_factor": args.indegree_factor,
        "null_trials": args.null_trials,
        "null_quantile": args.null_quantile,
        "seed": args.audit_seed,
        "orthogonal_tolerance": args.orthogonal_tolerance,
        "use_residual_splice_gate": args.use_residual_splice_gate,
        "residual_splice_similarity_threshold": args.residual_splice_similarity_threshold,
    }
    return (
        validate_cospro_config(CrpAuditConfig(**grouping)),
        validate_cospro_config(CrpAuditConfig(**audit)),
    )


def _student_manifest(args: argparse.Namespace, graph_path: Path) -> dict:
    flags = []
    if args.keep_checkpoints:
        flags.append("keep_checkpoints")
    if args.cosine:
        flags.append("cosine")
    if args.use_wandb:
        flags.append("use_wandb")
    locked_test = args.linear_eval_split == "test"
    if locked_test and args.linear_probe_mode != "final":
        raise ValueError("Held-out test evaluation requires --linear-probe-mode final.")
    common = {
        "dataset": args.dataset,
        "data_folder": str(args.data_folder),
        "device": args.student_device,
        "model": args.model,
        "head": args.head,
        "feat_dim": args.feat_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "lr_decay_epochs": args.lr_decay_epochs,
        "lr_decay_rate": args.lr_decay_rate,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "optimizer": args.optimizer,
        "temp": args.temp,
        "simclr_weight": args.simclr_weight,
        "ssl_crop_min": args.ssl_crop_min,
        "splice_mode": "cospro_relational",
        "splice_weight": args.splice_weight,
        "cospro_teacher_graph": str(graph_path),
        "cospro_temperature": args.crp_temperature,
        "cospro_start_epoch": args.crp_start_epoch,
        "cospro_warmup_epochs": args.crp_warmup_epochs,
        "cospro_decay_start_epoch": args.crp_decay_start_epoch,
        "cospro_decay_end_epoch": args.crp_decay_end_epoch,
        "rank_eval_freq": args.rank_eval_freq,
        "print_freq": args.print_freq,
        "save_freq": args.save_freq,
        "checkpoint_keep_count": args.checkpoint_keep_count,
        "delete_checkpoints_after_training": args.delete_checkpoints_after_training,
        "amp": args.amp,
        "channels_last": args.channels_last,
        "cudnn_enabled": args.cudnn_enabled,
        "train_set_linear_layer": args.linear_train_split,
        "linear_eval_split": "val" if locked_test else args.linear_eval_split,
        "linear_probe_mode": "periodic" if locked_test else args.linear_probe_mode,
        "linear_probe_freq": args.linear_probe_freq,
        "linear_probe_solver": args.linear_probe_solver,
        "linear_probe_epochs": args.linear_probe_epochs,
        "linear_probe_l2": args.linear_probe_l2,
        "linear_probe_tolerance": args.linear_probe_tolerance,
        "linear_probe_max_epochs": args.linear_probe_max_epochs,
        "linear_spurious_probe": args.linear_spurious_probe,
        "wandb_name": args.wandb_name,
        "wandb_group": args.wandb_group or args.study,
        "wandb_tags": args.wandb_tags or f"{args.dataset},cospro,full-pipeline",
        "entity": args.entity,
    }
    manifest = {
        "name": args.study,
        "description": f"Full {args.dataset} CoSpRo pipeline student run.",
        "seeds": [args.seed],
        "flags": flags,
        "common": common,
        "arms": {"cospro": {"args": {}}},
    }
    if locked_test:
        manifest["locked_test"] = {
            "description": f"Explicit final-only evaluation on the {args.dataset} test split.",
            "flags": ["final_test"],
            "args": {
                "linear_eval_split": "test",
                "linear_probe_mode": "final",
                "linear_probe_freq": 0,
            },
        }
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.study = args.study.strip() or f"{args.dataset}_cospro_pipeline"
    args.data_folder = args.data_folder.expanduser().resolve()
    args.feature_root = args.feature_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    grouping_config, audit_config = _configs(args)

    cache_args = argparse.Namespace(
        dataset=args.dataset,
        output_root=args.feature_root,
        splice_model=args.splice_model,
        splice_pretrained=args.splice_pretrained,
        splice_vocab=args.splice_vocab,
        splice_vocab_size=args.splice_vocab_size,
        splice_l1_penalty=args.splice_l1_penalty,
    )
    cache_path = resolve_cache_path(cache_args)
    groups_root = args.output_root / "shared" / args.dataset / "graphs" / "concept_groups"
    groups_path = concept_group_directory(groups_root, grouping_config) / "concept_groups.json"
    graph_path = teacher_graph_path(groups_path, audit_config)
    # Validate run identity and test-protocol combinations before any expensive
    # preprocessing begins.
    run_directory(
        args.seed,
        args.study,
        "cospro",
        args.attempt_id or ("locked-test" if args.linear_eval_split == "test" else "primary"),
        root=args.output_root,
    )
    manifest = _student_manifest(args, graph_path)
    python = str(Path(args.python).expanduser()) if os.sep in args.python else args.python
    env = {**os.environ, "SPUR_SPLICE_OUTPUT_ROOT": str(args.output_root)}

    expected_cache_provenance = {
        "dataset": args.dataset,
        "split": "train",
        "splice_model": args.splice_model,
        "splice_pretrained": args.splice_pretrained,
        "splice_vocab": args.splice_vocab,
        "splice_vocab_size": args.splice_vocab_size,
        "splice_l1_penalty": args.splice_l1_penalty,
    }

    def validate_cache() -> None:
        cached = validate_splice_dataset_cache(
            torch.load(cache_path, map_location="cpu", weights_only=True)
        )
        if cached.get("provenance") != expected_cache_provenance:
            raise RuntimeError(f"Existing cache has incompatible provenance: {cache_path}")
        if not all(str(sample).startswith(f"{args.dataset}:") for sample in cached["sample_ids"]):
            raise RuntimeError(f"Existing cache uses non-canonical sample IDs: {cache_path}")

    cache_command = [
        python, "-u", "-m", "scripts.tools.cache_splice_dataset",
        "--dataset", args.dataset,
        "--data-folder", str(args.data_folder),
        "--output-root", str(args.feature_root),
        "--batch-size", str(args.cache_batch_size),
        "--num-workers", str(args.cache_num_workers),
        "--device", args.cache_device,
        "--splice-model", args.splice_model,
        "--splice-pretrained", args.splice_pretrained,
        "--splice-vocab", args.splice_vocab,
        "--splice-vocab-size", str(args.splice_vocab_size),
        "--splice-l1-penalty", str(args.splice_l1_penalty),
    ]
    _stage(
        "SpLiCE dataset cache", cache_path, cache_command, validate_cache,
        rebuild=args.rebuild_preprocessing, dry_run=args.dry_run, env=env,
    )

    grouping_values = {field: getattr(grouping_config, field) for field in GROUPING_CONFIG_FIELDS}

    def validate_groups() -> None:
        if args.dry_run and not cache_path.is_file():
            return
        cached = validate_splice_dataset_cache(
            torch.load(cache_path, map_location="cpu", weights_only=True)
        )
        artifact = load_concept_groups_json(groups_path)
        if artifact["config"] != grouping_values:
            raise RuntimeError(f"Existing concept groups have an incompatible config: {groups_path}")
        if artifact["sample_ids"] != cached["sample_ids"] or artifact["vocabulary"] != cached["vocabulary"]:
            raise RuntimeError(f"Existing concept groups do not match the dataset cache: {groups_path}")

    groups_command = [
        python, "-u", "-m", "scripts.tools.generate_cospro_concept_groups",
        "--splice-dataset-cache", str(cache_path),
        "--output-root", str(groups_root),
        "--data-folder", str(args.data_folder),
        "--text-similarity-threshold", str(args.text_similarity_threshold),
        "--coactivation-threshold", str(args.coactivation_threshold),
        "--config", _json(grouping_values),
    ]
    _stage(
        "concept groups", groups_path, groups_command, validate_groups,
        rebuild=args.rebuild_preprocessing, dry_run=args.dry_run, env=env,
    )
    if not args.dry_run:
        artifact = load_concept_groups_json(groups_path)
        expected_images = sum(
            len(group.get("representative_samples", []))
            for group in artifact.get("report_diagnostics", {}).get("composite_groups", [])
        )
        report_path = groups_path.with_suffix(".html")
        rendered_images = (
            report_path.read_text(encoding="utf-8").count("<img ")
            if report_path.is_file()
            else 0
        )
        if rendered_images != expected_images:
            print(
                f"[PIPELINE] Repairing concept report thumbnails: "
                f"found={rendered_images} expected={expected_images}",
                flush=True,
            )
            resolver = _dataset_image_resolver(artifact, args.data_folder)
            render_concept_groups_report(artifact, report_path, image_resolver=resolver)
            repaired_images = report_path.read_text(encoding="utf-8").count("<img ")
            if repaired_images != expected_images:
                raise RuntimeError(
                    f"Concept report contains {repaired_images}/{expected_images} thumbnails: {report_path}"
                )

    audit_values = {
        key: value for key, value in asdict(audit_config).items() if key not in GROUPING_CONFIG_FIELDS
    }

    def validate_graph() -> None:
        graph = validate_teacher_graph(load_graph_json(graph_path))
        if graph.get("config") != asdict(audit_config):
            raise RuntimeError(f"Existing teacher graph has an incompatible config: {graph_path}")
        source = graph.get("concept_groups_source", {})
        if Path(str(source.get("path", ""))).resolve() != groups_path.resolve():
            raise RuntimeError(f"Existing teacher graph points to different concept groups: {graph_path}")
        if source.get("sha256") != _sha256(groups_path):
            raise RuntimeError(f"Existing teacher graph was built from changed concept groups: {graph_path}")

    graph_command = [
        python, "-u", "-m", "scripts.tools.build_cospro_teacher_graphs",
        "--splice-dataset-cache", str(cache_path),
        "--concept-groups", str(groups_path),
        "--config", _json(audit_values),
    ]
    _stage(
        "teacher graph", graph_path, graph_command, validate_graph,
        rebuild=args.rebuild_preprocessing, dry_run=args.dry_run, env=env,
    )

    manifest_id = manifest_fingerprint(manifest)[:12]
    manifest_path = (
        args.output_root / "shared" / args.dataset / "pipelines"
        / f"{args.study}_seed_{args.seed:02d}_{manifest_id}.json"
    )
    if args.dry_run:
        print(f"[PIPELINE] Would write student manifest: {manifest_path}", flush=True)
    else:
        atomic_write_json(manifest_path, manifest)

    runner_command = [
        python, "-u", "-m", "experiments.runner", str(manifest_path),
        "--seed", str(args.seed), "--arm", "cospro",
        "--existing", args.student_existing,
        "--output-root", str(args.output_root),
    ]
    if args.attempt_id:
        runner_command.extend(("--attempt-id", args.attempt_id))
    if args.linear_eval_split == "test":
        runner_command.append("--locked-test")
    if args.dry_run:
        runner_command.append("--dry-run")
        print("[PIPELINE] " + " ".join(runner_command), flush=True)
    else:
        _run(runner_command, env)

    results_path = args.output_root / "reports" / args.study / "results.json"
    if args.collect_results and not args.dry_run:
        _run(
            [
                python, "-u", "-m", "scripts.tools.collect_results", str(manifest_path),
                "--output", str(results_path),
                "--output-root", str(args.output_root),
            ],
            env,
        )
    print(f"[PIPELINE] Teacher graph: {graph_path}", flush=True)
    print(f"[PIPELINE] Student manifest: {manifest_path}", flush=True)
    if args.collect_results:
        print(f"[PIPELINE] Results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
