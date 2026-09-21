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

from third_party import splice
from experiments.runner import manifest_fingerprint
from cospro.data.registry import (
    canonical_dataset_name,
    dataset_class,
    dataset_names,
)
from cospro.models.resnet import SSL_RESNET_MODEL_NAMES
from cospro.cli.build_cospro_teacher_graphs import teacher_graph_path
from cospro.pipeline.dictionary import DICTIONARY_KINDS, ORDERS
from cospro.cli.cache_splice_dataset import cache_provenance, concept_dictionary, resolve_cache_path
from cospro.cli.generate_cospro_concept_groups import (
    _dataset_image_resolver,
    concept_group_directory,
)
from cospro.config import DEFAULT_PERIODIC_PROBE_FREQ, preset_values, training_defaults
from cospro.config.training import LINEAR_TRAIN_SPLITS
from cospro.tracking.artifacts import PROJECT_ROOT, atomic_write_json, resolve_output_root, run_directory, scratch_root
from cospro.config.settings import data_folder, wandb_entity
from cospro.pipeline import (
    GROUPING_CONFIG_FIELDS,
    CoSpRoAuditConfig,
    load_concept_groups_json,
    validate_cospro_config,
    validate_splice_dataset_cache,
)
from cospro.methods.relational_graph import validate_teacher_graph
from cospro.pipeline.reporting import render_concept_groups_report
from cospro.pipeline.graph_io import load_graph_json


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
        "--dataset", type=canonical_dataset_name, choices=dataset_names(),
        default="celeba",
    )
    data_folder_default = data_folder()
    paths.add_argument(
        "--data-folder", type=Path, default=data_folder_default, required=data_folder_default is None,
        help="Dataset root; defaults to $DATA_FOLDER.",
    )
    paths.add_argument(
        "--feature-root", type=Path, default=scratch_root() / "features" / "Spur_SpLiCE",
        help="Frozen SpLiCE cache root; defaults to <scratch root>/features/Spur_SpLiCE.",
    )
    paths.add_argument("--output-root", type=Path, default=resolve_output_root())
    paths.add_argument("--python", default=sys.executable, help="Python interpreter used by every stage.")
    paths.add_argument("--rebuild-preprocessing", action="store_true")
    paths.add_argument("--dry-run", action="store_true")

    cache = parser.add_argument_group("SpLiCE dataset cache")
    cache.add_argument("--cache-batch-size", type=int, default=64)
    cache.add_argument("--cache-num-workers", type=int, default=4)
    cache.add_argument("--cache-device", default="cuda" if torch.cuda.is_available() else "cpu")
    cache.add_argument("--splice-model", default="open_clip:ViT-B-32")
    cache.add_argument("--splice-pretrained", default="laion2b_s34b_b79k")
    cache.add_argument("--splice-vocab", default=splice.DEFAULT_VOCABULARY, choices=DICTIONARY_KINDS)
    cache.add_argument("--splice-vocab-size", type=int, default=splice.DEFAULT_VOCABULARY_SIZE)
    cache.add_argument("--splice-vocab-file", type=Path, help="Word file of a 'file' dictionary.")
    cache.add_argument("--splice-vocab-order", choices=ORDERS, help="Which end a size keeps (default: head).")
    cache.add_argument("--splice-l1-penalty", type=float, default=0.25)

    # Grouping and audit defaults come from CoSpRoAuditConfig; student defaults are the training
    # defaults with the cospro_student preset applied (cospro/config/presets.py).
    audit_defaults = CoSpRoAuditConfig()
    student_defaults = {**training_defaults(), **preset_values("cospro_student")}

    grouping = parser.add_argument_group("concept grouping")
    grouping.add_argument("--min-concept-frequency", type=float, default=audit_defaults.min_concept_frequency)
    grouping.add_argument("--max-concept-frequency", type=float, default=audit_defaults.max_concept_frequency)
    grouping.add_argument("--text-similarity-threshold", type=float, default=audit_defaults.text_similarity_threshold)
    grouping.add_argument("--coactivation-threshold", type=float, default=audit_defaults.coactivation_threshold)
    grouping.add_argument("--min-group-size", type=int, default=audit_defaults.min_group_size)
    grouping.add_argument("--similarity-chunk-size", type=int, default=audit_defaults.similarity_chunk_size)

    audit = parser.add_argument_group("teacher graph audit")
    audit.add_argument("--max-selected-groups", type=int, default=audit_defaults.max_selected_groups)
    audit.add_argument("--projected-neighbors", type=int, default=audit_defaults.projected_neighbors)
    audit.add_argument(
        "--activation-difference-quantile", type=float, default=audit_defaults.activation_difference_quantile,
    )
    audit.add_argument("--min-intervention-gain", type=float, default=audit_defaults.min_intervention_gain)
    audit.add_argument("--min-coverage", type=float, default=audit_defaults.min_coverage)
    audit.add_argument("--graph-top-k", type=int, default=audit_defaults.graph_top_k)
    audit.add_argument("--max-indegree", type=int, default=audit_defaults.max_indegree)
    audit.add_argument("--indegree-factor", type=float, default=audit_defaults.indegree_factor)
    audit.add_argument("--null-trials", type=int, default=audit_defaults.null_trials)
    audit.add_argument("--null-quantile", type=float, default=audit_defaults.null_quantile)
    audit.add_argument("--audit-seed", type=int, default=audit_defaults.seed)
    audit.add_argument("--orthogonal-tolerance", type=float, default=audit_defaults.orthogonal_tolerance)
    audit.add_argument("--graph-device", default="auto")
    audit.add_argument("--neighbor-backend", choices=("auto", "exact", "lsh"), default=audit_defaults.neighbor_backend)
    audit.add_argument("--ann-threshold", type=int, default=audit_defaults.ann_threshold)
    audit.add_argument("--ann-tables", type=int, default=audit_defaults.ann_tables)
    audit.add_argument("--ann-bucket-size", type=int, default=audit_defaults.ann_bucket_size)
    audit.add_argument(
        "--use-residual-splice-gate", action=argparse.BooleanOptionalAction,
        default=audit_defaults.use_residual_splice_gate,
    )
    audit.add_argument(
        "--residual-splice-similarity-threshold", type=float,
        default=audit_defaults.residual_splice_similarity_threshold,
    )

    student = parser.add_argument_group("student training")
    student.add_argument("--study", default="", help="Defaults to <dataset>_cospro_pipeline.")
    student.add_argument("--seed", type=int, default=1, help="The pipeline trains seed 1 unless told otherwise.")
    student.add_argument("--student-existing", choices=("error", "reuse", "resume", "new-attempt"), default="error")
    student.add_argument("--attempt-id")
    student.add_argument("--student-device", default=student_defaults["device"])
    student.add_argument("--model", choices=SSL_RESNET_MODEL_NAMES, default=None)
    student.add_argument("--head", choices=("linear", "mlp", "identity"), default=student_defaults["head"])
    for option, value_type in (
        ("feat_dim", int), ("epochs", int), ("batch_size", int), ("num_workers", int), ("learning_rate", float),
        ("lr_decay_epochs", str), ("lr_decay_rate", float), ("weight_decay", float), ("momentum", float),
    ):
        student.add_argument("--" + option.replace("_", "-"), type=value_type, default=student_defaults[option])
    student.add_argument("--optimizer", choices=("SGD", "AdamW"), default=student_defaults["optimizer"])
    student.add_argument("--temp", type=float, default=student_defaults["temp"])
    student.add_argument("--simclr-weight", type=float, default=student_defaults["simclr_weight"])
    student.add_argument("--splice-weight", type=float, default=student_defaults["splice_weight"])
    for option, value_type in (
        ("cospro_temperature", float), ("cospro_start_epoch", int), ("cospro_warmup_epochs", int),
        ("cospro_decay_start_epoch", int), ("cospro_decay_end_epoch", int),
    ):
        flag = option.replace("_", "-")
        student.add_argument(
            "--" + flag, "--" + flag.replace("cospro-", "crp-", 1), dest=option, type=value_type,
            default=student_defaults[option],
        )
    for option, value_type in (
        ("ssl_crop_min", float), ("rank_eval_freq", int), ("print_freq", int), ("save_freq", int),
        ("checkpoint_keep_count", int),
    ):
        student.add_argument("--" + option.replace("_", "-"), type=value_type, default=student_defaults[option])
    student.add_argument(
        "--keep-checkpoints", action=argparse.BooleanOptionalAction, default=True,
        help="The pipeline retains the final student checkpoint on scratch by default.",
    )
    for option in ("delete_checkpoints_after_training", "amp", "channels_last", "cudnn_enabled", "cosine"):
        student.add_argument(
            "--" + option.replace("_", "-"), action=argparse.BooleanOptionalAction, default=student_defaults[option],
        )

    probe = parser.add_argument_group("linear evaluation")
    probe.add_argument(
        "--linear-train-split", choices=LINEAR_TRAIN_SPLITS, default=student_defaults["train_set_linear_layer"],
    )
    probe.add_argument("--linear-eval-split", choices=("val", "test"), default="val")
    probe.add_argument("--linear-probe-mode", choices=("final", "periodic", "none"), default="periodic")
    probe.add_argument("--linear-probe-freq", type=int, default=DEFAULT_PERIODIC_PROBE_FREQ)
    probe.add_argument(
        "--linear-probe-solver", choices=("logistic", "sgd"), default=student_defaults["linear_probe_solver"],
    )
    for option, value_type in (
        ("linear_probe_epochs", int), ("linear_probe_l2", float), ("linear_probe_tolerance", float),
        ("linear_probe_max_epochs", int),
    ):
        probe.add_argument("--" + option.replace("_", "-"), type=value_type, default=student_defaults[option])
    probe.add_argument(
        "--linear-spurious-probe", action=argparse.BooleanOptionalAction,
        default=student_defaults["linear_spurious_probe"],
    )

    tracking = parser.add_argument_group("tracking and collection")
    tracking.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=False)
    tracking.add_argument("--wandb-name", default="CoSpRo")
    tracking.add_argument("--wandb-group", default="")
    tracking.add_argument("--wandb-tags", default="")
    tracking.add_argument("--entity", default=wandb_entity())
    tracking.add_argument("--collect-results", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    dataset = dataset_class(args.dataset)
    if args.model is None:
        args.model = dataset.default_model()
    incompatible_model = dataset.model_error(args.model)
    if incompatible_model is not None:
        parser.error(incompatible_model)
    return args


def _configs(args: argparse.Namespace) -> tuple[CoSpRoAuditConfig, CoSpRoAuditConfig]:
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
        "neighbor_backend": args.neighbor_backend,
        "ann_threshold": args.ann_threshold,
        "ann_tables": args.ann_tables,
        "ann_bucket_size": args.ann_bucket_size,
    }
    return (
        validate_cospro_config(CoSpRoAuditConfig(**grouping)),
        validate_cospro_config(CoSpRoAuditConfig(**audit)),
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
        "cospro_temperature": args.cospro_temperature,
        "cospro_start_epoch": args.cospro_start_epoch,
        "cospro_warmup_epochs": args.cospro_warmup_epochs,
        "cospro_decay_start_epoch": args.cospro_decay_start_epoch,
        "cospro_decay_end_epoch": args.cospro_decay_end_epoch,
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
        splice_vocab_file=args.splice_vocab_file,
        splice_vocab_order=args.splice_vocab_order,
        splice_l1_penalty=args.splice_l1_penalty,
    )
    dictionary = concept_dictionary(cache_args)
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

    expected_cache_provenance = cache_provenance(cache_args, dictionary)

    def validate_cache() -> None:
        cached = validate_splice_dataset_cache(
            torch.load(cache_path, map_location="cpu", weights_only=True)
        )
        if cached.get("provenance") != expected_cache_provenance:
            raise RuntimeError(f"Existing cache has incompatible provenance: {cache_path}")
        if not all(str(sample).startswith(f"{args.dataset}:") for sample in cached["sample_ids"]):
            raise RuntimeError(f"Existing cache uses non-canonical sample IDs: {cache_path}")

    cache_command = [
        python, "-u", "-m", "cospro.cli.cache_splice_dataset",
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
        *(["--splice-vocab-file", str(args.splice_vocab_file)] if args.splice_vocab_file else []),
        *(["--splice-vocab-order", args.splice_vocab_order] if args.splice_vocab_order else []),
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
        python, "-u", "-m", "cospro.cli.generate_cospro_concept_groups",
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
        python, "-u", "-m", "cospro.cli.build_cospro_teacher_graphs",
        "--splice-dataset-cache", str(cache_path),
        "--concept-groups", str(groups_path),
        "--config", _json(audit_values),
        "--device", args.graph_device,
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
                python, "-u", "-m", "cospro.cli.collect_results", str(manifest_path),
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
