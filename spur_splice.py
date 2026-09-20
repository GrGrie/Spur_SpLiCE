"""Entry point of one SSL training run: parse the command line, build the run, fit it.

The command line resolves into the flat training namespace (``cospro.config.training`` plus the
filesystem checks and the run naming here). The namespace builds a ``TrainingState`` around the
configured ``TrainingMethod`` and ``cospro.training.Trainer`` runs the epochs while the callbacks
log, probe and write checkpoints. The namespace stays flat because four stored identities are
derived from it: the storage name, ``args.json``, the run record and the checkpoint options.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from pathlib import Path

# Required by deterministic CUDA matrix multiplications; must be set before CUDA is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from experiments.spurious_eval.datasets.registry import build_loader, dataset_class
from experiments.spurious_eval.losses.contrastive import SimCLRLoss
from experiments.spurious_eval.models.simclr import SimCLRModel
from experiments.spurious_eval.training.optim import build_optimizer
from experiments.spurious_eval.training.reproducibility import (
    make_dataloader_kwargs,
    preserve_rng_state,
    seed_worker,
)
from cospro.config.options import ConfigError, str_to_bool  # noqa: F401  (str_to_bool re-exported)
from cospro.evaluation import ProbeArtifacts, ProbeOptions, probe_checkpoint, seed_probe
from cospro.methods import LoaderContext, build_method
from cospro.training import (
    CheckpointPolicy,
    PeriodicProbe,
    RankMetrics,
    RunRecordLogger,
    StoragePolicy,
    Trainer,
    TrainingState,
    WandbLogger,
    artifact_identity,
)
from cospro.config.training import (
    PROBE_MOMENTUM,
    RELATIONAL_GRAPH_MODES,
    TrainingConfig,
    build_training_parser,
    normalize_training_options,
    parse_training_arguments,
    resolve_epoch_schedule,  # noqa: F401  (re-exported for callers of spur_splice)
)
from splice.compat import with_legacy_option_names
from splice.graph_io import graph_fingerprint
from splice.artifacts import artifact_uri, atomic_write_json
from splice.run_recording import RunRecorder, portable_json
from splice.concept_distillation import load_target_artifact



# Run-record artifact kind per training method.
METHOD_INPUT_ARTIFACT_KINDS = {
    "cospro_relational": "teacher_graph",
    "frozen_concept_distill": "frozen_transfer_targets",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Resolve the command line into the flat training namespace.

    Options, defaults and filesystem-free checks live in :mod:`cospro.config.training`; this function
    adds the checks that read files (teacher graph, target bank) and the run naming.
    """

    parser = build_training_parser()
    args = parse_training_arguments(parser, argv)
    try:
        normalize_training_options(args)
    except ConfigError as exc:
        parser.error(str(exc))
    if args.splice_mode == "frozen_concept_distill":
        if not Path(args.concept_transfer_targets).is_file():
            parser.error("--concept_transfer_targets must point to the retained target bank.")
        target_artifact = load_target_artifact(args.concept_transfer_targets)
        if int(target_artifact["target_dim"]) != 512:
            parser.error("Frozen transfer requires 512-dimensional targets.")
        args.concept_transfer_target_artifact = target_artifact["artifact"]
        args.concept_transfer_cache_fingerprint = target_artifact.get("cache_fingerprint", "")
    if args.splice_mode in RELATIONAL_GRAPH_MODES:
        graph_path = Path(args.cospro_teacher_graph)
        if not graph_path.is_file():
            parser.error(f"--cospro_teacher_graph does not exist: {graph_path}")
        args.cospro_graph_fingerprint = graph_fingerprint(graph_path)
    else:
        args.cospro_graph_fingerprint = None
    args.runtime_versions = runtime_versions()
    args.model_name = format_run_name(args)
    args.wandb_run_name = args.wandb_run_name.strip() or format_wandb_run_name(args)
    args.storage_name = format_storage_name(args)
    artifact_base = Path(args.artifact_dir or args.checkpoint_dir or f"./outputs/seeds/adhoc/seed_{args.seed:02d}/training")
    args.save_folder = str(artifact_base / args.storage_name)
    args.run_record = args.run_record or str(artifact_base.parent / "run.json")
    os.makedirs(args.save_folder, exist_ok=True)
    write_run_config(args)
    return args


def training_config(args: argparse.Namespace) -> TrainingConfig:
    """Typed sections of a namespace returned by :func:`parse_args`."""

    return TrainingConfig.from_namespace(args)


def format_wandb_run_name(args: argparse.Namespace) -> str:
    prefix = f"{args.dataset}_s{args.seed:g}"
    suffix = f"_e{args.epochs}"
    if args.splice_mode in RELATIONAL_GRAPH_MODES:
        return f"{prefix}_CoSpRo_w{args.splice_weight:g}_t{args.cospro_temperature:g}{suffix}"
    return f"{prefix}_SimCLR{suffix}"


def format_storage_name(args: argparse.Namespace) -> str:
    """Return a short, deterministic checkpoint directory name safe for Windows paths."""
    if args.splice_mode in RELATIONAL_GRAPH_MODES:
        experiment = "cospro-relational" if args.splice_mode == "cospro_relational" else "crp-v2-relational"
    else:
        experiment = "la-ssl" if getattr(args, "la_ssl", False) else ("concept-transfer" if args.splice_mode == "frozen_concept_distill" else "base")

    excluded_from_fingerprint = {
        "checkpoint_dir",
        "artifact_dir",
        "attempt_id",
        "arm",
        "data_folder",
        "manifest_path",
        "resume",
        "run_record",
        "runtime_versions",
        "save_folder",
        "storage_name",
        "study",
        "use_wandb",
        "wandb_group",
        "wandb_name",
        "wandb_notes",
        "wandb_run_name",
        "wandb_tags",
    }
    # Historical option names keep storage names stable, so a resumed run reuses its checkpoint folder.
    fingerprint_payload = with_legacy_option_names({
        key: value
        for key, value in vars(args).items()
        if key not in excluded_from_fingerprint
    })
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]

    def path_token(value: object) -> str:
        token = "".join(character if character.isalnum() else "-" for character in str(value))
        return token.strip("-") or "run"

    return (
        f"{path_token(args.dataset)}_s{path_token(f'{args.seed:g}')}_"
        f"{path_token(experiment)}_e{args.epochs}_{fingerprint}"
    )


def format_run_name(args: argparse.Namespace) -> str:
    if args.splice_mode in RELATIONAL_GRAPH_MODES:
        splice_name = (f"cospro_relational_w{args.splice_weight:g}_t{args.cospro_temperature:g}_"
                       f"start{args.cospro_start_epoch}_warm{args.cospro_warmup_epochs}")
    else:
        splice_name = "nosplice"
    run_name = (
        f"SimCLR_{args.dataset}_{args.optimizer}_{args.model}_{args.head}_{splice_name}_"
        f"seed{args.seed:g}_lr{args.learning_rate:g}_bs{args.batch_size}_temp{args.temp:g}_"
        f"amp{int(args.amp)}_cl{int(args.channels_last)}_cudnn{int(args.cudnn_enabled)}_"
        f"bench{int(args.cudnn_benchmark)}"
    )
    return run_name


def write_run_config(args: argparse.Namespace) -> None:
    config_path = Path(args.save_folder) / "args.json"
    payload = {key: value for key, value in vars(args).items() if key != "run_recorder_instance"}
    atomic_write_json(config_path, portable_json(payload))


def _cudnn_version() -> str:
    # cudnn.version() raises on CUDA builds of torch when no GPU is visible.
    try:
        return str(torch.backends.cudnn.version() or "not-available")
    except (RuntimeError, ValueError):
        return "not-available"


def runtime_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda": str(torch.version.cuda or "not-available"),
        "cudnn": _cudnn_version(),
    }
    for distribution in [
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-learn",
        "open-clip-torch",
        "wandb",
    ]:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def set_seed(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)


def configure_training_backend(args: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        cudnn.enabled = args.cudnn_enabled
        cudnn.benchmark = args.cudnn_enabled and args.cudnn_benchmark
        cudnn.deterministic = args.cudnn_enabled
    print(
        "[INFO] Training backend: "
        f"cudnn_enabled={cudnn.enabled} cudnn_benchmark={cudnn.benchmark} "
        f"amp={args.amp} channels_last={args.channels_last} seeded_reproducibility=True",
        flush=True,
    )


def build_dataset_config(args: argparse.Namespace):
    return dataset_class(args.dataset).Config(
        root_dir=args.data_folder, ssl_crop_min=args.ssl_crop_min,
    )


def build_ssl_loader(args: argparse.Namespace, method):
    """The dataset's two-crop loader, wrapped by whatever loader the training method needs."""

    loader = build_loader(
        dataset_class(args.dataset), "ssl", build_dataset_config(args), args.batch_size,
        **make_dataloader_kwargs(args, shuffle=True),
    )
    context = LoaderContext(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        generator=loader.generator,
        worker_init_fn=seed_worker,
    )
    return method.wrap_loader(loader, context)


def build_rank_loader(args: argparse.Namespace):
    """Build an observational loader that cannot advance the training sampler."""

    if args.rank_eval_freq <= 0:
        return None
    return build_loader(
        dataset_class(args.dataset), "rank", build_dataset_config(args), args.batch_size,
        **make_dataloader_kwargs(args, shuffle=False, seed=args.seed + 1_000_000),
    )


def probe_options(args: argparse.Namespace) -> ProbeOptions:
    """The probe's own options, named by the training command line."""

    return ProbeOptions(
        data_folder=args.data_folder,
        train_split=args.train_set_linear_layer,
        eval_split=args.linear_eval_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        head=args.head,
        solver=args.linear_probe_solver,
        l2=args.linear_probe_l2,
        tolerance=args.linear_probe_tolerance,
        max_epochs=args.linear_probe_max_epochs,
        epochs=args.linear_probe_epochs,
        learning_rate=args.linear_learning_rate,
        lr_decay_epochs=tuple(args.linear_lr_decay_epochs),
        lr_decay_rate=args.linear_lr_decay_rate,
        weight_decay=args.linear_weight_decay,
        momentum=PROBE_MOMENTUM,
        cosine=args.cosine,
        spurious_probe=args.linear_spurious_probe,
    )


def training_wandb_run(args: argparse.Namespace):
    """The run this training job already opened. A probe never starts a second one."""

    if not args.use_wandb:
        return None
    import wandb

    return wandb.run


def run_linear_probe(args: argparse.Namespace, ckpt_path: str, epoch: int) -> dict[str, float]:
    """Measure the frozen encoder of one checkpoint, leaving training RNG and backend untouched."""

    artifacts = ProbeArtifacts(
        directory=Path(args.save_folder),
        identity=artifact_identity(args),
        ssl_epoch=epoch,
        ssl_total_epochs=args.epochs,
        retain_every=args.retain_probe_artifacts_every,
        recorder=args.run_recorder_instance,
    )
    with preserve_rng_state():
        try:
            seed_probe(args.seed)
            result = probe_checkpoint(
                dataset_class(args.dataset),
                probe_options(args),
                artifacts,
                model=args.model,
                checkpoint=ckpt_path,
                wandb_run=training_wandb_run(args),
            )
            return result.metrics
        finally:
            configure_training_backend(args)


def build_training_state(args: argparse.Namespace, device: torch.device) -> TrainingState:
    """The method, loaders, model, objective, optimizer and scaler of this run."""

    method = build_method(training_config(args), graph_fingerprint=args.cospro_graph_fingerprint)
    with preserve_rng_state():
        train_loader = build_ssl_loader(args, method)
    with preserve_rng_state():
        rank_loader = build_rank_loader(args)
    for key, value in method.provenance().items():
        setattr(args, key, value)
    configure_training_backend(args)
    model = SimCLRModel(
        name=args.model,
        head=args.head,
        feat_dim=args.feat_dim,
        clip_distillation_dim=method.clip_distillation_dim,
    )
    if args.channels_last and device.type == "cuda":
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and device.type == "cuda":
        model.encoder = torch.nn.DataParallel(model.encoder)
    criterion = SimCLRLoss(temperature=args.temp).to(device)
    optimizer = build_optimizer(args, model)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    return TrainingState(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        method=method,
        train_loader=train_loader,
        rank_loader=rank_loader,
    )


def record_resolved_training_config(args: argparse.Namespace, state: TrainingState, wandb_run, recorder) -> None:
    """Persist values that are resolved only while constructing the dataset."""

    method = state.method
    write_run_config(args)
    recorder.update_config({
        **{key: value for key, value in vars(args).items() if key != "run_recorder_instance"},
        "dataset_identity": {
            "name": args.dataset,
            "ssl_training_examples": len(state.train_loader.dataset),
            "linear_train_split": args.train_set_linear_layer,
            "linear_eval_split": args.linear_eval_split,
        },
    })
    for artifact in method.input_artifacts():
        recorder.register_artifact(
            artifact, kind=METHOD_INPUT_ARTIFACT_KINDS[method.name], stage="input", retention_state="retained",
        )
    if wandb_run is not None:
        wandb_run.config.update(method.provenance(), allow_val_change=True)
        for artifact in method.input_artifacts():
            if artifact.suffix == ".json":
                resolved_path = artifact.resolve()
                wandb_run.save(str(resolved_path), base_path=str(resolved_path.parent), policy="now")


def write_run_status(args: argparse.Namespace, payload: dict[str, object]) -> None:
    """Persist completion, W&B finish and cleanup state beside run artifacts."""

    status_path = Path(args.save_folder) / "run_status.json"
    temporary_path = status_path.with_suffix(status_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(status_path)


def build_run_recorder(args: argparse.Namespace) -> RunRecorder:
    """Open the run record, embedding the manifest that produced this command."""

    manifest_payload = {}
    if args.manifest_path and Path(args.manifest_path).is_file():
        from experiments.runner import read_manifest_file

        manifest_payload = read_manifest_file(args.manifest_path)
    return RunRecorder(
        args.run_record,
        identity=artifact_identity(args),
        config=vars(args),
        runtime=args.runtime_versions,
        manifest=manifest_payload,
    )


def build_callbacks(args: argparse.Namespace, state: TrainingState, storage: StoragePolicy, recorder,
                    wandb_logger: WandbLogger) -> tuple[list, PeriodicProbe]:
    """The observers of a run, in the order they act on a finished epoch."""

    probe = PeriodicProbe(
        state,
        args,
        storage,
        lambda checkpoint, epoch: run_linear_probe(args, checkpoint, epoch),
        mode=args.linear_probe_mode,
        every=args.linear_probe_freq,
        total_epochs=args.epochs,
    )
    callbacks = [
        RankMetrics(state, args, every=args.rank_eval_freq),
        RunRecordLogger(recorder),
        wandb_logger,
        probe,
        CheckpointPolicy(state, args, storage, recorder),
    ]
    return callbacks, probe


def main() -> None:
    args = parse_args()
    print(args)
    set_seed(args)
    device = torch.device(args.device)
    args.device = str(device)

    recorder = build_run_recorder(args)
    args.run_recorder_instance = recorder
    storage = StoragePolicy(args)
    wandb_logger = WandbLogger.start(args)
    cleanup_status: dict[str, object] = {}
    # The logger keeps writing its finish state into this dictionary until the status is persisted.
    status: dict[str, object] = {
        "status": "running",
        "run_identity": {"storage_name": args.storage_name, "save_folder": artifact_uri(args.save_folder)},
        "wandb": wandb_logger.status,
    }
    try:
        if wandb_logger.run is not None:
            status["run_identity"]["wandb"] = wandb_logger.identity
            recorder.set_wandb(wandb_logger.identity)

        state = build_training_state(args, device)
        record_resolved_training_config(args, state, wandb_logger.run, recorder)
        callbacks, probe = build_callbacks(args, state, storage, recorder, wandb_logger)
        Trainer(state, args, callbacks, device).fit()

        wandb_logger.finish()
        cleanup_status = storage.cleanup_after_training()
        status.update({"status": "complete", "cleanup": cleanup_status})
        write_run_status(args, status)
        recorder.finish(final_metrics=probe.metrics, cleanup=cleanup_status)
    except Exception as exc:
        wandb_logger.on_failure(exc)
        status.update({
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "cleanup": cleanup_status,
        })
        write_run_status(args, status)
        recorder.fail(exc, cleanup_status)
        raise


if __name__ == "__main__":
    main()
