from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path

# Required by deterministic CUDA matrix multiplications; must be set before CUDA is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from experiments.spurious_eval import linear_probe
from experiments.spurious_eval.datasets.registry import DATASET_REGISTRY
from experiments.spurious_eval.evaluation_protocol import resolve_evaluation_split, resolve_probe_mode
from experiments.spurious_eval.losses.contrastive import SimCLRLoss
from experiments.spurious_eval.models.resnet import SSL_RESNET_MODEL_NAMES
from experiments.spurious_eval.models.simclr import SimCLRModel
from experiments.spurious_eval.training.checkpointing import load_checkpoint, save_checkpoint
from experiments.spurious_eval.training.optim import adjust_learning_rate, build_optimizer
from experiments.spurious_eval.training.ssl_loop import log_rank_metrics, train_one_epoch
from splice.crp_training import (
    CrpRelationalRegularizer,
    build_crp_training_loader,
    load_teacher_graph,
    save_crp_concept_report,
)
from splice.graph_io import graph_fingerprint
from splice.splice import DEFAULT_VOCABULARY, DEFAULT_VOCABULARY_SIZE


RELATIONAL_GRAPH_MODES = {"crp_relational"}


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Spur_SpLiCE SimCLR SSL training")
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=50, help="Checkpoint frequency when --keep_checkpoints is enabled.")
    parser.add_argument(
        "--rank_eval_freq",
        type=int,
        default=100,
        help="Compute full-dataset representation-rank metrics every N epochs; 0 disables them.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)

    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument(
        "--lr_decay_epochs",
        type=str,
        default="auto",
        help="Comma-separated SSL LR milestones, or 'auto' for 70%%, 80%%, and 90%% of --epochs.",
    )
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "AdamW"])

    parser.add_argument("--dataset", type=str, default="waterbirds", choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--data_folder", type=str, default="./datasets")
    parser.add_argument("--model", type=str, default="resnet18_large", choices=SSL_RESNET_MODEL_NAMES)
    parser.add_argument("--head", type=str, default="mlp", choices=["linear", "mlp", "identity"])
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument(
        "--simclr_weight",
        type=float,
        default=1.0,
        help="Weight of the SimCLR/NT-Xent objective. Set to 0 only for relational KL-only ablations.",
    )
    parser.add_argument("--ssl_crop_min", "--ssl-crop-min", dest="ssl_crop_min", type=float, default=0.2)

    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--channels_last", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument(
        "--cudnn_enabled",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable cuDNN for SimCLR training. SpLiCE scoring still starts with cuDNN disabled.",
    )
    parser.add_argument("--cudnn_benchmark", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument(
        "--keep_checkpoints",
        action="store_true",
        help="Persist epoch/last checkpoints. By default checkpoints are temporary and only support linear probing.",
    )
    parser.add_argument(
        "--checkpoint_keep_count",
        type=int,
        default=2,
        help="Number of most recent epoch checkpoints to retain while training.",
    )
    parser.add_argument(
        "--delete_checkpoints_after_training",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Delete epoch checkpoint files after successful training while preserving last.pth.",
    )
    parser.add_argument(
        "--delete_epoch_checkpoints_after_training",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Delete only epoch checkpoint files after training; preserve Linear Probe artifacts.",
    )
    parser.add_argument(
        "--retain_probe_artifacts_every",
        type=int,
        default=100,
        help="Retain bulky downstream probe tensors only at these SSL epochs; 0 keeps all probe tensors.",
    )
    parser.add_argument("--resume", type=str, default="")

    parser.add_argument("--train_set_linear_layer", type=str, default="ds_train", choices=["train", "ds_train", "us_train", "balanced_train", "val"])
    parser.add_argument(
        "--linear_eval_split",
        type=str,
        default=None,
        choices=["val", "test"],
        help="Linear-probe evaluation split. Defaults to val; test requires --final_test.",
    )
    parser.add_argument(
        "--final_test",
        action="store_true",
        help="Evaluate a locked final configuration on test instead of the validation default.",
    )
    parser.add_argument(
        "--linear_probe_mode",
        type=str,
        default=None,
        choices=["final", "periodic", "none"],
        help="Defaults to periodic on val. --final_test restricts evaluation to one final probe.",
    )
    parser.add_argument("--linear_probe_epochs", type=int, default=100)
    parser.add_argument("--linear_probe_solver", choices=["logistic", "sgd"], default="logistic")
    parser.add_argument("--linear_probe_l2", type=float, default=1e-3)
    parser.add_argument("--linear_probe_tolerance", type=float, default=1e-6)
    parser.add_argument("--linear_probe_max_epochs", type=int, default=200)
    parser.add_argument(
        "--linear_probe_freq",
        type=int,
        default=None,
        help="Run periodic linear evaluation every N SSL epochs (default: 25, independent of save_freq).",
    )
    parser.add_argument("--linear_learning_rate", type=float, default=1.0)
    parser.add_argument(
        "--linear_lr_decay_epochs",
        type=str,
        default="auto",
        help="Comma-separated probe LR milestones, or 'auto' for 60%%, 75%%, and 90%% of probe epochs.",
    )
    parser.add_argument("--linear_lr_decay_rate", type=float, default=0.2)
    parser.add_argument("--linear_weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--linear_spurious_probe",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Log an auxiliary linear probe for residual spurious-attribute predictability.",
    )

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_name", default="Spur_SpLiCE")
    parser.add_argument(
        "--wandb_run_name",
        default="",
        help="Optional concise W&B display name. Checkpoint directories retain the full reproducibility name.",
    )
    parser.add_argument("--entity", default="gsgrechkin-rptu")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_tags", default="", help="Comma-separated W&B tags.")
    parser.add_argument(
        "--splice_mode",
        type=str,
        default="none",
        choices=["none", "crp_relational"],
    )
    parser.add_argument("--splice_weight", type=float, default=0.0)
    parser.add_argument(
        "--crp_teacher_graph",
        type=str,
        default="",
        help="Label-free CRP teacher graph used by the relational graph mode.",
    )
    parser.add_argument(
        "--crp_temperature",
        type=float,
        default=0.1,
        help="Temperature of the student relation distribution.",
    )
    parser.add_argument(
        "--crp_start_epoch",
        type=int,
        default=10,
        help="Number of pure-SimCLR epochs before relational distillation starts.",
    )
    parser.add_argument(
        "--crp_warmup_epochs",
        type=int,
        default=10,
        help="Linear warm-up duration for the CRP relational loss weight; 0 disables warm-up.",
    )
    parser.add_argument(
        "--crp_decay_start_epoch",
        type=int,
        default=0,
        help="Epoch at which relational-loss decay begins; 0 with end=0 disables decay.",
    )
    parser.add_argument(
        "--crp_decay_end_epoch",
        type=int,
        default=0,
        help="Epoch at which relational-loss weight reaches zero.",
    )
    args = parser.parse_args()
    try:
        args.linear_eval_split = resolve_evaluation_split(args.linear_eval_split, args.final_test)
        args.linear_probe_mode = resolve_probe_mode(args.linear_probe_mode, args.final_test)
    except ValueError as exc:
        parser.error(str(exc))
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.linear_probe_epochs <= 0:
        parser.error("--linear_probe_epochs must be positive.")
    if args.simclr_weight < 0:
        parser.error("--simclr_weight must be non-negative.")
    try:
        args.lr_decay_epochs = resolve_epoch_schedule(args.lr_decay_epochs, args.epochs, (0.70, 0.80, 0.90))
        args.linear_lr_decay_epochs = resolve_epoch_schedule(
            args.linear_lr_decay_epochs,
            args.linear_probe_epochs,
            (0.60, 0.75, 0.90),
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.use_splice = args.splice_mode != "none"
    if args.splice_mode in RELATIONAL_GRAPH_MODES and args.splice_weight < 0:
        parser.error("--splice_weight must be non-negative for relational graph modes.")
    if args.simclr_weight == 0 and args.splice_mode not in RELATIONAL_GRAPH_MODES:
        parser.error("--simclr_weight 0 is supported only for CRP relational training.")
    if args.simclr_weight == 0 and args.splice_weight <= 0:
        parser.error("KL-only relational training requires --splice_weight to be positive.")
    if args.splice_mode in RELATIONAL_GRAPH_MODES and not args.crp_teacher_graph.strip():
        parser.error("--crp_teacher_graph is required for relational graph modes.")
    if args.splice_mode in RELATIONAL_GRAPH_MODES:
        graph_path = Path(args.crp_teacher_graph)
        if not graph_path.is_file():
            parser.error(f"--crp_teacher_graph does not exist: {graph_path}")
        args.crp_graph_fingerprint = graph_fingerprint(graph_path)
    else:
        args.crp_graph_fingerprint = None
    if args.crp_temperature <= 0:
        parser.error("--crp_temperature must be positive.")
    if args.crp_start_epoch < 0 or args.crp_warmup_epochs < 0:
        parser.error("--crp_start_epoch and --crp_warmup_epochs must be non-negative.")
    if args.crp_decay_start_epoch < 0 or args.crp_decay_end_epoch < 0:
        parser.error("--crp_decay_start_epoch and --crp_decay_end_epoch must be non-negative.")
    if bool(args.crp_decay_start_epoch) != bool(args.crp_decay_end_epoch):
        parser.error("CRP decay start/end must both be zero or both be set.")
    if args.crp_decay_end_epoch and args.crp_decay_end_epoch <= args.crp_decay_start_epoch:
        parser.error("--crp_decay_end_epoch must be greater than --crp_decay_start_epoch.")
    if not 0 < args.ssl_crop_min <= 1:
        parser.error("--ssl-crop-min must be in the interval (0, 1].")
    if args.dataset == "spur_cifar10" and (
        args.model.endswith("_large") or args.model == "resnet50_pretrained"
    ):
        parser.error("spur_cifar10 uses 32x32 images; choose --model resnet18 or --model resnet50.")
    if args.cudnn_benchmark and not args.cudnn_enabled:
        parser.error("--cudnn_benchmark true requires --cudnn_enabled true.")
    if args.cudnn_benchmark:
        parser.error("--cudnn_benchmark must remain false because training is reproducible by default.")
    if args.rank_eval_freq < 0:
        parser.error("--rank_eval_freq must be non-negative.")
    if args.linear_probe_freq is not None and args.linear_probe_freq < 0:
        parser.error("--linear_probe_freq must be non-negative.")
    if args.keep_checkpoints and args.save_freq <= 0:
        parser.error("--save_freq must be positive when --keep_checkpoints is enabled.")
    if args.checkpoint_keep_count <= 0:
        parser.error("--checkpoint_keep_count must be positive.")
    if args.retain_probe_artifacts_every < 0:
        parser.error("--retain_probe_artifacts_every must be non-negative.")
    if args.batch_size > 256:
        args.warm = True
    if args.warm:
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate**3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                1 + math.cos(math.pi * args.warm_epochs / args.epochs)
            ) / 2
        else:
            args.warmup_to = args.learning_rate
    else:
        args.warmup_from = 0.0
        args.warmup_to = args.learning_rate
        args.warm_epochs = 0
    if args.linear_probe_epochs is None:
        args.linear_probe_epochs = 100
    if args.linear_learning_rate is None:
        args.linear_learning_rate = 1.0
    if args.linear_probe_freq is None:
        args.linear_probe_freq = 25 if args.linear_probe_mode == "periodic" else 0
    args.n_cls = DATASET_REGISTRY[args.dataset]["num_classes"]
    args.runtime_versions = runtime_versions()
    args.model_name = format_run_name(args)
    args.wandb_run_name = args.wandb_run_name.strip() or format_wandb_run_name(args)
    args.storage_name = format_storage_name(args)
    args.save_folder = str(
        Path(args.checkpoint_dir or f"./save/SimCLR/{args.dataset}_models") / args.storage_name
    )
    os.makedirs(args.save_folder, exist_ok=True)
    write_run_config(args)
    return args


def resolve_epoch_schedule(value: str, total_epochs: int, fractions: tuple[float, ...]) -> list[int]:
    """Resolve explicit milestones or scale an automatic schedule to a run length."""
    normalized = str(value).strip().lower()
    if normalized == "auto":
        milestones = sorted(
            {
                int(round(total_epochs * fraction))
                for fraction in fractions
                if 0 < int(round(total_epochs * fraction)) < total_epochs
            }
        )
    else:
        try:
            milestones = [int(epoch.strip()) for epoch in normalized.split(",") if epoch.strip()]
        except ValueError as exc:
            raise ValueError("LR milestones must be comma-separated integers or 'auto'.") from exc
    if any(epoch <= 0 or epoch >= total_epochs for epoch in milestones):
        raise ValueError(f"LR milestones must be between 1 and {total_epochs - 1}; got {milestones}.")
    if milestones != sorted(set(milestones)):
        raise ValueError(f"LR milestones must be unique and increasing; got {milestones}.")
    return milestones


def format_wandb_run_name(args: argparse.Namespace) -> str:
    prefix = f"{args.dataset}_s{args.seed:g}"
    suffix = f"_e{args.epochs}"
    if args.splice_mode == "crp_relational":
        return f"{prefix}_CRP_w{args.splice_weight:g}_t{args.crp_temperature:g}{suffix}"
    return f"{prefix}_SimCLR{suffix}"


def format_storage_name(args: argparse.Namespace) -> str:
    """Return a short, deterministic checkpoint directory name safe for Windows paths."""
    if args.splice_mode == "crp_relational":
        experiment = "crp-v2-relational"
    else:
        experiment = "base"

    excluded_from_fingerprint = {
        "checkpoint_dir",
        "data_folder",
        "resume",
        "runtime_versions",
        "save_folder",
        "storage_name",
        "use_wandb",
        "wandb_group",
        "wandb_name",
        "wandb_notes",
        "wandb_run_name",
        "wandb_tags",
    }
    fingerprint_payload = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded_from_fingerprint
    }
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
    if args.splice_mode == "crp_relational":
        splice_name = (f"crp_relational_w{args.splice_weight:g}_t{args.crp_temperature:g}_"
                       f"start{args.crp_start_epoch}_warm{args.crp_warmup_epochs}")
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
    payload = vars(args).copy()
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def runtime_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
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


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader_kwargs(args: argparse.Namespace, shuffle: bool, seed: int | None = None) -> dict:
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed if seed is None else seed)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "generator": loader_generator,
    }
    if shuffle or args.num_workers > 0:
        loader_kwargs["worker_init_fn"] = seed_worker
    return loader_kwargs


def build_dataset_config(args: argparse.Namespace):
    return DATASET_REGISTRY[args.dataset]["config"](
        root_dir=args.data_folder, ssl_crop_min=args.ssl_crop_min,
    )


def build_ssl_loader(args: argparse.Namespace):
    dataset_spec = DATASET_REGISTRY[args.dataset]
    config = build_dataset_config(args)
    loader_kwargs = make_dataloader_kwargs(args, shuffle=True)
    loader = dataset_spec["ssl_loader"](
        config,
        args.batch_size,
        splice_mode=args.splice_mode,
        **loader_kwargs,
    )
    if args.splice_mode not in RELATIONAL_GRAPH_MODES:
        return loader

    args.relational_graph_empty = False
    source_indices = getattr(loader.dataset, "indices", None)
    if source_indices is None:
        raise ValueError("CRP training requires an SSL dataset with stable source indices.")
    graph, loaded_graph_fingerprint = load_teacher_graph(
        args.crp_teacher_graph,
        args.dataset,
        source_indices,
    )
    if loaded_graph_fingerprint != args.crp_graph_fingerprint:
        raise ValueError("Relational teacher graph changed after argument validation; restart the run.")
    args.crp_graph_fingerprint = loaded_graph_fingerprint
    args.teacher_graph_artifact = graph["artifact"]
    args.teacher_graph_config = graph.get("config", {})
    stats = graph.get("degree_stats", {})
    args.teacher_graph_degree_stats = stats
    args.teacher_graph_selected_group_ids = graph.get("selected_group_ids", [])
    args.teacher_graph_removed_concepts = sorted(
        {
            concept
            for group in graph.get("groups", [])
            if group.get("selected")
            for concept in group.get("concepts", [])
        }
    )
    if graph["artifact"] in {
        "splice_crp_v2_teacher_graph",
        "splice_crp_v3_teacher_graph",
        "splice_crp_v4_teacher_graph",
    }:
        report_path = save_crp_concept_report(graph, args.crp_teacher_graph)
        concept_report = json.loads(report_path.read_text(encoding="utf-8"))
        top_concepts = [
            item["concept"]
            for item in concept_report["important_concepts"]
            if item["training_edge_count"] > 0
        ][:10]
        print(
            f"[INFO] CRP concept report: path={report_path}, "
            f"teacher_projected={concept_report['teacher_projected_concepts']}, "
            f"top_training_concepts={top_concepts}",
            flush=True,
        )
    print(
        f"[INFO] Loaded {graph['artifact']} teacher graph: "
        f"edges={stats.get('edge_count', int((graph['neighbor_indices'] >= 0).sum()))}, "
        f"coverage={stats.get('coverage', float((graph['weights'].sum(dim=1) > 0).float().mean())):.4f}, "
        f"path={args.crp_teacher_graph}",
        flush=True,
    )
    if not torch.any(graph["weights"].sum(dim=1) > 0):
        if getattr(args, "simclr_weight", 1.0) == 0:
            raise ValueError(
                "KL-only relational training requires a non-empty teacher graph; "
                "the resolved graph contains no supported anchors."
            )
        args.relational_graph_empty = True
        print(
            "[WARNING] Relational teacher graph is empty; using the standard SimCLR "
            "DataLoader and disabling relational regularization.",
            flush=True,
        )
        return loader

    crp_loader = build_crp_training_loader(
        loader.dataset,
        graph,
        args.batch_size,
        args.num_workers,
        loader.generator,
        worker_init_fn=seed_worker,
    )
    return crp_loader


def build_rank_loader(args: argparse.Namespace):
    """Build an observational loader that cannot advance the training sampler."""

    if args.rank_eval_freq <= 0:
        return None
    dataset_spec = DATASET_REGISTRY[args.dataset]
    loader_kwargs = make_dataloader_kwargs(args, shuffle=False, seed=args.seed + 1_000_000)
    return dataset_spec["rank_loader"](
        build_dataset_config(args),
        args.batch_size,
        **loader_kwargs,
    )


@contextmanager
def preserve_rng_state():
    """Prevent observational probes from changing subsequent SSL randomness."""

    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def run_linear_probe(args: argparse.Namespace, ckpt_path: str, epoch: int) -> dict[str, float]:
    with preserve_rng_state():
        try:
            return linear_probe.main(build_linear_probe_args(args, ckpt_path), supcon_epoch=epoch)
        finally:
            configure_training_backend(args)


def build_linear_probe_args(args: argparse.Namespace, ckpt_path: str) -> argparse.Namespace:
    probe_settings = {
        "dataset": args.dataset,
        "data_folder": args.data_folder,
        "train_set_linear_layer": args.train_set_linear_layer,
        "eval_split": args.linear_eval_split,
        "model": args.model,
        "ckpt": ckpt_path,
        "head": args.head,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": args.linear_probe_epochs,
        "probe_solver": getattr(args, "linear_probe_solver", "logistic"),
        "probe_l2": getattr(args, "linear_probe_l2", 1e-3),
        "probe_tolerance": getattr(args, "linear_probe_tolerance", 1e-6),
        "probe_max_epochs": getattr(args, "linear_probe_max_epochs", 200),
        "learning_rate": args.linear_learning_rate,
        "lr_decay_epochs": args.linear_lr_decay_epochs,
        "lr_decay_rate": args.linear_lr_decay_rate,
        "weight_decay": args.linear_weight_decay,
        "momentum": 0.9,
        "cosine": args.cosine,
        "seed": args.seed,
        "device": args.device,
        "use_wandb": args.use_wandb,
        "wandb_name": args.wandb_name,
        "entity": args.entity,
        "spurious_probe": args.linear_spurious_probe,
    }
    return argparse.Namespace(**probe_settings)


def build_training_state(args: argparse.Namespace, device: torch.device):
    with preserve_rng_state():
        train_loader = build_ssl_loader(args)
    with preserve_rng_state():
        rank_loader = build_rank_loader(args)
    configure_training_backend(args)
    model = SimCLRModel(
        name=args.model,
        head=args.head,
        feat_dim=args.feat_dim,
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
    if args.splice_mode in RELATIONAL_GRAPH_MODES:
        if getattr(args, "relational_graph_empty", False):
            splice_regularizer = None
        else:
            splice_regularizer = CrpRelationalRegularizer(
                train_loader.crp_graph,
                weight=args.splice_weight,
                temperature=args.crp_temperature,
                start_epoch=args.crp_start_epoch,
                warmup_epochs=args.crp_warmup_epochs,
                decay_start_epoch=args.crp_decay_start_epoch,
                decay_end_epoch=args.crp_decay_end_epoch,
            )
    else:
        splice_regularizer = None
    return train_loader, rank_loader, model, criterion, optimizer, scaler, splice_regularizer


def record_resolved_training_config(args: argparse.Namespace, train_loader, wandb_run) -> None:
    """Persist values that are resolved only while constructing the dataset."""

    write_run_config(args)
    if wandb_run is not None:
        resolved = {}
        if args.splice_mode in RELATIONAL_GRAPH_MODES:
            resolved.update(
                {
                    "teacher_graph_artifact": args.teacher_graph_artifact,
                    "teacher_graph_config": args.teacher_graph_config,
                    "teacher_graph_degree_stats": args.teacher_graph_degree_stats,
                    "teacher_graph_selected_group_ids": args.teacher_graph_selected_group_ids,
                    "teacher_graph_removed_concepts": args.teacher_graph_removed_concepts,
                    "relational_graph_empty": getattr(args, "relational_graph_empty", False),
                }
            )
        wandb_run.config.update(resolved, allow_val_change=True)
        if args.splice_mode in RELATIONAL_GRAPH_MODES:
            graph_path = Path(args.crp_teacher_graph).resolve()
            wandb_run.save(str(graph_path), base_path=str(graph_path.parent), policy="now")


def get_probe_score(metrics: dict[str, float]) -> float:
    preferred_keys = [
        "Average over last 10 linear val worst-group acc",
        "Average over 10 last linear val worst-group acc",
        "Average over last 10 linear test worst-group acc",
        "Average over 10 last linear test worst-group acc",
        "Last linear val worst-group acc",
        "Linear val worst-group acc",
    ]

    for key in preferred_keys:
        if key in metrics:
            return float(metrics[key])

    raise KeyError(
        "Could not find averaged worst-group accuracy in probe metrics. "
        f"Available keys: {list(metrics.keys())}"
    )


def maybe_run_periodic_probe(args: argparse.Namespace, save_file: str, epoch: int) -> dict[str, float] | None:
    if args.linear_probe_mode != "periodic":
        return None
    if not args.linear_probe_freq or epoch % args.linear_probe_freq != 0:
        return None
    return run_linear_probe(args, save_file, epoch)


def maybe_run_final_probe(args: argparse.Namespace, save_file: str, already_probed_epoch: int) -> dict[str, float] | None:
    if args.linear_probe_mode == "none":
        return None
    if already_probed_epoch == args.epochs:
        return None
    return run_linear_probe(args, save_file, args.epochs)


def prune_epoch_checkpoints(args: argparse.Namespace) -> None:
    """Keep only the newest periodic epoch checkpoints for crash recovery."""

    if not args.keep_checkpoints:
        return
    checkpoint_dir = Path(args.save_folder)
    epoch_checkpoints: list[tuple[int, Path]] = []
    for checkpoint_path in checkpoint_dir.glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", checkpoint_path.name)
        if match:
            epoch_checkpoints.append((int(match.group(1)), checkpoint_path))
    epoch_checkpoints.sort(key=lambda item: item[0], reverse=True)
    for _, checkpoint_path in epoch_checkpoints[args.checkpoint_keep_count :]:
        checkpoint_path.unlink()


def cleanup_default_checkpoints(args: argparse.Namespace) -> None:
    temporary_paths = [Path(args.save_folder) / "probe_tmp.pth", Path(args.save_folder) / "probe_tmp.pth.tmp"]
    for temporary_path in temporary_paths:
        if temporary_path.exists():
            temporary_path.unlink()


def cleanup_all_checkpoints(args: argparse.Namespace) -> dict[str, object]:
    """Delete epoch checkpoint artifacts while preserving the final last.pth."""

    checkpoint_dir = Path(args.save_folder)
    removed_count = 0
    for checkpoint_path in checkpoint_dir.iterdir():
        if not checkpoint_path.is_file():
            continue
        if checkpoint_path.name == "last.pth":
            continue
        if not (checkpoint_path.name.endswith(".pth") or checkpoint_path.name.endswith(".pth.tmp")):
            continue
        checkpoint_path.unlink()
        removed_count += 1
    print(f"[INFO] Removed {removed_count} checkpoint files from {checkpoint_dir}")
    return {"requested": True, "removed_count": removed_count, "completed": True}


def cleanup_probe_artifacts(args: argparse.Namespace) -> dict[str, object]:
    """Keep small probe JSONs and only selected bulky feature tensors."""

    interval = args.retain_probe_artifacts_every
    if interval == 0:
        return {"requested": True, "removed_count": 0, "retained_epochs": "all", "completed": True}

    removed_count = 0
    retained_epochs: set[int] = set()
    checkpoint_dir = Path(args.save_folder)
    for feature_path in checkpoint_dir.glob("probe_features_epoch_*.pt"):
        match = re.fullmatch(r"probe_features_epoch_(\d+)(?:_.+)?\.pt", feature_path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch > 0 and epoch % interval == 0:
            retained_epochs.add(epoch)
            continue
        feature_path.unlink()
        removed_count += 1
    return {
        "requested": True,
        "removed_count": removed_count,
        "retained_epochs": sorted(retained_epochs),
        "completed": True,
    }


def _wandb_identity(wandb_run) -> dict[str, object] | None:
    if wandb_run is None:
        return None
    return {
        key: value
        for key, value in {
            "id": getattr(wandb_run, "id", None),
            "name": getattr(wandb_run, "name", None),
            "entity": getattr(wandb_run, "entity", None),
            "project": getattr(wandb_run, "project", None),
            "url": getattr(wandb_run, "url", None),
        }.items()
        if value is not None
    }


def write_run_status(args: argparse.Namespace, payload: dict[str, object]) -> None:
    """Persist completion, W&B finish, and cleanup state beside run artifacts."""

    status_path = Path(args.save_folder) / "run_status.json"
    temporary_path = status_path.with_suffix(status_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(status_path)


def main() -> None:
    args = parse_args()
    print(args)
    set_seed(args)
    device = torch.device(args.device)
    args.device = str(device)

    wandb_run = None
    wandb_finished = False
    cleanup_status: dict[str, object] = {}
    status: dict[str, object] = {
        "status": "running",
        "run_identity": {"storage_name": args.storage_name, "save_folder": args.save_folder},
        "wandb": {"enabled": bool(args.use_wandb), "finish_called": False, "finish_succeeded": False},
    }
    try:
        if args.use_wandb:
            with preserve_rng_state():
                import wandb

                wandb_config = vars(args).copy()
                wandb_tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
                wandb_run = wandb.init(
                    project=args.wandb_name,
                    name=args.wandb_run_name,
                    config=wandb_config,
                    entity=args.entity,
                    group=args.wandb_group or None,
                    tags=wandb_tags or None,
                )
            status["run_identity"]["wandb"] = _wandb_identity(wandb_run)

        train_loader, rank_loader, model, criterion, optimizer, scaler, splice_regularizer = build_training_state(
            args, device
        )
        record_resolved_training_config(args, train_loader, wandb_run)
        start_epoch = (
            load_checkpoint(
                model,
                optimizer,
                args.resume,
                device,
                scaler=scaler,
                loader_generator=train_loader.generator,
                expected_crp_graph_fingerprint=getattr(args, "crp_graph_fingerprint", None),
            )
            + 1
            if args.resume
            else 1
        )
        last_probe_epoch = 0
        probe_file = os.path.join(args.save_folder, "probe_tmp.pth")
        prune_epoch_checkpoints(args)

        for epoch in range(start_epoch, args.epochs + 1):
            adjust_learning_rate(args, optimizer, epoch)
            time1 = time.time()
            train_metrics = train_one_epoch(
                train_loader, model, criterion, optimizer, scaler, epoch, args, splice_regularizer
            )
            time2 = time.time()
            print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

            log_metrics = wandb_run is not None or epoch % args.print_freq == 0
            log_rank = args.rank_eval_freq > 0 and epoch % args.rank_eval_freq == 0
            if log_metrics or log_rank:
                with preserve_rng_state():
                    log_rank_metrics(
                        model,
                        rank_loader,
                        optimizer,
                        train_metrics,
                        epoch,
                        args,
                        wandb_run,
                        compute_rank=log_rank,
                    )

            should_probe = (
                args.linear_probe_mode == "periodic"
                and args.linear_probe_freq > 0
                and epoch % args.linear_probe_freq == 0
            )
            if should_probe:
                save_checkpoint(
                    model,
                    optimizer,
                    args,
                    epoch,
                    probe_file,
                    scaler=scaler,
                    loader_generator=train_loader.generator,
                )
                run_linear_probe(args, probe_file, epoch)
                last_probe_epoch = epoch
                if os.path.exists(probe_file):
                    os.remove(probe_file)

            if args.keep_checkpoints and epoch % args.save_freq == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    args,
                    epoch,
                    os.path.join(args.save_folder, f"epoch_{epoch}.pth"),
                    scaler=scaler,
                    loader_generator=train_loader.generator,
                )
                prune_epoch_checkpoints(args)

        if args.linear_probe_mode != "none" and last_probe_epoch != args.epochs:
            save_checkpoint(
                model,
                optimizer,
                args,
                args.epochs,
                probe_file,
                scaler=scaler,
                loader_generator=train_loader.generator,
            )
            run_linear_probe(args, probe_file, args.epochs)
            if os.path.exists(probe_file):
                os.remove(probe_file)

        if args.keep_checkpoints:
            save_checkpoint(
                model,
                optimizer,
                args,
                args.epochs,
                os.path.join(args.save_folder, "last.pth"),
                scaler=scaler,
                loader_generator=train_loader.generator,
            )

        if wandb_run is not None:
            wandb_run.finish()
            wandb_finished = True
            status["wandb"].update({"finish_called": True, "finish_succeeded": True})

        if args.delete_checkpoints_after_training or args.delete_epoch_checkpoints_after_training:
            cleanup_status["ssl_checkpoints"] = cleanup_all_checkpoints(args)
            if args.delete_checkpoints_after_training:
                cleanup_status["probe_artifacts"] = cleanup_probe_artifacts(args)
            else:
                cleanup_status["probe_artifacts"] = {
                    "requested": False,
                    "removed_count": 0,
                    "completed": True,
                }
        else:
            cleanup_default_checkpoints(args)
            cleanup_status["ssl_checkpoints"] = {
                "requested": False,
                "removed_count": 0,
                "completed": True,
            }
            cleanup_status["probe_artifacts"] = {
                "requested": False,
                "removed_count": 0,
                "completed": True,
            }
        status.update({"status": "complete", "cleanup": cleanup_status})
        write_run_status(args, status)
    except Exception as exc:
        if wandb_run is not None and not wandb_finished:
            try:
                wandb_run.finish()
                status["wandb"].update({"finish_called": True, "finish_succeeded": True})
            except Exception as finish_error:  # preserve the original failure and recovery artifacts
                status["wandb"].update({"finish_called": True, "finish_succeeded": False})
                status["wandb"]["finish_error"] = repr(finish_error)
        status.update({
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "cleanup": cleanup_status,
        })
        write_run_status(args, status)
        raise


if __name__ == "__main__":
    main()
