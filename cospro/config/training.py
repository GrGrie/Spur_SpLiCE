"""Typed configuration of an SSL training run (``spur_splice.py``).

Every field name equals the historical argparse destination. ``TrainingConfig.flat()`` therefore
reproduces the namespace that feeds the storage-name hash, args.json, run.json, the W&B config and
checkpoint options (see tests/golden/resolved_configs.json).

Resolution runs in two layers. :func:`normalize_training_options` performs every check and derived
value that needs no filesystem access; ``spur_splice.parse_args`` adds the filesystem checks (teacher
graph, target bank) and the run naming.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, fields
from typing import Any

import torch

from cospro.config.options import ConfigError, add_section_arguments, option, section_defaults, section_from_namespace
from cospro.config.presets import PRESETS, preset_values
from cospro.data.registry import canonical_dataset_name, dataset_class, dataset_names
from cospro.evaluation.protocol import resolve_evaluation_split, resolve_probe_mode
from cospro.models.resnet import SSL_RESNET_MODEL_NAMES
from cospro.compat import LEGACY_RELATIONAL_MODE
from cospro.config.settings import wandb_entity

RELATIONAL_GRAPH_MODES = frozenset({"cospro_relational", LEGACY_RELATIONAL_MODE})
TRAINING_MODES = ("none", "cospro_relational", LEGACY_RELATIONAL_MODE, "frozen_concept_distill", "concept_factors")
LINEAR_TRAIN_SPLITS = ("train", "ds_train", "us_train", "balanced_train", "val")
DEFAULT_PERIODIC_PROBE_FREQ = 25


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class LoggingOptions:
    print_freq: int = option(10, parse=int)
    rank_eval_freq: int = option(
        100, parse=int, help="Compute full-dataset representation-rank metrics every N epochs; 0 disables them.",
    )


@dataclass(frozen=True)
class OptimizationOptions:
    batch_size: int = option(256, parse=int)
    epochs: int = option(500, parse=int)
    learning_rate: float = option(0.01, parse=float)
    lr_decay_epochs: Any = option(
        "auto", parse=str, help="Comma-separated SSL LR milestones, or 'auto' for 70%%, 80%% and 90%% of --epochs.",
    )
    lr_decay_rate: float = option(0.1, parse=float)
    weight_decay: float = option(1e-4, parse=float)
    momentum: float = option(0.9, parse=float)
    optimizer: str = option("SGD", parse=str, choices=("SGD", "AdamW"))
    cosine: bool = option(style="flag")
    warm: bool = option(style="flag")


@dataclass(frozen=True)
class DataOptions:
    dataset: str = option("waterbirds", parse=canonical_dataset_name, choices=dataset_names())
    data_folder: str = option("./datasets", parse=str)
    num_workers: int = option(32, parse=int)
    ssl_crop_min: float = option(0.2, flags=("--ssl_crop_min", "--ssl-crop-min"), parse=float)


@dataclass(frozen=True)
class ModelOptions:
    model: str | None = option(None, parse=str, choices=SSL_RESNET_MODEL_NAMES)
    head: str = option("mlp", parse=str, choices=("linear", "mlp", "identity"))
    feat_dim: int = option(128, parse=int)


@dataclass(frozen=True)
class SSLOptions:
    temp: float = option(0.5, parse=float)
    simclr_weight: float = option(
        1.0, parse=float,
        help="Weight of the SimCLR/NT-Xent objective. Set to 0 only for relational KL-only ablations.",
    )


@dataclass(frozen=True)
class RuntimeOptions:
    seed: int = option(0, parse=int)
    device: str = option(parse=str, default_factory=_default_device)
    amp: bool = option(True, style="switch")
    channels_last: bool = option(True, style="switch")
    cudnn_enabled: bool = option(
        True, style="switch",
        help="Enable cuDNN for SimCLR training. SpLiCE scoring still starts with cuDNN disabled.",
    )
    cudnn_benchmark: bool = option(False, style="switch")


@dataclass(frozen=True)
class RunIdentityOptions:
    study: str = option("adhoc", parse=str)
    arm: str = option("training", parse=str)
    attempt_id: str = option("standalone", parse=str)
    run_record: str = option("", parse=str)
    manifest_path: str = option("", parse=str)


@dataclass(frozen=True)
class StorageOptions:
    checkpoint_dir: str | None = option(None, parse=str)
    artifact_dir: str = option("", parse=str)
    save_freq: int = option(50, parse=int, help="Checkpoint frequency when --keep_checkpoints is enabled.")
    keep_checkpoints: bool = option(
        style="flag",
        help="Persist epoch/last checkpoints. By default checkpoints are temporary and only support linear probing.",
    )
    checkpoint_keep_count: int = option(
        2, parse=int, help="Number of most recent epoch checkpoints to retain while training.",
    )
    delete_checkpoints_after_training: bool = option(
        True, style="switch", help="Delete epoch checkpoint files after successful training while preserving last.pth.",
    )
    delete_epoch_checkpoints_after_training: bool = option(
        False, style="switch",
        help="Delete only epoch checkpoint files after training; preserve probe tensors and the final probe result.",
    )
    retain_probe_artifacts_every: int = option(
        0, parse=int,
        help="Additionally retain bulky probe tensors every N SSL epochs; 0 retains only the final tensor.",
    )
    resume: str = option("", parse=str)


@dataclass(frozen=True)
class ProbeOptions:
    train_set_linear_layer: str = option("ds_train", parse=str, choices=LINEAR_TRAIN_SPLITS)
    linear_eval_split: str | None = option(
        None, parse=str, choices=("val", "test"),
        help="Linear-probe evaluation split. Defaults to val; test requires --final_test.",
    )
    final_test: bool = option(
        style="flag", help="Evaluate a locked final configuration on test instead of the validation default.",
    )
    linear_probe_mode: str | None = option(
        None, parse=str, choices=("final", "periodic", "none"),
        help="Defaults to periodic on val. --final_test restricts evaluation to one final probe.",
    )
    linear_probe_epochs: int = option(100, parse=int)
    linear_probe_solver: str = option("logistic", choices=("logistic", "sgd"))
    linear_probe_l2: float = option(1e-3, parse=float)
    linear_probe_tolerance: float = option(1e-6, parse=float)
    linear_probe_max_epochs: int = option(200, parse=int)
    linear_probe_freq: int | None = option(
        None, parse=int,
        help="Run periodic linear evaluation every N SSL epochs (default: 25, independent of save_freq).",
    )
    linear_learning_rate: float = option(1.0, parse=float)
    linear_lr_decay_epochs: Any = option(
        "auto", parse=str,
        help="Comma-separated probe LR milestones, or 'auto' for 60%%, 75%% and 90%% of probe epochs.",
    )
    linear_lr_decay_rate: float = option(0.2, parse=float)
    linear_weight_decay: float = option(0.0, parse=float)
    linear_spurious_probe: bool = option(
        True, style="switch", help="Log an auxiliary linear probe for residual spurious-attribute predictability.",
    )


@dataclass(frozen=True)
class TrackingOptions:
    use_wandb: bool = option(
        True, style="toggle",
        help="Log the run to Weights & Biases (default). --no-use_wandb disables it for local checks and tests.",
    )
    wandb_name: str = option("CoSpRo", help="W&B project; every run lands in wandb.ai/<entity>/CoSpRo unless overridden.")
    wandb_run_name: str = option(
        "", help="Optional concise W&B display name. Checkpoint directories retain the full reproducibility name.",
    )
    entity: str = option(default_factory=wandb_entity)
    wandb_group: str = option("")
    wandb_tags: str = option("", help="Comma-separated W&B tags.")


@dataclass(frozen=True)
class MethodOptions:
    splice_mode: str = option("none", parse=str, choices=TRAINING_MODES)
    splice_weight: float = option(0.0, parse=float)


@dataclass(frozen=True)
class CoSpRoOptions:
    cospro_teacher_graph: str = option(
        "", flags=("--cospro_teacher_graph", "--crp_teacher_graph"), parse=str,
        help="Label-free CoSpRo teacher graph used by the relational graph mode.",
    )
    cospro_temperature: float = option(
        0.1, flags=("--cospro_temperature", "--crp_temperature"), parse=float,
        help="Temperature of the student relation distribution.",
    )
    cospro_start_epoch: int = option(
        10, flags=("--cospro_start_epoch", "--crp_start_epoch"), parse=int,
        help="Number of pure-SimCLR epochs before relational distillation starts.",
    )
    cospro_warmup_epochs: int = option(
        10, flags=("--cospro_warmup_epochs", "--crp_warmup_epochs"), parse=int,
        help="Linear warm-up duration for the CoSpRo relational loss weight; 0 disables warm-up.",
    )
    cospro_decay_start_epoch: int = option(
        0, flags=("--cospro_decay_start_epoch", "--crp_decay_start_epoch"), parse=int,
        help="Epoch at which relational-loss decay begins; 0 with end=0 disables decay.",
    )
    cospro_decay_end_epoch: int = option(
        0, flags=("--cospro_decay_end_epoch", "--crp_decay_end_epoch"), parse=int,
        help="Epoch at which relational-loss weight reaches zero.",
    )


@dataclass(frozen=True)
class ConceptTransferOptions:
    concept_transfer_targets: str = option("")
    concept_transfer_target_kind: str = option(
        "reconstruction", choices=("raw", "reconstruction", "shuffled_reconstruction"),
    )
    concept_transfer_alpha_max: float = option(0.1, parse=float)
    concept_transfer_start_epoch: int = option(10, parse=int)
    concept_transfer_warmup_epochs: int = option(10, parse=int)


@dataclass(frozen=True)
class LaSSLOptions:
    la_ssl: bool = option(style="flag")
    la_ssl_eta: float = option(0.1, parse=float)
    la_ssl_gamma: float = option(10.0, parse=float)
    la_ssl_quantile: float = option(0.1, parse=float)
    la_ssl_warmup_epochs: int = option(10, parse=int)
    la_ssl_update_freq: int = option(2, parse=int)


@dataclass(frozen=True)
class LateTVGOptions:
    latetvg_prune_rate: float = option(
        0.0, parse=float,
        help="LateTVG: fraction of smallest-magnitude weights pruned in the second view's encoder; 0 disables it.",
    )
    latetvg_layers: int = option(
        5, parse=int, help="LateTVG: number of final encoder convolutions the magnitude pruning covers.",
    )


@dataclass(frozen=True)
class ConceptFactorOptions:
    factor_concept_groups: str = option(
        "", help="Concept groups defining the factors; default: the single concept_groups.json under "
        "outputs/shared/<dataset>/graphs/concept_groups/.",
    )
    factor_splice_cache: str = option(
        "", help="SpLiCE dataset cache of those groups; default: found under <scratch>/features/Spur_SpLiCE/<dataset>/.",
    )
    factor_min_frequency: float = option(0.02, parse=float, help="Least fraction of images a factor must appear in.")
    factor_max_frequency: float = option(0.9, parse=float, help="Largest fraction of images a factor may appear in.")
    factor_max_count: int = option(64, parse=int, help="Most factors kept, the most balanced first.")
    factor_condition_pairs: int = option(8, parse=int, help="Entangled factor pairs whose factors condition batches.")
    factor_min_correlation: float = option(
        0.2, parse=float, help="Least presence correlation (phi) for two factors to form an entangled pair.",
    )
    factor_max_text_similarity: float = option(
        0.75, parse=float, help="Largest text similarity of a pair; higher pairs are near-synonyms.",
    )
    factor_condition_fraction: float = option(
        0.0, parse=float, help="F1: fraction of batches drawn from the images that show one factor; 0 disables F1.",
    )
    factor_distill_weight: float = option(
        0.0, parse=float, help="F2: weight of the linear factor-distillation loss; 0 disables F2.",
    )
    factor_targets: str = option(
        "whitened", choices=("whitened", "standardized"),
        help="F2 targets: ZCA-whitened factors (decorrelated) or standardized factors (ablation).",
    )
    factor_whitening_eps: float = option(0.1, parse=float, help="Ridge of the ZCA whitening.")
    factor_start_epoch: int = option(10, parse=int, help="Pure-SimCLR epochs before F2 starts.")
    factor_warmup_epochs: int = option(10, parse=int, help="Linear warm-up of the F2 weight; 0 disables it.")


# (section, argparse group title) in --help order.
TRAINING_SECTIONS: tuple[tuple[type, str], ...] = (
    (LoggingOptions, "logging"),
    (OptimizationOptions, "optimization"),
    (DataOptions, "data"),
    (ModelOptions, "model"),
    (SSLOptions, "self-supervised objective"),
    (RuntimeOptions, "runtime and reproducibility"),
    (RunIdentityOptions, "run identity"),
    (StorageOptions, "checkpoints and storage"),
    (ProbeOptions, "linear evaluation"),
    (TrackingOptions, "experiment tracking"),
    (MethodOptions, "training method"),
    (CoSpRoOptions, "CoSpRo relational distillation"),
    (ConceptTransferOptions, "frozen concept transfer"),
    (LaSSLOptions, "LA-SSL"),
    (LateTVGOptions, "LateTVG late-layer pruned view"),
    (ConceptFactorOptions, "concept factors (F1 conditioned batches, F2 factor distillation)"),
)

# Standalone linear-probe option -> trainer ProbeOptions field it shares its default with.
_PROBE_OPTION_SOURCES = {
    "train_set_linear_layer": "train_set_linear_layer",
    "epochs": "linear_probe_epochs",
    "probe_solver": "linear_probe_solver",
    "probe_l2": "linear_probe_l2",
    "probe_tolerance": "linear_probe_tolerance",
    "probe_max_epochs": "linear_probe_max_epochs",
    "learning_rate": "linear_learning_rate",
    "lr_decay_epochs": "linear_lr_decay_epochs",
    "lr_decay_rate": "linear_lr_decay_rate",
    "weight_decay": "linear_weight_decay",
    "spurious_probe": "linear_spurious_probe",
}
PROBE_MOMENTUM = 0.9
LINEAR_PROBE_DEFAULTS: dict[str, Any] = {
    **{name: section_defaults(ProbeOptions)[source] for name, source in _PROBE_OPTION_SOURCES.items()},
    "momentum": PROBE_MOMENTUM,
}


def training_defaults() -> dict[str, Any]:
    """Default of every training option, keyed by its destination."""

    defaults: dict[str, Any] = {}
    for section, _ in TRAINING_SECTIONS:
        defaults.update(section_defaults(section))
    return defaults


def build_training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Spur_SpLiCE SimCLR SSL training")
    parser.add_argument(
        "--preset", choices=sorted(PRESETS),
        help="Apply a named set of option defaults (see cospro/config/presets.py); explicit options win.",
    )
    for section, title in TRAINING_SECTIONS:
        add_section_arguments(parser.add_argument_group(title), section)
    return parser


def parse_training_arguments(parser: argparse.ArgumentParser, argv: list[str] | None) -> argparse.Namespace:
    """Parse ``argv`` with an optional ``--preset`` applied beneath the explicit options."""

    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset", choices=sorted(PRESETS))
    known, _ = preset_parser.parse_known_args(argv)
    parser.set_defaults(**preset_values(known.preset))
    namespace = parser.parse_args(argv)
    del namespace.preset
    return namespace


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def normalize_training_options(args: argparse.Namespace) -> argparse.Namespace:
    """Validate ``args`` and fill every derived value that needs no filesystem access.

    Checks run in their historical order with their historical messages. Raises :class:`ConfigError`.
    """

    dataset = dataset_class(args.dataset)
    if args.model is None:
        args.model = dataset.default_model()
    if args.la_ssl:
        _require(args.splice_mode == "none" and args.simclr_weight == 1,
                 "LA-SSL uses the unchanged SimCLR objective without a teacher.")
        _require(0 < args.la_ssl_eta <= 1 and 0 < args.la_ssl_quantile < 1,
                 "LA-SSL requires eta in (0,1] and quantile in (0,1).")
        _require(args.la_ssl_gamma > 0 and args.la_ssl_update_freq > 0 and args.la_ssl_warmup_epochs >= 0,
                 "Invalid LA-SSL scaling or schedule.")
    if args.splice_mode == "frozen_concept_distill":
        _require(args.dataset == "waterbirds", "The restored direct-transfer protocol supports Waterbirds only.")
        _require(
            min(args.concept_transfer_alpha_max, args.concept_transfer_start_epoch, args.concept_transfer_warmup_epochs) >= 0,
            "Concept-transfer schedule values must be non-negative.",
        )
    try:
        args.linear_eval_split = resolve_evaluation_split(args.linear_eval_split, args.final_test)
        args.linear_probe_mode = resolve_probe_mode(args.linear_probe_mode, args.final_test)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    _require(args.epochs > 0, "--epochs must be positive.")
    _require(args.linear_probe_epochs > 0, "--linear_probe_epochs must be positive.")
    _require(args.simclr_weight >= 0, "--simclr_weight must be non-negative.")
    try:
        args.lr_decay_epochs = resolve_epoch_schedule(args.lr_decay_epochs, args.epochs, (0.70, 0.80, 0.90))
        args.linear_lr_decay_epochs = resolve_epoch_schedule(
            args.linear_lr_decay_epochs, args.linear_probe_epochs, (0.60, 0.75, 0.90),
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    args.use_splice = args.splice_mode != "none"
    relational = args.splice_mode in RELATIONAL_GRAPH_MODES
    _require(not relational or args.splice_weight >= 0, "--splice_weight must be non-negative for relational graph modes.")
    _require(args.simclr_weight != 0 or relational, "--simclr_weight 0 is supported only for CoSpRo relational training.")
    _require(args.simclr_weight != 0 or args.splice_weight > 0,
             "KL-only relational training requires --splice_weight to be positive.")
    _require(not relational or bool(args.cospro_teacher_graph.strip()),
             "--cospro_teacher_graph is required for CoSpRo relational training.")
    _require(args.cospro_temperature > 0, "--cospro_temperature must be positive.")
    _require(args.cospro_start_epoch >= 0 and args.cospro_warmup_epochs >= 0,
             "--cospro_start_epoch and --cospro_warmup_epochs must be non-negative.")
    _require(args.cospro_decay_start_epoch >= 0 and args.cospro_decay_end_epoch >= 0,
             "--cospro_decay_start_epoch and --cospro_decay_end_epoch must be non-negative.")
    _require(bool(args.cospro_decay_start_epoch) == bool(args.cospro_decay_end_epoch),
             "CoSpRo decay start/end must both be zero or both be set.")
    _require(not args.cospro_decay_end_epoch or args.cospro_decay_end_epoch > args.cospro_decay_start_epoch,
             "--cospro_decay_end_epoch must be greater than --cospro_decay_start_epoch.")
    _require(0 <= args.latetvg_prune_rate < 1, "--latetvg_prune_rate must lie in [0, 1).")
    _require(args.latetvg_layers >= 1, "--latetvg_layers must be positive.")
    _require(not args.latetvg_prune_rate or args.simclr_weight > 0,
             "LateTVG builds its pruned view inside the SimCLR objective, so --simclr_weight must be positive.")
    _require(not (args.latetvg_prune_rate and args.la_ssl), "LA-SSL uses the unchanged SimCLR objective.")
    if args.splice_mode == "concept_factors":
        _require(args.factor_condition_fraction > 0 or args.factor_distill_weight > 0,
                 "concept_factors needs --factor_condition_fraction or --factor_distill_weight above 0.")
        _require(0 <= args.factor_condition_fraction <= 1, "--factor_condition_fraction must lie in [0, 1].")
        _require(args.factor_distill_weight >= 0, "--factor_distill_weight must be non-negative.")
        _require(0 <= args.factor_min_frequency < args.factor_max_frequency <= 1,
                 "Factor frequencies need 0 <= --factor_min_frequency < --factor_max_frequency <= 1.")
        _require(args.factor_max_count >= 2 and args.factor_condition_pairs >= 1,
                 "--factor_max_count must be at least 2 and --factor_condition_pairs at least 1.")
        _require(args.factor_whitening_eps > 0, "--factor_whitening_eps must be positive.")
        _require(args.factor_start_epoch >= 0 and args.factor_warmup_epochs >= 0,
                 "--factor_start_epoch and --factor_warmup_epochs must be non-negative.")
    _require(0 < args.ssl_crop_min <= 1, "--ssl-crop-min must be in the interval (0, 1].")
    incompatible_model = dataset.model_error(args.model)
    _require(incompatible_model is None, incompatible_model or "")
    _require(not args.cudnn_benchmark or args.cudnn_enabled, "--cudnn_benchmark true requires --cudnn_enabled true.")
    _require(not args.cudnn_benchmark, "--cudnn_benchmark must remain false because training is reproducible by default.")
    _require(args.rank_eval_freq >= 0, "--rank_eval_freq must be non-negative.")
    _require(args.linear_probe_freq is None or args.linear_probe_freq >= 0, "--linear_probe_freq must be non-negative.")
    _require(not args.keep_checkpoints or args.save_freq > 0,
             "--save_freq must be positive when --keep_checkpoints is enabled.")
    _require(1 <= args.checkpoint_keep_count <= 2, "--checkpoint_keep_count must be 1 or 2.")
    _require(args.retain_probe_artifacts_every >= 0, "--retain_probe_artifacts_every must be non-negative.")

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
    if args.linear_probe_freq is None:
        args.linear_probe_freq = DEFAULT_PERIODIC_PROBE_FREQ if args.linear_probe_mode == "periodic" else 0
    args.n_cls = dataset.num_classes
    return args


@dataclass(frozen=True)
class TrainingConfig:
    """Typed view of a resolved training configuration, one attribute per section."""

    logging: LoggingOptions
    optimization: OptimizationOptions
    data: DataOptions
    model: ModelOptions
    ssl: SSLOptions
    runtime: RuntimeOptions
    identity: RunIdentityOptions
    storage: StorageOptions
    probe: ProbeOptions
    tracking: TrackingOptions
    method: MethodOptions
    cospro: CoSpRoOptions
    concept_transfer: ConceptTransferOptions
    la_ssl: LaSSLOptions
    latetvg: LateTVGOptions
    concept_factors: ConceptFactorOptions

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "TrainingConfig":
        return cls(**{
            item.name: section_from_namespace(section, namespace)
            for item, (section, _) in zip(fields(cls), TRAINING_SECTIONS)
        })

    def flat(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for item in fields(self):
            section = getattr(self, item.name)
            values.update({entry.name: getattr(section, entry.name) for entry in fields(section)})
        return values
