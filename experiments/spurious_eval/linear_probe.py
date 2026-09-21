"""Command line for the linear probe.

The measurement itself lives in :mod:`cospro.evaluation.probe`; this module resolves the command
line into ``ProbeOptions`` and ``ProbeArtifacts`` and dispatches. Programmatic callers build those
two objects directly instead of assembling a namespace.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cospro.config import LINEAR_PROBE_DEFAULTS, training_defaults
from cospro.evaluation import ProbeArtifacts, ProbeOptions, probe_checkpoint, seed_probe
from experiments.spurious_eval.datasets.registry import canonical_dataset_name, dataset_class, dataset_names
from experiments.spurious_eval.evaluation_protocol import resolve_evaluation_split
from experiments.spurious_eval.models.resnet import RESNET_MODEL_NAMES
from cospro.config.settings import wandb_entity


def resolve_lr_decay_epochs(value: str | list[int], total_epochs: int) -> list[int]:
    if not isinstance(value, str):
        return list(value)
    if value.strip().lower() == "auto":
        return sorted(
            {
                int(round(total_epochs * fraction))
                for fraction in (0.60, 0.75, 0.90)
                if 0 < int(round(total_epochs * fraction)) < total_epochs
            }
        )
    milestones = [int(epoch.strip()) for epoch in value.split(",") if epoch.strip()]
    if milestones != sorted(set(milestones)):
        raise ValueError(f"LR milestones must be unique and increasing; got {milestones}.")
    if any(epoch <= 0 or epoch >= total_epochs for epoch in milestones):
        raise ValueError(f"LR milestones must be between 1 and {total_epochs - 1}; got {milestones}.")
    return milestones


def build_parser() -> argparse.ArgumentParser:
    """Standalone probe options; shared defaults come from cospro.config.LINEAR_PROBE_DEFAULTS."""

    defaults = LINEAR_PROBE_DEFAULTS
    parser = argparse.ArgumentParser("Linear probing on spurious-correlation datasets")
    parser.add_argument(
        "--dataset",
        type=canonical_dataset_name,
        default="waterbirds",
        choices=dataset_names(),
    )
    parser.add_argument("--data_folder", default="./datasets")
    parser.add_argument(
        "--train_set_linear_layer", default=defaults["train_set_linear_layer"],
        choices=["train", "val", "ds_train", "us_train", "balanced_train"],
    )
    parser.add_argument(
        "--eval_split",
        default=None,
        choices=["val", "test"],
        help="Evaluation split. Defaults to val; test requires --final_test.",
    )
    parser.add_argument(
        "--final_test",
        action="store_true",
        help="Evaluate a locked final configuration on test instead of the validation default.",
    )
    parser.add_argument("--model", default="resnet18_large", choices=RESNET_MODEL_NAMES)
    parser.add_argument("--ckpt", default="", help="SpurSSL checkpoint containing encoder.* weights")
    parser.add_argument(
        "--artifact_dir",
        default="",
        help="Directory for downstream probe artifacts; defaults to the checkpoint directory.",
    )
    parser.add_argument("--head", default="mlp", choices=["mlp", "linear", "fixed", "identity"], help="Accepted for SpurSSL command compatibility")
    parser.add_argument("--batch_size", type=int, default=training_defaults()["batch_size"])
    parser.add_argument("--num_workers", type=int, default=training_defaults()["num_workers"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument(
        "--ssl_epoch",
        type=int,
        default=0,
        help="SSL checkpoint epoch recorded in downstream feature/result artifacts.",
    )
    parser.add_argument("--probe_solver", choices=["logistic", "sgd"], default=defaults["probe_solver"])
    parser.add_argument("--probe_l2", type=float, default=defaults["probe_l2"])
    parser.add_argument("--probe_tolerance", type=float, default=defaults["probe_tolerance"])
    parser.add_argument("--probe_max_epochs", type=int, default=defaults["probe_max_epochs"])
    parser.add_argument("--learning_rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--lr_decay_epochs", default=defaults["lr_decay_epochs"])
    parser.add_argument("--lr_decay_rate", type=float, default=defaults["lr_decay_rate"])
    parser.add_argument("--weight_decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--momentum", type=float, default=defaults["momentum"])
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_name", default=training_defaults()["wandb_name"])
    parser.add_argument("--entity", default=wandb_entity())
    parser.add_argument(
        "--spurious_probe",
        action=argparse.BooleanOptionalAction,
        default=defaults["spurious_probe"],
        help="Also measure how linearly predictable the spurious attribute remains.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.eval_split = resolve_evaluation_split(args.eval_split, args.final_test)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.lr_decay_epochs = resolve_lr_decay_epochs(args.lr_decay_epochs, args.epochs)
    except ValueError as exc:
        parser.error(str(exc))
    return args


# Options a trainer-built namespace may omit, beyond the parser defaults. --final_test only
# guards the command line, so programmatic callers pass eval_split directly.
_PROGRAMMATIC_DEFAULTS = {
    "run_recorder": None,
    "study": "adhoc",
    "arm": "linear_probe",
    "attempt_id": "standalone",
    "retain_probe_artifacts_every": 0,
    "ssl_total_epochs": 0,
}


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = vars(build_parser().parse_args([]))
    defaults.pop("final_test")
    defaults["eval_split"] = resolve_evaluation_split(None, False)
    for key, value in {**defaults, **_PROGRAMMATIC_DEFAULTS}.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.lr_decay_epochs = resolve_lr_decay_epochs(args.lr_decay_epochs, args.epochs)
    return args


def probe_options(args: argparse.Namespace) -> ProbeOptions:
    """The measurement options a resolved command line describes."""

    return ProbeOptions(
        data_folder=args.data_folder,
        train_split=args.train_set_linear_layer,
        eval_split=args.eval_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        head=args.head,
        solver=args.probe_solver,
        l2=args.probe_l2,
        tolerance=args.probe_tolerance,
        max_epochs=args.probe_max_epochs,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_decay_epochs=tuple(args.lr_decay_epochs),
        lr_decay_rate=args.lr_decay_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        cosine=args.cosine,
        spurious_probe=args.spurious_probe,
    )


def probe_artifacts(args: argparse.Namespace, ssl_epoch: int) -> ProbeArtifacts:
    """Where this probe writes, defaulting to the directory the checkpoint came from."""

    return ProbeArtifacts(
        directory=Path(args.artifact_dir) if args.artifact_dir else Path(args.ckpt).parent,
        identity={name: getattr(args, name) for name in ("study", "seed", "arm", "attempt_id")},
        ssl_epoch=ssl_epoch,
        ssl_total_epochs=int(args.ssl_total_epochs),
        retain_every=int(args.retain_probe_artifacts_every),
        recorder=args.run_recorder,
    )


def open_wandb_run(args: argparse.Namespace):
    """The active W&B run, or a probe run of its own. The second value says which."""

    if not args.use_wandb:
        return None, False
    import wandb

    if wandb.run is not None:
        return wandb.run, False
    return wandb.init(
        project=args.wandb_name,
        entity=args.entity,
        config=vars(args),
        name=f"{args.dataset}_S{args.seed}_Probe",
    ), True


def main(args: argparse.Namespace | None = None, supcon_epoch: int | None = None) -> dict[str, float]:
    args = parse_args() if args is None else normalize_args(args)
    # Trainer calls that pass supcon_epoch stay authoritative. The value stays visible to callers
    # that inspect the normalized arguments.
    args.ssl_epoch = int(args.ssl_epoch if supcon_epoch is None else supcon_epoch)
    seed_probe(args.seed)
    run, created = open_wandb_run(args)
    result = probe_checkpoint(
        dataset_class(args.dataset),
        probe_options(args),
        probe_artifacts(args, args.ssl_epoch),
        model=args.model,
        checkpoint=args.ckpt,
        wandb_run=run,
    )
    if created:
        run.finish()
    return result.metrics


if __name__ == "__main__":
    main()
