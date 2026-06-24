from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from experiments.spurious_eval import linear_probe
from experiments.spurious_eval.datasets.registry import DATASET_REGISTRY
from experiments.spurious_eval.losses.contrastive import SimCLRLoss
from experiments.spurious_eval.models.simclr import SimCLRModel
from experiments.spurious_eval.training.checkpointing import load_checkpoint, save_checkpoint
from experiments.spurious_eval.training.optim import adjust_learning_rate, build_optimizer
from experiments.spurious_eval.training.ssl_loop import log_rank_metrics, train_one_epoch
from splice.ssl_regularization import SpliceConceptScorer, SpliceConfig, build_splice_regularizer, splice_mode_uses_scores


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def optional_bool(value: str) -> bool | None:
    value = str(value).strip().lower()
    if value in {"true", "yes", "y", "on"}:
        return True
    if value in {"", "false", "no", "n", "off", "none", "null"}:
        return False
    return None


def parse_float_tuple(value: str, expected_len: int, option_name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option_name} must be a comma-separated list of floats.") from exc
    if len(values) != expected_len:
        raise argparse.ArgumentTypeError(f"{option_name} expects {expected_len} comma-separated floats.")
    return values


def parse_optional_float_or_bool(value: str, default_value: float, option_name: str) -> float | None:
    bool_value = optional_bool(value)
    if bool_value is not None:
        return default_value if bool_value else None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option_name} expects true, false, or a float value.") from exc


def parse_optional_color_jitter(value: str) -> tuple[float, float, float, float] | None:
    bool_value = optional_bool(value)
    if bool_value is not None:
        return (0.8, 0.8, 0.8, 0.2) if bool_value else None
    values = parse_float_tuple(value, 4, "--splice_strong_color_jitter")
    return values[0], values[1], values[2], values[3]


def parse_optional_blur_sigma(value: str) -> tuple[float, float] | None:
    bool_value = optional_bool(value)
    if bool_value is not None:
        return (0.1, 2.0) if bool_value else None
    values = parse_float_tuple(value, 2, "--splice_strong_blur_sigma")
    return values[0], values[1]


def validate_probability(value: float | None, option_name: str) -> None:
    if value is not None and not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"{option_name} must be in the interval [0, 1].")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Spur_SpLiCE SimCLR SSL training")
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)

    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--lr_decay_epochs", type=str, default="700,800,900")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "SAM", "AdamW"])
    parser.add_argument("--sam_base_optimizer", type=str, default="SGD", choices=["SGD", "AdamW"])
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--sam_no_grad_norm", action="store_true")
    parser.add_argument("--only_sam_step_size", action="store_true")

    parser.add_argument("--dataset", type=str, default="waterbirds", choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--data_folder", type=str, default="./datasets")
    parser.add_argument("--model", type=str, default="resnet18_large", choices=["resnet18", "resnet18_large", "resnet50", "resnet50_large"])
    parser.add_argument("--method", type=str, default="SimCLR", choices=["SimCLR"])
    parser.add_argument("--head", type=str, default="mlp", choices=["linear", "mlp", "identity"])
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--ssl_crop_min", "--ssl-crop-min", dest="ssl_crop_min", type=float, default=0.2)

    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--trial", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default="")

    parser.add_argument("--train_set_linear_layer", type=str, default="ds_train", choices=["train", "ds_train", "us_train", "balanced_train", "val"])
    parser.add_argument("--linear_eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--linear_probe_mode", type=str, default="periodic", choices=["final", "periodic", "none"])
    parser.add_argument("--linear_probe_epochs", type=int, default=None)
    parser.add_argument("--linear_probe_freq", type=int, default=None)
    parser.add_argument("--linear_learning_rate", type=float, default=None)
    parser.add_argument("--linear_lr_decay_epochs", type=str, default="60,75,90")
    parser.add_argument("--linear_lr_decay_rate", type=float, default=0.2)
    parser.add_argument("--linear_weight_decay", type=float, default=0.0)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_name", default="Spur_SpLiCE")
    parser.add_argument("--entity", default="gsgrechkin-rptu")
    parser.add_argument("--energy_threshold", type=float, default=0.9)
    parser.add_argument("--rank_threshold", type=float, default=0.1)

    parser.add_argument("--use_splice", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--splice_mode", type=str, default="none", choices=["none", "augment", "corr_reg", "augment_corr_reg"])
    parser.add_argument("--splice_concepts", type=str, default="")
    parser.add_argument("--splice_score_threshold", type=float, default=0.01)
    parser.add_argument("--splice_score_reduction", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--splice_weight", type=float, default=0.0)
    parser.add_argument("--splice_l1_penalty", type=float, default=0.25)
    parser.add_argument("--splice_vocab", type=str, default="laion")
    parser.add_argument("--splice_vocab_size", type=int, default=10000)
    parser.add_argument("--splice_model", type=str, default="open_clip:ViT-B-32")
    parser.add_argument("--splice_batch_size", type=int, default=128)
    parser.add_argument("--splice_num_workers", type=int, default=0)
    parser.add_argument(
        "--splice_strong_crop",
        type=lambda value: parse_optional_float_or_bool(value, 0.08, "--splice_strong_crop"),
        nargs="?",
        const="true",
        default=None,
        help="Enable a stronger crop for high-score samples. Accepts true, false, or a crop min scale. True/no value uses 0.08.",
    )
    parser.add_argument(
        "--splice_strong_color_jitter",
        type=parse_optional_color_jitter,
        nargs="?",
        const="true",
        default=None,
        help="Enable stronger ColorJitter. Accepts true, false, or brightness,contrast,saturation,hue. True/no value uses 0.8,0.8,0.8,0.2.",
    )
    parser.add_argument(
        "--splice_strong_color_jitter_p",
        type=lambda value: parse_optional_float_or_bool(value, 0.9, "--splice_strong_color_jitter_p"),
        default=None,
        help="Probability for strong ColorJitter. Accepts true, false, or a probability. True uses 0.9.",
    )
    parser.add_argument(
        "--splice_strong_grayscale_p",
        type=lambda value: parse_optional_float_or_bool(value, 0.3, "--splice_strong_grayscale_p"),
        nargs="?",
        const="true",
        default=None,
        help="Enable stronger RandomGrayscale probability. Accepts true, false, or a probability. True/no value uses 0.3.",
    )
    parser.add_argument(
        "--splice_strong_blur_p",
        type=lambda value: parse_optional_float_or_bool(value, 0.5, "--splice_strong_blur_p"),
        nargs="?",
        const="true",
        default=None,
        help="Enable GaussianBlur for high-score samples. Accepts true, false, or a probability. True/no value uses 0.5.",
    )
    parser.add_argument(
        "--splice_strong_blur_kernel_size",
        type=int,
        default=None,
        help="GaussianBlur kernel size. Also enables blur with default probability if used alone.",
    )
    parser.add_argument(
        "--splice_strong_blur_sigma",
        type=parse_optional_blur_sigma,
        default=None,
        help="GaussianBlur sigma as min,max. Accepts true, false, or min,max. True uses 0.1,2.0.",
    )

    args = parser.parse_args()
    args.lr_decay_epochs = [int(epoch.strip()) for epoch in args.lr_decay_epochs.split(",") if epoch.strip()]
    args.linear_lr_decay_epochs = [int(epoch.strip()) for epoch in args.linear_lr_decay_epochs.split(",") if epoch.strip()]
    if args.use_splice and args.splice_mode == "none":
        args.splice_mode = "corr_reg"
    args.use_splice = args.splice_mode != "none"
    if args.use_splice and not args.splice_concepts.strip():
        parser.error("--splice_concepts must be provided when --splice_mode is not none.")
    if args.splice_mode in {"corr_reg", "augment_corr_reg"} and args.splice_weight <= 0:
        parser.error("--splice_weight must be positive for SpLiCE correlation regularization modes.")
    if not 0 < args.ssl_crop_min <= 1:
        parser.error("--ssl-crop-min must be in the interval (0, 1].")
    if args.splice_strong_crop is not None and not 0 < args.splice_strong_crop <= 1:
        parser.error("--splice_strong_crop must be in the interval (0, 1].")
    probability_args = {
        "--splice_strong_color_jitter_p": args.splice_strong_color_jitter_p,
        "--splice_strong_grayscale_p": args.splice_strong_grayscale_p,
        "--splice_strong_blur_p": args.splice_strong_blur_p,
    }
    for option_name, value in probability_args.items():
        try:
            validate_probability(value, option_name)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    if args.splice_strong_blur_kernel_size is not None and args.splice_strong_blur_kernel_size <= 0:
        parser.error("--splice_strong_blur_kernel_size must be positive.")
    if args.splice_strong_blur_sigma is not None and args.splice_strong_blur_sigma[0] > args.splice_strong_blur_sigma[1]:
        parser.error("--splice_strong_blur_sigma min must be <= max.")
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
        args.linear_probe_epochs = args.epochs
    if args.linear_learning_rate is None:
        args.linear_learning_rate = args.learning_rate
    if args.linear_probe_freq is None:
        args.linear_probe_freq = args.save_freq if args.linear_probe_mode == "periodic" else 0
    args.n_cls = DATASET_REGISTRY[args.dataset]["num_classes"]
    args.model_name = format_run_name(args)
    args.save_folder = str(Path(args.checkpoint_dir or f"./save/{args.method}/{args.dataset}_models") / args.model_name)
    os.makedirs(args.save_folder, exist_ok=True)
    write_run_config(args)
    return args


def format_run_name(args: argparse.Namespace) -> str:
    optimizer_name = args.optimizer
    if optimizer_name.lower() == "sam":
        optimizer_name = f"SAM{args.rho:g}-{args.sam_base_optimizer}"
    if not args.use_splice:
        splice_name = "nosplice"
    elif args.splice_mode == "augment":
        score_reduction = args.splice_score_reduction[:1].upper() + args.splice_score_reduction[1:]
        splice_name = f"augment{args.splice_score_threshold:g}_{format_strong_aug_name(args)}"
    else:
        splice_name = f"{args.splice_mode}_w{args.splice_weight:g}"
        if args.splice_mode == "augment_corr_reg":
            splice_name = f"{splice_name}_{format_strong_aug_name(args)}"
    run_name = (
        f"{args.method}_{args.dataset}_{optimizer_name}_{args.model}_{args.head}_{splice_name}_"
        f"seed{args.seed:g}_lr{args.learning_rate:g}_bs{args.batch_size}_temp{args.temp:g}"
    )
    if args.use_splice and args.splice_mode == "augment":
        run_name = f"{run_name}_score{score_reduction}"
    return run_name


def format_strong_aug_name(args: argparse.Namespace) -> str:
    parts = []
    if args.splice_strong_crop is not None:
        parts.append(f"crop{args.splice_strong_crop:g}")
    if args.splice_strong_color_jitter is not None or args.splice_strong_color_jitter_p is not None:
        jitter = args.splice_strong_color_jitter or (0.8, 0.8, 0.8, 0.2)
        jitter_values = "-".join(f"{value:g}" for value in jitter)
        probability = 0.9 if args.splice_strong_color_jitter_p is None else args.splice_strong_color_jitter_p
        parts.append(f"cj{jitter_values}p{probability:g}")
    if args.splice_strong_grayscale_p is not None:
        parts.append(f"gray{args.splice_strong_grayscale_p:g}")
    if (
        args.splice_strong_blur_p is not None
        or args.splice_strong_blur_kernel_size is not None
        or args.splice_strong_blur_sigma is not None
    ):
        probability = 0.5 if args.splice_strong_blur_p is None else args.splice_strong_blur_p
        kernel_size = 23 if args.splice_strong_blur_kernel_size is None else args.splice_strong_blur_kernel_size
        sigma = args.splice_strong_blur_sigma or (0.1, 2.0)
        parts.append(f"blur{probability:g}k{kernel_size}s{sigma[0]:g}-{sigma[1]:g}")
    return "standardAug" if not parts else "_".join(parts)


def write_run_config(args: argparse.Namespace) -> None:
    config_path = Path(args.save_folder) / "args.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, sort_keys=True)
        file.write("\n")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader_kwargs(args: argparse.Namespace, shuffle: bool) -> dict:
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "generator": loader_generator,
    }
    if shuffle or args.num_workers > 0:
        loader_kwargs["worker_init_fn"] = seed_worker
    return loader_kwargs


def build_ssl_loader(args: argparse.Namespace):
    dataset_spec = DATASET_REGISTRY[args.dataset]
    config = dataset_spec["config"](
        root_dir=args.data_folder,
        ssl_crop_min=args.ssl_crop_min,
        splice_strong_crop=args.splice_strong_crop,
        splice_strong_color_jitter=args.splice_strong_color_jitter,
        splice_strong_color_jitter_p=args.splice_strong_color_jitter_p,
        splice_strong_grayscale_p=args.splice_strong_grayscale_p,
        splice_strong_blur_p=args.splice_strong_blur_p,
        splice_strong_blur_kernel_size=args.splice_strong_blur_kernel_size,
        splice_strong_blur_sigma=args.splice_strong_blur_sigma,
    )
    loader_kwargs = make_dataloader_kwargs(args, shuffle=True)
    concept_scorer = build_splice_concept_scorer(args) if splice_mode_uses_scores(args.splice_mode) else None
    return dataset_spec["ssl_loader"](
        config,
        args.batch_size,
        concept_scorer=concept_scorer,
        splice_mode=args.splice_mode,
        splice_score_threshold=args.splice_score_threshold,
        **loader_kwargs,
    )


def build_splice_config(args: argparse.Namespace) -> SpliceConfig:
    return SpliceConfig(
        use_splice=args.use_splice,
        splice_weight=args.splice_weight,
        mode=args.splice_mode,
        concepts=args.splice_concepts,
        l1_penalty=args.splice_l1_penalty,
        vocab=args.splice_vocab,
        vocab_size=args.splice_vocab_size,
        model=args.splice_model,
        score_threshold=args.splice_score_threshold,
        score_reduction=args.splice_score_reduction,
        batch_size=args.splice_batch_size,
        num_workers=args.splice_num_workers,
        device=args.device,
    )


def build_splice_concept_scorer(args: argparse.Namespace) -> SpliceConceptScorer:
    scorer = SpliceConceptScorer(build_splice_config(args))
    print("[INFO] SpLiCE concepts:", [(idx, scorer.vocabulary[idx]) for idx in scorer.concept_indices])
    return scorer


def run_linear_probe(args: argparse.Namespace, ckpt_path: str, epoch: int) -> dict[str, float]:
    return linear_probe.main(build_linear_probe_args(args, ckpt_path), supcon_epoch=epoch)


def build_linear_probe_args(args: argparse.Namespace, ckpt_path: str) -> argparse.Namespace:
    probe_settings = {
        "dataset": args.dataset,
        "data_folder": args.data_folder,
        "train_set_linear_layer": args.train_set_linear_layer,
        "eval_split": args.linear_eval_split,
        "model": args.model,
        "ckpt": ckpt_path,
        "method": args.method,
        "head": args.head,
        "kappa": 1.0,
        "trial": args.trial,
        "augmented_features": False,
        "plot_path": "",
        "energy_threshold": args.energy_threshold,
        "rank_threshold": args.rank_threshold,
        "spur_str": 0.0,
        "num_zero_high": 0,
        "num_zero_low": 0,
        "batch_size": args.batch_size,
        "num_workers": 32,
        "epochs": 100,
        "learning_rate": 1.0,
        "lr_decay_epochs": [60, 75, 90],
        "lr_decay_rate": 0.2,
        "weight_decay": 0,
        "momentum": 0.9,
        "cosine": args.cosine,
        "seed": args.seed,
        "device": args.device,
        "use_wandb": args.use_wandb,
        "wandb_name": args.wandb_name,
        "entity": args.entity,
    }
    return argparse.Namespace(**probe_settings)


def build_training_state(args: argparse.Namespace, device: torch.device):
    train_loader = build_ssl_loader(args)
    model = SimCLRModel(name=args.model, head=args.head, feat_dim=args.feat_dim).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and device.type == "cuda":
        model.encoder = torch.nn.DataParallel(model.encoder)
    criterion = SimCLRLoss(temperature=args.temp).to(device)
    optimizer = build_optimizer(args, model)
    splice_regularizer = build_splice_regularizer(build_splice_config(args))
    return train_loader, model, criterion, optimizer, splice_regularizer


def maybe_run_periodic_probe(args: argparse.Namespace, save_file: str, epoch: int) -> bool:
    if args.linear_probe_mode != "periodic":
        return False
    if not args.linear_probe_freq or epoch % args.linear_probe_freq != 0:
        return False
    run_linear_probe(args, save_file, epoch)
    return True


def maybe_run_final_probe(args: argparse.Namespace, save_file: str, already_probed_epoch: int) -> None:
    if args.linear_probe_mode == "none":
        return
    if already_probed_epoch == args.epochs:
        return
    run_linear_probe(args, save_file, args.epochs)


def cleanup_default_checkpoints(args: argparse.Namespace) -> None:
    if args.checkpoint_dir:
        return
    save_folder = Path(args.save_folder)
    checkpoint_paths = sorted(save_folder.glob("*.pth"))
    for checkpoint_path in checkpoint_paths:
        checkpoint_path.unlink()
    if checkpoint_paths:
        print(f"[INFO] Removed {len(checkpoint_paths)} checkpoint file(s) from {save_folder}")


def main() -> None:
    args = parse_args()
    print(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    args.device = str(device)

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_name, name=args.model_name, config=vars(args), entity=args.entity)

    train_loader, model, criterion, optimizer, splice_regularizer = build_training_state(args, device)
    start_epoch = load_checkpoint(model, optimizer, args.resume, device) + 1 if args.resume else 1
    if device.type == "cuda":
        cudnn.benchmark = False
        cudnn.enabled = False

    last_probe_epoch = 0
    for epoch in range(start_epoch, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        time1 = time.time()
        train_metrics = train_one_epoch(train_loader, model, criterion, optimizer, epoch, args, splice_regularizer)
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        if epoch % args.print_freq == 0:
            log_rank_metrics(model, train_loader, optimizer, train_metrics, epoch, args, wandb_run)

        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.save_folder, f"ckpt_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, args, epoch, save_file)
            if maybe_run_periodic_probe(args, save_file, epoch):
                last_probe_epoch = epoch

    save_file = os.path.join(args.save_folder, "last.pth")
    save_checkpoint(model, optimizer, args, args.epochs, save_file)
    maybe_run_final_probe(args, save_file, last_probe_epoch)
    cleanup_default_checkpoints(args)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
