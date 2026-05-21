from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from experiments.spurious_eval import linear_probe
from experiments.spurious_eval.metrics import entropy_effective_rank
from experiments.spurious_eval.optim import adjust_learning_rate, build_optimizer, warmup_learning_rate
from experiments.spurious_eval.simclr import SimCLRLoss, SimCLRModel
from experiments.spurious_eval.splice_regularization import SpliceConfig, build_splice_regularizer
from experiments.spurious_eval.waterbirds import DATASET_REGISTRY, WaterbirdsConfig


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
    parser.add_argument("--splice_weight", type=float, default=0.0)

    args = parser.parse_args()
    args.lr_decay_epochs = [int(epoch.strip()) for epoch in args.lr_decay_epochs.split(",") if epoch.strip()]
    args.linear_lr_decay_epochs = [int(epoch.strip()) for epoch in args.linear_lr_decay_epochs.split(",") if epoch.strip()]
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
    splice_name = f"splice{args.splice_weight:g}" if args.use_splice else "nosplice"
    return (
        f"{args.method}_{args.dataset}_{optimizer_name}_{args.model}_{args.head}_{splice_name}_"
        f"seed{args.seed:g}_lr{args.learning_rate:g}_bs{args.batch_size}_temp{args.temp:g}"
    )


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_ssl_loader(args: argparse.Namespace):
    if args.dataset != "waterbirds":
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    config = WaterbirdsConfig(root_dir=args.data_folder)
    return DATASET_REGISTRY[args.dataset]["ssl_loader"](config, args.batch_size, args.num_workers)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, args: argparse.Namespace, epoch: int, path: str) -> None:
    print("==> Saving...")
    torch.save(
        {
            "opt": args,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str, device: torch.device) -> int:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


def simclr_forward_loss(model: SimCLRModel, criterion: SimCLRLoss, image, splice_regularizer=None) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    bsz = image[0].size(0)
    images = torch.cat([image[0], image[1]], dim=0)
    projections = model(images)
    f1, f2 = torch.split(projections, [bsz, bsz], dim=0)
    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    loss, decor_loss, entropy_loss, _, _ = criterion(features)
    splice_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
    if splice_regularizer is not None:
        splice_loss = splice_regularizer(projections)
        loss = loss + splice_loss
    parts = {
        "decor": decor_loss,
        "entropy": entropy_loss,
        "splice": splice_loss,
    }
    return loss, parts, bsz


def train_one_epoch(train_loader, model, criterion, optimizer, epoch: int, args: argparse.Namespace, splice_regularizer) -> dict[str, float]:
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    decor_losses = AverageMeter()
    entropy_losses = AverageMeter()
    splice_losses = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        image = data[0]
        image[0] = image[0].to(args.device, non_blocking=True)
        image[1] = image[1].to(args.device, non_blocking=True)
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        loss, parts, bsz = simclr_forward_loss(model, criterion, image, splice_regularizer)
        losses.update(loss.item(), bsz)
        decor_losses.update(parts["decor"].item(), bsz)
        entropy_losses.update(parts["entropy"].item(), bsz)
        splice_losses.update(parts["splice"].item(), bsz)

        if args.optimizer == "SAM":
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step()
            loss, _, _ = simclr_forward_loss(model, criterion, image, splice_regularizer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.second_step()
            optimizer.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % args.print_freq == 0:
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})".format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses
                )
            )
            sys.stdout.flush()

    return {
        "loss": losses.avg,
        "decor_loss": decor_losses.avg,
        "entropy_loss": entropy_losses.avg
    }


def extract_normalized_train_features(model: SimCLRModel, train_loader, args: argparse.Namespace) -> torch.Tensor:
    model.eval()
    features = []
    with torch.no_grad():
        for data in train_loader:
            image = data[0]
            images = image[0].to(args.device, non_blocking=True)
            embeddings = model.encoder(images)
            features.append(embeddings.cpu())
    features = F.normalize(torch.cat(features, dim=0), dim=1)
    print("Extracted features shape:", features.shape)
    return features


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
        "num_workers": args.num_workers,
        "epochs": args.linear_probe_epochs,
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
    }
    return argparse.Namespace(**probe_settings)


def build_training_state(args: argparse.Namespace, device: torch.device):
    train_loader = build_ssl_loader(args)
    model = SimCLRModel(name=args.model, head=args.head, feat_dim=args.feat_dim).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and device.type == "cuda":
        model.encoder = torch.nn.DataParallel(model.encoder)
    criterion = SimCLRLoss(temperature=args.temp).to(device)
    optimizer = build_optimizer(args, model)
    splice_regularizer = None
    if args.use_splice:
        splice_regularizer = build_splice_regularizer(SpliceConfig(use_splice=True, splice_weight=args.splice_weight))
    return train_loader, model, criterion, optimizer, splice_regularizer


def log_rank_metrics(model: SimCLRModel, train_loader, optimizer: torch.optim.Optimizer, train_metrics: dict[str, float], epoch: int, args: argparse.Namespace, wandb_run) -> None:
    train_features = extract_normalized_train_features(model, train_loader, args)
    entropy, effective_rank, energy_based_rank = entropy_effective_rank(train_features)
    print(
        "epoch {}, entropy {:.2f}, effective rank {}, and energy-based rank {}".format(
            epoch, entropy, effective_rank, energy_based_rank
        )
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "Entropy": entropy,
                "Effective rank": effective_rank,
                "Energy-based rank": energy_based_rank,
                "SSL train loss": train_metrics["loss"],
                "SSL decor loss": train_metrics["decor_loss"],
                "SSL entropy loss": train_metrics["entropy_loss"],
                "SSL learning rate": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )


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

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
