from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from experiments.spurious_eval.datasets.registry import DATASET_REGISTRY
from experiments.spurious_eval.metrics import entropy_effective_rank
from experiments.spurious_eval.models.resnet import LinearClassifier, build_resnet_encoder
from experiments.spurious_eval.training.checkpointing import load_encoder_checkpoint
from experiments.spurious_eval.training.probe_loop import extract_features, make_feature_loader, train_one_epoch, validate


@dataclass
class ProbeHistory:
    val_accuracy: list[float]
    val_worst_group: list[float]
    val_best_group: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Linear probing on spurious-correlation datasets")
    parser.add_argument("--dataset", default="waterbirds", choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--data_folder", default="./datasets")
    parser.add_argument("--train_set_linear_layer", default="ds_train", choices=["train", "val", "ds_train", "us_train", "balanced_train"])
    parser.add_argument("--eval_split", default="val", choices=["val", "test"])
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet18_large", "resnet50_large"])
    parser.add_argument("--ckpt", default="", help="SpurSSL checkpoint containing encoder.* weights")
    parser.add_argument("--method", default="SimCLR", help="Accepted for SpurSSL command compatibility")
    parser.add_argument("--head", default="mlp", choices=["mlp", "linear", "fixed", "identity"], help="Accepted for SpurSSL command compatibility")
    parser.add_argument("--kappa", type=float, default=1.0, help="Accepted for SpurSSL command compatibility")
    parser.add_argument("--trial", default="0", help="Accepted for SpurSSL command compatibility")
    parser.add_argument("--augmented_features", action="store_true", help="Accepted for SpurSSL command compatibility")
    parser.add_argument("--plot_path", default="", help="Accepted for SpurSSL command compatibility")
    parser.add_argument("--energy_threshold", type=float, default=0.9, help="Accepted for SpurSSL command compatibility")
    parser.add_argument("--rank_threshold", type=float, default=0.1, help="Accepted for SpurSSL command compatibility")
    parser.add_argument("--spur_str", type=float, default=0.0, help="Accepted for SpurSSL command compatibility")
    parser.add_argument("--num_zero_high", type=int, default=0, help="Accepted for SpurSSL command compatibility")
    parser.add_argument("--num_zero_low", type=int, default=0, help="Accepted for SpurSSL command compatibility")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--lr_decay_epochs", default="60,75,90")
    parser.add_argument("--lr_decay_rate", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_name", default="Spur_SpLiCE")
    parser.add_argument("--entity", default="gsgrechkin-rptu")
    args = parser.parse_args()
    args.lr_decay_epochs = [int(epoch.strip()) for epoch in args.lr_decay_epochs.split(",") if epoch.strip()]
    return args


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "dataset": "waterbirds",
        "data_folder": "./datasets",
        "train_set_linear_layer": "ds_train",
        "eval_split": "val",
        "model": "resnet18",
        "ckpt": "",
        "method": "SimCLR",
        "head": "mlp",
        "kappa": 1.0,
        "batch_size": 256,
        "num_workers": 32,
        "epochs": 100,
        "learning_rate": 1.0,
        "lr_decay_epochs": [60, 75, 90],
        "lr_decay_rate": 0.2,
        "weight_decay": 0.0,
        "momentum": 0.9,
        "cosine": False,
        "seed": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "use_wandb": False,
        "wandb_name": "Spur_SpLiCE",
        "entity": "gsgrechkin-rptu",
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if isinstance(args.lr_decay_epochs, str):
        args.lr_decay_epochs = [int(epoch.strip()) for epoch in args.lr_decay_epochs.split(",") if epoch.strip()]
    return args


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def adjust_learning_rate(args: argparse.Namespace, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate**3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate**steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def consume_spurssl_head_rng(feature_dim: int, args: argparse.Namespace) -> None:
    """Instantiate the unused SpurSSL projection head to preserve classifier RNG state."""

    if args.method != "SimCLR":
        return
    if args.head == "linear":
        torch.nn.Linear(feature_dim, 128)
    elif args.head == "mlp":
        torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 128),
        )
    elif args.head in {"identity", "fixed"}:
        return
    else:
        raise ValueError(f"Unsupported SpurSSL head: {args.head}")


def main(args: argparse.Namespace | None = None, supcon_epoch: int = 0) -> dict[str, float]:
    args = parse_args() if args is None else normalize_args(args)
    set_seed(args.seed)
    device = torch.device(args.device)

    dataset_spec = DATASET_REGISTRY[args.dataset]
    config = dataset_spec["config"](
        root_dir=args.data_folder,
        train_split=args.train_set_linear_layer,
        eval_split=args.eval_split,
    )
    train_loader_kwargs = make_dataloader_kwargs(args, shuffle=True)
    val_loader_kwargs = make_dataloader_kwargs(args, shuffle=False)
    train_loader, val_loader = dataset_spec["probe_loaders"](
        config,
        args.batch_size,
        train_loader_kwargs=train_loader_kwargs,
        eval_loader_kwargs=val_loader_kwargs,
    )

    encoder, feature_dim = build_resnet_encoder(args.model)
    if args.ckpt:
        print(f"[INFO] Loading encoder checkpoint from {args.ckpt}")
        load_encoder_checkpoint(encoder, args.ckpt)
    else:
        print("[INFO] No checkpoint provided. Using randomly initialized frozen encoder.")

    encoder = encoder.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if device.type == "cuda":
        cudnn.benchmark = False

    consume_spurssl_head_rng(feature_dim, args)
    classifier = LinearClassifier(feature_dim=feature_dim, num_classes=dataset_spec["num_classes"]).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    print("[INFO] Extracting frozen train features")
    train_features = extract_features(encoder, train_loader, device)
    print("[INFO] Extracting frozen validation features")
    val_features = extract_features(encoder, val_loader, device)
    feature_loader = make_feature_loader(train_features, args.batch_size, args.seed, shuffle=True)
    val_feature_loader = make_feature_loader(val_features, args.batch_size, args.seed, shuffle=False)

    history = ProbeHistory([], [], [])
    best_val_acc = best_val_wg_acc = best_val_bg_acc = 0.0
    best_train_acc = best_train_wg_acc = best_train_bg_acc = 0.0

    wandb_run = None
    created_wandb_run = False
    if args.use_wandb:
        import wandb

        wandb_run = wandb.run
        if wandb_run is None:
            wandb_run = wandb.init(
                project=args.wandb_name,
                entity=args.entity,
                config=vars(args),
                name=f"linear_{args.dataset}_{args.model}_{args.seed}",
            )
            created_wandb_run = True

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        start = time.time()
        train_loss, train_acc, train_pred, train_labels, train_metadata = train_one_epoch(
            feature_loader, classifier, criterion, optimizer, device
        )
        train_results, _ = train_loader.dataset.eval(train_pred, train_labels, train_metadata)
        train_wg_acc = train_results["acc_wg"] * 100
        train_bg_acc = train_results["best_acc"] * 100
        print(
            "Train epoch {}, total time {:.2f}, loss {:.4f}, accuracy {:.2f}, wg accuracy {:.2f}, bg accuracy {:.2f}".format(
                epoch, time.time() - start, train_loss, train_acc, train_wg_acc, train_bg_acc
            )
        )

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_train_wg_acc = train_wg_acc
            best_train_bg_acc = train_bg_acc

        val_loss, val_acc, val_pred, val_labels, val_metadata = validate(
            val_feature_loader, classifier, criterion, device
        )
        val_results, _ = val_loader.dataset.eval(val_pred, val_labels, val_metadata)
        val_wg_acc = val_results["acc_wg"] * 100
        val_bg_acc = val_results["best_acc"] * 100
        print(
            "Val epoch {}, loss {:.4f}, accuracy {:.2f}, wg accuracy {:.2f}, bg accuracy {:.2f}".format(
                epoch, val_loss, val_acc, val_wg_acc, val_bg_acc
            )
        )

        history.val_accuracy.append(val_acc)
        history.val_worst_group.append(val_wg_acc)
        history.val_best_group.append(val_bg_acc)

        if val_acc > best_val_acc or (
            val_acc == best_val_acc and (val_wg_acc, val_bg_acc) > (best_val_wg_acc, best_val_bg_acc)
        ):
            best_val_acc = val_acc
            best_val_wg_acc = val_wg_acc
            best_val_bg_acc = val_bg_acc

    last_acc = history.val_accuracy[-1]
    last_wg_acc = history.val_worst_group[-1]
    last_bg_acc = history.val_best_group[-1]
    window = min(10, len(history.val_accuracy))
    avg_last_10_acc = float(np.mean(history.val_accuracy[-window:]))
    avg_last_10_wg_acc = float(np.mean(history.val_worst_group[-window:]))
    avg_last_10_bg_acc = float(np.mean(history.val_best_group[-window:]))
    group_counts = val_results["group_counts"]
    group_accuracies = val_results["group_accuracy"]
    nonempty_group_ids = torch.where(group_counts > 0)[0]
    if len(nonempty_group_ids):
        worst_offset = torch.argmin(group_accuracies[nonempty_group_ids])
        last_worst_group_id = int(nonempty_group_ids[worst_offset].item())
        last_worst_group_count = int(group_counts[last_worst_group_id].item())
    else:
        last_worst_group_id = -1
        last_worst_group_count = 0
    print(
        "Average of last 10 accuracies: {:.2f}, Average of last 10 worst-group accuracies: {:.2f}, Average of last 10 best-group accuracies: {:.2f}".format(
            avg_last_10_acc, avg_last_10_wg_acc, avg_last_10_bg_acc
        )
    )

    train_feature_tensor = train_features.tensors[0]
    val_feature_tensor = val_features.tensors[0]
    entropy, effective_rank, energy_based_rank = entropy_effective_rank(train_feature_tensor)
    val_entropy, val_effective_rank, val_energy_based_rank = entropy_effective_rank(val_feature_tensor)

    print(f"Train - Entropy: {entropy:.4f}, Effective Rank: {effective_rank:.2f}, Energy-Based Rank: {energy_based_rank:.2f}")
    print(f"Val   - Entropy: {val_entropy:.4f}, Effective Rankuse_wandb: {val_effective_rank:.2f}, Energy-Based Rank: {val_energy_based_rank:.2f}")

    final_metrics = {
        "Linear train acc": best_train_acc,
        "Linear train worst-group acc": best_train_wg_acc,
        "Linear train best-group acc": best_train_bg_acc,
        "Linear val acc": best_val_acc,
        "Linear val worst-group acc": best_val_wg_acc,
        "Linear val best-group acc": best_val_bg_acc,
        "Train linear entropy": entropy,
        "Train linear effective rank": effective_rank,
        "Train linear energy-based rank": energy_based_rank,
        "Val linear entropy": val_entropy,
        "Val linear effective rank": val_effective_rank,
        "Val linear energy-based rank": val_energy_based_rank,
        "Last linear val acc": last_acc,
        "Last linear val worst-group acc": last_wg_acc,
        "Last linear val best-group acc": last_bg_acc,
        "Average over 10 last linear val acc": avg_last_10_acc,
        "Average over last 10 linear val worst-group acc": avg_last_10_wg_acc,
        "Average over last 10 linear val best-group acc": avg_last_10_bg_acc,
        "Last linear val worst-group id": last_worst_group_id,
        "Last linear val worst-group count": last_worst_group_count,
    }
    if wandb_run is not None:
        wandb_run.log(final_metrics, step=supcon_epoch)
        if created_wandb_run:
            wandb_run.finish()

    print(
        "best accuracy: {:.2f} and worst-group accuracy: {:.2f} and best-group accuracy: {:.2f}".format(
            best_val_acc, best_val_wg_acc, best_val_bg_acc
        )
    )
    print(
        "Last accuracy: {:.2f}, Last worst-group accuracy: {:.2f}, Last best-group accuracy: {:.2f}".format(
            last_acc, last_wg_acc, last_bg_acc
        )
    )
    print("Train entropy: {:.2f}, effective rank: {}, and energy-based rank: {}".format(entropy, effective_rank, energy_based_rank))
    print("Val entropy: {:.2f}, effective rank: {}, and energy-based rank: {}".format(val_entropy, val_effective_rank, val_energy_based_rank))
    print(
        "Average last 10 accuracies: {:.2f}, Average last 10 worst-group accuracies: {:.2f}, Average last 10 best-group accuracies: {:.2f}".format(
            avg_last_10_acc, avg_last_10_wg_acc, avg_last_10_bg_acc
        )
    )
    return final_metrics


if __name__ == "__main__":
    main()
