from __future__ import annotations

import math

import numpy as np
import torch


def build_optimizer(args, model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer_name = args.optimizer.lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer '{args.optimizer}'")


def adjust_learning_rate(args, optimizer: torch.optim.Optimizer, epoch: int) -> None:
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


def warmup_learning_rate(args, epoch: int, batch_id: int, total_batches: int, optimizer: torch.optim.Optimizer) -> None:
    if not args.warm or epoch > args.warm_epochs:
        return
    progress = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
    lr = args.warmup_from + progress * (args.warmup_to - args.warmup_from)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
