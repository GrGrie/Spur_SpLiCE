from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

from experiments.spurious_eval.losses.contrastive import SimCLRLoss
from experiments.spurious_eval.metrics import entropy_effective_rank
from experiments.spurious_eval.models.simclr import SimCLRModel
from experiments.spurious_eval.training.optim import warmup_learning_rate


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


def simclr_forward_loss(
    model: SimCLRModel,
    criterion: SimCLRLoss,
    image,
    splice_concepts=None,
    targets=None,
    splice_regularizer=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    bsz = image[0].size(0)
    images = torch.cat([image[0], image[1]], dim=0)
    embeddings = model.encoder(images)
    projections = F.normalize(model.head(embeddings), dim=1)
    f1, f2 = torch.split(projections, [bsz, bsz], dim=0)
    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    loss, decor_loss, entropy_loss, _, _ = criterion(features)
    splice_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
    if splice_regularizer is not None:
        repeated_concepts = None
        repeated_targets = None
        if splice_concepts is not None:
            repeated_concepts = torch.cat([splice_concepts, splice_concepts], dim=0)
        if targets is not None:
            repeated_targets = torch.cat([targets, targets], dim=0)
        splice_loss = splice_regularizer(embeddings, repeated_concepts, repeated_targets)
        loss = loss + splice_loss
    parts = {
        "decor": decor_loss,
        "entropy": entropy_loss,
        "splice": splice_loss,
    }
    return loss, parts, bsz


def train_one_epoch(
    train_loader, model, criterion, optimizer, scaler, epoch: int, args, splice_regularizer
) -> dict[str, float]:
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
        if args.channels_last and str(args.device).startswith("cuda"):
            image[0] = image[0].contiguous(memory_format=torch.channels_last)
            image[1] = image[1].contiguous(memory_format=torch.channels_last)
        targets = data[1].to(args.device, non_blocking=True)
        splice_concepts = data[3].to(args.device, non_blocking=True) if len(data) > 3 else None
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=args.amp and str(args.device).startswith("cuda"),
        ):
            loss, parts, bsz = simclr_forward_loss(
                model,
                criterion,
                image,
                splice_concepts,
                targets,
                splice_regularizer,
            )
        losses.update(loss.item(), bsz)
        decor_losses.update(parts["decor"].item(), bsz)
        entropy_losses.update(parts["entropy"].item(), bsz)
        splice_losses.update(parts["splice"].item(), bsz)

        if args.optimizer == "SAM":
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step()
            loss, _, _ = simclr_forward_loss(
                model,
                criterion,
                image,
                splice_concepts,
                targets,
                splice_regularizer,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.second_step()
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % args.print_freq == 0:
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "splice {splice.val:.3f} ({splice.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    splice=splice_losses,
                )
            )
            sys.stdout.flush()

    return {
        "loss": losses.avg,
        "decor_loss": decor_losses.avg,
        "entropy_loss": entropy_losses.avg,
        "splice_loss": splice_losses.avg,
    }


def extract_normalized_train_features(model: SimCLRModel, rank_loader, args) -> torch.Tensor:
    was_training = model.training
    model.eval()
    features = []
    try:
        with torch.no_grad():
            for data in rank_loader:
                images = data[0].to(args.device, non_blocking=True)
                if args.channels_last and str(args.device).startswith("cuda"):
                    images = images.contiguous(memory_format=torch.channels_last)
                embeddings = model.encoder(images)
                features.append(embeddings.cpu())
    finally:
        model.train(was_training)
    features = F.normalize(torch.cat(features, dim=0), dim=1)
    print("Extracted features shape:", features.shape)
    return features


def log_rank_metrics(
    model: SimCLRModel,
    rank_loader,
    optimizer: torch.optim.Optimizer,
    train_metrics: dict[str, float],
    epoch: int,
    args,
    wandb_run,
    compute_rank: bool = True,
) -> None:
    rank_metrics = {}
    if compute_rank:
        if rank_loader is None:
            raise ValueError("Rank metrics require a dedicated rank loader.")
        train_features = extract_normalized_train_features(model, rank_loader, args)
        entropy, effective_rank, energy_based_rank = entropy_effective_rank(train_features)
        print(
            "epoch {}, entropy {:.2f}, effective rank {}, and energy-based rank {}".format(
                epoch, entropy, effective_rank, energy_based_rank
            )
        )
        rank_metrics = {
            "Entropy": entropy,
            "Effective rank": effective_rank,
            "Energy-based rank": energy_based_rank,
        }
    if wandb_run is not None:
        wandb_run.log(
            {
                **rank_metrics,
                "SSL train loss": train_metrics["loss"],
                "SSL decor loss": train_metrics["decor_loss"],
                "SSL entropy loss": train_metrics["entropy_loss"],
                "SSL splice loss": train_metrics["splice_loss"],
                "SSL learning rate": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )
