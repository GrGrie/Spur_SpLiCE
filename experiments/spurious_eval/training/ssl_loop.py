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
    metadata=None,
    sample_indices=None,
    simclr_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    if simclr_weight < 0:
        raise ValueError("simclr_weight must be non-negative.")
    bsz = image[0].size(0)
    images = torch.cat([image[0], image[1]], dim=0)
    embeddings = model.encoder(images)
    if simclr_weight > 0:
        projections = F.normalize(model.head(embeddings), dim=1)
        f1, f2 = torch.split(projections, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        extra_positive_mask = None
        if splice_regularizer is not None and getattr(splice_regularizer, "uses_graph_positives", False):
            if sample_indices is None:
                raise ValueError("Graph-positive contrastive learning requires graph-row sample indices.")
            extra_positive_mask = splice_regularizer.batch_positive_mask(sample_indices)
        simclr_loss, decor_loss, entropy_loss, _, _ = criterion(
            features, extra_positive_mask=extra_positive_mask
        )
        loss = simclr_weight * simclr_loss
    else:
        # Keep a differentiable zero so an unsupported relational batch is still safe to backpropagate.
        simclr_loss = embeddings.sum() * 0.0
        decor_loss = simclr_loss
        entropy_loss = simclr_loss
        loss = simclr_loss
    splice_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
    if splice_regularizer is not None:
        if getattr(splice_regularizer, "requires_crp_indices", False):
            if sample_indices is None:
                raise ValueError("CRP relational regularization requires graph-row sample indices.")
            splice_loss = splice_regularizer(embeddings, sample_indices)
            loss = loss + splice_loss
            parts = {
                "simclr": simclr_loss,
                "decor": decor_loss,
                "entropy": entropy_loss,
                "splice": splice_loss,
            }
            return loss, parts, bsz
        repeated_concepts = None
        repeated_targets = None
        if splice_concepts is not None:
            repeated_concepts = torch.cat([splice_concepts, splice_concepts], dim=0)
        if targets is not None:
            repeated_targets = torch.cat([targets, targets], dim=0)
        if getattr(splice_regularizer, "requires_oracle_metadata", False):
            if metadata is None:
                raise ValueError("Oracle relational regularization requires batch metadata.")
            repeated_targets = torch.cat([metadata, metadata], dim=0)
        regularized_embeddings = embeddings
        if getattr(splice_regularizer, "requires_clip_distillation", False):
            if model.clip_distillation_head is None:
                raise ValueError("SpLiCE synthesis distillation requires a g_clip head.")
            regularized_embeddings = model.clip_distillation_head(embeddings)
        splice_loss = splice_regularizer(regularized_embeddings, repeated_concepts, repeated_targets)
        loss = loss + splice_loss
    parts = {
        "simclr": simclr_loss,
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
    simclr_losses = AverageMeter()
    decor_losses = AverageMeter()
    entropy_losses = AverageMeter()
    splice_losses = AverageMeter()
    relational_diagnostics = {
        "scheduled_weight": AverageMeter(),
        "supported_anchor_fraction": AverageMeter(),
        "mean_anchor_confidence": AverageMeter(),
        "unweighted_kl": AverageMeter(),
        "confidence_weighted_kl": AverageMeter(),
    }

    if hasattr(splice_regularizer, "set_epoch"):
        splice_regularizer.set_epoch(epoch)

    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        image = data[0]
        image[0] = image[0].to(args.device, non_blocking=True)
        image[1] = image[1].to(args.device, non_blocking=True)
        if args.channels_last and str(args.device).startswith("cuda"):
            image[0] = image[0].contiguous(memory_format=torch.channels_last)
            image[1] = image[1].contiguous(memory_format=torch.channels_last)
        crp_training = getattr(splice_regularizer, "requires_crp_indices", False)
        targets = None if crp_training else data[1].to(args.device, non_blocking=True)
        metadata = (
            data[2].to(args.device, non_blocking=True)
            if getattr(splice_regularizer, "requires_oracle_metadata", False)
            else None
        )
        splice_concepts = (
            data[3].to(args.device, non_blocking=True)
            if len(data) > 3 and not crp_training
            else None
        )
        sample_indices = data[1] if crp_training else None
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
                metadata=metadata,
                sample_indices=sample_indices,
                simclr_weight=getattr(args, "simclr_weight", 1.0),
            )
        losses.update(loss.item(), bsz)
        simclr_losses.update(parts["simclr"].item(), bsz)
        decor_losses.update(parts["decor"].item(), bsz)
        entropy_losses.update(parts["entropy"].item(), bsz)
        splice_losses.update(parts["splice"].item(), bsz)
        for name, meter in relational_diagnostics.items():
            value = getattr(splice_regularizer, "last_diagnostics", {}).get(name)
            if value is not None:
                meter.update(float(value), bsz)

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
                metadata=metadata,
                sample_indices=sample_indices,
                simclr_weight=getattr(args, "simclr_weight", 1.0),
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

    metrics = {
        "loss": losses.avg,
        "simclr_loss": simclr_losses.avg,
        "decor_loss": decor_losses.avg,
        "entropy_loss": entropy_losses.avg,
        "splice_loss": splice_losses.avg,
    }
    metrics.update(
        {
            f"relational_{name}": meter.avg
            for name, meter in relational_diagnostics.items()
            if meter.count
        }
    )
    return metrics


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
                "SSL SimCLR loss": train_metrics["simclr_loss"],
                "SSL decor loss": train_metrics["decor_loss"],
                "SSL entropy loss": train_metrics["entropy_loss"],
                "SSL splice loss": train_metrics["splice_loss"],
                "SSL learning rate": optimizer.param_groups[0]["lr"],
                "SSL relational scheduled weight": train_metrics.get(
                    "relational_scheduled_weight", 0.0
                ),
                "SSL relational supported anchor fraction": train_metrics.get(
                    "relational_supported_anchor_fraction", 0.0
                ),
                "SSL relational mean anchor confidence": train_metrics.get(
                    "relational_mean_anchor_confidence", 0.0
                ),
                "SSL relational unweighted KL": train_metrics.get(
                    "relational_unweighted_kl", 0.0
                ),
                "SSL relational confidence-weighted KL": train_metrics.get(
                    "relational_confidence_weighted_kl", 0.0
                ),
            },
            step=epoch,
        )
