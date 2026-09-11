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
    splice_regularizer=None,
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
        simclr_loss, decor_loss, entropy_loss, _, _ = criterion(
            features
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
                raise ValueError("CoSpRo relational regularization requires graph-row sample indices.")
            splice_loss = splice_regularizer(embeddings, sample_indices)
            loss = loss + splice_loss
            parts = {
                "simclr": simclr_loss,
                "decor": decor_loss,
                "entropy": entropy_loss,
                "splice": splice_loss,
                "_embeddings": embeddings,
            }
            return loss, parts, bsz
        if getattr(splice_regularizer, "requires_concept_transfer", False):
            if sample_indices is None or model.clip_distillation_head is None:
                raise ValueError("Frozen transfer requires target-bank indices and its prediction head.")
            target_rows, valid_rows = splice_regularizer.targets_for_indices(sample_indices, embeddings.device)
            predictions = model.clip_distillation_head(embeddings)
            splice_loss = splice_regularizer(predictions, torch.cat([target_rows, target_rows]), valid_rows)
            loss = loss + splice_loss
        else:
            raise ValueError("Unsupported SSL regularizer.")
    parts = {
        "simclr": simclr_loss,
        "decor": decor_loss,
        "entropy": entropy_loss,
        "splice": splice_loss,
        "_embeddings": embeddings,
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
        "q": AverageMeter(),
        "row_mass_before_renorm": AverageMeter(),
        "effective_donor_count": AverageMeter(),
        "row_mass_after_renorm": AverageMeter(),
        "valid_fraction": AverageMeter(),
        "cosine_loss": AverageMeter(),
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
        concept_transfer = getattr(splice_regularizer, "requires_concept_transfer", False)
        sample_indices = data[1] if (crp_training or concept_transfer) else None
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
                splice_regularizer=splice_regularizer,
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
    run_recorder=None,
) -> dict[str, float]:
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
    payload = {
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
    }
    payload.update({f"SSL {key}": value for key, value in train_metrics.items()
                    if key.startswith("la_ssl_") or key in {"relational_valid_fraction", "relational_cosine_loss"}})
    if run_recorder is not None:
        run_recorder.log_metrics("ssl", epoch, payload)
    if wandb_run is not None:
        wandb_run.log(payload, step=epoch)
    return payload
