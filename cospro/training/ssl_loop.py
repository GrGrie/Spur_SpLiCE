from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

from cospro.training.contrastive import SimCLRLoss
from cospro.models.simclr import SimCLRModel
from cospro.training.late_pruning import LatePruning
from cospro.training.optim import warmup_learning_rate


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
    method=None,
    sample_indices=None,
    simclr_weight: float = 1.0,
    late_pruning: LatePruning | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    """SimCLR loss plus whatever the training method adds for this batch.

    With ``late_pruning`` the second view passes through the LateTVG-pruned encoder.
    """

    if simclr_weight < 0:
        raise ValueError("simclr_weight must be non-negative.")
    bsz = image[0].size(0)
    if late_pruning is None:
        embeddings = model.encoder(torch.cat([image[0], image[1]], dim=0))
    else:
        embeddings = torch.cat([model.encoder(image[0]), late_pruning(model.encoder, image[1])], dim=0)
    if simclr_weight > 0:
        projections = F.normalize(model.head(embeddings), dim=1)
        f1, f2 = torch.split(projections, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        simclr_loss, decor_loss, entropy_loss, _, _ = criterion(features)
        loss = simclr_weight * simclr_loss
    else:
        # Keep a differentiable zero so an unsupported relational batch is still safe to backpropagate.
        simclr_loss = embeddings.sum() * 0.0
        decor_loss = simclr_loss
        entropy_loss = simclr_loss
        loss = simclr_loss
    splice_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
    terms = None if method is None else method.extra_loss(
        model=model, embeddings=embeddings, sample_indices=sample_indices,
    )
    if terms is not None:
        splice_loss = terms.value
        loss = loss + splice_loss
    parts = {
        "simclr": simclr_loss,
        "decor": decor_loss,
        "entropy": entropy_loss,
        "splice": splice_loss,
        "_embeddings": embeddings,
    }
    return loss, parts, bsz


def train_one_epoch(
    train_loader, model, criterion, optimizer, scaler, epoch: int, args, method
) -> dict[str, float]:
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    simclr_losses = AverageMeter()
    decor_losses = AverageMeter()
    entropy_losses = AverageMeter()
    splice_losses = AverageMeter()
    # One meter per diagnostic the method reports, created when the method first reports it.
    method_diagnostics: dict[str, AverageMeter] = {}
    needs_indices = method is not None and method.needs_sample_indices
    late_pruning = LatePruning.from_args(args)
    if method is not None:
        method.set_epoch(epoch)

    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        image = data[0]
        image[0] = image[0].to(args.device, non_blocking=True)
        image[1] = image[1].to(args.device, non_blocking=True)
        if args.channels_last and str(args.device).startswith("cuda"):
            image[0] = image[0].contiguous(memory_format=torch.channels_last)
            image[1] = image[1].contiguous(memory_format=torch.channels_last)
        sample_indices = data[1] if needs_indices else None
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
                method=method,
                sample_indices=sample_indices,
                simclr_weight=getattr(args, "simclr_weight", 1.0),
                late_pruning=late_pruning,
            )
        losses.update(loss.item(), bsz)
        simclr_losses.update(parts["simclr"].item(), bsz)
        decor_losses.update(parts["decor"].item(), bsz)
        entropy_losses.update(parts["entropy"].item(), bsz)
        splice_losses.update(parts["splice"].item(), bsz)
        for name, value in ({} if method is None else method.diagnostics()).items():
            if value is not None:
                method_diagnostics.setdefault(name, AverageMeter()).update(float(value), bsz)
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
    # Historical metric names: every method diagnostic is reported under relational_<name>.
    metrics.update({f"relational_{name}": meter.avg for name, meter in method_diagnostics.items() if meter.count})
    if late_pruning is not None:
        metrics["latetvg_kept_fraction"] = late_pruning.last_kept_fraction
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
