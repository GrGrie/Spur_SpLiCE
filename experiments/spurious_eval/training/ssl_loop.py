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


def _gradient_diagnostic(
    model,
    parts: dict[str, torch.Tensor],
    splice_regularizer,
    scaler,
    epoch: int,
    batch: int,
) -> dict:
    """Inspect AMP-scaled objective gradients without touching model state.

    AMP can underflow an FP16 objective before a later ``.float()`` cast.  The
    loss therefore has to be scaled *before* ``autograd.grad``; the returned
    parameter gradients are then converted to float32 and unscaled.
    """

    encoder_parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    head_parameters = []
    if model.clip_distillation_head is not None:
        head_parameters = [
            parameter for parameter in model.clip_distillation_head.parameters() if parameter.requires_grad
        ]
    scaler_enabled = bool(scaler.is_enabled()) if hasattr(scaler, "is_enabled") else False
    scale = float(scaler.get_scale()) if scaler_enabled else 1.0

    def gradients(loss, parameters):
        if not parameters:
            return []
        return torch.autograd.grad(
            loss * scale, parameters, retain_graph=True, allow_unused=True
        )

    simclr_encoder_grads = gradients(parts["simclr"], encoder_parameters)
    splice_encoder_grads = gradients(parts["splice"], encoder_parameters)
    simclr_head_grads = gradients(parts["simclr"], head_parameters)
    splice_head_grads = gradients(parts["splice"], head_parameters)

    def flatten(grads, parameters):
        pieces = [
            (
                gradient.detach().float() / scale
                if gradient is not None
                else torch.zeros_like(parameter, dtype=torch.float32)
            ).reshape(-1)
            for gradient, parameter in zip(grads, parameters)
        ]
        return torch.cat(pieces) if pieces else torch.zeros(1, dtype=torch.float32)

    simclr_encoder = flatten(simclr_encoder_grads, encoder_parameters)
    splice_encoder = flatten(splice_encoder_grads, encoder_parameters)
    simclr_head = flatten(simclr_head_grads, head_parameters)
    splice_head = flatten(splice_head_grads, head_parameters)
    simclr_norm = float(torch.linalg.vector_norm(simclr_encoder))
    splice_norm = float(torch.linalg.vector_norm(splice_encoder))
    finite = bool(
        torch.isfinite(simclr_encoder).all()
        and torch.isfinite(splice_encoder).all()
        and torch.isfinite(simclr_head).all()
        and torch.isfinite(splice_head).all()
    )
    if not finite:
        raise FloatingPointError(f"Non-finite encoder gradient in diagnostic at epoch={epoch}, batch={batch}.")
    cosine = None
    if simclr_norm > 0 and splice_norm > 0:
        cosine = float(F.cosine_similarity(simclr_encoder.view(1, -1), splice_encoder.view(1, -1)).item())
    embeddings = parts["_embeddings"].detach().float()
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    variance = float(centered.square().mean())
    singular_values = torch.linalg.svdvals(centered) if centered.shape[0] > 1 else torch.zeros(1)
    effective_rank = float(
        (singular_values.square().sum() ** 2 / singular_values.pow(4).sum().clamp_min(1e-12)).item()
    )
    diagnostic = {
        "epoch": int(epoch),
        "batch": int(batch),
        "simclr_gradient_norm": simclr_norm,
        "kl_gradient_norm": splice_norm,
        "encoder_simclr_gradient_norm": simclr_norm,
        "encoder_kl_gradient_norm": splice_norm,
        "direct_head_simclr_gradient_norm": float(torch.linalg.vector_norm(simclr_head)),
        "direct_head_kl_gradient_norm": float(torch.linalg.vector_norm(splice_head)),
        "gradient_ratio_kl_to_simclr": None if simclr_norm == 0 else splice_norm / simclr_norm,
        "gradient_cosine": cosine,
        "simclr_gradient_zero": simclr_norm == 0,
        "kl_gradient_zero": splice_norm == 0,
        "direct_head_present": bool(head_parameters),
        "amp_scale": scale,
        "embedding_norm_mean": float(embeddings.norm(dim=1).mean()),
        "embedding_variance": variance,
        "embedding_effective_rank": effective_rank,
        "finite": finite,
    }
    diagnostic.update(getattr(splice_regularizer, "last_diagnostics", {}))
    return diagnostic


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
                raise ValueError("CRP relational regularization requires graph-row sample indices.")
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
            if sample_indices is None:
                raise ValueError("Frozen concept transfer requires target-bank row indices.")
            if model.clip_distillation_head is None:
                raise ValueError("Frozen concept transfer requires the g_clip head.")
            target_rows, valid_rows = splice_regularizer.targets_for_indices(
                sample_indices, embeddings.device
            )
            repeated_targets = torch.cat([target_rows, target_rows], dim=0)
            repeated_valid = valid_rows
            predictions = model.clip_distillation_head(embeddings)
            splice_loss = splice_regularizer(predictions, repeated_targets, repeated_valid)
            loss = loss + splice_loss
            parts = {
                "simclr": simclr_loss,
                "decor": decor_loss,
                "entropy": entropy_loss,
                "splice": splice_loss,
                "_embeddings": embeddings,
            }
            return loss, parts, bsz
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
    }
    gradient_records = []
    diagnostic_epochs = set(getattr(args, "gradient_diagnostics_epochs", (1, 11, 20, 25, 500)))
    diagnostic_batches = int(getattr(args, "gradient_diagnostics_batches", 4))

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
        if (
            getattr(args, "gradient_diagnostics", False)
            and epoch in diagnostic_epochs
            and idx < diagnostic_batches
        ):
            gradient_records.append(_gradient_diagnostic(model, parts, splice_regularizer, scaler, epoch, idx))

        if args.optimizer == "SAM":
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step()
            loss, _, _ = simclr_forward_loss(
                model,
                criterion,
                image,
                splice_regularizer=splice_regularizer,
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
    if gradient_records:
        metrics["gradient_diagnostics"] = gradient_records
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
        for diagnostic in train_metrics.get("gradient_diagnostics", []):
            for key, value in diagnostic.items():
                if isinstance(value, (int, float)) and value is not None:
                    payload[f"gradient/{key}/batch{diagnostic.get('batch', 0)}"] = value
            # W&B step is epoch-level; preserve all four batches in the summary JSON.
        wandb_run.log(payload, step=epoch)
