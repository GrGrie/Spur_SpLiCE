from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GroupMetrics:
    """Binary classification metrics grouped by Waterbirds metadata."""

    average: float
    worst_group: float
    best_group: float
    group_accuracy: torch.Tensor
    group_counts: torch.Tensor

    def as_spurssl_dict(self) -> dict[str, float | torch.Tensor]:
        return {
            "acc_avg": self.average,
            "acc_wg": self.worst_group,
            "best_acc": self.best_group,
            "group_accuracy": self.group_accuracy,
            "group_counts": self.group_counts,
        }


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    """Computes top-k accuracy in percent, matching the SpurSSL helper."""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results


def waterbirds_group_ids(metadata: torch.Tensor) -> torch.Tensor:
    """Return SpurSSL/WILDS group IDs for metadata columns [background, y]."""

    if metadata.ndim == 1:
        metadata = metadata.unsqueeze(0)
    return metadata[:, 0].long() + 2 * metadata[:, 1].long()


def compute_group_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    metadata: torch.Tensor,
    n_groups: int = 4,
) -> GroupMetrics:
    """Compute average, worst-group, and best-group accuracy as fractions."""

    predictions = predictions.cpu().long()
    labels = labels.cpu().long()
    groups = waterbirds_group_ids(metadata.cpu())
    correct = predictions.eq(labels).float()

    group_accuracy = torch.zeros(n_groups, dtype=torch.float32)
    group_counts = torch.zeros(n_groups, dtype=torch.float32)
    for group_idx in range(n_groups):
        mask = groups == group_idx
        group_counts[group_idx] = mask.sum()
        if group_counts[group_idx] > 0:
            group_accuracy[group_idx] = correct[mask].mean()

    nonempty = group_counts > 0
    average = correct.mean().item() if correct.numel() else 0.0
    worst_group = group_accuracy[nonempty].min().item() if nonempty.any() else 0.0
    best_group = group_accuracy[nonempty].max().item() if nonempty.any() else 0.0

    return GroupMetrics(
        average=average,
        worst_group=worst_group,
        best_group=best_group,
        group_accuracy=group_accuracy,
        group_counts=group_counts,
    )


def entropy_effective_rank(features: torch.Tensor) -> tuple[float, float, float]:
    """Match SpurSSL's SVD-based representation rank metrics."""

    features = features.view(features.shape[0], -1)
    _, singular_values, _ = torch.linalg.svd(features, full_matrices=False)
    squared_singular_values = singular_values**2
    total_energy = torch.sum(squared_singular_values)
    sum_squared_energy = torch.sum(squared_singular_values**2)

    energy_based_rank = (total_energy**2) / (sum_squared_energy + 1e-12)
    probabilities = squared_singular_values / (total_energy + 1e-12)
    spectral_entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12))
    effective_rank = torch.exp(spectral_entropy)

    return spectral_entropy.item(), effective_rank.item(), energy_based_rank.item()
