from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler


class ConceptBalancedSampler(Sampler[int]):
    """Algorithm 1: uniform concepts, inverse-frequency classes, uniform samples."""

    def __init__(
        self,
        concepts: torch.Tensor,
        labels: torch.Tensor,
        sampling_lambda: float,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        concepts = concepts.cpu().long()
        labels = labels.cpu().long()
        if concepts.ndim != 2 or concepts.shape[0] != labels.numel():
            raise ValueError("Concepts must be [samples, slots] and align with labels.")
        self.num_samples = int(num_samples or labels.numel())
        self.sampling_lambda = float(sampling_lambda)
        self.seed = int(seed)
        self.epoch = 0
        unique_concepts = torch.unique(concepts[concepts >= 0]).tolist()
        self.buckets: dict[int, dict[int, np.ndarray]] = {}
        for concept in unique_concepts:
            contains = concepts.eq(int(concept)).any(dim=1)
            class_buckets: dict[int, np.ndarray] = {}
            for label in torch.unique(labels[contains]).tolist():
                indices = torch.where(contains & labels.eq(int(label)))[0].numpy()
                if indices.size:
                    class_buckets[int(label)] = indices
            if class_buckets:
                self.buckets[int(concept)] = class_buckets
        if not self.buckets:
            raise ValueError("No active concepts were found.")
        self.concept_ids = np.asarray(sorted(self.buckets), dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.num_samples):
            concept = int(rng.choice(self.concept_ids))
            buckets = self.buckets[concept]
            classes = np.asarray(sorted(buckets), dtype=np.int64)
            counts = np.asarray([buckets[int(label)].size for label in classes], dtype=np.float64)
            weights = np.power(1.0 / counts, self.sampling_lambda)
            probabilities = weights / weights.sum()
            label = int(rng.choice(classes, p=probabilities))
            yield int(rng.choice(buckets[label]))


def inferred_worst_group_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    concepts: torch.Tensor,
) -> float:
    """Worst accuracy over all non-empty (class, discovered concept) memberships."""

    predictions = predictions.cpu().long()
    labels = labels.cpu().long()
    concepts = concepts.cpu().long()
    accuracies: list[float] = []
    for concept in torch.unique(concepts[concepts >= 0]).tolist():
        contains = concepts.eq(int(concept)).any(dim=1)
        for label in torch.unique(labels[contains]).tolist():
            group = contains & labels.eq(int(label))
            if group.any():
                accuracies.append(predictions[group].eq(labels[group]).float().mean().item())
    return min(accuracies) if accuracies else 0.0
