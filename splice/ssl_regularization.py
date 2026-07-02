from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import DataLoader

import splice


@dataclass(frozen=True)
class SpliceConfig:
    use_splice: bool = False
    splice_weight: float = 0.0
    mode: str = "none"
    concepts: str = ""
    l1_penalty: float = 0.25
    vocab: str = "laion"
    vocab_size: int = 10000
    model: str = "open_clip:ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    score_threshold: float = 0.01
    score_reduction: str = "mean"
    batch_size: int = 128
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def splice_mode_uses_scores(mode: str) -> bool:
    return mode in {"augment", "corr_reg", "augment_corr_reg"}


def splice_mode_uses_regularizer(mode: str) -> bool:
    return mode in {"corr_reg", "augment_corr_reg"}


def identity_collate(batch):
    return batch


def resolve_concept_indices(concepts: str, vocabulary: Sequence[str]) -> list[int]:
    if not concepts.strip():
        raise ValueError("--splice_concepts must be non-empty when SpLiCE modes are enabled.")

    vocab_to_idx = {concept: idx for idx, concept in enumerate(vocabulary)}
    indices: list[int] = []
    for raw_token in concepts.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token.lstrip("-").isdigit():
            index = int(token)
            if index < 0 or index >= len(vocabulary):
                raise ValueError(f"SpLiCE concept index {index} is outside vocabulary size {len(vocabulary)}.")
            indices.append(index)
            continue
        if token not in vocab_to_idx:
            raise ValueError(f"SpLiCE concept {token!r} was not found in the selected vocabulary.")
        indices.append(vocab_to_idx[token])

    if not indices:
        raise ValueError("--splice_concepts did not contain any valid concept names or indices.")
    return sorted(set(indices))


class SpliceConceptScorer:
    """Frozen SpLiCE helper that maps raw PIL images to selected concept scores."""

    def __init__(self, config: SpliceConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.preprocess = splice.get_preprocess(config.model, pretrained=config.pretrained)
        self.vocabulary = splice.get_vocabulary(config.vocab, config.vocab_size)
        self.concept_indices = resolve_concept_indices(config.concepts, self.vocabulary)
        self.model = splice.load(
            config.model,
            config.vocab,
            config.vocab_size,
            config.device,
            pretrained=config.pretrained,
            l1_penalty=config.l1_penalty,
            return_weights=True,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def score_weights(self, weights: torch.Tensor) -> torch.Tensor:
        selected = weights[:, self.concept_indices]
        if self.config.score_reduction == "mean":
            return selected.mean(dim=1)
        if self.config.score_reduction == "max":
            return selected.max(dim=1).values
        raise ValueError(f"Unsupported SpLiCE score reduction: {self.config.score_reduction}")

    @torch.no_grad()
    def score_images(self, images) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image) for image in images], dim=0).to(self.device)
        weights = self.model.encode_image(batch)
        return self.score_weights(weights).detach().cpu()

    @torch.no_grad()
    def score_dataset(self, dataset, batch_size: int | None = None, num_workers: int | None = None) -> torch.Tensor:
        loader = DataLoader(
            dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers if num_workers is None else num_workers,
            collate_fn=identity_collate,
        )
        scores = []
        for batch in loader:
            images = [item[0] for item in batch]
            scores.append(self.score_images(images))
        return torch.cat(scores, dim=0)


class CorrelationSpliceRegularizer:
    enabled = True

    def __init__(self, weight: float) -> None:
        self.weight = weight

    def __call__(self, embeddings: torch.Tensor, scores: torch.Tensor | None = None) -> torch.Tensor:
        if scores is None:
            raise ValueError("SpLiCE correlation regularization requires per-sample concept scores.")
        if self.weight <= 0:
            return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

        embeddings = embeddings.float()
        scores = scores.to(device=embeddings.device, dtype=embeddings.dtype).view(-1)
        if embeddings.shape[0] != scores.shape[0]:
            raise ValueError(
                f"Expected one SpLiCE score per embedding, got {scores.shape[0]} scores for {embeddings.shape[0]} embeddings."
            )
        if scores.numel() < 2:
            return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

        centered_embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        centered_scores = scores - scores.mean()
        score_std = centered_scores.norm()
        feature_std = centered_embeddings.norm(dim=0)
        valid = (feature_std > 1e-12) & (score_std > 1e-12)
        if not torch.any(valid):
            return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

        correlations = (centered_embeddings[:, valid] * centered_scores.unsqueeze(1)).sum(dim=0)
        correlations = correlations / (feature_std[valid] * score_std + 1e-12)
        return self.weight * correlations.pow(2).mean().to(dtype=embeddings.dtype)


class DisabledSpliceRegularizer:
    """No-op placeholder for future SpLiCE regularization/intervention work."""

    enabled = False

    def __call__(self, embeddings: torch.Tensor, scores: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)


def build_splice_regularizer(config: SpliceConfig):
    if not config.use_splice or not splice_mode_uses_regularizer(config.mode):
        return DisabledSpliceRegularizer()
    return CorrelationSpliceRegularizer(config.splice_weight)
