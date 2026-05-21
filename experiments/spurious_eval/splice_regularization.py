from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpliceConfig:
    use_splice: bool = False
    splice_weight: float = 0.0


class DisabledSpliceRegularizer:
    """No-op placeholder for future SpLiCE regularization/intervention work."""

    enabled = False

    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)


def build_splice_regularizer(config: SpliceConfig):
    if not config.use_splice:
        return DisabledSpliceRegularizer()
    raise NotImplementedError(
        "SpLiCE regularization is intentionally not implemented yet. "
        "The default no-SpLiCE path does not allocate SpLiCE models or alter embeddings."
    )
