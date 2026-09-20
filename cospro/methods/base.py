"""The seam between the SSL trainer and a training method.

A method owns everything that distinguishes it from plain SimCLR: the loader it needs, the loss term
it adds, the diagnostics it reports, the artifacts it consumes and the state it saves for resume.
The trainer calls the same four entry points for every method, so datasets, the epoch loop and the
run record stay free of per-method branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping

import torch


@dataclass(frozen=True)
class LossTerms:
    """The loss a method adds to the SimCLR objective, with the diagnostics it reports."""

    value: torch.Tensor
    diagnostics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LoaderContext:
    """What a method needs to rebuild the SSL loader around its own dataset or sampler."""

    dataset: str
    batch_size: int
    num_workers: int
    seed: int
    generator: torch.Generator | None = None
    worker_init_fn: Callable[[int], None] | None = None


class TrainingMethod:
    """Plain SimCLR: no extra loader, no extra loss. Other methods override what they change."""

    name: ClassVar[str] = "simclr"
    #: ``splice_mode`` values this method serves.
    modes: ClassVar[tuple[str, ...]] = ()
    #: True for the method selected by the --la_ssl flag.
    requires_la_ssl: ClassVar[bool] = False
    needs_sample_indices: ClassVar[bool] = False
    clip_distillation_dim: ClassVar[int | None] = None

    def wrap_loader(self, loader, context: LoaderContext):
        return loader

    def set_epoch(self, epoch: int) -> None:
        return None

    def extra_loss(self, *, model, embeddings, sample_indices) -> LossTerms | None:
        return None

    def diagnostics(self) -> Mapping[str, float]:
        return {}

    def provenance(self) -> dict[str, Any]:
        """Values recorded in run.json and the W&B config under their historical names."""

        return {}

    def input_artifacts(self) -> list[Path]:
        """Files this method consumes, registered as run inputs."""

        return []

    def sampling_state(self):
        """State a method saves in the checkpoint, such as an adaptive sampler."""

        return None

    def refresh_sampling(self, model, device, temperature: float, epoch: int) -> dict[str, float]:
        return {}


METHOD_REGISTRY: dict[str, type[TrainingMethod]] = {}


def register_method(cls: type[TrainingMethod]) -> type[TrainingMethod]:
    if cls.name in METHOD_REGISTRY:
        raise ValueError(f"Training method {cls.name!r} is registered twice.")
    for other in METHOD_REGISTRY.values():
        clash = set(cls.modes) & set(other.modes)
        if clash:
            raise ValueError(f"Methods {cls.name!r} and {other.name!r} both claim modes {sorted(clash)}.")
    METHOD_REGISTRY[cls.name] = cls
    return cls


def method_for(*, splice_mode: str, la_ssl: bool) -> type[TrainingMethod]:
    """The registered method a configuration selects."""

    if la_ssl:
        selected = [cls for cls in METHOD_REGISTRY.values() if cls.requires_la_ssl]
    else:
        selected = [cls for cls in METHOD_REGISTRY.values() if splice_mode in cls.modes]
    if len(selected) != 1:
        raise ValueError(f"Expected one method for splice_mode={splice_mode!r} la_ssl={la_ssl}; found {len(selected)}.")
    return selected[0]
