"""Frozen concept transfer: cosine distillation against a retained CLIP target bank."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from cospro.methods.base import LoaderContext, LossTerms, TrainingMethod, register_method
from cospro.methods.concept_targets import (
    ConceptDistillationRegularizer,
    FrozenConceptTransferSubset,
    load_target_artifact,
)


@register_method
class FrozenConceptDistill(TrainingMethod):
    """Predicts frozen CLIP targets from the backbone through a dedicated head."""

    name = "frozen_concept_distill"
    modes = ("frozen_concept_distill",)
    needs_sample_indices = True
    clip_distillation_dim = 512

    def __init__(
        self,
        *,
        targets_path: str = "",
        target_kind: str = "reconstruction",
        alpha_max: float = 0.1,
        start_epoch: int = 0,
        warmup_epochs: int = 0,
        regularizer: ConceptDistillationRegularizer | None = None,
    ) -> None:
        self.targets_path = targets_path
        self._settings = (target_kind, alpha_max, start_epoch, warmup_epochs)
        self.regularizer = regularizer

    @classmethod
    def from_config(cls, config, **resolved) -> "FrozenConceptDistill":
        transfer = config.concept_transfer
        return cls(
            targets_path=transfer.concept_transfer_targets,
            target_kind=transfer.concept_transfer_target_kind,
            alpha_max=transfer.concept_transfer_alpha_max,
            start_epoch=transfer.concept_transfer_start_epoch,
            warmup_epochs=transfer.concept_transfer_warmup_epochs,
        )

    def wrap_loader(self, loader, context: LoaderContext):
        if not self.targets_path:
            raise ValueError("frozen_concept_distill requires a target bank.")
        targets = load_target_artifact(self.targets_path)
        dataset = FrozenConceptTransferSubset(
            loader.dataset, loader.dataset.transform, targets, dataset_name=context.dataset,
        )
        self.regularizer = ConceptDistillationRegularizer(targets, *self._settings)
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=context.batch_size,
            collate_fn=dataset.collate,
            num_workers=context.num_workers,
            pin_memory=True,
            generator=loader.generator,
            worker_init_fn=context.worker_init_fn,
        )

    def set_epoch(self, epoch: int) -> None:
        if self.regularizer is not None:
            self.regularizer.set_epoch(epoch)

    def extra_loss(self, *, model, embeddings, sample_indices) -> LossTerms | None:
        if self.regularizer is None:
            return None
        if sample_indices is None or model.clip_distillation_head is None:
            raise ValueError("Frozen transfer requires target-bank indices and its prediction head.")
        target_rows, valid_rows = self.regularizer.targets_for_indices(sample_indices, embeddings.device)
        predictions = model.clip_distillation_head(embeddings)
        value = self.regularizer(predictions, torch.cat([target_rows, target_rows]), valid_rows)
        return LossTerms(value=value, diagnostics=self.regularizer.last_diagnostics)

    def diagnostics(self) -> Mapping[str, float]:
        return getattr(self.regularizer, "last_diagnostics", {})

    def provenance(self) -> dict[str, Any]:
        return {}

    def input_artifacts(self) -> list[Path]:
        return [Path(self.targets_path)] if self.targets_path else []
