"""Concept-factor training: keep entangled concept factors apart in the student, without labels.

Three mechanisms share one set of factors (``cospro.pipeline.concept_factors``):

* F1, concept-conditioned batches. A fraction of the batches holds only images that show one factor
  of an entangled pair. Inside such a batch that factor tells no image from another, so the
  contrastive loss has to separate the images by everything else, including the factor it is
  correlated with. This is conditional contrastive learning (Tsai et al., 2021) with the discovered
  factor in place of an annotated attribute.
* F2, decorrelated factor distillation. A linear head on the backbone predicts the whitened factor
  activations. Whitening turns each factor into its part the other factors leave unexplained, so
  the backbone must encode correlated factors along separate linear directions. Per-image weights can
  make the factors independent in the weighted data (``balanced``), so the co-occurrence of two
  concepts stops paying off in the regression.
* F3, concept blocks. One small head per factor maps the backbone into its own block, and a supervised
  contrastive loss inside block k pulls together the images that show factor k and pushes away the
  images without it. Pairs that share k but differ most in their other factors weigh most, so a
  direction that fuses k with its usual context gives the wrong answer on exactly those pairs: the
  backbone needs a direction for k that holds across contexts. Every factor has a block, so no factor
  is declared spurious.

Every image still occurs exactly once per epoch, so the optimizer-step budget matches SimCLR.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from cospro.methods.base import LoaderContext, LossTerms, TrainingMethod, register_method
from cospro.methods.relational_graph import IndexedCoSpRoDataset
from cospro.pipeline.concept_factors import (
    SHUFFLE_SEED,
    FactorConfig,
    balancing_weights,
    block_factor_columns,
    factor_name,
    expected_condition_pool,
    factor_report,
    format_factor_report,
    load_concept_factors,
    rows_for_subset,
)


class ConceptConditionedBatchSampler(Sampler[list[int]]):
    """Batches that each show one conditioning factor, mixed with ordinary random batches."""

    def __init__(self, active: torch.Tensor, batch_size: int, condition_fraction: float,
                 generator: torch.Generator | None) -> None:
        if batch_size < 2:
            raise ValueError("Concept-conditioned training requires batch_size >= 2.")
        if not 0 <= condition_fraction <= 1:
            raise ValueError("The conditioned batch fraction lies in [0, 1].")
        active = torch.as_tensor(active, dtype=torch.bool).cpu()
        self.count = int(active.shape[0])
        self.batch_size = int(batch_size)
        self.condition_fraction = float(condition_fraction)
        self.generator = generator
        self.members = [np.flatnonzero(column) for column in active.T.numpy()]
        self.sample_factors = [np.flatnonzero(row) for row in active.numpy()]
        self.min_pool = expected_condition_pool(batch_size)
        self.last_conditioned_fraction = 0.0

    def __len__(self) -> int:
        return math.ceil(self.count / self.batch_size)

    def _permutation(self, size: int) -> np.ndarray:
        return torch.randperm(size, generator=self.generator).numpy()

    def __iter__(self) -> Iterator[list[int]]:
        order = self._permutation(self.count)
        queues = [members[self._permutation(len(members))] for members in self.members]
        heads = [0] * len(queues)
        remaining = np.array([len(members) for members in self.members], dtype=np.int64)
        used = np.zeros(self.count, dtype=bool)
        taken = cursor = batches = conditioned = 0

        def take(index: int, batch: list[int]) -> None:
            nonlocal taken
            used[index] = True
            remaining[self.sample_factors[index]] -= 1
            batch.append(int(index))
            taken += 1

        while taken < self.count:
            batch: list[int] = []
            if queues and float(torch.rand(1, generator=self.generator)) < self.condition_fraction:
                eligible = np.flatnonzero(remaining >= self.min_pool)
                if len(eligible):
                    factor = int(eligible[int(torch.randint(len(eligible), (1,), generator=self.generator))])
                    queue, head = queues[factor], heads[factor]
                    while len(batch) < self.batch_size and head < len(queue):
                        if not used[queue[head]]:
                            take(queue[head], batch)
                        head += 1
                    heads[factor] = head
                    conditioned += 1
            while len(batch) < self.batch_size and cursor < self.count:
                if not used[order[cursor]]:
                    take(order[cursor], batch)
                cursor += 1
            batches += 1
            yield batch
        self.last_conditioned_fraction = conditioned / max(batches, 1)


class ConceptBlockContrast:
    """Supervised contrastive loss per factor block, with pairs weighted by how much their contexts differ."""

    def __init__(self, presence: torch.Tensor, weight: float, temperature: float, context_weight: float,
                 start_epoch: int, warmup_epochs: int) -> None:
        if weight < 0 or temperature <= 0 or context_weight < 0 or start_epoch < 0 or warmup_epochs < 0:
            raise ValueError("Concept-block settings must be non-negative and the temperature positive.")
        self.presence = torch.as_tensor(presence, dtype=torch.bool).cpu()
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.context_weight = float(context_weight)
        self.start_epoch = int(start_epoch)
        self.warmup_epochs = int(warmup_epochs)
        self.epoch = 0
        self.last_diagnostics: dict[str, float] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def scheduled_weight(self) -> float:
        if self.epoch <= self.start_epoch:
            return 0.0
        if self.warmup_epochs == 0:
            return self.weight
        return self.weight * min(1.0, (self.epoch - self.start_epoch) / self.warmup_epochs)

    def pair_weights(self, presence: torch.Tensor) -> torch.Tensor:
        """Weight of every positive pair of every block: 1 plus the scaled difference of the whole factor sets."""

        present = presence.float()
        difference = torch.cdist(present, present, p=1) / present.shape[1]
        off_diagonal = ~torch.eye(len(present), dtype=torch.bool, device=present.device)
        scale = difference[off_diagonal].mean().clamp_min(1e-6)
        context = 1.0 + self.context_weight * difference / scale
        shared = present.T[:, :, None] * present.T[:, None, :]
        return shared * context[None] * off_diagonal[None]

    def __call__(self, blocks: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
        rows = torch.as_tensor(sample_indices, dtype=torch.long).detach().cpu().view(-1)
        presence = self.presence.index_select(0, rows).to(blocks.device)
        presence = torch.cat([presence, presence], dim=0)
        if blocks.ndim != 3 or blocks.shape[:2] != presence.shape:
            raise ValueError("Concept blocks expect [2 * batch, blocks, dim] with one presence row per view.")
        embedded = torch.nn.functional.normalize(blocks.float(), dim=-1)
        logits = torch.einsum("ikd,jkd->kij", embedded, embedded) / self.temperature
        diagonal = torch.eye(logits.shape[1], dtype=torch.bool, device=logits.device)
        log_probability = torch.log_softmax(logits.masked_fill(diagonal[None], float("-inf")), dim=2)
        log_probability = log_probability.masked_fill(diagonal[None], 0.0)
        weights = self.pair_weights(presence)
        total = weights.sum(dim=2)
        anchors = total > 0
        per_anchor = -(weights * log_probability).sum(dim=2) / total.clamp_min(1e-12)
        loss = per_anchor[anchors].mean() if anchors.any() else blocks.sum() * 0.0
        self.last_diagnostics = {
            "factor_block_scheduled_weight": self.scheduled_weight,
            "factor_block_loss": float(loss.detach()),
            "factor_block_anchor_fraction": float(anchors.float().mean()),
            "factor_block_positives_per_anchor": float((weights > 0).sum(dim=2)[anchors].float().mean())
            if anchors.any() else 0.0,
        }
        return self.scheduled_weight * loss


class FactorDistillationRegularizer:
    """Scheduled mean squared error between a linear head and the per-image factor targets."""

    def __init__(self, targets: torch.Tensor, weight: float, start_epoch: int, warmup_epochs: int,
                 sample_weights: torch.Tensor | None = None) -> None:
        if weight < 0 or start_epoch < 0 or warmup_epochs < 0:
            raise ValueError("Factor distillation schedule values must be non-negative.")
        self.targets = torch.as_tensor(targets).detach().float().cpu()
        self.sample_weights = None if sample_weights is None else torch.as_tensor(sample_weights).float().cpu()
        self.weight = float(weight)
        self.start_epoch = int(start_epoch)
        self.warmup_epochs = int(warmup_epochs)
        self.epoch = 0
        self.last_diagnostics: dict[str, float] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def scheduled_weight(self) -> float:
        if self.epoch <= self.start_epoch:
            return 0.0
        if self.warmup_epochs == 0:
            return self.weight
        return self.weight * min(1.0, (self.epoch - self.start_epoch) / self.warmup_epochs)

    def __call__(self, predictions: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
        rows = torch.as_tensor(sample_indices, dtype=torch.long).detach().cpu().view(-1)
        targets = self.targets.index_select(0, rows).to(predictions.device)
        targets = torch.cat([targets, targets], dim=0)
        if predictions.shape != targets.shape:
            raise ValueError("Factor distillation expects two aligned views per batch sample.")
        per_image = (predictions.float() - targets).pow(2).mean(dim=1)
        mse = per_image.mean()
        loss = mse
        if self.sample_weights is not None:
            weights = self.sample_weights.index_select(0, rows).to(predictions.device)
            loss = (per_image * torch.cat([weights, weights])).mean()
        # Targets have unit variance per factor, so 1 - mse is the explained variance.
        self.last_diagnostics = {
            "factor_scheduled_weight": self.scheduled_weight,
            "factor_mse": float(mse.detach()),
            "factor_explained_variance": 1.0 - float(mse.detach()),
        }
        return self.scheduled_weight * loss


@register_method
class ConceptFactors(TrainingMethod):
    """F1 conditioned batches and/or F2 decorrelated factor distillation."""

    name = "concept_factors"
    modes = ("concept_factors",)
    needs_sample_indices = True

    def __init__(
        self,
        *,
        dataset: str = "",
        concept_groups: str = "",
        splice_cache: str = "",
        config: FactorConfig | None = None,
        condition_fraction: float = 0.0,
        distill_weight: float = 0.0,
        target_kind: str = "whitened",
        sample_weighting: str = "none",
        block_weight: float = 0.0,
        block_count: int = 32,
        block_dim: int = 16,
        block_temperature: float = 0.1,
        block_context_weight: float = 1.0,
        block_presence: str = "real",
        start_epoch: int = 0,
        warmup_epochs: int = 0,
    ) -> None:
        self.dataset = dataset
        self.concept_groups = concept_groups
        self.splice_cache = splice_cache
        self.config = config or FactorConfig()
        self.condition_fraction = float(condition_fraction)
        self.distill_weight = float(distill_weight)
        self.target_kind = target_kind
        self.sample_weighting = sample_weighting
        self.block_settings = {
            "weight": float(block_weight), "count": int(block_count), "dim": int(block_dim),
            "temperature": float(block_temperature), "context_weight": float(block_context_weight),
            "presence": block_presence,
        }
        self.schedule = (int(start_epoch), int(warmup_epochs))
        self.factor_head_dim: int | None = None
        self.factor_block_shape: tuple[int, int] | None = None
        self.blocks: ConceptBlockContrast | None = None
        self.block_names: list[str] = []
        self.balancing: dict[str, float] = {}
        self.sampler: ConceptConditionedBatchSampler | None = None
        self.regularizer: FactorDistillationRegularizer | None = None
        self.report: dict[str, Any] = {}
        self.paths: tuple[Path, Path] | None = None

    @classmethod
    def from_config(cls, config, **resolved) -> "ConceptFactors":
        options = config.concept_factors
        return cls(
            dataset=config.data.dataset,
            concept_groups=options.factor_concept_groups,
            splice_cache=options.factor_splice_cache,
            config=FactorConfig(
                min_frequency=options.factor_min_frequency,
                max_frequency=options.factor_max_frequency,
                max_count=options.factor_max_count,
                merge_similarity=options.factor_merge_similarity,
                condition_pairs=options.factor_condition_pairs,
                min_correlation=options.factor_min_correlation,
                max_text_similarity=options.factor_max_text_similarity,
                whitening_eps=options.factor_whitening_eps,
            ),
            condition_fraction=options.factor_condition_fraction,
            distill_weight=options.factor_distill_weight,
            target_kind=options.factor_targets,
            sample_weighting=options.factor_sample_weighting,
            block_weight=options.factor_block_weight,
            block_count=options.factor_block_count,
            block_dim=options.factor_block_dim,
            block_temperature=options.factor_block_temperature,
            block_context_weight=options.factor_block_context_weight,
            block_presence=options.factor_block_presence,
            start_epoch=options.factor_start_epoch,
            warmup_epochs=options.factor_warmup_epochs,
        )

    def wrap_loader(self, loader, context: LoaderContext):
        source_indices = getattr(loader.dataset, "indices", None)
        if source_indices is None:
            raise ValueError("Concept-factor training requires an SSL dataset with stable source indices.")
        factors, groups_path, cache_path = load_concept_factors(
            context.dataset, self.config, concept_groups=self.concept_groups, splice_cache=self.splice_cache,
        )
        self.paths = (groups_path, cache_path)
        self.report = factor_report(factors)
        print(f"[INFO] Concept factors from {groups_path} and {cache_path}", flush=True)
        print(format_factor_report(self.report), flush=True)
        rows = rows_for_subset(factors["sample_ids"], context.dataset, source_indices)
        if self.condition_fraction > 0 and not factors["condition_factors"]:
            raise ValueError("Conditioned batches need at least one entangled factor pair.")
        if self.distill_weight > 0:
            targets = factors["targets"][self.target_kind].index_select(0, rows)
            sample_weights = None
            if self.sample_weighting == "atypicality":
                sample_weights = factors["atypicality_weights"].index_select(0, rows)
            elif self.sample_weighting == "balanced":
                weights, self.balancing = balancing_weights(factors["active"])
                sample_weights = weights.index_select(0, rows)
                print(f"[INFO] Balancing weights: {self.balancing}", flush=True)
            self.regularizer = FactorDistillationRegularizer(
                targets, self.distill_weight, *self.schedule, sample_weights=sample_weights,
            )
            self.factor_head_dim = int(targets.shape[1])
        settings = self.block_settings
        if settings["weight"] > 0:
            columns = block_factor_columns(factors["active"], settings["count"])
            presence = factors["active"][:, columns]
            if settings["presence"] == "shuffled":
                # Control: each image carries the concept set of another image.
                presence = presence[torch.randperm(len(presence), generator=torch.Generator().manual_seed(SHUFFLE_SEED))]
            self.blocks = ConceptBlockContrast(
                presence.index_select(0, rows), settings["weight"], settings["temperature"],
                settings["context_weight"], *self.schedule,
            )
            self.factor_block_shape = (len(columns), settings["dim"])
            self.block_names = [factor_name(factors["factors"][column]) for column in columns]
            print(f"[INFO] Concept blocks ({settings['presence']} presence): {self.block_names}", flush=True)
        active = factors["active"].index_select(0, rows)[:, factors["condition_factors"]]
        self.sampler = ConceptConditionedBatchSampler(
            active, context.batch_size, self.condition_fraction, loader.generator,
        )
        indexed = IndexedCoSpRoDataset(loader.dataset)
        return DataLoader(
            indexed,
            batch_sampler=self.sampler,
            num_workers=context.num_workers,
            pin_memory=True,
            collate_fn=indexed.collate,
            worker_init_fn=context.worker_init_fn if context.num_workers > 0 else None,
            generator=loader.generator,
        )

    def set_epoch(self, epoch: int) -> None:
        for part in (self.regularizer, self.blocks):
            if part is not None:
                part.set_epoch(epoch)

    def extra_loss(self, *, model, embeddings, sample_indices) -> LossTerms | None:
        if self.regularizer is None and self.blocks is None:
            return None
        if sample_indices is None:
            raise ValueError("Concept-factor losses require sample indices.")
        value = embeddings.sum() * 0.0
        if self.regularizer is not None:
            if getattr(model, "factor_head", None) is None:
                raise ValueError("Factor distillation requires the model's factor head.")
            value = value + self.regularizer(model.factor_head(embeddings), sample_indices)
        if self.blocks is not None:
            if getattr(model, "factor_blocks", None) is None:
                raise ValueError("Concept blocks require the model's block heads.")
            count, dim = self.factor_block_shape
            value = value + self.blocks(model.factor_blocks(embeddings).view(-1, count, dim), sample_indices)
        return LossTerms(value=value, diagnostics=self.diagnostics())

    def diagnostics(self) -> Mapping[str, float]:
        values = dict(getattr(self.regularizer, "last_diagnostics", {}))
        values.update(getattr(self.blocks, "last_diagnostics", {}))
        if self.sampler is not None:
            values["factor_conditioned_batch_fraction"] = self.sampler.last_conditioned_fraction
        return values

    def provenance(self) -> dict[str, Any]:
        if self.paths is None:
            return {}
        return {
            "concept_factor_groups": str(self.paths[0]),
            "concept_factor_splice_cache": str(self.paths[1]),
            "concept_factor_count": len(self.report["factors"]),
            "concept_factor_pairs": [
                {"concepts": pair["concepts"], "phi": round(pair["phi"], 4)} for pair in self.report["pairs"]
            ],
            "concept_block_factors": self.block_names,
            "concept_factor_balancing": self.balancing,
        }

    def input_artifacts(self) -> list[Path]:
        # The cache is gigabytes on the slow scratch disk; the provenance names it instead.
        return [self.paths[0]] if self.paths else []
