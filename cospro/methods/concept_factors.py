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
* F2 with cross-fitting (``cross_fit``). Instead of a trained head, a ridge regression fitted in closed
  form on one half of the batch predicts the factors of the other half, and the loss is that held-out
  error. A feature that only identifies an image, which lets a trained head memorize any per-image
  target, carries nothing from one half to the other; only concept directions that transfer between
  images lower the loss. With sample weights in the fit and the loss, a direction that fuses a
  concept with its usual context mispredicts exactly the heavily weighted images that break the
  co-occurrence.
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
from cospro.data.transforms import RecordedTwoCropTransform
from cospro.methods.relational_graph import IndexedCoSpRoDataset
from cospro.pipeline.concept_maps import spatial_cross_fit_loss, warp_maps
from cospro.pipeline.concept_factors import (
    LEARNABILITY_RIDGE,
    SHUFFLE_SEED,
    explained_variance,
    group_explained_variance,
    heldout_explained_variance,
    out_of_fold_predictions,
    partial_residuals,
    FactorConfig,
    balancing_weights,
    block_factor_columns,
    factor_name,
    expected_condition_pool,
    factor_report,
    format_factor_report,
    load_concept_factors,
    rows_for_subset,
    unseen_explained_variance,
    unseen_predictions,
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


def cross_fit_predictions(features: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor,
                          fit: torch.Tensor, ridge: float) -> torch.Tensor:
    """Predict the targets of the rows outside ``fit`` with a weighted ridge regression fitted on ``fit``.

    Kernel form: with the fitted rows H, their targets T and weights D, the prediction for rows G is
    G H' (D H H' + lambda I)^-1 D T, which equals the primal weighted ridge solution and needs only an
    n-by-n solve for n fitted rows. Every feature row is scaled to unit norm first, so the backbone
    cannot weaken the ridge by shrinking its features, and the system's eigenvalues stay at least
    lambda: the solve is well conditioned. Features and targets are then centred on the weighted mean
    of the fitted rows, so the regression has an intercept. Gradients flow through the solve.
    """

    held_out = ~fit
    features = torch.nn.functional.normalize(features, dim=1)
    fitted, other = features[fit], features[held_out]
    fit_weights = weights[fit] / weights[fit].mean().clamp_min(1e-12)
    feature_mean = (fit_weights[:, None] * fitted).mean(dim=0)
    target_mean = (fit_weights[:, None] * targets[fit]).mean(dim=0)
    fitted, other = fitted - feature_mean, other - feature_mean
    kernel = fitted @ fitted.T
    system = fit_weights[:, None] * kernel + ridge * torch.eye(len(fitted), device=features.device)
    coefficients = torch.linalg.solve(system, fit_weights[:, None] * (targets[fit] - target_mean))
    return other @ fitted.T @ coefficients + target_mean


class CrossFitDistillation:
    """F2 with cross-fitting: each half of the batch is predicted by a ridge regression fitted on the other."""

    def __init__(self, targets: torch.Tensor, weight: float, start_epoch: int, warmup_epochs: int,
                 sample_weights: torch.Tensor | None = None, ridge: float = 1.0) -> None:
        if weight < 0 or start_epoch < 0 or warmup_epochs < 0 or ridge <= 0:
            raise ValueError("Cross-fitted distillation needs non-negative schedule values and a positive ridge.")
        self.targets = torch.as_tensor(targets).detach().float().cpu()
        self.sample_weights = None if sample_weights is None else torch.as_tensor(sample_weights).float().cpu()
        self.weight = float(weight)
        self.start_epoch = int(start_epoch)
        self.warmup_epochs = int(warmup_epochs)
        self.ridge = float(ridge)
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

    def __call__(self, embeddings: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
        rows = torch.as_tensor(sample_indices, dtype=torch.long).detach().cpu().view(-1)
        count = len(rows)
        if count < 4:
            self.last_diagnostics = {"factor_scheduled_weight": self.scheduled_weight}
            return embeddings.sum() * 0.0
        device = embeddings.device
        targets = self.targets.index_select(0, rows).to(device)
        weights = (torch.ones(count) if self.sample_weights is None else self.sample_weights.index_select(0, rows)).to(device)
        # Both views of an image stay in one half, so no half sees a view of an image it predicts.
        half = torch.zeros(count, dtype=torch.bool, device=device)
        half[: count // 2] = True
        features = embeddings.float()
        targets, weights, half = torch.cat([targets, targets]), torch.cat([weights, weights]), torch.cat([half, half])
        loss = embeddings.sum() * 0.0
        errors = []
        for fit in (half, ~half):
            predictions = cross_fit_predictions(features, targets, weights, fit, self.ridge)
            per_image = (predictions - targets[~fit]).pow(2).mean(dim=1)
            held_out_weights = weights[~fit] / weights[~fit].mean().clamp_min(1e-12)
            loss = loss + (held_out_weights * per_image).mean() / 2
            errors.append(per_image.detach().mean())
        mse = float(torch.stack(errors).mean())
        # Targets have unit variance over the dataset, so this is the held-out explained variance.
        self.last_diagnostics = {
            "factor_scheduled_weight": self.scheduled_weight,
            "factor_heldout_mse": mse,
            "factor_heldout_explained_variance": 1.0 - mse,
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

    def __call__(self, predictions: torch.Tensor, sample_indices: torch.Tensor,
                 embeddings: torch.Tensor | None = None) -> torch.Tensor:
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
        if embeddings is not None and len(rows) >= 4:
            with torch.no_grad():
                count = len(rows)
                half = torch.zeros(count, dtype=torch.bool, device=predictions.device)
                half[: count // 2] = True
                half = torch.cat([half, half])
                predicted = cross_fit_predictions(embeddings.float(), targets, torch.ones_like(half, dtype=torch.float),
                                                  half, 1.0)
                heldout = float((predicted - targets[~half]).pow(2).mean())
            self.last_diagnostics["factor_heldout_explained_variance"] = 1.0 - heldout
        return self.scheduled_weight * loss


class IndexedViewBoxDataset(IndexedCoSpRoDataset):
    """Two augmented views, the sample row and each view's crop box and flip, [2, 5]."""

    def __init__(self, dataset) -> None:
        super().__init__(dataset)
        two_crop = self.transform
        if not hasattr(two_crop, "transform"):
            raise ValueError("Spatial concept maps need the SSL two-view transform.")
        self.recorded = RecordedTwoCropTransform(two_crop.transform)

    def __getitem__(self, index: int):
        image = self.source_dataset.get_input(int(self.source_indices[index]))
        views, boxes = self.recorded(image)
        return views, int(index), boxes


class SpatialConceptDistillation:
    """Scheduled per-location cross-fit from the encoder's last feature map to the warped concept maps."""

    def __init__(self, maps: torch.Tensor, weight: float, start_epoch: int, warmup_epochs: int,
                 sample_weights: torch.Tensor, ridge: float) -> None:
        self.maps = maps
        self.sample_weights = sample_weights.float()
        self.weight = float(weight)
        self.start_epoch = int(start_epoch)
        self.warmup_epochs = int(warmup_epochs)
        self.ridge = float(ridge)
        self.epoch = 0
        self.feature_map: torch.Tensor | None = None
        self.boxes: torch.Tensor | None = None
        self._hooked = False
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

    def hook(self, model) -> None:
        """Keep the output of the encoder's last stage, the feature map before global pooling."""

        if self._hooked:
            return
        encoder = getattr(model.encoder, "module", model.encoder)
        stage = getattr(encoder, "layer4", None)
        if stage is None:
            raise ValueError("The spatial concept loss needs a ResNet encoder with a layer4 stage.")
        stage.register_forward_hook(lambda module, inputs, output: setattr(self, "feature_map", output))
        self._hooked = True

    def __call__(self, embeddings: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
        feature_map, boxes = self.feature_map, self.boxes
        if feature_map is None or boxes is None or feature_map.shape[0] != embeddings.shape[0]:
            # The hook attaches during the first step, whose forward pass has already run.
            self.last_diagnostics = {"factor_spatial_scheduled_weight": self.scheduled_weight}
            return embeddings.sum() * 0.0
        rows = torch.as_tensor(sample_indices, dtype=torch.long).detach().cpu().view(-1)
        device = feature_map.device
        maps = self.maps.index_select(0, rows).float().to(device)
        boxes = torch.cat([boxes[:, 0], boxes[:, 1]]).to(device)
        targets = warp_maps(torch.cat([maps, maps]), boxes, tuple(feature_map.shape[-2:]))
        weights = self.sample_weights.index_select(0, rows).to(device)
        loss, explained = spatial_cross_fit_loss(feature_map, targets, torch.cat([weights, weights]), self.ridge)
        self.last_diagnostics = {
            "factor_spatial_scheduled_weight": self.scheduled_weight,
            "factor_spatial_heldout_explained_variance": explained,
        }
        return self.scheduled_weight * loss


def load_concept_maps(path: str, factor_names: list[str], dataset: str, source_indices,
                      shuffled: bool) -> torch.Tensor:
    """Concept maps of the training subset in its order, each factor standardized over images and locations."""

    stored = torch.load(path, map_location="cpu", weights_only=False)
    if list(stored["factor_names"]) != list(factor_names):
        raise ValueError(f"Concept maps {path} were built for other factors than this run's.")
    maps = stored["maps"].float()
    mean = maps.mean(dim=(0, 2, 3), keepdim=True)
    std = maps.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    maps = (maps - mean) / std
    if shuffled:
        # Control: every image carries another image's maps.
        maps = maps[torch.randperm(len(maps), generator=torch.Generator().manual_seed(SHUFFLE_SEED))]
    rows = rows_for_subset([str(sample_id) for sample_id in stored["sample_ids"]], dataset, source_indices)
    return maps.index_select(0, rows).half()


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
        cross_fit: bool = False,
        ridge: float = 1.0,
        holdout_fraction: float = 0.0,
        spatial_weight: float = 0.0,
        concept_maps: str = "",
        spatial_maps: str = "real",
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
        self.cross_fit = bool(cross_fit)
        self.ridge = float(ridge)
        self.holdout_fraction = float(holdout_fraction)
        self.spatial_settings = {"weight": float(spatial_weight), "maps": concept_maps, "kind": spatial_maps}
        self.spatial: SpatialConceptDistillation | None = None
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
        self.learnability: dict[str, Any] = {}
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
                pca_components=options.factor_pca_components,
            ),
            condition_fraction=options.factor_condition_fraction,
            distill_weight=options.factor_distill_weight,
            target_kind=options.factor_targets,
            sample_weighting=options.factor_sample_weighting,
            cross_fit=options.factor_cross_fit,
            ridge=options.factor_ridge,
            holdout_fraction=options.factor_holdout_fraction,
            spatial_weight=options.factor_spatial_weight,
            concept_maps=options.factor_concept_maps,
            spatial_maps=options.factor_spatial_maps,
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
        # The same images stay out of the concept loss in every arm and seed, so their scores compare.
        unseen = torch.rand(len(factors["sample_ids"]), generator=torch.Generator().manual_seed(SHUFFLE_SEED))
        unseen = (unseen < self.holdout_fraction).index_select(0, rows)
        self._prepare_learnability(factors, rows, source_indices, unseen, context.dataset)
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
            if unseen.any():
                sample_weights = torch.ones(len(rows)) if sample_weights is None else sample_weights.clone()
                sample_weights[unseen] = 0.0
                sample_weights = sample_weights / sample_weights.mean()
            if self.cross_fit:
                self.regularizer = CrossFitDistillation(
                    targets, self.distill_weight, *self.schedule, sample_weights=sample_weights, ridge=self.ridge,
                )
            else:
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
        spatial = self.spatial_settings
        if spatial["weight"] > 0:
            names = [factor_name(factor) for factor in factors["factors"]]
            maps = load_concept_maps(spatial["maps"], names, context.dataset, source_indices,
                                     shuffled=spatial["kind"] == "shuffled")
            self.spatial = SpatialConceptDistillation(
                maps, spatial["weight"], *self.schedule, sample_weights=(~unseen).float(), ridge=self.ridge,
            )
            print(f"[INFO] Spatial concept maps ({spatial['kind']}) from {spatial['maps']}: {tuple(maps.shape)}",
                  flush=True)
        active = factors["active"].index_select(0, rows)[:, factors["condition_factors"]]
        self.sampler = ConceptConditionedBatchSampler(
            active, context.batch_size, self.condition_fraction, loader.generator,
        )
        indexed = IndexedViewBoxDataset(loader.dataset) if self.spatial is not None else IndexedCoSpRoDataset(loader.dataset)
        return DataLoader(
            indexed,
            batch_sampler=self.sampler,
            num_workers=context.num_workers,
            pin_memory=True,
            collate_fn=indexed.collate,
            worker_init_fn=context.worker_init_fn if context.num_workers > 0 else None,
            generator=loader.generator,
        )

    def _prepare_learnability(self, factors: dict, rows: torch.Tensor, source_indices, unseen: torch.Tensor,
                              dataset: str = "") -> None:
        """The real targets, their residuals and the CLIP features, whatever targets the loss uses."""

        # Response targets and spatial maps are scored as responses, every other kind against the sparse
        # factor values.
        spatial = self.spatial_settings["weight"] > 0
        base = "response" if self.target_kind.startswith("response") or spatial else "standardized"
        self.learnability = {
            "dataset": dataset,
            "scored_targets": base,
            "source_indices": [int(index) for index in source_indices],
            "targets": factors["targets"][base].index_select(0, rows),
            # Each factor minus its regression on the others: one direction shared by two co-occurring
            # factors predicts their common part and misses this one.
            "residuals": partial_residuals(factors["targets"][base]).index_select(0, rows),
            "clip_features": factors["clip_embeddings"].index_select(0, rows),
            "unseen": unseen,
            "factors": [
                {"name": factor_name(factor), "concepts": factor["concepts"], "frequency": round(factor["frequency"], 4)}
                for factor in factors["factors"]
            ],
        }
        self._clip_scores = {}

    def _scores(self, features, targets, residuals, unseen, groups) -> dict[str, Any]:
        """Explained variance over all images, on unseen images, of the residuals and inside each group."""

        scores: dict[str, Any] = {"all": heldout_explained_variance(features, targets)}
        if unseen.any():
            predicted = unseen_predictions(features, targets, unseen)
            predicted_residuals = unseen_predictions(features, residuals, unseen)
            rows = unseen
            scores["unseen"] = explained_variance(predicted, targets[unseen])
        else:
            predicted = out_of_fold_predictions(features, targets)
            predicted_residuals = out_of_fold_predictions(features, residuals)
            rows = torch.ones(len(targets), dtype=torch.bool)
        scores["residual"] = explained_variance(predicted_residuals, residuals[rows])
        if groups is not None:
            scores["groups"] = group_explained_variance(predicted, targets[rows], groups[rows])
        return scores

    def factor_learnability(self, features, source_indices, groups=None, group_names=None) -> dict[str, Any] | None:
        """How well ``features`` encode every real factor, beside the same scores of CLIP features.

        ``student`` is the five-fold score over the training images. With a hold-out, ``student_unseen``
        scores the images the concept loss never saw. ``student_residual`` scores each factor's part
        the other factors leave unexplained, a label-free test that the factor has its own direction.
        ``student_groups`` scores the factor inside each (class, attribute) group: a factor fused with
        its usual context drops in the groups that break the co-occurrence. Groups enter only this
        evaluation. Shuffled-target arms score the real factors too, so their record shows what SimCLR
        learns of the concepts without the concept loss.
        """

        if not self.learnability:
            return None
        position = {index: row for row, index in enumerate(self.learnability["source_indices"])}
        order = torch.tensor([position[int(index)] for index in source_indices], dtype=torch.long)
        targets = self.learnability["targets"].index_select(0, order)
        residuals = self.learnability["residuals"].index_select(0, order)
        unseen = self.learnability["unseen"].index_select(0, order)
        groups = None if groups is None else torch.as_tensor(groups, dtype=torch.long)
        student = self._scores(features, targets, residuals, unseen, groups)
        key = (len(order), int(order[:64].sum()), groups is None)
        if key not in self._clip_scores:
            clip_features = self.learnability["clip_features"].index_select(0, order)
            self._clip_scores[key] = self._scores(clip_features, targets, residuals, unseen, groups)
        clip = self._clip_scores[key]

        def name(group: int) -> str:
            return group_names[group] if group_names is not None and group < len(group_names) else str(group)

        entries = []
        for column, factor in enumerate(self.learnability["factors"]):
            entry = {**factor}
            for label, scores in (("student", student), ("clip", clip)):
                entry[label] = round(float(scores["all"][column]), 4)
                if "unseen" in scores:
                    entry[f"{label}_unseen"] = round(float(scores["unseen"][column]), 4)
                entry[f"{label}_residual"] = round(float(scores["residual"][column]), 4)
                if "groups" in scores:
                    entry[f"{label}_groups"] = {name(group): round(float(values[column]), 4)
                                                for group, values in scores["groups"].items()}
            entries.append(entry)
        entries.sort(key=lambda entry: -entry["clip"])

        summary: dict[str, Any] = {
            "ridge": LEARNABILITY_RIDGE,
            "scored_targets": self.learnability.get("scored_targets", "standardized"),
            "evaluated_on": "unseen images" if unseen.any() else "out-of-fold training images",
            "student_mean": float(student["all"].mean()),
            "clip_mean": float(clip["all"].mean()),
            # A factor counts as learned once the student reaches half of what CLIP features explain.
            "learned_fraction": float((student["all"] >= 0.5 * clip["all"].clamp_min(0)).float().mean()),
            "student_residual_mean": float(student["residual"].mean()),
            "clip_residual_mean": float(clip["residual"].mean()),
        }
        if unseen.any():
            summary.update({
                "unseen_images": int(unseen.sum()),
                "student_unseen_mean": float(student["unseen"].mean()),
                "clip_unseen_mean": float(clip["unseen"].mean()),
            })
        if groups is not None:
            for label, scores in (("student", student), ("clip", clip)):
                stacked = torch.stack(list(scores["groups"].values()))
                summary[f"{label}_group_means"] = {name(group): float(values.mean())
                                                   for group, values in scores["groups"].items()}
                # Per factor its weakest group, averaged over the factors.
                summary[f"{label}_worst_group_mean"] = float(stacked.min(dim=0).values.mean())
            evaluated = groups[unseen] if unseen.any() else groups
            summary["group_counts"] = {name(int(group)): int((evaluated == group).sum())
                                       for group in torch.unique(evaluated).tolist()}
        return {**summary, "factors": entries}

    def set_epoch(self, epoch: int) -> None:
        for part in (self.regularizer, self.blocks, self.spatial):
            if part is not None:
                part.set_epoch(epoch)

    def observe_batch(self, batch) -> None:
        if self.spatial is not None:
            self.spatial.boxes = batch[2] if len(batch) > 2 else None

    def extra_loss(self, *, model, embeddings, sample_indices) -> LossTerms | None:
        if self.regularizer is None and self.blocks is None and self.spatial is None:
            return None
        if sample_indices is None:
            raise ValueError("Concept-factor losses require sample indices.")
        value = embeddings.sum() * 0.0
        if isinstance(self.regularizer, CrossFitDistillation):
            value = value + self.regularizer(embeddings, sample_indices)
        elif self.regularizer is not None:
            if getattr(model, "factor_head", None) is None:
                raise ValueError("Factor distillation requires the model's factor head.")
            value = value + self.regularizer(model.factor_head(embeddings), sample_indices, embeddings.detach())
        if self.blocks is not None:
            if getattr(model, "factor_blocks", None) is None:
                raise ValueError("Concept blocks require the model's block heads.")
            count, dim = self.factor_block_shape
            value = value + self.blocks(model.factor_blocks(embeddings).view(-1, count, dim), sample_indices)
        if self.spatial is not None:
            self.spatial.hook(model)
            value = value + self.spatial(embeddings, sample_indices)
            self.spatial.feature_map = None
        return LossTerms(value=value, diagnostics=self.diagnostics())

    def diagnostics(self) -> Mapping[str, float]:
        values = dict(getattr(self.regularizer, "last_diagnostics", {}))
        values.update(getattr(self.blocks, "last_diagnostics", {}))
        values.update(getattr(self.spatial, "last_diagnostics", {}))
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
