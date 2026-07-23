from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


DEFAULT_LOCALIZED_STAGES = ("stem", "layer1", "layer2", "layer3", "layer4", "encoder", "projection")


def _buffer_name(prefix: str, stage: str) -> str:
    return f"{prefix}_{stage}"


class LayerLocalizedScrubber(nn.Module):
    """Low-rank concept erasers installed at ResNet stage boundaries.

    Each eraser implements ``x <- x - alpha * x W^T (W W^T + eps I)^-1 W``.
    Readout and synthesis factors are cached separately, so convolutional maps
    are scrubbed with two inexpensive 1x1 convolutions rather than a dense
    channel-by-channel matrix multiplication.
    """

    def __init__(self, stage_dims: Mapping[str, int], max_concepts: int) -> None:
        super().__init__()
        if max_concepts <= 0:
            raise ValueError("max_concepts must be positive.")
        self.stage_dims = dict(stage_dims)
        self.max_concepts = int(max_concepts)
        self.runtime_enabled = True
        self.leakage_history: dict[str, dict[str, list[float]]] = {}
        self.onsets: dict[str, str] = {}
        self.last_probe_epoch = 0

        for stage, dimension in self.stage_dims.items():
            if dimension <= 0:
                raise ValueError(f"Stage {stage!r} has invalid dimension {dimension}.")
            self.register_buffer(
                _buffer_name("readout", stage),
                torch.zeros(self.max_concepts, dimension),
            )
            self.register_buffer(
                _buffer_name("synthesis", stage),
                torch.zeros(self.max_concepts, dimension),
            )
            self.register_buffer(_buffer_name("alpha", stage), torch.zeros(()))
            self.register_buffer(_buffer_name("rank", stage), torch.zeros((), dtype=torch.long))
            self.register_buffer(
                _buffer_name("concept_ids", stage),
                torch.full((self.max_concepts,), -1, dtype=torch.long),
            )

    def get_extra_state(self) -> dict:
        return {
            "leakage_history": self.leakage_history,
            "onsets": self.onsets,
            "last_probe_epoch": self.last_probe_epoch,
        }

    def set_extra_state(self, state: dict) -> None:
        state = state or {}
        self.leakage_history = state.get("leakage_history", {})
        self.onsets = state.get("onsets", {})
        self.last_probe_epoch = int(state.get("last_probe_epoch", 0))

    def clear_stage(self, stage: str) -> None:
        getattr(self, _buffer_name("readout", stage)).zero_()
        getattr(self, _buffer_name("synthesis", stage)).zero_()
        getattr(self, _buffer_name("alpha", stage)).zero_()
        getattr(self, _buffer_name("rank", stage)).zero_()
        getattr(self, _buffer_name("concept_ids", stage)).fill_(-1)

    @torch.no_grad()
    def update_stage(
        self,
        stage: str,
        directions: torch.Tensor,
        concept_ids: Iterable[int],
        alpha: float,
        projector_ridge: float,
    ) -> int:
        if stage not in self.stage_dims:
            raise KeyError(f"Unknown localized stage {stage!r}.")
        self.clear_stage(stage)
        directions = directions.detach().float()
        concept_ids = list(concept_ids)
        if not concept_ids or directions.numel() == 0 or alpha <= 0:
            return 0
        if directions.ndim != 2 or directions.shape[1] != self.stage_dims[stage]:
            raise ValueError(
                f"Directions for {stage} must have shape [n, {self.stage_dims[stage]}], "
                f"got {tuple(directions.shape)}."
            )
        if len(concept_ids) != directions.shape[0]:
            raise ValueError("Expected one concept id per direction.")
        if len(concept_ids) > self.max_concepts:
            raise ValueError("Too many localized concept directions.")

        nonzero = directions.norm(dim=1) > 1e-8
        directions = directions[nonzero]
        concept_ids = [concept_id for concept_id, keep in zip(concept_ids, nonzero.tolist()) if keep]
        if not concept_ids:
            return 0

        gram = directions @ directions.T
        gram = gram + projector_ridge * torch.eye(
            len(concept_ids),
            device=directions.device,
            dtype=directions.dtype,
        )
        synthesis = torch.linalg.solve(gram, directions)
        rank = len(concept_ids)
        readout_buffer = getattr(self, _buffer_name("readout", stage))
        synthesis_buffer = getattr(self, _buffer_name("synthesis", stage))
        readout_buffer[:rank].copy_(directions.to(readout_buffer))
        synthesis_buffer[:rank].copy_(synthesis.to(synthesis_buffer))
        getattr(self, _buffer_name("alpha", stage)).fill_(float(alpha))
        getattr(self, _buffer_name("rank", stage)).fill_(rank)
        concept_buffer = getattr(self, _buffer_name("concept_ids", stage))
        concept_buffer[:rank] = torch.tensor(concept_ids, device=concept_buffer.device)
        return rank

    def apply(self, stage: str, activations: torch.Tensor) -> torch.Tensor:
        if not self.runtime_enabled or stage not in self.stage_dims:
            return activations
        rank = int(getattr(self, _buffer_name("rank", stage)).item())
        alpha = getattr(self, _buffer_name("alpha", stage))
        if rank == 0 or float(alpha) <= 0:
            return activations
        readout = getattr(self, _buffer_name("readout", stage))[:rank].to(
            device=activations.device,
            dtype=activations.dtype,
        )
        synthesis = getattr(self, _buffer_name("synthesis", stage))[:rank].to(
            device=activations.device,
            dtype=activations.dtype,
        )
        if activations.ndim == 4:
            coefficients = F.conv2d(activations, readout[:, :, None, None])
            removed = F.conv2d(coefficients, synthesis.T[:, :, None, None])
        elif activations.ndim == 2:
            removed = (activations @ readout.T) @ synthesis
        else:
            raise ValueError(f"Cannot scrub {activations.ndim}D activations at stage {stage}.")
        return activations - alpha.to(dtype=activations.dtype) * removed


@dataclass(frozen=True)
class LocalizedProbeConfig:
    ridge: float = 1.0
    leakage_threshold: float = 0.05
    leakage_max: float = 0.50
    stability: int = 2
    shuffle_repeats: int = 3
    holdout_fraction: float = 0.25
    projector_ridge: float = 1e-4
    protect_target: bool = True
    seed: int = 0


def target_conditioned_residuals(
    train_values: np.ndarray,
    eval_values: np.ndarray,
    train_targets: np.ndarray,
    eval_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_residuals = train_values.copy()
    eval_residuals = eval_values.copy()
    global_mean = train_values.mean(axis=0, keepdims=True)
    for target in np.unique(train_targets):
        train_mask = train_targets == target
        eval_mask = eval_targets == target
        class_mean = train_values[train_mask].mean(axis=0, keepdims=True) if train_mask.any() else global_mean
        train_residuals[train_mask] -= class_mean
        eval_residuals[eval_mask] -= class_mean
    return train_residuals, eval_residuals


def _r2_columns(expected: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    residual_sum = np.square(expected - predicted).sum(axis=0)
    centered_sum = np.square(expected - expected.mean(axis=0, keepdims=True)).sum(axis=0)
    return np.where(centered_sum > 1e-12, 1.0 - residual_sum / centered_sum, 0.0)


def _fit_ridge(
    train_features: np.ndarray,
    train_values: np.ndarray,
    eval_features: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    estimator = Ridge(alpha=ridge, fit_intercept=True, solver="lsqr", tol=1e-5)
    estimator.fit(train_features, train_values)
    coefficients = np.asarray(estimator.coef_, dtype=np.float32)
    if coefficients.ndim == 1:
        coefficients = coefficients[None, :]
    predictions = np.asarray(estimator.predict(eval_features), dtype=np.float32)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    return coefficients, predictions


def estimate_leakage(
    features: np.ndarray,
    concept_values: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    ridge: float,
    shuffle_repeats: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_concepts, eval_concepts = target_conditioned_residuals(
        concept_values[train_indices],
        concept_values[eval_indices],
        targets[train_indices],
        targets[eval_indices],
    )
    coefficients, predictions = _fit_ridge(
        features[train_indices],
        train_concepts,
        features[eval_indices],
        ridge,
    )
    observed_r2 = _r2_columns(eval_concepts, predictions)
    shuffled_scores = []
    combined = np.concatenate((train_concepts, eval_concepts), axis=0)
    for _ in range(shuffle_repeats):
        shuffled = combined[rng.permutation(len(combined))]
        shuffled_train = shuffled[: len(train_indices)]
        shuffled_eval = shuffled[len(train_indices) :]
        _, shuffled_predictions = _fit_ridge(
            features[train_indices],
            shuffled_train,
            features[eval_indices],
            ridge,
        )
        shuffled_scores.append(_r2_columns(shuffled_eval, shuffled_predictions))
    shuffled_r2 = np.mean(shuffled_scores, axis=0)
    return observed_r2 - shuffled_r2, coefficients, shuffled_r2


def orthogonalize_against_target(
    directions: torch.Tensor,
    target_directions: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    if target_directions.numel() == 0:
        return directions
    target_gram = target_directions @ target_directions.T
    target_gram = target_gram + ridge * torch.eye(
        target_gram.shape[0],
        dtype=target_gram.dtype,
        device=target_gram.device,
    )
    target_coordinates = directions @ target_directions.T
    projected = target_coordinates @ torch.linalg.solve(target_gram, target_directions)
    return directions - projected


def _unwrap_scrubber(model) -> LayerLocalizedScrubber:
    encoder = model.encoder.module if isinstance(model.encoder, nn.DataParallel) else model.encoder
    scrubber = getattr(encoder, "localized_scrubber", None)
    if scrubber is None:
        raise ValueError("The model has no layer-localized scrubber.")
    return scrubber


@torch.no_grad()
def collect_probe_activations(
    model,
    rank_loader,
    concept_weights: torch.Tensor,
    device: torch.device,
    max_samples: int,
    seed: int,
    channels_last: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if len(concept_weights) != len(rank_loader.dataset):
        raise ValueError(
            "Localized probes require concept weights aligned with the ordered rank loader: "
            f"{len(concept_weights)} weights for {len(rank_loader.dataset)} samples."
        )
    scrubber = _unwrap_scrubber(model)
    was_training = model.training
    previous_runtime_state = scrubber.runtime_enabled
    model.eval()
    scrubber.runtime_enabled = False
    collected: dict[str, list[torch.Tensor]] = {}
    collected_targets = []
    sample_generator = torch.Generator().manual_seed(seed)
    selected_indices = torch.randperm(len(rank_loader.dataset), generator=sample_generator)[:max_samples]
    selected_indices = selected_indices.sort().values
    selection_offset = 0
    dataset_offset = 0
    try:
        for batch in rank_loader:
            batch_size = len(batch[0])
            batch_end = dataset_offset + batch_size
            selected_end = selection_offset
            while (
                selected_end < len(selected_indices)
                and int(selected_indices[selected_end]) < batch_end
            ):
                selected_end += 1
            if selected_end == selection_offset:
                dataset_offset = batch_end
                continue
            local_indices = selected_indices[selection_offset:selected_end] - dataset_offset
            images = batch[0][local_indices].to(device, non_blocking=True)
            if channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            targets = batch[1][local_indices]
            _, activations = model.forward_with_intermediates(images)
            for stage, values in activations.items():
                collected.setdefault(stage, []).append(values.detach().float().cpu())
            collected_targets.append(targets.detach().cpu())
            selection_offset = selected_end
            dataset_offset = batch_end
            if selection_offset >= len(selected_indices):
                break
    finally:
        scrubber.runtime_enabled = previous_runtime_state
        model.train(was_training)
    if selection_offset < 4:
        raise ValueError("Localized leakage probing needs at least four samples.")
    features = {
        stage: torch.cat(values, dim=0).numpy()
        for stage, values in collected.items()
    }
    return (
        features,
        concept_weights[selected_indices[:selection_offset]].float().numpy(),
        torch.cat(collected_targets).numpy(),
    )


def update_localized_scrubber(
    model,
    rank_loader,
    concept_weights: torch.Tensor,
    concept_names: list[str],
    epoch: int,
    device: torch.device,
    max_samples: int,
    config: LocalizedProbeConfig,
    diagnostics_path: str | Path,
    channels_last: bool = False,
) -> dict[str, float]:
    scrubber = _unwrap_scrubber(model)
    features, concepts, targets = collect_probe_activations(
        model,
        rank_loader,
        concept_weights,
        device,
        max_samples,
        config.seed + epoch * 97,
        channels_last=channels_last,
    )
    stage_order = [stage for stage in DEFAULT_LOCALIZED_STAGES if stage in features and stage in scrubber.stage_dims]
    rng = np.random.default_rng(config.seed + epoch * 1_000_003)
    eval_size = max(1, int(round(len(targets) * config.holdout_fraction)))
    eval_size = min(eval_size, len(targets) - 2)
    indices = np.arange(len(targets))
    try:
        train_indices, eval_indices = train_test_split(
            indices,
            test_size=eval_size,
            random_state=config.seed + epoch,
            shuffle=True,
            stratify=targets,
        )
    except ValueError:
        permutation = rng.permutation(len(targets))
        eval_indices = permutation[:eval_size]
        train_indices = permutation[eval_size:]

    leakage: dict[str, np.ndarray] = {}
    coefficients: dict[str, np.ndarray] = {}
    shuffled: dict[str, np.ndarray] = {}
    target_coefficients: dict[str, np.ndarray] = {}
    target_one_hot = np.eye(int(targets.max()) + 1, dtype=np.float32)[targets]
    for stage in stage_order:
        leakage[stage], coefficients[stage], shuffled[stage] = estimate_leakage(
            features[stage],
            concepts,
            targets,
            train_indices,
            eval_indices,
            config.ridge,
            config.shuffle_repeats,
            rng,
        )
        target_coefficients[stage], _ = _fit_ridge(
            features[stage][train_indices],
            target_one_hot[train_indices],
            features[stage][eval_indices],
            config.ridge,
        )

    for concept_index in range(concepts.shape[1]):
        concept_history = scrubber.leakage_history.setdefault(str(concept_index), {})
        for stage in stage_order:
            values = concept_history.setdefault(stage, [])
            values.append(float(leakage[stage][concept_index]))
            del values[: -max(config.stability, 1)]
        if str(concept_index) in scrubber.onsets:
            continue
        for stage in stage_order:
            values = concept_history[stage]
            if len(values) >= config.stability and min(values[-config.stability :]) >= config.leakage_threshold:
                scrubber.onsets[str(concept_index)] = stage
                break

    stage_details = {}
    scalar_metrics: dict[str, float] = {}
    for stage in stage_order:
        concept_ids = sorted(
            int(concept_id)
            for concept_id, onset in scrubber.onsets.items()
            if onset == stage
        )
        if concept_ids:
            directions = torch.from_numpy(coefficients[stage][concept_ids]).float()
            if config.protect_target:
                directions = orthogonalize_against_target(
                    directions,
                    torch.from_numpy(target_coefficients[stage]).float(),
                    config.projector_ridge,
                )
            current_leakage = float(np.mean(leakage[stage][concept_ids]))
            alpha = float(
                np.clip(
                    (current_leakage - config.leakage_threshold)
                    / (config.leakage_max - config.leakage_threshold),
                    0.0,
                    1.0,
                )
            )
            rank = scrubber.update_stage(
                stage,
                directions.to(device),
                concept_ids,
                alpha,
                config.projector_ridge,
            )
        else:
            scrubber.clear_stage(stage)
            current_leakage = 0.0
            alpha = 0.0
            rank = 0
        stage_details[stage] = {
            "concept_ids": concept_ids,
            "concepts": [concept_names[index] for index in concept_ids],
            "alpha": alpha,
            "rank": rank,
            "assigned_mean_leakage": current_leakage,
        }
        scalar_metrics[f"localized/{stage}/alpha"] = alpha
        scalar_metrics[f"localized/{stage}/rank"] = float(rank)
        scalar_metrics[f"localized/{stage}/mean_leakage"] = float(np.mean(leakage[stage]))

    scrubber.last_probe_epoch = epoch
    payload = {
        "epoch": epoch,
        "num_samples": len(targets),
        "holdout_samples": len(eval_indices),
        "concepts": concept_names,
        "stage_order": stage_order,
        "leakage": {stage: values.tolist() for stage, values in leakage.items()},
        "shuffled_r2": {stage: values.tolist() for stage, values in shuffled.items()},
        "onsets": {
            concept_names[int(concept_id)]: stage
            for concept_id, stage in sorted(scrubber.onsets.items(), key=lambda item: int(item[0]))
        },
        "stages": stage_details,
    }
    diagnostics_path = Path(diagnostics_path)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True))
        file.write("\n")
    print(
        "[localized] epoch={} onsets={} alphas={}".format(
            epoch,
            payload["onsets"],
            {stage: round(details["alpha"], 4) for stage, details in stage_details.items() if details["rank"]},
        ),
        flush=True,
    )
    scalar_metrics["localized/onset_count"] = float(len(scrubber.onsets))
    return scalar_metrics
