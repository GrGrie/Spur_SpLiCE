from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
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
    score_cache_dir: str = "outputs/splice_score_cache"
    score_threshold: float | None = None
    score_reduction: str = "mean"
    batch_size: int = 128
    num_workers: int = 0
    conditional_on_target: bool = True
    intervention: str = "class_neutralize"
    intervention_strength: float = 1.0
    intervention_seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def dataset_score_cache_key(dataset_name: str, full_dataset, split: str) -> str:
    root_dir = getattr(full_dataset, "root_dir", "unknown")
    if dataset_name == "spur_cifar10":
        return (
            f"spur_cifar10_{split}_corr{full_dataset.train_spurious_correlation:g}_"
            f"spurSeed{full_dataset.spurious_seed}_line{full_dataset.line_width}_root{root_dir}"
        )
    return f"{dataset_name}_{split}_root{root_dir}"


def score_cache_path(
    config: SpliceConfig,
    dataset_size: int,
    concept_indices: Sequence[int],
    cache_key: str,
    artifact: str = "scores",
) -> Path:
    fingerprint = {
        "cache_version": 3,
        "artifact": artifact,
        "dataset": cache_key,
        "dataset_size": dataset_size,
        "model": config.model,
        "pretrained": config.pretrained,
        "vocab": config.vocab,
        "vocab_size": config.vocab_size,
        "l1_penalty": config.l1_penalty,
        # Column order is semantically significant for cached weight matrices:
        # it must remain aligned with the corresponding dictionary directions.
        "concept_indices": [int(index) for index in concept_indices],
    }
    if artifact == "scores":
        fingerprint["score_reduction"] = config.score_reduction
    if artifact == "synthesis_targets":
        fingerprint.update(
            {
                "intervention": config.intervention,
                "intervention_strength": config.intervention_strength,
                "intervention_seed": config.intervention_seed,
            }
        )
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    safe_key = "".join(character if character.isalnum() or character in "-_" else "_" for character in cache_key)
    return Path(config.score_cache_dir) / f"{safe_key}_{artifact}_{digest}.pt"


def save_score_cache(scores: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(scores.detach().cpu(), temporary_path)
    os.replace(temporary_path, path)
    print(f"[INFO] Cached SpLiCE scores at {path}", flush=True)


def splice_mode_uses_scores(mode: str) -> bool:
    return mode in {"augment", "corr_reg", "augment_corr_reg", "synthesis_distill"}


def splice_mode_uses_regularizer(mode: str) -> bool:
    return mode in {"corr_reg", "augment_corr_reg", "synthesis_distill"}


def residual_preserving_intervention(
    clip_embeddings: torch.Tensor,
    concept_weights: torch.Tensor,
    edited_weights: torch.Tensor,
    concept_directions: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Edit selected concept contributions while retaining the CLIP residual."""

    if not 0 <= strength <= 2:
        raise ValueError("SpLiCE intervention strength must be in [0, 2].")
    if concept_weights.shape != edited_weights.shape:
        raise ValueError("Original and edited SpLiCE weights must have the same shape.")
    if concept_weights.shape[1] != concept_directions.shape[0]:
        raise ValueError("Expected one dictionary direction per selected concept weight.")
    delta = (edited_weights - concept_weights) @ concept_directions
    return F.normalize(clip_embeddings + strength * delta, dim=1)


def edit_spurious_concept_weights(
    intervention: str,
    concept_weights: torch.Tensor,
    clip_embeddings: torch.Tensor,
    concept_directions: torch.Tensor,
    targets: torch.Tensor,
    spurious_values: torch.Tensor,
    seed: int = 0,
) -> torch.Tensor:
    """Construct auditable sparse-code edits for synthesized CLIP targets."""

    weights = concept_weights.detach().float().cpu()
    targets = targets.detach().long().cpu().view(-1)
    spurious_values = spurious_values.detach().long().cpu().view(-1)
    if intervention == "original":
        return weights.clone()
    if intervention == "zero_out":
        return torch.zeros_like(weights)
    if intervention in {"class_neutralize", "random_coords"}:
        edited = weights.clone()
        for target in torch.unique(targets):
            mask = targets == target
            edited[mask] = weights[mask].median(dim=0).values
        return edited
    if intervention not in {"core_matched_swap", "shuffled_donor", "same_class_random_donor"}:
        raise ValueError(f"Unsupported SpLiCE synthesis intervention: {intervention!r}")

    embeddings = clip_embeddings.detach().float().cpu()
    directions = concept_directions.detach().float().cpu()
    core = F.normalize(embeddings - weights @ directions, dim=1)
    edited = weights.clone()
    generator = torch.Generator().manual_seed(int(seed))
    for target in torch.unique(targets):
        target_mask = targets == target
        if intervention == "same_class_random_donor":
            target_indices = torch.where(target_mask)[0]
            if target_indices.numel() < 2:
                continue
            donor_offsets = torch.randint(
                1,
                target_indices.numel(),
                (target_indices.numel(),),
                generator=generator,
            )
            donor_positions = (torch.arange(target_indices.numel()) + donor_offsets) % target_indices.numel()
            edited[target_indices] = weights[target_indices[donor_positions]]
            continue
        for spurious_value in torch.unique(spurious_values[target_mask]):
            source_indices = torch.where(target_mask & (spurious_values == spurious_value))[0]
            donor_indices = torch.where(target_mask & (spurious_values != spurious_value))[0]
            if source_indices.numel() == 0 or donor_indices.numel() == 0:
                continue
            if intervention == "shuffled_donor":
                sampled = torch.randint(
                    donor_indices.numel(),
                    (source_indices.numel(),),
                    generator=generator,
                )
                chosen = donor_indices[sampled]
            else:
                chosen_chunks = []
                for source_chunk in source_indices.split(256):
                    similarity = core[source_chunk] @ core[donor_indices].T
                    chosen_chunks.append(donor_indices[similarity.argmax(dim=1)])
                chosen = torch.cat(chosen_chunks)
            edited[source_indices] = weights[chosen]
    return edited


def random_dictionary_indices(
    vocabulary_size: int,
    excluded_indices: Sequence[int],
    count: int,
    seed: int,
) -> list[int]:
    """Select deterministic non-spurious coordinates for a matched control."""

    excluded = set(int(index) for index in excluded_indices)
    candidates = torch.tensor(
        [index for index in range(vocabulary_size) if index not in excluded],
        dtype=torch.long,
    )
    if candidates.numel() < count:
        raise ValueError("Not enough non-selected SpLiCE coordinates for the random-coordinate control.")
    generator = torch.Generator().manual_seed(int(seed))
    chosen = candidates[torch.randperm(candidates.numel(), generator=generator)[:count]]
    return [int(index) for index in chosen.tolist()]


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
        return self.reduce_selected_weights(selected)

    def reduce_selected_weights(self, selected: torch.Tensor) -> torch.Tensor:
        if self.config.score_reduction == "mean":
            return selected.mean(dim=1)
        if self.config.score_reduction == "max":
            return selected.max(dim=1).values
        raise ValueError(f"Unsupported SpLiCE score reduction: {self.config.score_reduction}")

    def select_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Keep concept dimensions separate for vector-valued regularization."""

        return weights[:, self.concept_indices]

    @torch.no_grad()
    def score_images(self, images) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image) for image in images], dim=0).to(self.device)
        weights = self.model.encode_image(batch)
        return self.score_weights(weights).detach().cpu()

    @torch.no_grad()
    def concept_weights_images(
        self,
        images,
        concept_indices: Sequence[int] | None = None,
    ) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image) for image in images], dim=0).to(self.device)
        weights = self.model.encode_image(batch)
        indices = self.concept_indices if concept_indices is None else list(concept_indices)
        return weights[:, indices].detach().cpu()

    @torch.no_grad()
    def _score_cache_path(
        self,
        dataset,
        cache_key: str,
        artifact: str = "scores",
        concept_indices: Sequence[int] | None = None,
    ) -> Path:
        return score_cache_path(
            self.config,
            len(dataset),
            self.concept_indices if concept_indices is None else concept_indices,
            cache_key,
            artifact=artifact,
        )

    def concept_weights_dataset(
        self,
        dataset,
        batch_size: int | None = None,
        num_workers: int | None = None,
        cache_key: str | None = None,
        concept_indices: Sequence[int] | None = None,
        artifact: str = "concept_weights",
    ) -> torch.Tensor:
        """Return an ``[n_images, n_selected_concepts]`` weight matrix.

        Keeping this matrix instead of immediately reducing it to one scalar is
        essential: mutually exclusive concepts such as ``water`` and ``forest``
        otherwise become indistinguishable.
        """

        indices = self.concept_indices if concept_indices is None else list(concept_indices)
        cache_path = (
            self._score_cache_path(dataset, cache_key, artifact, concept_indices=indices)
            if cache_key
            else None
        )
        if cache_path is not None and cache_path.is_file():
            weights = torch.load(cache_path, map_location="cpu", weights_only=True)
            expected_shape = (len(dataset), len(indices))
            if isinstance(weights, torch.Tensor) and tuple(weights.shape) == expected_shape:
                print(f"[INFO] Loaded cached SpLiCE concept weights {expected_shape} from {cache_path}", flush=True)
                return weights.float()
            print(f"[WARNING] Ignoring invalid SpLiCE concept-weight cache at {cache_path}", flush=True)

        loader = DataLoader(
            dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers if num_workers is None else num_workers,
            collate_fn=identity_collate,
        )
        total_batches = len(loader)
        report_every = max(1, total_batches // 20)
        started_at = time.monotonic()
        print(
            f"[INFO] Precomputing SpLiCE concept vectors for {len(dataset)} images "
            f"({total_batches} batches). Training starts after this one-time step.",
            flush=True,
        )
        selected_weights = []
        for batch_index, batch in enumerate(loader, start=1):
            images = [item[0] for item in batch]
            selected_weights.append(self.concept_weights_images(images, indices))
            if batch_index == 1 or batch_index % report_every == 0 or batch_index == total_batches:
                elapsed = time.monotonic() - started_at
                print(
                    f"[INFO] SpLiCE scoring: {batch_index}/{total_batches} batches "
                    f"({100.0 * batch_index / total_batches:.1f}%, {elapsed:.1f}s)",
                    flush=True,
                )
        result = torch.cat(selected_weights, dim=0).float()
        if cache_path is not None:
            save_score_cache(result, cache_path)
        return result

    def clip_embeddings_dataset(
        self,
        dataset,
        batch_size: int | None = None,
        num_workers: int | None = None,
        cache_key: str | None = None,
    ) -> torch.Tensor:
        """Return frozen, normalized CLIP image embeddings in dataset order."""

        cache_path = self._score_cache_path(dataset, cache_key, "clip_embeddings") if cache_key else None
        if cache_path is not None and cache_path.is_file():
            embeddings = torch.load(cache_path, map_location="cpu", weights_only=True)
            if isinstance(embeddings, torch.Tensor) and embeddings.ndim == 2 and len(embeddings) == len(dataset):
                return embeddings.float()
            print(f"[WARNING] Ignoring invalid CLIP embedding cache at {cache_path}", flush=True)

        loader = DataLoader(
            dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers if num_workers is None else num_workers,
            collate_fn=identity_collate,
        )
        embeddings = []
        for batch in loader:
            images = torch.stack([self.preprocess(item[0]) for item in batch], dim=0).to(self.device)
            with torch.no_grad():
                encoded = self.model.clip.encode_image(images)
            embeddings.append(F.normalize(encoded.float(), dim=1).cpu())
        result = torch.cat(embeddings, dim=0)
        if cache_path is not None:
            save_score_cache(result, cache_path)
        return result

    def synthesis_targets_dataset(
        self,
        dataset,
        cache_key: str | None = None,
        spurious_metadata_index: int = 0,
    ) -> torch.Tensor:
        """Synthesize stop-gradient CLIP targets by editing selected sparse coordinates."""

        cache_path = self._score_cache_path(dataset, cache_key, "synthesis_targets") if cache_key else None
        if cache_path is not None and cache_path.is_file():
            targets = torch.load(cache_path, map_location="cpu", weights_only=True)
            if isinstance(targets, torch.Tensor) and targets.ndim == 2 and len(targets) == len(dataset):
                return targets.float()
            print(f"[WARNING] Ignoring invalid synthesis-target cache at {cache_path}", flush=True)

        clip_embeddings = self.clip_embeddings_dataset(dataset, cache_key=cache_key)
        labels = torch.as_tensor(dataset.y_array).long().view(-1)
        metadata = torch.as_tensor(dataset.metadata_array)
        if metadata.ndim != 2 or not 0 <= spurious_metadata_index < metadata.shape[1]:
            raise ValueError("SpLiCE synthesis requires a valid spurious metadata column.")
        spurious_values = metadata[:, spurious_metadata_index].long()
        edit_indices = self.concept_indices
        if self.config.intervention == "random_coords":
            edit_indices = random_dictionary_indices(
                len(self.model.dictionary),
                self.concept_indices,
                count=len(self.concept_indices),
                seed=self.config.intervention_seed,
            )
        concept_weights = self.concept_weights_dataset(
            dataset,
            cache_key=cache_key,
            concept_indices=edit_indices,
            artifact="concept_weights" if edit_indices == self.concept_indices else "random_concept_weights",
        )
        concept_directions = self.model.dictionary[edit_indices].detach().float().cpu()
        edited_weights = edit_spurious_concept_weights(
            self.config.intervention,
            concept_weights,
            clip_embeddings,
            concept_directions,
            labels,
            spurious_values,
            seed=self.config.intervention_seed,
        )
        result = residual_preserving_intervention(
            clip_embeddings,
            concept_weights,
            edited_weights,
            concept_directions,
            self.config.intervention_strength,
        ).cpu()
        if cache_path is not None:
            save_score_cache(result, cache_path)
        return result

    def score_dataset(
        self,
        dataset,
        batch_size: int | None = None,
        num_workers: int | None = None,
        cache_key: str | None = None,
    ) -> torch.Tensor:
        cache_path = self._score_cache_path(dataset, cache_key, "scores") if cache_key else None
        if cache_path is not None and cache_path.is_file():
            scores = torch.load(cache_path, map_location="cpu", weights_only=True)
            if isinstance(scores, torch.Tensor) and len(scores) == len(dataset):
                print(f"[INFO] Loaded {len(scores)} cached SpLiCE scores from {cache_path}", flush=True)
                return scores
            print(f"[WARNING] Ignoring invalid SpLiCE score cache at {cache_path}", flush=True)

        concept_weights = self.concept_weights_dataset(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            cache_key=cache_key,
        )
        result = self.reduce_selected_weights(concept_weights)
        if cache_path is not None:
            save_score_cache(result, cache_path)
        return result


class CorrelationSpliceRegularizer:
    enabled = True
    requires_clip_distillation = False

    def __init__(self, weight: float, conditional_on_target: bool = True) -> None:
        self.weight = weight
        self.conditional_on_target = conditional_on_target

    def __call__(
        self,
        embeddings: torch.Tensor,
        concept_weights: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if concept_weights is None:
            raise ValueError("SpLiCE correlation regularization requires per-sample concept vectors.")
        if self.weight <= 0:
            return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

        embeddings = embeddings.float()
        concept_weights = concept_weights.to(device=embeddings.device, dtype=embeddings.dtype)
        if concept_weights.ndim == 1:
            concept_weights = concept_weights.unsqueeze(1)
        if embeddings.shape[0] != concept_weights.shape[0]:
            raise ValueError(
                "Expected one SpLiCE concept vector per embedding, got "
                f"{concept_weights.shape[0]} vectors for {embeddings.shape[0]} embeddings."
            )
        if embeddings.shape[0] < 2:
            return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

        centered_embeddings = embeddings.clone()
        centered_concepts = concept_weights.clone()
        if self.conditional_on_target and targets is not None:
            targets = targets.to(device=embeddings.device).view(-1)
            if targets.shape[0] != embeddings.shape[0]:
                raise ValueError("Expected one target label per embedding for conditional regularization.")
            for target in torch.unique(targets):
                mask = targets == target
                centered_embeddings[mask] -= centered_embeddings[mask].mean(dim=0, keepdim=True)
                centered_concepts[mask] -= centered_concepts[mask].mean(dim=0, keepdim=True)
        else:
            centered_embeddings -= centered_embeddings.mean(dim=0, keepdim=True)
            centered_concepts -= centered_concepts.mean(dim=0, keepdim=True)

        feature_norms = centered_embeddings.norm(dim=0)
        concept_norms = centered_concepts.norm(dim=0)
        valid_features = feature_norms > 1e-12
        valid_concepts = concept_norms > 1e-12
        if not torch.any(valid_features) or not torch.any(valid_concepts):
            return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

        features = centered_embeddings[:, valid_features]
        concepts = centered_concepts[:, valid_concepts]
        correlations = features.T @ concepts
        denominator = feature_norms[valid_features].unsqueeze(1) * concept_norms[valid_concepts].unsqueeze(0)
        correlations = correlations / (denominator + 1e-12)
        return self.weight * correlations.pow(2).mean().to(dtype=embeddings.dtype)


class DisabledSpliceRegularizer:
    """No-op placeholder for future SpLiCE regularization/intervention work."""

    enabled = False
    requires_clip_distillation = False

    def __call__(
        self,
        embeddings: torch.Tensor,
        concept_weights: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)


class SpliceSynthesisDistillation:
    """Stop-gradient cosine distillation from a synthesized CLIP target."""

    enabled = True
    requires_clip_distillation = True

    def __init__(self, weight: float) -> None:
        self.weight = weight

    def __call__(
        self,
        embeddings: torch.Tensor,
        teacher_targets: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del targets
        if teacher_targets is None:
            raise ValueError("SpLiCE synthesis distillation requires one teacher target per view.")
        if embeddings.shape != teacher_targets.shape:
            raise ValueError(
                "g_clip predictions and synthesized CLIP targets must have the same shape, got "
                f"{tuple(embeddings.shape)} and {tuple(teacher_targets.shape)}."
            )
        teacher_targets = teacher_targets.to(device=embeddings.device, dtype=embeddings.dtype).detach()
        return self.weight * (1 - F.cosine_similarity(embeddings, teacher_targets, dim=1)).mean()


def build_splice_regularizer(config: SpliceConfig):
    if not config.use_splice or not splice_mode_uses_regularizer(config.mode):
        return DisabledSpliceRegularizer()
    if config.mode == "synthesis_distill":
        return SpliceSynthesisDistillation(config.splice_weight)
    return CorrelationSpliceRegularizer(
        config.splice_weight,
        conditional_on_target=config.conditional_on_target,
    )
