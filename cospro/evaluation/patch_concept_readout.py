"""Patch-concept readout of a frozen encoder: which concepts its patch tokens carry and which decide.

Three measurements on frozen patch tokens [images, patches, dim] and CLIP concept maps
[images, factors, side, side] (``cospro.cli.build_concept_maps`` on the concept-factors branch):

1. **Readout.** A ridge regression from every patch token to the concept probabilities at that patch,
   cross-fitted over image folds, gives the held-out explained variance per factor. Two controls bound
   it: the same ridge on the image's mean-pooled token (image-level information only) and the ridge on
   maps shuffled across images (the marginal map alone).
2. **Concept bottleneck.** The ridge turns any image into predicted concept maps; their patch mean is
   a concept vector that a group-balanced linear probe classifies. Its standardized weights name the
   concepts the decision uses.
3. **Concept evidence.** A linear probe on mean-pooled tokens gives every patch an evidence value, the
   margin of that patch alone. Averaging it over each concept's patches, weighted by the predicted
   probabilities, gives the evidence per patch of the concept, independent of the area it covers.
   Signed towards the true class, a negative value means the concept's regions argue for the wrong
   class; on misclassified images it names the concepts behind the errors.

The ridge has an intercept and the maps sum to one over the factors at every patch, so the predicted
concepts also sum to one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def resize_maps(maps: torch.Tensor, side: int) -> torch.Tensor:
    """Concept maps [images, factors, s, s] resampled to the encoder's patch grid, [images, side*side, factors].

    Bilinear resampling is affine with weights summing to one, so probabilities still sum to one.
    """

    resized = F.interpolate(maps.float(), size=(side, side), mode="bilinear", align_corners=False)
    return resized.flatten(2).transpose(1, 2).contiguous()


def image_folds(images: int, folds: int, seed: int = 0) -> torch.Tensor:
    """A fold id per image from a seeded permutation, sizes differing by at most one."""

    if folds < 2 or folds > images:
        raise ValueError("Cross-fitting needs 2 <= folds <= images.")
    order = torch.randperm(images, generator=torch.Generator().manual_seed(seed))
    assignment = torch.empty(images, dtype=torch.long)
    assignment[order] = torch.arange(images) % folds
    return assignment


@dataclass
class Moments:
    """Sufficient statistics of a ridge with intercept: sums, Gram matrix and cross moments."""

    count: float
    feature_sum: torch.Tensor
    target_sum: torch.Tensor
    gram: torch.Tensor
    cross: torch.Tensor

    @classmethod
    def of(cls, features: torch.Tensor, targets: torch.Tensor) -> "Moments":
        x, y = features.double(), targets.double()
        return cls(float(len(x)), x.sum(0), y.sum(0), x.T @ x, x.T @ y)

    def __add__(self, other: "Moments") -> "Moments":
        return Moments(self.count + other.count, self.feature_sum + other.feature_sum,
                       self.target_sum + other.target_sum, self.gram + other.gram, self.cross + other.cross)

    def __sub__(self, other: "Moments") -> "Moments":
        return Moments(self.count - other.count, self.feature_sum - other.feature_sum,
                       self.target_sum - other.target_sum, self.gram - other.gram, self.cross - other.cross)

    def solve(self, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Weights [dim, factors] and intercept [factors] of the centred ridge."""

        feature_mean = self.feature_sum / self.count
        target_mean = self.target_sum / self.count
        gram = self.gram - self.count * torch.outer(feature_mean, feature_mean)
        cross = self.cross - self.count * torch.outer(feature_mean, target_mean)
        identity = torch.eye(len(gram), dtype=gram.dtype, device=gram.device)
        weights = torch.linalg.solve(gram + ridge * identity, cross)
        return weights, target_mean - feature_mean @ weights


def unit_rows(tokens: torch.Tensor) -> torch.Tensor:
    """Tokens scaled to unit norm, the row normalization the spatial cross-fit of SCD uses."""

    return F.normalize(tokens.float(), dim=-1)


def fold_moments(tokens: torch.Tensor, targets: torch.Tensor, folds: torch.Tensor, count: int,
                 chunk: int = 64, device: str | torch.device = "cpu") -> list[Moments]:
    """Per-fold moments of patch rows: ``tokens`` [images, patches, dim], ``targets`` [images, patches, factors]."""

    moments: list[Moments | None] = [None] * count
    for start in range(0, len(tokens), chunk):
        stop = start + chunk
        rows = unit_rows(tokens[start:stop].to(device))
        values = targets[start:stop].to(device).float()
        for fold in folds[start:stop].unique().tolist():
            mask = (folds[start:stop] == fold).to(device)
            part = Moments.of(rows[mask].flatten(0, 1), values[mask].flatten(0, 1))
            moments[fold] = part if moments[fold] is None else moments[fold] + part
    if any(moment is None for moment in moments):
        raise ValueError("Every fold needs at least one image.")
    return moments  # type: ignore[return-value]


def explained_variance(errors: torch.Tensor, totals: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Overall and per-factor explained variance from summed squared errors and total sums of squares."""

    per_factor = 1.0 - errors / totals.clamp_min(1e-12)
    return float(1.0 - errors.sum() / totals.sum().clamp_min(1e-12)), per_factor


def cross_fit_readout(tokens: torch.Tensor, targets: torch.Tensor, folds: torch.Tensor,
                      ridges: list[float], chunk: int = 64, device: str | torch.device = "cpu") -> dict:
    """Held-out explained variance of a patch-level ridge for every ridge strength.

    Returns, per ridge, the overall and per-factor explained variance; for the best ridge also the
    out-of-fold predicted concept vector of every image (patch mean, [images, factors]).
    """

    count = int(folds.max()) + 1
    moments = fold_moments(tokens, targets, folds, count, chunk, device)
    total = sum(moments[1:], moments[0])
    solutions = {ridge: [(total - moments[fold]).solve(ridge) for fold in range(count)] for ridge in ridges}
    factors = targets.shape[-1]
    errors = {ridge: torch.zeros(factors, dtype=torch.float64, device=device) for ridge in ridges}
    oof_vectors = {ridge: torch.zeros(len(tokens), factors) for ridge in ridges}
    # Total sum of squares around each held-out fold's training mean, as the ridge's intercept sees it.
    totals = torch.zeros(factors, dtype=torch.float64, device=device)
    for start in range(0, len(tokens), chunk):
        stop = start + chunk
        rows = unit_rows(tokens[start:stop].to(device)).double()
        values = targets[start:stop].to(device).double()
        for fold in folds[start:stop].unique().tolist():
            mask = folds[start:stop] == fold
            held_rows, held_values = rows[mask.to(device)], values[mask.to(device)]
            rest = total - moments[fold]
            totals += (held_values - rest.target_sum / rest.count).pow(2).sum(dim=(0, 1))
            for ridge in ridges:
                weights, intercept = solutions[ridge][fold]
                predicted = held_rows @ weights + intercept
                errors[ridge] += (predicted - held_values).pow(2).sum(dim=(0, 1))
                oof_vectors[ridge][start:stop][mask] = predicted.mean(dim=1).float().cpu()
    scores = {ridge: explained_variance(errors[ridge].cpu(), totals.cpu()) for ridge in ridges}
    best = max(ridges, key=lambda ridge: scores[ridge][0])
    return {
        "per_ridge": {ridge: scores[ridge][0] for ridge in ridges},
        "best_ridge": best,
        "explained_variance": scores[best][0],
        "per_factor": scores[best][1].float(),
        "oof_concept_vectors": oof_vectors[best],
        "full_fit": tuple(part.cpu() for part in total.solve(best)),
    }


def mean_pool(tokens: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Mean-pooled tokens [images, dim], computed in chunks to keep a float16 store in float16."""

    return torch.cat([tokens[start:start + chunk].float().mean(dim=1) for start in range(0, len(tokens), chunk)])


def pooled_control(tokens: torch.Tensor, targets: torch.Tensor, folds: torch.Tensor, ridge: float,
                   device: str | torch.device = "cpu") -> float:
    """Held-out explained variance when every patch only sees its image's mean-pooled token."""

    pooled = mean_pool(tokens)[:, None].expand(-1, tokens.shape[1], -1)
    return cross_fit_readout(pooled, targets, folds, [ridge], device=device)["explained_variance"]


def shuffled_control(tokens: torch.Tensor, targets: torch.Tensor, folds: torch.Tensor, ridge: float,
                     seed: int = 0, device: str | torch.device = "cpu") -> float:
    """Held-out explained variance when each image is paired with another image's maps."""

    order = torch.randperm(len(targets), generator=torch.Generator().manual_seed(seed + 1))
    return cross_fit_readout(tokens, targets[order], folds, [ridge], device=device)["explained_variance"]


def predict_concepts(tokens: torch.Tensor, solution: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Predicted concept probabilities of every patch, [images, patches, factors]."""

    weights, intercept = solution
    return (unit_rows(tokens).double() @ weights.to(tokens.device) + intercept.to(tokens.device)).float()


def margin_direction(weight: torch.Tensor, scale: torch.Tensor, mean: torch.Tensor) -> tuple[torch.Tensor, float]:
    """The binary probe as a direction on raw pooled features: class-1 minus class-0 logit.

    ``weight`` [2, dim] acts on standardized features ``(x - mean) / scale``; the returned direction
    ``d`` and offset ``c`` give the margin without bias as ``d @ x - c``.
    """

    direction = (weight[1] - weight[0]) / scale
    return direction, float(direction @ mean)


def concept_evidence(tokens: torch.Tensor, concepts: torch.Tensor, direction: torch.Tensor,
                     offset: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Probe evidence and mass of every concept region per image, both [images, factors].

    A patch's evidence is ``d @ z_p - c``: the margin the probe would give an image made of that patch
    alone, measured from the training-average image. The image margin without bias is its patch mean.
    Each patch passes its evidence to the concepts by its predicted probabilities, clipped at zero;
    dividing summed evidence by summed mass gives the evidence per patch of a concept, free of its area.
    """

    weights = concepts.float().clamp_min(0.0)
    evidence = tokens.float() @ direction.float() - offset
    return torch.einsum("ip,ipk->ik", evidence, weights), weights.sum(dim=1)


def evidence_density(evidence: torch.Tensor, mass: torch.Tensor, sign: torch.Tensor,
                     min_mass: float) -> torch.Tensor:
    """Mean evidence per patch of each concept over a set of images, signed towards the true class.

    ``sign`` is +1 where the true class is class 1, -1 otherwise. Concepts whose total mass stays below
    ``min_mass`` patches per image return NaN: too rare in the set to say anything.
    """

    total = mass.sum(dim=0)
    density = (sign[:, None] * evidence).sum(dim=0) / total.clamp_min(1e-12)
    return torch.where(total >= min_mass * len(mass), density, torch.full_like(density, float("nan")))


def typed_evidence(evidence: torch.Tensor, mass: torch.Tensor, sign: torch.Tensor, types: dict[str, torch.Tensor],
                   contrasts: list[tuple[str, str]], resamples: int = 1000, seed: int = 0) -> dict:
    """Evidence per patch of each concept type over a set of images, with percentile bootstrap intervals.

    ``types`` maps a type name to a boolean column mask over the factors. A type's density is its signed
    evidence summed over its factors and images, divided by its summed mass. ``contrasts`` lists
    (a, b) pairs whose difference ``a - b`` gets a paired interval from the same resampled images.
    """

    columns = torch.stack([types[name] for name in types], dim=1).double()  # [factors, types]
    signed = (sign[:, None].double() * evidence.double()) @ columns  # [images, types]
    weights = mass.double() @ columns
    names = list(types)
    position = {name: index for index, name in enumerate(names)}

    def densities(rows: torch.Tensor) -> torch.Tensor:
        return signed[rows].sum(dim=0) / weights[rows].sum(dim=0).clamp_min(1e-12)

    point = densities(torch.arange(len(signed)))
    generator = torch.Generator().manual_seed(seed)
    draws = torch.stack([densities(torch.randint(len(signed), (len(signed),), generator=generator))
                         for _ in range(resamples)]) if len(signed) else torch.zeros(0, len(names))
    low, high = (draws.quantile(0.025, dim=0), draws.quantile(0.975, dim=0)) if len(draws) else (point, point)
    result = {
        name: {"evidence_per_patch": round(float(point[index]), 4),
               "ci95": [round(float(low[index]), 4), round(float(high[index]), 4)],
               "patches_per_image": round(float(weights[:, index].mean()), 2) if len(weights) else 0.0}
        for name, index in position.items()
    }
    for first, second in contrasts:
        a, b = position[first], position[second]
        difference = draws[:, a] - draws[:, b]
        result[f"{first}_minus_{second}"] = {
            "evidence_per_patch": round(float(point[a] - point[b]), 4),
            "ci95": [round(float(difference.quantile(0.025)), 4), round(float(difference.quantile(0.975)), 4)],
        }
    return result
