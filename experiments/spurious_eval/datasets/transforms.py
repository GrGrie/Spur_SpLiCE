import torch


def resolve_augmentation_threshold(
    scores: torch.Tensor,
    threshold: float | None,
    quantile: float,
) -> float:
    if threshold is not None:
        return float(threshold)
    if scores.numel() == 0:
        raise ValueError("Cannot auto-calibrate a SpLiCE threshold from an empty score tensor.")
    resolved = torch.quantile(scores.float(), torch.tensor(quantile, dtype=torch.float32)).item()
    print(f"[INFO] Auto-calibrated SpLiCE augmentation threshold at q={quantile:g}: {resolved:.8f}")
    return resolved


class TwoCropTransform:
    """Create two independently augmented views of the same image."""

    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, image):
        return [self.transform(image), self.transform(image)]


class ConceptAwareTwoCropTransform:
    """Create SimCLR views with stronger augmentation for high SpLiCE-score images."""

    def __init__(self, standard_transform, strong_transform, threshold: float) -> None:
        self.standard_transform = standard_transform
        self.strong_transform = strong_transform
        self.threshold = threshold

    def __call__(self, image, score: float):
        first_view = self.standard_transform(image)
        second_transform = self.strong_transform if score >= self.threshold else self.standard_transform
        return [first_view, second_transform(image)]


class ConceptAwareSSLSubset(torch.utils.data.Dataset):
    """Attach cached SpLiCE controls to a WILDS-style SSL subset.

    ``scores`` decide whether the second view receives the targeted transform.
    When ``concept_weights`` are provided, the full selected concept vector is
    returned to the SSL loop instead of the lossy scalar score.
    """

    def __init__(self, subset, scores: torch.Tensor, transform, concept_weights: torch.Tensor | None = None) -> None:
        if len(subset) != len(scores):
            raise ValueError(f"Expected one SpLiCE score per sample, got {len(scores)} for {len(subset)} samples.")
        if concept_weights is not None and len(subset) != len(concept_weights):
            raise ValueError(
                f"Expected one SpLiCE concept vector per sample, got {len(concept_weights)} for {len(subset)} samples."
            )
        self.subset = subset
        self.scores = scores.float()
        self.concept_weights = concept_weights.float() if concept_weights is not None else None
        self.transform = transform
        self.og_group_counts = getattr(subset, "og_group_counts", None)

    def __len__(self) -> int:
        return len(self.subset)

    @property
    def collate(self):
        return getattr(self.subset, "collate", None)

    @property
    def metadata_array(self) -> torch.Tensor:
        return self.subset.metadata_array

    def __getitem__(self, idx: int):
        image, label, metadata = self.subset[idx]
        score = self.scores[idx]
        views = self.transform(image, float(score.item()))
        regularization_control = self.concept_weights[idx] if self.concept_weights is not None else score
        return views, label, metadata, regularization_control

    def eval(self, y_pred: torch.Tensor, y_true: torch.Tensor, metadata: torch.Tensor):
        return self.subset.eval(y_pred, y_true, metadata)
