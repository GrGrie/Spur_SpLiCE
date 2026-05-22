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
        transform = self.strong_transform if score >= self.threshold else self.standard_transform
        return [transform(image), transform(image)]
