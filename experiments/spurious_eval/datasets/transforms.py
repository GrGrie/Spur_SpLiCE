class TwoCropTransform:
    """Create two independently augmented views of the same image."""

    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, image):
        return [self.transform(image), self.transform(image)]
