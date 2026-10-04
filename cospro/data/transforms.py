import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


class TwoCropTransform:
    """Create two independently augmented views of the same image."""

    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, image):
        return [self.transform(image), self.transform(image)]


class _RecordedResizedCrop:
    """A RandomResizedCrop that remembers its box as fractions of the input image."""

    def __init__(self, crop: transforms.RandomResizedCrop) -> None:
        self.crop = crop
        self.box = (0.0, 0.0, 1.0, 1.0)

    def __call__(self, image):
        width, height = TF.get_image_size(image)
        top, left, crop_height, crop_width = self.crop.get_params(image, self.crop.scale, self.crop.ratio)
        self.box = (top / height, left / width, crop_height / height, crop_width / width)
        return TF.resized_crop(image, top, left, crop_height, crop_width, self.crop.size,
                               self.crop.interpolation, antialias=self.crop.antialias)


class _RecordedFlip:
    """A RandomHorizontalFlip that remembers whether it flipped."""

    def __init__(self, flip: transforms.RandomHorizontalFlip) -> None:
        self.p = flip.p
        self.flipped = False

    def __call__(self, image):
        self.flipped = bool(torch.rand(1) < self.p)
        return TF.hflip(image) if self.flipped else image


class RecordedTwoCropTransform:
    """Two views of an image plus, per view, its crop box and flip: [2, 5] of top, left, height, width, flip.

    The box locates each view in the whole image, so maps computed on the whole image can be cut out for
    the view. The views follow the wrapped pipeline exactly; only its crop and flip steps record.
    """

    def __init__(self, transform: transforms.Compose) -> None:
        steps = []
        self.crop = self.flip = None
        for step in transform.transforms:
            if isinstance(step, transforms.RandomResizedCrop) and self.crop is None:
                self.crop = _RecordedResizedCrop(step)
                steps.append(self.crop)
            elif isinstance(step, transforms.RandomHorizontalFlip) and self.flip is None:
                self.flip = _RecordedFlip(step)
                steps.append(self.flip)
            else:
                steps.append(step)
        if self.crop is None:
            raise ValueError("Recording view boxes needs a RandomResizedCrop as the geometric step.")
        self.transform = transforms.Compose(steps)

    def _view(self, image):
        view = self.transform(image)
        flipped = float(self.flip.flipped) if self.flip is not None else 0.0
        return view, torch.tensor([*self.crop.box, flipped], dtype=torch.float32)

    def __call__(self, image):
        first, first_box = self._view(image)
        second, second_box = self._view(image)
        return [first, second], torch.stack([first_box, second_box])
