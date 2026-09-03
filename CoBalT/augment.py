from __future__ import annotations

import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class RecordedViewTransform:
    """SlotCon-style augmentation that also records crop geometry.

    The normalized source-image box and flip flag let the model compare only
    spatial locations visible in both augmented views.
    """

    def __init__(
        self, image_size: int = 224, crop_min: float = 0.2, clip_normalize: bool = False
    ) -> None:
        self.image_size = image_size
        self.crop_min = crop_min
        self.color = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        mean, std = (CLIP_MEAN, CLIP_STD) if clip_normalize else (IMAGENET_MEAN, IMAGENET_STD)
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width, height = image.size
        top, left, crop_h, crop_w = transforms.RandomResizedCrop.get_params(
            image, scale=(self.crop_min, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
        )
        view = TF.resized_crop(
            image,
            top,
            left,
            crop_h,
            crop_w,
            [self.image_size, self.image_size],
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        flipped = random.random() < 0.5
        if flipped:
            view = TF.hflip(view)
        if random.random() < 0.8:
            view = self.color(view)
        if random.random() < 0.2:
            view = TF.rgb_to_grayscale(view, num_output_channels=3)
        tensor = self.normalize(TF.to_tensor(view))
        box = torch.tensor(
            [left / width, top / height, (left + crop_w) / width, (top + crop_h) / height],
            dtype=torch.float32,
        )
        return tensor, box, torch.tensor(flipped, dtype=torch.bool)


class TwoRecordedViews:
    def __init__(self, transform: RecordedViewTransform) -> None:
        self.transform = transform

    def __call__(self, image: Image.Image):
        first = self.transform(image)
        second = self.transform(image)
        return (
            torch.stack((first[0], second[0])),
            torch.stack((first[1], second[1])),
            torch.stack((first[2], second[2])),
        )


def evaluation_transform(image_size: int = 224, clip_normalize: bool = False):
    mean, std = (CLIP_MEAN, CLIP_STD) if clip_normalize else (IMAGENET_MEAN, IMAGENET_STD)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def classifier_train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
