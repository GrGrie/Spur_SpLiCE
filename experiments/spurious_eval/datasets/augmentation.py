from __future__ import annotations

from dataclasses import dataclass

from torchvision import transforms


@dataclass(frozen=True)
class StrongAugmentationConfig:
    splice_strong_crop: float | None = None
    splice_strong_color_jitter: tuple[float, float, float, float] | None = None
    splice_strong_color_jitter_p: float | None = None
    splice_strong_grayscale_p: float | None = None
    splice_strong_blur_p: float | None = None
    splice_strong_blur_kernel_size: int | None = None
    splice_strong_blur_sigma: tuple[float, float] | None = None


def strong_color_jitter_enabled(config: StrongAugmentationConfig) -> bool:
    return config.splice_strong_color_jitter is not None or config.splice_strong_color_jitter_p is not None


def strong_blur_enabled(config: StrongAugmentationConfig) -> bool:
    return (
        config.splice_strong_blur_p is not None
        or config.splice_strong_blur_kernel_size is not None
        or config.splice_strong_blur_sigma is not None
    )


def build_ssl_transform(
    image_size: int,
    crop_min: float,
    color_jitter: tuple[float, float, float, float],
    color_jitter_p: float,
    grayscale_p: float,
    normalize: transforms.Normalize,
    blur_p: float | None = None,
    blur_kernel_size: int = 23,
    blur_sigma: tuple[float, float] = (0.1, 2.0),
) -> transforms.Compose:
    transform_steps = [
        transforms.RandomResizedCrop(size=image_size, scale=(crop_min, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(*color_jitter)], p=color_jitter_p),
        transforms.RandomGrayscale(p=grayscale_p),
    ]
    if blur_p is not None:
        transform_steps.append(
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)], p=blur_p)
        )
    transform_steps.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(transform_steps)


def build_standard_and_strong_ssl_transforms(
    image_size: int,
    ssl_crop_min: float,
    normalize: transforms.Normalize,
    strong_config: StrongAugmentationConfig,
) -> tuple[transforms.Compose, transforms.Compose]:
    standard_transform = build_ssl_transform(
        image_size=image_size,
        crop_min=ssl_crop_min,
        color_jitter=(0.4, 0.4, 0.4, 0.1),
        color_jitter_p=0.8,
        grayscale_p=0.2,
        normalize=normalize,
    )

    use_strong_color_jitter = strong_color_jitter_enabled(strong_config)
    strong_color_jitter = strong_config.splice_strong_color_jitter
    if strong_color_jitter is None:
        strong_color_jitter = (0.8, 0.8, 0.8, 0.2) if use_strong_color_jitter else (0.4, 0.4, 0.4, 0.1)
    strong_color_jitter_p = strong_config.splice_strong_color_jitter_p
    if strong_color_jitter_p is None:
        strong_color_jitter_p = 0.9 if use_strong_color_jitter else 0.8
    strong_grayscale_p = strong_config.splice_strong_grayscale_p
    if strong_grayscale_p is None:
        strong_grayscale_p = 0.2
    strong_blur_p = None
    if strong_blur_enabled(strong_config):
        strong_blur_p = 0.5 if strong_config.splice_strong_blur_p is None else strong_config.splice_strong_blur_p

    strong_transform = build_ssl_transform(
        image_size=image_size,
        crop_min=strong_config.splice_strong_crop if strong_config.splice_strong_crop is not None else ssl_crop_min,
        color_jitter=strong_color_jitter,
        color_jitter_p=strong_color_jitter_p,
        grayscale_p=strong_grayscale_p,
        normalize=normalize,
        blur_p=strong_blur_p,
        blur_kernel_size=strong_config.splice_strong_blur_kernel_size or 23,
        blur_sigma=strong_config.splice_strong_blur_sigma or (0.1, 2.0),
    )
    return standard_transform, strong_transform
