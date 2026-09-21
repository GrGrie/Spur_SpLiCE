from __future__ import annotations


from torchvision import transforms










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
