"""Concept maps: where in an image each concept factor sits, from frozen CLIP without training.

The patch tokens of CLIP's vision transformer are projected into the text space through the last
block with its residual and feed-forward branch removed and query-query attention in place of
query-key attention (ClearCLIP, Lan et al. 2024; MaskCLIP, Zhou et al. 2022). Their cosine with each
factor's text direction, turned into a softmax over the factors, gives a soft segmentation of the
image into the factors. Images are read at 448 pixels through 224-pixel windows and the maps are kept
at 14 x 14, so a training view can be cut out of them by its crop box.

Training warps the maps onto each SimCLR view (``warp_maps``) and regresses the student's feature map
on them location by location (``spatial_cross_fit_loss``): a cat patch has to encode "cat" whether
the image shows a couch or a lawn, which an image-level target cannot ask for when cat and couch
co-occur on most images.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MAP_SIZE = 14
WINDOW = 224
PATCH = 32
#: Accumulation grid: 16-pixel cells, so a 112-pixel window stride is a whole number of cells.
CELL = 16


@torch.no_grad()
def patch_embeddings(model, images: torch.Tensor, attention: str = "qq") -> torch.Tensor:
    """Unit-norm patch embeddings in CLIP's text space, [batch, patches, dim], from 224-pixel images.

    ``attention="qq"`` uses query-query attention in the last block (ClearCLIP); ``"value"`` keeps
    only the value path (MaskCLIP). Both drop the last block's residual and feed-forward branch.
    """

    visual = model.visual
    block = visual.transformer.resblocks[-1]
    captured = {}
    handle = block.ln_1.register_forward_hook(lambda module, inputs, output: captured.__setitem__("x", output))
    try:
        model.encode_image(images)
    finally:
        handle.remove()
    x = captured["x"]
    if x.shape[0] != images.shape[0]:  # sequence-first layout
        x = x.transpose(0, 1)
    batch, tokens, width = x.shape
    weight, bias = block.attn.in_proj_weight, block.attn.in_proj_bias
    value = F.linear(x, weight[2 * width:], bias[2 * width:])
    if attention == "qq":
        heads = block.attn.num_heads
        query = F.linear(x, weight[:width], bias[:width]).view(batch, tokens, heads, -1).transpose(1, 2)
        value = value.view(batch, tokens, heads, -1).transpose(1, 2)
        scores = query @ query.transpose(-1, -2) / query.shape[-1] ** 0.5
        value = (scores.softmax(dim=-1) @ value).transpose(1, 2).reshape(batch, tokens, width)
    elif attention != "value":
        raise ValueError("attention is 'qq' or 'value'.")
    projected = visual.ln_post(block.attn.out_proj(value)) @ visual.proj
    return F.normalize(projected[:, 1:].float(), dim=-1)


def window_offsets(resolution: int, stride: int) -> list[int]:
    """Top-left pixel offsets of the 224-pixel windows along one side of a ``resolution`` image."""

    if (resolution - WINDOW) % stride or stride % CELL:
        raise ValueError("The window stride must tile the image and be a multiple of 16 pixels.")
    return list(range(0, resolution - WINDOW + 1, stride))


def assemble_windows(window_scores: torch.Tensor, resolution: int, stride: int) -> torch.Tensor:
    """Average overlapping window scores into one map per image, [images, factors, 14, 14].

    ``window_scores`` is [images, windows, factors, 7, 7] with the windows in row-major order of
    ``window_offsets``. Scores are spread onto a 16-pixel grid, averaged where windows overlap and
    pooled to 14 x 14.
    """

    offsets = window_offsets(resolution, stride)
    images, windows, factors, side, _ = window_scores.shape
    if windows != len(offsets) ** 2 or side != WINDOW // PATCH:
        raise ValueError("Window scores do not match the window layout.")
    cells = resolution // CELL
    per_patch = PATCH // CELL
    total = window_scores.new_zeros(images, factors, cells, cells)
    count = window_scores.new_zeros(1, 1, cells, cells)
    for position, (top, left) in enumerate((top, left) for top in offsets for left in offsets):
        block = window_scores[:, position].repeat_interleave(per_patch, dim=-1).repeat_interleave(per_patch, dim=-2)
        rows = slice(top // CELL, top // CELL + side * per_patch)
        columns = slice(left // CELL, left // CELL + side * per_patch)
        total[:, :, rows, columns] += block
        count[:, :, rows, columns] += 1
    return F.adaptive_avg_pool2d(total / count, MAP_SIZE)


def warp_maps(maps: torch.Tensor, boxes: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Cut every view's crop out of its image's map and resize it to the feature-map ``size``.

    ``maps`` is [views, factors, S, S] over the whole image; ``boxes`` is [views, 5] with the crop's
    top, left, height and width as fractions of the image and a flip flag, as recorded by the SSL
    transform.
    """

    from torchvision.ops import roi_align

    side = maps.shape[-1]
    boxes = boxes.to(maps.device, maps.dtype)
    regions = torch.stack([
        torch.arange(len(maps), device=maps.device, dtype=maps.dtype),
        boxes[:, 1] * side, boxes[:, 0] * side,
        (boxes[:, 1] + boxes[:, 3]) * side, (boxes[:, 0] + boxes[:, 2]) * side,
    ], dim=1)
    warped = roi_align(maps, regions, output_size=size, spatial_scale=1.0, sampling_ratio=2, aligned=True)
    flipped = boxes[:, 4] > 0.5
    warped[flipped] = warped[flipped].flip(-1)
    return warped


def spatial_cross_fit_loss(features: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor,
                           ridge: float) -> tuple[torch.Tensor, float]:
    """Held-out squared error of a per-location ridge fitted on one half of the images, both ways.

    ``features`` is [views, channels, h, w], ``targets`` [views, factors, h, w] and ``weights`` one
    value per view (0 keeps a view out). The views of the first half of the images fit a ridge from
    each location's feature to its concept values, which predicts every location of the other half,
    and the reverse; the two views of an image stay in one half. Rows are unit-norm features, the
    ridge is absolute and the primal solve is channels by channels. Returns the loss and the held-out
    explained variance of standardized targets.
    """

    with torch.autocast(device_type=features.device.type, enabled=False):
        return _spatial_cross_fit_loss(features.float(), targets.float(), weights.float(), ridge)


def _spatial_cross_fit_loss(features: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor,
                            ridge: float) -> tuple[torch.Tensor, float]:
    views, channels, height, width = features.shape
    images = views // 2
    rows = F.normalize(features.float().permute(0, 2, 3, 1).reshape(views, height * width, channels), dim=-1)
    values = targets.float().permute(0, 2, 3, 1).reshape(views, height * width, -1)
    half = torch.zeros(images, dtype=torch.bool, device=features.device)
    half[: images // 2] = True
    half = torch.cat([half, half])
    identity = torch.eye(channels, device=features.device)
    loss = features.sum() * 0.0
    errors = []
    for fit in (half, ~half):
        fit_weights = weights[fit][:, None].expand(-1, height * width).reshape(-1)
        fitted_rows, fitted_values = rows[fit].reshape(-1, channels), values[fit].reshape(-1, values.shape[-1])
        scale = fit_weights.sum().clamp_min(1e-12)
        row_mean = (fit_weights[:, None] * fitted_rows).sum(dim=0) / scale
        value_mean = (fit_weights[:, None] * fitted_values).sum(dim=0) / scale
        centred = fitted_rows - row_mean
        gram = (fit_weights[:, None] * centred).T @ centred
        solution = torch.linalg.solve(gram + ridge * identity,
                                      (fit_weights[:, None] * centred).T @ (fitted_values - value_mean))
        predicted = (rows[~fit] - row_mean) @ solution + value_mean
        per_view = (predicted - values[~fit]).pow(2).mean(dim=(1, 2))
        held_out = weights[~fit]
        loss = loss + (held_out * per_view).sum() / held_out.sum().clamp_min(1e-12) / 2
        errors.append(float(((held_out * per_view).sum() / held_out.sum().clamp_min(1e-12)).detach()))
    return loss, 1.0 - sum(errors) / len(errors)
