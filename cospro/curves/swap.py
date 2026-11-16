"""The early concept swap: exchange the flagged concept regions between training images.

Every image keeps the part outside its flagged regions and receives, inside them, the flagged-type
content of a donor image; the label stays with the kept part. Flagged concepts then carry no
information about the label (Section 3.6 of docs/PROJECT_PLAN.md, Theorem "Early concept swap" in
paper/ConceptLearningCurves.tex).
"""

from __future__ import annotations

import torch

from cospro.pipeline.concept_maps import composite_images, sharpen, warp_maps

#: An image is swapped only when its kept part covers this range of the view.
MIN_KEEP, MAX_KEEP = 0.05, 0.95
#: A view serves as a donor when its flagged regions cover at least this share.
MIN_DONOR = 0.25


def flagged_masks(regions: torch.Tensor, flagged: list[int], boxes: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Pixel masks [b, 1, H, W] of the flagged regions on each training view.

    ``regions`` is the boolean [b, units, h, w] of the batch's images over the whole image and ``boxes``
    [b, 5] the crop box and flip of each view.
    """

    index = torch.as_tensor(flagged, dtype=torch.long, device=regions.device)
    mass = regions.index_select(1, index).any(dim=1, keepdim=True).float()
    return sharpen(warp_maps(mass, boxes, size))


def swap_flagged(views: torch.Tensor, flagged: torch.Tensor, probability: float,
                 generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Views with the flagged regions of a random share replaced by donor content, and who was swapped.

    ``flagged`` is [b, 1, H, W] in [0, 1]. An eligible image is swapped with ``probability``. Its donor is
    another view of the batch with enough flagged content, whose own kept part is painted over before it
    fills the image's flagged regions. ``generator`` lives on the CPU, so the draws are reproducible.
    """

    count = len(views)
    swapped = torch.zeros(count, dtype=torch.bool, device=views.device)
    if probability <= 0 or count < 2:
        return views, swapped
    keep = 1.0 - flagged
    kept_share = keep.mean(dim=(1, 2, 3))
    eligible = (kept_share >= MIN_KEEP) & (kept_share <= MAX_KEEP)
    pool = ((1.0 - kept_share) >= MIN_DONOR).nonzero().view(-1)
    drawn = torch.rand(count, generator=generator).to(views.device) < probability
    chosen = (eligible & drawn).nonzero().view(-1)
    if len(chosen) == 0 or len(pool) < 2:
        return views, swapped
    picks = torch.randint(len(pool), (len(chosen),), generator=generator).to(views.device)
    donors = pool[picks]
    # Pool members are distinct, so the next one differs from an image drawn as its own donor.
    donors = torch.where(donors == chosen, pool[(picks + 1) % len(pool)], donors)
    composites = composite_images(views[chosen].float(), views[donors].float(), keep[chosen], keep[donors])
    result = views.clone()
    result[chosen] = composites.to(views.dtype)
    swapped[chosen] = True
    return result, swapped
