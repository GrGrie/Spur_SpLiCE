"""Concept attribution at inference: what the frozen encoder and its linear probe lose when one concept is hidden.

For a trained encoder, every audited concept unit ``k`` is occluded in the evaluation images that show it, with
the same hard regions and inpainting as the training-time audit (``cospro.curves.core``). The probe, fitted on
the clean features of the probe training split, then classifies the occluded images. The change of each
group's accuracy says how much the encoder and probe use the concept: a concept whose occlusion raises the
worst-group accuracy acts as a shortcut, one whose occlusion lowers it is needed. docs/PROJECT_CSA.md,
Section 6, defines the measurement.

The control occludes, in the same images, the region unit ``k`` holds in another evaluation image that also
shows it: the same kind of shape and size at a place unrelated to the concept. The net effect is the change of
the concept occlusion minus the change of the control, so it removes what any occlusion of that size does.

The functions here read no label; labels enter only the group metrics of the caller.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cospro.curves.core import occlude, pixel_masks, unit_presence

#: Seed of the donor images of the control.
CONTROL_SEED = 0


def control_donors(present: torch.Tensor, seed: int = CONTROL_SEED) -> torch.Tensor:
    """Per unit and image the row of another image that shows the unit, or -1: a long [n, units].

    The images that show a unit are put in a random order and each one borrows the region of the next, so
    every image borrows from a different image and the donors keep the region sizes of that unit.
    """

    generator = torch.Generator().manual_seed(seed)
    donors = torch.full(present.shape, -1, dtype=torch.long)
    for unit in range(present.shape[1]):
        shown = present[:, unit].nonzero().view(-1)
        if len(shown) < 2:
            continue
        order = shown[torch.randperm(len(shown), generator=generator)]
        donors[order, unit] = order.roll(-1)
    return donors


@torch.no_grad()
def occlusion_features(encoder, loader, regions: torch.Tensor, units: list[int], device,
                       seed: int = CONTROL_SEED) -> dict:
    """Encoder features of the evaluation images, clean and with each unit occluded, and the control.

    ``loader`` yields ``(images, rows)`` in a fixed order; ``rows`` index ``regions`` [n, units, h, w]. The
    occluded features of unit ``k`` equal the clean ones on the images without the unit. Returns ``clean``
    [n, d], ``occluded`` and ``control`` {unit: [n, d]} and ``present`` [n, units].
    """

    present = unit_presence(regions)
    donors = control_donors(present, seed)
    was_training = encoder.training
    encoder.eval()
    clean_parts, occluded_parts, control_parts = [], {unit: [] for unit in units}, {unit: [] for unit in units}
    try:
        for images, rows in loader:
            images = images.to(device)
            rows = torch.as_tensor(rows, dtype=torch.long)
            clean = encoder(images).float()
            clean_parts.append(clean.cpu())
            for unit in units:
                for parts, source in ((occluded_parts, rows), (control_parts, donors[rows, unit])):
                    features = clean.clone()
                    chosen = present[rows, unit] & (source >= 0)
                    if bool(chosen.any()):
                        masks = pixel_masks(regions[source[chosen], unit:unit + 1].to(device), tuple(images.shape[-2:]))
                        picked = chosen.to(device)
                        features[picked] = encoder(occlude(images[picked].float(), masks).to(images.dtype)).float()
                    parts[unit].append(features.cpu())
    finally:
        encoder.train(was_training)
    return {
        "clean": torch.cat(clean_parts),
        "occluded": {unit: torch.cat(parts) for unit, parts in occluded_parts.items()},
        "control": {unit: torch.cat(parts) for unit, parts in control_parts.items()},
        "present": present,
    }


def cosine_shift(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean of one minus the cosine of two feature sets over the rows in ``mask``; 0 without rows."""

    if not bool(mask.any()):
        return 0.0
    return float((1.0 - (F.normalize(first[mask], dim=1) * F.normalize(second[mask], dim=1)).sum(dim=1)).mean())
