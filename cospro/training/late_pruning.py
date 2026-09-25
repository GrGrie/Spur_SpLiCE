"""LateTVG: the second SimCLR view passes through a magnitude-pruned copy of the late encoder layers.

Hamidieh et al. (ICLR 2024, "Views Can Be Deceiving", Algorithm 1) build a positive pair from the
full encoder on view 1 and a transformed encoder on view 2. The transformation zeroes the fraction
``prune_rate`` of smallest-magnitude weights, pooled across the last ``layers`` convolutions. The
mask is recomputed from the current weights at every step and the pruned encoder shares its weights
with the full one, so the gradient of view 2 reaches every weight the mask keeps. Batch-norm
statistics of the pruned pass stay out of the full encoder.

Evaluation measures the full encoder, like every other arm of this project.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.func import functional_call


class LatePruning:
    """Magnitude pruning of the last ``layers`` convolutions, applied functionally per forward."""

    def __init__(self, prune_rate: float, layers: int) -> None:
        if not 0 < prune_rate < 1:
            raise ValueError("LateTVG prune rate must lie in (0, 1).")
        if layers < 1:
            raise ValueError("LateTVG must prune at least one convolution.")
        self.prune_rate = float(prune_rate)
        self.layers = int(layers)
        self.last_kept_fraction = 1.0

    @classmethod
    def from_args(cls, args) -> "LatePruning | None":
        rate = float(getattr(args, "latetvg_prune_rate", 0.0))
        return cls(rate, int(args.latetvg_layers)) if rate > 0 else None

    def target_names(self, encoder: nn.Module) -> list[str]:
        """Weight names of the last ``layers`` convolutions in module registration order."""

        convolutions = [name for name, module in encoder.named_modules() if isinstance(module, nn.Conv2d)]
        if len(convolutions) < self.layers:
            raise ValueError(f"LateTVG asks for {self.layers} convolutions; the encoder has {len(convolutions)}.")
        return [f"{name}.weight" for name in convolutions[-self.layers:]]

    def masks(self, encoder: nn.Module) -> dict[str, torch.Tensor]:
        """Keep-masks from one magnitude threshold pooled over every targeted weight."""

        parameters = dict(encoder.named_parameters())
        names = self.target_names(encoder)
        with torch.no_grad():
            magnitudes = torch.cat([parameters[name].detach().abs().flatten().float() for name in names])
            pruned = int(self.prune_rate * magnitudes.numel())
            threshold = torch.kthvalue(magnitudes, pruned).values if pruned else magnitudes.min() - 1
            masks = {name: parameters[name].detach().abs() > threshold for name in names}
            kept = sum(int(mask.sum()) for mask in masks.values())
        self.last_kept_fraction = kept / magnitudes.numel()
        return masks

    def __call__(self, encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """Encode ``images`` with the pruned encoder; the full encoder's weights stay unchanged."""

        parameters = dict(encoder.named_parameters())
        pruned = {name: parameters[name] * mask for name, mask in self.masks(encoder).items()}
        # The pruned encoder keeps its own batch-norm statistics, so the evaluated full encoder never
        # averages pruned activations into its running means and variances.
        buffers = {name: buffer.clone() for name, buffer in encoder.named_buffers()}
        return functional_call(encoder, {**pruned, **buffers}, (images,), strict=False)
