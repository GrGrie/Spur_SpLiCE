"""Supervised fine-tuning of the last blocks of an I-JEPA encoder with a linear head.

The first ``depth - trainable_blocks`` blocks, the patch embedding and the position table stay frozen
and run without gradients. The last ``trainable_blocks`` blocks, the final LayerNorm and the head train.
Image features are the average of the last-layer patch tokens, as for the frozen probe, so a head
fitted after fine-tuning reads the same kind of vector.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from cospro.models.ijepa import IJEPAEncoder


class PartialFineTune(nn.Module):
    """An I-JEPA encoder whose last ``trainable_blocks`` blocks and final norm train with a linear head."""

    def __init__(self, encoder: IJEPAEncoder, num_classes: int, trainable_blocks: int,
                 gradient_checkpointing: bool = False) -> None:
        super().__init__()
        depth = len(encoder.blocks)
        if not 0 <= trainable_blocks <= depth:
            raise ValueError(f"trainable_blocks must lie in [0, {depth}]; got {trainable_blocks}.")
        self.encoder = encoder
        self.first_trainable = depth - trainable_blocks
        self.gradient_checkpointing = gradient_checkpointing
        self.head = nn.Linear(encoder.spec.embed_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        for module in (*encoder.blocks[self.first_trainable:], encoder.norm):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Average-pooled last-layer patch tokens [batch, embed_dim] in float32."""

        encoder = self.encoder
        expected = (encoder.spec.image_size, encoder.spec.image_size)
        if tuple(images.shape[-2:]) != expected:
            raise ValueError(f"I-JEPA expects {expected} inputs; got {tuple(images.shape[-2:])}.")
        with torch.autocast("cuda", dtype=torch.float16, enabled=images.is_cuda):
            with torch.no_grad():
                x = encoder.patch_embed(images) + encoder.pos_embed
                for block in encoder.blocks[:self.first_trainable]:
                    x = block(x)
            for block in encoder.blocks[self.first_trainable:]:
                if self.gradient_checkpointing and self.training:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            x = encoder.norm(x)
        return x.float().mean(dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))

    def train(self, mode: bool = True) -> "PartialFineTune":
        super().train(mode)
        # The frozen prefix always runs in evaluation mode.
        self.encoder.patch_embed.eval()
        for block in self.encoder.blocks[:self.first_trainable]:
            block.eval()
        return self


def parameter_groups(model: PartialFineTune, lr: float, head_lr: float, weight_decay: float) -> list[dict]:
    """AdamW groups: encoder matrices with decay, encoder biases and norms without, the head at ``head_lr``."""

    decay, no_decay = [], []
    for parameter in model.encoder.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": list(model.head.parameters()), "lr": head_lr, "weight_decay": 0.0},
    ]
    return [group for group in groups if group["params"]]


def warmup_cosine(step: int, total_steps: int, warmup_steps: int) -> float:
    """Learning-rate factor: linear warmup to 1, then cosine decay to 0 at ``total_steps``."""

    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def trainable_state(model: PartialFineTune) -> dict[str, torch.Tensor]:
    """The parameters fine-tuning changed, in float16, keyed as in the full model."""

    return {name: parameter.detach().half().cpu() for name, parameter in model.named_parameters()
            if parameter.requires_grad}
