"""Supervised fine-tuning of the last blocks of a pretrained encoder with a linear head.

A backbone is a frozen stem, an ordered list of blocks and a pooling step. The last
``trainable_blocks`` blocks, the backbone's trainable tail (the final LayerNorm of a ViT) and the head
train; the stem and the earlier blocks run without gradients and in evaluation mode. Two backbones
exist: the I-JEPA ViT (32 transformer blocks, features averaged over patch tokens, as in the frozen
probe) and the torchvision ImageNet ResNet-50 (16 bottleneck blocks, global average pooling) that
serves as the non-JEPA reference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from cospro.models.ijepa import IJEPAEncoder


class IJEPABackbone(nn.Module):
    """An I-JEPA ViT split into patch embedding, transformer blocks and the final norm with token pooling."""

    def __init__(self, encoder: IJEPAEncoder) -> None:
        super().__init__()
        self.spec = encoder.spec
        self.patch_embed = encoder.patch_embed
        self.pos_embed = encoder.pos_embed
        self.blocks = encoder.blocks
        self.tail = encoder.norm
        self.feature_dim = encoder.spec.embed_dim

    def stem(self, images: torch.Tensor) -> torch.Tensor:
        expected = (self.spec.image_size, self.spec.image_size)
        if tuple(images.shape[-2:]) != expected:
            raise ValueError(f"I-JEPA expects {expected} inputs; got {tuple(images.shape[-2:])}.")
        return self.patch_embed(images) + self.pos_embed

    def stem_modules(self) -> list[nn.Module]:
        return [self.patch_embed]

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(x).float().mean(dim=1)


class ResNetBackbone(nn.Module):
    """A torchvision ResNet split into its stem, its bottleneck blocks in order and global pooling."""

    def __init__(self, resnet: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.conv1, self.bn1, self.relu, self.maxpool = resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        self.blocks = nn.ModuleList([block for layer in (resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
                                     for block in layer])
        self.tail = nn.Identity()
        self.feature_dim = feature_dim

    def stem(self, images: torch.Tensor) -> torch.Tensor:
        return self.maxpool(self.relu(self.bn1(self.conv1(images))))

    def stem_modules(self) -> list[nn.Module]:
        return [self.conv1, self.bn1]

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().mean(dim=(2, 3))


class PartialFineTune(nn.Module):
    """A backbone whose last ``trainable_blocks`` blocks and tail train together with a linear head."""

    def __init__(self, backbone: IJEPABackbone | ResNetBackbone, num_classes: int, trainable_blocks: int,
                 gradient_checkpointing: bool = False) -> None:
        super().__init__()
        depth = len(backbone.blocks)
        if not 0 <= trainable_blocks <= depth:
            raise ValueError(f"trainable_blocks must lie in [0, {depth}]; got {trainable_blocks}.")
        self.backbone = backbone
        self.first_trainable = depth - trainable_blocks
        self.gradient_checkpointing = gradient_checkpointing
        self.head = nn.Linear(backbone.feature_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        for module in (*backbone.blocks[self.first_trainable:], backbone.tail):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Pooled last-layer features [batch, feature_dim] in float32."""

        backbone = self.backbone
        with torch.autocast("cuda", dtype=torch.float16, enabled=images.is_cuda):
            with torch.no_grad():
                x = backbone.stem(images)
                for block in backbone.blocks[:self.first_trainable]:
                    x = block(x)
            for block in backbone.blocks[self.first_trainable:]:
                if self.gradient_checkpointing and self.training:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            return backbone.pool(x)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))

    def train(self, mode: bool = True) -> "PartialFineTune":
        super().train(mode)
        # The frozen stem and blocks always run in evaluation mode, so ResNet batch norms keep their statistics.
        for module in (*self.backbone.stem_modules(), *self.backbone.blocks[:self.first_trainable]):
            module.eval()
        return self


def parameter_groups(model: PartialFineTune, lr: float, head_lr: float, weight_decay: float) -> list[dict]:
    """AdamW groups: backbone matrices and kernels with decay, biases and norms without, the head at ``head_lr``."""

    decay, no_decay = [], []
    for parameter in model.backbone.parameters():
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


def held_out_halves(groups: torch.Tensor, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Positions of a group-balanced head-fitting half and of the held-out rest.

    Each group is shuffled with ``seed`` and split in half; the fitting half is then cut to its
    smallest group, so the refitted head sees every group equally often. The held-out positions are
    the other half of every group, untouched by the head.
    """

    generator = torch.Generator().manual_seed(seed)
    fit, holdout = [], []
    for group in groups.unique():
        members = torch.nonzero(groups == group).reshape(-1)
        members = members[torch.randperm(len(members), generator=generator)]
        half = len(members) // 2
        fit.append(members[:half])
        holdout.append(members[half:])
    size = min(len(part) for part in fit)
    return torch.cat([part[:size] for part in fit]).sort().values, torch.cat(holdout).sort().values


def logistic_logits(parameters: dict, features: torch.Tensor) -> torch.Tensor:
    """Logits of a head fitted by ``fit_logistic_probe`` with its ``parameters`` dict."""

    x = (features.double() - parameters["mean"].double()) / parameters["scale"].double()
    return x @ parameters["weight"].double().T + parameters["bias"].double()


def mean_and_spread(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    spread = float(tensor.std()) if len(values) > 1 else 0.0
    return {"mean": round(float(tensor.mean()), 2), "std": round(spread, 2)}
