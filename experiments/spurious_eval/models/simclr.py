from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.spurious_eval.models.resnet import build_resnet_encoder


class SimCLRModel(nn.Module):
    """SpurSSL-compatible encoder plus normalized projection head."""

    def __init__(
        self,
        name: str = "resnet18_large",
        head: str = "mlp",
        feat_dim: int = 128,
        clip_distillation_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder, dim_in = build_resnet_encoder(name)
        self.feature_dim = dim_in
        if head == "linear":
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(dim_in, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, feat_dim),
            )
        elif head == "identity":
            self.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported SimCLR projection head '{head}'. Use linear, mlp, or identity.")
        self.clip_distillation_head = None
        if clip_distillation_dim is not None:
            self.clip_distillation_head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, clip_distillation_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return F.normalize(self.head(features), dim=1)
