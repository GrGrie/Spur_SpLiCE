from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.spurious_eval.models.resnet import build_resnet_encoder


class SimCLRModel(nn.Module):
    """SpurSSL-compatible encoder plus normalized projection head."""

    def __init__(self, name: str = "resnet18_large", head: str = "mlp", feat_dim: int = 128) -> None:
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

    def enable_localized_scrubber(self, max_concepts: int) -> None:
        self.encoder.enable_localized_scrubber(max_concepts, projection_dim=self._projection_dim())

    def _projection_dim(self) -> int:
        if isinstance(self.head, nn.Identity):
            return self.feature_dim
        if isinstance(self.head, nn.Linear):
            return self.head.out_features
        return self.head[-1].out_features

    def _encoder_scrubber(self):
        encoder = self.encoder.module if isinstance(self.encoder, nn.DataParallel) else self.encoder
        return getattr(encoder, "localized_scrubber", None)

    def project(self, features: torch.Tensor) -> torch.Tensor:
        projections = self.head(features)
        scrubber = self._encoder_scrubber()
        if scrubber is not None:
            projections = scrubber.apply("projection", projections)
        return F.normalize(projections, dim=1)

    def forward_with_intermediates(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features, intermediates = self.encoder(x, return_intermediates=True)
        projections = self.head(features)
        scrubber = self._encoder_scrubber()
        if scrubber is not None:
            projections = scrubber.apply("projection", projections)
        intermediates["projection"] = projections
        return features, intermediates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.project(features)
