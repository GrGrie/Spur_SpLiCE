from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.spurious_eval.resnet import build_resnet_encoder


class TwoCropTransform:
    """Create two independently augmented views of the same image."""

    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, image):
        return [self.transform(image), self.transform(image)]


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return F.normalize(self.head(features), dim=1)


class SimCLRLoss(nn.Module):
    """SupConLoss in unsupervised SimCLR mode, matching SpurSSL's implementation."""

    def __init__(self, temperature: float = 0.5, contrast_mode: str = "all", base_temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        start = time.perf_counter()
        if len(features.shape) < 3:
            raise ValueError("features must be shaped [batch_size, n_views, ...]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        device = features.device
        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32, device=device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        zero = torch.tensor(0.0, device=device)
        return loss, zero, zero, zero, time.perf_counter() - start
