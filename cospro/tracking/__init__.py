"""Experiment tracking: the stable metric names runs report to Weights & Biases."""

from cospro.tracking.metrics import (
    WGA_KEY,
    canonical_probe_metrics,
    canonical_train_metrics,
    define_wandb_metrics,
    epoch_payload,
)

__all__ = [
    "WGA_KEY", "canonical_probe_metrics", "canonical_train_metrics", "define_wandb_metrics", "epoch_payload",
]
