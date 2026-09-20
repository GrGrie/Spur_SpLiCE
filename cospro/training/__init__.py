"""The SSL training loop and everything that observes it.

``Trainer`` runs the epochs, ``TrainingState`` is what a checkpoint holds, ``StoragePolicy`` decides
where binaries live and how long they stay. The callbacks carry the diagnostics, the log sinks,
the linear probe and the checkpoint writes. ``spur_splice.py`` parses the command line, builds these
objects and calls ``fit``.
"""

from __future__ import annotations

from cospro.training.callbacks import (
    Callback,
    CheckpointPolicy,
    EpochReport,
    PeriodicProbe,
    RankMetrics,
    RunRecordLogger,
    WandbLogger,
)
from cospro.training.state import TrainingState
from cospro.training.storage import StoragePolicy, artifact_identity
from cospro.training.trainer import Trainer

__all__ = [
    "Callback", "CheckpointPolicy", "EpochReport", "PeriodicProbe", "RankMetrics", "RunRecordLogger",
    "StoragePolicy", "Trainer", "TrainingState", "WandbLogger", "artifact_identity",
]
