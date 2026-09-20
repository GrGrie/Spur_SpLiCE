"""What one training attempt holds and what it writes into a checkpoint.

Every save and every resume goes through this object, so the checkpoint payload has one definition:
a new piece of state reaches the checkpoint, the run record and the resumed run by being added here
once. The method contributes its own sampling state, which keeps adaptive samplers resumable
without the trainer knowing they exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cospro.methods import TrainingMethod
from experiments.spurious_eval.training.checkpointing import load_checkpoint, save_checkpoint


@dataclass
class TrainingState:
    """Model, objective, optimizer, scaler, loaders and method of one attempt."""

    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scaler: Any
    method: TrainingMethod
    train_loader: Any
    rank_loader: Any = None

    @property
    def loader_generator(self) -> torch.Generator | None:
        """The seeded generator of the SSL loader, saved so a resumed run draws the same batches."""

        return getattr(self.train_loader, "generator", None)

    def save(self, args, epoch: int, path: str | Path) -> Path:
        """Write the checkpoint and return where it landed, which may be scratch."""

        return save_checkpoint(
            self.model,
            self.optimizer,
            args,
            epoch,
            str(path),
            scaler=self.scaler,
            loader_generator=self.loader_generator,
            training_state=self.method.sampling_state(),
        )

    def resume(self, args, device: torch.device) -> int:
        """The first epoch to train: one past ``--resume``, or one for a fresh run."""

        if not args.resume:
            return 1
        completed = load_checkpoint(
            self.model,
            self.optimizer,
            args.resume,
            device,
            scaler=self.scaler,
            loader_generator=self.loader_generator,
            training_state=self.method.sampling_state(),
            expected_cospro_graph_fingerprint=getattr(args, "cospro_graph_fingerprint", None),
        )
        return completed + 1
