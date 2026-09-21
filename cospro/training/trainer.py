"""The SSL epoch loop.

``Trainer.fit`` owns exactly one thing: how many epochs run, in which order their steps happen and
what each epoch reports. Everything observed about the run happens in callbacks, so a new
diagnostic, log sink, evaluation or retention rule never touches this file.

Two RNG rules hold the loop together. Adaptive sampling draws from its own seeded stream, so a
method that resamples every epoch leaves the training stream untouched. The callbacks run inside
``preserve_rng_state``, so probes and rank diagnostics cannot move the next epoch's batches.
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch

from cospro.training.callbacks import Callback, EpochReport
from cospro.training.state import TrainingState
from cospro.training.optim import adjust_learning_rate
from cospro.training.reproducibility import preserve_rng_state
from cospro.training.ssl_loop import train_one_epoch

#: Offset of the adaptive-sampling RNG stream from the run seed.
SAMPLING_SEED_OFFSET = 2_000_000


class Trainer:
    """Run the configured epochs and report each one to the callbacks."""

    def __init__(self, state: TrainingState, args, callbacks: list[Callback], device: torch.device) -> None:
        self.state = state
        self.args = args
        self.callbacks = list(callbacks)
        self.device = device
        self.total_epochs = int(args.epochs)

    def fit(self) -> None:
        """Resume where a checkpoint left off, train to ``--epochs`` and close the run out."""

        try:
            start_epoch = self.state.resume(self.args, self.device)
            self._dispatch("on_train_start")
            for epoch in range(start_epoch, self.total_epochs + 1):
                self.run_epoch(epoch)
            with preserve_rng_state():
                self._dispatch("on_train_end")
        except Exception as error:
            for callback in self.callbacks:
                callback.on_failure(error)
            raise

    def run_epoch(self, epoch: int) -> EpochReport:
        """One epoch: schedule, resample, train, then report."""

        adjust_learning_rate(self.args, self.state.optimizer, epoch)
        started = time.time()
        sampling_metrics = self.refresh_sampling(epoch)
        train_metrics = train_one_epoch(
            self.state.train_loader,
            self.state.model,
            self.state.criterion,
            self.state.optimizer,
            self.state.scaler,
            epoch,
            self.args,
            self.state.method,
        )
        train_metrics.update(sampling_metrics)
        print("epoch {}, total time {:.2f}".format(epoch, time.time() - started))

        report = EpochReport(
            epoch=epoch,
            train=train_metrics,
            learning_rate=self.state.optimizer.param_groups[0]["lr"],
        )
        with preserve_rng_state():
            self._dispatch("on_epoch_end", report)
        return report

    def refresh_sampling(self, epoch: int) -> dict[str, float]:
        """Let a method rebuild its sampling from its own stream, leaving training RNG intact."""

        if self.state.method.sampling_state() is None:
            return {}
        with preserve_rng_state():
            seed = self.args.seed + SAMPLING_SEED_OFFSET + epoch
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            return self.state.method.refresh_sampling(
                self.state.model, self.device, self.args.temp, epoch,
            )

    def _dispatch(self, hook: str, *arguments) -> None:
        for callback in self.callbacks:
            getattr(callback, hook)(*arguments)
