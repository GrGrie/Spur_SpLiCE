"""Everything observed about a training run, outside the epoch loop.

The trainer runs epochs and reports each one; a callback decides what that means for a diagnostic,
a log sink, an evaluation or a file on disk. Adding an observation means adding a callback while the
loop stays the same length. A callback holds what it needs from construction, so the hooks carry
only the epoch report or the failure.

Order matters and follows the order the run historically produced its side effects:
``RankMetrics`` fills the diagnostics of the report, ``RunRecordLogger`` and ``WandbLogger`` write
the resulting metric event, ``PeriodicProbe`` evaluates the encoder and ``CheckpointPolicy`` writes
the recovery checkpoint. The trainer restores the RNG state around the whole dispatch, so no
observation can move the training stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from cospro.tracking import canonical_train_metrics, epoch_payload
from cospro.training.state import TrainingState
from cospro.training.storage import StoragePolicy
from experiments.spurious_eval.metrics import entropy_effective_rank
from experiments.spurious_eval.training.reproducibility import preserve_rng_state
from experiments.spurious_eval.training.ssl_loop import extract_normalized_train_features

ProbeFunction = Callable[[str, int], Mapping[str, Any]]


@dataclass
class EpochReport:
    """What one finished epoch produced, plus what the callbacks add to it."""

    epoch: int
    train: dict[str, Any]
    learning_rate: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """The metric event under its historical key names."""

        return epoch_payload(self.train, learning_rate=self.learning_rate, diagnostics=self.diagnostics)


class Callback:
    """The hooks a trainer emits. Every hook is optional."""

    def on_train_start(self) -> None:
        return None

    def on_epoch_end(self, report: EpochReport) -> None:
        return None

    def on_train_end(self) -> None:
        return None

    def on_failure(self, error: BaseException) -> None:
        return None


class RankMetrics(Callback):
    """Representation-rank diagnostics of the full training set on the observational loader."""

    def __init__(self, state: TrainingState, args, *, every: int) -> None:
        self.state = state
        self.args = args
        self.every = int(every)

    def on_epoch_end(self, report: EpochReport) -> None:
        if self.every <= 0 or report.epoch % self.every != 0:
            return
        if self.state.rank_loader is None:
            raise ValueError("Rank metrics require a dedicated rank loader.")
        features = extract_normalized_train_features(self.state.model, self.state.rank_loader, self.args)
        entropy, effective_rank, energy_based_rank = entropy_effective_rank(features)
        print(
            "epoch {}, entropy {:.2f}, effective rank {} and energy-based rank {}".format(
                report.epoch, entropy, effective_rank, energy_based_rank
            )
        )
        report.diagnostics.update({
            "Entropy": entropy,
            "Effective rank": effective_rank,
            "Energy-based rank": energy_based_rank,
        })


class RunRecordLogger(Callback):
    """Append each epoch's metric event to the run record."""

    def __init__(self, recorder) -> None:
        self.recorder = recorder

    def on_epoch_end(self, report: EpochReport) -> None:
        self.recorder.log_metrics("ssl", report.epoch, report.payload())


class WandbLogger(Callback):
    """Own the W&B run: the per-epoch event under both key sets, plus exactly one finish.

    The run closes after the trainer returns rather than in ``on_train_end``, so the final linear
    probe still logs into it. A failure closes it through ``on_failure`` and records why.
    """

    def __init__(self, run) -> None:
        self.run = run
        self.finished = False
        self.status: dict[str, Any] = {
            "enabled": run is not None, "finish_called": False, "finish_succeeded": False,
        }

    @classmethod
    def start(cls, args) -> "WandbLogger":
        """Start the configured W&B run, or none when tracking is disabled."""

        if not args.use_wandb:
            return cls(None)
        with preserve_rng_state():
            import wandb

            run = wandb.init(
                project=args.wandb_name,
                name=args.wandb_run_name,
                config={key: value for key, value in vars(args).items() if key != "run_recorder_instance"},
                entity=args.entity,
                group=args.wandb_group or None,
                tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()] or None,
            )
        return cls(run)

    @property
    def identity(self) -> dict[str, Any] | None:
        if self.run is None:
            return None
        return {
            key: value
            for key, value in {
                "id": getattr(self.run, "id", None),
                "name": getattr(self.run, "name", None),
                "entity": getattr(self.run, "entity", None),
                "project": getattr(self.run, "project", None),
                "url": getattr(self.run, "url", None),
            }.items()
            if value is not None
        }

    def on_epoch_end(self, report: EpochReport) -> None:
        if self.run is None:
            return
        payload = report.payload()
        self.run.log({**payload, **canonical_train_metrics(payload)}, step=report.epoch)

    def finish(self) -> None:
        """Close the run, letting a W&B failure reach the caller."""

        if self.run is None or self.finished:
            return
        self.run.finish()
        self.finished = True
        self.status.update({"finish_called": True, "finish_succeeded": True})

    def on_failure(self, error: BaseException) -> None:
        if self.run is None or self.finished:
            return
        try:
            self.finish()
        except Exception as finish_error:  # preserve the original failure and recovery artifacts
            self.status.update({
                "finish_called": True, "finish_succeeded": False, "finish_error": repr(finish_error),
            })


class PeriodicProbe(Callback):
    """Evaluate the encoder with a linear probe on a checkpoint that exists only for the probe."""

    def __init__(
        self,
        state: TrainingState,
        args,
        storage: StoragePolicy,
        probe: ProbeFunction,
        *,
        mode: str,
        every: int,
        total_epochs: int,
    ) -> None:
        self.state = state
        self.args = args
        self.storage = storage
        self.probe = probe
        self.mode = mode
        self.every = int(every or 0)
        self.total_epochs = int(total_epochs)
        self.metrics: dict[str, Any] = {}
        self.last_epoch = 0

    def on_epoch_end(self, report: EpochReport) -> None:
        if self.mode != "periodic" or self.every <= 0 or report.epoch % self.every != 0:
            return
        self._evaluate(report.epoch)

    def on_train_end(self) -> None:
        """Probe the final encoder unless the last periodic probe already covered it."""

        if self.mode == "none" or self.last_epoch == self.total_epochs:
            return
        self._evaluate(self.total_epochs)

    def _evaluate(self, epoch: int) -> None:
        checkpoint = self.state.save(self.args, epoch, self.storage.probe_checkpoint_path)
        try:
            self.metrics = dict(self.probe(str(checkpoint), epoch))
        finally:
            if checkpoint.exists():
                checkpoint.unlink()
        self.last_epoch = epoch


class CheckpointPolicy(Callback):
    """Write recovery checkpoints while training and one final checkpoint, both attested."""

    def __init__(self, state: TrainingState, args, storage: StoragePolicy, recorder) -> None:
        self.state = state
        self.args = args
        self.storage = storage
        self.recorder = recorder

    def on_train_start(self) -> None:
        self.storage.prune_epoch_checkpoints()

    def on_epoch_end(self, report: EpochReport) -> None:
        if not self.storage.saves_epoch(report.epoch):
            return
        checkpoint = self.state.save(self.args, report.epoch, self.storage.epoch_checkpoint_path(report.epoch))
        self.recorder.register_artifact(
            checkpoint, kind="ssl_checkpoint", stage="ssl", epoch=report.epoch, retention_state="recovery",
        )
        self.storage.prune_epoch_checkpoints()

    def on_train_end(self) -> None:
        if not self.storage.keep_checkpoints:
            return
        epochs = self.storage.total_epochs
        checkpoint = self.state.save(self.args, epochs, self.storage.final_checkpoint_path)
        self.recorder.register_artifact(
            checkpoint, kind="ssl_checkpoint", stage="ssl", epoch=epochs, retention_state="final",
        )
