"""The trainer seam: callback order, RNG isolation and the checkpoint retention guard.

Ten epochs run with a stubbed training step, so the loop, the callbacks and the storage policy are
the real ones while the gradients are not. The central assertion is the peak number of checkpoint
files that exist at once: ``/home`` carries a 100 GB quota, so a 500-epoch run must never hold more
than ``--checkpoint_keep_count`` recovery checkpoints plus the temporary one a probe reads.
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from cospro.methods import TrainingMethod
from cospro.training import (
    Callback,
    CheckpointPolicy,
    EpochReport,
    PeriodicProbe,
    RunRecordLogger,
    StoragePolicy,
    Trainer,
    TrainingState,
)

EPOCHS = 10
KEEP_COUNT = 2
SAVE_FREQ = 1
PROBE_FREQ = 5


def training_arguments(save_folder: Path) -> argparse.Namespace:
    return argparse.Namespace(
        study="golden", seed=0, arm="training", attempt_id="attempt",
        save_folder=str(save_folder), epochs=EPOCHS, resume="",
        keep_checkpoints=True, save_freq=SAVE_FREQ, checkpoint_keep_count=KEEP_COUNT,
        retain_probe_artifacts_every=0,
        delete_checkpoints_after_training=True, delete_epoch_checkpoints_after_training=False,
        linear_probe_mode="periodic", linear_probe_freq=PROBE_FREQ,
        learning_rate=0.1, lr_decay_epochs=[8], lr_decay_rate=0.1, cosine=False,
        device="cpu", temp=0.5,
    )


class FakeRecorder:
    """Records what a real run record would attest."""

    def __init__(self) -> None:
        self.artifacts: list[tuple[str, int, bool]] = []
        self.metrics: list[tuple[int, dict]] = []

    def register_artifact(self, path, *, kind, stage, epoch=None, retention_state="retained"):
        self.artifacts.append((retention_state, epoch, Path(path).is_file()))

    def log_metrics(self, stage, step, metrics):
        self.metrics.append((step, dict(metrics)))


class CheckpointCensus(Callback):
    """Count the checkpoint files alive at the end of every epoch."""

    def __init__(self, storage: StoragePolicy) -> None:
        self.storage = storage
        self.counts: list[int] = []

    def on_epoch_end(self, report: EpochReport) -> None:
        self.counts.append(count_checkpoints(self.storage))


class NoisyCallback(Callback):
    """An observer that draws random numbers, as a probe or an augmented evaluation would."""

    def on_epoch_end(self, report: EpochReport) -> None:
        torch.rand(4)
        random.random()


def count_checkpoints(storage: StoragePolicy) -> int:
    return sum(
        len([path for path in directory.glob("*.pth") if path.is_file()])
        for directory in storage.checkpoint_directories()
        if directory.exists()
    )


def training_state() -> TrainingState:
    model = torch.nn.Linear(4, 2)
    return TrainingState(
        model=model,
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        method=TrainingMethod(),
        train_loader=None,
    )


def stub_epoch(*arguments, **keywords) -> dict[str, float]:
    """Stand in for one SSL epoch: the losses a real epoch reports, without the gradients."""

    return {"loss": 1.0, "simclr_loss": 0.9, "decor_loss": 0.0, "entropy_loss": 0.0, "splice_loss": 0.1}


class TrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        scratch = patch.dict(os.environ, {"SPUR_SPLICE_SCRATCH_ROOT": str(root / "scratch")})
        scratch.start()
        self.addCleanup(scratch.stop)
        epoch = patch("cospro.training.trainer.train_one_epoch", stub_epoch)
        epoch.start()
        self.addCleanup(epoch.stop)
        self.args = training_arguments(root / "run")
        Path(self.args.save_folder).mkdir(parents=True)
        self.recorder = FakeRecorder()
        self.storage = StoragePolicy(self.args)
        self.state = training_state()
        self.probed: list[int] = []
        self.counts_while_probing: list[int] = []

    def probe(self, checkpoint: str, epoch: int) -> dict[str, float]:
        self.assertTrue(Path(checkpoint).is_file(), "the probe reads a checkpoint written for it")
        self.probed.append(epoch)
        self.counts_while_probing.append(count_checkpoints(self.storage))
        return {"Last linear val worst-group acc": 40.0 + epoch}

    def build_trainer(self, census: CheckpointCensus, extra: list[Callback] = ()) -> tuple[Trainer, PeriodicProbe]:
        probe = PeriodicProbe(
            self.state, self.args, self.storage, self.probe,
            mode=self.args.linear_probe_mode, every=self.args.linear_probe_freq, total_epochs=EPOCHS,
        )
        callbacks = [
            RunRecordLogger(self.recorder),
            *extra,
            probe,
            CheckpointPolicy(self.state, self.args, self.storage, self.recorder),
            census,
        ]
        return Trainer(self.state, self.args, callbacks, torch.device("cpu")), probe

    def test_ten_epochs_hold_at_most_the_configured_recovery_checkpoints(self):
        census = CheckpointCensus(self.storage)
        trainer, probe = self.build_trainer(census)
        trainer.fit()

        self.assertEqual(len(census.counts), EPOCHS)
        self.assertEqual(max(census.counts), KEEP_COUNT, "epoch checkpoints roll over instead of piling up")
        # A probe adds its own temporary checkpoint, so the run peaks one above the retention count.
        self.assertEqual(max(self.counts_while_probing), KEEP_COUNT + 1)
        self.assertEqual(self.probed, [5, 10], "periodic probes run on their schedule and cover the last epoch")
        self.assertEqual(probe.metrics["Last linear val worst-group acc"], 50.0)
        self.assertFalse(self.storage.probe_checkpoint_path.exists(), "the probe checkpoint is temporary")

        final = [entry for entry in self.recorder.artifacts if entry[0] == "final"]
        self.assertEqual(final, [("final", EPOCHS, True)], "the final checkpoint is attested once")
        self.assertEqual([step for step, _ in self.recorder.metrics], list(range(1, EPOCHS + 1)))

        cleanup = self.storage.cleanup_after_training()
        self.assertTrue(cleanup["ssl_checkpoints"]["completed"])
        self.assertEqual(
            sorted(path.name for directory in self.storage.checkpoint_directories()
                   if directory.exists() for path in directory.glob("*.pth")),
            ["last.pth"],
            "cleanup keeps the final checkpoint and nothing else",
        )

    def test_observing_callbacks_cannot_move_the_training_rng_stream(self):
        census = CheckpointCensus(self.storage)
        trainer, _ = self.build_trainer(census, extra=[NoisyCallback()])
        torch.manual_seed(7)
        random.seed(7)
        trainer.fit()
        after_noisy_run = (torch.rand(3).tolist(), random.random())

        torch.manual_seed(7)
        random.seed(7)
        expected = (torch.rand(3).tolist(), random.random())
        self.assertEqual(after_noisy_run, expected)

    def test_a_failing_epoch_reaches_the_callbacks_and_the_caller(self):
        class Failing(Callback):
            def __init__(self) -> None:
                self.errors: list[BaseException] = []

            def on_epoch_end(self, report: EpochReport) -> None:
                if report.epoch == 3:
                    raise RuntimeError("out of memory")

            def on_failure(self, error: BaseException) -> None:
                self.errors.append(error)

        failing = Failing()
        trainer, _ = self.build_trainer(CheckpointCensus(self.storage), extra=[failing])
        with self.assertRaises(RuntimeError):
            trainer.fit()
        self.assertEqual([str(error) for error in failing.errors], ["out of memory"])


if __name__ == "__main__":
    unittest.main()
