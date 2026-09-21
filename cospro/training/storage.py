"""Where a training attempt writes its binaries and how long they stay.

``/home`` carries a 100 GB quota and ``/scratch`` is a slow HDD, so one object owns every decision
about checkpoint and probe files: canonical checkpoints leave for scratch through
:func:`cospro.tracking.artifacts.binary_destination`, an epoch checkpoint exists only to recover a crashed
run and the newest ``--checkpoint_keep_count`` of them survive at any moment. Once training
completes everything except ``last.pth`` and the newest probe artifacts is deleted. The run record
attests every retained checkpoint with its SHA-256, so a deleted file stays identifiable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from cospro.tracking.artifacts import scratch_binary_directory

EPOCH_CHECKPOINT = re.compile(r"epoch_(\d+)\.pth")
PROBE_FEATURES = re.compile(r"probe_features_epoch_(\d+)(?:_.+)?\.pt")
PROBE_RESULT = re.compile(r"probe_features_epoch_(\d+)(?:_.+)?\.json")


def artifact_identity(args) -> dict[str, Any]:
    """The study, seed, arm and attempt that name every artifact of this run."""

    return {name: getattr(args, name) for name in ("study", "seed", "arm", "attempt_id")}


class StoragePolicy:
    """Checkpoint and probe-artifact placement, rolling retention and end-of-run cleanup."""

    def __init__(self, args) -> None:
        self.identity = artifact_identity(args)
        self.run_folder = Path(args.save_folder)
        self.total_epochs = int(args.epochs)
        self.keep_checkpoints = bool(args.keep_checkpoints)
        self.save_freq = int(args.save_freq)
        self.keep_count = int(args.checkpoint_keep_count)
        self.retain_probe_artifacts_every = int(args.retain_probe_artifacts_every)
        self.delete_checkpoints_after_training = bool(args.delete_checkpoints_after_training)
        self.delete_epoch_checkpoints_after_training = bool(args.delete_epoch_checkpoints_after_training)

    # Placement -----------------------------------------------------------------

    def checkpoint_directories(self) -> list[Path]:
        """The run folder and the scratch folder a checkpoint of this run can land in."""

        return [self.run_folder, scratch_binary_directory("checkpoints", self.identity)]

    def feature_directories(self) -> list[Path]:
        return [self.run_folder, scratch_binary_directory("features", self.identity)]

    @property
    def probe_checkpoint_path(self) -> Path:
        """The temporary checkpoint a probe reads; it is deleted as soon as the probe returns."""

        return self.run_folder / "probe_tmp.pth"

    def epoch_checkpoint_path(self, epoch: int) -> Path:
        return self.run_folder / f"epoch_{epoch}.pth"

    @property
    def final_checkpoint_path(self) -> Path:
        return self.run_folder / "last.pth"

    def saves_epoch(self, epoch: int) -> bool:
        """True for an epoch whose checkpoint is kept for crash recovery."""

        return self.keep_checkpoints and self.save_freq > 0 and epoch % self.save_freq == 0

    # Retention during training -------------------------------------------------

    def prune_epoch_checkpoints(self) -> int:
        """Keep only the newest epoch checkpoints and report how many files were removed."""

        if not self.keep_checkpoints:
            return 0
        epoch_checkpoints: list[tuple[int, Path]] = []
        for checkpoint_dir in self.checkpoint_directories():
            for checkpoint_path in checkpoint_dir.glob("epoch_*.pth"):
                match = EPOCH_CHECKPOINT.fullmatch(checkpoint_path.name)
                if match:
                    epoch_checkpoints.append((int(match.group(1)), checkpoint_path))
        epoch_checkpoints.sort(key=lambda item: item[0], reverse=True)
        removed = 0
        for _, checkpoint_path in epoch_checkpoints[self.keep_count :]:
            checkpoint_path.unlink()
            removed += 1
        return removed

    # Cleanup after training ----------------------------------------------------

    def cleanup_after_training(self) -> dict[str, dict[str, Any]]:
        """Apply the configured retention and report what each part removed."""

        if self.delete_checkpoints_after_training or self.delete_epoch_checkpoints_after_training:
            probe_cleanup = (
                self.cleanup_probe_artifacts() if self.delete_checkpoints_after_training
                else self.cleanup_probe_results()
            )
            return {"ssl_checkpoints": self.cleanup_all_checkpoints(), "probe_artifacts": probe_cleanup}
        self.cleanup_temporary_checkpoints()
        # Probe result JSON retention is independent of checkpoint retention: keep the newest
        # result even when every other artifact is preserved.
        return {
            "ssl_checkpoints": {"requested": False, "removed_count": 0, "completed": True},
            "probe_artifacts": self.cleanup_probe_results(),
        }

    def cleanup_temporary_checkpoints(self) -> None:
        """Delete the probe scratch checkpoint and any interrupted write beside it."""

        for directory in self.checkpoint_directories():
            for name in ("probe_tmp.pth", "probe_tmp.pth.tmp"):
                temporary_path = directory / name
                if temporary_path.exists():
                    temporary_path.unlink()

    def cleanup_all_checkpoints(self) -> dict[str, Any]:
        """Delete epoch checkpoint artifacts while preserving the final last.pth."""

        removed_count = 0
        for checkpoint_dir in self.checkpoint_directories():
            if not checkpoint_dir.exists():
                continue
            for checkpoint_path in checkpoint_dir.iterdir():
                if not checkpoint_path.is_file() or checkpoint_path.name == "last.pth":
                    continue
                if not (checkpoint_path.name.endswith(".pth") or checkpoint_path.name.endswith(".pth.tmp")):
                    continue
                checkpoint_path.unlink()
                removed_count += 1
        print(f"[INFO] Removed {removed_count} recovery checkpoint files")
        return {"requested": True, "removed_count": removed_count, "completed": True}

    def cleanup_probe_results(self) -> dict[str, Any]:
        """Keep only the newest probe result JSON."""

        removed_result_count = 0
        result_paths: list[tuple[int, Path]] = []
        for feature_dir in self.feature_directories():
            for result_path in feature_dir.glob("probe_features_epoch_*.json"):
                match = PROBE_RESULT.fullmatch(result_path.name)
                if match is not None:
                    result_paths.append((int(match.group(1)), result_path))

        if result_paths:
            # There is one result JSON per probe epoch. Keep the newest one and remove older
            # periodic snapshots.
            result_paths.sort(key=lambda item: (item[0], item[1].name))
            for _, result_path in result_paths[:-1]:
                if result_path.exists():
                    result_path.unlink()
                    removed_result_count += 1

        return {
            "requested": True,
            "removed_count": removed_result_count,
            "removed_probe_result_count": removed_result_count,
            "completed": True,
        }

    def cleanup_probe_artifacts(self) -> dict[str, Any]:
        """Keep only the newest probe JSON and the selected bulky feature tensors."""

        interval = self.retain_probe_artifacts_every
        result_cleanup = self.cleanup_probe_results()
        removed_count = 0
        retained_epochs: set[int] = set()
        feature_paths = []
        for feature_dir in self.feature_directories():
            feature_paths.extend(feature_dir.glob("probe_features_epoch_*.pt"))

        for feature_path in feature_paths:
            match = PROBE_FEATURES.fullmatch(feature_path.name)
            if match is None:
                continue
            epoch = int(match.group(1))
            if epoch == self.total_epochs or (interval > 0 and epoch > 0 and epoch % interval == 0):
                retained_epochs.add(epoch)
                continue
            feature_path.unlink()
            removed_count += 1
        return {
            "requested": True,
            "removed_count": removed_count + result_cleanup["removed_probe_result_count"],
            "removed_probe_result_count": result_cleanup["removed_probe_result_count"],
            "retained_epochs": sorted(retained_epochs),
            "completed": True,
        }
