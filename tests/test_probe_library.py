"""The probe seam: measuring writes nothing, persisting writes exactly what run.json attests.

``evaluate_probe`` is the only part of the probe that computes, so these tests hold it to having no
file, no W&B and no run-record side effect. The side effects are then checked where they belong.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from cospro.evaluation import (
    ProbeArtifacts,
    ProbeOptions,
    evaluate_probe,
    log_probe_result,
    persist_probe_result,
)
from cospro.evaluation import probe as probe_module
from cospro.metrics import compute_group_metrics


class GroupedFeatures(TensorDataset):
    """A split that reports group metrics the way a real adapter does."""

    def eval(self, predictions, targets, metadata):
        return compute_group_metrics(predictions, targets, metadata).as_spurssl_dict(), ""


class ToyDataset:
    name = "toy"
    num_classes = 2
    Config = dict


class FakeRecorder:
    def __init__(self) -> None:
        self.artifacts: list[tuple[str, str]] = []
        self.events: list[tuple[str, int]] = []

    def register_artifact(self, path, *, kind, stage, epoch=None, retention_state="retained"):
        self.artifacts.append((kind, retention_state))

    def log_metrics(self, stage, step, metrics):
        self.events.append((stage, step))


class FakeWandbRun:
    def __init__(self) -> None:
        self.logged: dict = {}
        self.summaries: dict[str, str] = {}
        self.config = self

    def log(self, payload, step=None):
        self.logged.update(payload)

    def define_metric(self, name, summary=None):
        self.summaries[name] = summary

    def update(self, payload, allow_val_change=False):
        self.logged.update(payload)


def toy_loader() -> DataLoader:
    torch.manual_seed(0)
    features = torch.tensor([[-1.0, 0.0], [-0.8, 0.1], [1.0, 0.0], [0.8, -0.1]] * 4)
    targets = torch.tensor([0, 1, 0, 1] * 4)
    spurious = torch.tensor([0, 0, 1, 1] * 4)
    metadata = torch.stack((spurious, targets), dim=1)
    return DataLoader(GroupedFeatures(features, targets, metadata), batch_size=8)


def measure(options: ProbeOptions | None = None):
    loader = toy_loader()
    options = options or ProbeOptions(spurious_probe=False, max_epochs=20)
    with patch.object(probe_module, "build_probe_loaders", lambda *a, **k: (loader, loader)):
        return evaluate_probe(torch.nn.Identity(), ToyDataset, options, feature_dim=2)


class EvaluateProbeTests(unittest.TestCase):
    def test_measuring_writes_no_file_and_opens_no_run(self):
        refuse = AssertionError("evaluate_probe must not write")
        with patch.object(probe_module.torch, "save", side_effect=refuse), patch.object(
            probe_module, "atomic_write_json", side_effect=refuse
        ), patch.dict("sys.modules", {"wandb": None}):
            result = measure()
        self.assertIn("Last linear val worst-group acc", result.metrics)
        self.assertEqual(len(result.epoch_metrics), len(result.history.val_accuracy))
        self.assertEqual(sorted(result.sample_ids), ["evaluation", "train"])
        self.assertEqual(result.history_window, min(10, len(result.history.val_accuracy)))

    def test_the_spurious_probe_is_optional(self):
        without = measure(ProbeOptions(spurious_probe=False, max_epochs=20))
        with_probe = measure(ProbeOptions(spurious_probe=True, max_epochs=20))
        self.assertNotIn("Spurious probe last val acc", without.metrics)
        self.assertIn("Spurious probe last val acc", with_probe.metrics)


class PersistProbeResultTests(unittest.TestCase):
    def persist(self, directory: Path, **artifact_fields):
        result = measure()
        options = ProbeOptions(spurious_probe=False, train_split="ds_train", eval_split="val")
        recorder = FakeRecorder()
        artifacts = ProbeArtifacts(
            directory=directory,
            identity={"study": "golden", "seed": 1, "arm": "training", "attempt_id": "attempt"},
            recorder=recorder,
            **artifact_fields,
        )
        return result, recorder, persist_probe_result(result, options, artifacts)

    def test_it_writes_the_features_and_the_result_and_attests_both(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict("os.environ", {"SPUR_SPLICE_SCRATCH_ROOT": str(root / "scratch")}):
                result, recorder, paths = self.persist(root, ssl_epoch=100, ssl_total_epochs=100)

            self.assertEqual(paths["result"].name, "probe_features_epoch_100_ds_train_val.json")
            self.assertTrue(paths["features"].is_file())
            payload = json.loads(paths["result"].read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "linear-probe-result-v3")
            self.assertEqual(payload["ssl_epoch"], 100)
            self.assertEqual(payload["metrics"], result.metrics)
            self.assertNotIn("wandb", payload["group_metrics"]["val"], "W&B keys stay out of the record")

            self.assertEqual(
                recorder.artifacts, [("probe_features", "final"), ("probe_result", "final")],
                "the last probe of a run keeps its artifacts",
            )
            stages = [stage for stage, _ in recorder.events]
            self.assertEqual(stages.count("linear_probe"), len(result.epoch_metrics))
            self.assertEqual(stages[-1], "linear_probe_final")

    def test_an_intermediate_probe_keeps_its_artifacts_only_when_asked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict("os.environ", {"SPUR_SPLICE_SCRATCH_ROOT": str(root / "scratch")}):
                _, temporary, _ = self.persist(root, ssl_epoch=50, ssl_total_epochs=500)
                _, retained, _ = self.persist(root, ssl_epoch=50, ssl_total_epochs=500, retain_every=25)
        self.assertEqual(temporary.artifacts[0], ("probe_features", "temporary"))
        self.assertEqual(retained.artifacts[0], ("probe_features", "retained"))


class LogProbeResultTests(unittest.TestCase):
    def test_a_probe_reaches_wandb_under_both_key_sets(self):
        result = measure()
        options = ProbeOptions(spurious_probe=False, eval_split="val")
        run = FakeWandbRun()
        log_probe_result(result, options, run=run, ssl_epoch=7)
        self.assertEqual(run.summaries["probe/val/wga"], "max")
        self.assertEqual(
            run.logged["probe/val/wga"], result.metrics["Last linear val worst-group acc"],
        )
        self.assertIn("Linear val group 0 count", run.logged)
        self.assertIn("Linear val group (target,context)=(0,0) acc", run.logged)
        self.assertEqual(run.logged["probe_solver"], "logistic")
        # Lists stay out of the scalar panel.
        self.assertNotIn("Linear val group accuracies", run.logged)

    def test_logging_without_a_run_does_nothing(self):
        log_probe_result(measure(), ProbeOptions(), run=None, ssl_epoch=1)


if __name__ == "__main__":
    unittest.main()
