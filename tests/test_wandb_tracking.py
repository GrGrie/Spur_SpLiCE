"""What one training run sends to Weights & Biases, from the run it opens to the run it closes.

Every other test disables W&B, so the tracking path was the one part of a cluster run that nothing
exercised. This test runs ``spur_splice.py`` against a recording stand-in for the ``wandb`` module
and asserts the contract the cluster depends on: one run per job, worst-group accuracy as the run
summary, both key sets on every event, the graph provenance in the config and exactly one finish.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from golden_support import write_waterbirds_fixture
from cospro.tracking.artifacts import PROJECT_ROOT

WANDB_STUB = '''
"""A recording stand-in for wandb: the training run sees an ordinary run object."""
import json, os

_RECORD = os.environ["WANDB_STUB_RECORD"]
run = None


class _Config:
    def __init__(self, owner):
        self.owner = owner

    def update(self, payload, allow_val_change=False):
        self.owner.events.append({"kind": "config.update", "keys": sorted(payload)})


class _Run:
    def __init__(self, **keywords):
        self.id, self.name = "stub", keywords.get("name")
        self.entity, self.project = keywords.get("entity"), keywords.get("project")
        self.url = "https://wandb.test/stub"
        self.events = [{"kind": "init", "project": keywords.get("project"), "name": keywords.get("name"),
                        "entity": keywords.get("entity"), "group": keywords.get("group"),
                        "tags": keywords.get("tags"), "config_keys": sorted(keywords.get("config") or {})}]
        self.config = _Config(self)

    def _write(self):
        with open(_RECORD, "w", encoding="utf-8") as handle:
            json.dump(self.events, handle)

    def log(self, payload, step=None):
        self.events.append({"kind": "log", "step": step, "keys": sorted(payload)})
        self._write()

    def define_metric(self, name, summary=None):
        self.events.append({"kind": "define_metric", "name": name, "summary": summary})

    def save(self, path, base_path=None, policy=None):
        self.events.append({"kind": "save", "path": os.path.basename(str(path)), "policy": policy})

    def finish(self):
        self.events.append({"kind": "finish"})
        self._write()


def init(**keywords):
    global run
    run = _Run(**keywords)
    return run
'''


def train_with_stub(root: Path) -> list[dict]:
    """Run one short training job with the stub in place and return what it recorded."""

    stub_dir = root / "stub"
    stub_dir.mkdir(parents=True)
    (stub_dir / "wandb.py").write_text(WANDB_STUB, encoding="utf-8")
    write_waterbirds_fixture(root / "datasets")
    record = root / "wandb_calls.json"
    run_dir = root / "run"
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "WANDB_STUB_RECORD": str(record),
        "PYTHONHASHSEED": "0",
        "SPUR_SPLICE_SCRATCH_ROOT": str(root / "scratch"),
        "SPUR_SPLICE_OUTPUT_ROOT": str(root / "outputs"),
        "PYTHONPATH": os.pathsep.join([str(stub_dir), str(PROJECT_ROOT), os.environ.get("PYTHONPATH", "")]),
    }
    command = [
        sys.executable, "-u", str(PROJECT_ROOT / "spur_splice.py"),
        "--dataset", "waterbirds", "--model", "resnet18_large", "--epochs", "1", "--batch_size", "16",
        "--num_workers", "0", "--device", "cpu", "--amp", "false", "--channels_last", "false",
        "--rank_eval_freq", "1", "--linear_probe_mode", "final", "--linear_probe_max_epochs", "30",
        "--seed", "1", "--study", "tracking", "--arm", "simclr", "--attempt_id", "check",
        "--splice_mode", "none", "--data_folder", str(root / "datasets"),
        "--artifact_dir", str(run_dir / "training"), "--run_record", str(run_dir / "run.json"),
        "--use_wandb", "--wandb_name", "TrackingCheck", "--entity", "stub-entity",
        "--wandb_group", "audit", "--wandb_tags", "audit, tracking",
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, text=True, capture_output=True)
    if result.returncode != 0:
        raise AssertionError(f"training failed:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}")
    return json.loads(record.read_text(encoding="utf-8")), json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


class WandbTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        cls.events, cls.record = train_with_stub(Path(cls._directory.name))

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def kinds(self, kind: str) -> list[dict]:
        return [event for event in self.events if event["kind"] == kind]

    def test_one_run_opens_with_the_job_identity_and_closes_once(self):
        started = self.kinds("init")
        self.assertEqual(len(started), 1, "a training job opens exactly one run")
        self.assertEqual(started[0]["project"], "TrackingCheck")
        self.assertEqual(started[0]["entity"], "stub-entity")
        self.assertEqual(started[0]["group"], "audit")
        self.assertEqual(started[0]["tags"], ["audit", "tracking"])
        self.assertIn("storage_name", started[0]["config_keys"])
        self.assertNotIn("run_recorder_instance", started[0]["config_keys"], "the recorder stays out of W&B")
        self.assertEqual(len(self.kinds("finish")), 1, "the run is finished exactly once")
        self.assertEqual(self.events[-1]["kind"], "finish")

    def test_worst_group_accuracy_is_the_run_summary(self):
        summaries = {event["name"]: event["summary"] for event in self.kinds("define_metric")}
        self.assertEqual(summaries.get("probe/val/wga"), "max")

    def test_every_epoch_reports_both_key_sets(self):
        logged = {key for event in self.kinds("log") for key in event["keys"]}
        for key in ("SSL train loss", "Entropy", "Last linear val worst-group acc"):
            self.assertIn(key, logged, "historical keys stay the primary record")
        for key in ("train/loss/total", "train/representation/entropy", "probe/val/wga"):
            self.assertIn(key, logged, "canonical keys are logged next to them")
        self.assertTrue(all(event["step"] is not None for event in self.kinds("log")))

    def test_the_run_record_and_wandb_agree_on_the_run(self):
        self.assertEqual(self.record["status"], "complete")
        self.assertEqual(self.record["wandb"]["project"], "TrackingCheck")
        self.assertEqual(self.record["wandb"]["id"], "stub")
        self.assertEqual(self.record["wandb"]["url"], "https://wandb.test/stub", "the run link stays a link")
        self.assertIn("Last linear val worst-group acc", self.record["final_metrics"])


if __name__ == "__main__":
    unittest.main()
