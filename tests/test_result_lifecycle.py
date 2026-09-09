import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.tools.collect_results import collect
from scripts.tools.migrate_outputs import apply_migration, discover_result_root, migration_plan
from splice.run_recording import RunRecorder


class ResultLifecycleTests(unittest.TestCase):
    def test_recorder_redacts_secrets_and_attests_scratch_artifact(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SPUR_SPLICE_ARTIFACT_ROOT": directory}
        ):
            root = Path(directory)
            record_path = root / "record.json"
            checkpoint = root / "checkpoints" / "Spur_SpLiCE" / "study" / "final.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint_bytes = b"checkpoint"
            checkpoint.write_bytes(checkpoint_bytes)
            recorder = RunRecorder(
                record_path,
                identity={"study": "study", "seed": 1, "arm": "arm", "attempt_id": "attempt"},
                config={"api_token": "secret", "learning_rate": 0.1, "data_folder": "/home/user/data"},
            )
            recorder.log_metrics("ssl", 1, {"loss": 2.0})
            recorder.register_artifact(
                checkpoint, kind="ssl_checkpoint", stage="ssl", epoch=1, retention_state="final"
            )
            recorder.finish(final_metrics={"Linear val acc": 50.0})
            payload = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["api_token"], "<redacted>")
            self.assertTrue(payload["config"]["data_folder"].startswith("external://"))
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["artifacts"][0]["availability"], "cluster-only")
            self.assertEqual(payload["artifacts"][0]["bytes"], len(checkpoint_bytes))
            self.assertEqual(len(payload["artifacts"][0]["sha256"]), 64)

    def test_collector_marks_missing_matrix_entries_partial(self):
        manifest = {"name": "study", "seeds": [1, 2], "common": {}, "arms": {"arm": {}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            record_dir = root / "outputs" / "seeds" / "study" / "seed_01" / "arm" / "attempt"
            record_dir.mkdir(parents=True)
            (record_dir / "run.json").write_text(
                json.dumps({
                    "schema": "run-record-v1",
                    "identity": {"study": "study", "seed": 1, "arm": "arm", "attempt_id": "attempt"},
                    "status": "complete",
                    "timestamps": {"updated_at": "2026-01-01T00:00:00Z"},
                    "artifacts": [],
                    "final_metrics": {"Linear val acc": 50.0},
                }),
                encoding="utf-8",
            )
            output = root / "results.json"
            with patch("scripts.tools.collect_results.OUTPUT_ROOT", root / "outputs"), patch(
                "scripts.tools.collect_results.PROJECT_ROOT", root
            ):
                payload = collect(manifest_path, output)
            self.assertEqual(payload["status"], "partial")
            self.assertEqual(payload["matrix"]["missing"], [{"seed": 2, "arm": "arm"}])
            self.assertTrue(output.is_file())

    def test_migration_discovers_direct_and_nested_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            direct = base / "direct"
            (direct / "seeds").mkdir(parents=True)
            (direct / "reports").mkdir()
            self.assertEqual(discover_result_root(direct), direct.resolve())
            wrapper = base / "wrapper"
            nested = wrapper / "outputs"
            (nested / "seeds").mkdir(parents=True)
            (nested / "shared").mkdir()
            self.assertEqual(discover_result_root(wrapper), nested.resolve())

    def test_migration_backfills_legacy_run_record(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SPUR_SPLICE_ARTIFACT_ROOT": str(Path(directory) / "scratch")}
        ):
            base = Path(directory)
            source = base / "legacy"
            run = source / "seeds" / "seed_01" / "paper" / "simclr" / "training" / "attempt"
            run.mkdir(parents=True)
            (source / "reports").mkdir()
            (run / "run_status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
            (run / "args.json").write_text(json.dumps({"dataset": "waterbirds"}), encoding="utf-8")
            destination = base / "canonical"
            with patch("scripts.tools.migrate_outputs.OUTPUT_ROOT", destination):
                plan = migration_plan(source)
                summary = apply_migration(plan)
                repeated = apply_migration(plan)
            record = destination / "seeds" / "paper" / "seed_01" / "simclr" / "attempt" / "run.json"
            self.assertEqual(summary["backfilled_run_records"], 1)
            self.assertTrue(record.is_file())
            self.assertEqual(json.loads(record.read_text(encoding="utf-8"))["status"], "complete")
            self.assertEqual(summary["files"], repeated["files"])


if __name__ == "__main__":
    unittest.main()
