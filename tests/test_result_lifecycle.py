import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.tools.archive_legacy import archive_legacy
from scripts.tools.collect_results import collect
from scripts.tools.migrate_outputs import apply_migration, discover_result_root, migration_plan
from splice.artifacts import BINARY_SIZE_THRESHOLD
from splice.run_recording import RunRecorder


class ResultLifecycleTests(unittest.TestCase):
    def test_legacy_archive_is_verified_before_source_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy"
            (source / "study").mkdir(parents=True)
            (source / "study" / "result.json").write_text('{"metric": 1}\n', encoding="utf-8")
            (source / "notes.txt").write_text("historical\n", encoding="utf-8")
            archive = root / "scratch" / "legacy.tar.gz"
            manifest = root / "reports" / "legacy.json"

            payload = archive_legacy(source, archive, manifest, delete_source=True)

            self.assertEqual(payload["file_count"], 2)
            self.assertEqual(payload["suffix_counts"], {".json": 1, ".txt": 1})
            self.assertTrue(payload["source_deleted_after_verification"])
            self.assertTrue(archive.is_file())
            self.assertTrue(manifest.is_file())
            self.assertFalse(source.exists())

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

    def test_migration_discovers_flat_legacy_outputs_and_routes_last_pt(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SPUR_SPLICE_ARTIFACT_ROOT": str(Path(directory) / "scratch")}
        ):
            base = Path(directory)
            source = base / "outputs"
            run = source / "legacy_study" / "seed_01"
            run.mkdir(parents=True)
            checkpoint = run / "last.pt"
            with checkpoint.open("wb") as stream:
                stream.seek(BINARY_SIZE_THRESHOLD)
                stream.write(b"x")
            graph = run / "teacher_graph.json"
            graph.write_text(json.dumps({"schema": "teacher-graph-v1"}), encoding="utf-8")
            readme = source / "README.md"
            readme.write_text("# Outputs\n", encoding="utf-8")

            self.assertEqual(discover_result_root(source), source.resolve())
            with patch("scripts.tools.migrate_outputs.OUTPUT_ROOT", base / "canonical"):
                plan = migration_plan(source)
                by_name = {item["source"].name: item for item in plan}
                self.assertTrue(by_name["last.pt"]["heavy_binary"])
                self.assertIn("scratch", by_name["last.pt"]["destination"].parts)
                self.assertFalse(by_name["teacher_graph.json"]["heavy_binary"])
                self.assertEqual(by_name["README.md"]["destination"], base / "canonical" / "README.md")
                graph_destination = (
                    base / "canonical" / "shared" / "legacy" / "legacy_study" / "seed_01" / "teacher_graph.json"
                )
                self.assertEqual(by_name["teacher_graph.json"]["destination"], graph_destination)
                summary = apply_migration(plan, delete_source=True)

            self.assertTrue(graph_destination.is_file())
            self.assertTrue(by_name["last.pt"]["destination"].is_file())
            self.assertFalse(run.exists())
            self.assertTrue(summary["deleted_source"])

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
