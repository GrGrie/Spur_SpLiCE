import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.runner import command_for, load_manifest, main, matrix, run


class ExperimentRunnerTests(unittest.TestCase):
    def test_manifest_expands_to_seed_major_matrix(self):
        manifest = {"name": "study", "seeds": [1, 2], "common": {}, "arms": {"a": {}, "b": {}}}
        self.assertEqual(matrix(manifest), [(1, "a"), (1, "b"), (2, "a"), (2, "b")])

    def test_command_uses_seed_first_output(self):
        manifest = {"name": "study", "seeds": [1], "common": {}, "arms": {"arm": {"args": {"splice_mode": "none"}}}}
        command, output = command_for(manifest, 1, "arm")
        self.assertIn("seed_01", str(output))
        self.assertIn("--checkpoint_dir", command)

    def test_missing_manifest_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"name": "broken"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(path)

    @staticmethod
    def _manifest():
        return {
            "name": "study",
            "seeds": [1],
            "common": {},
            "arms": {"arm": {"args": {"splice_mode": "none"}}},
        }

    def test_existing_execution_is_refused_without_an_explicit_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "seeds" / "seed_01" / "study" / "arm"
            output.mkdir(parents=True)
            marker = output / "do-not-overwrite.txt"
            marker.write_text("kept", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                run(self._manifest(), 1, "arm", artifact_root=root)
            self.assertEqual(marker.read_text(encoding="utf-8"), "kept")

    def test_completed_execution_can_only_be_reused_with_matching_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, output = command_for(self._manifest(), 1, "arm", artifact_root=root)
            status_dir = output / "training" / "run"
            status_dir.mkdir(parents=True)
            (output / "command.json").write_text(json.dumps(command), encoding="utf-8")
            (status_dir / "run_status.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            with patch("experiments.runner.subprocess.run") as subprocess_run:
                self.assertEqual(
                    run(self._manifest(), 1, "arm", existing="reuse", artifact_root=root),
                    output,
                )
            subprocess_run.assert_not_called()

    def test_resume_adds_the_discovered_checkpoint_without_overwriting_identity_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, output = command_for(self._manifest(), 1, "arm", artifact_root=root)
            run_dir = output / "training" / "run"
            run_dir.mkdir(parents=True)
            (output / "command.json").write_text(json.dumps(command), encoding="utf-8")
            checkpoint = run_dir / "epoch_3.pth"
            checkpoint.write_bytes(b"checkpoint")
            (run_dir / "run_status.json").write_text(
                json.dumps({"status": "failed"}), encoding="utf-8"
            )
            with patch("experiments.runner.subprocess.run") as subprocess_run:
                run(self._manifest(), 1, "arm", existing="resume", artifact_root=root)
            executed = subprocess_run.call_args.args[0]
            self.assertEqual(executed[-2:], ["--resume", str(checkpoint)])
            record = json.loads((output / "execution.json").read_text(encoding="utf-8"))
            self.assertEqual(record["schema_version"], 1)
            self.assertEqual(len(record["manifest_sha256"]), 64)

    def test_new_attempt_gets_a_separate_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "seeds" / "seed_01" / "study" / "arm"
            base.mkdir(parents=True)
            (base / "command.json").write_text("[]", encoding="utf-8")
            with patch("experiments.runner.subprocess.run"):
                output = run(
                    self._manifest(), 1, "arm", existing="new-attempt", artifact_root=root
                )
            self.assertEqual(output, base / "attempts" / "attempt_0001")
            self.assertTrue((output / "execution.json").is_file())

    def test_negative_task_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self._manifest()), encoding="utf-8")
            with patch("sys.argv", ["runner.py", str(manifest_path), "--task", "-1"]):
                with self.assertRaisesRegex(SystemExit, "task must be in 0..0"):
                    main()


if __name__ == "__main__":
    unittest.main()
