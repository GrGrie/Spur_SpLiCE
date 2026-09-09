import json
import tempfile
import unittest
from pathlib import Path

from experiments.runner import command_for, load_manifest, matrix


class ExperimentRunnerTests(unittest.TestCase):
    def test_manifest_expands_to_seed_major_matrix(self):
        manifest = {"name": "study", "seeds": [1, 2], "common": {}, "arms": {"a": {}, "b": {}}}
        self.assertEqual(matrix(manifest), [(1, "a"), (1, "b"), (2, "a"), (2, "b")])

    def test_command_uses_seed_first_output(self):
        manifest = {"name": "study", "seeds": [1], "common": {}, "arms": {"arm": {"args": {"splice_mode": "none"}}}}
        command, output = command_for(manifest, 1, "arm", "attempt")
        self.assertIn("seed_01", str(output))
        self.assertEqual(output.name, "attempt")
        self.assertIn("--checkpoint_dir", command)
        self.assertIn("--run_record", command)

    def test_missing_manifest_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"name": "broken"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
