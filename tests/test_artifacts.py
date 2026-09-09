import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from splice.artifacts import OUTPUT_ROOT, reference, report, resolve_output_root, seed_run, shared


class ArtifactPathTests(unittest.TestCase):
    def test_seed_runs_are_grouped_by_seed_then_study_and_arm(self):
        self.assertEqual(
            seed_run(1, "paper", "simclr"),
            OUTPUT_ROOT / "seeds" / "seed_01" / "paper" / "simclr",
        )

    def test_shared_and_report_paths_have_distinct_meanings(self):
        self.assertEqual(shared("waterbirds", "graphs"), OUTPUT_ROOT / "shared" / "waterbirds" / "graphs")
        self.assertEqual(report("paper", "summary.json"), OUTPUT_ROOT / "reports" / "paper" / "summary.json")
        self.assertEqual(reference("wandb_exports"), OUTPUT_ROOT / "reference" / "wandb_exports")

    def test_names_cannot_escape_outputs(self):
        with self.assertRaises(ValueError):
            seed_run(1, "../outside")

    def test_artifact_root_can_be_configured_explicitly_or_by_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(seed_run(2, "paper", root=root), root / "seeds" / "seed_02" / "paper")
            with patch.dict("os.environ", {"SPUR_SPLICE_ARTIFACT_ROOT": str(root)}):
                self.assertEqual(resolve_output_root(), root)
                self.assertEqual(shared("waterbirds"), root / "shared" / "waterbirds")


if __name__ == "__main__":
    unittest.main()
