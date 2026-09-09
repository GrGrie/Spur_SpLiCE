import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from splice.artifacts import (
    BINARY_SIZE_THRESHOLD,
    OUTPUT_ROOT,
    binary_destination,
    reference,
    report,
    seed_run,
    shared,
)


class ArtifactPathTests(unittest.TestCase):
    def test_seed_runs_are_grouped_by_seed_then_study_and_arm(self):
        self.assertEqual(
            seed_run(1, "paper", "simclr"),
            OUTPUT_ROOT / "seeds" / "paper" / "seed_01" / "simclr",
        )

    def test_shared_and_report_paths_have_distinct_meanings(self):
        self.assertEqual(shared("waterbirds", "graphs"), OUTPUT_ROOT / "shared" / "waterbirds" / "graphs")
        self.assertEqual(report("paper", "summary.json"), OUTPUT_ROOT / "reports" / "paper" / "summary.json")
        self.assertEqual(reference("wandb_exports"), OUTPUT_ROOT / "reference" / "wandb_exports")

    def test_names_cannot_escape_outputs(self):
        with self.assertRaises(ValueError):
            seed_run(1, "../outside")

    def test_binary_routing_uses_strict_ten_mib_boundary(self):
        identity = {"study": "paper", "seed": 1, "arm": "simclr", "attempt_id": "attempt"}
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SPUR_SPLICE_ARTIFACT_ROOT": directory}
        ):
            local = Path(directory).parent / "result.pth"
            self.assertEqual(
                binary_destination(local, BINARY_SIZE_THRESHOLD, kind="checkpoints", identity=identity),
                local,
            )
            routed = binary_destination(
                local, BINARY_SIZE_THRESHOLD + 1, kind="checkpoints", identity=identity
            )
            self.assertTrue(str(routed).startswith(directory))
            self.assertIn("checkpoints", routed.parts)


if __name__ == "__main__":
    unittest.main()
