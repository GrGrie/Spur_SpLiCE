import unittest

from splice.artifacts import OUTPUT_ROOT, reference, report, seed_run, shared


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


if __name__ == "__main__":
    unittest.main()
