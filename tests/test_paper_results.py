import json
import unittest

from scripts.tools.build_paper_results import PROJECT_ROOT, _distribution, _quantile, build


class PaperResultsTests(unittest.TestCase):
    def test_quantiles_use_linear_interpolation(self):
        self.assertEqual(_quantile([0.0, 10.0], 0.25), 2.5)
        self.assertEqual(_quantile([0.0, 10.0], 0.95), 9.5)

    def test_distribution_has_reproducible_named_summary(self):
        summary = _distribution([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["mean"], 2.5)

    def test_checked_registry_rebuilds_from_committed_outputs(self):
        expected = json.loads((PROJECT_ROOT / "paper_results.json").read_text(encoding="utf-8"))
        self.assertEqual(build(PROJECT_ROOT / "outputs"), expected)


if __name__ == "__main__":
    unittest.main()
