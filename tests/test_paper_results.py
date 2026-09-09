import unittest

from scripts.tools.build_paper_results import _distribution, _quantile


class PaperResultsTests(unittest.TestCase):
    def test_quantiles_use_linear_interpolation(self):
        self.assertEqual(_quantile([0.0, 10.0], 0.25), 2.5)
        self.assertEqual(_quantile([0.0, 10.0], 0.95), 9.5)

    def test_distribution_has_reproducible_named_summary(self):
        summary = _distribution([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["mean"], 2.5)


if __name__ == "__main__":
    unittest.main()
