import tempfile
import unittest
from pathlib import Path

from splice.reporting import render_report


class ReportingTests(unittest.TestCase):
    def test_report_is_escaped_and_self_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            render_report("A < B", [{"title": "Metrics", "data": {"score": 0.5}}], output)
            page = output.read_text(encoding="utf-8")
        self.assertIn("A &lt; B", page)
        self.assertIn('&quot;score&quot;: 0.5', page)
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)


if __name__ == "__main__":
    unittest.main()
