from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.tools.render_concept_ablation_examples import generate_report
from splice.graph_io import save_graph_json


class ConceptAblationReportTests(unittest.TestCase):
    def test_self_contained_crp_report_renders_both_required_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"img_filename": "waterbird_land.jpg", "y": 1, "place": 0, "split": 0},
                {"img_filename": "waterbird_water.jpg", "y": 1, "place": 1, "split": 0},
                {"img_filename": "landbird_land.jpg", "y": 0, "place": 0, "split": 0},
                {"img_filename": "landbird_water.jpg", "y": 0, "place": 1, "split": 0},
            ]
            pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
            for index, row in enumerate(rows):
                Image.new("RGB", (24, 18), color=(30 * index, 80, 140)).save(root / row["img_filename"])

            clip = F.normalize(
                torch.tensor(
                    [
                        [1.0, 1.0, 0.0],
                        [-1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0],
                        [0.0, -1.0, 1.0],
                    ]
                ),
                dim=1,
            )
            sample_ids = [f"waterbirds:{index}" for index in range(4)]
            cache = {
                "cache_version": 1,
                "sample_ids": sample_ids,
                "clip_embeddings": clip,
                "image_mean": torch.zeros(3),
                "splice_codes": torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
                "dictionary": torch.tensor([[1.0, 0.0, 0.0]]),
                "vocabulary": ["background direction"],
                "dino_embeddings": clip,
            }
            cache_path = root / "cache.pt"
            torch.save(cache, cache_path)
            graph_path = root / "graph.json"
            save_graph_json(
                {
                    "artifact": "splice_crp_v2_teacher_graph",
                    "sample_ids": sample_ids,
                    "config": {"orthogonal_tolerance": 1e-6},
                    "groups": [
                        {
                            "group_id": 0,
                            "concept_indices": [0],
                            "concepts": ["background direction"],
                            "basis_rank": 1,
                            "selected": True,
                            "score": 0.2,
                            "null_threshold": 0.1,
                            "coverage": 0.5,
                        }
                    ],
                },
                graph_path,
            )

            output_path = generate_report(cache_path, graph_path, root, root / "report.html")
            report = output_path.read_text(encoding="utf-8")

        self.assertIn("Пара 1: одинаковый label", report)
        self.assertIn("Пара 2: разные labels", report)
        self.assertIn("waterbird", report)
        self.assertIn("landbird", report)
        self.assertIn("background direction", report)
        self.assertIn("data:image/jpeg;base64,", report)
        self.assertIn("+1.000000", report)


if __name__ == "__main__":
    unittest.main()
