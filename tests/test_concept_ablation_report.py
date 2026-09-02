from __future__ import annotations

import re
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
                {"img_filename": "landbird_land_2.jpg", "y": 0, "place": 0, "split": 0},
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
                        [0.0, 2.0, 1.0],
                    ]
                ),
                dim=1,
            )
            sample_ids = [f"waterbirds:{index}" for index in range(5)]
            cache = {
                "cache_version": 1,
                "sample_ids": sample_ids,
                "clip_embeddings": clip,
                "image_mean": torch.zeros(3),
                "splice_codes": torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0]]),
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
                    "selected_group_ids": [0],
                    "neighbor_indices": torch.tensor([[1], [-1], [-1], [-1], [-1]]),
                    "weights": torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0]]),
                    "edge_confidences": torch.tensor([[0.2], [0.0], [0.0], [0.0], [0.0]]),
                    "group_ids": torch.tensor([[0], [-1], [-1], [-1], [-1]]),
                    "intervention_gains": torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0]]),
                    "anchor_confidence": torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0]),
                    "degree_stats": {"edge_count": 1, "supported_anchors": 1, "coverage": 0.2},
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
            compact_path = generate_report(
                cache_path,
                graph_path,
                root,
                root / "compact.html",
                max_interventions=1,
                edges_per_group=0,
            )
            compact_report = compact_path.read_text(encoding="utf-8")

        self.assertIn("Pair 1: same target", report)
        self.assertIn("Pair 2: different targets", report)
        self.assertIn("waterbird", report)
        self.assertIn("landbird", report)
        self.assertIn("background direction", report)
        self.assertIn("data:image/jpeg;base64,", report)
        self.assertIn("+1.000000", report)
        self.assertIn("Pair 3: same target and spurious", report)
        self.assertIn("Pair 4: different target and spurious", report)
        self.assertIn("Typical retained teacher edges", report)
        self.assertIn("median edge confidence", report)
        self.assertIn('<html lang="en">', report)
        self.assertIsNone(re.search(r"[\u0400-\u04FF]", report))
        self.assertEqual(report.count('<figure class="card">'), 8)
        self.assertEqual(report.count('<figure class="card compact">'), 2)
        self.assertEqual(compact_report.count('<figure class="card">'), 8)


if __name__ == "__main__":
    unittest.main()
