from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.tools.render_group_similarity_sweep import generate_report
from splice.graph_io import save_graph_json


class GroupSimilaritySweepTests(unittest.TestCase):
    def test_report_contains_shared_anchor_pair_sweeps_for_both_triplets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = []
            vectors = []
            # Three examples in every Waterbirds joint metadata group provide
            # enough candidates for all shared-anchor high/low selections.
            for label in (0, 1):
                for background in (0, 1):
                    for repeat in range(3):
                        filename = f"y{label}_a{background}_{repeat}.jpg"
                        metadata.append({"img_filename": filename, "y": label, "place": background, "split": 0})
                        Image.new("RGB", (28, 20), color=(40 + label * 90, 40 + background * 90, 40 + repeat * 50)).save(root / filename)
                        vectors.append([1.0 + repeat, float(label * 2 - 1), float(background * 2 - 1), 0.2 * repeat])
            pd.DataFrame(metadata).to_csv(root / "metadata.csv", index=False)
            embeddings = F.normalize(torch.tensor(vectors), dim=1)
            sample_ids = [f"waterbirds:{index}" for index in range(len(metadata))]
            cache_path = root / "cache.pt"
            torch.save(
                {
                    "cache_version": 1,
                    "sample_ids": sample_ids,
                    "clip_embeddings": embeddings,
                    "image_mean": torch.zeros(4),
                    "splice_codes": torch.ones((len(metadata), 2)),
                    "dictionary": torch.tensor(
                        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
                    ),
                    "vocabulary": ["selected direction", "rejected direction"],
                },
                cache_path,
            )
            graph_path = root / "graph.json"
            save_graph_json(
                {
                    "artifact": "splice_crp_v4_teacher_graph",
                    "sample_ids": sample_ids,
                    "config": {"orthogonal_tolerance": 1e-6},
                    "selected_group_ids": [0],
                    "neighbor_indices": torch.full((len(metadata), 1), -1),
                    "weights": torch.zeros((len(metadata), 1)),
                    "edge_confidences": torch.zeros((len(metadata), 1)),
                    "group_ids": torch.full((len(metadata), 1), -1),
                    "intervention_gains": torch.zeros((len(metadata), 1)),
                    "anchor_confidence": torch.zeros(len(metadata)),
                    "degree_stats": {},
                    "groups": [
                        {
                            "group_id": 0,
                            "concept_indices": [0],
                            "concepts": ["selected direction"],
                            "basis_rank": 1,
                            "selected": True,
                            "score": 0.2,
                            "coverage": 1.0,
                        },
                        {
                            "group_id": 1,
                            "concept_indices": [1],
                            "concepts": ["rejected direction"],
                            "basis_rank": 1,
                            "selected": False,
                            "score": 0.1,
                            "coverage": 1.0,
                        }
                    ],
                },
                graph_path,
            )
            output = generate_report(cache_path, graph_path, root, root / "sweep.html")
            report = output.read_text(encoding="utf-8")
            all_output = generate_report(
                cache_path, graph_path, root, root / "all.html", scope="all"
            )
            all_report = all_output.read_text(encoding="utf-8")

        self.assertIn("A + B · same label, different spurious data", report)
        self.assertIn("D + F · same spurious data, different labels", report)
        self.assertIn("K + N · high cosine", report)
        self.assertIn("K + L · low cosine", report)
        self.assertIn("X + Y · high cosine", report)
        self.assertIn("X + Z · low cosine", report)
        self.assertEqual(report.count('<article class="sub-sweep">'), 8)
        self.assertIn("data:image/jpeg;base64,", report)
        self.assertIn("gain(G)", report)
        self.assertIn("selected direction", report)
        self.assertNotIn("rejected direction", report)
        self.assertIn("rejected direction", all_report)
        self.assertIn("Σ individual gains", report)
        self.assertIn("Joint removal gain", report)


if __name__ == "__main__":
    unittest.main()
