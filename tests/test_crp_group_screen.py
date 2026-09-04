from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.tools.render_crp_group_screen import render_group_screen
from splice.crp import CrpAuditConfig
from splice.crp_group_screen import (
    GROUP_SCREEN_ARTIFACT,
    MiniInterventionConfig,
    ReconstructionScreenConfig,
    _rank_spaced_group_ids,
    build_group_screen,
)


class CrpGroupScreenTests(unittest.TestCase):
    def test_rank_spaced_group_sample_includes_head_middle_and_tail(self):
        self.assertEqual(_rank_spaced_group_ids([4, 9, 2, 7, 11], 3), [4, 2, 11])

    @staticmethod
    def _cache(sample_count: int = 12) -> dict:
        generator = torch.Generator().manual_seed(11)
        dictionary = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.99, 0.01, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
            dim=1,
        )
        codes = torch.zeros(sample_count, 4)
        for index in range(sample_count):
            codes[index, index % 2] = 1.0
            codes[index, 2 + index % 2] = 0.35
        return {
            "cache_version": 1,
            "provenance": {
                "dataset": "waterbirds",
                "split": "train",
                "splice_model": "fixture",
                "splice_pretrained": "fixture",
                "splice_vocab": "fixture",
                "splice_vocab_size": 4,
                "splice_l1_penalty": 0.25,
            },
            "sample_ids": [f"waterbirds:{index}" for index in range(sample_count)],
            "clip_embeddings": F.normalize(
                torch.randn(sample_count, 4, generator=generator), dim=1
            ),
            "image_mean": torch.zeros(4),
            "splice_codes": codes,
            "dictionary": dictionary,
            "vocabulary": ["tree sparrow", "field sparrow", "water", "grass"],
        }

    def test_group_screen_collapses_semantic_variants_and_measures_fidelity(self):
        report = build_group_screen(
            self._cache(),
            CrpAuditConfig(
                min_concept_frequency=0.0,
                max_concept_frequency=1.0,
                text_similarity_threshold=0.95,
                coactivation_threshold=0.0,
                min_group_size=1,
            ),
            ReconstructionScreenConfig(
                fidelity_threshold=0.80,
                target_image_coverage=0.90,
            ),
            MiniInterventionConfig(enabled=False),
        )
        sparrow_groups = [
            group for group in report["groups"]
            if set(group["concepts"]) == {"tree sparrow", "field sparrow"}
        ]
        self.assertEqual(report["artifact"], GROUP_SCREEN_ARTIFACT)
        self.assertEqual(len(sparrow_groups), 1)
        self.assertGreaterEqual(report["metrics"]["source_coverage"], 0.90)
        self.assertEqual(report["decision"]["status"], "REVIEW_GROUPS")
        self.assertIsNone(report["mini_intervention"])

    def test_mini_audit_is_sampled_and_does_not_build_a_teacher_graph(self):
        report = build_group_screen(
            self._cache(),
            CrpAuditConfig(
                min_concept_frequency=0.0,
                max_concept_frequency=1.0,
                text_similarity_threshold=0.95,
                coactivation_threshold=0.0,
                min_group_size=1,
                use_residual_splice_gate=False,
            ),
            ReconstructionScreenConfig(
                fidelity_threshold=0.70,
                target_image_coverage=0.50,
            ),
            MiniInterventionConfig(
                enabled=True,
                sample_count=8,
                max_groups=2,
                projected_neighbors=2,
                null_trials=1,
                null_quantile=0.0,
                activation_difference_quantile=0.5,
                min_intervention_gain=0.0,
                min_coverage=0.0,
                example_edges_per_group=1,
            ),
        )
        mini = report["mini_intervention"]
        self.assertEqual(mini["sample_count"], 8)
        self.assertLessEqual(mini["audited_group_count"], 2)
        self.assertEqual(
            mini["device"], "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.assertNotIn("neighbor_indices", report)
        self.assertTrue(all("mean_jaccard_at_k" in group for group in mini["groups"]))
        selected = sorted(
            (group for group in report["groups"] if group["selected_for_reconstruction"]),
            key=lambda group: group["activation_rank"],
        )
        self.assertEqual(
            [group["group_id"] for group in mini["groups"]],
            list(
                dict.fromkeys(
                    [selected[0]["group_id"], selected[-1]["group_id"]]
                )
            ),
        )
        self.assertTrue(
            all("neighbor_triplets" in group for group in mini["groups"])
        )

    def test_html_contains_reconstruction_bands_groups_and_posthoc_classes(self):
        report = build_group_screen(
            self._cache(),
            CrpAuditConfig(
                min_concept_frequency=0.0,
                max_concept_frequency=1.0,
                text_similarity_threshold=0.95,
                coactivation_threshold=0.0,
                min_group_size=1,
            ),
            ReconstructionScreenConfig(
                fidelity_threshold=0.80,
                target_image_coverage=0.90,
                top_images_per_group=2,
            ),
            MiniInterventionConfig(
                enabled=True,
                sample_count=8,
                max_groups=2,
                projected_neighbors=2,
                null_trials=1,
                null_quantile=0.0,
                activation_difference_quantile=0.5,
                min_intervention_gain=0.0,
                min_coverage=0.0,
                example_edges_per_group=1,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(12):
                filename = f"image_{index}.jpg"
                rows.append(
                    {
                        "img_filename": filename,
                        "y": index % 2,
                        "place": (index // 2) % 2,
                        "split": 0,
                    }
                )
                Image.new("RGB", (30, 22), color=(20 * index, 80, 150)).save(root / filename)
            pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
            output = render_group_screen(
                report, root, root / "screen.html", max_groups=3, images_per_band=2
            )
            document = output.read_text(encoding="utf-8")

        self.assertIn("Poor reconstruction", document)
        self.assertIn("Median reconstruction", document)
        self.assertIn("Good reconstruction", document)
        self.assertIn("Class and context slices", document)
        self.assertIn("tree sparrow", document)
        self.assertIn("data:image/jpeg;base64,", document)
        self.assertIn("Post-hoc only", document)
        self.assertIn("Mini groups", document)
        self.assertIn("Raw vs projected nearest neighbours", document)


if __name__ == "__main__":
    unittest.main()
