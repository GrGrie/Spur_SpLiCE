"""Concept learning curves and the early concept swap: units, reliance, flagging, the swap and both CLIs."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cospro.cli import summarize_concept_curves, train_concept_curves
from cospro.curves.core import (
    auroc,
    build_units,
    concept_shares,
    describe_units,
    detection_auroc,
    flag_units,
    hard_regions,
    margins,
    onset_scores,
    reliance_audit,
    spearman,
    unit_presence,
    unit_typing,
)
from cospro.curves.swap import flagged_masks, swap_flagged


def left_right_maps(images: int) -> torch.Tensor:
    """Two factors on a 14 x 14 grid: factor 0 owns the left half, factor 1 the right half."""

    maps = torch.zeros(images, 2, 14, 14)
    maps[:, 0, :, :7] = 1.0
    maps[:, 1, :, 7:] = 1.0
    return maps


class LeftMean(torch.nn.Module):
    """Logits that read only the mean of the left half of the image."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        value = images[:, :, :, : images.shape[-1] // 2].mean(dim=(1, 2, 3))
        return torch.stack([-value, value], dim=1)


class UnitTests(unittest.TestCase):
    def test_regions_partition_every_image(self):
        maps = torch.rand(5, 6, 14, 14)
        units = build_units(maps, 3)
        self.assertEqual(sorted(set(units.tolist())), list(range(int(units.max()) + 1)))
        regions = hard_regions(maps, units)
        self.assertTrue(torch.equal(regions.sum(dim=1), torch.ones(5, 14, 14, dtype=torch.long)))

    def test_zero_units_keeps_single_factors(self):
        self.assertEqual(build_units(torch.rand(2, 4, 14, 14), 0).tolist(), [0, 1, 2, 3])

    def test_descriptions_report_presence_and_area(self):
        maps = left_right_maps(4)
        maps[0, 1] = 0.0
        maps[0, 0] = 1.0  # image 0 shows only factor 0
        units = build_units(maps, 0)
        described = describe_units(maps, units, hard_regions(maps, units), ["left", "right"])
        self.assertEqual(described[0]["presence"], 1.0)
        self.assertEqual(described[1]["presence"], 0.75)
        self.assertAlmostEqual(described[1]["area"], 0.5, places=3)
        self.assertEqual(described[0]["name"], "left")


class RelianceTests(unittest.TestCase):
    def test_margin_is_correct_logit_minus_best_other(self):
        logits = torch.tensor([[2.0, 0.5, 1.0], [0.0, 3.0, 1.0]])
        self.assertEqual(margins(logits, torch.tensor([0, 2])).tolist(), [1.0, -2.0])

    def test_reliance_finds_the_region_the_model_reads(self):
        labels = torch.tensor([0, 1] * 4)
        images = torch.zeros(8, 3, 56, 56)
        images[:, :, :, :28] = (2.0 * labels.float() - 1.0)[:, None, None, None]  # the left half carries the class
        maps = left_right_maps(8)
        regions = hard_regions(maps, build_units(maps, 0))
        loader = [(images, torch.arange(8), labels)]
        result = reliance_audit(LeftMean(), loader, regions, [0, 1], torch.device("cpu"))
        self.assertGreater(result["reliance"][0], 0.5)
        self.assertAlmostEqual(result["reliance"][1], 0.0, places=2)
        self.assertEqual(result["images"], [8.0, 8.0])
        shares = concept_shares(result["reliance"], [0, 1])
        self.assertAlmostEqual(shares[0], 1.0, places=2)

    def test_shares_ignore_negative_reliance_and_unaudited_units(self):
        self.assertEqual(concept_shares([1.0, -2.0, 3.0, 4.0], [0, 1, 2]), [0.25, 0.0, 0.75, 0.0])
        self.assertEqual(concept_shares([-1.0, -2.0], [0, 1]), [0.0, 0.0])

    def test_onset_averages_the_early_window_only(self):
        onset = onset_scores([0, 1, 2, 10], [[0.5, 0.5], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], early_epochs=2)
        self.assertEqual(onset, [0.75, 0.25])


class FlaggingTests(unittest.TestCase):
    def setUp(self):
        self.units = [{"unit": unit, "audited": unit != 4, "presence": 1.0, "area": 0.1} for unit in range(5)]
        self.onset = [0.4, 0.3, 0.2, 0.1, 0.9]

    def test_top_takes_the_shortest_prefix_reaching_tau(self):
        self.assertEqual(flag_units(self.onset, self.units, "top", tau=0.5), [0, 1])
        self.assertEqual(flag_units(self.onset, self.units, "top", tau=0.3), [0])

    def test_one_audited_unit_always_stays_unflagged(self):
        self.assertEqual(flag_units(self.onset, self.units, "top", tau=1.0), [0, 1, 2])

    def test_area_cap_ends_the_prefix(self):
        units = [dict(unit, area=0.5) for unit in self.units]
        self.assertEqual(flag_units(self.onset, units, "top", tau=0.9, max_area=0.75), [0])

    def test_controls_flag_as_many_units_as_top(self):
        self.assertEqual(flag_units(self.onset, self.units, "bottom", tau=0.5), [2, 3])
        drawn = flag_units(self.onset, self.units, "random", tau=0.5, seed=3)
        self.assertEqual(len(drawn), 2)
        self.assertNotIn(4, drawn)
        self.assertEqual(drawn, flag_units(self.onset, self.units, "random", tau=0.5, seed=3))

    def test_explicit_units(self):
        self.assertEqual(flag_units(self.onset, self.units, "units", explicit=[3, 1, 3]), [1, 3])
        with self.assertRaises(ValueError):
            flag_units(self.onset, self.units, "units")


class DiagnosticTests(unittest.TestCase):
    def test_typing_separates_attribute_and_class_units(self):
        generator = torch.Generator().manual_seed(0)
        labels = torch.randint(2, (4000,), generator=generator)
        attributes = torch.where(torch.rand(4000, generator=generator) < 0.8, labels, 1 - labels)
        noise = torch.rand(4000, 3, generator=generator) < 0.05
        present = torch.stack([attributes.bool(), labels.bool(), torch.rand(4000, generator=generator) < 0.5], dim=1)
        present = present ^ noise
        typing = unit_typing(present, labels, attributes)
        self.assertEqual([entry["type"] for entry in typing], ["attribute", "class", "untyped"])
        units = [{"unit": unit, "audited": True} for unit in range(3)]
        self.assertEqual(detection_auroc([0.6, 0.3, 0.1], units, typing), 1.0)
        self.assertEqual(detection_auroc([0.2, 0.7, 0.1], units, typing), 0.0)

    def test_auroc_and_spearman(self):
        self.assertEqual(auroc([3.0, 1.0, 2.0, 0.0], [True, True, False, False]), 0.75)
        self.assertTrue(math.isnan(auroc([1.0, 2.0], [True, True])))
        self.assertAlmostEqual(spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]), 1.0)
        self.assertAlmostEqual(spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]), -1.0)


class SwapTests(unittest.TestCase):
    def setUp(self):
        self.views = torch.arange(6, dtype=torch.float32)[:, None, None, None].expand(6, 3, 56, 56).clone()
        regions = hard_regions(left_right_maps(6), torch.tensor([0, 1]))
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]]).repeat(6, 1)
        self.masks = flagged_masks(regions, [0], boxes, (56, 56))

    def test_masks_cover_the_flagged_half(self):
        self.assertAlmostEqual(float(self.masks[:, :, :, :24].mean()), 1.0, places=4)
        self.assertAlmostEqual(float(self.masks[:, :, :, 32:].mean()), 0.0, places=4)

    def test_swap_keeps_the_unflagged_part_and_takes_donor_content(self):
        swapped, changed = swap_flagged(self.views, self.masks, 1.0, generator=torch.Generator().manual_seed(0))
        self.assertTrue(bool(changed.all()))
        for image in range(6):
            self.assertAlmostEqual(float(swapped[image, :, :, 40:].mean()), float(image), places=4)
            donor_value = float(swapped[image, :, :, :16].mean())
            self.assertNotAlmostEqual(donor_value, float(image), places=3)
            self.assertAlmostEqual(donor_value, round(donor_value), places=3)

    def test_probability_zero_changes_nothing(self):
        swapped, changed = swap_flagged(self.views, self.masks, 0.0)
        self.assertTrue(torch.equal(swapped, self.views))
        self.assertFalse(bool(changed.any()))

    def test_images_without_flagged_content_stay(self):
        masks = self.masks.clone()
        masks[0] = 0.0
        _, changed = swap_flagged(self.views, masks, 1.0, generator=torch.Generator().manual_seed(0))
        self.assertFalse(bool(changed[0]))
        self.assertTrue(bool(changed[1:].all()))


def write_synthetic_metashift(root: Path, per_group: int = 6) -> None:
    """A MetaShift tree of 64-pixel images: the class is a bright or dark centre, the context the border colour."""

    generator = np.random.RandomState(0)
    for split in ("train", "test"):
        for target, name in enumerate(("cat", "dog")):
            for context, context_name in enumerate(("indoor", "outdoor")):
                directory = root / "metashift" / split / name / f"{name}({context_name})"
                directory.mkdir(parents=True)
                aligned = target == context
                count = per_group * (3 if aligned and split == "train" else 1)
                for index in range(count):
                    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
                    pixels[..., 1 if context == 0 else 2] = 200
                    pixels[20:44, 20:44] = 230 if target == 0 else 30
                    pixels = np.clip(pixels + generator.randint(-10, 10, pixels.shape), 0, 255).astype(np.uint8)
                    Image.fromarray(pixels).save(directory / f"{index:03d}.jpg")


class EndToEndTests(unittest.TestCase):
    def test_erm_audit_then_swap_then_summary(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_synthetic_metashift(root / "data")
            from cospro.data.metashift import MetaShiftDataset

            dataset = MetaShiftDataset(str(root / "data"))
            train = dataset.get_subset("train").indices
            maps = torch.zeros(len(train), 2, 14, 14)
            maps[:, 1] = 1.0
            maps[:, 0, 4:10, 4:10] = 2.0  # factor "centre" wins the centre cells, "border" the rest
            maps = maps / maps.sum(dim=1, keepdim=True)
            maps_path = root / "concept_maps.pt"
            torch.save({"sample_ids": [f"metashift:{int(index)}" for index in train],
                        "factor_names": ["centre", "border"], "maps": maps.half(), "kind": "concepts"}, maps_path)
            common = ["--dataset", "metashift", "--data-folder", str(root / "data"), "--concept-maps", str(maps_path),
                      "--seed", "1", "--study", "curves_test", "--output-root", str(root / "outputs"),
                      "--model", "resnet18", "--epochs", "2", "--early-epochs", "2", "--eval-every", "1",
                      "--batch-size", "16", "--unit-count", "0", "--num-workers", "0", "--device", device,
                      "--learning-rate", "0.01"]
            erm = train_concept_curves.main([*common, "--mode", "erm", "--arm", "erm"])
            curves = json.loads((erm / "concept_curves.json").read_text(encoding="utf-8"))
            self.assertEqual(curves["epochs"], [0, 1, 2])
            self.assertEqual(len(curves["onset"]), 2)
            self.assertEqual([unit["name"] for unit in curves["units"]], ["centre", "border"])
            self.assertEqual([entry["type"] for entry in curves["evaluation_only"]["typing"]], ["untyped", "untyped"])
            run = json.loads((erm / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["status"], "complete")
            self.assertEqual(len(run["final"]["val_group_acc"]), 4)

            swap = train_concept_curves.main([*common, "--mode", "swap", "--arm", "swap_units", "--flag-rule", "units",
                                              "--flag-units", "1", "--swap-probability", "1.0"])
            flags = json.loads((swap / "flagged_units.json").read_text(encoding="utf-8"))
            self.assertEqual(flags["names"], ["border"])
            swapped = json.loads((swap / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(swapped["status"], "complete")
            self.assertGreater(swapped["training"][0]["swapped_fraction"], 0.5)

            summary = summarize_concept_curves.main(["--study", "curves_test", "--output-root", str(root / "outputs")])
            text = summary.read_text(encoding="utf-8")
            self.assertIn("| swap_units | 1 |", text)
            self.assertIn("## Audit of `erm`", text)


if __name__ == "__main__":
    unittest.main()
