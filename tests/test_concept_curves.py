"""Concept learning curves and the early concept swap: units, the audit, flagging, the swap and the method."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import spur_splice
from cospro.cli import summarize_concept_curves
from cospro.curves.core import (
    acquired,
    auroc,
    build_units,
    concept_shares,
    describe_units,
    detection_auroc,
    embedding_audit,
    flag_units,
    hard_regions,
    onset_scores,
    spearman,
    unit_typing,
)
from cospro.curves.swap import flagged_masks, swap_flagged
from cospro.data.base import build_loader
from cospro.data.metashift import MetaShiftDataset
from cospro.methods import LoaderContext, build_method
from cospro.methods.concept_curves import ConceptCurves
from cospro.models.simclr import SimCLRModel
from cospro.training.callbacks import ConceptAudit, EpochReport


def left_right_maps(images: int) -> torch.Tensor:
    """Two factors on a 14 x 14 grid: factor 0 owns the left half, factor 1 the right half."""

    maps = torch.zeros(images, 2, 14, 14)
    maps[:, 0, :, :7] = 1.0
    maps[:, 1, :, 7:] = 1.0
    return maps


class LeftReader(torch.nn.Module):
    """An encoder and head that read only the left half of the image: two features per image."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Identity()

    def encoder(self, images: torch.Tensor) -> torch.Tensor:
        left = images[:, :, :, : images.shape[-1] // 2]
        return torch.stack([left[:, :, : left.shape[2] // 2].mean(dim=(1, 2, 3)),
                            left[:, :, left.shape[2] // 2:].mean(dim=(1, 2, 3))], dim=1)


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


class AuditTests(unittest.TestCase):
    def test_audit_finds_the_region_the_encoder_reads(self):
        generator = torch.Generator().manual_seed(0)
        images = torch.zeros(8, 3, 56, 56)
        images[:, :, :28, :28] = torch.randn(8, 1, 1, 1, generator=generator)  # the left half carries the content
        images[:, :, 28:, :28] = torch.randn(8, 1, 1, 1, generator=generator)
        images[:, :, :, 28:] = torch.randn(8, 1, 1, 1, generator=generator)
        maps = left_right_maps(8)
        regions = hard_regions(maps, build_units(maps, 0))
        result = embedding_audit(LeftReader(), [(images, torch.arange(8))], regions, [0, 1], torch.device("cpu"))
        self.assertGreater(result["shift"][0], 10 * abs(result["shift"][1]) + 0.05)
        self.assertGreater(result["margin"][0], 10 * abs(result["margin"][1]) + 0.05)
        self.assertEqual(result["images"], [8.0, 8.0])

    def test_acquired_subtracts_the_initial_value(self):
        self.assertEqual(acquired([0.5, 0.25], [0.25, 0.5]), [0.25, -0.25])

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


def synthetic_maps(root: Path, dataset: MetaShiftDataset) -> Path:
    """Maps with two factors for the synthetic tree: "centre" wins the centre cells and "border" the rest."""

    train = dataset.get_subset("train").indices
    maps = torch.ones(len(train), 2, 14, 14)
    maps[:, 0] = 0.0
    maps[:, 0, 4:10, 4:10] = 2.0
    maps = maps / maps.sum(dim=1, keepdim=True)
    path = root / "concept_maps.pt"
    torch.save({"sample_ids": [f"metashift:{int(index)}" for index in train], "factor_names": ["centre", "border"],
                "maps": maps.half(), "kind": "concepts"}, path)
    return path


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        write_synthetic_metashift(self.root / "data")
        self.dataset = MetaShiftDataset(str(self.root / "data"))
        self.maps = synthetic_maps(self.root, self.dataset)

    def tearDown(self):
        self.folder.cleanup()

    def build(self, **options):
        method = ConceptCurves(concept_maps=str(self.maps), seed=1, unit_count=0, early_epochs=2, audit_every=1,
                               late_audit_every=0, total_epochs=4, swap_warmup_epochs=0, image_size=64, **options)
        loader = build_loader(MetaShiftDataset, "ssl", MetaShiftDataset.Config(root_dir=str(self.root / "data"), image_size=64),
                              16, dataset=self.dataset, num_workers=0, generator=torch.Generator().manual_seed(1))
        wrapped = method.wrap_loader(loader, LoaderContext(dataset="metashift", batch_size=16, num_workers=0, seed=1))
        return method, loader, wrapped

    def test_loader_keeps_the_simclr_batch_order_and_adds_boxes(self):
        method, loader, wrapped = self.build()
        self.assertIs(wrapped.sampler, loader.sampler)
        views, rows, boxes = next(iter(wrapped))
        self.assertEqual(tuple(views[0].shape), (16, 3, 64, 64))
        self.assertEqual(tuple(boxes.shape), (16, 2, 5))
        self.assertEqual([unit["name"] for unit in method.units], ["centre", "border"])

    def test_audit_flags_and_swap_follow_the_schedule(self):
        torch.manual_seed(0)
        method, _, wrapped = self.build(flag_rule="units", flag_units_list="1", swap_probability=1.0)
        model = SimCLRModel(name="resnet18")
        batch = next(iter(wrapped))
        self.assertEqual([epoch for epoch in range(5) if method.audit_due(epoch)], [0, 1, 2, 4])
        for epoch in (0, 1, 2):
            method.set_epoch(epoch)
            before = [view.clone() for view in batch[0]]
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(method.compose_views(list(batch[0]), batch), before)))
            method.audit(model, torch.device("cpu"), False, epoch)
        self.assertEqual(method.curves["epochs"], [0, 1, 2])
        self.assertEqual(len(method.curves["onset"]), 2)
        self.assertEqual(method.curves["share"][0], [0.0, 0.0])  # nothing is acquired at initialization
        labels = self.dataset.get_subset("train").metadata_array
        method.score_audit(labels[:, 1], labels[:, 0])
        self.assertEqual(len(method.curves["evaluation_only"]["typing"]), 2)
        flags = method.decide_flags()
        self.assertEqual(flags["names"], ["border"])
        method.set_epoch(3)
        views = method.compose_views([view.clone() for view in batch[0]], batch)
        self.assertGreater(method.diagnostics()["curve_swapped_fraction"], 0.3)
        self.assertFalse(torch.equal(views[0], batch[0][0]))
        self.assertFalse(torch.equal(views[1], batch[0][1]))

    def test_rule_none_never_swaps(self):
        method, _, wrapped = self.build(flag_rule="none")
        batch = next(iter(wrapped))
        model = SimCLRModel(name="resnet18")
        for epoch in (0, 1, 2):
            method.audit(model, torch.device("cpu"), False, epoch)
        self.assertIsNone(method.decide_flags())
        method.set_epoch(3)
        self.assertEqual(method.scheduled_probability, 0.0)
        self.assertTrue(torch.equal(method.compose_views(list(batch[0]), batch)[0], batch[0][0]))

    def test_callback_writes_curves_and_flags_and_the_summary_reads_them(self):
        method, _, _ = self.build(flag_rule="top")
        run_folder = self.root / "outputs" / "seeds" / "study" / "seed_01" / "curves" / "attempt" / "training" / "run"
        args = SimpleNamespace(save_folder=str(run_folder), device="cpu", amp=False, resume="", dataset="metashift",
                               data_folder=str(self.root / "data"))
        state = SimpleNamespace(method=method, model=SimCLRModel(name="resnet18"))
        callback = ConceptAudit(state, args)
        callback.on_train_start()
        for epoch in (1, 2):
            callback.on_epoch_end(EpochReport(epoch=epoch, train={}, learning_rate=0.1))
        curves = json.loads((run_folder / "concept_curves.json").read_text(encoding="utf-8"))
        self.assertEqual(curves["epochs"], [0, 1, 2])
        self.assertIn("typing", curves["evaluation_only"])
        # The untrained encoder acquires no reliance, so nothing is flagged and the run would continue as SimCLR.
        flags = json.loads((run_folder / "flagged_units.json").read_text(encoding="utf-8"))
        self.assertEqual(flags["flagged"], [])
        self.assertEqual(method.scheduled_probability, 0.0)
        summary = summarize_concept_curves.main(["--study", "study", "--output-root", str(self.root / "outputs")])
        self.assertIn("## curves", summary.read_text(encoding="utf-8"))

    def test_attribution_pairs_single_unit_arms_with_simclr_by_seed(self):
        def audit(onset):
            units = [{"unit": unit, "name": name, "presence": 0.5, "area": 0.3, "audited": unit < 3}
                     for unit, name in enumerate(["grass", "sofa", "dog", "rare"])]
            typing = [{"unit": unit, "type": kind} for unit, kind in enumerate(["attribute", "attribute", "class", "untyped"])]
            return {"curves": {"onset": onset, "units": units, "evaluation_only": {"typing": typing}}, "flags": None}

        onset = [0.5, 0.3, 0.2, 0.0]
        audits = {f"curves_unit_{unit}": {1: audit(onset), 2: audit(onset)} for unit in range(4)}
        results = {"simclr": {1: {"wga": 40.0, "acc": 80.0}, 2: {"wga": 44.0, "acc": 80.0}, 3: {"wga": 90.0, "acc": 80.0}}}
        for unit, gain in enumerate([6.0, 2.0, -4.0, 0.0]):
            results[f"curves_unit_{unit}"] = {1: {"wga": 40.0 + gain, "acc": 81.0}, 2: {"wga": 44.0 + gain, "acc": 81.0}}
        result = summarize_concept_curves.attribution(audits, results)
        self.assertEqual(result["reference_arm"], "simclr")
        self.assertEqual([row["name"] for row in result["units"]], ["grass", "sofa", "dog"])  # the unaudited unit is left out
        self.assertEqual([row["wga_effect"] for row in result["units"]], [6.0, 2.0, -4.0])
        self.assertEqual(result["units"][0]["seeds"], [1, 2])
        self.assertEqual(result["units"][2]["positive_seeds"], 0)
        self.assertAlmostEqual(result["onset_effect_spearman"], 1.0)
        self.assertIn("| 1 | 0 | grass |", "\n".join(summarize_concept_curves.render_attribution(result)))
        self.assertIsNone(summarize_concept_curves.attribution(audits, {"simclr": results["simclr"]}))

    def test_configuration_selects_the_method_and_leaves_other_storage_names_alone(self):
        common = ["--dataset", "metashift", "--seed", "1", "--artifact_dir", str(self.root / "runs")]
        args = spur_splice.parse_args([*common, "--splice_mode", "concept_curves", "--curve_concept_maps", str(self.maps)])
        method = build_method(spur_splice.training_config(args), graph_fingerprint=None)
        self.assertIsInstance(method, ConceptCurves)
        self.assertIn("concept-curves", args.storage_name)
        plain = spur_splice.parse_args([*common, "--splice_mode", "none"])
        other = spur_splice.parse_args([*common, "--splice_mode", "none", "--curve_early_epochs", "7"])
        self.assertEqual(plain.storage_name, other.storage_name)


if __name__ == "__main__":
    unittest.main()
