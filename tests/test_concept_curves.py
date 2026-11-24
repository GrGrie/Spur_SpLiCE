"""Concept learning curves and the early concept swap: units, the audit, flagging, the swap and the method."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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

    def test_ecs_kill_test_verdicts(self):
        def audit(auroc, flagged, leader_onset=0.6):
            units = [{"unit": unit, "name": name, "presence": 0.5, "area": 0.3, "audited": True}
                     for unit, name in enumerate(["grass", "sofa", "dog"])]
            typing = [{"unit": 0, "type": "attribute"}, {"unit": 1, "type": "attribute"}, {"unit": 2, "type": "class"}]
            share = [[0.0, 0.0, 0.0], [0.6, 0.2, 0.2], [0.2, 0.3, 0.5]]
            curves = {"onset": [leader_onset, 0.3, 0.1], "units": units, "epochs": [0, 50, 500], "early_epochs": 50,
                      "share": share, "evaluation_only": {"typing": typing, "detection_auroc": auroc}}
            return {"curves": curves, "flags": {"flagged": flagged}}

        audits = {"curves": {seed: audit(0.9, [0]) for seed in (1, 2, 3, 4)}}
        audits["curves"][4] = audit(0.6, [0, 2])
        k1 = summarize_concept_curves.k1_verdict(audits["curves"])
        self.assertEqual((k1["verdict"], k1["seeds_auroc_at_least_0.7"], k1["seeds_with_clean_flags"]), ("pass", 3, 3))
        self.assertEqual(summarize_concept_curves.k1_verdict({1: audit(0.4, [0])})["verdict"], "fail")
        self.assertEqual(summarize_concept_curves.k1_verdict({1: audit(0.65, [0])})["verdict"], "weak")

        def wga(values):
            return {seed: {"wga": value, "acc": 80.0} for seed, value in zip((1, 2, 3, 4), values)}

        results = {"simclr": wga([40, 44, 46, 48]), "curves": wga([48, 50, 52, 50]), "curves_random": wga([42, 45, 47, 51]),
                   "curves_oracle": wga([50, 52, 54, 55])}
        self.assertEqual(summarize_concept_curves.k2_verdict(results)["verdict"], "pass")
        results["curves_random"] = wga([49, 51, 53, 51])
        self.assertEqual(summarize_concept_curves.k2_verdict(results)["verdict"], "partial")
        results["curves_oracle"] = wga([41, 44, 46, 48])
        self.assertEqual(summarize_concept_curves.k2_verdict(results)["verdict"], "fail")
        self.assertEqual(summarize_concept_curves.k2_verdict({"simclr": results["simclr"]})["verdict"], "pending")

        change = summarize_concept_curves.flagged_share_change(audits["curves"][1])
        self.assertEqual((change["at_early_end"], change["at_last_audit"], change["fell"]), (0.6, 0.2, True))
        result = summarize_concept_curves.ecs_kill_test(audits, results)
        self.assertEqual(result["flagged_share_after_swap"]["curves"]["fell"], 4)
        text = "\n".join(summarize_concept_curves.render_kill_test(result))
        self.assertIn("Stage K1 (audit finds the spurious concept): pass", text)
        self.assertIn("| curves_random | 4 |", text)

    def test_every_curve_arm_of_the_launcher_parses(self):
        launcher = (Path(spur_splice.__file__).parent / "scripts" / "submit_factor_study.sh").read_text(encoding="utf-8")
        arms = dict(re.findall(r'ARMS\[(curves[a-z0-9_]*)\]="\$\{CURVES\[\*\]\}([^"]*)"', launcher))
        self.assertIn("curves_oracle_t1", arms)
        self.assertGreaterEqual(len(arms), 17)
        for arm, extra in arms.items():
            extra = extra.replace("${CURVE_UNITS}", "12").replace("${unit}", "3")
            args = spur_splice.parse_args(["--dataset", "metashift", "--seed", "1", "--artifact_dir", str(self.root / "runs"),
                                           "--splice_mode", "concept_curves", "--curve_concept_maps", str(self.maps),
                                           *extra.split()])
            self.assertEqual(args.splice_mode, "concept_curves", arm)

    def test_area_normalized_shares_divide_reliance_by_area(self):
        method, _, _ = self.build(flag_rule="none", area_normalize=True)
        self.assertTrue(method.curves["area_normalize"])
        method.curves.update(epochs=[0], shift=[[0.0, 0.0]], margin=[[0.0, 0.0]], share=[[0.0, 0.0]])
        result = {"shift": [0.1, 0.1], "margin": [0.0, 0.0], "images": [48, 48]}
        with mock.patch("cospro.methods.concept_curves.embedding_audit", return_value=result):
            method.audit(None, torch.device("cpu"), False, 1)
        areas = [unit["area"] for unit in method.units]
        expected = [0.1 / areas[0], 0.1 / areas[1]]
        self.assertAlmostEqual(method.curves["share"][-1][0], expected[0] / sum(expected), places=6)

    def test_audit_only_run_reproduces_simclr_and_the_swap_starts_after_the_window(self):
        """The control arm curves_audit must equal simclr loss for loss; a swapping run departs only after T0."""

        common = ["--dataset", "metashift", "--data_folder", str(self.root / "data"), "--model", "resnet18_large",
                  "--epochs", "3", "--batch_size", "16", "--num_workers", "0", "--device", "cpu", "--learning_rate",
                  "0.01", "--temp", "0.5", "--seed", "1", "--save_freq", "50"]
        curves = ["--splice_mode", "concept_curves", "--curve_concept_maps", str(self.maps), "--curve_unit_count", "0",
                  "--curve_early_epochs", "2", "--curve_audit_every", "1"]
        arms = {"simclr": ["--splice_mode", "none"], "audit": [*curves, "--curve_flag_rule", "none"],
                "swap": [*curves, "--curve_flag_rule", "units", "--curve_flag_units", "1",
                         "--curve_swap_warmup_epochs", "0", "--curve_swap_probability", "1.0"]}
        losses = {}
        for arm, extra in arms.items():
            folder = self.root / "runs" / arm
            done = subprocess.run(
                [sys.executable, str(Path(spur_splice.__file__)), *common, *extra, "--artifact_dir",
                 str(folder / "training"), "--run_record", str(folder / "run.json")],
                cwd=Path(spur_splice.__file__).parent, capture_output=True, text=True,
                env={**os.environ, "WANDB_MODE": "disabled"},
            )
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            losses[arm] = [event["values"]["SSL SimCLR loss"] for event in record["metrics"]
                           if event.get("stage") == "ssl" and "SSL SimCLR loss" in event["values"]]
        self.assertEqual(len(losses["simclr"]), 3)
        self.assertEqual(losses["audit"], losses["simclr"])
        self.assertEqual(losses["swap"][:2], losses["simclr"][:2])
        self.assertNotEqual(losses["swap"][2], losses["simclr"][2])

    def test_configuration_selects_the_method_and_leaves_other_storage_names_alone(self):
        common = ["--dataset", "metashift", "--seed", "1", "--artifact_dir", str(self.root / "runs")]
        args = spur_splice.parse_args([*common, "--splice_mode", "concept_curves", "--curve_concept_maps", str(self.maps)])
        method = build_method(spur_splice.training_config(args), graph_fingerprint=None)
        self.assertIsInstance(method, ConceptCurves)
        self.assertIn("concept-curves", args.storage_name)
        plain = spur_splice.parse_args([*common, "--splice_mode", "none"])
        other = spur_splice.parse_args([*common, "--splice_mode", "none", "--curve_early_epochs", "7"])
        self.assertEqual(plain.storage_name, other.storage_name)

    def test_csa_audit_arm_of_the_launcher_parses_and_keeps_the_encoder(self):
        launcher = (Path(spur_splice.__file__).parent / "scripts" / "submit_factor_study.sh").read_text(encoding="utf-8")
        extra = re.search(r'ARMS\[csa_audit\]="\$\{CURVES\[\*\]\}([^"]*)"', launcher).group(1).replace("${CURVE_UNITS}", "12")
        args = spur_splice.parse_args(["--dataset", "metashift", "--seed", "1", "--artifact_dir", str(self.root / "runs"),
                                       "--splice_mode", "concept_curves", "--curve_concept_maps", str(self.maps),
                                       *extra.split()])
        self.assertEqual(args.curve_flag_rule, "none")
        self.assertTrue(args.keep_checkpoints)

    def test_attribution_at_inference_end_to_end(self):
        """Train an audit-only run that keeps its encoder, attribute its concepts at inference and summarize."""

        from cospro.cli import attribute_concepts
        from cospro.cli.build_concept_maps import split_maps_path

        maps_folder = self.root / "maps" / "synthetic"
        maps_folder.mkdir(parents=True)
        training_maps = maps_folder / "concept_maps.pt"
        training_maps.write_bytes(self.maps.read_bytes())
        validation = self.dataset.get_subset("val").indices
        maps = torch.ones(len(validation), 2, 14, 14)
        maps[:, 0] = 0.0
        maps[:, 0, 4:10, 4:10] = 2.0
        maps = maps / maps.sum(dim=1, keepdim=True)
        validation_maps = split_maps_path(training_maps, "val")
        validation_maps.parent.mkdir(parents=True)
        torch.save({"sample_ids": [f"metashift:{int(index)}" for index in validation], "factor_names": ["centre", "border"],
                    "maps": maps.half(), "kind": "concepts"}, validation_maps)
        outputs = self.root / "outputs"
        attempt = outputs / "seeds" / "study" / "seed_01" / "csa_audit" / "attempt"
        environment = {**os.environ, "WANDB_MODE": "disabled", "SPUR_SPLICE_SCRATCH_ROOT": str(self.root / "scratch")}
        done = subprocess.run(
            [sys.executable, str(Path(spur_splice.__file__)), "--dataset", "metashift", "--data_folder",
             str(self.root / "data"), "--model", "resnet18_large", "--epochs", "2", "--batch_size", "16", "--num_workers",
             "0", "--device", "cpu", "--learning_rate", "0.01", "--temp", "0.5", "--seed", "1", "--study", "study",
             "--arm", "csa_audit", "--attempt_id", "attempt", "--splice_mode", "concept_curves", "--curve_concept_maps",
             str(training_maps), "--curve_unit_count", "0", "--curve_flag_rule", "none", "--curve_early_epochs", "1",
             "--curve_audit_every", "1", "--keep_checkpoints", "--save_freq", "500",
             "--artifact_dir", str(attempt / "training"), "--run_record", str(attempt / "run.json")],
            cwd=Path(spur_splice.__file__).parent, capture_output=True, text=True, env=environment,
        )
        self.assertEqual(done.returncode, 0, done.stderr[-3000:])
        with mock.patch.dict(os.environ, {"SPUR_SPLICE_SCRATCH_ROOT": str(self.root / "scratch")}):
            output = attribute_concepts.main(["--study", "study", "--arm", "csa_audit", "--seed", "1", "--output-root",
                                              str(outputs), "--data_folder", str(self.root / "data"), "--device", "cpu",
                                              "--num_workers", "0", "--batch_size", "16"])
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(output, attempt / "concept_attribution_val.json")
        self.assertEqual([unit["name"] for unit in record["units"]], ["centre", "border"])
        self.assertEqual(record["units"][0]["images"], len(validation))
        self.assertLess(record["probe"]["clean_feature_max_difference"], 1e-3)
        self.assertEqual(len(record["clean"]["group_acc"]), 4)
        for unit in record["units"]:
            self.assertGreater(unit["embedding_shift"], 0.0)
            self.assertAlmostEqual(unit["net_change"]["wga"], unit["occluded"]["wga"] - unit["control"]["wga"])
        summary = summarize_concept_curves.main(["--study", "study", "--output-root", str(outputs)])
        self.assertIn("## Concept attribution at inference (`csa_audit`, 1 seeds)", summary.read_text(encoding="utf-8"))


class InferenceAttributionTests(unittest.TestCase):
    def test_control_donors_borrow_from_another_image_that_shows_the_unit(self):
        from cospro.curves.attribution import control_donors

        present = torch.tensor([[True, False], [True, True], [False, True], [True, False]])
        donors = control_donors(present)
        for unit in range(2):
            shown = present[:, unit].nonzero().view(-1).tolist()
            self.assertEqual(sorted(donors[shown, unit].tolist()), shown)
            self.assertTrue(all(int(donors[row, unit]) != row for row in shown))
            self.assertTrue(all(int(donors[row, unit]) == -1 for row in range(4) if row not in shown))

    def test_occlusion_moves_only_the_features_of_images_that_show_the_unit(self):
        from cospro.curves.attribution import occlusion_features

        generator = torch.Generator().manual_seed(0)
        images = torch.zeros(6, 3, 56, 56)
        images[:, :, :28, :28] = torch.randn(6, 1, 1, 1, generator=generator)  # the left half carries the content
        images[:, :, 28:, :28] = torch.randn(6, 1, 1, 1, generator=generator)
        images[:, :, :, 28:] = torch.randn(6, 1, 1, 1, generator=generator)
        maps = left_right_maps(6)
        maps[0, 0] = 0.0  # image 0 shows no left unit
        maps[0, 1] = 1.0
        regions = hard_regions(maps, build_units(maps, 0))
        class Encoder(torch.nn.Module):
            def forward(self, batch):
                return LeftReader().encoder(batch)

        features = occlusion_features(Encoder(), [(images, torch.arange(6))], regions, [0, 1], torch.device("cpu"))
        clean = features["clean"]
        self.assertTrue(torch.equal(features["occluded"][0][0], clean[0]))
        left_shift = (features["occluded"][0][1:] - clean[1:]).abs().sum()
        # Image 0 is all "right" unit, so occluding it hides everything; the other images keep their left half.
        right_shift = (features["occluded"][1][1:] - clean[1:]).abs().sum()
        self.assertGreater(float(left_shift), 10 * float(right_shift) + 0.01)
        self.assertFalse(bool(features["present"][0, 0]))

    def test_verdicts_contrast_and_hypotheses(self):
        verdict = summarize_concept_curves.effect_verdict
        self.assertEqual(verdict([3.0, 2.5, 3.5, 2.0])["verdict"], "shortcut")
        self.assertEqual(verdict([-3.0, -2.5, -3.5, -2.0])["verdict"], "needed")
        self.assertEqual(verdict([5.0, -4.0, 6.0, -5.0])["verdict"], "neutral")
        results = {f"curves_unit_{unit}": {seed: {"wga": 40.0 + 10 * seed + offset + (seed % 2) * 0.5, "acc": 70.0}
                                          for seed in (1, 2, 3, 4)}
                   for unit, offset in enumerate([4.0, 0.0, -4.0])}
        contrast = summarize_concept_curves.unit_contrast(results, {unit: f"curves_unit_{unit}" for unit in range(3)})
        self.assertAlmostEqual(contrast["units"][0]["mean"], 4.0)
        self.assertEqual(contrast["units"][2]["verdict"], "needed")
        self.assertEqual(contrast["seeds"], [1, 2, 3, 4])
        rows = [{"type": "attribute", "verdict": "shortcut", "seeds": [1, 2, 3, 4], "reliance": {"holds": True}},
                {"type": "class", "verdict": "needed", "seeds": [1, 2, 3, 4], "reliance": {"holds": True}}]
        hypotheses = summarize_concept_curves.csa_hypotheses(rows, 0.8)
        self.assertEqual((hypotheses["h1"], hypotheses["h2"], hypotheses["h3"], hypotheses["outcome"]),
                         (True, True, True, "pass"))
        neutral = [{**row, "verdict": "neutral", "seeds": list(range(8))} for row in rows]
        self.assertEqual(summarize_concept_curves.csa_hypotheses(neutral, 0.1)["outcome"], "fail")

    def test_unit_share_falls_against_the_audit_only_arm(self):
        def entry(late):
            return {"curves": {"epochs": [0, 50, 500], "early_epochs": 50, "share": [[0.0, 0.0], [0.5, 0.5], late]}}

        swapped = {seed: entry([0.6, 0.1]) for seed in (1, 2)}
        plain = {seed: entry([0.4, 0.6]) for seed in (1, 2)}
        alone = summarize_concept_curves.unit_share_change(swapped, 1)
        self.assertEqual((alone["fell"], alone["against"]), (2, "early window"))
        against = summarize_concept_curves.unit_share_change(swapped, 0, plain)
        self.assertEqual((against["fell"], against["reference"]), (0, 0.4))

    def test_inference_summary_nets_the_control_over_seeds(self):
        def record(net):
            unit = {"unit": 3, "name": "lawn", "type": "attribute", "eval_presence": 0.3, "images": 10,
                    "embedding_shift": 0.2, "control_shift": 0.1,
                    "change": {"wga": net + 1.0, "acc": 0.0, "group_acc": [0.0] * 4},
                    "control_change": {"wga": 1.0, "acc": 0.0, "group_acc": [0.0] * 4},
                    "net_change": {"wga": net, "acc": 0.5, "group_acc": [net, 0.0, 0.0, 0.0]}}
            return {"group_names": [f"(target,context)=({g // 2},{g % 2})" for g in range(4)], "clean": {"wga": 40.0},
                    "units": [unit]}

        seeds = {seed: record(net) for seed, net in zip((1, 2, 3, 4), (3.0, 2.0, 4.0, 3.0))}
        result = summarize_concept_curves.summarize_inference("csa_audit", seeds, {}, None)
        self.assertAlmostEqual(result["units"][0]["net_wga_change"], 3.0)
        self.assertEqual(result["units"][0]["verdict"], "shortcut")
        self.assertIn("| 3 | lawn | attribute |", "\n".join(summarize_concept_curves.render_inference(result)))

    def test_split_maps_sit_beside_the_training_maps(self):
        from cospro.cli.build_concept_maps import maps_name, split_maps_path

        self.assertEqual(split_maps_path(Path("/s/maps/name/concept_maps.pt"), "val"),
                         Path("/s/maps/name_val/concept_maps.pt"))
        groups = Path("outputs/groups/laion_text/concept_groups.json")
        self.assertEqual(maps_name(groups, 0.02, 0, "qq", 448, split="val"), "laion_text_min0.02_merge0_qq448_val")
        self.assertEqual(maps_name(groups, 0.02, 0, "qq", 448), "laion_text_min0.02_merge0_qq448")

    def test_like_copies_groups_cache_and_settings_of_stored_maps(self):
        from cospro.cli.build_concept_maps import apply_like, parse_args

        with tempfile.TemporaryDirectory() as folder:
            stored = Path(folder) / "concept_maps.pt"
            config = {"factor_min_frequency": 0.02, "merge_similarity": 0.0, "attention": "qq", "resolution": 448,
                      "stride": 112, "temperature": 0.01, "targets": "concepts"}
            torch.save({"maps": torch.zeros(1, 2, 14, 14).half(), "config": config, "concept_groups": "g/groups.json",
                        "splice_cache": "c/cache.pt"}, stored)
            args = parse_args(["--dataset", "metashift", "--like", str(stored), "--split", "val"])
            apply_like(args)
            self.assertEqual((args.concept_groups, args.splice_cache), (Path("g/groups.json"), Path("c/cache.pt")))
            self.assertEqual((args.factor_min_frequency, args.resolution, args.split), (0.02, 448, "val"))
        with self.assertRaises(SystemExit):
            parse_args(["--dataset", "metashift"])


if __name__ == "__main__":
    unittest.main()
