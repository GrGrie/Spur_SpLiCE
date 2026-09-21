"""Typed training configuration: sections, presets and the flat projection."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import fields
from unittest.mock import patch

import spur_splice
from cospro.config import PRESETS, TRAINING_SECTIONS, TrainingConfig, training_defaults
from cospro.tracking.artifacts import PROJECT_ROOT

GRAPH = "tests/fixtures/storage_identity_graph.json"


def parse(argv: list[str]):
    with tempfile.TemporaryDirectory() as directory:
        cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            with patch.object(sys, "argv", ["spur_splice.py"]), patch("torch.cuda.is_available", return_value=False):
                args = spur_splice.parse_args([*argv, "--artifact_dir", directory])
        finally:
            os.chdir(cwd)
    values = vars(args).copy()
    values.pop("runtime_versions")
    values.pop("save_folder")
    values.pop("artifact_dir")
    values.pop("run_record")
    return args, values


class TrainingConfigTests(unittest.TestCase):
    def test_every_option_has_one_destination(self):
        names = Counter(item.name for section, _ in TRAINING_SECTIONS for item in fields(section))
        self.assertEqual([name for name, count in names.items() if count > 1], [])

    def test_typed_sections_project_back_to_the_namespace(self):
        args, _ = parse(["--dataset", "celeba", "--seed", "3"])
        flat = TrainingConfig.from_namespace(args).flat()
        self.assertEqual(flat, {key: getattr(args, key) for key in flat})
        self.assertEqual(spur_splice.training_config(args).data.dataset, "celeba")

    def test_preset_matches_the_same_values_spelled_out(self):
        _, with_preset = parse(["--preset", "cospro_student", "--cospro_teacher_graph", GRAPH])
        explicit = [
            "--batch_size", "128", "--num_workers", "4", "--temp", "0.05", "--splice_mode", "cospro_relational",
            "--splice_weight", "0.5", "--cospro_temperature", "0.25", "--cospro_teacher_graph", GRAPH,
        ]
        _, spelled_out = parse(explicit)
        self.assertEqual(with_preset, spelled_out)
        self.assertNotIn("preset", with_preset)

    def test_explicit_options_override_the_preset(self):
        _, values = parse(["--preset", "cospro_student", "--cospro_teacher_graph", GRAPH, "--temp", "0.1"])
        self.assertEqual(values["temp"], 0.1)
        self.assertEqual(values["batch_size"], 128)

    def test_presets_name_existing_options(self):
        defaults = training_defaults()
        for name, values in PRESETS.items():
            with self.subTest(preset=name):
                self.assertEqual(set(values) - set(defaults), set())


if __name__ == "__main__":
    unittest.main()
