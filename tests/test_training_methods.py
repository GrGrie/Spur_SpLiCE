"""The training-method seam: registry, selection and the loader each method builds."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import torch

import spur_splice
from cospro.config.training import TRAINING_MODES, TrainingConfig
from cospro.methods import (
    METHOD_REGISTRY,
    CoSpRoRelational,
    FrozenConceptDistill,
    LaSSL,
    LoaderContext,
    SimCLROnly,
    build_method,
    method_class,
    method_for,
    register_method,
)
from cospro.methods.base import TrainingMethod


def config_for(argv: list[str]) -> TrainingConfig:
    with patch.object(sys, "argv", ["spur_splice.py"]), patch("torch.cuda.is_available", return_value=False):
        parser = spur_splice.build_training_parser()
        args = spur_splice.parse_training_arguments(parser, argv)
        spur_splice.normalize_training_options(args)
    return TrainingConfig.from_namespace(args)


class MethodRegistryTests(unittest.TestCase):
    def test_every_training_mode_has_exactly_one_method(self):
        for mode in TRAINING_MODES:
            with self.subTest(mode=mode):
                self.assertTrue(method_for(splice_mode=mode, la_ssl=False).modes)
        self.assertIs(method_for(splice_mode="none", la_ssl=True), LaSSL)

    def test_configuration_selects_the_method(self):
        self.assertIs(method_class(config_for([])), SimCLROnly)
        self.assertIs(method_class(config_for(["--la_ssl"])), LaSSL)
        self.assertIs(
            method_class(config_for(["--splice_mode", "crp_relational", "--cospro_teacher_graph", "g.json",
                                     "--splice_weight", "0.5"])),
            CoSpRoRelational,
        )
        self.assertIs(
            method_class(config_for(["--splice_mode", "frozen_concept_distill"])), FrozenConceptDistill,
        )

    def test_a_new_method_needs_a_class_and_a_registration(self):
        @register_method
        class Dummy(TrainingMethod):
            name = "dummy_method"
            modes = ("dummy_mode",)

        try:
            self.assertIs(method_for(splice_mode="dummy_mode", la_ssl=False), Dummy)
            with self.assertRaisesRegex(ValueError, "both claim modes"):
                register_method(type("Clash", (TrainingMethod,), {"name": "clash", "modes": ("dummy_mode",)}))
        finally:
            METHOD_REGISTRY.pop("dummy_method")

    def test_plain_simclr_adds_no_loss_and_keeps_the_loader(self):
        method = build_method(config_for([]))
        loader = object()
        context = LoaderContext(dataset="waterbirds", batch_size=4, num_workers=0, seed=0)
        self.assertIs(method.wrap_loader(loader, context), loader)
        self.assertIsNone(method.extra_loss(model=None, embeddings=torch.zeros(2), sample_indices=None))
        self.assertEqual(method.provenance(), {})
        self.assertEqual(method.input_artifacts(), [])
        self.assertIsNone(method.sampling_state())

    def test_relational_method_reports_its_graph_and_input_artifact(self):
        method = CoSpRoRelational(graph_path="graph.json", weight=0.5, temperature=0.25)
        self.assertEqual([path.name for path in method.input_artifacts()], ["graph.json"])
        self.assertEqual(method.provenance(), {"relational_graph_empty": False})
        method.graph = {"artifact": "cospro_teacher_graph_v3", "degree_stats": {"edge_count": 3},
                        "selected_group_ids": [1], "groups": [{"selected": True, "concepts": ["water"]}]}
        provenance = method.provenance()
        self.assertEqual(provenance["teacher_graph_artifact"], "cospro_teacher_graph_v3")
        self.assertEqual(provenance["teacher_graph_removed_concepts"], ["water"])

    def test_frozen_transfer_declares_its_prediction_head(self):
        self.assertEqual(FrozenConceptDistill.clip_distillation_dim, 512)
        self.assertIsNone(SimCLROnly.clip_distillation_dim)


if __name__ == "__main__":
    unittest.main()
