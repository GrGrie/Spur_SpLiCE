from __future__ import annotations

import unittest

from scripts.tools.crp_config_path import config_path


def _config(**overrides):
    values = {
        "use_dino": True,
        "cobalt": False,
        "min_concept_frequency": 0.01,
        "max_concept_frequency": 0.95,
        "text_similarity_threshold": 0.8,
        "coactivation_threshold": 0.35,
        "min_group_size": 1,
        "max_selected_groups": 0,
        "projected_neighbors": 20,
        "dino_neighbors": 50,
        "activation_difference_quantile": 0.85,
        "min_intervention_gain": 5e-4,
        "min_coverage": 0.01,
        "graph_top_k": 3,
        "max_indegree": 10,
        "indegree_factor": 3.0,
        "null_trials": 32,
        "null_quantile": 0.95,
        "similarity_chunk_size": 512,
        "orthogonal_tolerance": 1e-6,
        "use_residual_splice_gate": True,
        "residual_splice_similarity_threshold": 0.25,
        "use_cross_fold_validation": True,
        "cross_fold_count": 2,
        "cross_fold_min_edge_persistence": 0.5,
        "use_cobalt_confidence": True,
        "spatial_balance": False,
        "spatial_balance_variant": "",
        "spatial_balance_floor": 0.25,
        "spatial_frequency_power": 0.0,
    }
    values.update(overrides)
    return values


class CrpConfigPathTests(unittest.TestCase):
    def test_equivalent_numeric_values_have_the_same_path(self):
        self.assertEqual(config_path(_config()), config_path(_config(indegree_factor=3)))

    def test_every_graph_parameter_changes_the_path(self):
        active = _config(spatial_balance=True, spatial_balance_variant="vanilla_slots")
        baseline = config_path(active)
        for key, value in active.items():
            if isinstance(value, bool):
                replacement = not value
            elif isinstance(value, str):
                replacement = "sclip_slots" if value == "vanilla_slots" else "vanilla_slots"
            else:
                replacement = value + 1
            with self.subTest(key=key):
                self.assertNotEqual(baseline, config_path({**active, key: replacement}))

    def test_variant_is_sanitized_without_replacing_configuration(self):
        baseline = config_path(_config())
        labelled = config_path(_config(), "precision graph")
        self.assertEqual(baseline.parts[-4:], labelled.parts[-4:])
        self.assertIn("named-precision-graph", labelled.parts)


if __name__ == "__main__":
    unittest.main()
