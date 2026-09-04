from __future__ import annotations

import unittest

from scripts.tools.crp_config_path import config_path


def _config(**overrides):
    values = {
        "cobalt": False,
        "min_concept_frequency": 0.01,
        "max_concept_frequency": 0.95,
        "text_similarity_threshold": 0.8,
        "coactivation_threshold": 0.35,
        "min_group_size": 1,
        "max_selected_groups": 0,
        "projected_neighbors": 20,
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
    }
    values.update(overrides)
    return values


class CrpConfigPathTests(unittest.TestCase):
    def test_equivalent_numeric_values_have_the_same_path(self):
        self.assertEqual(config_path(_config()), config_path(_config(indegree_factor=3)))

    def test_every_graph_parameter_changes_the_path(self):
        baseline = config_path(_config())
        for key, value in _config().items():
            replacement = not value if isinstance(value, bool) else value + 1
            with self.subTest(key=key):
                self.assertNotEqual(baseline, config_path(_config(**{key: replacement})))

    def test_variant_is_sanitized_without_replacing_configuration(self):
        baseline = config_path(_config())
        labelled = config_path(_config(), "precision graph")
        self.assertEqual(baseline.parts[-4:], labelled.parts[-4:])
        self.assertIn("named-precision-graph", labelled.parts)


if __name__ == "__main__":
    unittest.main()
