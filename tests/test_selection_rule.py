"""Selection as a stage: which audited groups reach the graph, and how that is recorded.

Phase 2 showed selection decides graph quality, so the rule is swappable and the graph says which
one produced it. The default rule must keep reproducing the paper protocol, which the golden graph
snapshot pins; these tests pin the seam around it.
"""

from __future__ import annotations

import unittest

import torch

from golden_support import synthetic_splice_cache, train_sample_ids

from cospro.pipeline import CoSpRoAuditConfig, build_concept_groups, build_teacher_graph
from cospro.pipeline.selection import (
    SELECTION_RULES,
    NullQuantilePass,
    SelectionRule,
    selection_rule,
)

CONFIG = CoSpRoAuditConfig(
    text_similarity_threshold=0.8,
    coactivation_threshold=0.35,
    projected_neighbors=6,
    graph_top_k=3,
    max_indegree=6,
    null_trials=4,
)


def pipeline_inputs():
    sample_ids, y, place = train_sample_ids()
    cache = synthetic_splice_cache(sample_ids, y, place)
    return cache, build_concept_groups(cache, CONFIG)


class RejectEverything(SelectionRule):
    name = "reject_everything"

    def accepts(self, group, config) -> bool:
        return False

    def retain(self, candidates, groups, config):
        return []


class KeepTheFirstAccepted(SelectionRule):
    """Accept whatever the default accepts, then keep only the lowest group id."""

    name = "keep_first"

    def accepts(self, group, config) -> bool:
        return NullQuantilePass().accepts(group, config)

    def retain(self, candidates, groups, config):
        return sorted(candidates, key=lambda item: item[0])[:1]


class SelectionRuleRegistryTests(unittest.TestCase):
    def test_the_default_rule_is_the_paper_protocol(self):
        self.assertEqual(sorted(SELECTION_RULES), ["null_quantile"])
        self.assertIsInstance(selection_rule(), NullQuantilePass)
        self.assertIsInstance(selection_rule("null_quantile"), NullQuantilePass)
        rule = RejectEverything()
        self.assertIs(selection_rule(rule), rule, "a rule instance passes through")

    def test_an_unknown_rule_names_the_rules_there_are(self):
        with self.assertRaisesRegex(ValueError, "Unknown selection rule"):
            selection_rule("keep_everything")

    def test_the_default_rule_accepts_on_coverage_and_the_null_threshold(self):
        rule, config = NullQuantilePass(), CoSpRoAuditConfig(min_coverage=0.5)
        self.assertTrue(rule.accepts({"coverage": 0.9, "score": 1.0, "null_threshold": 0.5}, config))
        self.assertFalse(rule.accepts({"coverage": 0.1, "score": 1.0, "null_threshold": 0.5}, config))
        self.assertFalse(rule.accepts({"coverage": 0.9, "score": 0.5, "null_threshold": 0.5}, config))


class SelectionInTheGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache, cls.groups = pipeline_inputs()

    def build(self, selection=None):
        return build_teacher_graph(self.cache, self.groups, CONFIG, device="cpu", selection=selection)

    def test_naming_the_default_rule_changes_nothing(self):
        default = self.build()
        named = self.build("null_quantile")
        self.assertEqual(default["selected_group_ids"], named["selected_group_ids"])
        torch.testing.assert_close(
            torch.as_tensor(default["weights"]), torch.as_tensor(named["weights"]),
        )
        self.assertTrue(default["selected_group_ids"], "the fixture selects at least one group")

    def test_the_graph_records_the_rule_that_produced_it(self):
        graph = self.build()
        self.assertEqual(graph["selection"]["rule"], "null_quantile")
        self.assertEqual(graph["selection"]["max_selected_groups"], CONFIG.max_selected_groups or None)

    def test_a_rule_that_rejects_everything_produces_an_empty_graph(self):
        graph = self.build(RejectEverything())
        self.assertEqual(graph["selected_group_ids"], [])
        self.assertFalse(any(group["selected"] for group in graph["groups"]))
        self.assertEqual(float(torch.as_tensor(graph["weights"]).sum()), 0.0)
        self.assertEqual(graph["selection"]["rule"], "reject_everything")

    def test_a_capping_rule_marks_the_groups_it_dropped(self):
        default = self.build()
        capped = self.build(KeepTheFirstAccepted())
        self.assertEqual(len(capped["selected_group_ids"]), 1)
        self.assertLessEqual(len(capped["selected_group_ids"]), len(default["selected_group_ids"]))
        dropped = [group for group in capped["groups"] if group.get("rejection_reason")]
        self.assertEqual(
            len(dropped), len(default["selected_group_ids"]) - 1,
            "every accepted group the rule did not retain says why it was dropped",
        )


if __name__ == "__main__":
    unittest.main()
