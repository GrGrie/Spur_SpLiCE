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
from cospro.pipeline.concept_type import group_scores
from cospro.pipeline.selection import (
    SELECTION_RULES,
    ConceptTypeGate,
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


class FakeTextEncoder:
    """A text tower whose embedding of a word is fixed by a table, for the concept-type gate."""

    def __init__(self, table):
        self.table = table
        self.parameters = lambda: iter([torch.zeros(1)])

    def encode_text(self, tokens):
        return torch.stack([self.table[int(index)] for index in tokens])


def fake_tokenizer(table):
    words = list(table)
    return lambda texts: torch.tensor([[words.index(next(w for w in words if w in text))] for text in texts])


class ConceptTypeGateTests(unittest.TestCase):
    def setUp(self):
        # Two directions: the first is "context", the second is "object".
        self.table = {"a place or scene": torch.tensor([1.0, 0.0]), "the background of a photo": torch.tensor([1.0, 0.0]),
                      "furniture or a room": torch.tensor([1.0, 0.0]), "an outdoor landscape": torch.tensor([1.0, 0.0]),
                      "weather or time of day": torch.tensor([1.0, 0.0]), "a photo of an animal": torch.tensor([0.0, 1.0]),
                      "a kind of animal": torch.tensor([0.0, 1.0]), "a photo of a person": torch.tensor([0.0, 1.0]),
                      "an object in the foreground": torch.tensor([0.0, 1.0]),
                      "couch": torch.tensor([1.0, 0.0]), "cats": torch.tensor([0.0, 1.0])}
        encoder = FakeTextEncoder({index: value for index, value in enumerate(self.table.values())})
        self.scores = group_scores([{"group_id": 0, "concepts": ["couch"]}, {"group_id": 1, "concepts": ["cats"]}],
                                   encoder, fake_tokenizer(self.table))

    def test_words_that_name_an_object_score_below_words_that_name_context(self):
        self.assertGreater(self.scores["0"], self.scores["1"])

    def test_the_gate_drops_the_most_object_like_quantile_then_ranks_as_the_default(self):
        groups = [{"group_id": 0, "null_excess_score": 1.0, "score": 1.0}, {"group_id": 1, "null_excess_score": 2.0, "score": 2.0}]
        candidates = [(0, {}), (1, {})]
        config = CoSpRoAuditConfig(max_selected_groups=0)
        kept = ConceptTypeGate(self.scores, quantile=0.5).retain(candidates, groups, config)
        self.assertEqual([index for index, _ in kept], [0], "the object-like group loses despite its higher score")
        kept = ConceptTypeGate(self.scores, quantile=0.0).retain(candidates, groups, config)
        self.assertEqual([index for index, _ in kept], [1, 0], "quantile zero keeps every candidate")

    def test_the_gate_records_what_it_cut_and_refuses_to_run_blind(self):
        config = CoSpRoAuditConfig()
        rule = ConceptTypeGate(self.scores, quantile=0.5)
        rule.retain([(0, {}), (1, {})], [{"group_id": 0, "null_excess_score": 1.0, "score": 1.0},
                                         {"group_id": 1, "null_excess_score": 2.0, "score": 2.0}], config)
        provenance = rule.provenance(config)
        self.assertEqual(provenance["rule"], "concept_type_gate")
        self.assertEqual(provenance["object_quantile"], 0.5)
        self.assertEqual(provenance["scored_groups"], 2)
        with self.assertRaisesRegex(ValueError, "needs per-group concept-type scores"):
            ConceptTypeGate().retain([(0, {})], [{"group_id": 0}], config)
        with self.assertRaisesRegex(ValueError, "quantile must lie"):
            ConceptTypeGate(self.scores, quantile=1.0)


class SelectionRuleRegistryTests(unittest.TestCase):
    def test_the_default_rule_is_the_paper_protocol(self):
        self.assertEqual(sorted(SELECTION_RULES), ["concept_type_gate", "null_quantile"])
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
