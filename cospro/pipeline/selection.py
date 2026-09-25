"""Which audited concept groups become edges in the teacher graph.

Phase 2 showed that selection decides graph quality as much as the audit does, so it is a stage of
its own rather than two conditions buried in the audit loop. A rule answers two questions: whether
one group beats its own null controls, and which of the survivors are kept once they are ranked
against each other.

``NullQuantilePass`` is the rule the paper protocol uses and the default everywhere. A rule is
passed to ``build_teacher_graph`` as an argument rather than named in ``CoSpRoAuditConfig``,
because the configuration is hashed into the group-checkpoint identity: adding a field there would
invalidate every audit checkpoint on the cluster. Naming rules on the command line is a deliberate
step to take later, together with that cost.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Sequence

#: One audited group and the evidence its edges were scored from.
Candidate = tuple[int, dict]


class SelectionRule:
    """The two decisions that turn audited groups into the graph's edge sources."""

    name: ClassVar[str] = ""

    def accepts(self, group: Mapping[str, Any], config) -> bool:
        """True when one group beats its own null controls."""

        raise NotImplementedError

    def retain(self, candidates: Sequence[Candidate], groups: Sequence[dict], config) -> list[Candidate]:
        """The accepted groups that survive ranking against each other."""

        raise NotImplementedError

    def provenance(self, config) -> dict[str, Any]:
        """What the teacher graph records about how its groups were selected."""

        return {"rule": self.name, "max_selected_groups": config.max_selected_groups or None}


SELECTION_RULES: dict[str, type[SelectionRule]] = {}


def register_rule(cls: type[SelectionRule]) -> type[SelectionRule]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} must declare a rule name.")
    if cls.name in SELECTION_RULES:
        raise ValueError(f"Selection rule {cls.name!r} is registered twice.")
    SELECTION_RULES[cls.name] = cls
    return cls


@register_rule
class NullQuantilePass(SelectionRule):
    """Pass the null quantile with enough coverage, then keep the largest null excess.

    A group is accepted when its score beats the configured quantile of its own null scores and it
    covers enough anchors. The survivors are ranked by how far past their null threshold they
    reached, and ``--max-selected-groups`` caps how many contribute edges.
    """

    name = "null_quantile"

    def accepts(self, group: Mapping[str, Any], config) -> bool:
        return bool(
            float(group["coverage"]) >= config.min_coverage
            and float(group["score"]) > float(group["null_threshold"])
        )

    def retain(self, candidates: Sequence[Candidate], groups: Sequence[dict], config) -> list[Candidate]:
        ordered = sorted(
            candidates,
            key=lambda item: (
                -float(groups[item[0]]["null_excess_score"]),
                -float(groups[item[0]]["score"]),
                item[0],
            ),
        )
        return list(ordered[: config.max_selected_groups] if config.max_selected_groups else ordered)


@register_rule
class ConceptTypeGate(NullQuantilePass):
    """The null rule, minus the candidates whose concepts name the object rather than its context.

    ``scores`` maps a group id to its ``cospro.pipeline.concept_type`` context preference, read
    from the dictionary's own text encoder. The cut is a quantile of this dataset's candidates
    rather than an absolute number, because the scale of the scores depends on the vocabulary: one
    dimensionless setting therefore carries across datasets and dictionaries.
    """

    name = "concept_type_gate"

    def __init__(self, scores: Mapping[str, float] | None = None, quantile: float = 0.25) -> None:
        if not 0 <= quantile < 1:
            raise ValueError(f"The object quantile must lie in [0, 1); got {quantile}.")
        self.scores = {str(key): float(value) for key, value in (scores or {}).items()}
        self.quantile = float(quantile)
        self._threshold: float | None = None

    def preference(self, group_id: int) -> float:
        """A group without a score is treated as the most object-like, so it never survives."""

        return self.scores.get(str(group_id), float("-inf"))

    def retain(self, candidates: Sequence[Candidate], groups: Sequence[dict], config) -> list[Candidate]:
        if not self.scores:
            raise ValueError("The concept_type_gate rule needs per-group concept-type scores.")
        values = sorted(self.preference(groups[index]["group_id"]) for index, _ in candidates)
        cut = int(self.quantile * len(values))
        self._threshold = values[cut] if 0 < cut < len(values) else float("-inf")
        survivors = [item for item in candidates if self.preference(groups[item[0]]["group_id"]) >= self._threshold]
        return super().retain(survivors, groups, config)

    def provenance(self, config) -> dict[str, Any]:
        return {
            **super().provenance(config),
            "object_quantile": self.quantile,
            "context_preference_threshold": self._threshold,
            "scored_groups": len(self.scores),
        }


DEFAULT_SELECTION_RULE = "null_quantile"


def selection_rule(rule: str | SelectionRule | None = None) -> SelectionRule:
    """The rule a name selects, or the default when nothing is named."""

    if isinstance(rule, SelectionRule):
        return rule
    name = rule or DEFAULT_SELECTION_RULE
    rule_class = SELECTION_RULES.get(name)
    if rule_class is None:
        raise ValueError(f"Unknown selection rule: {name!r}. Rules: {sorted(SELECTION_RULES)}")
    return rule_class()
