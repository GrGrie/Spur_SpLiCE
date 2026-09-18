"""Readers for names that historical CRP-era artifacts and runs still carry.

Current code uses CoSpRo names throughout. The mappings below keep checkpoint folders, resumed
checkpoints and saved graphs readable after the rename.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Training options renamed from crp_* to cospro_*. Run storage names hash the configuration
# under the historical names, so a resumed run keeps writing into its existing folder.
LEGACY_OPTION_NAMES = {
    "cospro_teacher_graph": "crp_teacher_graph",
    "cospro_temperature": "crp_temperature",
    "cospro_start_epoch": "crp_start_epoch",
    "cospro_warmup_epochs": "crp_warmup_epochs",
    "cospro_decay_start_epoch": "crp_decay_start_epoch",
    "cospro_decay_end_epoch": "crp_decay_end_epoch",
    "cospro_graph_fingerprint": "crp_graph_fingerprint",
}

# Relational mode value accepted for historical manifests next to "cospro_relational".
LEGACY_RELATIONAL_MODE = "crp_relational"

# Teacher-graph and concept-group artifact types written before the rename.
LEGACY_TEACHER_GRAPH_ARTIFACTS = ("splice_crp_v2_teacher_graph", "splice_crp_v3_teacher_graph")
LEGACY_CONCEPT_GROUP_ARTIFACTS = frozenset({"splice_crp_concept_groups"})


def with_legacy_option_names(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``options`` keyed by the historical option names."""

    return {LEGACY_OPTION_NAMES.get(key, key): value for key, value in options.items()}


def saved_graph_fingerprint(options: Mapping[str, Any] | Any) -> str | None:
    """Read the teacher-graph fingerprint from checkpoint options of either generation."""

    for name in ("cospro_graph_fingerprint", "crp_graph_fingerprint"):
        value = options.get(name) if isinstance(options, Mapping) else getattr(options, name, None)
        if value is not None:
            return value
    return None
