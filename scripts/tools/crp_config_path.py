"""Build a stable, human-readable directory path from a complete CRP config."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import PurePosixPath


REQUIRED_KEYS = (
    "cobalt",
    "min_concept_frequency",
    "max_concept_frequency",
    "text_similarity_threshold",
    "coactivation_threshold",
    "min_group_size",
    "max_selected_groups",
    "projected_neighbors",
    "activation_difference_quantile",
    "min_intervention_gain",
    "min_coverage",
    "graph_top_k",
    "max_indegree",
    "indegree_factor",
    "null_trials",
    "null_quantile",
    "similarity_chunk_size",
    "orthogonal_tolerance",
    "use_residual_splice_gate",
    "residual_splice_similarity_threshold",
    "use_cobalt_confidence",
)


def _value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".12g")
    raise TypeError(f"Unsupported CRP config value: {value!r}")


def _safe_label(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    if not normalized:
        raise ValueError("variant must contain at least one path-safe character.")
    return normalized


def config_path(config: dict, variant: str = "") -> PurePosixPath:
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"CRP config is missing path fields: {missing}")
    def value(key: str) -> str:
        return _value(config[key])

    root = PurePosixPath("configs")
    if variant:
        root /= f"named-{_safe_label(variant)}"
    return root.joinpath(
        f"cobalt-{value('cobalt')}",
        f"freq-{value('min_concept_frequency')}-{value('max_concept_frequency')}_"
        f"group-{value('min_group_size')}-{value('text_similarity_threshold')}-"
        f"{value('coactivation_threshold')}-cap{value('max_selected_groups')}",
        f"search-{value('projected_neighbors')}_"
        f"actq-{value('activation_difference_quantile')}_gain-{value('min_intervention_gain')}_"
        f"cov-{value('min_coverage')}",
        f"graph-k{value('graph_top_k')}-maxdeg{value('max_indegree')}-indeg{value('indegree_factor')}_"
        f"null-{value('null_trials')}-{value('null_quantile')}_"
        f"num-{value('similarity_chunk_size')}-{value('orthogonal_tolerance')}_"
        f"resid-{value('use_residual_splice_gate')}-{value('residual_splice_similarity_threshold')}_"
        f"cbconf-{value('use_cobalt_confidence')}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default="")
    args = parser.parse_args()
    print(config_path(json.loads(args.config), args.variant).as_posix())


if __name__ == "__main__":
    main()
