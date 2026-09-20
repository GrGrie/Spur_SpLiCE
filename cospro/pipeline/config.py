"""The settings one CoSpRo pipeline run is described by.

Grouping, the audit and graph assembly read the same configuration, so it has one definition and
one validator. ``GROUPING_CONFIG_FIELDS`` names the subset a concept-group artifact carries, which
is what lets a graph build resume from groups produced by an earlier run.
"""

from __future__ import annotations

from dataclasses import dataclass

GROUPING_CONFIG_FIELDS = (
    "min_concept_frequency",
    "max_concept_frequency",
    "text_similarity_threshold",
    "coactivation_threshold",
    "min_group_size",
    "similarity_chunk_size",
)


@dataclass(frozen=True)
class CoSpRoAuditConfig:
    min_concept_frequency: float = 0.01
    max_concept_frequency: float = 0.95
    text_similarity_threshold: float = 0.82
    coactivation_threshold: float = 0.35
    min_group_size: int = 1
    max_selected_groups: int = 0
    projected_neighbors: int = 20
    activation_difference_quantile: float = 0.75
    min_intervention_gain: float = 1e-4
    min_coverage: float = 0.01
    graph_top_k: int = 3
    max_indegree: int = 10
    # Retained as a legacy fallback for callers constructing a config without
    # max_indegree; CoSpRo v3 uses the absolute cap above.
    indegree_factor: float = 3.0
    null_trials: int = 16
    null_quantile: float = 0.95
    seed: int = 0
    similarity_chunk_size: int = 512
    orthogonal_tolerance: float = 1e-6
    use_residual_splice_gate: bool = True
    residual_splice_similarity_threshold: float = 0.25
    # Exact all-pairs search is useful for small audits and regression tests.
    # Large datasets use deterministic random-hyperplane LSH followed by exact
    # cosine reranking inside the candidate buckets.
    neighbor_backend: str = "auto"
    ann_threshold: int = 20_000
    ann_tables: int = 8
    ann_bucket_size: int = 512


def _validate_config(config: CoSpRoAuditConfig) -> None:
    boolean_fields = {
        "use_residual_splice_gate": config.use_residual_splice_gate,
    }
    if any(not isinstance(value, bool) for value in boolean_fields.values()):
        raise ValueError("CoSpRo boolean settings must be booleans.")
    probabilities = {
        "min_concept_frequency": config.min_concept_frequency,
        "max_concept_frequency": config.max_concept_frequency,
        "activation_difference_quantile": config.activation_difference_quantile,
        "min_coverage": config.min_coverage,
        "null_quantile": config.null_quantile,
        "residual_splice_similarity_threshold": config.residual_splice_similarity_threshold,
    }
    for name, value in probabilities.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
    if config.min_concept_frequency > config.max_concept_frequency:
        raise ValueError("min_concept_frequency cannot exceed max_concept_frequency.")
    integer_fields = {
        "min_group_size": config.min_group_size,
        "projected_neighbors": config.projected_neighbors,
        "graph_top_k": config.graph_top_k,
        "max_indegree": config.max_indegree,
        "null_trials": config.null_trials,
        "similarity_chunk_size": config.similarity_chunk_size,
        "ann_threshold": config.ann_threshold,
        "ann_tables": config.ann_tables,
        "ann_bucket_size": config.ann_bucket_size,
    }
    for name, value in integer_fields.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if config.max_selected_groups < 0:
        raise ValueError("max_selected_groups must be non-negative; 0 disables the cap.")
    if config.neighbor_backend not in {"auto", "exact", "lsh"}:
        raise ValueError("neighbor_backend must be one of: auto, exact, lsh.")


def validate_cospro_config(config: CoSpRoAuditConfig) -> CoSpRoAuditConfig:
    """Validate a complete grouping/audit configuration and return it unchanged."""

    _validate_config(config)
    return config
