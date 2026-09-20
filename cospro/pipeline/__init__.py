"""The label-free CoSpRo pipeline: cache, concept groups, audit, teacher graph.

Each stage reads the artifact the previous one wrote, so a stage can be rerun, swapped or resumed
on its own. ``cache`` validates the frozen SpLiCE representations, ``grouping`` merges vocabulary
entries into concept groups, ``neighbors`` answers neighbourhood queries, ``audit`` scores one
group against its null controls and ``graph`` assembles the teacher graph training consumes.
"""

from cospro.pipeline.cache import (
    FORBIDDEN_CACHE_KEYS,
    REQUIRED_CACHE_KEYS,
    SPLICE_DATASET_CACHE_VERSION,
    save_splice_dataset_cache,
    validate_splice_dataset_cache,
)
from cospro.pipeline.config import GROUPING_CONFIG_FIELDS, CoSpRoAuditConfig, validate_cospro_config
from cospro.pipeline.graph import (
    COSPRO_GRAPH_VERSION,
    COSPRO_TEACHER_GRAPH_ARTIFACT,
    GRAPH_VERSION,
    build_teacher_graph,
)
from cospro.pipeline.grouping import (
    CONCEPT_GROUPS_VERSION,
    COSPRO_CONCEPT_GROUP_ARTIFACT,
    build_concept_groups,
    load_concept_groups_json,
    save_concept_groups_json,
    validate_concept_groups,
)
from cospro.pipeline.neighbors import (
    NEIGHBOR_INDEXES,
    ExactNeighbors,
    LshNeighbors,
    NeighborIndex,
    build_index,
    orthonormal_basis,
    project_out,
    topk_neighbors,
)
from cospro.pipeline.selection import (
    SELECTION_RULES,
    NullQuantilePass,
    SelectionRule,
    selection_rule,
)

__all__ = [
    "CONCEPT_GROUPS_VERSION", "COSPRO_CONCEPT_GROUP_ARTIFACT", "COSPRO_GRAPH_VERSION",
    "COSPRO_TEACHER_GRAPH_ARTIFACT", "CoSpRoAuditConfig", "ExactNeighbors", "FORBIDDEN_CACHE_KEYS",
    "GRAPH_VERSION", "GROUPING_CONFIG_FIELDS", "LshNeighbors", "NEIGHBOR_INDEXES", "NeighborIndex",
    "NullQuantilePass", "REQUIRED_CACHE_KEYS", "SELECTION_RULES", "SPLICE_DATASET_CACHE_VERSION",
    "SelectionRule", "build_concept_groups", "build_index", "build_teacher_graph",
    "load_concept_groups_json", "orthonormal_basis", "project_out", "save_concept_groups_json",
    "save_splice_dataset_cache", "selection_rule", "topk_neighbors", "validate_concept_groups",
    "validate_cospro_config", "validate_splice_dataset_cache",
]
