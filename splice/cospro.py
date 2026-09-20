"""Re-exports of the CoSpRo pipeline, which now lives in :mod:`cospro.pipeline`.

The pipeline was one 1524-line module holding cache validation, grouping, neighbour search, the
audit, graph assembly and a duplicate CLI. It is now one module per stage. This module keeps the
historical import path working for scripts and notebooks written against it; new code imports from
``cospro.pipeline``.
"""

from cospro.pipeline.audit import (  # noqa: F401  (historical import path)
    _AuditGeometry,
    _candidate_code_dot,
    _gini,
    _neighbor_geometry,
    _null_scores,
    _relation_geometry,
    _residual_splice_similarity,
    _score_relations,
)
from cospro.pipeline.cache import (  # noqa: F401
    FORBIDDEN_CACHE_KEYS,
    REQUIRED_CACHE_KEYS,
    SPLICE_DATASET_CACHE_VERSION,
    _atomic_torch_save,
    _normalized_rows,
    save_splice_dataset_cache,
    validate_splice_dataset_cache,
)
from cospro.pipeline.config import (  # noqa: F401
    GROUPING_CONFIG_FIELDS,
    CoSpRoAuditConfig,
    _validate_config,
    validate_cospro_config,
)
from cospro.pipeline.graph import (  # noqa: F401
    COSPRO_GRAPH_VERSION,
    COSPRO_TEACHER_GRAPH_ARTIFACT,
    GRAPH_VERSION,
    _build_teacher_graph,
    build_teacher_graph,
)
from cospro.pipeline.grouping import (  # noqa: F401
    CONCEPT_GROUPS_VERSION,
    COSPRO_CONCEPT_GROUP_ARTIFACT,
    _active_concept_indices,
    _concept_group_diagnostics,
    _concept_group_report_diagnostics,
    _group_concepts,
    _grouping_config,
    _lexical_key,
    build_concept_groups,
    load_concept_groups_json,
    save_concept_groups_json,
    validate_concept_groups,
)
from cospro.pipeline.neighbors import (  # noqa: F401
    _exact_topk_neighbors,
    _lsh_topk_neighbors,
    orthonormal_basis,
    project_out,
    topk_neighbors,
)
