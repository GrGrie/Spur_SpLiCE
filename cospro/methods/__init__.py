"""Training methods behind one interface.

``build_method`` turns a resolved training configuration into the method the run uses. Adding a
method means adding a module here with ``@register_method`` plus its options section in
``cospro.config``; the trainer, the datasets and the epoch loop stay unchanged.
"""

from __future__ import annotations

from cospro.config.training import TrainingConfig
from cospro.methods.base import (
    METHOD_REGISTRY,
    LoaderContext,
    LossTerms,
    TrainingMethod,
    method_for,
    register_method,
)
from cospro.methods.concept_transfer import FrozenConceptDistill
from cospro.methods.la_ssl import LaSSL
from cospro.methods.relational import CoSpRoRelational
from cospro.methods.simclr import SimCLROnly


def method_class(config: TrainingConfig) -> type[TrainingMethod]:
    """The registered method class a configuration selects."""

    return method_for(splice_mode=config.method.splice_mode, la_ssl=config.la_ssl.la_ssl)


def build_method(config: TrainingConfig, **resolved) -> TrainingMethod:
    """Build the configured method. ``resolved`` carries values produced during argument
    resolution, such as ``graph_fingerprint``."""

    return method_class(config).from_config(config, **resolved)


__all__ = [
    "METHOD_REGISTRY", "LoaderContext", "LossTerms", "TrainingMethod", "register_method",
    "CoSpRoRelational", "FrozenConceptDistill", "LaSSL", "SimCLROnly", "build_method", "method_class",
    "method_for",
]
