"""Typed configuration for CoSpRo training, probing and the teacher pipeline.

Each configuration section is a frozen dataclass whose field names equal the historical command-line
destinations. The generated parser keeps every existing flag, so runner commands, stored namespaces
and storage names stay unchanged while the defaults and checks live in one typed place.
"""

from cospro.config.options import ConfigError, option, str_to_bool
from cospro.config.presets import PRESETS, preset_values
from cospro.config.training import (
    DEFAULT_PERIODIC_PROBE_FREQ,
    LINEAR_PROBE_DEFAULTS,
    TRAINING_SECTIONS,
    TrainingConfig,
    build_training_parser,
    normalize_training_options,
    training_defaults,
)

__all__ = [
    "ConfigError", "option", "str_to_bool", "PRESETS", "preset_values", "DEFAULT_PERIODIC_PROBE_FREQ",
    "LINEAR_PROBE_DEFAULTS", "TRAINING_SECTIONS", "TrainingConfig", "build_training_parser",
    "normalize_training_options", "training_defaults",
]
