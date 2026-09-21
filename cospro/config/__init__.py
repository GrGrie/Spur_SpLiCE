"""Typed configuration for CoSpRo training, probing and the teacher pipeline.

Each configuration section is a frozen dataclass whose field names equal the historical command-line
destinations. The generated parser keeps every existing flag, so runner commands, stored namespaces
and storage names stay unchanged while the defaults and checks live in one typed place.

Names are imported on first use, so ``cospro.config.settings`` imports without the rest of
the package and its dependencies.
"""

import importlib

_EXPORTS = {
    "ConfigError": "cospro.config.options",
    "DEFAULT_PERIODIC_PROBE_FREQ": "cospro.config.training",
    "LINEAR_PROBE_DEFAULTS": "cospro.config.training",
    "PRESETS": "cospro.config.presets",
    "TRAINING_SECTIONS": "cospro.config.training",
    "TrainingConfig": "cospro.config.training",
    "build_training_parser": "cospro.config.training",
    "normalize_training_options": "cospro.config.training",
    "option": "cospro.config.options",
    "preset_values": "cospro.config.presets",
    "str_to_bool": "cospro.config.options",
    "training_defaults": "cospro.config.training",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
