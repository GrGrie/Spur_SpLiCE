"""Evaluation of a trained encoder.

The linear probe is the project's headline measurement. ``evaluate_probe`` performs it and returns
a ``ProbeResult``; ``persist_probe_result`` and ``log_probe_result`` do the side effects, and
``probe_checkpoint`` is the composition the trainer, the CLI and the submission tool all use.

Names are imported on first use, so ``cospro.evaluation.protocol`` imports without the rest of
the package and its dependencies.
"""

import importlib

_EXPORTS = {
    "ProbeArtifacts": "cospro.evaluation.probe",
    "ProbeHistory": "cospro.evaluation.probe",
    "ProbeOptions": "cospro.evaluation.probe",
    "ProbeResult": "cospro.evaluation.probe",
    "evaluate_probe": "cospro.evaluation.probe",
    "log_probe_result": "cospro.evaluation.probe",
    "persist_probe_result": "cospro.evaluation.probe",
    "probe_checkpoint": "cospro.evaluation.probe",
    "seed_probe": "cospro.evaluation.probe",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
