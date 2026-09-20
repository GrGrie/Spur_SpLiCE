"""Evaluation of a trained encoder.

The linear probe is the project's headline measurement. ``evaluate_probe`` performs it and returns
a ``ProbeResult``; ``persist_probe_result`` and ``log_probe_result`` do the side effects, and
``probe_checkpoint`` is the composition the trainer, the CLI and the submission tool all use.
"""

from cospro.evaluation.probe import (
    ProbeArtifacts,
    ProbeHistory,
    ProbeOptions,
    ProbeResult,
    evaluate_probe,
    log_probe_result,
    persist_probe_result,
    probe_checkpoint,
    seed_probe,
)

__all__ = [
    "ProbeArtifacts", "ProbeHistory", "ProbeOptions", "ProbeResult", "evaluate_probe",
    "log_probe_result", "persist_probe_result", "probe_checkpoint", "seed_probe",
]
