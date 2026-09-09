"""Canonical placement of experiment artifacts.

All new runs cross this seam instead of spelling output-directory conventions
in launchers and experiment code.
"""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _safe_name(value: str, kind: str) -> str:
    value = str(value).strip().lower()
    if not _NAME.fullmatch(value):
        raise ValueError(f"Invalid {kind} name {value!r}; use lowercase letters, numbers, '.', '_' or '-'.")
    return value


def seed_run(seed: int, study: str, arm: str | None = None) -> Path:
    """Return the directory for one seed-specific run."""

    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    path = OUTPUT_ROOT / "seeds" / f"seed_{int(seed):02d}" / _safe_name(study, "study")
    return path / _safe_name(arm, "arm") if arm else path


def shared(dataset: str, *parts: str) -> Path:
    """Return a path for seed-independent cache or graph artifacts."""

    path = OUTPUT_ROOT / "shared" / _safe_name(dataset, "dataset")
    for part in parts:
        path /= _safe_name(part, "path")
    return path


def report(study: str, *parts: str) -> Path:
    """Return a path for aggregate reports and study-level provenance."""

    path = OUTPUT_ROOT / "reports" / _safe_name(study, "study")
    for part in parts:
        path /= _safe_name(part, "path")
    return path


def reference(*parts: str) -> Path:
    """Return a path for vocabularies and exports that are not experiment runs."""

    path = OUTPUT_ROOT / "reference"
    for part in parts:
        path /= _safe_name(part, "path")
    return path
