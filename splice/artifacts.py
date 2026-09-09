"""Canonical placement of experiment artifacts.

All new runs cross this seam instead of spelling output-directory conventions
in launchers and experiment code.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
OUTPUT_ROOT_ENV = "SPUR_SPLICE_ARTIFACT_ROOT"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def resolve_output_root(root: str | Path | None = None) -> Path:
    """Resolve the artifact tree without guessing from directories on disk."""

    configured = root if root is not None else os.environ.get(OUTPUT_ROOT_ENV)
    if configured is None or not str(configured).strip():
        return OUTPUT_ROOT
    path = Path(configured).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _safe_name(value: str, kind: str) -> str:
    value = str(value).strip().lower()
    if not _NAME.fullmatch(value):
        raise ValueError(f"Invalid {kind} name {value!r}; use lowercase letters, numbers, '.', '_' or '-'.")
    return value


def seed_run(seed: int, study: str, arm: str | None = None, *, root: str | Path | None = None) -> Path:
    """Return the directory for one seed-specific run."""

    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    path = resolve_output_root(root) / "seeds" / f"seed_{int(seed):02d}" / _safe_name(study, "study")
    return path / _safe_name(arm, "arm") if arm else path


def shared(dataset: str, *parts: str, root: str | Path | None = None) -> Path:
    """Return a path for seed-independent cache or graph artifacts."""

    path = resolve_output_root(root) / "shared" / _safe_name(dataset, "dataset")
    for part in parts:
        path /= _safe_name(part, "path")
    return path


def report(study: str, *parts: str, root: str | Path | None = None) -> Path:
    """Return a path for aggregate reports and study-level provenance."""

    path = resolve_output_root(root) / "reports" / _safe_name(study, "study")
    for part in parts:
        path /= _safe_name(part, "path")
    return path


def reference(*parts: str, root: str | Path | None = None) -> Path:
    """Return a path for vocabularies and exports that are not experiment runs."""

    path = resolve_output_root(root) / "reference"
    for part in parts:
        path /= _safe_name(part, "path")
    return path
