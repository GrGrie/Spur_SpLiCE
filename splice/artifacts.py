"""Canonical placement and integrity metadata for experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
OUTPUT_ROOT_ENV = "SPUR_SPLICE_OUTPUT_ROOT"
SCRATCH_ROOT_ENV = "SPUR_SPLICE_SCRATCH_ROOT"
LEGACY_SCRATCH_ROOT_ENV = "SPUR_SPLICE_ARTIFACT_ROOT"
DEFAULT_SCRATCH_ROOT = Path("/scratch/xar68reb/CoSpRo")
BINARY_SIZE_THRESHOLD = 10 * 1024 * 1024
BINARY_SUFFIXES = {".pt", ".pth", ".ckpt"}
_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _configured_path(value: str | Path, *, relative_to_project: bool = False) -> Path:
    path = Path(value).expanduser()
    if relative_to_project and not path.is_absolute():
        return (PROJECT_ROOT / path).resolve()
    return path


def resolve_output_root(root: str | Path | None = None) -> Path:
    """Return the Git-facing results root, independently of scratch storage."""

    configured = root if root is not None else os.environ.get(OUTPUT_ROOT_ENV)
    if configured is None or not str(configured).strip():
        return OUTPUT_ROOT
    return _configured_path(configured, relative_to_project=True)


def scratch_root() -> Path:
    """Return the large-binary root, accepting the old cluster variable as a fallback."""

    configured = os.environ.get(SCRATCH_ROOT_ENV) or os.environ.get(LEGACY_SCRATCH_ROOT_ENV) or DEFAULT_SCRATCH_ROOT
    return _configured_path(configured)


def _safe_name(value: str, kind: str) -> str:
    value = str(value).strip().lower()
    if not _NAME.fullmatch(value):
        raise ValueError(f"Invalid {kind} name {value!r}; use lowercase letters, numbers, '.', '_' or '-'.")
    return value


def seed_run(seed: int, study: str, arm: str | None = None, *, root: str | Path | None = None) -> Path:
    """Return the Git-facing directory for one seed in a study."""

    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    path = resolve_output_root(root) / "seeds" / _safe_name(study, "study") / f"seed_{int(seed):02d}"
    return path / _safe_name(arm, "arm") if arm else path


def run_directory(seed: int, study: str, arm: str, attempt_id: str, *, root: str | Path | None = None) -> Path:
    return seed_run(seed, study, arm, root=root) / _safe_name(attempt_id, "attempt")


def shared(dataset: str, *parts: str, root: str | Path | None = None) -> Path:
    path = resolve_output_root(root) / "shared" / _safe_name(dataset, "dataset")
    for part in parts:
        path /= _safe_name(part, "path")
    return path


def report(study: str, *parts: str, root: str | Path | None = None) -> Path:
    path = resolve_output_root(root) / "reports" / _safe_name(study, "study")
    for part in parts:
        path /= _safe_name(part, "path")
    return path


def reference(*parts: str, root: str | Path | None = None) -> Path:
    path = resolve_output_root(root) / "reference"
    for part in parts:
        path /= _safe_name(part, "path")
    return path


def make_attempt_id(environment: Mapping[str, str] | None = None) -> str:
    environment = os.environ if environment is None else environment
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    job = environment.get("SLURM_ARRAY_JOB_ID") or environment.get("SLURM_JOB_ID")
    task = environment.get("SLURM_ARRAY_TASK_ID")
    suffix = "-".join(value for value in (job, task) if value)
    return f"{stamp}-{suffix}" if suffix else f"{stamp}-{os.urandom(4).hex()}"


def tensor_payload_bytes(value: Any) -> int:
    """Estimate serialized tensor payload without first writing it to disk."""

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a project dependency
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(tensor_payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(tensor_payload_bytes(item) for item in value)
    return 0


def scratch_binary_directory(kind: str, identity: Mapping[str, Any]) -> Path:
    if kind not in {"checkpoints", "features"}:
        raise ValueError(f"Unsupported scratch artifact kind: {kind!r}")
    return (scratch_root() / kind / "Spur_SpLiCE" / _safe_name(str(identity["study"]), "study")
            / f"seed_{int(identity['seed']):02d}" / _safe_name(str(identity["arm"]), "arm")
            / _safe_name(str(identity["attempt_id"]), "attempt"))


def binary_destination(local_path: str | Path, payload_bytes: int, *, kind: str, identity: Mapping[str, Any] | None) -> Path:
    """Route only binary SSL/probe payloads larger than 10 MiB to scratch."""

    path = Path(local_path)
    if path.suffix.lower() not in BINARY_SUFFIXES or payload_bytes <= BINARY_SIZE_THRESHOLD:
        return path
    if identity is None:
        raise ValueError("Large binary artifacts require study/seed/arm/attempt identity")
    return scratch_binary_directory(kind, identity) / path.name


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_uri(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return "project://" + path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        pass
    try:
        return "artifact://" + path.relative_to(scratch_root().resolve()).as_posix()
    except ValueError:
        return path.as_posix()
