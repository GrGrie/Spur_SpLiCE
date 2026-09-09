"""Atomic, local-first recording for reproducible experiment runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from splice.artifacts import PROJECT_ROOT, artifact_uri, atomic_write_json, scratch_root, sha256_file


SCHEMA = "run-record-v1"
_SECRET = re.compile(r"(token|secret|password|api[_-]?key)", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any, key: str = "") -> Any:
    if _SECRET.search(key):
        return "<redacted>"
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and (Path(value).is_absolute() or "/" in value or "\\" in value):
            normalized = value.replace("\\", "/")
            for root, prefix in ((PROJECT_ROOT, "project://"), (scratch_root(), "artifact://")):
                root_text = root.as_posix().rstrip("/") + "/"
                if normalized.casefold().startswith(root_text.casefold()):
                    return prefix + normalized[len(root_text):]
                try:
                    return prefix + Path(value).resolve().relative_to(root.resolve()).as_posix()
                except (OSError, ValueError):
                    pass
            path = Path(value)
            if path.is_absolute() or normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
                return "external://" + normalized.lstrip("/").replace(":", "")
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, key) for item in value]
    return str(value)


def portable_json(value: Any) -> Any:
    """Return a secret-redacted payload with portable path references."""

    return _json_safe(value)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    commit = run("rev-parse", "HEAD")
    status = run("status", "--short", "--untracked-files=no")
    return {
        "commit": commit or None,
        "dirty": bool(status),
        "dirty_digest": hashlib.sha256(status.encode("utf-8")).hexdigest() if status else None,
    }


class RunRecorder:
    """Own the lifecycle and integrity record for one training attempt."""

    def __init__(self, path: str | Path, *, identity: Mapping[str, Any], config: Mapping[str, Any], runtime: Mapping[str, Any] | None = None, manifest: Mapping[str, Any] | None = None) -> None:
        manifest_payload = _json_safe(manifest or {})
        self.path = Path(path)
        self.data: dict[str, Any] = {
            "schema": SCHEMA,
            "identity": _json_safe(identity),
            "status": "running",
            "timestamps": {"started_at": _utc_now(), "updated_at": _utc_now()},
            "source": _git_state(),
            "config": _json_safe(config),
            "manifest": {
                "definition": manifest_payload,
                "sha256": hashlib.sha256(
                    json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
            "runtime": {"hostname": socket.gethostname(), **_json_safe(runtime or {})},
            "slurm": {key: os.environ[key] for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID") if key in os.environ},
            "metrics": [],
            "artifacts": [],
        }
        self._write()

    @classmethod
    def open(cls, path: str | Path) -> "RunRecorder":
        recorder = cls.__new__(cls)
        recorder.path = Path(path)
        recorder.data = json.loads(recorder.path.read_text(encoding="utf-8"))
        if recorder.data.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported run record schema: {recorder.data.get('schema')!r}")
        return recorder

    def _write(self) -> None:
        self.data["timestamps"]["updated_at"] = _utc_now()
        atomic_write_json(self.path, self.data)

    def update_config(self, values: Mapping[str, Any]) -> None:
        self.data["config"].update(_json_safe(values))
        self._write()

    def set_wandb(self, identity: Mapping[str, Any] | None) -> None:
        self.data["wandb"] = _json_safe(identity or {})
        self._write()

    def log_metrics(self, stage: str, step: int, metrics: Mapping[str, Any]) -> None:
        self.data["metrics"].append({"stage": stage, "step": int(step), "values": _json_safe(metrics)})
        self._write()

    def register_artifact(self, path: str | Path, *, kind: str, stage: str, epoch: int | None = None, retention_state: str = "retained") -> dict[str, Any]:
        artifact_path = Path(path).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        try:
            artifact_path.relative_to(scratch_root().resolve())
            in_scratch = True
        except ValueError:
            in_scratch = False
        entry = {
            "kind": kind,
            "stage": stage,
            "epoch": epoch,
            "uri": artifact_uri(artifact_path),
            "storage_path": artifact_path.as_posix() if in_scratch else None,
            "availability": "cluster-only" if in_scratch else "repository-copy",
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256_file(artifact_path),
            "verified_at": _utc_now(),
            "retention_state": retention_state,
        }
        self.data["artifacts"].append(entry)
        self._write()
        return entry

    def record_promotion(self, *, label: str, reason: str, metrics: Mapping[str, Any], artifact_sha256: str) -> None:
        self.data.setdefault("promotions", []).append({
            "label": label,
            "reason": reason,
            "metric_snapshot": _json_safe(metrics),
            "artifact_sha256": artifact_sha256,
        })
        self._write()

    def _verify_retained_artifacts(self, *, strict: bool = True) -> None:
        for artifact in self.data["artifacts"]:
            uri = artifact["uri"]
            if uri.startswith("artifact://"):
                path = scratch_root() / uri.removeprefix("artifact://")
            elif uri.startswith("project://"):
                path = PROJECT_ROOT / uri.removeprefix("project://")
            else:
                path = Path(uri)
            artifact["present_at_finish"] = path.is_file()
            if not strict or artifact["retention_state"] not in {"final", "promoted", "retained"}:
                continue
            if not artifact["present_at_finish"] or path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
                raise RuntimeError(f"Artifact integrity verification failed: {uri}")

    def finish(self, *, final_metrics: Mapping[str, Any] | None = None, cleanup: Mapping[str, Any] | None = None) -> None:
        self._verify_retained_artifacts()
        self.data["status"] = "complete"
        self.data["final_metrics"] = _json_safe(final_metrics or {})
        self.data["cleanup"] = _json_safe(cleanup or {})
        self.data["timestamps"]["finished_at"] = _utc_now()
        self._write()

    def fail(self, error: BaseException, cleanup: Mapping[str, Any] | None = None) -> None:
        self._verify_retained_artifacts(strict=False)
        self.data["status"] = "failed"
        self.data["error"] = {"type": type(error).__name__, "message": str(error)}
        self.data["cleanup"] = _json_safe(cleanup or {})
        self.data["timestamps"]["finished_at"] = _utc_now()
        self._write()
