"""Discover and safely migrate legacy result trees into the current layout."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from splice.artifacts import (
    BINARY_SIZE_THRESHOLD,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    artifact_uri,
    atomic_write_json,
    scratch_root,
    sha256_file,
)


MARKERS = {"seeds", "shared", "reports", "reference"}


def discover_result_root(source: str | Path) -> Path:
    source = Path(source).resolve()
    candidates = []
    for candidate in (source, *[path for path in source.rglob("*") if path.is_dir()]):
        try:
            relative_depth = len(candidate.relative_to(source).parts)
        except ValueError:
            continue
        if relative_depth > 3:
            continue
        children = {path.name for path in candidate.iterdir() if path.is_dir()}
        if len(children & MARKERS) >= 2:
            candidates.append(candidate)
    if not candidates and source.name.lower() == "outputs":
        legacy_entries = [path for path in source.iterdir() if path.name not in {"SLURM", "README.md"}]
        if legacy_entries:
            return source
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates) or "none"
        raise ValueError(f"Expected exactly one result root; found {rendered}")
    return candidates[0]


def _heavy_binary_kind(path: Path, relative: Path) -> str | None:
    if path.stat().st_size <= BINARY_SIZE_THRESHOLD:
        return None
    suffix = path.suffix.lower()
    name = path.name.lower()
    parts = {part.lower() for part in relative.parts}
    if suffix in {".pth", ".ckpt"}:
        return "checkpoints"
    if suffix != ".pt":
        return None
    if name.startswith("probe_features_epoch_"):
        return "features"
    checkpoint_name = name == "last.pt" or name.startswith(("epoch_", "checkpoint"))
    checkpoint_context = bool(parts & {"ssl", "training", "linear_probe", "linear-probe", "probe"})
    return "checkpoints" if checkpoint_name or checkpoint_context else None


def _normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        scratch_prefix = scratch_root().as_posix().rstrip("/") + "/"
        if normalized.startswith(scratch_prefix):
            return "artifact://" + normalized[len(scratch_prefix):]
        marker = "/outputs/"
        if marker in normalized and (normalized.startswith("/") or ":/" in normalized):
            return "project://outputs/" + normalized.split(marker, 1)[1]
    return value


def migration_plan(source: str | Path) -> list[dict[str, Any]]:
    root = discover_result_root(source)
    root_children = {path.name for path in root.iterdir() if path.is_dir()}
    flat_legacy_root = len(root_children & MARKERS) < 2
    plan = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        binary_kind = _heavy_binary_kind(path, relative)
        heavy_binary = binary_kind is not None
        if heavy_binary:
            destination = scratch_root() / binary_kind / "Spur_SpLiCE" / "legacy" / relative
        elif flat_legacy_root and relative.parts[0] not in MARKERS | {"SLURM", "README.md"}:
            destination = OUTPUT_ROOT / "shared" / "legacy" / relative
        else:
            destination = OUTPUT_ROOT / relative
        plan.append({
            "root": root,
            "source": path,
            "destination": destination,
            "bytes": path.stat().st_size,
            "heavy_binary": heavy_binary,
        })
    return plan


def _legacy_identity(path: Path) -> dict[str, Any] | None:
    parts = list(path.parts)
    try:
        seeds_index = parts.index("seeds")
    except ValueError:
        return None
    tail = parts[seeds_index + 1:]
    if len(tail) < 5:
        return None
    if tail[0].startswith("seed_"):
        seed_token, study = tail[0], tail[1]
        before_training = tail[2:tail.index("training")] if "training" in tail else tail[2:-2]
    elif len(tail) > 1 and tail[1].startswith("seed_"):
        study, seed_token = tail[0], tail[1]
        before_training = tail[2:tail.index("training")] if "training" in tail else tail[2:-2]
    else:
        return None
    try:
        seed = int(seed_token.removeprefix("seed_"))
    except ValueError:
        return None
    arm = "-".join(before_training) if before_training else "training"
    return {"study": study, "seed": seed, "arm": arm, "attempt_id": path.parent.name}


def backfill_run_records(plan: list[dict[str, Any]]) -> int:
    destination_by_source = {item["source"].resolve(): item["destination"] for item in plan}
    created = 0
    for item in plan:
        status_source = item["source"]
        if status_source.name != "run_status.json":
            continue
        identity = _legacy_identity(status_source)
        if identity is None:
            continue
        source_run_dir = status_source.parent
        status = json.loads(status_source.read_text(encoding="utf-8"))
        args_path = source_run_dir / "args.json"
        config = json.loads(args_path.read_text(encoding="utf-8")) if args_path.is_file() else {}
        probe_results = []
        for probe_path in source_run_dir.glob("probe_features_epoch_*.json"):
            try:
                probe = json.loads(probe_path.read_text(encoding="utf-8"))
                probe_results.append(probe)
            except json.JSONDecodeError:
                continue
        probe_results.sort(key=lambda result: int(result.get("ssl_epoch", 0)))
        artifacts = []
        for source_path, destination in destination_by_source.items():
            if source_run_dir not in source_path.parents:
                continue
            suffix = source_path.suffix.lower()
            if suffix not in {".pth", ".pt", ".json"}:
                continue
            if not destination.is_file():
                continue
            in_scratch = str(destination.resolve()).startswith(str(scratch_root().resolve()))
            artifacts.append({
                "kind": "ssl_checkpoint" if suffix == ".pth" else ("probe_features" if suffix == ".pt" else "result_json"),
                "stage": "legacy",
                "epoch": next((result.get("ssl_epoch") for result in probe_results if str(result.get("ssl_epoch")) in source_path.name), None),
                "uri": artifact_uri(destination),
                "storage_path": destination.resolve().as_posix() if in_scratch else None,
                "availability": "cluster-only" if in_scratch else "repository-copy",
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "retention_state": "final" if source_path.name == "last.pth" else "retained",
                "present_at_finish": True,
            })
        final_probe = probe_results[-1] if probe_results else {}
        record = {
            "schema": "run-record-v1",
            "identity": identity,
            "status": status.get("status", "incomplete"),
            "timestamps": {"updated_at": datetime.fromtimestamp(status_source.stat().st_mtime, timezone.utc).isoformat()},
            "source": {"migration": "legacy-output-backfill", "original_status": _normalize_json(status)},
            "config": _normalize_json(config),
            "manifest": {"definition": {}, "sha256": None},
            "runtime": config.get("runtime_versions", {}),
            "slurm": {},
            "wandb": status.get("run_identity", {}).get("wandb", {}),
            "metrics": [
                {"stage": "linear_probe_final", "step": int(result.get("ssl_epoch", 0)), "values": result}
                for result in probe_results
            ],
            "final_metrics": final_probe.get("metrics", {}),
            "cleanup": status.get("cleanup", {}),
            "artifacts": artifacts,
        }
        if status.get("error"):
            record["error"] = status["error"]
        target = (
            OUTPUT_ROOT / "seeds" / str(identity["study"]) / f"seed_{int(identity['seed']):02d}"
            / str(identity["arm"]) / str(identity["attempt_id"]) / "run.json"
        )
        atomic_write_json(target, record)
        created += 1
    return created


def apply_migration(plan: list[dict[str, Any]], *, delete_source: bool = False) -> dict[str, Any]:
    copied = []
    for item in plan:
        source = item["source"]
        destination = item["destination"]
        source_sha256 = sha256_file(source)
        if source.resolve() == destination.resolve():
            copied.append({**item, "source_sha256": source_sha256, "destination_sha256": source_sha256, "unchanged": True})
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            normalized_payload = _normalize_json(payload)
            if destination.exists():
                existing_payload = json.loads(destination.read_text(encoding="utf-8"))
                if existing_payload != normalized_payload:
                    raise FileExistsError(f"Refusing to overwrite different JSON: {destination}")
            else:
                atomic_write_json(destination, normalized_payload)
        else:
            if destination.exists() and sha256_file(destination) != source_sha256:
                raise FileExistsError(f"Refusing to overwrite different artifact: {destination}")
            if not destination.exists():
                shutil.copy2(source, destination)
        if not destination.is_file():
            raise RuntimeError(f"Migration did not create {destination}")
        destination_sha256 = sha256_file(destination)
        if source.suffix.lower() != ".json" and source_sha256 != destination_sha256:
            raise RuntimeError(f"Migration checksum mismatch: {source} -> {destination}")
        copied.append({
            **item,
            "source_sha256": source_sha256,
            "destination_sha256": destination_sha256,
            "unchanged": False,
        })
    backfilled_records = backfill_run_records(copied)
    if delete_source:
        for item in copied:
            if item["unchanged"]:
                continue
            destination = item["destination"]
            if destination.stat().st_size != item["bytes"] and item["source"].suffix.lower() != ".json":
                raise RuntimeError(f"Refusing deletion after size mismatch: {destination}")
            if item["source"].suffix.lower() != ".json" and item["source_sha256"] != item["destination_sha256"]:
                raise RuntimeError(f"Refusing deletion after checksum mismatch: {destination}")
        for item in copied:
            if not item["unchanged"]:
                item["source"].unlink()
        for root in {item["root"] for item in copied}:
            for directory in sorted(
                (path for path in root.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                try:
                    directory.rmdir()
                except OSError:
                    pass
    return {"files": len(copied), "bytes": sum(item["bytes"] for item in copied), "backfilled_run_records": backfilled_records, "deleted_source": delete_source}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--mode", choices=["scan", "dry-run", "apply"], default="scan")
    parser.add_argument("--delete-source-after-verify", action="store_true")
    args = parser.parse_args()
    root = discover_result_root(args.source)
    plan = migration_plan(root)
    print(json.dumps({"root": str(root), "files": len(plan), "bytes": sum(item["bytes"] for item in plan), "heavy_files": sum(item["heavy_binary"] for item in plan)}, indent=2))
    if args.mode == "dry-run":
        for item in plan:
            print(f"{item['source']} -> {item['destination']}")
    elif args.mode == "apply":
        print(json.dumps(apply_migration(plan, delete_source=args.delete_source_after_verify), indent=2))


if __name__ == "__main__":
    main()
