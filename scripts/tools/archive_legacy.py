"""Archive migrated legacy outputs outside Git with a verified manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from splice.artifacts import artifact_uri, atomic_write_json, report, resolve_output_root, scratch_root, sha256_file


CHUNK_SIZE = 1024 * 1024


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
        digest.update(chunk)
    return digest.hexdigest()


def inventory(source: str | Path) -> list[dict[str, Any]]:
    source = Path(source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Legacy directory does not exist: {source}")
    entries = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Refusing to archive symlink: {path}")
        if not path.is_file():
            continue
        entries.append({
            "path": path.relative_to(source).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "suffix": path.suffix.lower() or "<none>",
        })
    if not entries:
        raise ValueError(f"Legacy directory is empty: {source}")
    return entries


def _verify_archive(archive: Path, entries: list[dict[str, Any]]) -> None:
    expected = {entry["path"]: entry for entry in entries}
    with tarfile.open(archive, "r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers() if member.isfile()}
        if set(members) != set(expected):
            raise RuntimeError("Archive member list does not match the legacy inventory")
        for name, entry in expected.items():
            member = members[name]
            if member.size != entry["bytes"]:
                raise RuntimeError(f"Archive size mismatch: {name}")
            stream = bundle.extractfile(member)
            if stream is None or _sha256_stream(stream) != entry["sha256"]:
                raise RuntimeError(f"Archive SHA-256 mismatch: {name}")


def archive_legacy(
    source: str | Path,
    archive: str | Path,
    manifest_path: str | Path,
    *,
    delete_source: bool = False,
) -> dict[str, Any]:
    source = Path(source).resolve()
    archive = Path(archive).resolve()
    manifest_path = Path(manifest_path).resolve()
    if archive == source or source in archive.parents:
        raise ValueError("Archive must be stored outside the legacy source directory")
    if manifest_path == source or source in manifest_path.parents:
        raise ValueError("Manifest must be stored outside the legacy source directory")
    entries = inventory(source)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {archive}")
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    try:
        with tarfile.open(temporary, "w:gz") as bundle:
            for entry in entries:
                bundle.add(source / entry["path"], arcname=entry["path"], recursive=False)
        _verify_archive(temporary, entries)
        temporary.replace(archive)
    finally:
        if temporary.exists():
            temporary.unlink()

    suffix_counts: dict[str, int] = {}
    for entry in entries:
        suffix_counts[entry["suffix"]] = suffix_counts.get(entry["suffix"], 0) + 1
    payload = {
        "schema": "legacy-archive-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source.as_posix(),
        "archive_uri": artifact_uri(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "file_count": len(entries),
        "source_bytes": sum(entry["bytes"] for entry in entries),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "files": entries,
        "source_deleted_after_verification": False,
    }
    atomic_write_json(manifest_path, payload)

    if delete_source:
        for entry in entries:
            path = source / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"Source changed after archiving; refusing deletion: {path}")
        for entry in entries:
            (source / entry["path"]).unlink()
        for directory in sorted(
            (path for path in source.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.rmdir()
        source.rmdir()
        payload["source_deleted_after_verification"] = True
        atomic_write_json(manifest_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--apply", action="store_true", help="Create and verify the archive.")
    parser.add_argument("--delete-source-after-verify", action="store_true")
    args = parser.parse_args()
    if args.delete_source_after_verify and not args.apply:
        parser.error("--delete-source-after-verify requires --apply")

    source = args.source or resolve_output_root() / "shared" / "legacy"
    entries = inventory(source)
    summary = {"source": str(source.resolve()), "file_count": len(entries), "bytes": sum(entry["bytes"] for entry in entries)}
    if not args.apply:
        print(json.dumps(summary, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    archive = args.archive or scratch_root() / "legacy_archives" / "Spur_SpLiCE" / f"legacy-{stamp}.tar.gz"
    manifest_path = args.manifest or report("legacy-archive", f"legacy-{stamp}.json")
    payload = archive_legacy(source, archive, manifest_path, delete_source=args.delete_source_after_verify)
    print(json.dumps({**summary, "archive_uri": payload["archive_uri"], "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
