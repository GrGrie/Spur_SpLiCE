"""Promote an extracted pre-unification archive into canonical Git records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from splice.artifacts import atomic_write_json, sha256_file


_SEED = re.compile(r"^seed_?(\d+)$", re.IGNORECASE)
_SAFE = re.compile(r"[^a-z0-9]+")
_REPORT_WORDS = (
    "audit", "confirmation", "diagnosis", "discovery", "edge_overlap",
    "fingerprint", "integrity", "inventory", "paired_deltas", "preflight",
    "report", "results", "selection", "summary",
)


def safe_name(value: str) -> str:
    return _SAFE.sub("_", value.lower()).strip("_")


def infer_identity(relative_status: Path) -> dict[str, Any]:
    parts = list(relative_status.parts)
    training = parts.index("training")
    prefix = parts[:training]
    matches = [(index, _SEED.fullmatch(part)) for index, part in enumerate(prefix)]
    matches = [(index, match) for index, match in matches if match]
    if len(matches) != 1:
        raise ValueError(f"Expected one seed component in {relative_status}")
    seed_index, match = matches[0]
    assert match is not None
    if seed_index == 0 or seed_index == len(prefix) - 1:
        raise ValueError(f"Cannot infer study and arm from {relative_status}")
    study_parts = prefix[:seed_index]
    arm_parts = prefix[seed_index + 1:]
    return {
        "study": safe_name("_".join(study_parts)),
        "seed": int(match.group(1)),
        "arm": safe_name("_".join(arm_parts)),
        "attempt_id": safe_name(parts[training + 1]),
    }


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact(path: Path, root: Path, *, kind: str, state: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "stage": "legacy_recovery",
        "uri": "artifact://" + path.resolve().relative_to(root.resolve()).as_posix(),
        "storage_path": path.resolve().as_posix(),
        "availability": "cluster-only",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "retention_state": state,
        "present_at_finish": True,
    }


def build_run_record(status_path: Path, source: Path, scratch_root: Path) -> dict[str, Any]:
    relative_status = status_path.relative_to(source)
    identity = infer_identity(relative_status)
    run_dir = status_path.parent
    status = _load(status_path)
    args_path = run_dir / "args.json"
    config = _load(args_path) if args_path.is_file() else {}
    probes = []
    for probe_path in run_dir.glob("probe_features_epoch_*_ds_train_val.json"):
        try:
            probe = _load(probe_path)
            probes.append((int(probe.get("ssl_epoch", 0)), probe_path, probe))
        except (ValueError, json.JSONDecodeError):
            continue
    probes.sort(key=lambda item: item[0])
    metrics = [
        {
            "stage": "linear_probe",
            "step": step,
            "values": probe.get("metrics", {}),
            "group_metrics": probe.get("group_metrics", {}),
            "convergence": probe.get("convergence", {}),
            "source_sha256": sha256_file(path),
        }
        for step, path, probe in probes
    ]
    artifacts = []
    relative_run = run_dir.relative_to(source)
    checkpoint = scratch_root / "checkpoints" / "Spur_SpLiCE" / "legacy" / relative_run / "last.pth"
    if checkpoint.is_file():
        artifacts.append(_artifact(checkpoint, scratch_root, kind="ssl_checkpoint", state="final"))
    if probes:
        final_tensor = probes[-1][1].with_suffix(".pt")
        if final_tensor.is_file():
            artifacts.append(_artifact(final_tensor, scratch_root, kind="probe_features", state="recovery"))
    manifest = {"legacy_recovery": True, "original_path": relative_status.as_posix()}
    final_probe = probes[-1][2] if probes else {}
    modified = datetime.fromtimestamp(status_path.stat().st_mtime, timezone.utc).isoformat()
    return {
        "schema": "run-record-v1",
        "identity": identity,
        "status": status.get("status", "incomplete"),
        "timestamps": {"updated_at": modified, "finished_at": modified},
        "source": {
            "migration": "pre-unification-archive-recovery",
            "archive_uri": "artifact://legacy_archives/Spur_SpLiCE/pre-unification.tar.gz",
            "original_status_sha256": sha256_file(status_path),
        },
        "config": config,
        "manifest": {"definition": manifest, "sha256": _sha256_json(manifest)},
        "runtime": config.get("runtime_versions", {}),
        "slurm": {},
        "wandb": status.get("run_identity", {}).get("wandb", {}),
        "metrics": metrics,
        "final_metrics": final_probe.get("metrics", {}),
        "cleanup": status.get("cleanup", {}),
        "artifacts": artifacts,
    }


def _metric(record: dict[str, Any], name: str) -> float | None:
    value = record.get("final_metrics", {}).get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _study_report(study: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    arms: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        arms[record["identity"]["arm"]].append(record)
    summary = {}
    keys = {
        "average_accuracy": "Average over last 10 linear val acc",
        "worst_group_accuracy": "Average over last 10 linear val worst-group acc",
    }
    for arm, arm_records in sorted(arms.items()):
        summary[arm] = {}
        for label, key in keys.items():
            values = [value for record in arm_records if (value := _metric(record, key)) is not None]
            if values:
                summary[arm][label] = {
                    "count": len(values),
                    "mean": mean(values),
                    "sd": stdev(values) if len(values) > 1 else 0.0,
                }
    complete = all(record["status"] == "complete" for record in records)
    return {
        "schema": "study-results-v1",
        "study": study,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete else "partial",
        "legacy_snapshot": True,
        "matrix": {
            "observed": len(records),
            "successful": [record["identity"] for record in records if record["status"] == "complete"],
            "failed_or_incomplete": [record["identity"] for record in records if record["status"] != "complete"],
        },
        "summary": summary,
        "runs": records,
    }


def _archive_manifest(source: Path, archive: Path, scratch_root: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.resolve() == archive.resolve():
            continue
        entries.append({
            "path": path.relative_to(source).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "suffix": path.suffix.lower() or "<none>",
        })
    archive_members = 0
    if archive.is_file():
        with tarfile.open(archive, "r:gz") as bundle:
            archive_members = sum(member.isfile() for member in bundle.getmembers())
    suffix_counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        suffix_counts[entry["suffix"]] += 1
    return {
        "schema": "legacy-archive-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recovered_from_extracted_archive": True,
        "source_uri": "artifact://" + source.resolve().relative_to(scratch_root.resolve()).as_posix(),
        "archive_uri": "artifact://" + archive.resolve().relative_to(scratch_root.resolve()).as_posix(),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "archive_file_count": archive_members,
        "file_count": len(entries),
        "source_bytes": sum(entry["bytes"] for entry in entries),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "files": entries,
        "source_deleted_after_verification": False,
    }


def _promote_interpretive_files(source: Path, output_root: Path) -> int:
    count = 0
    reports_root = output_root / "reports" / "legacy_analyses"
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            continue
        relative = path.relative_to(source)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & {"training", "wandb", "logs"}:
            continue
        if path.suffix.lower() not in {".json", ".csv", ".yaml", ".yml", ".txt"}:
            continue
        if not any(word in path.stem.lower() for word in _REPORT_WORDS):
            continue
        destination = reports_root.joinpath(*(safe_name(part) for part in relative.parts[:-1]), path.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        count += 1
    return count


def promote(
    source: Path,
    output_root: Path,
    scratch_root: Path,
    archive: Path,
    *,
    rebuild_archive_manifest: bool = False,
) -> dict[str, Any]:
    records_by_study: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for status_path in sorted(source.rglob("run_status.json")):
        record = build_run_record(status_path, source, scratch_root)
        identity = record["identity"]
        destination = (
            output_root / "seeds" / identity["study"] / f"seed_{identity['seed']:02d}"
            / identity["arm"] / identity["attempt_id"]
        )
        atomic_write_json(destination / "run.json", record)
        atomic_write_json(destination / "command.json", {"recovered_config": record["config"]})
        atomic_write_json(destination / "execution.json", {
            "source": record["source"], "timestamps": record["timestamps"], "status": record["status"]
        })
        records_by_study[identity["study"]].append(record)
    for study, records in records_by_study.items():
        atomic_write_json(output_root / "reports" / study / "results.json", _study_report(study, records))
    promoted_reports = _promote_interpretive_files(source, output_root)
    result = {
        "runs": sum(len(records) for records in records_by_study.values()),
        "studies": len(records_by_study),
        "interpretive_files": promoted_reports,
        "archive_manifest_rebuilt": rebuild_archive_manifest,
    }
    if rebuild_archive_manifest:
        manifest = _archive_manifest(source, archive, scratch_root)
        atomic_write_json(output_root / "reports" / "legacy_archive" / "pre_unification.json", manifest)
        result["archive_files"] = manifest["file_count"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--scratch-root", type=Path, default=Path("/scratch/xar68reb/CoSpRo"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--rebuild-archive-manifest",
        action="store_true",
        help="Hash the complete extracted payload and tarball; optional and potentially slow.",
    )
    args = parser.parse_args()
    archive = args.archive or args.source / "pre-unification.tar.gz"
    result = promote(
        args.source.resolve(),
        args.output_root.resolve(),
        args.scratch_root.resolve(),
        archive.resolve(),
        rebuild_archive_manifest=args.rebuild_archive_manifest,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
