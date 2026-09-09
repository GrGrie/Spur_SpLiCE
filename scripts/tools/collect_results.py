"""Collect one manifest's run records into a Git-friendly results JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from experiments.runner import load_manifest, matrix
from splice.artifacts import OUTPUT_ROOT, PROJECT_ROOT, atomic_write_json, report, scratch_root, sha256_file


def _artifact_path(uri: str) -> Path | None:
    if uri.startswith("artifact://"):
        return scratch_root() / uri.removeprefix("artifact://")
    if uri.startswith("project://"):
        return PROJECT_ROOT / uri.removeprefix("project://")
    return None


def verify_attestations(record: dict[str, Any]) -> list[str]:
    errors = []
    for artifact in record.get("artifacts", []):
        if artifact.get("retention_state") not in {"final", "promoted", "retained"}:
            continue
        path = _artifact_path(str(artifact.get("uri", "")))
        if path is None or not path.is_file():
            errors.append(f"missing:{artifact.get('uri')}")
            continue
        if path.stat().st_size != artifact.get("bytes"):
            errors.append(f"size:{artifact.get('uri')}")
        elif sha256_file(path) != artifact.get("sha256"):
            errors.append(f"sha256:{artifact.get('uri')}")
    return errors


def _latest_records(study: str) -> dict[tuple[int, str], dict[str, Any]]:
    root = OUTPUT_ROOT / "seeds" / study
    records: dict[tuple[int, str], dict[str, Any]] = {}
    if not root.exists():
        return records
    for path in root.glob("seed_*/*/*/run.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            identity = record["identity"]
            key = (int(identity["seed"]), str(identity["arm"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        previous = records.get(key)
        updated = str(record.get("timestamps", {}).get("updated_at", ""))
        if previous is None or updated >= str(previous.get("timestamps", {}).get("updated_at", "")):
            record["record_uri"] = "project://" + path.relative_to(PROJECT_ROOT).as_posix()
            records[key] = record
    return records


def _metric(record: dict[str, Any], names: tuple[str, ...]) -> float | None:
    metrics = record.get("final_metrics", {})
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _summaries(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_arm[str(record["identity"]["arm"])].append(record)
    summary = {}
    for arm, arm_records in sorted(by_arm.items()):
        values = {
            "average_accuracy": [value for record in arm_records if (value := _metric(record, ("Average over last 10 linear val acc", "Linear val acc"))) is not None],
            "worst_group_accuracy": [value for record in arm_records if (value := _metric(record, ("Average over last 10 linear val worst-group acc", "Linear val worst-group acc"))) is not None],
        }
        summary[arm] = {
            name: {"count": len(items), "mean": mean(items), "sd": stdev(items) if len(items) > 1 else 0.0}
            for name, items in values.items()
            if items
        }
    return summary


def collect(manifest_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    study = str(manifest["name"])
    found = _latest_records(study)
    successful = []
    failed = []
    incomplete = []
    missing = []
    selected = []
    for seed, arm in matrix(manifest):
        key = (seed, arm)
        record = found.get(key)
        label = {"seed": seed, "arm": arm}
        if record is None:
            missing.append(label)
            continue
        errors = verify_attestations(record)
        if record.get("status") == "complete" and not errors:
            successful.append(label)
        elif record.get("status") == "failed":
            failed.append({**label, "error": record.get("error"), "attestation_errors": errors})
        else:
            incomplete.append({**label, "status": record.get("status"), "attestation_errors": errors})
        selected.append(record)
    complete = not failed and not incomplete and not missing and len(successful) == len(matrix(manifest))
    payload = {
        "schema": "study-results-v1",
        "study": study,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete else "partial",
        "manifest": manifest,
        "matrix": {"expected": len(matrix(manifest)), "successful": successful, "failed": failed, "incomplete": incomplete, "missing": missing},
        "summary": _summaries([record for record in selected if record.get("status") == "complete"]),
        "runs": selected,
    }
    atomic_write_json(output_path or report(study, "results.json"), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = collect(args.manifest, args.output)
    print(f"study={payload['study']} status={payload['status']} successful={len(payload['matrix']['successful'])}")


if __name__ == "__main__":
    main()
