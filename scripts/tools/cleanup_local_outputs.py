"""Summarize legacy Windows results and remove only disposable local tensors.

The default mode is a dry run.  ``--apply`` writes a compact, Git-trackable JSON
report and deletes files only when the report records why the file is disposable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BINARY_SUFFIXES = {".pt", ".pth", ".ckpt"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_MARKERS = ("cache", "features", "embeddings", "scores")
SUMMARY_KEYS = (
    "accuracy",
    "loss",
    "score",
    "metric",
    "epoch",
    "seed",
    "status",
    "completed",
    "coverage",
    "groups",
    "edges",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _io_path(path: Path) -> Path:
    """Use the Win32 long-path namespace without leaking it into manifests."""
    rendered = str(path.resolve())
    if os.name == "nt" and not rendered.startswith("\\\\?\\"):
        return Path("\\\\?\\" + rendered)
    return path


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _project_uri(path: Path) -> str:
    rendered = str(path)
    if rendered.startswith("\\\\?\\"):
        rendered = rendered[4:]
    return "project://" + Path(rendered).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _normalize_string(value: str) -> str:
    normalized = value.replace("\\", "/")
    project_prefix = PROJECT_ROOT.resolve().as_posix().rstrip("/") + "/"
    if normalized.lower().startswith(project_prefix.lower()):
        return "project://" + normalized[len(project_prefix):]
    outputs_marker = "/outputs/"
    if outputs_marker in normalized and (normalized.startswith("/") or ":/" in normalized):
        return "project://outputs/" + normalized.split(outputs_marker, 1)[1]
    scratch_prefix = "/scratch/xar68reb/CoSpRo/"
    if normalized.startswith(scratch_prefix):
        return "artifact://" + normalized[len(scratch_prefix):]
    return normalized


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, str):
        return _normalize_string(value)
    return value


def _compact_evidence(plan: dict[str, Any]) -> None:
    referenced: set[str] = set()
    for item in plan["delete"]:
        candidates = item.get("evidence_uris", [])
        if not candidates:
            continue
        exact = item["uri"].rsplit(".", 1)[0] + ".json"
        selected = [uri for uri in candidates if uri == exact or uri.endswith("/run_status.json")]
        probes = []
        for uri in candidates:
            match = re.search(r"/probe_features_epoch_(\d+).*\.json$", uri)
            if match:
                probes.append((int(match.group(1)), uri))
        if probes:
            selected.append(max(probes)[1])
        if not selected:
            selected.append(candidates[0])
        item["evidence_uris"] = list(dict.fromkeys(selected))
        referenced.update(item["evidence_uris"])
    compact = {}
    for uri in sorted(referenced):
        if uri not in plan["evidence"]:
            continue
        record = plan["evidence"][uri]
        record["summary"] = dict(list(record.get("summary", {}).items())[:40])
        compact[uri] = record
    plan["evidence"] = compact


def _coerce_csv(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return _normalize_string(value)


def _csv_results(source: Path) -> tuple[list[dict[str, Any]], dict[Path, list[dict[str, Any]]]]:
    exported: list[dict[str, Any]] = []
    by_path: dict[Path, list[dict[str, Any]]] = {}
    for path in sorted(source.rglob("results.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = [{key: _coerce_csv(value) for key, value in row.items()} for row in csv.DictReader(handle)]
        by_path[path.resolve()] = rows
        exported.append({"source_uri": _project_uri(path), "rows": rows})
    return exported, by_path


def _small_json_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if path.stat().st_size > 5 * 1024 * 1024:
        return summary
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return summary

    def visit(value: Any, prefix: str = "") -> None:
        if len(summary) >= 80:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            key = prefix.lower()
            if any(marker in key for marker in SUMMARY_KEYS):
                summary[prefix] = _normalize_string(value) if isinstance(value, str) else value

    visit(payload)
    return summary


def _json_evidence(binary: Path) -> list[Path]:
    directory = binary.parent
    candidates = [directory / f"{binary.stem}.json"]
    candidates.extend(directory.glob("probe_features_epoch_*.json"))
    status = directory / "run_status.json"
    if status.is_file():
        try:
            if json.loads(status.read_text(encoding="utf-8")).get("status") == "complete":
                candidates.append(status)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    return sorted({path.resolve() for path in candidates if path.is_file()})


def _is_checkpoint(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() in {".pth", ".ckpt"}:
        return True
    return name == "last.pt" or name.startswith(("epoch_", "checkpoint"))


def _find_results_csv(path: Path, source: Path) -> Path | None:
    current = path.parent
    while current == source or source in current.parents:
        candidate = current / "results.csv"
        if candidate.is_file():
            return candidate.resolve()
        if current == source:
            break
        current = current.parent
    return None


def _legacy_variant(binary: Path, args: dict[str, Any]) -> str | None:
    parts = binary.parts
    if "windows_crpv4_ablation" in parts:
        return parts[parts.index("training") + 1]
    if "windows_crpv4_diverse" in parts:
        return parts[parts.index("windows_crpv4_diverse") + 1]
    if "windows_splice_only_ablation" in parts or "windows_splice_only_ablation_seed1" in parts:
        if int(args.get("epochs", 0)) < 100:
            return None
        graph = str(args.get("crp_teacher_graph", "")).lower()
        if not graph:
            return "simclr"
        return "cobalt_crpv3" if "cobalt" in graph else "splice_only_crp"
    return None


def _csv_evidence(binary: Path, source: Path, csv_rows: dict[Path, list[dict[str, Any]]]) -> dict[str, Any] | None:
    csv_path = _find_results_csv(binary, source)
    args_path = binary.parent / "args.json"
    if csv_path is None or not args_path.is_file():
        return None
    try:
        args = json.loads(args_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    variant = _legacy_variant(binary, args)
    if variant is None:
        return None
    rows = csv_rows[csv_path]
    matches = [row for row in rows if row.get("variant", row.get("run")) == variant]
    if len(matches) != 1:
        return None
    return {"source_uri": _project_uri(csv_path), "result": matches[0]}


def build_plan(source: Path) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_dir() or source == PROJECT_ROOT.resolve():
        raise ValueError(f"Refusing unsafe source: {source}")
    try:
        source.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Source must be inside the project") from exc

    scan_source = _io_path(source)
    legacy_results, csv_rows = _csv_results(scan_source)
    evidence_catalog: dict[str, dict[str, Any]] = {}
    delete: list[dict[str, Any]] = []
    retain: list[dict[str, Any]] = []
    for binary in sorted(path for path in scan_source.rglob("*") if path.is_file() and path.suffix.lower() in BINARY_SUFFIXES):
        checkpoint = _is_checkpoint(binary)
        json_paths = _json_evidence(binary)
        csv_evidence = _csv_evidence(binary, scan_source, csv_rows) if checkpoint else None
        cache_like = (
            not checkpoint
            and binary.suffix.lower() == ".pt"
            and any(marker in binary.as_posix().lower() for marker in CACHE_MARKERS)
        )
        reasons = []
        evidence_uris = []
        for evidence in json_paths:
            uri = _project_uri(evidence)
            evidence_uris.append(uri)
            if uri not in evidence_catalog:
                evidence_catalog[uri] = {
                    "bytes": evidence.stat().st_size,
                    "sha256": _sha256_file(evidence),
                    "summary": _small_json_summary(evidence),
                }
        if json_paths:
            reasons.append("result-json-present")
        if csv_evidence is not None:
            reasons.append("result-promoted-from-csv")
        if cache_like:
            reasons.append("reproducible-cache")

        record: dict[str, Any] = {
            "uri": _project_uri(binary),
            "bytes": binary.stat().st_size,
            "kind": "checkpoint" if checkpoint else "tensor",
        }
        if reasons:
            record["reasons"] = reasons
            if evidence_uris:
                record["evidence_uris"] = evidence_uris
            if csv_evidence is not None:
                record["promoted_result"] = csv_evidence
            delete.append(record)
        else:
            record["reason"] = "no-result-evidence-or-ambiguous-tensor"
            retain.append(record)

    return {
        "schema": "local-output-cleanup-v1",
        "source_uri": _project_uri(source),
        "legacy_results": legacy_results,
        "evidence": evidence_catalog,
        "delete": delete,
        "retain": retain,
        "summary": {
            "delete_files": len(delete),
            "delete_bytes": sum(item["bytes"] for item in delete),
            "retain_files": len(retain),
            "retain_bytes": sum(item["bytes"] for item in retain),
        },
    }


def apply_plan(plan: dict[str, Any], report: Path) -> None:
    report = report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    current_delete = list(plan["delete"])
    now = datetime.now(timezone.utc).isoformat()
    if report.is_file():
        previous = json.loads(report.read_text(encoding="utf-8"))
        if previous.get("schema") == plan["schema"] and previous.get("deletion_applied") is True:
            plan["delete"] = list({item["uri"]: item for item in [*previous["delete"], *current_delete]}.values())
            plan["evidence"] = {**previous.get("evidence", {}), **plan["evidence"]}
            plan["legacy_results"] = list({
                item["source_uri"]: item for item in [*previous.get("legacy_results", []), *plan["legacy_results"]]
            }.values())
            plan["created_at"] = previous.get("created_at", now)
        else:
            plan["created_at"] = now
    else:
        plan["created_at"] = now
    plan["updated_at"] = now
    plan["summary"] = {
        "delete_files": len(plan["delete"]),
        "delete_bytes": sum(item["bytes"] for item in plan["delete"]),
        "retain_files": len(plan["retain"]),
        "retain_bytes": sum(item["bytes"] for item in plan["retain"]),
    }
    _compact_evidence(plan)
    plan = _normalize_value(plan)
    plan["deletion_applied"] = False
    _atomic_write_json(report, plan)
    for item in current_delete:
        path = _io_path(PROJECT_ROOT / item["uri"].removeprefix("project://"))
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"File changed after planning; refusing deletion: {path}")
    for item in current_delete:
        path = _io_path(PROJECT_ROOT / item["uri"].removeprefix("project://"))
        path.unlink()
    plan["deletion_applied"] = True
    _atomic_write_json(report, plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "shared" / "legacy" / "windows-main-pre-unification" / "outputs-root",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "local-cleanup" / "windows-pre-unification.json",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args.source)
    if args.apply:
        apply_plan(plan, args.report)
    print(json.dumps({**plan["summary"], "report": str(args.report), "applied": args.apply}, indent=2))


if __name__ == "__main__":
    main()
