"""Build CoSpRo teacher graphs from one concept-group artifact or a sweep directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from splice.cospro import (
    GROUPING_CONFIG_FIELDS,
    CrpAuditConfig,
    build_teacher_graph,
    load_concept_groups_json,
)
from splice.cospro_reporting import render_teacher_graph_report
from splice.graph_io import save_graph_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Concept-group path does not exist: {path}")
    artifacts = sorted(path.rglob("concept_groups.json"))
    if not artifacts:
        raise FileNotFoundError(f"No concept_groups.json artifacts found under {path}")
    return artifacts


def audit_config_name(config: CrpAuditConfig) -> str:
    values = {key: value for key, value in asdict(config).items() if key not in GROUPING_CONFIG_FIELDS}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    quantile = str(config.null_quantile).replace(".", "p")
    return f"audit_seed_{config.seed}_nullq_{quantile}_trials_{config.null_trials}_{digest}"


# Historical private name retained for local callers while the pipeline uses
# the public path helper below.
_audit_name = audit_config_name


def teacher_graph_path(concept_groups_path: Path, config: CrpAuditConfig) -> Path:
    """Return the graph path produced for an exact concept-group artifact."""

    return concept_groups_path.parent / "teacher_graphs" / audit_config_name(config) / "teacher_graph.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splice-dataset-cache",
        required=True,
        type=Path,
        help="Frozen SpLiCE dataset cache (.pt).",
    )
    parser.add_argument(
        "--concept-groups",
        required=True,
        type=Path,
        help="A concept_groups.json file or a directory containing sweep artifacts.",
    )
    parser.add_argument("--config", help="Optional JSON object overriding teacher-audit settings.")
    parser.add_argument("--seed", type=int, help="Override the null-control seed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_values = json.loads(args.config) if args.config else {}
    unknown = set(config_values).difference(CrpAuditConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown CoSpRo audit settings: {sorted(unknown)}")
    grouping_overrides = set(config_values).intersection(GROUPING_CONFIG_FIELDS)
    if grouping_overrides:
        raise ValueError(
            "Grouping settings come from concept_groups.json and cannot be overridden during audit: "
            f"{sorted(grouping_overrides)}"
        )
    if args.seed is not None:
        config_values["seed"] = args.seed
    config = CrpAuditConfig(**config_values)
    cache = torch.load(args.splice_dataset_cache, map_location="cpu", weights_only=True)
    artifacts = _artifact_paths(args.concept_groups)
    for artifact_path in artifacts:
        concept_groups = load_concept_groups_json(artifact_path)
        source = {
            "path": str(artifact_path.resolve()),
            "sha256": _sha256(artifact_path),
            "artifact": concept_groups["artifact"],
            "concept_groups_version": concept_groups["concept_groups_version"],
        }
        graph = build_teacher_graph(cache, concept_groups, config, source)
        output_directory = teacher_graph_path(artifact_path, config).parent
        json_path = save_graph_json(graph, output_directory / "teacher_graph.json")
        html_path = render_teacher_graph_report(graph, output_directory / "teacher_graph.html")
        print(f"[INFO] Wrote {json_path}", flush=True)
        print(f"[INFO] Wrote {html_path}", flush=True)
    print(f"[INFO] Generated {len(artifacts)} teacher graph(s) from explicit concept-group artifacts.")


if __name__ == "__main__":
    main()
