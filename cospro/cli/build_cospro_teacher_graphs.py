"""Build CoSpRo teacher graphs from one concept-group artifact or a sweep directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from cospro.pipeline.selection import ConceptTypeGate
from cospro.pipeline import (
    GROUPING_CONFIG_FIELDS,
    CoSpRoAuditConfig,
    build_teacher_graph,
    load_concept_groups_json,
)
from cospro.pipeline.reporting import render_teacher_graph_report
from cospro.pipeline.graph_io import save_graph_json


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


def audit_config_name(config: CoSpRoAuditConfig) -> str:
    values = {key: value for key, value in asdict(config).items() if key not in GROUPING_CONFIG_FIELDS}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    quantile = str(config.null_quantile).replace(".", "p")
    return f"audit_seed_{config.seed}_nullq_{quantile}_trials_{config.null_trials}_{digest}"


# Historical private name retained for local callers while the pipeline uses
# the public path helper below.
_audit_name = audit_config_name


def teacher_graph_path(concept_groups_path: Path, config: CoSpRoAuditConfig) -> Path:
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
    parser.add_argument(
        "--device", default="auto",
        help="Neighbour-search device: auto, cpu, cuda, or a concrete CUDA device.",
    )
    parser.add_argument("--neighbor-backend", choices=("auto", "exact", "lsh"))
    parser.add_argument("--ann-threshold", type=int)
    parser.add_argument("--ann-tables", type=int)
    parser.add_argument("--ann-bucket-size", type=int)
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        help="Checkpoint root. With a sweep, each audit gets an identity-named child directory.",
    )
    parser.add_argument(
        "--concept-type-scores", type=Path,
        help="Scores from cospro.cli.score_concept_types; selects the concept_type_gate rule.",
    )
    parser.add_argument(
        "--object-quantile", type=float, default=0.25,
        help="Share of the candidates the concept_type_gate drops, the most object-like first.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Write the graph here instead of beside its concept groups; one artifact only.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Recompute and atomically replace existing per-group checkpoints.",
    )
    return parser.parse_args(argv)


def concept_type_selection(args: argparse.Namespace, artifact_path: Path):
    """The concept-type gate when scores are given, otherwise the default rule.

    Scores address groups by id, so they only mean anything for the artifact they were computed
    from. Another grouping of the same dataset renumbers the groups, which would gate the wrong
    concepts, so the artifact's digest has to match the one the scores recorded.
    """

    if args.concept_type_scores is None:
        return None
    payload = json.loads(args.concept_type_scores.read_text(encoding="utf-8"))
    expected = payload.get("concept_groups_sha256")
    actual = _sha256(artifact_path)
    if expected is None:
        raise ValueError(
            f"{args.concept_type_scores} predates the digest check; recompute it with "
            "cospro.cli.score_concept_types for this concept-group artifact."
        )
    if expected != actual:
        raise ValueError(
            f"The concept-type scores were computed from {payload.get('concept_groups')} "
            f"(sha256 {expected[:12]}), not from {artifact_path} (sha256 {actual[:12]}). "
            "Score the artifact this graph is built from."
        )
    return ConceptTypeGate(payload["scores"], quantile=args.object_quantile)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output is not None and args.concept_groups.is_dir():
        raise ValueError("--output writes one graph, so pass a single concept_groups.json.")
    config_values = json.loads(args.config) if args.config else {}
    unknown = set(config_values).difference(CoSpRoAuditConfig.__dataclass_fields__)
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
    for name in ("neighbor_backend", "ann_threshold", "ann_tables", "ann_bucket_size"):
        value = getattr(args, name)
        if value is not None:
            config_values[name] = value
    config = CoSpRoAuditConfig(**config_values)
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
        output_directory = teacher_graph_path(artifact_path, config).parent
        if args.checkpoint_dir is None:
            checkpoint_directory = output_directory / "group_checkpoints"
        elif len(artifacts) == 1:
            checkpoint_directory = args.checkpoint_dir
        else:
            checkpoint_directory = args.checkpoint_dir / source["sha256"][:12] / output_directory.name
        graph = build_teacher_graph(
            cache,
            concept_groups,
            config,
            source,
            device=args.device,
            checkpoint_dir=checkpoint_directory,
            resume=not args.no_resume,
            selection=concept_type_selection(args, artifact_path),
        )
        destination = args.output if args.output is not None else output_directory / "teacher_graph.json"
        json_path = save_graph_json(graph, destination)
        html_path = render_teacher_graph_report(graph, destination.with_suffix(".html"))
        print(f"[INFO] Wrote {json_path}", flush=True)
        print(f"[INFO] Wrote {html_path}", flush=True)
        print(f"[INFO] Group checkpoints: {checkpoint_directory}", flush=True)
    print(f"[INFO] Generated {len(artifacts)} teacher graph(s) from explicit concept-group artifacts.")


if __name__ == "__main__":
    main()
