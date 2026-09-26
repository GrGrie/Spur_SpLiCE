"""Sweep concept grouping and factor merging, and score every factor set against the hidden labels.

For each dataset and dictionary the sweep makes sure the SpLiCE cache exists (building it on the GPU
when it does not), groups the cached concepts at every text-similarity and co-activation threshold,
builds the concept factors at every redundancy-merge threshold and types the factors and pairs with
``cospro.diagnostics.factor_validity``. The labels evaluate the factor sets; nothing chooses a
setting inside this command.

    python -m cospro.cli.sweep_concept_factors --data-folder ~/Datasets

Outputs, all compact JSON or Markdown that Git carries:

* ``outputs/shared/<dataset>/factor_sweep/<vocab>/text_<t>_coactivation_<c>/groups.json``: the groups;
* ``.../factors_merge_<m>.json``: factors, pairs and their post-hoc types for one merge threshold;
* ``outputs/reports/concept_factor_sweep/summary.{json,md}``: one row per factor set.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from cospro.cli.cache_splice_dataset import cache_config_name
from cospro.config.options import section_defaults
from cospro.config.training import ConceptFactorOptions
from cospro.data.registry import canonical_dataset_name, dataset_names
from cospro.diagnostics.factor_validity import diagnose_factors
from cospro.diagnostics.labels import load_labels
from cospro.pipeline import CoSpRoAuditConfig, build_concept_groups
from cospro.pipeline.cache import validate_splice_dataset_cache
from cospro.pipeline.concept_factors import FactorConfig, build_concept_factors, factor_name
from cospro.tracking.artifacts import atomic_write_json, report, scratch_root, shared

DEFAULTS = section_defaults(ConceptFactorOptions)
VOCABULARY_SIZES = {"laion": 10000, "openimages_v7": -1}
SPLICE_MODEL = "open_clip:ViT-B-32"
SPLICE_PRETRAINED = "laion2b_s34b_b79k"
SPLICE_L1_PENALTY = 0.25


def token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-folder", default=os.environ.get("DATA_FOLDER", ""),
                        help="Dataset root; defaults to $DATA_FOLDER.")
    parser.add_argument("--datasets", nargs="+", type=canonical_dataset_name, choices=dataset_names(),
                        default=["metashift", "spur_cifar10"])
    parser.add_argument("--vocabs", nargs="+", choices=sorted(VOCABULARY_SIZES), default=["laion", "openimages_v7"])
    parser.add_argument("--text-thresholds", nargs="+", type=float,
                        default=[0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--coactivation-thresholds", nargs="+", type=float, default=[0.20, 0.30, 0.40])
    parser.add_argument("--merge-similarities", nargs="+", type=float, default=[0.0, 0.70, 0.80, 0.90],
                        help="Redundancy-merge thresholds of the factors; 0 disables merging.")
    parser.add_argument("--feature-root", type=Path, default=None,
                        help="SpLiCE cache root; defaults to <scratch>/features/Spur_SpLiCE.")
    parser.add_argument("--cache-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--cache-num-workers", type=int, default=4)
    parser.add_argument("--study", default="concept_factor_sweep")
    return parser.parse_args(argv)


def ensure_cache(args: argparse.Namespace, dataset: str, vocab: str) -> Path:
    """The cache of ``dataset`` under ``vocab``, built by the cache stage when it is missing."""

    feature_root = args.feature_root or scratch_root() / "features" / "Spur_SpLiCE"
    size = VOCABULARY_SIZES[vocab]
    name = cache_config_name(argparse.Namespace(
        dataset=dataset, splice_model=SPLICE_MODEL, splice_pretrained=SPLICE_PRETRAINED, splice_vocab=vocab,
        splice_vocab_size=size, splice_l1_penalty=SPLICE_L1_PENALTY, splice_vocab_file=None,
        splice_vocab_order=None,
    ))
    path = feature_root / dataset / "splice_dataset_cache" / name / "splice_dataset_cache.pt"
    if path.is_file():
        print(f"[INFO] Reusing SpLiCE cache {path}", flush=True)
        return path
    print(f"[INFO] Building SpLiCE cache {path}", flush=True)
    subprocess.run([
        sys.executable, "-u", "-m", "cospro.cli.cache_splice_dataset",
        "--dataset", dataset, "--data-folder", str(args.data_folder), "--output-root", str(feature_root),
        "--batch-size", str(args.cache_batch_size), "--num-workers", str(args.cache_num_workers),
        "--device", args.cache_device, "--splice-model", SPLICE_MODEL, "--splice-pretrained", SPLICE_PRETRAINED,
        "--splice-vocab", vocab, "--splice-vocab-size", str(size), "--splice-l1-penalty", str(SPLICE_L1_PENALTY),
    ], check=True)
    return path


def compact_groups(groups: dict, presence: torch.Tensor) -> dict[str, Any]:
    """The groups with their frequencies, without the per-image lists a training run would need."""

    return {
        "config": groups["config"],
        "provenance": groups["provenance"],
        "active_concept_count": groups["active_concept_count"],
        "group_count": len(groups["groups"]),
        "diagnostics": groups["diagnostics"],
        "groups": [
            {
                "group_id": group["group_id"],
                "concepts": group["concepts"],
                "frequency": round(float(presence[:, group["concept_indices"]].any(dim=1).float().mean()), 4),
            }
            for group in groups["groups"]
        ],
    }


def factor_set_record(factors: dict, diagnosis: dict) -> dict[str, Any]:
    by_factor = {entry["factor"]: entry for entry in diagnosis["factors"]}
    return {
        "config": factors["config"],
        "group_count": factors["group_count"],
        "summary": diagnosis["summary"],
        "pairs": diagnosis["pairs"],
        "factors": [
            {
                "name": factor_name(factor),
                "concepts": factor["concepts"],
                "frequency": round(factor["frequency"], 4),
                "u_class": by_factor[position]["u_class"],
                "u_attribute": by_factor[position]["u_attribute"],
                "type": by_factor[position]["type"],
            }
            for position, factor in enumerate(factors["factors"])
        ],
    }


def summary_markdown(rows: list[dict]) -> str:
    lines = ["# Concept-factor sweep", "",
             "Pair precision: share of the top entangled pairs that join a class factor to an attribute factor "
             "(post-hoc, hidden labels). Attribute signal: the largest I(F; a | y) / H(F) over the factors.", ""]
    keys = sorted({(row["dataset"], row["vocab"]) for row in rows})
    for dataset, vocab in keys:
        lines += [f"## {dataset}, {vocab}", "",
                  "| text | coact | merge | groups | factors | cross/pairs | precision | attr. signal | attr. factors | "
                  "strongest attribute factor |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        selected = [row for row in rows if (row["dataset"], row["vocab"]) == (dataset, vocab)]
        selected.sort(key=lambda row: (-(row.get("pair_precision") or 0), -row.get("attribute_signal", 0)))
        for row in selected:
            if "error" in row:
                lines.append(f"| {row['text']:.2f} | {row['coactivation']:.2f} | {row['merge']:.2f} | "
                             f"{row.get('groups', '')} | error: {row['error']} | | | | | |")
                continue
            precision = "" if row["pair_precision"] is None else f"{row['pair_precision']:.2f}"
            lines.append(
                f"| {row['text']:.2f} | {row['coactivation']:.2f} | {row['merge']:.2f} | {row['groups']} | "
                f"{row['factors']} | {row['cross_pairs']}/{row['pairs']} | {precision} | "
                f"{row['attribute_signal']:.3f} | {row['attribute_factors']} | {row['top_attribute']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if not args.data_folder:
        raise SystemExit("Pass --data-folder or set $DATA_FOLDER.")
    rows: list[dict] = []
    summary_dir = report(args.study)
    for dataset in args.datasets:
        labels = load_labels(dataset, args.data_folder)
        for vocab in args.vocabs:
            cache = validate_splice_dataset_cache(
                torch.load(ensure_cache(args, dataset, vocab), map_location="cpu", weights_only=True)
            )
            y, a = labels.for_ids(cache["sample_ids"])
            presence = cache["splice_codes"] > 0
            for text in args.text_thresholds:
                for coactivation in args.coactivation_thresholds:
                    grouping = CoSpRoAuditConfig(text_similarity_threshold=text, coactivation_threshold=coactivation)
                    groups = build_concept_groups(cache, grouping)
                    folder = shared(dataset, "factor_sweep", vocab, f"text_{token(text)}_coactivation_{token(coactivation)}")
                    folder.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(folder / "groups.json", compact_groups(groups, presence))
                    for merge in args.merge_similarities:
                        row = {"dataset": dataset, "vocab": vocab, "text": text, "coactivation": coactivation,
                               "merge": merge, "groups": len(groups["groups"])}
                        try:
                            factors = build_concept_factors(cache, groups, FactorConfig(merge_similarity=merge))
                        except ValueError as error:
                            rows.append({**row, "error": str(error)})
                            continue
                        names = [factor_name(factor) for factor in factors["factors"]]
                        diagnosis = diagnose_factors(factors["active"].numpy(), factors["pairs"], names, y, a)
                        atomic_write_json(folder / f"factors_merge_{token(merge)}.json",
                                          factor_set_record(factors, diagnosis))
                        summary = diagnosis["summary"]
                        top = summary["top_attribute_factors"][0] if summary["top_attribute_factors"] else {}
                        rows.append({
                            **row,
                            "factors": summary["factor_count"],
                            "pairs": summary["pair_count"],
                            "cross_pairs": summary["cross_pairs"],
                            "pair_precision": summary["pair_precision"],
                            "attribute_signal": summary["best_attribute_signal"],
                            "attribute_factors": summary["type_counts"]["attribute"],
                            "top_attribute": top.get("name", ""),
                            "top_pairs": [f"{pair['verdict']}: {pair['concepts'][0]} <-> {pair['concepts'][1]}"
                                          for pair in diagnosis["pairs"][:3]],
                        })
                        print(f"[sweep] {dataset} {vocab} text={text:.2f} coact={coactivation:.2f} merge={merge:.2f}: "
                              f"{summary['cross_pairs']}/{summary['pair_count']} cross pairs, "
                              f"attribute signal {summary['best_attribute_signal']:.3f}", flush=True)
                    # Write after every grouping, so an interrupted sweep still leaves a readable summary.
                    summary_dir.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(summary_dir / "summary.json", {"rows": rows})
                    (summary_dir / "summary.md").write_text(summary_markdown(rows), encoding="utf-8")
            del cache, presence
    print(f"[results] Sweep summary: {summary_dir / 'summary.md'}")
    return summary_dir / "summary.md"


if __name__ == "__main__":
    main()
