"""Build meaning concept groups in the layout concept-factor training reads, and report their factors.

The groups join concepts whose raw CLIP text embeddings name the same thing, by average linkage
(``cospro.pipeline.grouping.group_by_meaning``). The command writes the full artifact, with concept
indices, sample ids and the cache provenance, so ``--factor_concept_groups`` can load it; the sweep
writes only a compact view. It then builds the factors at the given band and prints them, and with
``--diagnose`` types them against the hidden labels (evaluation only).

    python -m cospro.cli.build_meaning_groups --dataset metashift --vocab laion --data-folder ~/Datasets

Output: ``outputs/shared/<dataset>/graphs/concept_groups_meaning/<vocab>_text_<t>_response_<r>/concept_groups.json``
and ``factor_report.json`` beside it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from cospro.cli.sweep_concept_factors import VOCABULARY_SIZES, ensure_cache, raw_text_embeddings, token
from cospro.data.registry import canonical_dataset_name, dataset_names
from cospro.diagnostics.factor_validity import diagnose_factors
from cospro.diagnostics.labels import load_labels
from cospro.pipeline.cache import validate_splice_dataset_cache
from cospro.pipeline.concept_factors import FactorConfig, build_concept_factors, factor_name
from cospro.pipeline.grouping import build_meaning_groups
from cospro.tracking.artifacts import atomic_write_json, shared


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=canonical_dataset_name, choices=dataset_names())
    parser.add_argument("--vocab", required=True, choices=sorted(VOCABULARY_SIZES))
    parser.add_argument("--data-folder", default=os.environ.get("DATA_FOLDER", ""),
                        help="Dataset root, for a missing cache and for --diagnose; defaults to $DATA_FOLDER.")
    parser.add_argument("--feature-root", type=Path, default=None,
                        help="SpLiCE cache root; defaults to <scratch>/features/Spur_SpLiCE.")
    parser.add_argument("--text-threshold", type=float, default=0.85)
    parser.add_argument("--response-threshold", type=float, default=0.5)
    parser.add_argument("--group-min-frequency", type=float, default=0.002)
    parser.add_argument("--factor-min-frequency", type=float, default=0.01,
                        help="Factor band of the printed report; training sets its own with --factor_min_frequency.")
    parser.add_argument("--merge-similarity", type=float, default=0.0)
    parser.add_argument("--diagnose", action="store_true", help="Type the factors against the hidden labels.")
    parser.add_argument("--cache-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--cache-num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None, help="Default: see the module docstring.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    cache_path = ensure_cache(args, args.dataset, args.vocab)
    cache = validate_splice_dataset_cache(torch.load(cache_path, map_location="cpu", weights_only=True))
    text_embeddings = raw_text_embeddings(args, args.vocab, [str(word) for word in cache["vocabulary"]])
    groups = build_meaning_groups(cache, text_embeddings, text_threshold=args.text_threshold,
                                  response_threshold=args.response_threshold,
                                  min_frequency=args.group_min_frequency)
    output = args.output or shared(
        args.dataset, "graphs", "concept_groups_meaning",
        f"{args.vocab}_text_{token(args.text_threshold)}_response_{token(args.response_threshold)}",
    ) / "concept_groups.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, groups)

    factors = build_concept_factors(cache, groups, FactorConfig(min_frequency=args.factor_min_frequency,
                                                                merge_similarity=args.merge_similarity))
    names = [factor_name(factor) for factor in factors["factors"]]
    entries = [{"name": name, "concepts": factor["concepts"], "frequency": round(factor["frequency"], 4)}
               for name, factor in zip(names, factors["factors"])]
    summary = {}
    if args.diagnose:
        y, a = load_labels(args.dataset, args.data_folder).for_ids(cache["sample_ids"])
        diagnosis = diagnose_factors(factors["active"].numpy(), factors["pairs"], names, y, a)
        for entry, typed in zip(entries, diagnosis["factors"]):
            entry.update(type=typed["type"], u_class=typed["u_class"], u_attribute=typed["u_attribute"])
        summary = diagnosis["summary"]
    report = {"groups": str(output), "splice_cache": str(cache_path), "factor_min_frequency": args.factor_min_frequency,
              "merge_similarity": args.merge_similarity, "group_count": len(groups["groups"]),
              "factor_count": len(entries), "summary": summary, "factors": entries}
    atomic_write_json(output.parent / "factor_report.json", report)

    composite = sorted((entry for entry in entries if len(entry["concepts"]) > 1), key=lambda entry: -entry["frequency"])
    print(f"[INFO] {len(groups['groups'])} groups, {len(entries)} factors in the band, {len(composite)} of several concepts")
    for entry in composite[:20]:
        typed = f" {entry['type']}" if "type" in entry else ""
        print(f"  {100 * entry['frequency']:5.1f}%{typed}  {' | '.join(entry['concepts'][:8])}")
    if summary:
        print(f"[INFO] factor types {summary['type_counts']}, attribute signal {summary['best_attribute_signal']:.3f}")
    print(f"[results] Concept groups: {output}")
    print(f"[results] Factor report: {output.parent / 'factor_report.json'}")
    print(f"[results] SpLiCE cache for --factor_splice_cache: {cache_path}")
    return output


if __name__ == "__main__":
    main()
