"""Show the concept factors and entangled pairs a concept-factor run would train with.

Reads the dataset's concept groups and SpLiCE cache, exactly as ``--splice_mode concept_factors``
does, and writes the JSON report without training anything:

    python -m cospro.cli.inspect_concept_factors --dataset metashift
    python -m cospro.cli.inspect_concept_factors --dataset spur_cifar10 --min-correlation 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cospro.config.training import ConceptFactorOptions
from cospro.config.options import section_defaults
from cospro.data.registry import canonical_dataset_name, dataset_names
from cospro.pipeline.concept_factors import FactorConfig, factor_report, format_factor_report, load_concept_factors
from cospro.tracking.artifacts import atomic_write_json, shared

DEFAULTS = section_defaults(ConceptFactorOptions)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=canonical_dataset_name, choices=dataset_names(), required=True)
    parser.add_argument("--concept-groups", default="")
    parser.add_argument("--splice-cache", default="")
    parser.add_argument("--min-frequency", type=float, default=DEFAULTS["factor_min_frequency"])
    parser.add_argument("--max-frequency", type=float, default=DEFAULTS["factor_max_frequency"])
    parser.add_argument("--max-count", type=int, default=DEFAULTS["factor_max_count"])
    parser.add_argument("--condition-pairs", type=int, default=DEFAULTS["factor_condition_pairs"])
    parser.add_argument("--min-correlation", type=float, default=DEFAULTS["factor_min_correlation"])
    parser.add_argument("--max-text-similarity", type=float, default=DEFAULTS["factor_max_text_similarity"])
    parser.add_argument("--whitening-eps", type=float, default=DEFAULTS["factor_whitening_eps"])
    parser.add_argument("--output", type=Path, help="Report path; default outputs/shared/<dataset>/factors/.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    config = FactorConfig(
        min_frequency=args.min_frequency,
        max_frequency=args.max_frequency,
        max_count=args.max_count,
        condition_pairs=args.condition_pairs,
        min_correlation=args.min_correlation,
        max_text_similarity=args.max_text_similarity,
        whitening_eps=args.whitening_eps,
    )
    factors, groups_path, cache_path = load_concept_factors(
        args.dataset, config, concept_groups=args.concept_groups, splice_cache=args.splice_cache,
    )
    report = {**factor_report(factors), "concept_groups": str(groups_path), "splice_cache": str(cache_path)}
    output = args.output or shared(args.dataset, "factors", "concept_factors_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(format_factor_report(report))
    print(f"[results] Concept-factor report: {output}")
    return output


if __name__ == "__main__":
    main()
