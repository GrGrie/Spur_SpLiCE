"""Generate reusable CRP concept-group artifacts and pre-audit HTML reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from splice.crp import (
    GROUPING_CONFIG_FIELDS,
    CrpAuditConfig,
    build_concept_groups,
    save_concept_groups_json,
)
from splice.crp_reporting import render_concept_groups_report


DEFAULT_TEXT_THRESHOLDS = (0.70, 0.75, 0.82, 0.85, 0.90)
DEFAULT_COACTIVATION_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


def _threshold_name(value: float) -> str:
    return f"{value:.4g}".replace("-", "neg").replace(".", "p")


def _threshold_value(raw: str) -> float:
    """Accept both plain floats and shell-friendly bracketed list tokens."""
    return float(raw.strip().strip("[],"))


def _default_output_root(cache: dict) -> Path:
    repository = Path(__file__).resolve().parents[2]
    output_root = Path(os.environ.get("SPUR_SPLICE_OUTPUT_ROOT", repository / "outputs"))
    dataset = str(cache.get("provenance", {}).get("dataset", "crp"))
    return output_root / "shared" / dataset / "graphs" / "concept_groups"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splice-dataset-cache",
        required=True,
        type=Path,
        help="Frozen SpLiCE dataset cache (.pt).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Sweep directory (default: outputs/shared/<dataset>/graphs/concept_groups).",
    )
    parser.add_argument(
        "--text-similarity-threshold",
        "--text-similarity-thresholds",
        dest="text_thresholds",
        nargs="+",
        type=_threshold_value,
        help="One or more text thresholds; defaults to the predefined sweep grid.",
    )
    parser.add_argument(
        "--coactivation-threshold",
        "--coactivation-thresholds",
        dest="coactivation_thresholds",
        nargs="+",
        type=_threshold_value,
        help="One or more coactivation thresholds; defaults to the predefined sweep grid.",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON object overriding non-swept grouping settings.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_values = json.loads(args.config) if args.config else {}
    unknown = set(config_values).difference(GROUPING_CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"Unknown or non-grouping CRP settings: {sorted(unknown)}")
    configured_text = config_values.pop("text_similarity_threshold", None)
    text_thresholds = args.text_thresholds or (
        [configured_text] if configured_text is not None else list(DEFAULT_TEXT_THRESHOLDS)
    )
    configured_coactivation = config_values.pop("coactivation_threshold", None)
    coactivation_thresholds = args.coactivation_thresholds or (
        [configured_coactivation]
        if configured_coactivation is not None
        else list(DEFAULT_COACTIVATION_THRESHOLDS)
    )

    cache = torch.load(args.splice_dataset_cache, map_location="cpu", weights_only=True)
    output_root = args.output_root or _default_output_root(cache)
    produced = 0
    for text_threshold in dict.fromkeys(text_thresholds):
        for coactivation_threshold in dict.fromkeys(coactivation_thresholds):
            config = CrpAuditConfig(
                **config_values,
                text_similarity_threshold=text_threshold,
                coactivation_threshold=coactivation_threshold,
            )
            artifact = build_concept_groups(cache, config)
            name = (
                f"text_{_threshold_name(text_threshold)}_"
                f"coactivation_{_threshold_name(coactivation_threshold)}"
            )
            directory = output_root / name
            json_path = save_concept_groups_json(artifact, directory / "concept_groups.json")
            html_path = render_concept_groups_report(artifact, directory / "concept_groups.html")
            print(f"[INFO] Wrote {json_path}", flush=True)
            print(f"[INFO] Wrote {html_path}", flush=True)
            produced += 1
    print(f"[INFO] Generated {produced} concept-group configuration(s); no CRP audit or training ran.")


if __name__ == "__main__":
    main()
