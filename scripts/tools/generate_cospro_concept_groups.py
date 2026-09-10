"""Generate reusable CoSpRo concept groups and pre-audit HTML reports."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Callable

import torch

from experiments.spurious_eval.datasets.registry import get_dataset_spec
from splice.cospro import (
    GROUPING_CONFIG_FIELDS,
    CrpAuditConfig,
    build_concept_groups,
    load_concept_groups_json,
    save_concept_groups_json,
)
from splice.cospro_reporting import render_concept_groups_report


DEFAULT_TEXT_THRESHOLDS = (0.70, 0.75, 0.82, 0.85, 0.90)
DEFAULT_COACTIVATION_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


def _threshold_name(value: float) -> str:
    return f"{value:.12g}".replace("-", "neg").replace(".", "p")


def concept_group_directory(output_root: Path, config: CrpAuditConfig) -> Path:
    """Return a collision-safe directory for every grouping configuration."""

    name = (
        f"text_{_threshold_name(config.text_similarity_threshold)}_"
        f"coactivation_{_threshold_name(config.coactivation_threshold)}"
    )
    defaults = CrpAuditConfig()
    secondary = {
        field: getattr(config, field)
        for field in GROUPING_CONFIG_FIELDS
        if field not in {"text_similarity_threshold", "coactivation_threshold"}
    }
    default_secondary = {field: getattr(defaults, field) for field in secondary}
    if secondary != default_secondary:
        payload = json.dumps(secondary, sort_keys=True, separators=(",", ":"))
        name += "_config_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return output_root / name


def _threshold_value(raw: str) -> float:
    """Accept both plain floats and shell-friendly bracketed list tokens."""
    return float(raw.strip().strip("[],"))


def _default_output_root(cache: dict) -> Path:
    repository = Path(__file__).resolve().parents[2]
    output_root = Path(os.environ.get("SPUR_SPLICE_OUTPUT_ROOT", repository / "outputs"))
    dataset = str(cache.get("provenance", {}).get("dataset", "cospro"))
    return output_root / "shared" / dataset / "graphs" / "concept_groups"


def _dataset_image_resolver(cache: dict, data_folder: Path | None) -> Callable[[str], str | None] | None:
    """Return a cached resolver that embeds small dataset thumbnails as data URLs."""

    if data_folder is None:
        return None
    dataset_name = str(cache.get("provenance", {}).get("dataset", ""))
    if not dataset_name:
        raise ValueError("Dataset provenance is required to resolve report thumbnails.")
    dataset_class = get_dataset_spec(dataset_name)["dataset"]
    dataset = dataset_class(str(data_folder))
    resolved: dict[str, str] = {}

    def resolve(sample_id: str) -> str:
        if sample_id in resolved:
            return resolved[sample_id]
        sample_dataset, separator, raw_index = sample_id.rpartition(":")
        if not separator or sample_dataset != dataset_name:
            raise ValueError(f"Cannot resolve dataset sample ID {sample_id!r}.")
        image = dataset.get_input(int(raw_index)).convert("RGB")
        image.thumbnail((136, 116))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=78, optimize=True)
        source = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        resolved[sample_id] = source
        return source

    return resolve


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--splice-dataset-cache",
        type=Path,
        help="Frozen SpLiCE dataset cache (.pt).",
    )
    source.add_argument(
        "--render-existing",
        type=Path,
        help="Re-render one existing concept_groups.json with dataset thumbnails.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Sweep directory (default: outputs/shared/<dataset>/graphs/concept_groups).",
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=Path(os.environ["DATA_FOLDER"]) if os.environ.get("DATA_FOLDER") else None,
        help="Dataset root used to embed representative thumbnails (default: DATA_FOLDER).",
    )
    parser.add_argument(
        "--no-embed-images",
        action="store_true",
        help="Explicitly allow an HTML report with image placeholders instead of thumbnails.",
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
    if args.data_folder is None and not args.no_embed_images:
        raise ValueError(
            "Representative images require --data-folder (or DATA_FOLDER). "
            "Use --no-embed-images only when placeholders are intentional."
        )
    if args.render_existing is not None:
        artifact = load_concept_groups_json(args.render_existing)
        image_resolver = (
            None
            if args.no_embed_images
            else _dataset_image_resolver(artifact, args.data_folder)
        )
        html_path = render_concept_groups_report(
            artifact,
            args.render_existing.with_suffix(".html"),
            image_resolver=image_resolver,
        )
        print(f"[INFO] Re-rendered {html_path}", flush=True)
        return
    config_values = json.loads(args.config) if args.config else {}
    unknown = set(config_values).difference(GROUPING_CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"Unknown or non-grouping CoSpRo settings: {sorted(unknown)}")
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
    image_resolver = None if args.no_embed_images else _dataset_image_resolver(cache, args.data_folder)
    produced = 0
    for text_threshold in dict.fromkeys(text_thresholds):
        for coactivation_threshold in dict.fromkeys(coactivation_thresholds):
            config = CrpAuditConfig(
                **config_values,
                text_similarity_threshold=text_threshold,
                coactivation_threshold=coactivation_threshold,
            )
            artifact = build_concept_groups(cache, config)
            directory = concept_group_directory(output_root, config)
            json_path = save_concept_groups_json(artifact, directory / "concept_groups.json")
            html_path = render_concept_groups_report(
                artifact,
                directory / "concept_groups.html",
                image_resolver=image_resolver,
            )
            print(f"[INFO] Wrote {json_path}", flush=True)
            print(f"[INFO] Wrote {html_path}", flush=True)
            produced += 1
    print(f"[INFO] Generated {produced} concept-group configuration(s); no CoSpRo audit or training ran.")


if __name__ == "__main__":
    main()
