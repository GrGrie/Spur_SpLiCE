"""Score the concept groups of one artifact by how much their words name context, not the object.

Reads the dictionary's text encoder only: no images, no annotations and no property of the dataset,
so the scores can be produced anywhere in seconds. The ``concept_type_gate`` selection rule drops
the most object-like candidates by a quantile of these scores.

    python -m cospro.cli.score_concept_types --concept-groups concept_groups.json --output scores.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cospro.pipeline.concept_type import CONTEXT_PROMPTS, OBJECT_PROMPTS, group_scores
from cospro.tracking.artifacts import atomic_write_json


def load_text_encoder(model: str, pretrained: str, device: str):
    """The text tower of the backbone the dictionary was embedded with."""

    import open_clip

    name = model.split(":", 1)[1] if ":" in model else model
    encoder, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained, device=device)
    return encoder.eval(), open_clip.get_tokenizer(name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concept-groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splice-model", default="open_clip:ViT-B-32")
    parser.add_argument("--splice-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--show", type=int, default=10, help="Concept groups to print at each end.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    groups = json.loads(args.concept_groups.read_text(encoding="utf-8"))["groups"]
    model, tokenizer = load_text_encoder(args.splice_model, args.splice_pretrained, args.device)
    scores = group_scores(groups, model, tokenizer)
    atomic_write_json(args.output, {
        "schema": "concept-type-scores-v1",
        "concept_groups": str(args.concept_groups),
        "splice_model": args.splice_model,
        "splice_pretrained": args.splice_pretrained,
        "context_prompts": list(CONTEXT_PROMPTS),
        "object_prompts": list(OBJECT_PROMPTS),
        "scores": scores,
    })
    names = {str(group["group_id"]): ", ".join(group["concepts"][:3]) for group in groups}
    ranked = sorted(scores.items(), key=lambda item: item[1])
    print(f"[INFO] Scored {len(scores)} concept groups")
    print("[INFO] Most object-like (dropped first):")
    for key, value in ranked[: args.show]:
        print(f"    {value:+.3f}  {names.get(key, key)}")
    print("[INFO] Most context-like:")
    for key, value in ranked[-args.show:]:
        print(f"    {value:+.3f}  {names.get(key, key)}")
    print(f"[INFO] Wrote {args.output}")


if __name__ == "__main__":
    main()
