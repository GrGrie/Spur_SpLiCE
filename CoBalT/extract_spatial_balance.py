"""Extract sparse CRPv4 image-specific balance factors in the SpLiCE vocabulary."""

from __future__ import annotations

import argparse

import torch

from CoBalT.data import spatial_inference_loader
from CoBalT.runtime import atomic_torch_save, seed_everything
from CoBalT.spatial import (
    FrozenClipPatchEncoder,
    SpatialSpliceDiscoveryModel,
    patchwise_spatial_evidence,
)
from splice.crp import validate_feature_cache
from splice.spatial_balance import SPATIAL_BALANCE_ARTIFACT, SPATIAL_VARIANTS


VARIANTS = {
    "vanilla_patchwise": ("vanilla", False),
    "vanilla_slots": ("vanilla", True),
    "sclip_patchwise": ("sclip", False),
    "sclip_slots": ("sclip", True),
}
assert set(VARIANTS) == SPATIAL_VARIANTS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("Extract CRPv4 spatial SpLiCE evidence")
    parser.add_argument("--dataset", choices=["waterbirds", "celeba"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="vanilla_slots")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--concepts-per-region", type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.concepts_per_region <= 0:
        raise ValueError("batch-size and concepts-per-region must be positive.")

    cache = validate_feature_cache(
        torch.load(args.cache, map_location="cpu", weights_only=True)
    )
    provenance = cache.get("provenance", {})
    if str(provenance.get("dataset", "")).lower() != args.dataset:
        raise ValueError("CRP cache belongs to a different dataset.")
    source, use_slots = VARIANTS[args.variant]
    checkpoint = None
    if use_slots:
        if not args.checkpoint:
            raise ValueError("Slot variants require --checkpoint.")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("artifact") != "splice_spatial_slots_v1":
            raise ValueError("Checkpoint is not a CRPv4 spatial-slot artifact.")
        if checkpoint.get("dataset") != args.dataset:
            raise ValueError("Spatial-slot checkpoint belongs to a different dataset.")
        if checkpoint["model_config"]["feature_source"] != source:
            raise ValueError("Spatial-slot checkpoint uses a different feature source.")
        checkpoint_provenance = checkpoint.get("cache_provenance", {})
        for key in (
            "dataset",
            "split",
            "splice_model",
            "splice_pretrained",
            "splice_vocab",
            "splice_vocab_size",
            "splice_l1_penalty",
        ):
            if checkpoint_provenance.get(key) != provenance.get(key):
                raise ValueError(f"Spatial-slot checkpoint does not match cache field {key!r}.")
        seed_everything(int(checkpoint["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = str(provenance.get("splice_model", "open_clip:ViT-B-32")).split(":", 1)[-1]
    pretrained = str(provenance.get("splice_pretrained", "laion2b_s34b_b79k"))
    encoder = FrozenClipPatchEncoder(model_name, pretrained, source).to(device)
    model = None
    if checkpoint is not None:
        config = checkpoint["model_config"]
        model = SpatialSpliceDiscoveryModel(
            encoder,
            cache["dictionary"],
            num_slots=int(config["num_slots"]),
            student_temperature=float(config["student_temperature"]),
            teacher_temperature=float(config["teacher_temperature"]),
            teacher_momentum=float(config["teacher_momentum"]),
            center_momentum=float(config["center_momentum"]),
            semantic_weight=float(config["semantic_weight"]),
        ).to(device)
        model.student_grouping.load_state_dict(checkpoint["student_grouping"])
        model.teacher_grouping.load_state_dict(checkpoint["teacher_grouping"])
        model.teacher_center.copy_(checkpoint["teacher_center"])
        model.eval()

    loader = spatial_inference_loader(
        args.dataset, args.data_root, args.batch_size, args.workers, args.image_size
    )
    all_indices, all_evidence, all_confidence, all_source_ids = [], [], [], []
    dictionary = cache["dictionary"].to(device)
    for images, source_ids in loader:
        images = images.to(device, non_blocking=True)
        if model is None:
            indices, evidence, confidence = patchwise_spatial_evidence(
                encoder, dictionary, images, args.concepts_per_region
            )
        else:
            indices, evidence, confidence = model.spatial_evidence(
                images, args.concepts_per_region
            )
        all_indices.append(indices)
        all_evidence.append(evidence)
        all_confidence.append(confidence)
        all_source_ids.extend(int(value) for value in source_ids)

    sample_ids = [f"{args.dataset}:{source_id}" for source_id in all_source_ids]
    if sample_ids != cache["sample_ids"]:
        raise ValueError("Spatial inference order does not exactly match the CRP cache.")
    atomic_torch_save(
        {
            "artifact": SPATIAL_BALANCE_ARTIFACT,
            "dataset": args.dataset,
            "sample_ids": sample_ids,
            "vocabulary": cache["vocabulary"],
            "variant": args.variant,
            "concept_indices": torch.cat(all_indices),
            "evidence": torch.cat(all_evidence),
            "confidence": torch.cat(all_confidence),
            "config": {
                "feature_source": source,
                "use_slots": use_slots,
                "concepts_per_region": args.concepts_per_region,
                "image_size": args.image_size,
                "model_name": model_name,
                "pretrained": pretrained,
                "slot_model_config": dict(checkpoint["model_config"]) if checkpoint else None,
            },
            "cache_provenance": dict(provenance),
        },
        args.output,
    )
    print(f"[INFO] Wrote {args.variant} spatial balance for {len(sample_ids)} samples to {args.output}")


if __name__ == "__main__":
    main()
