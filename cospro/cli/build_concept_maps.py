"""Build the concept maps of a dataset's training images for spatial concept distillation.

For every training image the frozen CLIP ViT-B/32 of the SpLiCE cache gives a 14 x 14 map per concept
factor: the softmax over the factors of each patch's cosine with the factor's text direction
(``cospro.pipeline.concept_maps``). The factors are those training builds from the same concept
groups, SpLiCE cache and factor band, so the map channels match the factor columns.

    python -m cospro.cli.build_concept_maps --dataset metashift --data-folder ~/Datasets \\
        --concept-groups outputs/shared/metashift/graphs/concept_groups_meaning/laion_text_0p85_response_0p50/concept_groups.json \\
        --splice-cache <scratch>/features/Spur_SpLiCE/metashift/splice_dataset_cache/<cache>/splice_dataset_cache.pt

Outputs: the maps under ``<scratch>/features/Spur_SpLiCE/<dataset>/concept_maps/<name>/concept_maps.pt``
and a compact summary under ``outputs/shared/<dataset>/graphs/concept_maps/<name>/summary.json``, where
``<name>`` is ``<groups folder>_min<band>_merge<merge>_<attention><resolution>``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cospro.cli.sweep_concept_factors import SPLICE_MODEL, SPLICE_PRETRAINED, raw_text_embeddings
from cospro.data.registry import canonical_dataset_name, dataset_class, dataset_names
from cospro.pipeline.concept_factors import FactorConfig, factor_name, load_concept_factors
from cospro.pipeline.concept_maps import MAP_SIZE, WINDOW, assemble_windows, patch_embeddings, window_offsets
from cospro.tracking.artifacts import atomic_write_json, scratch_root, shared

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def maps_name(concept_groups: Path, min_frequency: float, merge: float, attention: str, resolution: int) -> str:
    return f"{Path(concept_groups).parent.name}_min{min_frequency:g}_merge{merge:g}_{attention}{resolution}"


class _Images(Dataset):
    def __init__(self, adapter, indices: list[int], resolution: int) -> None:
        self.adapter = adapter
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> torch.Tensor:
        return self.transform(self.adapter.get_input(self.indices[position]).convert("RGB"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=canonical_dataset_name, choices=dataset_names())
    parser.add_argument("--data-folder", default=os.environ.get("DATA_FOLDER", ""))
    parser.add_argument("--concept-groups", required=True, type=Path)
    parser.add_argument("--splice-cache", required=True, type=Path)
    parser.add_argument("--factor-min-frequency", type=float, default=0.05)
    parser.add_argument("--merge-similarity", type=float, default=0.0)
    parser.add_argument("--attention", choices=("qq", "value"), default="qq")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--temperature", type=float, default=0.01, help="Softmax temperature over the factors.")
    parser.add_argument("--feature-root", type=Path, default=None,
                        help="Root of caches and maps; defaults to <scratch>/features/Spur_SpLiCE.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None, help="Default: see the module docstring.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    args.cache_device = args.device
    feature_root = args.feature_root or scratch_root() / "features" / "Spur_SpLiCE"
    name = maps_name(args.concept_groups, args.factor_min_frequency, args.merge_similarity, args.attention,
                     args.resolution)
    output = args.output or feature_root / args.dataset / "concept_maps" / name / "concept_maps.pt"

    factors, _, cache_path = load_concept_factors(
        args.dataset, FactorConfig(min_frequency=args.factor_min_frequency, merge_similarity=args.merge_similarity),
        concept_groups=str(args.concept_groups), splice_cache=str(args.splice_cache),
    )
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    vocabulary = [str(word) for word in cache["vocabulary"]]
    vocab = json.loads(Path(args.concept_groups).read_text(encoding="utf-8"))["provenance"]["splice_vocab"]
    words = raw_text_embeddings(args, vocab, vocabulary)
    position = {word: index for index, word in enumerate(vocabulary)}
    directions = F.normalize(torch.stack([
        F.normalize(words[[position[concept] for concept in factor["concepts"]]].float(), dim=1).mean(dim=0)
        for factor in factors["factors"]
    ]), dim=1).to(args.device)
    del cache

    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(SPLICE_MODEL.split(":", 1)[1], pretrained=SPLICE_PRETRAINED)
    model = model.to(args.device).eval()
    sample_ids = factors["sample_ids"]
    adapter = dataset_class(args.dataset)(str(args.data_folder))
    indices = [int(sample_id.rpartition(":")[2]) for sample_id in sample_ids]
    loader = DataLoader(_Images(adapter, indices, args.resolution), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False)
    offsets = window_offsets(args.resolution, args.stride)
    maps = torch.zeros(len(indices), len(directions), MAP_SIZE, MAP_SIZE, dtype=torch.float16)
    side = WINDOW // 32
    start, done = time.time(), 0
    for images in loader:
        images = images.to(args.device)
        windows = torch.stack([images[:, :, top:top + WINDOW, left:left + WINDOW]
                               for top in offsets for left in offsets], dim=1)
        count = windows.shape[1]
        patches = patch_embeddings(model, windows.flatten(0, 1), attention=args.attention)
        scores = (patches @ directions.T).view(len(images), count, side, side, -1).permute(0, 1, 4, 2, 3)
        assembled = assemble_windows(scores, args.resolution, args.stride)
        maps[done:done + len(images)] = (assembled / args.temperature).softmax(dim=1).cpu().half()
        done += len(images)
        if done % (args.batch_size * 20) < args.batch_size:
            print(f"[INFO] {done}/{len(indices)} images, {time.time() - start:.0f} s", flush=True)

    names = [factor_name(factor) for factor in factors["factors"]]
    config = {key: getattr(args, key) for key in ("factor_min_frequency", "merge_similarity", "attention",
                                                   "resolution", "stride", "temperature")}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sample_ids": sample_ids, "factor_names": names, "maps": maps, "config": config,
                "concept_groups": str(args.concept_groups), "splice_cache": str(cache_path)}, output)
    winners = maps.float().argmax(dim=1)
    summary = {
        "maps": str(output), "images": len(indices), "factors": len(names), "config": config,
        "per_factor": [
            {"name": name, "mean_probability": round(float(maps[:, column].float().mean()), 4),
             "patch_share": round(float((winners == column).float().mean()), 4)}
            for column, name in enumerate(names)
        ],
    }
    summary["per_factor"].sort(key=lambda entry: -entry["patch_share"])
    summary_path = shared(args.dataset, "graphs", "concept_maps", name) / "summary.json"
    atomic_write_json(summary_path, summary)
    print(f"[results] Concept maps: {output}")
    print(f"[results] Summary: {summary_path}")
    return output


if __name__ == "__main__":
    main()
