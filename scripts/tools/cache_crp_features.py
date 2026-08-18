from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import splice
from experiments.spurious_eval.datasets.registry import get_dataset_spec
from splice.crp import CACHE_VERSION, save_feature_cache


class IndexedImages(Dataset):
    """Expose train images and stable source indices without labels or metadata."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.indices = dataset.get_subset("train", transform=None).indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        return index, self.dataset.get_input(index)


def identity_collate(batch):
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen CLIP, SpLiCE, and DINOv3 features.")
    parser.add_argument("--dataset", choices=("waterbirds", "celeba", "spur_cifar10"), required=True)
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--splice-model", default="open_clip:ViT-B-32")
    parser.add_argument("--splice-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--splice-vocab", default="laion")
    parser.add_argument("--splice-vocab-size", type=int, default=10000)
    parser.add_argument("--splice-l1-penalty", type=float, default=0.25)
    parser.add_argument("--dino-model", default="vit_small_patch16_dinov3.lvd1689m")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers must be non-negative.")

    device = torch.device(args.device)
    dataset_class = get_dataset_spec(args.dataset)["dataset"]
    images = IndexedImages(dataset_class(args.data_folder))
    loader = DataLoader(
        images,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
    )

    clip_preprocess = splice.get_preprocess(args.splice_model, pretrained=args.splice_pretrained)
    splice_model = splice.load(
        args.splice_model,
        args.splice_vocab,
        args.splice_vocab_size,
        device=device,
        pretrained=args.splice_pretrained,
        l1_penalty=args.splice_l1_penalty,
        return_weights=True,
    ).eval()

    import timm
    from timm.data import create_transform, resolve_model_data_config

    dino_model = timm.create_model(args.dino_model, pretrained=True, num_classes=0).to(device).eval()
    dino_preprocess = create_transform(**resolve_model_data_config(dino_model), is_training=False)

    sample_ids, clip_embeddings, splice_codes, dino_embeddings = [], [], [], []
    for batch_number, batch in enumerate(loader, start=1):
        indices = [item[0] for item in batch]
        raw_images = [item[1] for item in batch]
        clip_input = torch.stack([clip_preprocess(image) for image in raw_images]).to(device)
        dino_input = torch.stack([dino_preprocess(image) for image in raw_images]).to(device)
        with torch.inference_mode():
            clip_batch = F.normalize(splice_model.clip.encode_image(clip_input).float(), dim=1)
            centered = F.normalize(clip_batch - splice_model.image_mean, dim=1)
            code_batch = splice_model.decompose(centered)
            dino_batch = F.normalize(dino_model(dino_input).float(), dim=1)
        sample_ids.extend(f"{args.dataset}:{index}" for index in indices)
        clip_embeddings.append(clip_batch.cpu())
        splice_codes.append(code_batch.cpu())
        dino_embeddings.append(dino_batch.cpu())
        print(f"[INFO] Cached batch {batch_number}/{len(loader)}", flush=True)

    cache = {
        "cache_version": CACHE_VERSION,
        "sample_ids": sample_ids,
        "clip_embeddings": torch.cat(clip_embeddings),
        "image_mean": splice_model.image_mean.detach().cpu(),
        "splice_codes": torch.cat(splice_codes),
        "dictionary": splice_model.dictionary.detach().cpu(),
        "vocabulary": splice.get_vocabulary(args.splice_vocab, args.splice_vocab_size),
        "dino_embeddings": torch.cat(dino_embeddings),
        "provenance": {
            "dataset": args.dataset,
            "split": "train",
            "splice_model": args.splice_model,
            "splice_pretrained": args.splice_pretrained,
            "splice_vocab": args.splice_vocab,
            "splice_vocab_size": args.splice_vocab_size,
            "splice_l1_penalty": args.splice_l1_penalty,
            "dino_model": args.dino_model,
        },
    }
    save_feature_cache(cache, Path(args.output))
    print(f"[INFO] Wrote {len(sample_ids)} aligned frozen samples to {args.output}")


if __name__ == "__main__":
    main()
