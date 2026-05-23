from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import splice
from experiments.spurious_eval.datasets.waterbirds import WaterbirdsDataset
from splice.ssl_regularization import identity_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Discover Waterbirds spurious concepts with frozen SpLiCE")
    parser.add_argument("--data_folder", default="./datasets")
    parser.add_argument("--split", default="train", choices=["train", "ds_train", "us_train", "balanced_train", "val", "test"])
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--splice_model", default="open_clip:ViT-B-32")
    parser.add_argument("--splice_vocab", default="laion")
    parser.add_argument("--splice_vocab_size", type=int, default=10000)
    parser.add_argument("--splice_l1_penalty", type=float, default=0.25)
    parser.add_argument("--min_mean_weight", type=float, default=0.0)
    parser.add_argument("--label_penalty", type=float, default=1.0)
    parser.add_argument("--use_abs_score", action="store_true")
    return parser.parse_args()


def load_splice(args: argparse.Namespace):
    preprocess = splice.get_preprocess(args.splice_model)
    vocabulary = splice.get_vocabulary(args.splice_vocab, args.splice_vocab_size)
    splicemodel = splice.load(
        args.splice_model,
        args.splice_vocab,
        args.splice_vocab_size,
        args.device,
        l1_penalty=args.splice_l1_penalty,
        return_weights=True,
    )
    splicemodel.eval()
    for parameter in splicemodel.parameters():
        parameter.requires_grad = False
    return preprocess, vocabulary, splicemodel


def decompose_by_group(args: argparse.Namespace):
    preprocess, vocabulary, splicemodel = load_splice(args)
    full_dataset = WaterbirdsDataset(args.data_folder)
    subset = full_dataset.get_subset(args.split, transform=None)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
    )

    vocab_size = len(vocabulary)
    group_sums = torch.zeros(4, vocab_size, dtype=torch.float64)
    group_counts = torch.zeros(4, dtype=torch.float64)
    total_sum = torch.zeros(vocab_size, dtype=torch.float64)
    total_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = torch.stack([preprocess(item[0]) for item in batch], dim=0).to(args.device)
            metadata = torch.stack([item[2] for item in batch], dim=0)
            weights = splicemodel.encode_image(images).detach().cpu().double()
            groups = metadata[:, 0].long() + 2 * metadata[:, 1].long()

            for group in range(4):
                mask = groups == group
                if mask.any():
                    group_sums[group] += weights[mask].sum(dim=0)
                    group_counts[group] += mask.sum().item()
            total_sum += weights.sum(dim=0)
            total_count += weights.shape[0]
            if batch_idx % 10 == 0:
                print(f"[INFO] Processed {total_count} images", flush=True)

    group_means = group_sums / group_counts.clamp_min(1).unsqueeze(1)
    dataset_mean = total_sum / max(total_count, 1)
    return vocabulary, group_means, group_counts, dataset_mean, total_count


def rank_concepts(
    vocabulary: list[str],
    group_means: torch.Tensor,
    group_counts: torch.Tensor,
    dataset_mean: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict]:
    land_background_mean = (group_means[0] * group_counts[0] + group_means[2] * group_counts[2]) / (
        group_counts[0] + group_counts[2]
    ).clamp_min(1)
    water_background_mean = (group_means[1] * group_counts[1] + group_means[3] * group_counts[3]) / (
        group_counts[1] + group_counts[3]
    ).clamp_min(1)
    landbird_mean = (group_means[0] * group_counts[0] + group_means[1] * group_counts[1]) / (
        group_counts[0] + group_counts[1]
    ).clamp_min(1)
    waterbird_mean = (group_means[2] * group_counts[2] + group_means[3] * group_counts[3]) / (
        group_counts[2] + group_counts[3]
    ).clamp_min(1)

    background_effect = water_background_mean - land_background_mean
    label_effect = waterbird_mean - landbird_mean
    signed_score = background_effect.abs() - args.label_penalty * label_effect.abs()
    score = signed_score.abs() if args.use_abs_score else signed_score
    eligible = dataset_mean >= args.min_mean_weight

    candidates = []
    for index in torch.argsort(score, descending=True).tolist():
        if not eligible[index] or score[index].item() <= 0:
            continue
        candidates.append(
            {
                "index": index,
                "concept": vocabulary[index],
                "score": round(score[index].item(), 8),
                "background_effect": round(background_effect[index].item(), 8),
                "label_effect": round(label_effect[index].item(), 8),
                "mean_weight": round(dataset_mean[index].item(), 8),
                "group_means": {
                    "landbird_on_land": round(group_means[0, index].item(), 8),
                    "landbird_on_water": round(group_means[1, index].item(), 8),
                    "waterbird_on_land": round(group_means[2, index].item(), 8),
                    "waterbird_on_water": round(group_means[3, index].item(), 8),
                },
            }
        )
        if len(candidates) >= args.top_k:
            break
    return candidates


def write_outputs(args: argparse.Namespace, candidates: list[dict], group_counts: torch.Tensor, total_count: int) -> None:
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "background_effect_abs_minus_label_effect_abs",
        "split": args.split,
        "total_count": total_count,
        "group_counts": {
            "landbird_on_land": int(group_counts[0].item()),
            "landbird_on_water": int(group_counts[1].item()),
            "waterbird_on_land": int(group_counts[2].item()),
            "waterbird_on_water": int(group_counts[3].item()),
        },
        "settings": {
            "top_k": args.top_k,
            "splice_model": args.splice_model,
            "splice_vocab": args.splice_vocab,
            "splice_vocab_size": args.splice_vocab_size,
            "splice_l1_penalty": args.splice_l1_penalty,
            "min_mean_weight": args.min_mean_weight,
            "label_penalty": args.label_penalty,
            "use_abs_score": args.use_abs_score,
        },
        "concepts": candidates,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    concepts_path = out_path.with_suffix(".concepts.txt")
    concepts_path.write_text(",".join(candidate["concept"] for candidate in candidates) + "\n", encoding="utf-8")

    indices_path = out_path.with_suffix(".indices.txt")
    indices_path.write_text(",".join(str(candidate["index"]) for candidate in candidates) + "\n", encoding="utf-8")

    print(f"[INFO] Wrote discovery JSON to {out_path}")
    print(f"[INFO] Wrote concept list to {concepts_path}")
    print(f"[INFO] Wrote index list to {indices_path}")
    print("[INFO] Top concepts:")
    for candidate in candidates:
        print(
            f"  {candidate['index']:5d} {candidate['concept']:<30} "
            f"score={candidate['score']:.6f} bg={candidate['background_effect']:.6f} "
            f"label={candidate['label_effect']:.6f}"
        )


def main() -> None:
    args = parse_args()
    vocabulary, group_means, group_counts, dataset_mean, total_count = decompose_by_group(args)
    candidates = rank_concepts(vocabulary, group_means, group_counts, dataset_mean, args)
    write_outputs(args, candidates, group_counts, total_count)


if __name__ == "__main__":
    main()
