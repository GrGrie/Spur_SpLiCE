from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.spurious_eval.datasets.registry import DATASET_REGISTRY
from splice.ssl_regularization import SpliceConceptScorer, SpliceConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize SpLiCE concept-score distribution for a spurious-eval dataset")
    parser.add_argument("--dataset", default="waterbirds", choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--data_folder", default="./datasets")
    parser.add_argument("--split", default="train", choices=["train", "ds_train", "us_train", "balanced_train", "val", "test"])
    parser.add_argument("--splice_concepts", required=True, help="Comma-separated concept names or vocabulary indices.")
    parser.add_argument("--out_path", default="", help="Optional JSON path for the summary.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--disable_cudnn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable cuDNN for SpLiCE/OpenCLIP image encoding on CUDA.",
    )
    parser.add_argument("--splice_model", default="open_clip:ViT-B-32")
    parser.add_argument("--splice_vocab", default="laion")
    parser.add_argument("--splice_vocab_size", type=int, default=10000)
    parser.add_argument("--splice_l1_penalty", type=float, default=0.25)
    parser.add_argument("--splice_score_reduction", default="mean", choices=["mean", "max"])
    parser.add_argument(
        "--candidate_thresholds",
        default="0.005,0.01,0.02,0.05,0.1",
        help="Comma-separated thresholds to report counts for.",
    )
    return parser.parse_args()


def configure_torch_backend(args: argparse.Namespace) -> None:
    if str(args.device).startswith("cuda") and args.disable_cudnn:
        cudnn.enabled = False
        cudnn.benchmark = False
        cudnn.deterministic = True


def parse_thresholds(raw_thresholds: str) -> list[float]:
    thresholds = []
    for raw_value in raw_thresholds.split(","):
        value = raw_value.strip()
        if value:
            thresholds.append(float(value))
    return thresholds


def percentile(scores: torch.Tensor, q: float) -> float:
    return torch.quantile(scores.float(), torch.tensor(q, dtype=torch.float32)).item()


def summarize(scores: torch.Tensor, thresholds: list[float]) -> dict:
    scores = scores.float()
    count = scores.numel()
    threshold_counts = {}
    for threshold in thresholds:
        above = int((scores >= threshold).sum().item())
        threshold_counts[str(threshold)] = {
            "count": above,
            "fraction": above / count if count else 0.0,
        }

    return {
        "count": count,
        "min": scores.min().item() if count else 0.0,
        "p10": percentile(scores, 0.10) if count else 0.0,
        "p25": percentile(scores, 0.25) if count else 0.0,
        "median": percentile(scores, 0.50) if count else 0.0,
        "p75": percentile(scores, 0.75) if count else 0.0,
        "p90": percentile(scores, 0.90) if count else 0.0,
        "p95": percentile(scores, 0.95) if count else 0.0,
        "max": scores.max().item() if count else 0.0,
        "threshold_counts": threshold_counts,
    }


def main() -> None:
    args = parse_args()
    configure_torch_backend(args)
    config = SpliceConfig(
        use_splice=True,
        mode="augment",
        concepts=args.splice_concepts,
        l1_penalty=args.splice_l1_penalty,
        vocab=args.splice_vocab,
        vocab_size=args.splice_vocab_size,
        model=args.splice_model,
        score_reduction=args.splice_score_reduction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    scorer = SpliceConceptScorer(config)
    dataset_spec = DATASET_REGISTRY[args.dataset]
    full_dataset = dataset_spec["dataset"](args.data_folder)
    subset = full_dataset.get_subset(args.split, transform=None)
    scores = scorer.score_dataset(subset)
    thresholds = parse_thresholds(args.candidate_thresholds)
    summary = summarize(scores, thresholds)
    summary["split"] = args.split
    summary["dataset"] = args.dataset
    summary["splice_concepts"] = args.splice_concepts
    summary["resolved_concepts"] = [
        {"index": index, "concept": scorer.vocabulary[index]} for index in scorer.concept_indices
    ]
    summary["score_reduction"] = args.splice_score_reduction

    print("[INFO] Resolved concepts:")
    for item in summary["resolved_concepts"]:
        print(f"  {item['index']:5d} {item['concept']}")
    print("[INFO] Score distribution:")
    for key in ["count", "min", "p10", "p25", "median", "p75", "p90", "p95", "max"]:
        print(f"  {key}: {summary[key]}")
    print("[INFO] Counts above candidate thresholds:")
    for threshold, values in summary["threshold_counts"].items():
        print(f"  >= {threshold}: {values['count']} ({values['fraction']:.3%})")

    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
