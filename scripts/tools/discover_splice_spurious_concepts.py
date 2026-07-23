from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import splice
from experiments.spurious_eval.datasets.registry import DATASET_REGISTRY
from splice.ssl_regularization import (
    SpliceConfig,
    dataset_score_cache_key,
    identity_collate,
    save_score_cache,
    score_cache_path,
)


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Discover spurious concepts with frozen SpLiCE")
    parser.add_argument("--dataset", default="waterbirds", choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--data_folder", default="./datasets")
    parser.add_argument("--split", default="train", choices=["train", "ds_train", "us_train", "balanced_train", "val", "test"])
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--per_image_top_k",
        type=int,
        default=0,
        help="Write an optional per-image JSONL audit; disabled by default to avoid large files.",
    )
    parser.add_argument(
        "--target_metadata_index",
        type=int,
        default=None,
        help="Metadata column containing the target label. Defaults to the dataset spec or column 1.",
    )
    parser.add_argument(
        "--spurious_metadata_index",
        type=int,
        default=None,
        help="Metadata column containing the spurious attribute. Defaults to the dataset spec or column 0.",
    )
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
    parser.add_argument("--splice_pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--splice_vocab", default="laion")
    parser.add_argument("--splice_vocab_size", type=int, default=10000)
    parser.add_argument("--splice_l1_penalty", type=float, default=0.25)
    parser.add_argument("--splice_score_cache_dir", default="outputs/splice_score_cache")
    parser.add_argument("--min_mean_weight", type=float, default=0.0)
    parser.add_argument("--label_penalty", type=float, default=1.0)
    parser.add_argument("--instability_penalty", type=float, default=1.0)
    parser.add_argument("--use_abs_score", action="store_true")
    parser.add_argument(
        "--require_consistent_spurious_direction",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Keep only concepts whose signed spurious effect has the same direction in every target class.",
    )
    parser.add_argument(
        "--deduplicate_concepts",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Automatically collapse simple lexical variants such as forest/forests.",
    )
    return parser.parse_args()


def configure_torch_backend(args: argparse.Namespace) -> None:
    if str(args.device).startswith("cuda") and args.disable_cudnn:
        cudnn.enabled = False
        cudnn.benchmark = False
        cudnn.deterministic = True


def load_splice(args: argparse.Namespace):
    configure_torch_backend(args)
    preprocess = splice.get_preprocess(args.splice_model, pretrained=args.splice_pretrained)
    vocabulary = splice.get_vocabulary(args.splice_vocab, args.splice_vocab_size)
    splicemodel = splice.load(
        args.splice_model,
        args.splice_vocab,
        args.splice_vocab_size,
        args.device,
        pretrained=args.splice_pretrained,
        l1_penalty=args.splice_l1_penalty,
        return_weights=True,
    )
    splicemodel.eval()
    for parameter in splicemodel.parameters():
        parameter.requires_grad = False
    return preprocess, vocabulary, splicemodel


@dataclass
class SparseConceptWeights:
    """Sparse per-image decompositions retained without an ``N x vocabulary`` allocation."""

    rows: torch.Tensor
    columns: torch.Tensor
    values: torch.Tensor
    n_rows: int
    n_columns: int

    def select_columns(self, concept_indices: list[int]) -> torch.Tensor:
        selected = torch.zeros((self.n_rows, len(concept_indices)), dtype=torch.float32)
        for output_column, concept_index in enumerate(concept_indices):
            mask = self.columns == concept_index
            if mask.any():
                selected[self.rows[mask], output_column] = self.values[mask]
        return selected


def resolve_metadata_indices(args: argparse.Namespace, dataset_spec: dict) -> tuple[int, int]:
    spurious_index = args.spurious_metadata_index
    target_index = args.target_metadata_index
    if spurious_index is None:
        spurious_index = dataset_spec.get("spurious_metadata_index", 0)
    if target_index is None:
        target_index = dataset_spec.get("target_metadata_index", 1)
    if spurious_index == target_index:
        raise ValueError("--spurious_metadata_index and --target_metadata_index must point to different metadata columns.")
    return int(spurious_index), int(target_index)


def metadata_value_names(dataset, metadata_index: int, values: torch.Tensor) -> dict[int, str]:
    fields = getattr(dataset, "_metadata_fields", None)
    metadata_map = getattr(dataset, "_metadata_map", None)
    if fields is None or metadata_map is None or metadata_index >= len(fields):
        return {int(value.item()): str(int(value.item())) for value in values}
    field_name = fields[metadata_index]
    names = metadata_map.get(field_name)
    if names is None:
        return {int(value.item()): str(int(value.item())) for value in values}
    result = {}
    for value in values:
        index = int(value.item())
        result[index] = names[index].strip() if index < len(names) else str(index)
    return result


def build_dataset_subset(args: argparse.Namespace):
    dataset_spec = DATASET_REGISTRY[args.dataset]
    full_dataset = dataset_spec["dataset"](args.data_folder) if "dataset" in dataset_spec else None
    if full_dataset is None:
        raise ValueError(
            f"Dataset spec for {args.dataset!r} must expose a 'dataset' class for concept discovery."
        )
    return dataset_spec, full_dataset, full_dataset.get_subset(args.split, transform=None)


def decompose_by_group(args: argparse.Namespace):
    preprocess, vocabulary, splicemodel = load_splice(args)
    dataset_spec, full_dataset, subset = build_dataset_subset(args)
    spurious_index, target_index = resolve_metadata_indices(args, dataset_spec)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
    )

    vocab_size = len(vocabulary)
    group_sums: dict[tuple[int, int], torch.Tensor] = {}
    group_counts: dict[tuple[int, int], int] = {}
    total_sum = torch.zeros(vocab_size, dtype=torch.float64)
    total_count = 0
    spurious_values = set()
    target_values = set()
    sparse_rows = []
    sparse_columns = []
    sparse_values = []
    row_offset = 0
    per_image_top_k = max(0, int(getattr(args, "per_image_top_k", 0)))
    audit_path = Path(args.out_path).with_suffix(f".per_image_top{per_image_top_k}.jsonl") if per_image_top_k else None
    audit_file = None
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_file = audit_path.open("w", encoding="utf-8")

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                images = torch.stack([preprocess(item[0]) for item in batch], dim=0).to(args.device)
                metadata = torch.stack([item[2] for item in batch], dim=0)
                weights = splicemodel.encode_image(images).detach().cpu().double()
                nonzero = torch.nonzero(weights, as_tuple=False)
                if nonzero.numel():
                    sparse_rows.append(nonzero[:, 0].long() + row_offset)
                    sparse_columns.append(nonzero[:, 1].long())
                    sparse_values.append(weights[nonzero[:, 0], nonzero[:, 1]].float())

                spurious = metadata[:, spurious_index].long()
                target = metadata[:, target_index].long()

                if audit_file is not None:
                    top_count = min(per_image_top_k, weights.shape[1])
                    top_values, top_indices = torch.topk(weights, k=top_count, dim=1)
                    subset_indices = getattr(subset, "indices", None)
                    for local_index in range(weights.shape[0]):
                        concepts = [
                            {"index": int(index), "concept": vocabulary[int(index)], "weight": round(float(value), 8)}
                            for index, value in zip(top_indices[local_index].tolist(), top_values[local_index].tolist())
                            if value > 0
                        ]
                        sample_index = row_offset + local_index
                        dataset_index = int(subset_indices[sample_index]) if subset_indices is not None else sample_index
                        audit_file.write(
                            json.dumps(
                                {
                                    "sample_index": sample_index,
                                    "dataset_index": dataset_index,
                                    "target": int(target[local_index]),
                                    "spurious": int(spurious[local_index]),
                                    "concepts": concepts,
                                }
                            )
                            + "\n"
                        )

                for spurious_value in torch.unique(spurious).tolist():
                    for target_value in torch.unique(target).tolist():
                        mask = (spurious == spurious_value) & (target == target_value)
                        if not mask.any():
                            continue
                        key = (int(spurious_value), int(target_value))
                        if key not in group_sums:
                            group_sums[key] = torch.zeros(vocab_size, dtype=torch.float64)
                            group_counts[key] = 0
                        group_sums[key] += weights[mask].sum(dim=0)
                        group_counts[key] += int(mask.sum().item())
                        spurious_values.add(int(spurious_value))
                        target_values.add(int(target_value))
                total_sum += weights.sum(dim=0)
                total_count += weights.shape[0]
                row_offset += weights.shape[0]
                if batch_idx % 10 == 0:
                    print(f"[INFO] Processed {total_count} images", flush=True)
    finally:
        if audit_file is not None:
            audit_file.close()
            print(f"[INFO] Wrote per-image concept audit to {audit_path}", flush=True)

    spurious_values_tensor = torch.tensor(sorted(spurious_values), dtype=torch.long)
    target_values_tensor = torch.tensor(sorted(target_values), dtype=torch.long)
    if len(spurious_values_tensor) < 2:
        raise ValueError("Conditional concept scoring requires at least two spurious attribute values.")
    if len(target_values_tensor) < 2:
        raise ValueError("Conditional concept scoring requires at least two target values.")
    group_means = {
        key: group_sums[key] / max(group_counts[key], 1)
        for key in group_sums
    }
    dataset_mean = total_sum / max(total_count, 1)
    metadata_names = {
        "spurious": metadata_value_names(full_dataset, spurious_index, spurious_values_tensor),
        "target": metadata_value_names(full_dataset, target_index, target_values_tensor),
    }
    sparse_weights = SparseConceptWeights(
        rows=torch.cat(sparse_rows) if sparse_rows else torch.empty(0, dtype=torch.long),
        columns=torch.cat(sparse_columns) if sparse_columns else torch.empty(0, dtype=torch.long),
        values=torch.cat(sparse_values) if sparse_values else torch.empty(0, dtype=torch.float32),
        n_rows=total_count,
        n_columns=vocab_size,
    )
    return (
        vocabulary,
        group_means,
        group_counts,
        dataset_mean,
        total_count,
        spurious_values_tensor,
        target_values_tensor,
        metadata_names,
        sparse_weights,
        full_dataset,
    )


def cache_discovered_scores(args, candidates: list[dict], weights: SparseConceptWeights, full_dataset) -> None:
    concept_indices = sorted(candidate["index"] for candidate in candidates)
    if not concept_indices:
        return
    selected_weights = weights.select_columns(concept_indices)
    cache_key = dataset_score_cache_key(args.dataset, full_dataset, args.split)
    for reduction in ("mean", "max"):
        config = SpliceConfig(
            concepts=",".join(str(index) for index in concept_indices),
            l1_penalty=args.splice_l1_penalty,
            vocab=args.splice_vocab,
            vocab_size=args.splice_vocab_size,
            model=args.splice_model,
            pretrained=args.splice_pretrained,
            score_reduction=reduction,
            score_cache_dir=args.splice_score_cache_dir,
        )
        scores = selected_weights.mean(dim=1) if reduction == "mean" else selected_weights.max(dim=1).values
        path = score_cache_path(config, weights.n_rows, concept_indices, cache_key, artifact="scores")
        save_score_cache(scores, path)
    vector_config = SpliceConfig(
        concepts=",".join(str(index) for index in concept_indices),
        l1_penalty=args.splice_l1_penalty,
        vocab=args.splice_vocab,
        vocab_size=args.splice_vocab_size,
        model=args.splice_model,
        pretrained=args.splice_pretrained,
        score_cache_dir=args.splice_score_cache_dir,
    )
    vector_path = score_cache_path(
        vector_config,
        weights.n_rows,
        concept_indices,
        cache_key,
        artifact="concept_weights",
    )
    save_score_cache(selected_weights, vector_path)
    print("[INFO] Discovery scores are ready for training; no second SpLiCE image pass is needed.", flush=True)


def concept_family_key(concept: str) -> str:
    """Return a conservative lexical family key without a curated concept list."""

    tokens = re.findall(r"[a-z0-9]+", concept.lower().replace("_", " ").replace("-", " "))
    normalized = []
    for token in tokens:
        if len(token) > 4 and token.endswith("ies"):
            token = f"{token[:-3]}y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        normalized.append(token)
    return " ".join(normalized)


def rank_concepts(
    vocabulary: list[str],
    group_means: dict[tuple[int, int], torch.Tensor],
    group_counts: dict[tuple[int, int], int],
    dataset_mean: torch.Tensor,
    spurious_values: torch.Tensor,
    target_values: torch.Tensor,
    metadata_names: dict,
    args: argparse.Namespace,
) -> list[dict]:
    spurious_list = [int(value.item()) for value in spurious_values]
    target_list = [int(value.item()) for value in target_values]
    required_groups = [
        (spurious_value, target_value)
        for spurious_value in spurious_list
        for target_value in target_list
    ]
    missing_groups = [key for key in required_groups if key not in group_means]
    if missing_groups:
        raise ValueError(f"Missing required spurious/target groups for conditional scoring: {missing_groups}")

    concept_means = torch.stack(
        [
            torch.stack([group_means[(spurious_value, target_value)] for spurious_value in spurious_list])
            for target_value in target_list
        ]
    )
    spurious_pairwise_differences = (
        concept_means[:, :, None, :] - concept_means[:, None, :, :]
    ).abs()
    spurious_pairs = torch.triu_indices(len(spurious_list), len(spurious_list), offset=1)
    spurious_effect_by_target = spurious_pairwise_differences[
        :, spurious_pairs[0], spurious_pairs[1], :
    ].mean(dim=1)
    spurious_effect = spurious_effect_by_target.mean(dim=0)
    signed_spurious_effect_by_target = concept_means[:, -1, :] - concept_means[:, 0, :]
    direction_epsilon = 1e-8
    direction_consistent = (
        (signed_spurious_effect_by_target > direction_epsilon).all(dim=0)
        | (signed_spurious_effect_by_target < -direction_epsilon).all(dim=0)
    )

    target_means_by_spurious = concept_means.permute(1, 0, 2)
    target_pairwise_differences = (
        target_means_by_spurious[:, :, None, :] - target_means_by_spurious[:, None, :, :]
    ).abs()
    target_pairs = torch.triu_indices(len(target_list), len(target_list), offset=1)
    target_effect_by_spurious = target_pairwise_differences[
        :, target_pairs[0], target_pairs[1], :
    ].mean(dim=1)
    target_effect = target_effect_by_spurious.mean(dim=0)
    instability = spurious_effect_by_target.std(dim=0, unbiased=False)
    signed_score = spurious_effect - args.label_penalty * target_effect - args.instability_penalty * instability
    score = signed_score.abs() if args.use_abs_score else signed_score
    eligible = dataset_mean >= args.min_mean_weight
    if getattr(args, "require_consistent_spurious_direction", False):
        eligible = eligible & direction_consistent

    group_mean_payload = {}
    for (spurious_value, target_value), means in group_means.items():
        key = (
            f"{metadata_names['target'].get(target_value, str(target_value))}_"
            f"on_{metadata_names['spurious'].get(spurious_value, str(spurious_value))}"
        )
        group_mean_payload[key] = means

    candidates = []
    seen_families: set[str] = set()
    for index in torch.argsort(score, descending=True).tolist():
        if not eligible[index] or score[index].item() <= 0:
            continue
        family_key = concept_family_key(vocabulary[index])
        if getattr(args, "deduplicate_concepts", False) and family_key in seen_families:
            continue
        seen_families.add(family_key)
        candidates.append(
            {
                "index": index,
                "concept": vocabulary[index],
                "score": round(score[index].item(), 8),
                "spurious_effect": round(spurious_effect[index].item(), 8),
                "target_effect": round(target_effect[index].item(), 8),
                "instability": round(instability[index].item(), 8),
                "direction_consistent": bool(direction_consistent[index].item()),
                "signed_spurious_effect_by_target": {
                    metadata_names["target"].get(target_value, str(target_value)): round(
                        signed_spurious_effect_by_target[target_idx, index].item(),
                        8,
                    )
                    for target_idx, target_value in enumerate(target_list)
                },
                "spurious_effect_by_target": {
                    metadata_names["target"].get(target_value, str(target_value)): round(
                        spurious_effect_by_target[target_idx, index].item(),
                        8,
                    )
                    for target_idx, target_value in enumerate(target_list)
                },
                "target_effect_by_spurious": {
                    metadata_names["spurious"].get(spurious_value, str(spurious_value)): round(
                        target_effect_by_spurious[spurious_idx, index].item(),
                        8,
                    )
                    for spurious_idx, spurious_value in enumerate(spurious_list)
                },
                "mean_weight": round(dataset_mean[index].item(), 8),
                "group_means": {
                    key: round(values[index].item(), 8)
                    for key, values in group_mean_payload.items()
                },
            }
        )
        if len(candidates) >= args.top_k:
            break
    return candidates


def write_outputs(args: argparse.Namespace, candidates: list[dict], group_counts: dict[tuple[int, int], int], total_count: int) -> None:
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "conditional_spurious_effect_minus_target_effect_minus_instability",
        "formula": "mean_y mean_{s_i<s_j} abs(E[c|s_i,y]-E[c|s_j,y]) - label_penalty * mean_s mean_{y_i<y_j} abs(E[c|s,y_i]-E[c|s,y_j]) - instability_penalty * std_y(pairwise_spurious_effect_y)",
        "dataset": args.dataset,
        "split": args.split,
        "total_count": total_count,
        "group_counts": {
            f"spurious_{spurious_value}_target_{target_value}": count
            for (spurious_value, target_value), count in group_counts.items()
        },
        "settings": {
            "top_k": args.top_k,
            "per_image_top_k": int(getattr(args, "per_image_top_k", 0)),
            "target_metadata_index": args.target_metadata_index,
            "spurious_metadata_index": args.spurious_metadata_index,
            "splice_model": args.splice_model,
            "splice_vocab": args.splice_vocab,
            "splice_vocab_size": args.splice_vocab_size,
            "splice_l1_penalty": args.splice_l1_penalty,
            "min_mean_weight": args.min_mean_weight,
            "label_penalty": args.label_penalty,
            "instability_penalty": args.instability_penalty,
            "use_abs_score": args.use_abs_score,
            "require_consistent_spurious_direction": bool(
                getattr(args, "require_consistent_spurious_direction", False)
            ),
            "deduplicate_concepts": bool(getattr(args, "deduplicate_concepts", False)),
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
            f"score={candidate['score']:.6f} spurious={candidate['spurious_effect']:.6f} "
            f"target={candidate['target_effect']:.6f} instability={candidate['instability']:.6f}"
        )


def main() -> None:
    args = parse_args()
    (
        vocabulary,
        group_means,
        group_counts,
        dataset_mean,
        total_count,
        spurious_values,
        target_values,
        metadata_names,
        per_image_weights,
        full_dataset,
    ) = decompose_by_group(args)
    candidates = rank_concepts(
        vocabulary,
        group_means,
        group_counts,
        dataset_mean,
        spurious_values,
        target_values,
        metadata_names,
        args,
    )
    write_outputs(args, candidates, group_counts, total_count)
    cache_discovered_scores(args, candidates, per_image_weights, full_dataset)


if __name__ == "__main__":
    main()
