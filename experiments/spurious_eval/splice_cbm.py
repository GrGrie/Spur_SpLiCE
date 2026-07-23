from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

import splice
from experiments.spurious_eval.datasets.registry import DATASET_REGISTRY
from experiments.spurious_eval.evaluation_protocol import resolve_evaluation_split
from experiments.spurious_eval.metrics import compute_group_metrics
from splice.ssl_regularization import identity_collate, resolve_concept_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Sparse SpLiCE concept-bottleneck intervention baseline")
    parser.add_argument("--dataset", default="waterbirds", choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--data_folder", default="./datasets")
    parser.add_argument("--train_split", default="train", choices=["train", "ds_train", "val"])
    parser.add_argument(
        "--eval_split",
        default=None,
        choices=["val", "test"],
        help="Evaluation split. Defaults to val; test requires --final_test.",
    )
    parser.add_argument(
        "--final_test",
        action="store_true",
        help="Evaluate a locked final configuration on test instead of the validation default.",
    )
    parser.add_argument(
        "--intervention_concepts",
        default="auto",
        help="Comma-separated concepts/indices, or 'auto' (default) for metadata-conditioned discovery.",
    )
    parser.add_argument("--auto_top_k", type=int, default=10)
    parser.add_argument("--auto_out_dir", default="outputs")
    parser.add_argument("--auto_label_penalty", type=float, default=1.0)
    parser.add_argument("--auto_instability_penalty", type=float, default=1.0)
    parser.add_argument("--splice_score_cache_dir", default="outputs/splice_score_cache")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--splice_model", default="open_clip:ViT-B-32")
    parser.add_argument("--splice_pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--splice_vocab", default="laion")
    parser.add_argument("--splice_vocab_size", type=int, default=10000)
    parser.add_argument("--splice_l1_penalty", type=float, default=0.25)
    parser.add_argument("--probe_c", type=float, default=1.0)
    parser.add_argument("--probe_max_iter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", default="outputs/splice_cbm_cache")
    parser.add_argument("--out_path", default="")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_name", default="Spur_SpLiCE")
    parser.add_argument("--entity", default="gsgrechkin-rptu")
    args = parser.parse_args()
    try:
        args.eval_split = resolve_evaluation_split(args.eval_split, args.final_test)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def cache_stem(args: argparse.Namespace, split: str, dataset_size: int) -> Path:
    fingerprint = {
        "version": 1,
        "dataset": args.dataset,
        "data_folder": str(Path(args.data_folder).expanduser().resolve()),
        "split": split,
        "dataset_size": dataset_size,
        "model": args.splice_model,
        "pretrained": args.splice_pretrained,
        "vocab": args.splice_vocab,
        "vocab_size": args.splice_vocab_size,
        "l1_penalty": args.splice_l1_penalty,
    }
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    return Path(args.cache_dir) / f"{args.dataset}_{split}_{digest}"


def decompose_subset(args, subset, split: str, preprocess, splicemodel):
    stem = cache_stem(args, split, len(subset))
    matrix_path = stem.with_suffix(".npz")
    metadata_path = stem.with_suffix(".metadata.npz")
    if matrix_path.exists() and metadata_path.exists():
        matrix = sparse.load_npz(matrix_path)
        payload = np.load(metadata_path)
        if matrix.shape[0] == len(subset):
            print(f"[INFO] Loaded cached sparse SpLiCE matrix {matrix.shape} from {matrix_path}")
            return matrix, payload["labels"], payload["metadata"]

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
    )
    matrix_chunks = []
    labels = []
    metadata_rows = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = torch.stack([preprocess(item[0]) for item in batch]).to(args.device)
            weights = splicemodel.encode_image(images).detach().cpu()
            nonzero = torch.nonzero(weights, as_tuple=False)
            chunk = sparse.csr_matrix(
                (
                    weights[nonzero[:, 0], nonzero[:, 1]].numpy(),
                    (nonzero[:, 0].numpy(), nonzero[:, 1].numpy()),
                ),
                shape=tuple(weights.shape),
                dtype=np.float32,
            )
            matrix_chunks.append(chunk)
            labels.extend(int(item[1]) for item in batch)
            metadata_rows.extend(item[2].tolist() for item in batch)
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
                print(f"[INFO] SpLiCE-CBM decomposition {split}: {batch_index}/{len(loader)} batches", flush=True)

    matrix = sparse.vstack(matrix_chunks, format="csr")
    labels_array = np.asarray(labels, dtype=np.int64)
    metadata_array = np.asarray(metadata_rows, dtype=np.int64)
    stem.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(matrix_path, matrix)
    np.savez_compressed(metadata_path, labels=labels_array, metadata=metadata_array)
    print(f"[INFO] Cached sparse SpLiCE matrix at {matrix_path}")
    return matrix, labels_array, metadata_array


def save_decomposition_cache(args, split: str, matrix, labels, metadata) -> None:
    stem = cache_stem(args, split, matrix.shape[0])
    stem.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(stem.with_suffix(".npz"), matrix.tocsr())
    np.savez_compressed(
        stem.with_suffix(".metadata.npz"),
        labels=np.asarray(labels, dtype=np.int64),
        metadata=np.asarray(metadata, dtype=np.int64),
    )


def automatically_discover_interventions(args, train_subset):
    from scripts.tools import discover_splice_spurious_concepts as discovery

    discovery_path = Path(args.auto_out_dir) / f"{args.dataset}_splice_concepts.json"
    concepts_path = discovery_path.with_suffix(".concepts.txt")
    stem = cache_stem(args, args.train_split, len(train_subset))
    matrix_path = stem.with_suffix(".npz")
    metadata_path = stem.with_suffix(".metadata.npz")
    discovery_matches = False
    if discovery_path.exists():
        previous = json.loads(discovery_path.read_text(encoding="utf-8"))
        settings = previous.get("settings", {})
        discovery_matches = (
            previous.get("dataset") == args.dataset
            and previous.get("split") == args.train_split
            and settings.get("top_k") == args.auto_top_k
            and settings.get("splice_model") == args.splice_model
            and settings.get("splice_vocab") == args.splice_vocab
            and settings.get("splice_vocab_size") == args.splice_vocab_size
            and settings.get("splice_l1_penalty") == args.splice_l1_penalty
            and settings.get("label_penalty") == args.auto_label_penalty
            and settings.get("instability_penalty") == args.auto_instability_penalty
        )
    if discovery_matches and concepts_path.exists() and matrix_path.exists() and metadata_path.exists():
        cached_metadata = np.load(metadata_path)
        cached_matrix = sparse.load_npz(matrix_path)
        if cached_matrix.shape[0] == len(train_subset):
            print(f"[INFO] Reusing automatic SpLiCE-CBM discovery from {concepts_path}")
            return (
                concepts_path.read_text(encoding="utf-8").strip(),
                cached_matrix,
                cached_metadata["labels"],
            )
    discovery_args = argparse.Namespace(
        dataset=args.dataset,
        data_folder=args.data_folder,
        split=args.train_split,
        out_path=str(discovery_path),
        top_k=args.auto_top_k,
        per_image_top_k=0,
        target_metadata_index=None,
        spurious_metadata_index=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        disable_cudnn=True,
        splice_model=args.splice_model,
        splice_pretrained=args.splice_pretrained,
        splice_vocab=args.splice_vocab,
        splice_vocab_size=args.splice_vocab_size,
        splice_l1_penalty=args.splice_l1_penalty,
        splice_score_cache_dir=args.splice_score_cache_dir,
        min_mean_weight=0.0,
        label_penalty=args.auto_label_penalty,
        instability_penalty=args.auto_instability_penalty,
        use_abs_score=False,
    )
    (
        vocabulary,
        group_means,
        group_counts,
        dataset_mean,
        total_count,
        spurious_values,
        target_values,
        metadata_names,
        sparse_weights,
        discovery_dataset,
    ) = discovery.decompose_by_group(discovery_args)
    candidates = discovery.rank_concepts(
        vocabulary,
        group_means,
        group_counts,
        dataset_mean,
        spurious_values,
        target_values,
        metadata_names,
        discovery_args,
    )
    if not candidates:
        raise ValueError("Automatic SpLiCE-CBM discovery found no positive-scoring concepts.")
    discovery.write_outputs(discovery_args, candidates, group_counts, total_count)
    discovery.cache_discovered_scores(discovery_args, candidates, sparse_weights, discovery_dataset)

    train_matrix = sparse.csr_matrix(
        (
            sparse_weights.values.numpy(),
            (sparse_weights.rows.numpy(), sparse_weights.columns.numpy()),
        ),
        shape=(sparse_weights.n_rows, sparse_weights.n_columns),
        dtype=np.float32,
    )
    train_labels = train_subset.y_array.numpy()
    train_metadata = train_subset.metadata_array.numpy()
    if train_matrix.shape[0] != len(train_labels):
        raise ValueError("Automatic discovery and SpLiCE-CBM train subsets have different sizes.")
    save_decomposition_cache(args, args.train_split, train_matrix, train_labels, train_metadata)
    return (
        ",".join(candidate["concept"] for candidate in candidates),
        train_matrix,
        train_labels,
    )


def metric_payload(predictions: np.ndarray, labels: np.ndarray, metadata: np.ndarray) -> dict[str, object]:
    metrics = compute_group_metrics(
        torch.from_numpy(predictions),
        torch.from_numpy(labels),
        torch.from_numpy(metadata),
    )
    return {
        "average_accuracy": metrics.average,
        "worst_group_accuracy": metrics.worst_group,
        "best_group_accuracy": metrics.best_group,
        "group_accuracy": metrics.group_accuracy.tolist(),
        "group_counts": metrics.group_counts.tolist(),
    }


def zero_sparse_columns(matrix: sparse.csr_matrix, indices: list[int]) -> sparse.csr_matrix:
    if not indices:
        return matrix
    keep = np.ones(matrix.shape[1], dtype=np.float32)
    keep[indices] = 0.0
    return matrix.multiply(keep).tocsr()


def main(args: argparse.Namespace | None = None) -> dict[str, object]:
    args = parse_args() if args is None else args
    dataset_spec = DATASET_REGISTRY[args.dataset]
    full_dataset = dataset_spec["dataset"](args.data_folder)
    train_subset = full_dataset.get_subset(args.train_split, transform=None)
    eval_subset = full_dataset.get_subset(args.eval_split, transform=None)

    vocabulary = splice.get_vocabulary(args.splice_vocab, args.splice_vocab_size)
    auto_train_payload = None
    if args.intervention_concepts.strip().lower() == "auto":
        concepts, train_weights, train_labels = automatically_discover_interventions(args, train_subset)
        args.intervention_concepts = concepts
        auto_train_payload = (train_weights, train_labels)

    preprocess = splice.get_preprocess(args.splice_model, pretrained=args.splice_pretrained)
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
    intervention_indices = resolve_concept_indices(args.intervention_concepts, vocabulary)

    if auto_train_payload is None:
        train_weights, train_labels, _ = decompose_subset(
            args, train_subset, args.train_split, preprocess, splicemodel
        )
    else:
        train_weights, train_labels = auto_train_payload
    eval_weights, eval_labels, eval_metadata = decompose_subset(
        args, eval_subset, args.eval_split, preprocess, splicemodel
    )

    probe = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=args.probe_c,
        fit_intercept=False,
        max_iter=args.probe_max_iter,
        random_state=args.seed,
    )
    probe.fit(train_weights, train_labels)
    baseline = metric_payload(probe.predict(eval_weights), eval_labels, eval_metadata)

    representation_intervention = metric_payload(
        probe.predict(zero_sparse_columns(eval_weights, intervention_indices)),
        eval_labels,
        eval_metadata,
    )
    original_coefficients = probe.coef_.copy()
    if intervention_indices:
        probe.coef_[:, intervention_indices] = 0.0
    probe_intervention = metric_payload(probe.predict(eval_weights), eval_labels, eval_metadata)
    probe.coef_ = original_coefficients

    results = {
        "dataset": args.dataset,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "intervention_concepts": [
            {"index": index, "concept": vocabulary[index]} for index in intervention_indices
        ],
        "baseline": baseline,
        "representation_intervention": representation_intervention,
        "probe_intervention": probe_intervention,
        "nonzero_probe_concepts": int(np.count_nonzero(original_coefficients)),
    }
    out_path = Path(args.out_path) if args.out_path else Path("outputs") / f"{args.dataset}_splice_cbm_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

    if args.use_wandb:
        import wandb

        run = wandb.init(project=args.wandb_name, entity=args.entity, config=vars(args), name=f"{args.dataset}_CBM")
        run.log(
            {
                "SpLiCE-CBM average accuracy": baseline["average_accuracy"] * 100,
                "SpLiCE-CBM worst-group accuracy": baseline["worst_group_accuracy"] * 100,
                "SpLiCE-CBM representation intervention average accuracy": representation_intervention["average_accuracy"] * 100,
                "SpLiCE-CBM representation intervention worst-group accuracy": representation_intervention["worst_group_accuracy"] * 100,
                "SpLiCE-CBM probe intervention average accuracy": probe_intervention["average_accuracy"] * 100,
                "SpLiCE-CBM probe intervention worst-group accuracy": probe_intervention["worst_group_accuracy"] * 100,
            }
        )
        run.finish()
    return results


if __name__ == "__main__":
    main()
