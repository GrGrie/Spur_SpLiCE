from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
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
        "--ranking_method",
        default="conditional_group",
        choices=["conditional_group", "error_contrast", "gradient_probe", "intervention_utility"],
        help=(
            "conditional_group ranks by group-conditioned means and REQUIRES spurious metadata. "
            "error_contrast contrasts correctly classified against misclassified examples of the "
            "same class and needs class labels only. gradient_probe is the published gradient "
            "estimator, kept for comparison. intervention_utility is the proposed group-free "
            "method: it ranks exact held-out error repair against damage after concept ablation."
        ),
    )
    parser.add_argument(
        "--gradient_step_scale",
        type=float,
        default=0.1,
        help=(
            "Relative size of the error-correcting step: the displacement norm is this fraction of "
            "the embedding norm. Relative units keep the step scale-free across datasets."
        ),
    )
    parser.add_argument(
        "--gradient_score",
        default="indicator",
        choices=["indicator", "signed"],
        help=(
            "indicator counts sparse-support changes (faithful port of the published estimator, "
            "chance level 0.5). signed averages the continuous concept displacement instead, which "
            "degrades more gracefully when the support rarely changes."
        ),
    )
    parser.add_argument("--probe_c", type=float, default=1.0, help="Inverse L2 strength of the audit probe.")
    parser.add_argument("--probe_max_iter", type=int, default=2000)
    parser.add_argument(
        "--probe_cv_folds",
        type=int,
        default=5,
        help="Cross-validation folds used to obtain honest probe errors on the discovery split.",
    )
    parser.add_argument(
        "--utility_max_samples",
        type=int,
        default=20000,
        help="Maximum stratified audit-set size for intervention-utility cross-fitting; <=0 uses all rows.",
    )
    parser.add_argument(
        "--utility_candidate_pool",
        type=int,
        default=100,
        help="Number of positive individual-utility concepts considered by joint greedy selection.",
    )
    parser.add_argument(
        "--utility_min_repair",
        type=int,
        default=1,
        help="Minimum number of held-out wrong-to-correct transitions required for a candidate.",
    )
    parser.add_argument(
        "--utility_min_marginal",
        type=float,
        default=0.0,
        help="Stop greedy selection when the class-balanced soft utility gain is not above this value.",
    )
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--splice_vocab", default=splice.DEFAULT_VOCABULARY)
    parser.add_argument("--splice_vocab_size", type=int, default=splice.DEFAULT_VOCABULARY_SIZE)
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
        help="Automatically collapse simple lexical variants such as signal/signals.",
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

    def masked_column_means(self, row_mask: torch.Tensor) -> torch.Tensor:
        """Mean of every concept column over the selected rows.

        Accumulating straight from the sparse triplets keeps this usable on
        datasets where an ``n_rows x vocabulary`` dense matrix would not fit.
        """

        count = int(row_mask.sum())
        sums = torch.zeros(self.n_columns, dtype=torch.float64)
        if count == 0 or self.rows.numel() == 0:
            return sums.float()
        keep = row_mask[self.rows]
        if bool(keep.any()):
            sums.index_add_(0, self.columns[keep], self.values[keep].double())
        return (sums / count).float()

    def to_csr(self) -> sparse.csr_matrix:
        return sparse.csr_matrix(
            (
                self.values.numpy(),
                (self.rows.numpy(), self.columns.numpy()),
            ),
            shape=(self.n_rows, self.n_columns),
            dtype=np.float32,
        )


@dataclass
class UtilityProbeFold:
    """One held-out fold of a sparse linear probe used for exact interventions."""

    features: sparse.csc_matrix
    labels: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    coefficients: np.ndarray


def _stratified_audit_indices(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or len(labels) <= max_samples:
        return np.arange(len(labels), dtype=np.int64)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
    indices, _ = next(splitter.split(np.zeros(len(labels)), labels))
    return np.sort(indices.astype(np.int64))


def _expanded_probe_parameters(
    probe: LogisticRegression,
    features: sparse.spmatrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one logit row per class, including sklearn's binary special case."""

    raw_logits = probe.decision_function(features)
    if probe.coef_.shape[0] == 1:
        logits = np.column_stack((np.zeros(features.shape[0], dtype=np.float64), raw_logits))
        coefficients = np.vstack((np.zeros_like(probe.coef_[0]), probe.coef_[0]))
    else:
        logits = np.asarray(raw_logits, dtype=np.float64)
        coefficients = np.asarray(probe.coef_, dtype=np.float64)
    expected_classes = np.arange(coefficients.shape[0])
    if not np.array_equal(probe.classes_, expected_classes):
        raise ValueError(
            "Intervention utility expects contiguous integer class labels starting at zero; "
            f"got {probe.classes_.tolist()}."
        )
    return logits, coefficients


def fit_cross_fitted_sparse_probes(
    weights: SparseConceptWeights,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[UtilityProbeFold], dict]:
    """Fit L1 probes on fold complements and retain honest held-out logits."""

    matrix = weights.to_csr()
    targets = labels.numpy().astype(np.int64, copy=False)
    audit_indices = _stratified_audit_indices(
        targets,
        int(getattr(args, "utility_max_samples", 20000)),
        int(args.seed),
    )
    matrix = matrix[audit_indices]
    targets = targets[audit_indices]

    smallest_class = int(np.bincount(targets).min())
    folds = min(max(2, int(args.probe_cv_folds)), smallest_class)
    if folds < 2:
        raise ValueError("Intervention utility needs at least two examples in every target class.")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(args.seed))
    fold_states: list[UtilityProbeFold] = []
    all_predictions = np.empty_like(targets)
    active_features: set[int] = set()
    for fold_index, (train_indices, eval_indices) in enumerate(splitter.split(matrix, targets), start=1):
        probe = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=float(args.probe_c),
            fit_intercept=False,
            max_iter=int(args.probe_max_iter),
            random_state=int(args.seed) + fold_index,
        )
        probe.fit(matrix[train_indices], targets[train_indices])
        eval_features = matrix[eval_indices].tocsc()
        logits, coefficients = _expanded_probe_parameters(probe, eval_features)
        probabilities = softmax(logits, axis=1)
        predictions = logits.argmax(axis=1)
        all_predictions[eval_indices] = predictions
        active_features.update(np.flatnonzero(np.any(coefficients != 0, axis=0)).tolist())
        fold_states.append(
            UtilityProbeFold(
                features=eval_features,
                labels=targets[eval_indices],
                logits=logits,
                probabilities=probabilities,
                predictions=predictions,
                coefficients=coefficients,
            )
        )

    n_classes = int(targets.max()) + 1
    error_support = np.bincount(
        targets[all_predictions != targets], minlength=n_classes
    ).astype(np.int64)
    correct_support = np.bincount(
        targets[all_predictions == targets], minlength=n_classes
    ).astype(np.int64)
    diagnostics = {
        "probe_cv_folds": folds,
        "probe_cv_accuracy": round(float((all_predictions == targets).mean()), 6),
        "audit_sample_count": int(len(targets)),
        "full_sample_count": int(len(labels)),
        "active_probe_concepts": int(len(active_features)),
        "error_support_by_class": {str(index): int(value) for index, value in enumerate(error_support)},
        "correct_support_by_class": {str(index): int(value) for index, value in enumerate(correct_support)},
    }
    return fold_states, diagnostics


def _summarize_utility(
    soft_repair: np.ndarray,
    soft_damage: np.ndarray,
    repairs: np.ndarray,
    damages: np.ndarray,
    error_support: np.ndarray,
    correct_support: np.ndarray,
) -> dict[str, float | int]:
    error_classes = error_support > 0
    correct_classes = correct_support > 0
    if not bool(error_classes.any()) or not bool(correct_classes.any()):
        raise ValueError("Intervention utility requires both correct and misclassified held-out examples.")
    repair_soft_rate = float(np.mean(soft_repair[error_classes] / error_support[error_classes]))
    damage_soft_rate = float(np.mean(soft_damage[correct_classes] / correct_support[correct_classes]))
    repair_rate = float(np.mean(repairs[error_classes] / error_support[error_classes]))
    damage_rate = float(np.mean(damages[correct_classes] / correct_support[correct_classes]))
    smoothed_ratio = (repair_rate + 0.5 / max(int(error_support.sum()), 1)) / (
        damage_rate + 0.5 / max(int(correct_support.sum()), 1)
    )
    return {
        "score": repair_soft_rate - damage_soft_rate,
        "soft_repair_rate": repair_soft_rate,
        "soft_damage_rate": damage_soft_rate,
        "repair_rate": repair_rate,
        "damage_rate": damage_rate,
        "repair_damage_ratio": smoothed_ratio,
        "repaired": int(repairs.sum()),
        "damaged": int(damages.sum()),
    }


def _utility_support(folds: list[UtilityProbeFold]) -> tuple[np.ndarray, np.ndarray]:
    n_classes = folds[0].logits.shape[1]
    error_support = np.zeros(n_classes, dtype=np.int64)
    correct_support = np.zeros(n_classes, dtype=np.int64)
    for fold in folds:
        errors = fold.predictions != fold.labels
        error_support += np.bincount(fold.labels[errors], minlength=n_classes)
        correct_support += np.bincount(fold.labels[~errors], minlength=n_classes)
    return error_support, correct_support


def evaluate_single_concept_utility(
    folds: list[UtilityProbeFold],
    concept_index: int,
    error_support: np.ndarray,
    correct_support: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate one deletion in time proportional to its sparse activation count."""

    n_classes = len(error_support)
    soft_repair = np.zeros(n_classes, dtype=np.float64)
    soft_damage = np.zeros(n_classes, dtype=np.float64)
    repairs = np.zeros(n_classes, dtype=np.int64)
    damages = np.zeros(n_classes, dtype=np.int64)
    for fold in folds:
        coefficient = fold.coefficients[:, concept_index]
        if not np.any(coefficient):
            continue
        column = fold.features.getcol(concept_index).tocoo()
        if column.nnz == 0:
            continue
        rows = column.row
        ablated_logits = fold.logits[rows] - column.data[:, None] * coefficient[None, :]
        ablated_probabilities = softmax(ablated_logits, axis=1)
        ablated_predictions = ablated_logits.argmax(axis=1)
        labels = fold.labels[rows]
        base_predictions = fold.predictions[rows]
        base_true_probability = fold.probabilities[rows, labels]
        ablated_true_probability = ablated_probabilities[np.arange(len(rows)), labels]
        base_errors = base_predictions != labels
        repair_values = np.maximum(ablated_true_probability - base_true_probability, 0.0) * base_errors
        damage_values = np.maximum(base_true_probability - ablated_true_probability, 0.0) * ~base_errors
        soft_repair += np.bincount(labels, weights=repair_values, minlength=n_classes)
        soft_damage += np.bincount(labels, weights=damage_values, minlength=n_classes)
        repaired_mask = base_errors & (ablated_predictions == labels)
        damaged_mask = ~base_errors & (ablated_predictions != labels)
        repairs += np.bincount(labels[repaired_mask], minlength=n_classes)
        damages += np.bincount(labels[damaged_mask], minlength=n_classes)
    return _summarize_utility(
        soft_repair,
        soft_damage,
        repairs,
        damages,
        error_support,
        correct_support,
    )


def evaluate_concept_set_utility(
    folds: list[UtilityProbeFold],
    concept_indices: list[int],
    error_support: np.ndarray,
    correct_support: np.ndarray,
) -> dict[str, float | int]:
    n_classes = len(error_support)
    soft_repair = np.zeros(n_classes, dtype=np.float64)
    soft_damage = np.zeros(n_classes, dtype=np.float64)
    repairs = np.zeros(n_classes, dtype=np.int64)
    damages = np.zeros(n_classes, dtype=np.int64)
    for fold in folds:
        if concept_indices:
            contribution = fold.features[:, concept_indices] @ fold.coefficients[:, concept_indices].T
            ablated_logits = fold.logits - np.asarray(contribution)
        else:
            ablated_logits = fold.logits
        ablated_probabilities = softmax(ablated_logits, axis=1)
        ablated_predictions = ablated_logits.argmax(axis=1)
        row_indices = np.arange(len(fold.labels))
        base_true_probability = fold.probabilities[row_indices, fold.labels]
        ablated_true_probability = ablated_probabilities[row_indices, fold.labels]
        base_errors = fold.predictions != fold.labels
        repair_values = np.maximum(ablated_true_probability - base_true_probability, 0.0) * base_errors
        damage_values = np.maximum(base_true_probability - ablated_true_probability, 0.0) * ~base_errors
        soft_repair += np.bincount(fold.labels, weights=repair_values, minlength=n_classes)
        soft_damage += np.bincount(fold.labels, weights=damage_values, minlength=n_classes)
        repaired_mask = base_errors & (ablated_predictions == fold.labels)
        damaged_mask = ~base_errors & (ablated_predictions != fold.labels)
        repairs += np.bincount(fold.labels[repaired_mask], minlength=n_classes)
        damages += np.bincount(fold.labels[damaged_mask], minlength=n_classes)
    return _summarize_utility(
        soft_repair,
        soft_damage,
        repairs,
        damages,
        error_support,
        correct_support,
    )


def rank_concepts_by_intervention_utility(
    vocabulary: list[str],
    weights: SparseConceptWeights,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    """Select a non-redundant set whose exact deletion repairs held-out errors."""

    folds, diagnostics = fit_cross_fitted_sparse_probes(weights, labels, args)
    error_support, correct_support = _utility_support(folds)
    if not bool(error_support.any()):
        raise ValueError(
            "The cross-fitted sparse probe made no errors, so intervention utility is undefined. "
            "Reduce --probe_c or use a harder audit split."
        )

    active_indices = sorted(
        {
            int(index)
            for fold in folds
            for index in np.flatnonzero(np.any(fold.coefficients != 0, axis=0))
        }
    )
    individual: dict[int, dict[str, float | int]] = {}
    minimum_repairs = max(0, int(getattr(args, "utility_min_repair", 1)))
    for position, index in enumerate(active_indices, start=1):
        metrics = evaluate_single_concept_utility(folds, index, error_support, correct_support)
        if metrics["score"] > 0 and metrics["repaired"] >= minimum_repairs:
            individual[index] = metrics
        if position % 500 == 0:
            print(f"[INFO] Intervention-utility screening {position}/{len(active_indices)}", flush=True)

    pool_size = max(int(args.top_k), int(getattr(args, "utility_candidate_pool", 100)))
    pool = sorted(individual, key=lambda index: individual[index]["score"], reverse=True)[:pool_size]
    if not pool:
        raise ValueError(
            "No concept produced positive repair-minus-damage utility with the requested minimum repairs."
        )

    selected: list[int] = []
    selected_families: set[str] = set()
    current_score = 0.0
    candidates: list[dict] = []
    minimum_marginal = float(getattr(args, "utility_min_marginal", 0.0))
    while len(selected) < int(args.top_k):
        best_index = None
        best_metrics = None
        for index in pool:
            if index in selected:
                continue
            family = concept_family_key(vocabulary[index])
            if getattr(args, "deduplicate_concepts", False) and family in selected_families:
                continue
            metrics = evaluate_concept_set_utility(
                folds,
                selected + [index],
                error_support,
                correct_support,
            )
            if best_metrics is None or metrics["score"] > best_metrics["score"]:
                best_index = index
                best_metrics = metrics
        if best_index is None or best_metrics is None:
            break
        marginal = float(best_metrics["score"]) - current_score
        if marginal <= minimum_marginal:
            break
        selected.append(best_index)
        selected_families.add(concept_family_key(vocabulary[best_index]))
        current_score = float(best_metrics["score"])
        standalone = individual[best_index]
        candidates.append(
            {
                "index": best_index,
                "concept": vocabulary[best_index],
                "score": round(current_score, 8),
                "selection_step": len(selected),
                "marginal_utility": round(marginal, 8),
                "individual_utility": round(float(standalone["score"]), 8),
                "soft_repair_rate": round(float(best_metrics["soft_repair_rate"]), 8),
                "soft_damage_rate": round(float(best_metrics["soft_damage_rate"]), 8),
                "repair_rate": round(float(best_metrics["repair_rate"]), 8),
                "damage_rate": round(float(best_metrics["damage_rate"]), 8),
                "repair_damage_ratio": round(float(best_metrics["repair_damage_ratio"]), 8),
                "repaired": int(best_metrics["repaired"]),
                "damaged": int(best_metrics["damaged"]),
            }
        )

    diagnostics.update(
        {
            "positive_individual_candidates": len(individual),
            "greedy_candidate_pool": len(pool),
            "selected_count": len(candidates),
            "final_joint_utility": round(current_score, 8),
        }
    )
    return candidates, diagnostics


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


def splice_codes_from_embeddings(
    splicemodel,
    embeddings: torch.Tensor,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Decompose already-encoded CLIP embeddings, mirroring ``SPLICE.encode_image``.

    The decomposition is solved in the mean-centered space, so the same centering
    and renormalization must be applied here for the codes to be comparable with
    the ones produced by the regular image path.
    """

    codes = []
    with torch.no_grad():
        for chunk in embeddings.split(max(1, batch_size)):
            batch = F.normalize(chunk.to(device).float(), dim=1)
            centered = F.normalize(batch - splicemodel.image_mean, dim=1)
            codes.append(splicemodel.decompose(centered).detach().cpu().float())
    return torch.cat(codes, dim=0)


def collect_embeddings_and_codes(args: argparse.Namespace):
    """Single pass returning CLIP embeddings, sparse SpLiCE codes, and class labels.

    Unlike :func:`decompose_by_group` this never touches the spurious metadata
    column, which is what makes gradient-probe discovery group-annotation free.
    """

    preprocess, vocabulary, splicemodel = load_splice(args)
    _, full_dataset, subset = build_dataset_subset(args)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
    )

    vocab_size = len(vocabulary)
    embeddings: list[torch.Tensor] = []
    labels: list[int] = []
    sparse_rows, sparse_columns, sparse_values = [], [], []
    row_offset = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = torch.stack([preprocess(item[0]) for item in batch], dim=0).to(args.device)
            encoded = F.normalize(splicemodel.clip.encode_image(images).float(), dim=1)
            centered = F.normalize(encoded - splicemodel.image_mean, dim=1)
            weights = splicemodel.decompose(centered).detach().cpu().float()

            embeddings.append(encoded.detach().cpu())
            labels.extend(int(item[1]) for item in batch)
            nonzero = torch.nonzero(weights, as_tuple=False)
            if nonzero.numel():
                sparse_rows.append(nonzero[:, 0].long() + row_offset)
                sparse_columns.append(nonzero[:, 1].long())
                sparse_values.append(weights[nonzero[:, 0], nonzero[:, 1]])
            row_offset += weights.shape[0]
            if batch_index % 10 == 0:
                print(f"[INFO] Encoded {row_offset} images", flush=True)

    sparse_weights = SparseConceptWeights(
        rows=torch.cat(sparse_rows) if sparse_rows else torch.empty(0, dtype=torch.long),
        columns=torch.cat(sparse_columns) if sparse_columns else torch.empty(0, dtype=torch.long),
        values=torch.cat(sparse_values) if sparse_values else torch.empty(0, dtype=torch.float32),
        n_rows=row_offset,
        n_columns=vocab_size,
    )
    return (
        vocabulary,
        splicemodel,
        torch.cat(embeddings, dim=0),
        torch.tensor(labels, dtype=torch.long),
        sparse_weights,
        full_dataset,
    )


def fit_audit_probe(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[LogisticRegression, torch.Tensor, float]:
    """Fit the audit probe and return honest, cross-validated predictions.

    Errors drive both group-free ranking methods, so they must not be hidden by a
    probe that has memorized the discovery split.
    """

    features = embeddings.numpy().astype(np.float64)
    targets = labels.numpy()
    probe = LogisticRegression(
        C=args.probe_c,
        max_iter=args.probe_max_iter,
        random_state=args.seed,
    )
    folds = max(2, int(args.probe_cv_folds))
    smallest_class = int(np.bincount(targets).min())
    folds = min(folds, smallest_class) if smallest_class >= 2 else 2
    predictions = cross_val_predict(probe, features, targets, cv=folds)
    probe.fit(features, targets)
    accuracy = float((predictions == targets).mean())
    print(
        f"[INFO] Audit probe: {folds}-fold cross-validated accuracy {accuracy:.4f} "
        f"({int((predictions != targets).sum())} errors of {len(targets)})",
        flush=True,
    )
    return probe, torch.from_numpy(predictions).long(), accuracy


def error_correcting_displacement(
    probe: LogisticRegression,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Return the gradient step that would fix each example.

    The magnitude is expressed relative to the embedding norm so that no
    per-dataset step size has to be tuned.
    """

    features = embeddings.numpy().astype(np.float64)
    targets = labels.numpy()
    n_classes = int(targets.max()) + 1
    probabilities = probe.predict_proba(features)
    coefficients = probe.coef_
    if n_classes == 2 and coefficients.shape[0] == 1:
        # Binary sklearn keeps a single weight vector for the positive class.
        residual = (probabilities[:, 1] - targets)[:, None]
        gradients = residual * coefficients
    else:
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(len(targets)), targets] = 1.0
        gradients = (probabilities - one_hot) @ coefficients

    gradients = torch.from_numpy(gradients).float()
    gradient_norms = gradients.norm(dim=1, keepdim=True).clamp_min(1e-12)
    embedding_norms = embeddings.norm(dim=1, keepdim=True)
    # z' = z - d * grad, with d chosen so that ||d * grad|| = scale * ||z||.
    return -args.gradient_step_scale * embedding_norms * gradients / gradient_norms


def rank_concepts_by_error_contrast(
    vocabulary: list[str],
    weights: SparseConceptWeights,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    probe_accuracy: float,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    """Rank concepts by how much they differ between correct and misclassified examples.

    A biased model fails precisely where the shortcut points the wrong way, so its
    errors are enriched in the conflict group. Conditioning on ``(class, correct)``
    versus ``(class, error)`` therefore stands in for conditioning on the minority
    group, and needs class labels only. A core concept is present whether or not
    the example was classified correctly, so it cancels; a shortcut concept does
    not, because its value is what made the example easy or hard.
    """

    class_values = [int(value) for value in torch.unique(labels).tolist()]
    signed_effect = torch.zeros(len(class_values), weights.n_columns, dtype=torch.float32)
    class_support: dict[int, dict[str, int]] = {}
    for position, class_value in enumerate(class_values):
        in_class = labels == class_value
        correct_mask = in_class & (predictions == class_value)
        error_mask = in_class & (predictions != class_value)
        class_support[class_value] = {
            "correct": int(correct_mask.sum()),
            "errors": int(error_mask.sum()),
        }
        if not bool(correct_mask.any()) or not bool(error_mask.any()):
            continue
        signed_effect[position] = weights.masked_column_means(correct_mask) - weights.masked_column_means(
            error_mask
        )

    usable = [position for position, value in enumerate(class_values) if all(class_support[value].values())]
    if not usable:
        raise ValueError(
            "Every class is either perfectly classified or never correct, so no error contrast "
            "exists. Weaken the audit probe with a smaller --probe_c."
        )
    signed_effect = signed_effect[usable]
    score = signed_effect.abs().mean(dim=0)

    # A shortcut helps one class and hurts the other, so its signed effect flips
    # across classes. Core concepts drift the same way for everyone.
    direction_epsilon = 1e-12
    if len(usable) >= 2:
        positive = (signed_effect > direction_epsilon).any(dim=0)
        negative = (signed_effect < -direction_epsilon).any(dim=0)
        sign_flips = positive & negative
    else:
        sign_flips = torch.ones(weights.n_columns, dtype=torch.bool)
    if getattr(args, "require_consistent_spurious_direction", False):
        score = score * sign_flips.float()

    overall_mean = weights.masked_column_means(torch.ones_like(labels, dtype=torch.bool))
    eligible = overall_mean >= getattr(args, "min_mean_weight", 0.0)

    candidates: list[dict] = []
    seen_families: set[str] = set()
    for index in torch.argsort(score, descending=True).tolist():
        if not bool(eligible[index]) or score[index].item() <= 0:
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
                "sign_flips_across_classes": bool(sign_flips[index].item()),
                "signed_effect_by_class": {
                    str(class_values[usable[position]]): round(signed_effect[position, index].item(), 8)
                    for position in range(len(usable))
                },
                "mean_weight": round(overall_mean[index].item(), 8),
            }
        )
        if len(candidates) >= args.top_k:
            break

    diagnostics = {
        "probe_cv_accuracy": round(probe_accuracy, 6),
        "error_count": int((predictions != labels).sum()),
        "total_count": int(labels.numel()),
        "class_support": class_support,
    }
    return candidates, diagnostics


def rank_concepts_by_gradient_probe(
    vocabulary: list[str],
    splicemodel,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    probe: LogisticRegression,
    probe_accuracy: float,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    """Rank concepts by the asymmetric signature of error-correcting gradients.

    This is a direct port of the published gradient-probe estimator, kept as a
    comparison point. Note that the correction direction is the probe's weight
    vector, which mixes shortcut and core evidence, so in this sparse text-concept
    setting it ranks core concepts highly as well; see the report for the
    controlled comparison against :func:`rank_concepts_by_error_contrast`.
    """

    displacement = error_correcting_displacement(probe, embeddings, labels, args)
    error_mask = predictions != labels
    if not bool(error_mask.any()):
        raise ValueError(
            "The audit probe made no cross-validated errors, so no gradient signal is available. "
            "Lower --probe_c to weaken the probe, or discover concepts on a harder split."
        )

    error_indices = torch.nonzero(error_mask, as_tuple=False).view(-1)
    original = embeddings[error_indices]
    corrected = original + displacement[error_indices]
    # Only misclassified examples are decomposed twice: the audit set is small.
    codes = splice_codes_from_embeddings(splicemodel, original, args.device, args.batch_size)
    corrected_codes = splice_codes_from_embeddings(splicemodel, corrected, args.device, args.batch_size)
    delta = corrected_codes - codes

    activated = (codes <= 0) & (corrected_codes > 0)
    suppressed = (codes > 0) & (corrected_codes <= 0)
    print(
        f"[INFO] Support changes over {len(error_indices)} errors: "
        f"{int(activated.sum())} activations, {int(suppressed.sum())} suppressions",
        flush=True,
    )
    if args.gradient_score == "indicator" and int(activated.sum()) + int(suppressed.sum()) == 0:
        raise ValueError(
            "The correction step never changed the sparse support, so indicator scores are all zero. "
            "Increase --gradient_step_scale or use --gradient_score signed."
        )

    error_labels = labels[error_indices]
    error_predictions = predictions[error_indices]
    class_values = torch.unique(labels).tolist()
    vocab_size = len(vocabulary)
    per_class_scores = torch.zeros(len(class_values), vocab_size, dtype=torch.float32)
    class_support = {}

    for position, class_value in enumerate(class_values):
        # False negatives of this class: the true class was missed.
        false_negative = error_labels == class_value
        # False positives: another class was misread as this one.
        false_positive = error_predictions == class_value
        class_support[int(class_value)] = {
            "false_negatives": int(false_negative.sum()),
            "false_positives": int(false_positive.sum()),
        }
        terms = []
        if bool(false_negative.any()):
            if args.gradient_score == "indicator":
                terms.append(activated[false_negative].float().mean(dim=0))
            else:
                terms.append(delta[false_negative].mean(dim=0))
        if bool(false_positive.any()):
            if args.gradient_score == "indicator":
                terms.append(suppressed[false_positive].float().mean(dim=0))
            else:
                terms.append(-delta[false_positive].mean(dim=0))
        if terms:
            per_class_scores[position] = torch.stack(terms).mean(dim=0)

    # A concept only has to look spurious for one class to be worth removing.
    score, best_class_position = per_class_scores.max(dim=0)
    activation_rate = activated.float().mean(dim=0)
    suppression_rate = suppressed.float().mean(dim=0)
    mean_weight = codes.mean(dim=0)
    # The published 0.55 cut-off assumes dense NNLS coefficients that flip support
    # often. l1-sparse SpLiCE codes change support rarely, so an absolute floor
    # rejects everything; rank and take top-K instead, which also avoids a
    # per-dataset threshold.
    minimum_score = 0.0

    candidates: list[dict] = []
    seen_families: set[str] = set()
    for index in torch.argsort(score, descending=True).tolist():
        if score[index].item() <= minimum_score:
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
                "best_class": int(class_values[int(best_class_position[index].item())]),
                "per_class_score": {
                    str(int(class_value)): round(per_class_scores[position, index].item(), 8)
                    for position, class_value in enumerate(class_values)
                },
                "activation_rate_on_errors": round(activation_rate[index].item(), 8),
                "suppression_rate_on_errors": round(suppression_rate[index].item(), 8),
                "mean_delta_on_errors": round(delta[:, index].mean().item(), 8),
                "mean_weight_on_errors": round(mean_weight[index].item(), 8),
            }
        )
        if len(candidates) >= args.top_k:
            break

    diagnostics = {
        "probe_cv_accuracy": round(probe_accuracy, 6),
        "error_count": int(error_indices.numel()),
        "total_count": int(labels.numel()),
        "class_support": class_support,
        "minimum_score": minimum_score,
    }
    return candidates, diagnostics


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


def write_outputs(
    args: argparse.Namespace,
    candidates: list[dict],
    group_counts: dict[tuple[int, int], int],
    total_count: int,
    diagnostics: dict | None = None,
) -> None:
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Callers such as splice_cbm.py build this namespace by hand, so every
    # gradient-probe field has to degrade to the historical default.
    ranking_method = getattr(args, "ranking_method", "conditional_group")
    if ranking_method == "intervention_utility":
        method = "cross_fitted_intervention_repair_damage_utility"
        formula = (
            "greedy_S class_balanced_mean_{errors} positive_delta_p_true(delete S) - "
            "class_balanced_mean_{correct} positive_negative_delta_p_true(delete S)"
        )
    elif ranking_method == "error_contrast":
        method = "error_conditioned_contrast"
        formula = "mean_y abs( E[c | y, predicted correctly] - E[c | y, misclassified] )"
    elif ranking_method == "gradient_probe":
        method = "gradient_probe_error_correction_asymmetry"
        formula = (
            "mean_y 0.5 * ( E_{i in FN(y)}[activated_k(i)] + E_{i in FP(y)}[suppressed_k(i)] ), "
            "activation measured on SpLiCE codes before/after z' = z - d * grad_z L"
        )
    else:
        method = "conditional_spurious_effect_minus_target_effect_minus_instability"
        formula = (
            "mean_y mean_{s_i<s_j} abs(E[c|s_i,y]-E[c|s_j,y]) - label_penalty * mean_s "
            "mean_{y_i<y_j} abs(E[c|s,y_i]-E[c|s,y_j]) - instability_penalty * "
            "std_y(pairwise_spurious_effect_y)"
        )
    payload = {
        "method": method,
        "formula": formula,
        "uses_spurious_metadata": ranking_method == "conditional_group",
        "dataset": args.dataset,
        "split": args.split,
        "total_count": total_count,
        "group_counts": {
            f"spurious_{spurious_value}_target_{target_value}": count
            for (spurious_value, target_value), count in group_counts.items()
        },
        "settings": {
            "top_k": args.top_k,
            "ranking_method": ranking_method,
            "gradient_step_scale": getattr(args, "gradient_step_scale", None),
            "gradient_score": getattr(args, "gradient_score", None),
            "probe_c": getattr(args, "probe_c", None),
            "probe_cv_folds": getattr(args, "probe_cv_folds", None),
            "utility_max_samples": getattr(args, "utility_max_samples", None),
            "utility_candidate_pool": getattr(args, "utility_candidate_pool", None),
            "utility_min_repair": getattr(args, "utility_min_repair", None),
            "utility_min_marginal": getattr(args, "utility_min_marginal", None),
            "seed": getattr(args, "seed", None),
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
    if diagnostics is not None:
        payload["diagnostics"] = diagnostics
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
        head = f"  {candidate['index']:5d} {candidate['concept']:<30} score={candidate['score']:.6f}"
        if "marginal_utility" in candidate:
            print(
                f"{head} marginal={candidate['marginal_utility']:+.6f} "
                f"repair={candidate['repaired']} damage={candidate['damaged']} "
                f"ratio={candidate['repair_damage_ratio']:.3f}"
            )
        elif "spurious_effect" in candidate:
            print(
                f"{head} spurious={candidate['spurious_effect']:.6f} "
                f"target={candidate['target_effect']:.6f} instability={candidate['instability']:.6f}"
            )
        elif "signed_effect_by_class" in candidate:
            effects = " ".join(
                f"y{key}={value:+.6f}" for key, value in candidate["signed_effect_by_class"].items()
            )
            print(f"{head} {effects} flips={candidate['sign_flips_across_classes']}")
        else:
            print(
                f"{head} activated={candidate['activation_rate_on_errors']:.4f} "
                f"suppressed={candidate['suppression_rate_on_errors']:.4f} "
                f"class={candidate['best_class']}"
            )


def main() -> None:
    args = parse_args()
    if args.ranking_method in {"error_contrast", "gradient_probe", "intervention_utility"}:
        (
            vocabulary,
            splicemodel,
            embeddings,
            labels,
            per_image_weights,
            full_dataset,
        ) = collect_embeddings_and_codes(args)
        if args.ranking_method == "intervention_utility":
            candidates, diagnostics = rank_concepts_by_intervention_utility(
                vocabulary,
                per_image_weights,
                labels,
                args,
            )
        else:
            probe, predictions, probe_accuracy = fit_audit_probe(embeddings, labels, args)
        if args.ranking_method == "error_contrast":
            candidates, diagnostics = rank_concepts_by_error_contrast(
                vocabulary,
                per_image_weights,
                labels,
                predictions,
                probe_accuracy,
                args,
            )
        elif args.ranking_method == "gradient_probe":
            candidates, diagnostics = rank_concepts_by_gradient_probe(
                vocabulary,
                splicemodel,
                embeddings,
                labels,
                predictions,
                probe,
                probe_accuracy,
                args,
            )
        if not candidates:
            raise ValueError(
                f"{args.ranking_method} discovery found no concept with a positive score. "
                "Check the audit-probe diagnostics printed above."
            )
        write_outputs(args, candidates, {}, int(labels.numel()), diagnostics=diagnostics)
        cache_discovered_scores(args, candidates, per_image_weights, full_dataset)
        return

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
