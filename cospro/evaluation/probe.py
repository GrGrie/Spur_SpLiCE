"""The linear probe: what it measures, where its outputs go and how it reaches W&B.

The headline number of this project comes from a frozen encoder plus a linear classifier, so the
measurement is separated from everything around it. :func:`evaluate_probe` takes an encoder, a
dataset and :class:`ProbeOptions` and returns a :class:`ProbeResult`; it writes no file, touches no
W&B run and reads no run identity. :func:`persist_probe_result` and :func:`log_probe_result` take
that result and do one side effect each. :func:`probe_checkpoint` is the composition every caller
wants: load an encoder from a checkpoint, measure it, store the result, log it.

The metric names are historical on purpose: ``run.json``, ``collect_results`` and the paper registry
read them. ``cospro.tracking`` maps them to the canonical W&B keys.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset

from cospro.tracking import canonical_probe_metrics, define_wandb_metrics
from experiments.spurious_eval.datasets.registry import build_probe_loaders
from experiments.spurious_eval.metrics import compute_group_metrics, entropy_effective_rank
from experiments.spurious_eval.models.resnet import LinearClassifier, build_resnet_encoder
from experiments.spurious_eval.training.checkpointing import load_encoder_checkpoint
from experiments.spurious_eval.training.logistic_probe import fit_logistic_probe
from experiments.spurious_eval.training.probe_loop import (
    extract_features,
    make_feature_loader,
    train_one_epoch,
    validate,
)
from experiments.spurious_eval.training.reproducibility import make_dataloader_kwargs
from splice.artifacts import artifact_uri, atomic_write_json, binary_destination, tensor_payload_bytes

RESULT_SCHEMA = "linear-probe-result-v3"
FEATURE_ARTIFACT = "downstream_probe_features_v2"
FEATURE_ARTIFACT_VERSION = 2
#: Accuracies are averaged over at most this many final probe epochs.
HISTORY_WINDOW = 10


@dataclass(frozen=True)
class ProbeOptions:
    """What the measurement needs. No run identity, no destination, no tracking."""

    data_folder: str = "./datasets"
    train_split: str = "ds_train"
    eval_split: str = "val"
    batch_size: int = 256
    num_workers: int = 0
    seed: int = 0
    device: str = "cpu"
    #: The SpurSSL projection head of the encoder, instantiated to keep classifier RNG in step.
    head: str = "mlp"
    solver: str = "logistic"
    l2: float = 1e-3
    tolerance: float = 1e-6
    max_epochs: int = 200
    epochs: int = 100
    learning_rate: float = 1.0
    lr_decay_epochs: tuple[int, ...] = ()
    lr_decay_rate: float = 0.2
    weight_decay: float = 0.0
    momentum: float = 0.9
    cosine: bool = False
    spurious_probe: bool = True

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


@dataclass(frozen=True)
class ProbeArtifacts:
    """Where a probe's outputs go and how long they stay."""

    directory: Path
    #: study, seed, arm and attempt_id, which route binaries to scratch.
    identity: Mapping[str, Any]
    #: The SSL epoch this probe measured; it names the files.
    ssl_epoch: int = 0
    #: The last SSL epoch of the run, which marks the final probe of a training run.
    ssl_total_epochs: int = 0
    #: Additionally retain the bulky feature tensors every N SSL epochs.
    retain_every: int = 0
    recorder: Any = None


@dataclass
class ProbeHistory:
    val_accuracy: list[float] = field(default_factory=list)
    val_worst_group: list[float] = field(default_factory=list)
    val_best_group: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ProbeResult:
    """What one probe measured, with the tensors its artifacts are written from."""

    metrics: dict[str, Any]
    convergence: dict[str, Any]
    history: ProbeHistory
    train_features: TensorDataset
    eval_features: TensorDataset
    sample_ids: dict[str, list[str]]
    group_metrics: dict[str, dict[str, Any]]
    feature_dim: int
    #: One entry per probe epoch, which the run record logs when the result is persisted.
    epoch_metrics: list[dict[str, float]] = field(default_factory=list)

    @property
    def history_window(self) -> int:
        return min(HISTORY_WINDOW, len(self.history.val_accuracy))


def seed_probe(seed: int) -> None:
    """Seed the probe so every measurement of a run starts from the same state."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def probe_learning_rate(options: ProbeOptions, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    """Apply the probe's own schedule, which is independent of the SSL schedule."""

    lr = options.learning_rate
    if options.cosine:
        eta_min = lr * (options.lr_decay_rate**3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / options.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(options.lr_decay_epochs))
        if steps > 0:
            lr = lr * (options.lr_decay_rate**steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def consume_head_rng(feature_dim: int, head: str) -> None:
    """Instantiate the unused SpurSSL projection head to preserve classifier RNG state."""

    if head == "linear":
        torch.nn.Linear(feature_dim, 128)
    elif head == "mlp":
        torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 128),
        )
    elif head in {"identity", "fixed"}:
        return
    else:
        raise ValueError(f"Unsupported SpurSSL head: {head}")


def probe_sample_ids(dataset, *, seed: int, batch_size: int, shuffle: bool, dataset_name: str) -> list[str]:
    """Reconstruct the exact order used by feature extraction for v2 artifacts."""

    source_indices = getattr(dataset, "indices", None)
    if source_indices is None:
        return []
    order = torch.arange(len(source_indices))
    if shuffle:
        loader = torch.utils.data.DataLoader(
            range(len(source_indices)),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
        )
        order = torch.cat(list(loader)).long()
    return [f"{dataset_name}:{int(source_indices[int(index)])}" for index in order]


def _group_cardinalities(metadata: torch.Tensor) -> tuple[int, int]:
    metadata = torch.as_tensor(metadata).detach().cpu()
    if metadata.ndim != 2 or metadata.shape[1] < 2:
        raise ValueError("Group metrics require metadata columns [context, target].")
    contexts = metadata[:, 0].long()
    targets = metadata[:, 1].long()
    return (
        max(2, int(contexts.max().item()) + 1 if contexts.numel() else 1),
        max(2, int(targets.max().item()) + 1 if targets.numel() else 1),
    )


def build_named_group_metrics(group_accuracies, group_counts, metadata) -> dict[str, dict[str, float | int | None]]:
    """Return portable named group metrics, marking empty-group accuracy unavailable."""

    group_accuracies = torch.as_tensor(group_accuracies).detach().cpu().float().view(-1)
    group_counts = torch.as_tensor(group_counts).detach().cpu().long().view(-1)
    context_cardinality, target_cardinality = _group_cardinalities(metadata)
    named: dict[str, dict[str, float | int | None]] = {}
    for target in range(target_cardinality):
        for context in range(context_cardinality):
            group_id = context + context_cardinality * target
            count = int(group_counts[group_id]) if group_id < len(group_counts) else 0
            accuracy = (
                float(group_accuracies[group_id]) * 100
                if count > 0 and group_id < len(group_accuracies)
                else None
            )
            named[f"(target,context)=({target},{context})"] = {
                "accuracy": accuracy,
                "count": count,
            }
    return named


def build_wandb_group_metrics(group_accuracies, group_counts, metadata) -> dict[str, float | int | None]:
    """Name validation groups by the stable ``(target, context)`` convention."""

    group_accuracies = torch.as_tensor(group_accuracies).detach().cpu().float().view(-1)
    group_counts = torch.as_tensor(group_counts).detach().cpu().long().view(-1)
    metadata = torch.as_tensor(metadata).detach().cpu()
    if metadata.ndim != 2 or metadata.shape[1] < 2:
        raise ValueError("Group W&B metrics require metadata columns [context, target].")
    # Waterbirds (the cluster control dataset) has binary target/context metadata. Keep the
    # complete 2x2 W&B panel even if a validation split happens to contain an empty group.
    context_cardinality, target_cardinality = _group_cardinalities(metadata)
    metrics: dict[str, float | int | None] = {}
    for target in range(target_cardinality):
        for context in range(context_cardinality):
            group_id = context + context_cardinality * target
            count = int(group_counts[group_id]) if group_id < len(group_counts) else 0
            accuracy = (
                float(group_accuracies[group_id]) * 100
                if count > 0 and group_id < len(group_accuracies)
                else None
            )
            prefix = f"Linear val group (target,context)=({target},{context})"
            metrics[f"{prefix} acc"] = accuracy
            metrics[f"{prefix} count"] = count
    return metrics


def spurious_attribute_metrics(
    train_features: TensorDataset,
    eval_features: TensorDataset,
    feature_dim: int,
    options: ProbeOptions,
    device: torch.device,
) -> dict[str, float]:
    """Measure residual linear access to the spurious attribute."""

    train_x, _, train_metadata = train_features.tensors
    val_x, _, val_metadata = eval_features.tensors
    train_spurious = train_metadata[:, 0].long()
    val_spurious = val_metadata[:, 0].long()
    n_attributes = int(train_spurious.max().item()) + 1
    train_dataset = TensorDataset(train_x, train_spurious, train_metadata)
    val_dataset = TensorDataset(val_x, val_spurious, val_metadata)
    if options.solver == "logistic":
        records, convergence = fit_logistic_probe(
            train_x, train_spurious, val_x, val_spurious, num_classes=n_attributes,
            l2=options.l2, tolerance=options.tolerance, max_epochs=options.max_epochs,
        )
        auxiliary_metadata = torch.stack((val_metadata[:, 1], val_metadata[:, 0]), dim=1)
        metrics = [compute_group_metrics(r.eval_predictions, val_spurious, auxiliary_metadata) for r in records]
        return {
            "Spurious probe last val acc": metrics[-1].average * 100,
            "Spurious probe average over last 10 val acc": float(np.mean([m.average for m in metrics])) * 100,
            "Spurious probe last val worst-group acc": metrics[-1].worst_group * 100,
            "Spurious probe average over last 10 val worst-group acc": float(np.mean([m.worst_group for m in metrics])) * 100,
            "Spurious probe converged": convergence["converged"],
            "Spurious probe gradient max": convergence["gradient_max"],
        }
    train_loader = make_feature_loader(train_dataset, options.batch_size, options.seed + 10_000, shuffle=True)
    val_loader = make_feature_loader(val_dataset, options.batch_size, options.seed + 10_000, shuffle=False)

    classifier = LinearClassifier(feature_dim=feature_dim, num_classes=n_attributes).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=options.learning_rate,
        momentum=options.momentum,
        weight_decay=options.weight_decay,
    )
    accuracies = []
    worst_group_accuracies = []
    for epoch in range(1, options.epochs + 1):
        probe_learning_rate(options, optimizer, epoch)
        train_one_epoch(train_loader, classifier, criterion, optimizer, device)
        _, accuracy, predictions, labels, metadata = validate(val_loader, classifier, criterion, device)
        # Reorder metadata so groups are (target, spurious label) for this auxiliary task.
        auxiliary_metadata = torch.stack((metadata[:, 1], metadata[:, 0]), dim=1)
        group_metrics = compute_group_metrics(predictions, labels, auxiliary_metadata)
        accuracies.append(accuracy)
        worst_group_accuracies.append(group_metrics.worst_group * 100)

    window = min(HISTORY_WINDOW, len(accuracies))
    return {
        "Spurious probe last val acc": float(accuracies[-1]),
        "Spurious probe average over last 10 val acc": float(np.mean(accuracies[-window:])),
        "Spurious probe last val worst-group acc": float(worst_group_accuracies[-1]),
        "Spurious probe average over last 10 val worst-group acc": float(
            np.mean(worst_group_accuracies[-window:])
        ),
    }


def evaluate_probe(encoder, dataset, options: ProbeOptions, *, feature_dim: int) -> ProbeResult:
    """Measure a frozen encoder with a linear probe. Writes nothing and logs nothing."""

    device = options.torch_device
    config = dataset.Config(
        root_dir=options.data_folder,
        train_split=options.train_split,
        eval_split=options.eval_split,
    )
    # ProbeOptions carries the seed and the worker count make_dataloader_kwargs reads.
    train_loader, eval_loader = build_probe_loaders(
        dataset,
        config,
        options.batch_size,
        train_loader_kwargs=make_dataloader_kwargs(options, shuffle=True),
        eval_loader_kwargs=make_dataloader_kwargs(options, shuffle=False),
    )

    consume_head_rng(feature_dim, options.head)
    classifier = LinearClassifier(feature_dim=feature_dim, num_classes=dataset.num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=options.learning_rate,
        momentum=options.momentum,
        weight_decay=options.weight_decay,
    )

    print("[INFO] Extracting frozen train features")
    train_features = extract_features(encoder, train_loader, device)
    print("[INFO] Extracting frozen validation features")
    eval_features = extract_features(encoder, eval_loader, device)
    sample_ids = {
        "train": probe_sample_ids(
            train_loader.dataset, seed=options.seed, batch_size=options.batch_size,
            shuffle=True, dataset_name=dataset.name,
        ),
        "evaluation": probe_sample_ids(
            eval_loader.dataset, seed=options.seed, batch_size=options.batch_size,
            shuffle=False, dataset_name=dataset.name,
        ),
    }
    feature_loader = make_feature_loader(train_features, options.batch_size, options.seed, shuffle=True)
    val_feature_loader = make_feature_loader(eval_features, options.batch_size, options.seed, shuffle=False)

    history = ProbeHistory()
    best_val_acc = best_val_wg_acc = best_val_bg_acc = 0.0
    best_train_acc = best_train_wg_acc = best_train_bg_acc = 0.0
    epoch_metrics: list[dict[str, float]] = []

    convergence: dict[str, Any] = {}
    logistic_records = None
    if options.solver == "logistic":
        logistic_records, convergence = fit_logistic_probe(
            train_features.tensors[0], train_features.tensors[1],
            eval_features.tensors[0], eval_features.tensors[1],
            num_classes=dataset.num_classes, l2=options.l2,
            tolerance=options.tolerance, max_epochs=options.max_epochs,
        )
    for epoch in range(1, (len(logistic_records) if logistic_records is not None else options.epochs) + 1):
        probe_learning_rate(options, optimizer, epoch)
        start = time.time()
        if logistic_records is None:
            train_loss, train_acc, train_pred, train_labels, train_metadata = train_one_epoch(
                feature_loader, classifier, criterion, optimizer, device
            )
        else:
            record = logistic_records[epoch - 1]
            train_loss, train_pred = record.train_loss, record.train_predictions
            train_labels, train_metadata = train_features.tensors[1:]
            train_acc = float(train_pred.eq(train_labels).float().mean()) * 100
        display_epoch = record.epoch if logistic_records is not None else epoch
        train_results, _ = train_loader.dataset.eval(train_pred, train_labels, train_metadata)
        train_wg_acc = train_results["acc_wg"] * 100
        train_bg_acc = train_results["best_acc"] * 100
        print(
            "Train epoch {}, total time {:.2f}, loss {:.4f}, accuracy {:.2f}, wg accuracy {:.2f}, bg accuracy {:.2f}".format(
                display_epoch, time.time() - start, train_loss, train_acc, train_wg_acc, train_bg_acc
            )
        )

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_train_wg_acc = train_wg_acc
            best_train_bg_acc = train_bg_acc

        if logistic_records is None:
            val_loss, val_acc, val_pred, val_labels, val_metadata = validate(
                val_feature_loader, classifier, criterion, device
            )
        else:
            val_loss, val_pred = record.eval_loss, record.eval_predictions
            val_labels, val_metadata = eval_features.tensors[1:]
            val_acc = float(val_pred.eq(val_labels).float().mean()) * 100
        val_results, _ = eval_loader.dataset.eval(val_pred, val_labels, val_metadata)
        val_wg_acc = val_results["acc_wg"] * 100
        val_bg_acc = val_results["best_acc"] * 100
        print(
            "Val epoch {}, loss {:.4f}, accuracy {:.2f}, wg accuracy {:.2f}, bg accuracy {:.2f}".format(
                display_epoch, val_loss, val_acc, val_wg_acc, val_bg_acc
            )
        )

        history.val_accuracy.append(val_acc)
        history.val_worst_group.append(val_wg_acc)
        history.val_best_group.append(val_bg_acc)
        epoch_metrics.append({
            "epoch": display_epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_worst_group_accuracy": train_wg_acc,
            "train_best_group_accuracy": train_bg_acc,
            "eval_loss": val_loss,
            "eval_accuracy": val_acc,
            "eval_worst_group_accuracy": val_wg_acc,
            "eval_best_group_accuracy": val_bg_acc,
        })

        if val_acc > best_val_acc or (
            val_acc == best_val_acc and (val_wg_acc, val_bg_acc) > (best_val_wg_acc, best_val_bg_acc)
        ):
            best_val_acc = val_acc
            best_val_wg_acc = val_wg_acc
            best_val_bg_acc = val_bg_acc

    last_acc = history.val_accuracy[-1]
    last_wg_acc = history.val_worst_group[-1]
    last_bg_acc = history.val_best_group[-1]
    window = min(HISTORY_WINDOW, len(history.val_accuracy))
    avg_last_10_acc = float(np.mean(history.val_accuracy[-window:]))
    avg_last_10_wg_acc = float(np.mean(history.val_worst_group[-window:]))
    avg_last_10_bg_acc = float(np.mean(history.val_best_group[-window:]))
    group_counts = val_results["group_counts"]
    group_accuracies = val_results["group_accuracy"]
    train_group_counts = train_results["group_counts"]
    train_group_accuracies = train_results["group_accuracy"]
    nonempty_group_ids = torch.where(group_counts > 0)[0]
    if len(nonempty_group_ids):
        worst_offset = torch.argmin(group_accuracies[nonempty_group_ids])
        last_worst_group_id = int(nonempty_group_ids[worst_offset].item())
        last_worst_group_count = int(group_counts[last_worst_group_id].item())
    else:
        last_worst_group_id = -1
        last_worst_group_count = 0
    print(
        "Average of last 10 accuracies: {:.2f}, Average of last 10 worst-group accuracies: {:.2f}, Average of last 10 best-group accuracies: {:.2f}".format(
            avg_last_10_acc, avg_last_10_wg_acc, avg_last_10_bg_acc
        )
    )

    train_feature_tensor = train_features.tensors[0]
    val_feature_tensor = eval_features.tensors[0]
    entropy, effective_rank, energy_based_rank = entropy_effective_rank(train_feature_tensor)
    val_entropy, val_effective_rank, val_energy_based_rank = entropy_effective_rank(val_feature_tensor)

    print(f"Train - Entropy: {entropy:.4f}, Effective Rank: {effective_rank:.2f}, Energy-Based Rank: {energy_based_rank:.2f}")
    print(f"Val   - Entropy: {val_entropy:.4f}, Effective Rank: {val_effective_rank:.2f}, Energy-Based Rank: {val_energy_based_rank:.2f}")

    metrics: dict[str, Any] = {
        "Probe converged": convergence.get("converged", False),
        "Probe epochs": convergence.get("epochs", options.epochs),
        "Probe gradient max": convergence.get("gradient_max", 0.0),
        "Linear train acc": best_train_acc,
        "Linear train worst-group acc": best_train_wg_acc,
        "Linear train best-group acc": best_train_bg_acc,
        "Linear val acc": best_val_acc,
        "Linear val worst-group acc": best_val_wg_acc,
        "Linear val best-group acc": best_val_bg_acc,
        "Train linear entropy": entropy,
        "Train linear effective rank": effective_rank,
        "Train linear energy-based rank": energy_based_rank,
        "Val linear entropy": val_entropy,
        "Val linear effective rank": val_effective_rank,
        "Val linear energy-based rank": val_energy_based_rank,
        "Last linear val acc": last_acc,
        "Last linear val worst-group acc": last_wg_acc,
        "Last linear val best-group acc": last_bg_acc,
        "Average over 10 last linear val acc": avg_last_10_acc,
        "Average over last 10 linear val acc": avg_last_10_acc,
        "Average over last 10 linear val worst-group acc": avg_last_10_wg_acc,
        "Average over last 10 linear val best-group acc": avg_last_10_bg_acc,
        "Last linear val worst-group id": last_worst_group_id,
        "Last linear val worst-group count": last_worst_group_count,
        # Persist every group, including empty groups as zero-count entries. The lists are
        # deliberately detached from tensors so result JSON is portable.
        "Linear val group accuracies": (group_accuracies.detach().cpu().float() * 100).tolist(),
        "Linear val group counts": group_counts.detach().cpu().long().tolist(),
        "Linear train group accuracies": (train_group_accuracies.detach().cpu().float() * 100).tolist(),
        "Linear train group counts": train_group_counts.detach().cpu().long().tolist(),
    }
    if options.spurious_probe:
        print("[INFO] Training auxiliary spurious-attribute leakage probe")
        metrics.update(
            spurious_attribute_metrics(train_features, eval_features, feature_dim, options, device)
        )

    group_metrics = {
        "val": {
            "accuracy": metrics["Linear val group accuracies"],
            "count": metrics["Linear val group counts"],
            "named": build_named_group_metrics(group_accuracies, group_counts, eval_features.tensors[2]),
            "wandb": build_wandb_group_metrics(group_accuracies, group_counts, eval_features.tensors[2]),
        },
        "train": {
            "accuracy": metrics["Linear train group accuracies"],
            "count": metrics["Linear train group counts"],
            "named": build_named_group_metrics(
                train_group_accuracies, train_group_counts, train_features.tensors[2],
            ),
        },
    }
    result = ProbeResult(
        metrics=metrics,
        convergence=convergence,
        history=history,
        train_features=train_features,
        eval_features=eval_features,
        sample_ids=sample_ids,
        group_metrics=group_metrics,
        feature_dim=feature_dim,
        epoch_metrics=epoch_metrics,
    )
    print_probe_summary(result)
    return result


def print_probe_summary(result: ProbeResult) -> None:
    """The closing lines a cluster log is read for."""

    metrics = result.metrics
    print(
        "best accuracy: {:.2f} and worst-group accuracy: {:.2f} and best-group accuracy: {:.2f}".format(
            metrics["Linear val acc"], metrics["Linear val worst-group acc"], metrics["Linear val best-group acc"]
        )
    )
    print(
        "Last accuracy: {:.2f}, Last worst-group accuracy: {:.2f}, Last best-group accuracy: {:.2f}".format(
            metrics["Last linear val acc"], metrics["Last linear val worst-group acc"],
            metrics["Last linear val best-group acc"],
        )
    )
    print("Train entropy: {:.2f}, effective rank: {}, and energy-based rank: {}".format(
        metrics["Train linear entropy"], metrics["Train linear effective rank"],
        metrics["Train linear energy-based rank"],
    ))
    print("Val entropy: {:.2f}, effective rank: {}, and energy-based rank: {}".format(
        metrics["Val linear entropy"], metrics["Val linear effective rank"],
        metrics["Val linear energy-based rank"],
    ))
    print(
        "Average last 10 accuracies: {:.2f}, Average last 10 worst-group accuracies: {:.2f}, Average last 10 best-group accuracies: {:.2f}".format(
            metrics["Average over last 10 linear val acc"],
            metrics["Average over last 10 linear val worst-group acc"],
            metrics["Average over last 10 linear val best-group acc"],
        )
    )


def probe_artifact_paths(options: ProbeOptions, artifacts: ProbeArtifacts) -> tuple[Path, Path]:
    """The feature tensor and the result JSON this probe writes, before scratch routing."""

    stem = f"probe_features_epoch_{artifacts.ssl_epoch}_{options.train_split}_{options.eval_split}"
    feature_path = Path(artifacts.directory) / f"{stem}.pt"
    return feature_path, feature_path.with_suffix(".json")


def persist_probe_result(
    result: ProbeResult, options: ProbeOptions, artifacts: ProbeArtifacts,
) -> dict[str, Path]:
    """Write the feature tensors to scratch and the result JSON beside the run, then attest both."""

    feature_path, result_path = probe_artifact_paths(options, artifacts)
    feature_payload = {
        "train": result.train_features.tensors,
        "evaluation": result.eval_features.tensors,
        "ssl_epoch": artifacts.ssl_epoch,
        "train_split": options.train_split,
        "eval_split": options.eval_split,
        "seed": options.seed,
        "artifact": FEATURE_ARTIFACT,
        "artifact_version": FEATURE_ARTIFACT_VERSION,
        "sample_ids": result.sample_ids,
    }
    stored_feature_path = binary_destination(
        feature_path, tensor_payload_bytes(feature_payload), kind="features", identity=artifacts.identity,
    )
    stored_feature_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feature_payload, stored_feature_path)

    is_final = bool(artifacts.ssl_total_epochs and artifacts.ssl_epoch == artifacts.ssl_total_epochs)
    retention = "final" if is_final else (
        "retained"
        if artifacts.retain_every > 0 and artifacts.ssl_epoch > 0
        and artifacts.ssl_epoch % artifacts.retain_every == 0
        else "temporary"
    )
    recorder = artifacts.recorder
    if recorder is not None:
        recorder.register_artifact(
            stored_feature_path, kind="probe_features", stage="linear_probe",
            epoch=artifacts.ssl_epoch, retention_state=retention,
        )
        for event in result.epoch_metrics:
            recorder.log_metrics("linear_probe", event["epoch"], {
                "ssl_epoch": artifacts.ssl_epoch,
                **{key: value for key, value in event.items() if key != "epoch"},
            })

    result_payload = {
        "schema": RESULT_SCHEMA,
        "ssl_epoch": artifacts.ssl_epoch,
        "solver": options.solver,
        "train_split": options.train_split,
        "eval_split": options.eval_split,
        "selection_criterion": "max_eval_accuracy_then_worst_group_then_best_group",
        "metric_semantics": {
            "accuracy_unit": "percent",
            "average_accuracy": "sample-weighted accuracy on eval_split",
            "worst_group_accuracy": "minimum accuracy over non-empty (target,context) groups",
            "best_group_accuracy": "maximum accuracy over non-empty (target,context) groups",
            "history_window": result.history_window,
        },
        "feature_artifact": {
            "uri": artifact_uri(stored_feature_path),
            "payload_bytes": tensor_payload_bytes(feature_payload),
            "retention_state": retention,
        },
        "convergence": result.convergence,
        "metrics": result.metrics,
        "group_metrics": {
            split: {key: value for key, value in entry.items() if key != "wandb"}
            for split, entry in result.group_metrics.items()
        },
    }
    atomic_write_json(result_path, result_payload)
    if recorder is not None:
        result_retention = "final" if is_final else ("temporary" if artifacts.ssl_total_epochs else "retained")
        recorder.register_artifact(
            result_path, kind="probe_result", stage="linear_probe",
            epoch=artifacts.ssl_epoch, retention_state=result_retention,
        )
        recorder.log_metrics("linear_probe_final", artifacts.ssl_epoch, result_payload)
    return {"features": stored_feature_path, "result": result_path}


def log_probe_result(result: ProbeResult, options: ProbeOptions, *, run, ssl_epoch: int) -> None:
    """Send one probe to W&B under both the historical and the canonical keys."""

    if run is None:
        return
    metrics = result.metrics
    define_wandb_metrics(run, split=options.eval_split)
    run.log(
        {
            **{key: value for key, value in metrics.items() if not isinstance(value, list)},
            **canonical_probe_metrics(metrics, split=options.eval_split),
        },
        step=ssl_epoch,
    )
    run.log(
        {
            **{
                f"Linear val group {group_id} acc": (
                    float(accuracy) if int(metrics["Linear val group counts"][group_id]) > 0 else None
                )
                for group_id, accuracy in enumerate(metrics["Linear val group accuracies"])
            },
            **{
                f"Linear val group {group_id} count": int(count)
                for group_id, count in enumerate(metrics["Linear val group counts"])
            },
        },
        step=ssl_epoch,
    )
    run.log(result.group_metrics["val"]["wandb"], step=ssl_epoch)
    run.config.update(
        {
            "probe_solver": options.solver,
            "probe_l2": options.l2,
            "probe_tolerance": options.tolerance,
            "probe_max_epochs": options.max_epochs,
            "probe_normalization": result.convergence.get("normalization", "none"),
        },
        allow_val_change=True,
    )


def load_probe_encoder(model: str, checkpoint: str, device: torch.device):
    """The frozen encoder a probe measures, from a checkpoint or from pretrained weights."""

    encoder, feature_dim = build_resnet_encoder(model, load_pretrained_weights=not bool(checkpoint))
    if checkpoint:
        print(f"[INFO] Loading encoder checkpoint from {checkpoint}")
        load_encoder_checkpoint(encoder, checkpoint)
    elif model == "resnet50_pretrained":
        print("[INFO] No checkpoint provided. Using the frozen ImageNet-pretrained encoder.")
    else:
        print("[INFO] No checkpoint provided. Using randomly initialized frozen encoder.")
    encoder = encoder.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if device.type == "cuda":
        cudnn.benchmark = False
    return encoder, feature_dim


def probe_checkpoint(
    dataset,
    options: ProbeOptions,
    artifacts: ProbeArtifacts,
    *,
    model: str,
    checkpoint: str = "",
    wandb_run=None,
) -> ProbeResult:
    """Measure one checkpoint end to end: load the encoder, evaluate, persist, log."""

    encoder, feature_dim = load_probe_encoder(model, checkpoint, options.torch_device)
    result = evaluate_probe(encoder, dataset, options, feature_dim=feature_dim)
    persist_probe_result(result, options, artifacts)
    log_probe_result(result, options, run=wandb_run, ssl_epoch=artifacts.ssl_epoch)
    return result
