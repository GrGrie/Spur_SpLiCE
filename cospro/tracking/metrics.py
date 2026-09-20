"""Stable metric names for Weights & Biases.

Runs report two sets of keys. The historical sentence-style keys stay the primary record, since
``run.json``, `collect_results` and the paper registry read them. Next to them every run logs the
canonical keys below, which are stable across refactors and safe to chart, filter and compare:

    probe/<split>/wga            worst-group accuracy, the headline metric
    probe/<split>/wga_avg10      worst-group accuracy averaged over the last ten probe epochs
    probe/<split>/avg_acc        average accuracy
    probe/<split>/group_acc/<g>  per-group accuracy
    probe/spurious/wga           residual predictability of the spurious attribute
    train/loss/<term>            SSL loss terms
    train/representation/<name>  representation-rank diagnostics
    method/<name>                whatever the training method reports

``define_wandb_metrics`` marks ``probe/<split>/wga`` as the run summary metric, so the W&B run
table sorts on the best worst-group accuracy.
"""

from __future__ import annotations

from typing import Any, Mapping

WGA_KEY = "probe/{split}/wga"

# Historical probe key (with {split} filled in) -> canonical key.
PROBE_KEYS: dict[str, str] = {
    "Last linear {split} acc": "probe/{split}/avg_acc",
    "Last linear {split} worst-group acc": "probe/{split}/wga",
    "Last linear {split} best-group acc": "probe/{split}/best_group_acc",
    "Average over last 10 linear {split} acc": "probe/{split}/avg_acc_avg10",
    "Average over last 10 linear {split} worst-group acc": "probe/{split}/wga_avg10",
    "Average over last 10 linear {split} best-group acc": "probe/{split}/best_group_acc_avg10",
    "Linear train acc": "probe/train/avg_acc",
    "Linear train worst-group acc": "probe/train/wga",
    "Probe epochs": "probe/epochs",
    "Probe converged": "probe/converged",
    "Spurious probe last val acc": "probe/spurious/avg_acc",
    "Spurious probe last val worst-group acc": "probe/spurious/wga",
}
# Historical SSL key -> canonical key.
TRAIN_KEYS: dict[str, str] = {
    "SSL train loss": "train/loss/total",
    "SSL SimCLR loss": "train/loss/simclr",
    "SSL decor loss": "train/loss/decor",
    "SSL entropy loss": "train/loss/entropy",
    "SSL splice loss": "train/loss/method",
    "SSL learning rate": "train/learning_rate",
    "Entropy": "train/representation/entropy",
    "Effective rank": "train/representation/effective_rank",
    "Energy-based rank": "train/representation/energy_based_rank",
}
_METHOD_PREFIX = "SSL relational "
_METHOD_DIAGNOSTIC_NAMES = {
    "scheduled weight": "scheduled_weight",
    "supported anchor fraction": "supported_anchor_fraction",
    "mean anchor confidence": "mean_anchor_confidence",
    "unweighted KL": "unweighted_kl",
    "confidence-weighted KL": "confidence_weighted_kl",
}


def canonical_probe_metrics(metrics: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    """Canonical names for the probe metrics of one evaluation split."""

    canonical = {}
    for source, target in PROBE_KEYS.items():
        key = source.format(split=split)
        if key in metrics:
            canonical[target.format(split=split)] = metrics[key]
    accuracies = metrics.get("Linear val group accuracies") or []
    counts = metrics.get("Linear val group counts") or []
    for group, accuracy in enumerate(accuracies):
        if group < len(counts) and int(counts[group]) > 0:
            canonical[f"probe/{split}/group_acc/{group}"] = float(accuracy)
    return canonical


def canonical_train_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical names for the per-epoch SSL payload, including method diagnostics."""

    canonical = {target: payload[source] for source, target in TRAIN_KEYS.items() if source in payload}
    for key, value in payload.items():
        if not key.startswith(_METHOD_PREFIX):
            continue
        name = key[len(_METHOD_PREFIX):]
        canonical[f"method/{_METHOD_DIAGNOSTIC_NAMES.get(name, name.replace(' ', '_'))}"] = value
    for key, value in payload.items():
        if key.startswith("SSL relational_") or key.startswith("SSL la_ssl_"):
            canonical[f"method/{key.removeprefix('SSL ').removeprefix('relational_')}"] = value
    return canonical


def define_wandb_metrics(run, *, split: str) -> None:
    """Make worst-group accuracy the run summary metric."""

    if run is None:
        return
    run.define_metric(WGA_KEY.format(split=split), summary="max")
    run.define_metric(f"probe/{split}/wga_avg10", summary="max")
