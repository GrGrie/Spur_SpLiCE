"""Stable metric names for Weights & Biases.

Runs report two sets of keys. The historical sentence-style keys stay the primary record, since
``run.json``, `collect_results` and the paper registry read them. Next to them every run logs the
canonical keys below, which are stable across refactors and safe to chart, filter and compare:

    probe/<split>/wga            worst-group accuracy, the headline metric
    probe/<split>/wga_avg10      worst-group accuracy averaged over the last ten probe evaluations
                                 of the run (ten SSL checkpoints, not ten probe-solver steps)
    probe/<split>/avg_acc        average accuracy
    probe/<split>/group_acc/<g>  per-group accuracy
    probe/spurious/wga           residual predictability of the spurious attribute
    train/loss/<term>            SSL loss terms
    train/representation/<name>  representation-rank diagnostics
    method/<name>                whatever the training method reports

``epoch_payload`` assembles the per-epoch SSL event under the historical names and
``define_wandb_metrics`` marks ``probe/<split>/wga`` as the run summary metric, so the W&B run
table sorts on the best worst-group accuracy.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

WGA_KEY = "probe/{split}/wga"
ROLLING_WINDOW = 10
# Canonical probe metric -> its mean over the last ROLLING_WINDOW probe evaluations.
ROLLING_KEYS: dict[str, str] = {
    "probe/{split}/wga": "probe/{split}/wga_avg10",
    "probe/{split}/avg_acc": "probe/{split}/avg_acc_avg10",
    "probe/{split}/best_group_acc": "probe/{split}/best_group_acc_avg10",
}
# Keys W&B does not chart: the probe fits its training split perfectly, so they are flat lines.
UNCHARTED_KEYS = frozenset({"Linear train acc", "Linear train worst-group acc"})
_rolling_history: dict[tuple[Any, str], dict[str, deque]] = {}

# Historical probe key (with {split} filled in) -> canonical key.
PROBE_KEYS: dict[str, str] = {
    "Last linear {split} acc": "probe/{split}/avg_acc",
    "Last linear {split} worst-group acc": "probe/{split}/wga",
    "Last linear {split} best-group acc": "probe/{split}/best_group_acc",
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


def epoch_payload(
    train_metrics: Mapping[str, Any],
    *,
    learning_rate: float,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One epoch's metric event under its historical key names.

    ``train_metrics`` comes from the epoch loop and ``diagnostics`` from the observational
    callbacks, such as representation rank. The run record and W&B store this payload; W&B
    additionally receives :func:`canonical_train_metrics` of it.
    """

    payload: dict[str, Any] = dict(diagnostics or {})
    payload.update({
        "SSL train loss": train_metrics["loss"],
        "SSL SimCLR loss": train_metrics["simclr_loss"],
        "SSL decor loss": train_metrics["decor_loss"],
        "SSL entropy loss": train_metrics["entropy_loss"],
        "SSL splice loss": train_metrics["splice_loss"],
        "SSL learning rate": learning_rate,
        "SSL relational scheduled weight": train_metrics.get("relational_scheduled_weight", 0.0),
        "SSL relational supported anchor fraction": train_metrics.get("relational_supported_anchor_fraction", 0.0),
        "SSL relational mean anchor confidence": train_metrics.get("relational_mean_anchor_confidence", 0.0),
        "SSL relational unweighted KL": train_metrics.get("relational_unweighted_kl", 0.0),
        "SSL relational confidence-weighted KL": train_metrics.get("relational_confidence_weighted_kl", 0.0),
    })
    payload.update({
        f"SSL {key}": value for key, value in train_metrics.items()
        if key.startswith(("la_ssl_", "latetvg_")) or key in {"relational_valid_fraction", "relational_cosine_loss"}
    })
    return payload


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


def rolling_probe_metrics(run, canonical: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    """Means over the last ten probe evaluations of ``run``, the current one included.

    The historical "Average over last 10" keys average the probe solver's final steps, which
    have converged and so equal the last value. These keys average across SSL checkpoints.
    The window lives in memory, so a resumed run starts it afresh.
    """

    history = _rolling_history.setdefault((getattr(run, "id", None) or id(run), split), {})
    rolling = {}
    for source, target in ROLLING_KEYS.items():
        key = source.format(split=split)
        if key in canonical:
            values = history.setdefault(key, deque(maxlen=ROLLING_WINDOW))
            values.append(float(canonical[key]))
            rolling[target.format(split=split)] = sum(values) / len(values)
    return rolling


def canonical_train_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical names for the per-epoch SSL payload, including method diagnostics."""

    canonical = {target: payload[source] for source, target in TRAIN_KEYS.items() if source in payload}
    for key, value in payload.items():
        if not key.startswith(_METHOD_PREFIX):
            continue
        name = key[len(_METHOD_PREFIX):]
        canonical[f"method/{_METHOD_DIAGNOSTIC_NAMES.get(name, name.replace(' ', '_'))}"] = value
    for key, value in payload.items():
        if key.startswith(("SSL relational_", "SSL la_ssl_", "SSL latetvg_")):
            canonical[f"method/{key.removeprefix('SSL ').removeprefix('relational_')}"] = value
    return canonical


def define_wandb_metrics(run, *, split: str) -> None:
    """Make worst-group accuracy the run summary metric."""

    if run is None:
        return
    run.define_metric(WGA_KEY.format(split=split), summary="max")
    run.define_metric(f"probe/{split}/wga_avg10", summary="max")
