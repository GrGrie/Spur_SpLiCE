"""Refit saved probe features with both probe solvers and compare the results.

A diagnostic for runs whose numbers moved when the default solver changed from SGD to the
full-batch logistic probe. It reads the ``probe_features_epoch_*.pt`` tensors a probe already
wrote, so it needs neither the SSL checkpoint nor the images, and fits on identical features:

    sgd       the historical protocol: 100 epochs of SGD, lr 1.0, momentum 0.9, decay 0.2 at 60/75/90
    logistic  the current protocol: L-BFGS on standardized features, l2 1e-3, ten stable epochs

The logistic row should reproduce the number the run logged; the sgd row is what the old code
would have measured on the same encoder.

    python -m cospro.cli.compare_probe_solvers FEATURES.pt [FEATURES.pt | DIRECTORY ...] --seed 1
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import TensorDataset

from cospro.cli.linear_probe import resolve_lr_decay_epochs
from cospro.config import LINEAR_PROBE_DEFAULTS
from cospro.evaluation.logistic_probe import fit_logistic_probe
from cospro.evaluation.probe import (
    HISTORY_WINDOW,
    ProbeOptions,
    consume_head_rng,
    probe_learning_rate,
    seed_probe,
)
from cospro.evaluation.probe_loop import make_feature_loader, train_one_epoch, validate
from cospro.metrics import compute_group_metrics
from cospro.models.resnet import LinearClassifier


def feature_files(paths: list[str]) -> list[Path]:
    """The feature tensors named directly or found under the given directories."""

    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(path.rglob("probe_features_epoch_*.pt")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"No feature file or directory at {path}")
    if not found:
        raise FileNotFoundError(f"No probe_features_epoch_*.pt under {paths}")
    return found


def group_scores(predictions: torch.Tensor, labels: torch.Tensor, metadata: torch.Tensor) -> dict[str, Any]:
    metrics = compute_group_metrics(predictions, labels, metadata)
    return {
        "avg": metrics.average * 100,
        "wga": metrics.worst_group * 100,
        "best_group": metrics.best_group * 100,
        "groups": [round(float(value) * 100, 2) for value in metrics.group_accuracy],
    }


def summarize(epochs: list[dict[str, Any]]) -> dict[str, Any]:
    """Last-epoch scores plus the mean over the final probe epochs, as the probe reports them."""

    window = epochs[-HISTORY_WINDOW:]
    last = epochs[-1]
    return {
        "epochs": len(epochs),
        "train_avg": last["train"]["avg"],
        "train_wga": last["train"]["wga"],
        "eval_avg": last["eval"]["avg"],
        "eval_wga": last["eval"]["wga"],
        "eval_best_group": last["eval"]["best_group"],
        "eval_groups": last["eval"]["groups"],
        "eval_avg_last10": statistics.mean(e["eval"]["avg"] for e in window),
        "eval_wga_last10": statistics.mean(e["eval"]["wga"] for e in window),
    }


def fit_sgd(train: TensorDataset, evaluation: TensorDataset, options: ProbeOptions, num_classes: int) -> dict[str, Any]:
    seed_probe(options.seed)
    feature_dim = train.tensors[0].shape[1]
    device = options.torch_device
    consume_head_rng(feature_dim, options.head)
    classifier = LinearClassifier(feature_dim=feature_dim, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=options.learning_rate,
        momentum=options.momentum, weight_decay=options.weight_decay,
    )
    train_loader = make_feature_loader(train, options.batch_size, options.seed, shuffle=True)
    eval_loader = make_feature_loader(evaluation, options.batch_size, options.seed, shuffle=False)
    epochs = []
    for epoch in range(1, options.epochs + 1):
        probe_learning_rate(options, optimizer, epoch)
        _, _, train_pred, train_labels, train_meta = train_one_epoch(
            train_loader, classifier, criterion, optimizer, device,
        )
        _, _, eval_pred, eval_labels, eval_meta = validate(eval_loader, classifier, criterion, device)
        epochs.append({
            "train": group_scores(train_pred, train_labels, train_meta),
            "eval": group_scores(eval_pred, eval_labels, eval_meta),
        })
    return summarize(epochs)


def fit_logistic(train: TensorDataset, evaluation: TensorDataset, options: ProbeOptions, num_classes: int) -> dict[str, Any]:
    records, convergence = fit_logistic_probe(
        train.tensors[0], train.tensors[1], evaluation.tensors[0], evaluation.tensors[1],
        num_classes=num_classes, l2=options.l2, tolerance=options.tolerance, max_epochs=options.max_epochs,
    )
    epochs = [{
        "train": group_scores(record.train_predictions, *train.tensors[1:]),
        "eval": group_scores(record.eval_predictions, *evaluation.tensors[1:]),
    } for record in records]
    return {**summarize(epochs), "epochs": convergence["epochs"], "converged": convergence["converged"]}


def compare(path: Path, options: ProbeOptions) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    train = TensorDataset(*payload["train"])
    evaluation = TensorDataset(*payload["evaluation"])
    num_classes = int(torch.cat([train.tensors[1], evaluation.tensors[1]]).max().item()) + 1
    print(f"[INFO] {path}: train {tuple(train.tensors[0].shape)}, eval {tuple(evaluation.tensors[0].shape)}", flush=True)
    return {
        "features": str(path),
        "ssl_epoch": payload.get("ssl_epoch"),
        "eval_split": payload.get("eval_split"),
        "sgd": fit_sgd(train, evaluation, options, num_classes),
        "logistic": fit_logistic(train, evaluation, options, num_classes),
    }


def report(results: list[dict[str, Any]]) -> str:
    header = (f"{'solver':<9} {'train avg':>9} {'train wga':>9} {'avg':>7} {'wga':>7} "
              f"{'avg@10':>7} {'wga@10':>7}  groups")
    lines = []
    for result in results:
        lines += ["", f"{result['features']}  (ssl epoch {result['ssl_epoch']}, split {result['eval_split']})", header]
        for solver in ("sgd", "logistic"):
            row = result[solver]
            lines.append(
                f"{solver:<9} {row['train_avg']:9.2f} {row['train_wga']:9.2f} {row['eval_avg']:7.2f} "
                f"{row['eval_wga']:7.2f} {row['eval_avg_last10']:7.2f} {row['eval_wga_last10']:7.2f}  {row['eval_groups']}"
            )
        delta = result["sgd"]["eval_wga"] - result["logistic"]["eval_wga"]
        lines.append(f"sgd - logistic worst-group accuracy: {delta:+.2f}")
    return "\n".join(lines)


def main() -> None:
    defaults = LINEAR_PROBE_DEFAULTS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("features", nargs="+", help="probe_features_epoch_*.pt files or directories holding them")
    parser.add_argument("--seed", type=int, default=0, help="The probe seed; the training run uses its own --seed.")
    parser.add_argument("--output", default="", help="Write the comparison as JSON here.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=256, help="SGD probe batch size (the old run used 256).")
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--learning_rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--lr_decay_epochs", default="auto")
    parser.add_argument("--lr_decay_rate", type=float, default=defaults["lr_decay_rate"])
    parser.add_argument("--weight_decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--momentum", type=float, default=defaults["momentum"])
    parser.add_argument("--l2", type=float, default=defaults["probe_l2"])
    parser.add_argument("--tolerance", type=float, default=defaults["probe_tolerance"])
    parser.add_argument("--max_epochs", type=int, default=defaults["probe_max_epochs"])
    args = parser.parse_args()

    options = ProbeOptions(
        batch_size=args.batch_size, seed=args.seed, device=args.device, epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_decay_epochs=tuple(resolve_lr_decay_epochs(args.lr_decay_epochs, args.epochs)),
        lr_decay_rate=args.lr_decay_rate, weight_decay=args.weight_decay, momentum=args.momentum,
        l2=args.l2, tolerance=args.tolerance, max_epochs=args.max_epochs,
    )
    results = [compare(path, options) for path in feature_files(args.features)]
    print(report(results), flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"options": {**vars(args), "features": args.features}, "results": results}, indent=2))
        print(f"[INFO] Wrote {output}")


if __name__ == "__main__":
    main()
