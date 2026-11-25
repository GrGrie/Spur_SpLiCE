"""Fine-tune the last blocks of an official I-JEPA encoder on Waterbirds and refit its head.

    python -m cospro.cli.finetune_ijepa --model ijepa_vith14_in1k --data_folder ~/Datasets --use_wandb

Two heads are evaluated on the same fine-tuned encoder:

``finetuned``  the linear head trained jointly with the last blocks on ``--train_split`` (default
               ``cb_train``: equal classes, 95% background correlation), with cross-entropy
``dfr``        a logistic head refitted on the frozen fine-tuned features of the group-balanced
               ``--dfr_split`` (default ``ds_train``), as in deep feature reweighting
               (Kirichenko et al., ICLR 2023)

Validation group accuracy is measured after every epoch; the run reports the last epoch and selects
nothing on the evaluation split. ``--final_test`` evaluates the final model on test as well.
Outputs: ``outputs/seeds/<study>/seed_<NN>/<model>/<attempt>/run.json`` and, unless
``--no-save_checkpoint``, the trained parameters under
``<scratch>/checkpoints/Spur_SpLiCE/<study>/seed_<NN>/<model>/<attempt>/finetuned.pt``.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cospro.config.settings import wandb_entity
from cospro.data.base import dataset_transforms
from cospro.data.registry import dataset_class
from cospro.evaluation.logistic_probe import fit_logistic_probe
from cospro.methods.ijepa_finetune import PartialFineTune, parameter_groups, trainable_state, warmup_cosine
from cospro.metrics import compute_group_metrics
from cospro.models.ijepa import IJEPA_MODEL_NAMES, build_ijepa_encoder
from cospro.tracking.artifacts import atomic_write_json, run_directory, scratch_binary_directory

DATASET = "waterbirds"
GROUP_NAMES = ("landbird on land", "landbird on water", "waterbird on land", "waterbird on water")
TRAIN_SPLITS = ("cb_train", "train", "ds_train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="ijepa_vith14_in1k", choices=IJEPA_MODEL_NAMES)
    parser.add_argument("--ckpt", default="", help="Official I-JEPA checkpoint; empty downloads it to scratch.")
    parser.add_argument("--data_folder", default=os.environ.get("DATA_FOLDER", ""))
    parser.add_argument("--train_split", default="cb_train", choices=TRAIN_SPLITS)
    parser.add_argument("--dfr_split", default="ds_train", choices=("ds_train",))
    parser.add_argument("--final_test", action="store_true", help="Also evaluate the final model on test.")
    parser.add_argument("--trainable_blocks", type=int, default=8, help="Last encoder blocks that train (of 32).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5, help="Peak AdamW learning rate of the encoder blocks.")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="Peak AdamW learning rate of the head.")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute the trainable blocks in the backward pass to save memory.")
    parser.add_argument("--dfr_l2", type=float, default=1e-3, help="L2 of the refitted logistic head.")
    parser.add_argument("--save_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study", default="ijepa_finetune")
    parser.add_argument("--attempt_id", default="local")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_name", default="CoSpRo")
    parser.add_argument("--entity", default=wandb_entity())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch_size must be positive.")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def group_report(predictions: torch.Tensor, labels: torch.Tensor, metadata: torch.Tensor) -> dict:
    """Average, worst-group and per-group accuracy in percent."""

    metrics = compute_group_metrics(predictions, labels, metadata, n_groups=len(GROUP_NAMES))
    return {
        "accuracy": round(100 * metrics.average, 2),
        "worst_group_accuracy": round(100 * metrics.worst_group, 2),
        "group_accuracy": {name: round(100 * float(value), 2)
                           for name, value in zip(GROUP_NAMES, metrics.group_accuracy)},
        "group_count": {name: int(count) for name, count in zip(GROUP_NAMES, metrics.group_counts)},
    }


@torch.no_grad()
def extract(model: PartialFineTune, subset, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Features, head logits, labels and metadata of ``subset`` in order."""

    model.eval()
    loader = DataLoader(subset, batch_size=args.eval_batch_size, num_workers=args.num_workers, shuffle=False)
    features, logits, labels, metadata = [], [], [], []
    for images, batch_labels, batch_metadata in loader:
        batch_features = model.features(images.to(args.device, non_blocking=True))
        features.append(batch_features.cpu())
        logits.append(model.head(batch_features).cpu())
        labels.append(batch_labels)
        metadata.append(batch_metadata)
    return torch.cat(features), torch.cat(logits), torch.cat(labels), torch.cat(metadata)


def train_epoch(model, loader, optimizer, scaler, schedule, step: int, args) -> tuple[dict, int]:
    model.train()
    total_loss = correct = seen = 0.0
    for images, labels, _ in loader:
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        factor = schedule(step)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step += 1
        total_loss += float(loss) * len(labels)
        correct += float((logits.argmax(dim=1) == labels).sum())
        seen += len(labels)
    return {"loss": round(total_loss / seen, 5), "accuracy": round(100 * correct / seen, 2)}, step


def dfr_head(train_features, train_labels, eval_sets: dict, l2: float) -> dict:
    """Logistic heads refitted on ``train_features``, reported on every evaluation set."""

    reports = {}
    for split, (features, _, labels, metadata) in eval_sets.items():
        records, convergence = fit_logistic_probe(train_features, train_labels, features, labels,
                                                  num_classes=2, l2=l2)
        reports[split] = {**group_report(records[-1].eval_predictions, labels, metadata),
                          "converged": convergence["converged"]}
    return reports


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    seed_everything(args.seed)
    output_dir = run_directory(args.seed, args.study, args.model, args.attempt_id)
    identity = {"study": args.study, "seed": args.seed, "arm": args.model, "attempt_id": args.attempt_id}
    print(f"[results] Run JSON: {output_dir / 'run.json'}", flush=True)

    adapter_class = dataset_class(DATASET)
    dataset = adapter_class(str(args.data_folder))
    _, train_transform, eval_transform = dataset_transforms(adapter_class.image_size, adapter_class.mean,
                                                            adapter_class.std)
    train = dataset.get_subset(args.train_split, transform=train_transform)
    dfr_train = dataset.get_subset(args.dfr_split, transform=eval_transform)
    eval_splits = ["val", "test"] if args.final_test else ["val"]
    evaluation = {split: dataset.get_subset(split, transform=eval_transform) for split in eval_splits}
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=generator,
                        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))

    encoder, _ = build_ijepa_encoder(args.model, args.ckpt)
    model = PartialFineTune(encoder, num_classes=2, trainable_blocks=args.trainable_blocks,
                            gradient_checkpointing=args.gradient_checkpointing).to(args.device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"[INFO] {trainable / 1e6:.1f}M trainable parameters; {len(train)} {args.train_split} images", flush=True)
    groups = parameter_groups(model, args.lr, args.head_lr, args.weight_decay)
    for group in groups:
        group["base_lr"] = group["lr"]
    optimizer = torch.optim.AdamW(groups)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    total_steps = args.epochs * len(loader)
    warmup_steps = int(round(args.warmup_epochs * len(loader)))
    schedule = lambda step: warmup_cosine(step, total_steps, warmup_steps)  # noqa: E731

    run = None
    if args.use_wandb:
        import wandb

        run = wandb.init(project=args.wandb_name, entity=args.entity, config=vars(args),
                         name=f"{args.model}_ft{args.trainable_blocks}_{args.train_split}_S{args.seed}",
                         tags=["ijepa_finetune", args.train_split])

    history, step, start = [], 0, time.time()
    for epoch in range(1, args.epochs + 1):
        train_stats, step = train_epoch(model, loader, optimizer, scaler, schedule, step, args)
        _, logits, labels, metadata = extract(model, evaluation["val"], args)
        val = group_report(logits.argmax(dim=1), labels, metadata)
        history.append({"epoch": epoch, "train": train_stats, "val": val})
        print(f"[epoch {epoch}/{args.epochs}] loss {train_stats['loss']:.4f} train acc {train_stats['accuracy']:.2f} "
              f"| val acc {val['accuracy']:.2f} WGA {val['worst_group_accuracy']:.2f} "
              f"| {time.time() - start:.0f} s", flush=True)
        if run is not None:
            run.log({"epoch": epoch, "train/loss": train_stats["loss"], "train/accuracy": train_stats["accuracy"],
                     "val/accuracy": val["accuracy"], "val/worst_group_accuracy": val["worst_group_accuracy"],
                     **{f"val/group/{name}": value for name, value in val["group_accuracy"].items()}}, step=epoch)

    print("[INFO] Evaluating the fine-tuned head and refitting a DFR head", flush=True)
    eval_sets = {split: extract(model, subset, args) for split, subset in evaluation.items()}
    finetuned = {split: group_report(logits.argmax(dim=1), labels, metadata)
                 for split, (_, logits, labels, metadata) in eval_sets.items()}
    dfr_features, _, dfr_labels, _ = extract(model, dfr_train, args)
    dfr = dfr_head(dfr_features, dfr_labels, eval_sets, args.dfr_l2)

    checkpoint_path = None
    if args.save_checkpoint:
        checkpoint_path = scratch_binary_directory("checkpoints", identity) / "finetuned.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": args.model, "trainable_blocks": args.trainable_blocks,
                    "state_dict": trainable_state(model)}, checkpoint_path)
        print(f"[INFO] Saved trained parameters to {checkpoint_path}", flush=True)

    headline = "test" if args.final_test else "val"
    payload = {
        "schema": "ijepa-finetune-v1",
        "config": {key: value for key, value in vars(args).items() if key not in {"entity"}},
        "train_images": len(train),
        "dfr_images": len(dfr_train),
        "trainable_parameters": trainable,
        "history": history,
        "final": {"finetuned_head": finetuned, "dfr_head": dfr},
        "headline_split": headline,
        "worst_group_accuracy": {"finetuned_head": finetuned[headline]["worst_group_accuracy"],
                                 "dfr_head": dfr[headline]["worst_group_accuracy"]},
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "seconds": round(time.time() - start, 1),
    }
    path = atomic_write_json(output_dir / "run.json", payload)
    for head, reports in payload["final"].items():
        for split, report in reports.items():
            print(f"[final] {head} on {split}: acc {report['accuracy']:.2f} WGA {report['worst_group_accuracy']:.2f} "
                  f"{report['group_accuracy']}", flush=True)
    if run is not None:
        run.summary.update({f"final/{head}/{split}/{key}": report[key]
                            for head, reports in payload["final"].items() for split, report in reports.items()
                            for key in ("accuracy", "worst_group_accuracy")})
        run.summary.update({f"worst_group_accuracy/{head}": value
                            for head, value in payload["worst_group_accuracy"].items()})
        run.finish()
    print(f"[results] Wrote {path}", flush=True)
    return path


if __name__ == "__main__":
    main()
