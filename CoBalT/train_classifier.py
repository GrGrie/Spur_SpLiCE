from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from CoBalT.config import paper_config
from CoBalT.data import full_dataset, split_loader, validate_concept_artifact
from CoBalT.runtime import add_wandb_args, atomic_torch_save, init_wandb, seed_everything
from CoBalT.sampler import ConceptBalancedSampler, inferred_worst_group_accuracy
from experiments.spurious_eval.metrics import compute_group_metrics


def build_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.inference_mode()
def evaluate(model, loader, concepts: torch.Tensor | None, device: torch.device) -> dict:
    model.eval()
    predictions, labels, metadata = [], [], []
    for images, target, batch_metadata, _ in loader:
        logits = model(images.to(device, non_blocking=True))
        predictions.append(logits.argmax(dim=1).cpu())
        labels.append(target.cpu())
        metadata.append(batch_metadata.cpu())
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    metadata = torch.cat(metadata)
    human = compute_group_metrics(predictions, labels, metadata)
    result = {
        "average": human.average,
        "human_worst": human.worst_group,
        "human_best": human.best_group,
        "group_accuracy": human.group_accuracy,
        "group_counts": human.group_counts,
    }
    if concepts is not None:
        result["inferred_worst"] = inferred_worst_group_accuracy(predictions, labels, concepts)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("CoBalT stage 2: concept-balanced robust classifier")
    parser.add_argument("--dataset", choices=["waterbirds", "celeba"], default="waterbirds")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--concepts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--sampling-lambda", type=float, default=None)
    parser.add_argument("--selectors", default="avg,ig,hg")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    add_wandb_args(parser)
    args = parser.parse_args(argv)

    config = paper_config(args.dataset)
    args.epochs = args.epochs or config.classifier_epochs
    args.batch_size = args.batch_size or config.classifier_batch_size
    args.sampling_lambda = (
        config.sampling_lambda if args.sampling_lambda is None else args.sampling_lambda
    )
    if not args.smoke and not args.pretrained:
        raise ValueError("Full paper runs require an ImageNet-pretrained ResNet-50 classifier.")
    selectors = [value.strip() for value in args.selectors.split(",") if value.strip()]
    if not set(selectors).issubset({"avg", "ig", "hg"}):
        raise ValueError("Selectors must be a comma-separated subset of avg,ig,hg.")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.concepts, map_location="cpu", weights_only=True)
    validate_concept_artifact(artifact, args.dataset, args.data_root, ("train", "val"))
    train_concepts = artifact["splits"]["train"]["concepts"].long()
    val_concepts = artifact["splits"]["val"]["concepts"].long()
    dataset = full_dataset(args.dataset, args.data_root)
    train_subset = dataset.get_subset("train")
    sampler = ConceptBalancedSampler(
        train_concepts,
        train_subset.y_array,
        args.sampling_lambda,
        num_samples=len(train_subset),
        seed=args.seed,
    )
    train_loader = split_loader(
        args.dataset,
        args.data_root,
        "train",
        args.batch_size,
        args.workers,
        args.image_size,
        training=True,
        sampler=sampler,
    )
    val_loader = split_loader(
        args.dataset, args.data_root, "val", args.batch_size, args.workers, args.image_size
    )
    test_loader = split_loader(
        args.dataset, args.data_root, "test", args.batch_size, args.workers, args.image_size
    )
    model = build_classifier(config.num_classes, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    resolved = vars(args).copy()
    resolved.update(
        {
            "paper_defaults": config.as_dict(),
            "device": str(device),
            "concept_checkpoint_sha256": artifact["checkpoint_sha256"],
            "train_samples": len(train_subset),
        }
    )
    run = init_wandb(args, "robust_classifier", resolved)
    output_dir = Path(args.output_dir)
    best = {selector: (-float("inf"), None) for selector in selectors}

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        started = time.perf_counter()
        for step, (images, labels, _, _) in enumerate(train_loader):
            if args.max_steps_per_epoch is not None and step >= args.max_steps_per_epoch:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().item() * images.shape[0]
            correct += logits.argmax(dim=1).eq(labels).sum().item()
            seen += images.shape[0]
        if seen == 0:
            raise RuntimeError("The classifier loader produced no samples.")
        validation = evaluate(model, val_loader, val_concepts, device)
        metrics = {
            "epoch": epoch + 1,
            "train/loss": loss_sum / seen,
            "train/accuracy": correct / seen,
            "val/average": validation["average"],
            "val/inferred_worst": validation["inferred_worst"],
            "val/human_worst": validation["human_worst"],
            "runtime/epoch_seconds": time.perf_counter() - started,
        }
        criteria = {
            "avg": validation["average"],
            "ig": validation["inferred_worst"],
            "hg": validation["human_worst"],
        }
        for selector in selectors:
            if criteria[selector] > best[selector][0]:
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best[selector] = (criteria[selector], state)
                atomic_torch_save(
                    {
                        "artifact": "cobalt_classifier_v1",
                        "dataset": args.dataset,
                        "selector": selector,
                        "epoch": epoch + 1,
                        "selection_metric": criteria[selector],
                        "model": state,
                        "resolved_config": resolved,
                        "validation": validation,
                    },
                    output_dir / f"best_{selector}.pt",
                )
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={metrics['train/loss']:.5f} "
            f"val_avg={validation['average']:.4f} val_ig={validation['inferred_worst']:.4f} "
            f"val_hg={validation['human_worst']:.4f}"
        )
        if run is not None:
            run.log(metrics, step=epoch + 1)

    final_results = {}
    for selector in selectors:
        model.load_state_dict(best[selector][1])
        test = evaluate(model, test_loader, None, device)
        final_results[selector] = test
        summary = {
            f"test/{selector}/average": test["average"],
            f"test/{selector}/human_worst": test["human_worst"],
            f"test/{selector}/human_best": test["human_best"],
            f"selection/{selector}/best_validation": best[selector][0],
        }
        print(
            f"selector={selector} test_avg={test['average']:.4f} "
            f"test_worst_group={test['human_worst']:.4f}"
        )
        if run is not None:
            run.summary.update(summary)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
