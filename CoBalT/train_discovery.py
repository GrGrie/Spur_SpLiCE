from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from CoBalT.config import paper_config
from CoBalT.data import discovery_loader
from CoBalT.model import CoBalTDiscoveryModel
from CoBalT.runtime import add_wandb_args, atomic_torch_save, init_wandb, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("CoBalT stage 1: unsupervised concept discovery")
    parser.add_argument("--dataset", choices=["waterbirds", "celeba"], default="waterbirds")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-min", type=float, default=0.2)
    parser.add_argument("--num-slots", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=8)
    parser.add_argument("--slot-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--student-temperature", type=float, default=0.1)
    parser.add_argument("--teacher-temperature", type=float, default=0.07)
    parser.add_argument("--contrastive-temperature", type=float, default=0.2)
    parser.add_argument("--teacher-momentum", type=float, default=0.99)
    parser.add_argument("--codebook-momentum", type=float, default=0.9)
    parser.add_argument("--center-momentum", type=float, default=0.9)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--backbone", choices=["resnet50", "resnet18"], default="resnet50")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-nonpaper-backbone", action=argparse.BooleanOptionalAction, default=False)
    add_wandb_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = paper_config(args.dataset)
    args.epochs = args.epochs or config.discovery_epochs
    args.batch_size = args.batch_size or config.discovery_batch_size
    if not args.smoke and not args.allow_nonpaper_backbone and (args.backbone != "resnet50" or not args.pretrained):
        raise ValueError("Full paper runs require an ImageNet-pretrained ResNet-50 backbone.")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = discovery_loader(
        args.dataset,
        args.data_root,
        args.batch_size,
        args.workers,
        args.image_size,
        args.crop_min,
    )
    model_config = {
        "num_slots": args.num_slots,
        "codebook_size": args.codebook_size,
        "slot_dim": args.slot_dim,
        "hidden_dim": args.hidden_dim,
        "student_temperature": args.student_temperature,
        "teacher_temperature": args.teacher_temperature,
        "contrastive_temperature": args.contrastive_temperature,
        "teacher_momentum": args.teacher_momentum,
        "codebook_momentum": args.codebook_momentum,
        "center_momentum": args.center_momentum,
        "backbone": args.backbone,
        "pretrained": args.pretrained,
    }
    model = CoBalTDiscoveryModel(**model_config).to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    resolved = vars(args).copy()
    resolved.update({"paper_defaults": config.as_dict(), "device": str(device), "model": model_config})
    run = init_wandb(args, "concept_discovery", resolved)

    for epoch in range(args.epochs):
        model.train()
        totals = {"loss": 0.0, "distillation": 0.0, "contrastive": 0.0, "vq": 0.0}
        usage = torch.zeros(args.codebook_size)
        examples = 0
        started = time.perf_counter()
        for step, batch in enumerate(loader):
            if args.max_steps_per_epoch is not None and step >= args.max_steps_per_epoch:
                break
            (views, boxes, flips), _, _, _ = batch
            views = views.to(device, non_blocking=True)
            boxes = boxes.to(device, non_blocking=True)
            flips = flips.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(views, boxes, flips)
            scaler.scale(output.loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.update_teacher()
            batch_size = views.shape[0]
            examples += batch_size
            totals["loss"] += output.loss.detach().item() * batch_size
            totals["distillation"] += output.distillation.detach().item() * batch_size
            totals["contrastive"] += output.contrastive.detach().item() * batch_size
            totals["vq"] += output.vector_quantization.detach().item() * batch_size
            usage += output.code_usage.detach().cpu()
        if examples == 0:
            raise RuntimeError("The discovery loader produced no complete batch.")
        probabilities = usage / usage.sum().clamp_min(1)
        perplexity = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item()
        metrics = {
            "epoch": epoch + 1,
            "ssl/loss": totals["loss"] / examples,
            "ssl/distillation_loss": totals["distillation"] / examples,
            "ssl/contrastive_loss": totals["contrastive"] / examples,
            "ssl/vq_loss": totals["vq"] / examples,
            "ssl/codebook_perplexity": perplexity,
            "ssl/active_codes": int(usage.gt(0).sum().item()),
            "ssl/examples": examples,
            "runtime/epoch_seconds": time.perf_counter() - started,
        }
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={metrics['ssl/loss']:.5f} "
            f"codes={metrics['ssl/active_codes']}/{args.codebook_size} perplexity={perplexity:.3f}"
        )
        if run is not None:
            run.log(metrics, step=epoch + 1)
        atomic_torch_save(
            {
                "artifact": "cobalt_discovery_v1",
                "dataset": args.dataset,
                "epoch": epoch + 1,
                "seed": args.seed,
                "model_config": model_config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "resolved_config": resolved,
                "metrics": metrics,
            },
            args.output,
        )
    if run is not None:
        run.summary.update(metrics)
        run.summary["checkpoint"] = str(Path(args.output).resolve())
        run.finish()


if __name__ == "__main__":
    main()
