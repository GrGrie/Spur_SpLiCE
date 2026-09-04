"""Train the CRPv4 Slot Attention branch over frozen CLIP patch tokens."""

from __future__ import annotations

import argparse
import time

import torch

from CoBalT.data import spatial_train_loader
from CoBalT.runtime import add_wandb_args, atomic_torch_save, init_wandb, seed_everything
from CoBalT.spatial import FrozenClipPatchEncoder, SpatialSpliceDiscoveryModel
from splice.crp import validate_feature_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("CRPv4 label-free spatial slot discovery")
    parser.add_argument("--dataset", choices=["waterbirds", "celeba"], default="waterbirds")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache", required=True, help="Aligned SpLiCE CRP feature cache.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-min", type=float, default=0.2)
    parser.add_argument("--num-slots", type=int, default=4)
    parser.add_argument("--feature-source", choices=["vanilla", "sclip"], default="vanilla")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--student-temperature", type=float, default=0.1)
    parser.add_argument("--teacher-temperature", type=float, default=0.07)
    parser.add_argument("--teacher-momentum", type=float, default=0.99)
    parser.add_argument("--center-momentum", type=float, default=0.9)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    add_wandb_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_slots <= 0:
        raise ValueError("epochs, batch-size, and num-slots must be positive.")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = validate_feature_cache(
        torch.load(args.cache, map_location="cpu", weights_only=True)
    )
    provenance = cache.get("provenance", {})
    if str(provenance.get("dataset", "")).lower() != args.dataset:
        raise ValueError("CRP cache belongs to a different dataset.")
    model_name = str(provenance.get("splice_model", "open_clip:ViT-B-32")).split(":", 1)[-1]
    pretrained = str(provenance.get("splice_pretrained", "laion2b_s34b_b79k"))
    encoder = FrozenClipPatchEncoder(model_name, pretrained, args.feature_source).to(device)
    model = SpatialSpliceDiscoveryModel(
        encoder,
        cache["dictionary"],
        num_slots=args.num_slots,
        student_temperature=args.student_temperature,
        teacher_temperature=args.teacher_temperature,
        teacher_momentum=args.teacher_momentum,
        center_momentum=args.center_momentum,
        semantic_weight=args.semantic_weight,
    ).to(device)
    loader = spatial_train_loader(
        args.dataset, args.data_root, args.batch_size, args.workers, args.image_size, args.crop_min
    )
    optimizer = torch.optim.AdamW(
        model.student_grouping.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    model_config = {
        "model_name": model_name,
        "pretrained": pretrained,
        "feature_source": args.feature_source,
        "num_slots": args.num_slots,
        "student_temperature": args.student_temperature,
        "teacher_temperature": args.teacher_temperature,
        "teacher_momentum": args.teacher_momentum,
        "center_momentum": args.center_momentum,
        "semantic_weight": args.semantic_weight,
        "native_dim": encoder.native_dim,
        "output_dim": encoder.output_dim,
    }
    resolved = vars(args).copy()
    resolved.update({"device": str(device), "model": model_config})
    run = init_wandb(args, "crpv4_spatial_discovery", resolved)

    for epoch in range(args.epochs):
        model.train()
        totals = {"loss": 0.0, "distillation": 0.0, "semantic": 0.0, "agreement": 0.0}
        examples = 0
        started = time.perf_counter()
        for step, ((views, boxes, flips), _) in enumerate(loader):
            if args.max_steps_per_epoch is not None and step >= args.max_steps_per_epoch:
                break
            views, boxes, flips = views.to(device), boxes.to(device), flips.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(views, boxes, flips)
            scaler.scale(output.loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.update_teacher()
            batch_size = views.shape[0]
            examples += batch_size
            totals["loss"] += float(output.loss.detach()) * batch_size
            totals["distillation"] += float(output.distillation.detach()) * batch_size
            totals["semantic"] += float(output.semantic_consistency.detach()) * batch_size
            totals["agreement"] += float(output.concept_agreement.detach()) * batch_size
        if examples == 0:
            raise RuntimeError("The spatial discovery loader produced no complete batch.")
        metrics = {
            "epoch": epoch + 1,
            "ssl/loss": totals["loss"] / examples,
            "ssl/attention_distillation": totals["distillation"] / examples,
            "ssl/semantic_consistency": totals["semantic"] / examples,
            "ssl/concept_agreement": totals["agreement"] / examples,
            "ssl/examples": examples,
            "runtime/epoch_seconds": time.perf_counter() - started,
        }
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={metrics['ssl/loss']:.5f} "
            f"concept_agreement={metrics['ssl/concept_agreement']:.3f}",
            flush=True,
        )
        if run is not None:
            run.log(metrics, step=epoch + 1)
        atomic_torch_save(
            {
                "artifact": "splice_spatial_slots_v1",
                "dataset": args.dataset,
                "seed": args.seed,
                "epoch": epoch + 1,
                "model_config": model_config,
                "student_grouping": model.student_grouping.state_dict(),
                "teacher_grouping": model.teacher_grouping.state_dict(),
                "teacher_center": model.teacher_center.detach().cpu(),
                "cache_provenance": dict(provenance),
                "resolved_config": resolved,
                "metrics": metrics,
            },
            args.output,
        )
    if run is not None:
        run.summary.update(metrics)
        run.finish()


if __name__ == "__main__":
    main()
