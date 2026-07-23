#!/usr/bin/env python
"""Launch the matched 500-epoch baseline/localized validation experiment."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "localized", "both"], default="both")
    parser.add_argument("--dataset", choices=["waterbirds", "celeba", "spur_cifar10"], default="waterbirds")
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb_name", default="Spur_SpLiCE")
    parser.add_argument("--entity", default="gsgrechkin-rptu")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    return parser.parse_args()


def command_for(args: argparse.Namespace, variant: str) -> list[str]:
    model = "resnet18" if args.dataset == "spur_cifar10" else "resnet18_large"
    command = [
        args.python,
        "spur_splice.py",
        "--dataset",
        args.dataset,
        "--data_folder",
        args.data_folder,
        "--model",
        model,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--epochs",
        "500",
        "--lr_decay_epochs",
        "auto",
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--linear_probe_epochs",
        "100",
        "--linear_lr_decay_epochs",
        "auto",
        "--linear_probe_mode",
        "periodic",
        "--linear_probe_freq",
        "25",
        "--rank_eval_freq",
        "0",
        "--keep_checkpoints",
        "--save_freq",
        "50",
        "--wandb_group",
        f"{args.dataset}-layer-localized-500-seed-{args.seed}",
        "--wandb_tags",
        "layer_localized,protocol_500,matched_comparison",
    ]
    if args.use_wandb:
        command.extend(["--use_wandb", "--wandb_name", args.wandb_name, "--entity", args.entity])
    if variant == "baseline":
        command.extend(
            [
                "--splice_mode",
                "none",
                "--wandb_run_name",
                f"{args.dataset}_S{args.seed}_Baseline_E500",
            ]
        )
    else:
        command.extend(
            [
                "--splice_mode",
                "localized",
                "--splice_concepts",
                "auto",
                "--splice_auto_top_k",
                "10",
                "--localized_probe_freq",
                "10",
                "--localized_start_epoch",
                "10",
                "--localized_stability",
                "2",
                "--localized_leakage_threshold",
                "0.05",
                "--localized_leakage_max",
                "0.50",
                "--localized_probe_samples",
                "2048",
                "--localized_shuffle_repeats",
                "3",
                "--localized_protect_target",
                "true",
                "--wandb_run_name",
                f"{args.dataset}_S{args.seed}_Localized_E500",
            ]
        )
    command.extend(args.extra)
    return command


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    variants = ["baseline", "localized"] if args.variant == "both" else [args.variant]
    for variant in variants:
        command = command_for(args, variant)
        print(f"[{variant}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
