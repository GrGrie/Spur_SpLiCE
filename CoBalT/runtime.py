from __future__ import annotations

import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime_versions() -> dict[str, str]:
    import torchvision

    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def init_wandb(args, stage: str, resolved_config: dict):
    if args.no_wandb:
        if not args.smoke:
            raise ValueError("--no-wandb is allowed only together with --smoke.")
        return None

    import wandb

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    config = dict(resolved_config)
    config["runtime"] = runtime_versions()
    config["stage"] = stage
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or f"CoBalT_{args.dataset}_s{args.seed}_{stage}",
        group=args.wandb_group or f"CoBalT_{args.dataset}_s{args.seed}",
        tags=tags + ["CoBalT", f"dataset_{args.dataset}", f"stage_{stage}", f"seed_{args.seed}"],
        config=config,
    )


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def add_wandb_args(parser) -> None:
    parser.add_argument("--wandb-project", default="Spur_SpLiCE")
    parser.add_argument("--wandb-entity", default="gsgrechkin-rptu")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-tags", default="baseline_cobalt,paper_reproduction")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--smoke", action="store_true")
