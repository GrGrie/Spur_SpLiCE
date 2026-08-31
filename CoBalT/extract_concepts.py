from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from CoBalT.data import split_loader
from CoBalT.model import CoBalTDiscoveryModel
from CoBalT.runtime import atomic_torch_save, seed_everything


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("CoBalT: infer fixed concepts for classifier training")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("artifact") != "cobalt_discovery_v1":
        raise ValueError("Checkpoint is not a CoBalT discovery artifact.")
    seed_everything(int(checkpoint["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = dict(checkpoint["model_config"])
    model_config["pretrained"] = False
    model = CoBalTDiscoveryModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    records = {}
    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        loader = split_loader(
            checkpoint["dataset"],
            args.data_root,
            split,
            args.batch_size,
            args.workers,
            args.image_size,
        )
        all_concepts, all_ids = [], []
        with torch.inference_mode():
            for images, _, _, sample_ids in loader:
                concepts, _ = model.infer_concepts(images.to(device, non_blocking=True))
                all_concepts.append(concepts.cpu())
                all_ids.append(sample_ids.cpu())
        records[split] = {
            "sample_ids": torch.cat(all_ids),
            "concepts": torch.cat(all_concepts),
        }
        print(f"split={split} samples={len(records[split]['sample_ids'])}")

    atomic_torch_save(
        {
            "artifact": "cobalt_concepts_v1",
            "dataset": checkpoint["dataset"],
            "seed": checkpoint["seed"],
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "model_config": checkpoint["model_config"],
            "splits": records,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
