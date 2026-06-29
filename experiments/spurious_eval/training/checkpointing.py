from __future__ import annotations

import os
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args,
    epoch: int,
    path: str,
) -> None:
    print(f"==> Saving checkpoint to {path}")
    tmp_path = f"{path}.tmp"
    torch.save(
        {
            "opt": args,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> int:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


def load_encoder_checkpoint(encoder: torch.nn.Module, checkpoint_path: str) -> None:
    """Load only encoder weights from a SpurSSL checkpoint."""

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    encoder_state = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        if key.startswith("encoder."):
            encoder_state[key[len("encoder.") :]] = value
    if not encoder_state:
        raise ValueError(f"No encoder.* weights found in checkpoint: {checkpoint_path}")
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("fc_reduce.")]
    if unexpected:
        raise ValueError(f"Unexpected encoder checkpoint keys: {unexpected}")
    missing = [key for key in missing if not key.startswith("fc_reduce.")]
    if missing:
        raise ValueError(f"Missing encoder checkpoint keys: {missing}")
