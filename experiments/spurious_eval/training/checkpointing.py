from __future__ import annotations

import os
import random

import numpy as np
import torch

from splice.layer_localized import LayerLocalizedScrubber


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args,
    epoch: int,
    path: str,
    scaler=None,
    loader_generator: torch.Generator | None = None,
) -> None:
    print(f"==> Saving checkpoint to {path}")
    tmp_path = f"{path}.tmp"
    torch.save(
        {
            "opt": args,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "loader_generator_state": loader_generator.get_state() if loader_generator is not None else None,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
    scaler=None,
    loader_generator: torch.Generator | None = None,
) -> int:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    reproducibility_keys = {
        "scaler",
        "torch_rng_state",
        "numpy_rng_state",
        "python_rng_state",
        "loader_generator_state",
    }
    missing_reproducibility_keys = sorted(reproducibility_keys.difference(checkpoint))
    if missing_reproducibility_keys:
        print(
            "[WARN] Checkpoint predates full reproducibility state; exact continuation is not guaranteed. "
            f"Missing: {missing_reproducibility_keys}"
        )
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if loader_generator is not None and checkpoint.get("loader_generator_state") is not None:
        loader_generator.set_state(checkpoint["loader_generator_state"].cpu())
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])
    if checkpoint.get("numpy_rng_state") is not None:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if checkpoint.get("python_rng_state") is not None:
        random.setstate(checkpoint["python_rng_state"])
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
    localized_readouts = {
        key.removeprefix("localized_scrubber.readout_"): value
        for key, value in encoder_state.items()
        if key.startswith("localized_scrubber.readout_")
    }
    if localized_readouts:
        max_concepts = next(iter(localized_readouts.values())).shape[0]
        stage_dims = {stage: int(value.shape[1]) for stage, value in localized_readouts.items()}
        encoder.localized_scrubber = LayerLocalizedScrubber(stage_dims, max_concepts)
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("fc_reduce.")]
    if unexpected:
        raise ValueError(f"Unexpected encoder checkpoint keys: {unexpected}")
    missing = [key for key in missing if not key.startswith("fc_reduce.")]
    if missing:
        raise ValueError(f"Missing encoder checkpoint keys: {missing}")
