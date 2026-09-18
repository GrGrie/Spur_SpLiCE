"""Seeded data loading and RNG isolation shared by SSL training and linear probing."""

from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np
import torch


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader_kwargs(args, shuffle: bool, seed: int | None = None) -> dict:
    """DataLoader options with a dedicated generator seeded from ``seed`` or ``args.seed``."""

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed if seed is None else seed)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "generator": loader_generator,
    }
    if shuffle or args.num_workers > 0:
        loader_kwargs["worker_init_fn"] = seed_worker
    return loader_kwargs


@contextmanager
def preserve_rng_state():
    """Prevent observational probes from changing subsequent SSL randomness."""

    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
