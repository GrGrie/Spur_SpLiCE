"""LA-SSL Algorithm 1 / Eq. (3), arXiv:2311.16361v2.

Uses a complete, no-gradient scoring pass each epoch, including unsampled
images. See docs/SUBMISSION_RUNS.md for the declared Waterbirds adaptation.
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def sampling_probabilities(scores: torch.Tensor, quantile: float, gamma: float) -> torch.Tensor:
    scores = scores.detach().double().cpu()
    if scores.ndim != 1 or not scores.numel() or not torch.isfinite(scores).all() or (scores <= 0).any():
        raise ValueError("LA-SSL requires finite positive similarity scores.")
    if not 0 < quantile < 1 or gamma <= 0:
        raise ValueError("Invalid LA-SSL quantile or scale.")
    threshold = torch.quantile(scores, quantile)
    weights = (threshold - gamma * (scores - threshold)).clamp_min(0)
    return weights / weights.sum()


class LearningSpeedState:
    def __init__(self, score_loader, probabilities, *, eta, gamma, quantile, warmup_epochs, update_freq):
        self.score_loader = score_loader
        self.probabilities = probabilities
        self.ema = torch.zeros_like(probabilities)
        self.epoch = 0
        self.config = dict(eta=eta, gamma=gamma, quantile=quantile,
                           warmup_epochs=warmup_epochs, update_freq=update_freq)

    @torch.no_grad()
    def refresh(self, model, device, temperature, epoch):
        if epoch != self.epoch + 1:
            raise ValueError("LA-SSL requires consecutive epochs or a restored learning-speed state.")
        start = time.monotonic()
        modes = [(module, module.training) for module in model.modules()]
        model.eval()  # Scoring must not update BatchNorm or optimizer state.
        values = []
        try:
            for data in self.score_loader:
                views = data[0]
                images = torch.cat([views[0], views[1]]).to(device)
                first, second = model(images).float().chunk(2)
                # The paper defines sim as exp(normalized projection dot / tau).
                values.append(((first * second).sum(1).double() / temperature).exp().cpu())
        finally:
            for module, training in modes:
                module.training = training
        scores = torch.cat(values)
        if scores.shape != self.ema.shape or not torch.isfinite(scores).all():
            raise ValueError("LA-SSL scoring pass did not cover the complete dataset with finite scores.")
        eta = self.config["eta"]
        if self.epoch == 0:
            self.ema.copy_(scores)
        else:
            self.ema.mul_(1 - eta).add_(scores, alpha=eta)
        self.epoch = epoch
        updated = epoch > self.config["warmup_epochs"] and epoch % self.config["update_freq"] == 0
        if updated:
            self.probabilities.copy_(sampling_probabilities(self.ema, self.config["quantile"], self.config["gamma"]))
        positive = self.probabilities > 0
        return {
            "la_ssl_score_seconds": time.monotonic() - start,
            "la_ssl_score_examples": len(scores),
            "la_ssl_ema_mean": float(self.ema.mean()),
            "la_ssl_probability_max": float(self.probabilities.max()),
            "la_ssl_probability_nonzero_fraction": float(positive.double().mean()),
            "la_ssl_effective_sample_size": float(1 / self.probabilities.square().sum()),
            "la_ssl_probability_updated": int(updated),
        }

    def state_dict(self):
        return {"config": self.config, "epoch": self.epoch, "ema": self.ema.clone(),
                "probabilities": self.probabilities.clone(),
                "score_generator": self.score_loader.generator.get_state()}

    def load_state_dict(self, state):
        if state["config"] != self.config or state["ema"].shape != self.ema.shape:
            raise ValueError("Cannot resume LA-SSL with different learning-speed settings or dataset size.")
        self.epoch = int(state["epoch"])
        self.ema.copy_(state["ema"].cpu())
        self.probabilities.copy_(state["probabilities"].cpu())
        self.score_loader.generator.set_state(state["score_generator"].cpu())


def build_la_ssl_loader(loader, args, worker_init_fn):
    probabilities = torch.full((len(loader.dataset),), 1 / len(loader.dataset), dtype=torch.float64)
    # Replacement is essential: weighted permutations would still visit every
    # image once and would not implement LA-SSL's upsampling.
    sampler = WeightedRandomSampler(probabilities, len(loader.dataset), replacement=True, generator=loader.generator)
    common = dict(batch_size=loader.batch_size, num_workers=loader.num_workers,
                  pin_memory=loader.pin_memory, drop_last=False, collate_fn=loader.collate_fn,
                  worker_init_fn=worker_init_fn if loader.num_workers else None)
    training = DataLoader(loader.dataset, sampler=sampler, generator=loader.generator, **common)
    scoring = DataLoader(loader.dataset, shuffle=False,
                         generator=torch.Generator().manual_seed(args.seed + 2_000_000), **common)
    training.la_ssl = LearningSpeedState(
        scoring, sampler.weights, eta=args.la_ssl_eta, gamma=args.la_ssl_gamma,
        quantile=args.la_ssl_quantile, warmup_epochs=args.la_ssl_warmup_epochs,
        update_freq=args.la_ssl_update_freq,
    )
    return training
