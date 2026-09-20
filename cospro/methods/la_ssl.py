"""LA-SSL: the unchanged SimCLR objective with learning-speed-aware resampling."""

from __future__ import annotations

from typing import Any

from cospro.methods.base import LoaderContext, TrainingMethod, register_method
from experiments.spurious_eval.training.la_ssl import build_la_ssl_loader


@register_method
class LaSSL(TrainingMethod):
    name = "la_ssl"
    requires_la_ssl = True

    def __init__(self, *, seed: int = 0, eta: float = 0.1, gamma: float = 10.0, quantile: float = 0.1,
                 warmup_epochs: int = 10, update_freq: int = 2) -> None:
        self.seed = seed
        self.settings = {
            "eta": eta, "gamma": gamma, "quantile": quantile,
            "warmup_epochs": warmup_epochs, "update_freq": update_freq,
        }
        self.state = None

    @classmethod
    def from_config(cls, config, **resolved) -> "LaSSL":
        options = config.la_ssl
        return cls(
            seed=config.runtime.seed,
            eta=options.la_ssl_eta,
            gamma=options.la_ssl_gamma,
            quantile=options.la_ssl_quantile,
            warmup_epochs=options.la_ssl_warmup_epochs,
            update_freq=options.la_ssl_update_freq,
        )

    def wrap_loader(self, loader, context: LoaderContext):
        wrapped = build_la_ssl_loader(
            loader, seed=self.seed, worker_init_fn=context.worker_init_fn, **self.settings,
        )
        self.state = wrapped.la_ssl
        return wrapped

    def sampling_state(self):
        return self.state

    def refresh_sampling(self, model, device, temperature: float, epoch: int) -> dict[str, float]:
        if self.state is None:
            return {}
        return self.state.refresh(model, device, temperature, epoch)

    def provenance(self) -> dict[str, Any]:
        return {}
