"""Plain SimCLR, the method every other one starts from."""

from __future__ import annotations

from cospro.methods.base import TrainingMethod, register_method


@register_method
class SimCLROnly(TrainingMethod):
    name = "simclr"
    modes = ("none",)

    @classmethod
    def from_config(cls, config, **resolved) -> "SimCLROnly":
        return cls()
