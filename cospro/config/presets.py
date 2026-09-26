"""Named sets of training options for recurring setups.

A preset replaces the defaults of the options it names; explicit command-line options still win.
The preset name itself stays out of the resolved configuration, so a preset run and a run with
the same values spelled out share one storage name.
"""

from __future__ import annotations

from typing import Any

PRESETS: dict[str, dict[str, Any]] = {
    # The CoSpRo student used by the paper manifests, the standalone launcher in automatic graph
    # mode and the full pipeline.
    "cospro_student": {
        "batch_size": 128,
        "num_workers": 4,
        "temp": 0.05,
        "splice_mode": "cospro_relational",
        "splice_weight": 0.5,
        "cospro_temperature": 0.25,
    },
    # The shared SSL protocol of the paper runs without a method attached: batch 128, four loader
    # workers within the five-CPU job and the SimCLR temperature 0.05. Standalone baselines and
    # concept-factor runs start from it.
    "matched": {
        "batch_size": 128,
        "num_workers": 4,
        "temp": 0.05,
    },
}


def preset_values(name: str | None) -> dict[str, Any]:
    if name is None:
        return {}
    try:
        return dict(PRESETS[name])
    except KeyError as exc:
        raise ValueError(f"Unknown preset {name!r}; choose one of {sorted(PRESETS)}.") from exc
