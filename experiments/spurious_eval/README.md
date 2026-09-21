# Moved

The dataset adapters, models, training loop and linear probe that lived here are now in `cospro/`:
`cospro/data/`, `cospro/models/`, `cospro/training/`, `cospro/evaluation/` and
`cospro/cli/linear_probe.py`. The modules left in this directory are shims that keep old imports
and `python -m experiments.spurious_eval.linear_probe` working. See `PROJECT_MAP.md`.
