# Spur SpLiCE

Spur SpLiCE tests whether sparse semantic concepts can improve visual
representations under spurious object-context correlations. The current method
is **SpLiCE-CRP**: frozen OpenCLIP/SpLiCE features produce a sparse teacher
graph, then a SimCLR ResNet trains with graph-aware batches and relational KL.

The graph and SSL stages use no target or group annotations. Labels and group
metadata appear only in linear evaluation. The teacher is unnecessary at
student inference.

## Project shape

- `splice/` — sparse decomposition, CRP graph construction and graph training.
- `experiments/spurious_eval/` — datasets, models, training and linear probes.
- `experiments/runner.py` — the single seed/arm experiment runner.
- `experiments/manifests/` — reproducible experiment definitions.
- `scripts/` — one Slurm adapter plus small cache/report tools.
- `outputs/` — seed-first results, shared artifacts and aggregate reports.
- `tests/` — tests for the current method only.

See [PROJECT_MAP.md](PROJECT_MAP.md) for the data flow and
[outputs/README.md](outputs/README.md) for artifact navigation.

## Run the canonical experiment

On the cluster, set `DATA_FOLDER` if it differs from the default and submit the
20-task matrix:

```bash
sbatch --array=0-19 scripts/run_experiment.sbatch
```

Inspect the matrix without training:

```bash
python -m experiments.runner experiments/manifests/waterbirds_crp.json --list
python -m experiments.runner experiments/manifests/waterbirds_crp.json --task 0 --dry-run
```

Build shared inputs directly:

```bash
python -m scripts.tools.cache_crp_features --help
python -m splice.crp --help
```

## Verification

```bash
conda activate grgrie-train
pip install -e .
python -m unittest discover -s tests -p 'test_*.py'
```

The final four-seed paper aggregate is
`outputs/reports/paper/test_summary.json`. Raw run folders are grouped under
`outputs/seeds/seed_XX/`; shared Waterbirds cache and graphs are under
`outputs/shared/waterbirds/`.
