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
the README inside the selected artifact root for artifact navigation.

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

Existing execution directories are protected by default. Choose
`--existing reuse`, `resume`, or `new-attempt` explicitly; reuse/resume require
the recorded command to match. Use `--artifact-root outputs/output_cluster` or
set `SPUR_SPLICE_ARTIFACT_ROOT` to operate on the packaged artifact layout.

Build shared inputs directly:

```bash
python -m scripts.tools.cache_crp_features --help
python -m splice.crp --help
python -m scripts.tools.build_crp_baseline_graphs --help
python -m scripts.tools.build_paper_results --artifact-root outputs/output_cluster
```

## Verification

```bash
conda activate grgrie-train
pip install -e .
python -m unittest discover -s tests -p 'test_*.py'
```

The checked root `paper_results.json` is regenerated from the selected artifact
tree. In the packaged copy, its source aggregate is
`outputs/output_cluster/reports/paper/test_summary.json`.

For this study, DONE means: a predeclared split/protocol, unique run identity,
linked endpoint metrics, completed and converged status, and explicit limits.
It does not mean producing a new standalone report for every implementation
detail or rerunning training when no scientific claim would change.
