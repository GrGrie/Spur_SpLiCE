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
- `outputs/` — Git-friendly run records, shared JSON artifacts, aggregate reports and ignored Slurm logs.
- `tests/` — tests for the current method only.

See [PROJECT_MAP.md](PROJECT_MAP.md) for the data flow,
[`docs/REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md) for the canonical storage
policy, and `outputs/README.md` for result navigation.

## Run the canonical experiment

On the cluster, set `DATA_FOLDER` if it differs from the default. The submission
helper launches the matrix and a dependent result collector:

```bash
bash scripts/submit_experiment.sh experiments/manifests/waterbirds_crp.json
```

Inspect the matrix without training:

```bash
python -m experiments.runner experiments/manifests/waterbirds_crp.json --list
python -m experiments.runner experiments/manifests/waterbirds_crp.json --task 0 --dry-run
```

The held-out test protocol is declared in the manifest and must be requested
explicitly. It runs one final probe on `test` instead of periodic validation
probes:

```bash
python -m experiments.runner experiments/manifests/waterbirds_crp.json --task 0 --locked-test
bash scripts/submit_experiment.sh experiments/manifests/waterbirds_crp.json --locked-test
```

Locked-test runs default to the separate `locked-test` attempt ID; ordinary
runs use `primary`. Command identity checks prevent accidental reuse of one as
the other.

Existing execution directories are protected by default. Choose
`--existing reuse`, `resume`, or `new-attempt` explicitly; reuse/resume require
the recorded command to match. Use `--output-root PATH` or set
`SPUR_SPLICE_OUTPUT_ROOT` to inspect a packaged artifact layout. This is
independent of `SPUR_SPLICE_SCRATCH_ROOT`, which stores large binaries.

Build shared inputs directly:

```bash
python -m scripts.tools.cache_crp_features --help
python -m splice.crp --help
python -m scripts.tools.build_crp_baseline_graphs --help
python -m scripts.tools.build_paper_results --artifact-root /path/to/artifact-tree
```

## Verification

```bash
conda activate grgrie-train
pip install -e .
python -m unittest discover -s tests -p 'test_*.py'
```

The checked root `paper_results.json` is regenerated from the selected artifact
tree. Its historical source aggregate is `reports/paper/test_summary.json`
inside the archived pre-unification artifact package.

For this study, DONE means: a predeclared split/protocol, unique run identity,
linked endpoint metrics, completed and converged status, and explicit limits.
It does not mean producing a new standalone report for every implementation
detail or rerunning training when no scientific claim would change.

The final aggregate for a study is `outputs/reports/<study>/results.json`.
Run records are under `outputs/seeds/<study>/seed_XX/<arm>/<attempt_id>/`;
SSL checkpoints and large probe feature tensors are retained under
`/scratch/xar68reb/CoSpRo` and
attested by size and SHA-256 in each `run.json`.
