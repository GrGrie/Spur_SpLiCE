# Spur SpLiCE

CoSpRo (Concept-Guided Relational Pretraining for Spurious-Correlation
Robustness) tests whether sparse semantic concepts can improve visual
representations under spurious object-context correlations. The method uses
frozen OpenCLIP/SpLiCE features to produce a sparse teacher
graph, then a SimCLR ResNet trains with graph-aware batches and relational KL.

The graph and SSL stages use no target or group annotations. Labels and group
metadata appear only in linear evaluation. The teacher is unnecessary at
student inference.

## Project shape

- `splice/` — sparse decomposition, CoSpRo graph construction and graph training.
- `experiments/spurious_eval/` — datasets, models, training and linear probes.
- `experiments/runner.py` — the single seed/arm experiment runner.
- `experiments/manifests/` — reproducible experiment definitions.
- `scripts/` — one Slurm adapter plus small cache/report tools.
- `outputs/` — Git-friendly run records, shared JSON artifacts, aggregate reports and ignored Slurm logs.
- `tests/` — tests for the current method only.

See [PROJECT_MAP.md](PROJECT_MAP.md) for the data flow,
[`docs/REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md) for the canonical storage
policy, and `outputs/README.md` for result navigation.

For the seed-2/4 semantic/direct-transfer completion launcher and the matched
LA-SSL baseline, see [docs/SUBMISSION_RUNS.md](docs/SUBMISSION_RUNS.md).

## Run the canonical experiment

On the cluster, set `DATA_FOLDER` if it differs from the default. The submission
helper launches the matrix and a dependent result collector:

```bash
bash scripts/submit_experiment.sh experiments/manifests/waterbirds_cospro.yaml
```

Inspect the matrix without training:

```bash
python -m experiments.runner experiments/manifests/waterbirds_cospro.yaml --list
python -m experiments.runner experiments/manifests/waterbirds_cospro.yaml --task 0 --dry-run
```

The held-out test protocol is declared in the manifest and must be requested
explicitly. It runs one final probe on `test` instead of periodic validation
probes:

```bash
python -m experiments.runner experiments/manifests/waterbirds_cospro.yaml --task 0 --locked-test
bash scripts/submit_experiment.sh experiments/manifests/waterbirds_cospro.yaml --locked-test
```

Locked-test runs default to the separate `locked-test` attempt ID; ordinary
runs use `primary`. Command identity checks prevent accidental reuse of one as
the other.

Existing execution directories are protected by default. Choose
`--existing reuse`, `resume`, or `new-attempt` explicitly; reuse/resume require
the recorded command to match. Use `--output-root PATH` or set
`SPUR_SPLICE_OUTPUT_ROOT` to inspect a packaged artifact layout. This is
independent of `SPUR_SPLICE_SCRATCH_ROOT`, which stores large binaries.

Build the three teacher-input stages directly:

```bash
python -m scripts.tools.cache_splice_dataset --help
python -m scripts.tools.generate_cospro_concept_groups --help
python -m scripts.tools.build_cospro_teacher_graphs --help
python -m scripts.tools.build_cospro_baseline_graphs --help
python -m scripts.tools.build_paper_results --artifact-root /path/to/artifact-tree
```

Teacher-graph construction now selects deterministic cosine-LSH search for
datasets with at least 20,000 samples and runs projection/search on CUDA when
available. Small datasets retain the exact implementation. Each completed
group and the raw-neighbour index are checkpointed atomically beside the graph,
so resubmitting the same command resumes rather than restarting the audit. Use
`--neighbor-backend exact`, `--device cpu`, or the ANN tuning options shown by
`--help` when an explicit override is required.

## Run one custom training job

For an individual training job with a custom dataset, seed, hyperparameters and
checkpoint location, use:

```bash
sbatch scripts/run_training.sbatch --dataset waterbirds --seed 7 \
  --epochs 100 --checkpoint_dir /scratch/my-run --keep_checkpoints
```

If exactly one completed teacher graph exists below
`outputs/shared/<dataset>/graphs`, this launcher automatically selects
`cospro_relational` training and passes that graph. If several graphs exist,
select one explicitly with `--cospro_teacher_graph PATH`; if none exists, the
launcher warns and runs standard SimCLR.

See [scripts/README.md](scripts/README.md) for resume commands and the
differences between standalone runs and manifest matrices.

## Run the complete CelebA pipeline

`scripts/run_cospro_pipeline.sh` defaults to CelebA and runs all five stages in
order: frozen cache, one concept-group configuration, its teacher graph,
student training, and final result collection. On Slurm, launch it once from
the repository root:

```bash
sbatch scripts/run_cospro_pipeline.sh
```

`DATA_FOLDER` must resolve to either the CelebA directory itself or its parent;
the adapter expects `list_attr_celeba.csv`, `list_eval_partition.csv`, and the
`img_align_celeba/` image directory.

For a local command or a cheap configuration check:

```bash
bash scripts/run_cospro_pipeline.sh --dry-run
EPOCHS=10 USE_WANDB=0 bash scripts/run_cospro_pipeline.sh
```

Every default lives in the Python CLI (`python -m scripts.tools.run_cospro_pipeline --help`).
The launcher lists the environment variables it forwards, such as `EPOCHS`,
`TEXT_SIMILARITY_THRESHOLD` or `USE_WANDB`; unset variables keep the Python default
and trailing options pass through unchanged. Set `DATASET=waterbirds` (or
`spur_cifar10`) to use the same pipeline for another registered dataset.
Completed preprocessing artifacts are validated and reused; set
`REBUILD_PREPROCESSING=1` to rebuild them. Student output is protected by
`STUDENT_EXISTING=error` by default; use `resume` only after an interrupted run
that has a retained checkpoint.

Concept reports embed thumbnails directly in the HTML and now fail if dataset
access is unavailable instead of silently emitting placeholders. Repair an
already generated report without recomputing groups with:

```bash
python -m scripts.tools.generate_cospro_concept_groups \
  --render-existing outputs/shared/<dataset>/graphs/concept_groups/<config>/concept_groups.json \
  --data-folder /path/to/datasets
```

Use `--no-embed-images` only when a placeholder-only report is intentional.

## Verification

```bash
conda activate grgrie-train
pip install -e ".[dev]"
python -m pytest
```

The golden tests in `tests/test_golden_*.py` freeze runner commands, teacher-graph
structure and two-epoch training runs for every training mode on synthetic data.
A behaviour change appears as a snapshot diff. After an intended change, regenerate
the snapshots with `SPUR_SPLICE_UPDATE_GOLDEN=1 python -m pytest tests/test_golden_*.py`
and review the diff under `tests/golden/`. On the cluster the same suite runs on a
V100 through `sbatch scripts/run_golden_smoke.sbatch`.

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
