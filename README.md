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

- `cospro/` — the project's code: `config/` (typed options, presets, cluster settings), `data/`
  (dataset adapters and loader roles), `models/`, `methods/` (SimCLR, CoSpRo relational,
  concept transfer, LA-SSL), `training/` (trainer, callbacks, storage policy), `evaluation/`
  (linear probe), `pipeline/` (SpLiCE cache, concept groups, audit, teacher graph),
  `diagnostics/`, `tracking/` (W&B keys, run records, artifact placement) and `cli/` (every
  command-line stage).
- `spur_splice.py` — the SSL training entry point: parse, build, fit.
- `third_party/` — vendored SpLiCE and the WILDS slice, with their licenses and a NOTICE of local
  changes.
- `experiments/runner.py` — the single seed/arm experiment runner.
- `experiments/manifests/` — reproducible experiment definitions.
- `scripts/` — Slurm launchers.
- `splice/`, `experiments/spurious_eval/`, `scripts/tools/` — shims that keep the historical import
  paths and `python -m` commands working; new code imports from `cospro` and `third_party`.
- `tools/maintenance/` — output migration, legacy archiving, cleanup and W&B export.
- `tools/paper/` — paper registry, submission figures and checkpoint evaluation.
- `paper/` — the manuscript and its checked `paper_results.json`.
- `outputs/` — Git-friendly run records, shared JSON artifacts, aggregate reports and ignored Slurm logs.
- `tests/` — tests for the current method only.

See [PROJECT_MAP.md](PROJECT_MAP.md) for the data flow,
[`docs/REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md) for the canonical storage
policy and `outputs/README.md` for result navigation.

For the seed-2/4 semantic/direct-transfer completion launcher and the matched
LA-SSL baseline, see [docs/SUBMISSION_RUNS.md](docs/SUBMISSION_RUNS.md).

## Get the data

Everything except the images ships with the repository or downloads on first use: the LAION and
Open Images V7 concept vocabularies and the CLIP image mean are in `data/`, the OpenCLIP ViT-B-32
weights come from Hugging Face and Spur-CIFAR10 downloads through torchvision. Waterbirds and
CelebA come from their official sources through one command, which checks every file against the
data the paper was computed from:

```bash
python -m cospro.cli.download_datasets --data-folder ~/Datasets
```

Waterbirds is the Group DRO archive from Stanford; its archive and `metadata.csv` are checked by
SHA-256. CelebA is the official aligned release; every file is checked by the MD5 torchvision ships
and the annotations are rewritten into the CSV files the adapter reads, whose SHA-256 must equal
the paper's. The official CelebA files sit on Google Drive, which needs the `gdown` package
(`pip install -e .[data]`) and rate-limits large downloads. When it refuses, download
`img_align_celeba.zip`, `list_attr_celeba.txt` and `list_eval_partition.txt` from the CelebA site
into one directory and add `--celeba-archive-dir DIR`. On the cluster,
`sbatch scripts/download_datasets.sbatch` runs the same command for `DATA_FOLDER`.

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

### Presets and sweeps

Training options are declared once in `cospro/config/training.py`. Named presets in
`cospro/config/presets.py` bundle recurring values; `cospro_student` is the CoSpRo
student of the paper manifests:

```bash
python spur_splice.py --preset cospro_student --cospro_teacher_graph PATH --temp 0.1
```

Explicit options override the preset. A manifest can expand a grid of option values into
arms with a `sweeps` block; each grid point becomes an arm named
`prefix__option-value__option-value` that inherits the base arm:

```yaml
sweeps:
  weight:
    base: cospro
    grid:
      splice_weight: [0.25, 0.5, 1.0]
      cospro_temperature: [0.1, 0.25]
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
python -m cospro.cli.cache_splice_dataset --help
python -m cospro.cli.generate_cospro_concept_groups --help
python -m cospro.cli.build_cospro_teacher_graphs --help
python -m cospro.cli.build_cospro_baseline_graphs --help
python -m tools.paper.build_paper_results --artifact-root /path/to/artifact-tree
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

## Diagnose concept groups and teacher graphs

`cospro/diagnostics` measures how well a grouping and a teacher graph do their job and renders a
dashboard: sweep heatmaps over the grouping thresholds, a graph comparison with baselines,
(y, a) transition matrices, confidence calibration, per-group AUC selectivity and an edge gallery.

```bash
sbatch scripts/run_cospro_diagnostics.sbatch
python -m cospro.diagnostics dashboard outputs/reports/cospro_diagnostics/waterbirds/diagnostics.json \
  --data-folder /path/to/datasets --gallery-graph cospro --output diagnostics.html
```

The cluster job reads the SpLiCE cache from scratch and writes the compact JSON record, which Git
synchronizes. Rendering runs on any machine; the gallery needs a local copy of the dataset.
Label-based metrics are post-hoc: they guide manual selection and stay out of automatic tuning.

## Run the complete CelebA pipeline

`scripts/run_cospro_pipeline.sh` defaults to CelebA and runs all five stages in
order: frozen cache, one concept-group configuration, its teacher graph,
student training and final result collection. On Slurm, launch it once from
the repository root:

```bash
sbatch scripts/run_cospro_pipeline.sh
```

`DATA_FOLDER` must resolve to either the CelebA directory itself or its parent;
the adapter expects `list_attr_celeba.csv`, `list_eval_partition.csv` and the
`img_align_celeba/` image directory.

For a local command or a cheap configuration check:

```bash
bash scripts/run_cospro_pipeline.sh --dry-run
EPOCHS=10 USE_WANDB=0 bash scripts/run_cospro_pipeline.sh
```

Every default lives in the Python CLI (`python -m cospro.cli.run_cospro_pipeline --help`).
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
python -m cospro.cli.generate_cospro_concept_groups \
  --render-existing outputs/shared/<dataset>/graphs/concept_groups/<config>/concept_groups.json \
  --data-folder /path/to/datasets
```

Use `--no-embed-images` only when a placeholder-only report is intentional.

### Concept dictionaries

The SpLiCE cache decomposes every image into the words of one concept dictionary. The default is
Open Images V7; `SPLICE_VOCAB=laion` selects the SpLiCE LAION vocabulary. For an ablation, any text
file with one concept per line works:

```bash
SPLICE_VOCAB=file SPLICE_VOCAB_FILE=/path/to/words.txt SPLICE_VOCAB_SIZE=5000 \
  sbatch scripts/run_cospro_pipeline.sh
```

Blank lines are skipped and a repeated concept is an error; every other line is a concept.
`SPLICE_VOCAB_ORDER=head` (default) or `tail` says which end a size keeps. A file dictionary is
identified by the SHA-256 of its words, which names its cache directory and its embedding cache, so
an edited file never reuses stale embeddings. See `cospro/pipeline/dictionary.py`.

## Training methods and tracked metrics

Each training method lives in `cospro/methods/`: `SimCLROnly`, `CoSpRoRelational`,
`FrozenConceptDistill` and `LaSSL`. A method owns the loader it needs, the loss term it adds to
SimCLR, the diagnostics it reports, the artifacts it consumes and the sampler state it saves.
Adding a method means adding one module with `@register_method`, declaring the `splice_mode`
values it serves and adding its options section in `cospro/config`.

Every run logs two sets of W&B keys: the historical sentence-style names that `run.json`,
`collect_results` and the paper registry read, plus canonical names that stay stable across
refactors (`cospro/tracking/metrics.py`):

```text
probe/<split>/wga            worst-group accuracy, the run summary metric
probe/<split>/wga_avg10      averaged over the last ten probe epochs
probe/<split>/avg_acc        average accuracy
probe/<split>/group_acc/<g>  per-group accuracy
probe/spurious/wga           residual predictability of the spurious attribute
train/loss/<term>            SSL loss terms
method/<name>                diagnostics of the training method
```

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

The checked `paper/paper_results.json` is regenerated from the selected artifact
tree. Its historical source aggregate is `reports/paper/test_summary.json`
inside the archived pre-unification artifact package.

For this study, DONE means: a predeclared split/protocol, unique run identity,
linked endpoint metrics, completed and converged status and explicit limits.
It does not mean producing a new standalone report for every implementation
detail or rerunning training when no scientific claim would change.

The final aggregate for a study is `outputs/reports/<study>/results.json`.
Run records are under `outputs/seeds/<study>/seed_XX/<arm>/<attempt_id>/`;
SSL checkpoints and large probe feature tensors are retained under
`/scratch/xar68reb/CoSpRo` and
attested by size and SHA-256 in each `run.json`.
