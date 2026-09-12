# Scripts

For one training run with parameters chosen at submission time, use the general
launcher. All options after the script name are passed to `spur_splice.py`;
its CLI uses underscores in option names. Run `python3 spur_splice.py --help`
for the full list.

```bash
sbatch scripts/run_training.sbatch \
  --dataset waterbirds --seed 7 --epochs 100 --batch_size 128 \
  --learning_rate 0.003 --model resnet18_large \
  --data_folder /scratch/my-datasets --checkpoint_dir /scratch/my-run \
  --keep_checkpoints

# Continue from a particular checkpoint file. Set --checkpoint_dir as well
# when the continued run should write into a chosen directory.
sbatch scripts/run_training.sbatch \
  --dataset waterbirds --seed 7 --epochs 150 \
  --checkpoint_dir /scratch/my-run \
  --resume "$(find /scratch/my-run -name last.pth -print -quit)" \
  --keep_checkpoints
```

The launcher chooses a unique scratch directory, run record and attempt ID from
the Slurm job ID when none are supplied. The run record stays at
`$SPUR_SPLICE_SCRATCH_ROOT/manual/<job-id>/run.json` even if a checkpoint root
is supplied. `--checkpoint_dir` is the *root* of the training artifacts; the
trainer adds a configuration-named subdirectory, so use the actual `last.pth`
path saved there for `--resume`.
`--keep_checkpoints` must be set to retain checkpoints. Standalone runs are
recorded separately from manifest matrices and are not collected by the
matrix result collector.

For the predeclared seed/arm matrix below, keep the manifest as the source of
truth. A single matrix cell can also be submitted directly:

```bash
sbatch scripts/run_experiment.sbatch experiments/manifests/waterbirds_cospro.json \
  --seed 3 --arm cospro --existing error
```

Its hyperparameters and dataset still come from the manifest. To change them
for a *matrix* experiment, create a new manifest and use
`submit_experiment.sh` so training and collection read the same configuration.
`run_la_ssl.sh` is the fixed Waterbirds LA-SSL protocol (seed 1 by default;
pass `--seed N` to select another declared seed). `run_projection_controls.sh`
only completes the two historical Waterbirds controls, with verified inputs.
`run_cospro_pipeline.sh` runs preprocessing, graph construction, one CoSpRo
student, and collection; it accepts `--dataset`, `--seed`, `--epochs`,
`--batch-size`, `--output-root`, `--student-existing resume`, and the other
options in `python3 -m scripts.tools.run_cospro_pipeline --help` directly after
its script name. For `spur_cifar10`, both the standalone trainer and pipeline
default to `resnet18`; larger-image datasets retain `resnet18_large`.

Submit an experiment matrix and its dependent result collector with:

```bash
bash scripts/submit_experiment.sh experiments/manifests/waterbirds_cospro.json
```

Append `--locked-test` to apply the manifest's predeclared final-only held-out
test protocol. Locked-test executions use a separate `locked-test` attempt ID.

`run_experiment.sbatch` activates the cluster environment and calls
`experiments.runner`; `collect_results.sbatch` runs with an `afterany`
dependency. Slurm output is kept in `outputs/SLURM/`, and cluster setup verifies
`/scratch/xar68reb/CoSpRo` before training starts.

Inspect and migrate an existing cluster `outputs` tree through Slurm with:

```bash
bash scripts/submit_outputs_migration.sh --scan
bash scripts/submit_outputs_migration.sh --dry-run
bash scripts/submit_outputs_migration.sh --apply
```

`--apply` prints the complete dry-run plan first, then migrates files, verifies
sizes and SHA-256 hashes, and removes source copies only after successful
verification. The default source is `<project>/outputs`; an explicit legacy root
can be passed as the second argument. Discovery handles both direct and nested
layouts and stops if more than one plausible results root exists.

After migration, unknown historical files remain quarantined under
`outputs/shared/legacy/`. Inventory them without changing files:

```bash
bash scripts/submit_legacy_archive.sh --scan
```

Create and verify a compressed archive through Slurm:

```bash
bash scripts/submit_legacy_archive.sh --archive pre-unification
```

Only after the archive and its Git-friendly manifest have been reviewed, run:

```bash
bash scripts/submit_legacy_archive.sh --delete-after-verify pre-unification
```

The existing archive is fully reverified against the source before deletion;
it is never overwritten. Job output is written to `outputs/SLURM/`.

Small Python tools remain for operations that are not training matrices:

- `cache_splice_dataset.py` — build the frozen train-split SpLiCE dataset cache
  in a directory named from all cache-affecting hyperparameters;
- `generate_cospro_concept_groups.py` — generate reusable CoSpRo grouping JSON + HTML
  artifacts from the cache, including a threshold sweep with no audit or SSL;
- `build_cospro_teacher_graphs.py` — audit one saved CoSpRo grouping artifact or every
  `concept_groups.json` below a sweep directory, writing colocated graph JSON + HTML;
- `build_cospro_baseline_graphs.py` — build a raw-CLIP graph with anchor support,
  row degrees, weight profiles, confidence, and indegree cap matched to a
  canonical CoSpRo reference graph;
- `download_waterbirds_hf.py` — dataset helper;
- `summarize_cospro_audit.py` — concise CoSpRo graph summary;
- `render_report.py` — the single HTML report adapter.
- `collect_results.py` — validate a manifest matrix and build one results JSON;
- `promote_checkpoint.py` — explicitly retain a checkpoint with rationale;
- `migrate_outputs.py` — discover and safely migrate direct or nested legacy trees.
- `archive_legacy.py` — inventory and verify a recoverable archive of quarantined legacy files.
- `run_cospro_pipeline.py` — sequential, validated CoSpRo cache-to-results orchestration.

Use `python -m <module> --help` for their interfaces.

The shell entry points use the same interfaces directly or through `sbatch`:

```bash
# Complete CelebA cache -> groups -> graph -> student -> results pipeline.
# Its configuration block can be edited or overridden with environment variables.
sbatch scripts/run_cospro_pipeline.sh
bash scripts/run_cospro_pipeline.sh --dry-run

# Build a frozen SpLiCE dataset cache. The dataset argument is required;
# the output root is optional and uses the configured scratch feature root under Slurm.
bash scripts/cache_splice_dataset.sh waterbirds /scratch/path/features/Spur_SpLiCE

# The same options work under sbatch; later CLI values override defaults.
sbatch scripts/cache_splice_dataset.sh waterbirds \
  --data-folder /scratch/my-datasets --splice-l1-penalty 0.1 --batch-size 32

# The first cache command writes the following cache; pass this path to stage 2.
# /scratch/path/features/Spur_SpLiCE/waterbirds/splice_dataset_cache/
#   cache_v1__model_open_clip_ViT-B-32__pretrained_laion2b_s34b_b79k__vocab_openimages_v7_all__l1_0p25/
#     splice_dataset_cache.pt

# Other datasets use separate celeba/ and spur_cifar10/ subdirectories
# beneath the same output root. CelebA spellings are normalized to celeba.
bash scripts/cache_splice_dataset.sh celebA /scratch/path/features/Spur_SpLiCE
sbatch scripts/cache_splice_dataset.sh spur_cifar10 /scratch/path/features/Spur_SpLiCE

# Default 5 x 6 grouping grid. --output-root is optional.
bash scripts/generate_cospro_concept_groups.sh /scratch/path/splice_dataset_cache.pt \
  --output-root outputs/shared/waterbirds/graphs/concept_groups \
  --data-folder /path/to/datasets

# Reproduce the canonical grouping parameters only.
bash scripts/generate_cospro_concept_groups.sh /scratch/path/splice_dataset_cache.pt \
  --output-root outputs/shared/waterbirds/graphs/concept_groups_canonical \
  --text-similarity-threshold 0.82 --coactivation-threshold 0.35

# The second argument may be one JSON file or the whole sweep directory.
bash scripts/build_cospro_teacher_graphs.sh /scratch/path/splice_dataset_cache.pt \
  outputs/shared/waterbirds/graphs/concept_groups
```

Execution-only cache settings such as batch size, worker count, and device are
not part of its directory name. Cache-affecting settings are: model, pretrained
weights, vocabulary, vocabulary size, decomposition penalty, and cache schema
version.

Each grouping configuration is stored as
`<grouping-config>/concept_groups.{json,html}`. Teacher audits are placed below
that directory as `teacher_graphs/<audit-config>/teacher_graph.{json,html}`.
Grouping artifacts are the sole source of grouping thresholds for graph
construction; teacher-audit options cannot override them.
The grouping HTML embeds compact representative thumbnails when `--data-folder`
is supplied (or `DATA_FOLDER` is set); grouping itself remains cache-only.
Missing dataset access is an error unless `--no-embed-images` is explicitly
selected. An existing report can be repaired without regrouping via
`--render-existing /path/to/concept_groups.json --data-folder /path/to/datasets`.

Git-facing artifact paths default to `outputs/`. Set `SPUR_SPLICE_OUTPUT_ROOT`
or pass `experiments.runner --output-root PATH` when inspecting a packaged tree
such as `outputs/output_cluster`; the runner substitutes `{artifacts}` in
manifests. `SPUR_SPLICE_SCRATCH_ROOT` separately controls large binary storage.
