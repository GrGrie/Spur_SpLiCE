# Scripts

Submit an experiment matrix and its dependent result collector with:

```bash
bash scripts/submit_experiment.sh experiments/manifests/waterbirds_crp.json
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
- `generate_crp_concept_groups.py` — generate reusable grouping JSON + HTML
  artifacts from the cache, including a threshold sweep with no audit or SSL;
- `build_crp_teacher_graphs.py` — audit one saved grouping artifact or every
  `concept_groups.json` below a sweep directory, writing colocated graph JSON + HTML;
- `build_crp_baseline_graphs.py` — build a raw-CLIP graph with anchor support,
  row degrees, weight profiles, confidence, and indegree cap matched to a
  canonical CRP reference graph;
- `download_waterbirds_hf.py` — dataset helper;
- `summarize_crp_audit.py` — concise graph summary;
- `render_report.py` — the single HTML report adapter.
- `collect_results.py` — validate a manifest matrix and build one results JSON;
- `promote_checkpoint.py` — explicitly retain a checkpoint with rationale;
- `migrate_outputs.py` — discover and safely migrate direct or nested legacy trees.
- `archive_legacy.py` — inventory and verify a recoverable archive of quarantined legacy files.

Use `python -m <module> --help` for their interfaces.

The shell entry points use the same interfaces directly or through `sbatch`:

```bash
# Build the canonical frozen Waterbirds SpLiCE dataset cache. The output root
# is optional; under Slurm it defaults to the configured scratch feature root.
bash scripts/cache_splice_dataset.sh /scratch/path/features/Spur_SpLiCE

# The command above writes the following cache; pass this path to stage 2.
# /scratch/path/features/Spur_SpLiCE/waterbirds/splice_dataset_cache/
#   cache_v1__model_open_clip_ViT-B-32__pretrained_laion2b_s34b_b79k__vocab_openimages_v7_all__l1_0p25/
#     splice_dataset_cache.pt

# Default 5 x 6 grouping grid. --output-root is optional.
bash scripts/generate_crp_concept_groups.sh /scratch/path/splice_dataset_cache.pt \
  --output-root outputs/shared/waterbirds/graphs/concept_groups

# Reproduce the canonical grouping parameters only.
bash scripts/generate_crp_concept_groups.sh /scratch/path/splice_dataset_cache.pt \
  --output-root outputs/shared/waterbirds/graphs/concept_groups_canonical \
  --text-similarity-threshold 0.82 --coactivation-threshold 0.35

# The second argument may be one JSON file or the whole sweep directory.
bash scripts/build_crp_teacher_graphs.sh /scratch/path/splice_dataset_cache.pt \
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

Git-facing artifact paths default to `outputs/`. Set `SPUR_SPLICE_OUTPUT_ROOT`
or pass `experiments.runner --output-root PATH` when inspecting a packaged tree
such as `outputs/output_cluster`; the runner substitutes `{artifacts}` in
manifests. `SPUR_SPLICE_SCRATCH_ROOT` separately controls large binary storage.
