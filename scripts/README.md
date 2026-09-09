# Scripts

Submit an experiment matrix and its dependent result collector with:

```bash
bash scripts/submit_experiment.sh experiments/manifests/waterbirds_crp.json
```

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

- `cache_crp_features.py` — build the frozen CRP cache;
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

Git-facing artifact paths default to `outputs/`. Set `SPUR_SPLICE_OUTPUT_ROOT`
or pass `experiments.runner --output-root PATH` when inspecting a packaged tree
such as `outputs/output_cluster`; the runner substitutes `{artifacts}` in
manifests. `SPUR_SPLICE_SCRATCH_ROOT` separately controls large binary storage.
