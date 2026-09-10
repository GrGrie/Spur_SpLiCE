# Experiment outputs

This directory contains structured results intended for Git. Runtime binaries
and logs follow separate retention rules.

```text
outputs/
  seeds/<study>/seed_XX/<arm>/<attempt_id>/run.json
  reports/<study>/results.json
  shared/<dataset>/
  reference/
  SLURM/
```

`run.json` is the local source of truth for configuration, metric history,
group results, runtime provenance, failures, W&B identity and artifact
attestations. `results.json` combines the expected seed/arm matrix and remains
`partial` until every expected run is complete and its retained artifacts pass
size and SHA-256 verification.

The versioned contract is [`schemas/run-record-v1.schema.json`](../schemas/run-record-v1.schema.json).

`reports/paper_evidence/` contains compact inputs needed to rebuild the checked
root `paper_results.json`, including the 20 held-out probe records and the
frozen qualitative-panel selection. Claim-relevant reusable controls belong in
`shared/waterbirds/`: the intervention, raw-CLIP, and semantic-control graphs,
plus compact target metadata. Large feature and direct-transfer target tensors
remain in scratch and are identified by their retained metadata and hashes.

Canonical checkpoints are written under
`/scratch/xar68reb/CoSpRo/checkpoints/Spur_SpLiCE/` regardless of size. Feature
payloads larger than 10 MiB go under the sibling `features/` tree. Small local
feature tensors may remain beside a run but are ignored by Git. Slurm output is
kept in `outputs/SLURM/` and is also ignored.

Set `SPUR_SPLICE_SCRATCH_ROOT` to override the binary location. The legacy
cluster variable `SPUR_SPLICE_ARTIFACT_ROOT` is accepted as a fallback, but it
never changes the Git-facing `outputs/` root.

Git admits files only from the canonical `seeds/`, `reports/`, `shared/` and
`reference/` trees. Flat legacy directories are intentionally ignored until
the migration tool classifies them, preventing an accidental bulk commit of a
copied historical output tree.

`outputs/shared/legacy/` is a quarantine area and is also ignored. Use
`python -m scripts.tools.archive_legacy` to inventory it, then archive it to
scratch. The verified archive manifest is written under
`outputs/reports/legacy-archive/` and is suitable for Git.

On Windows, `python scripts/tools/cleanup_local_outputs.py` performs a dry-run
classification of quarantined tensors. Add `--apply` only after review: the
tool writes `outputs/reports/local-cleanup/windows-pre-unification.json`,
promotes legacy CSV results into that JSON, removes result-backed checkpoints
and reproducible caches, and retains ambiguous tensors.

Recovered pre-unification executions follow the same attempt layout. Their
`run.json` files embed periodic probe metrics so another machine can inspect
the training history without the scratch-only tensors. See
`docs/history/LEGACY_RECOVERY-2026-09-09.md` for the recovery boundary and the
optional full archive-manifest command.

Run the collector manually when needed:

```bash
python -m scripts.tools.collect_results experiments/manifests/waterbirds_crp.json
```

Inspect a legacy tree before migration:

```bash
python -m scripts.tools.migrate_outputs /path/to/copied/outputs
```
