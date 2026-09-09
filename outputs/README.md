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

Binary `.pth`, `.pt` and `.ckpt` payloads larger than 10 MiB are written under
`/scratch/xar68reb/CoSpRo/{checkpoints,features}/Spur_SpLiCE/`. Smaller binaries
may remain beside a run but are ignored by Git. Slurm output is kept in
`outputs/SLURM/` and is also ignored.

Set `SPUR_SPLICE_SCRATCH_ROOT` to override the binary location. The legacy
cluster variable `SPUR_SPLICE_ARTIFACT_ROOT` is accepted as a fallback, but it
never changes the Git-facing `outputs/` root.

Git admits files only from the canonical `seeds/`, `reports/`, `shared/` and
`reference/` trees. Flat legacy directories are intentionally ignored until
the migration tool classifies them, preventing an accidental bulk commit of a
copied historical output tree.

Run the collector manually when needed:

```bash
python -m scripts.tools.collect_results experiments/manifests/waterbirds_crp.json
```

Inspect a legacy tree before migration:

```bash
python -m scripts.tools.migrate_outputs /path/to/copied/outputs
```
