# Legacy result recovery status — 2026-09-09

The extracted cluster archive was audited after its original JSON manifest was
mistakenly deleted. Seventy-three completed training executions were recovered
into canonical `outputs/seeds/` records, including 1,400 associated periodic
probe metric events, final metrics, configuration, W&B identity, cleanup state, and
attestations for retained final checkpoints and final probe features.

Four standalone downstream-evaluator probe results had no training
`run_status.json`; their diagnostic report is retained with the compact legacy
analyses rather than inventing execution identities.

Twelve observed-study aggregates are stored under `outputs/reports/`. Compact
non-training summaries and audits are stored under
`outputs/reports/legacy_analyses/`. The checked `paper_results.json` remains the
primary compact registry for paper claims.

The extracted 4.18 GB tensor payload and the 2.8 GB tarball remain cluster-only
under `/scratch/xar68reb/CoSpRo/legacy_archives/Spur_SpLiCE/`. Rebuilding the
full per-file archive manifest is optional because it requires hashing the
entire payload and tarball. It can be performed later with:

```bash
python -m scripts.tools.promote_legacy_results \
  /scratch/xar68reb/CoSpRo/legacy_archives/Spur_SpLiCE \
  --rebuild-archive-manifest
```

The compact Git records are sufficient for comparing completed metrics on
another machine without downloading checkpoints or W&B runtime data.
