# Paper evidence

This directory contains compact evidence recovered from
`/scratch/xar68reb/CoSpRo/legacy_archives/Spur_SpLiCE/`. It is intentionally
limited to inputs needed by `scripts.tools.build_paper_results` that were not
already represented by canonical `run.json` records or legacy summaries.

- `final_test/probes/` contains one converged epoch-500 held-out probe record
  for every seed and arm in the final 4-by-5 matrix.
- `visual/concept_panels.json` freezes the label-free panel selection. The
  generated PDF, PNG, and source images remain outside Git.

Reusable teacher/control graphs and compact direct-transfer target metadata are
stored under `outputs/shared/waterbirds/`. The corresponding 29,625,858-byte
`targets_v1.pt` remains in scratch as required by the binary-artifact policy;
its SHA-256 is
`fde777c311169e14ac2381a418dd2d68434a0d5e923f7f2a77edc59501273123`.

No checkpoint, feature tensor, W&B runtime directory, Slurm log, or generated
visual binary was copied into Git.
