# Cluster copy cleanup — 2026-09-09

Scope: only this cluster-copy repository directory.

## Removed

- Completed-run intermediate model checkpoints (`epoch_*.pth`) while preserving
  each run's `last.pth`.
- Pre-500 periodic probe feature caches while preserving epoch-500 features and
  all lightweight JSON metrics.
- The obsolete singular `output/` tree (pre-paper CRPv2/CRPv3 artifacts and old
  loose Slurm logs), superseded by named studies in `outputs/` and consolidated
  `paper_results.json`.
- The legacy `save/` tree, which contained only small, old `args.json` files and
  no model weights.
- All 372 synchronized local W&B run caches (2.23 GB), three directories of
  historical Slurm stdout/stderr, Python bytecode/cache directories, and
  `tea_debug.log`. No offline/unsynchronized W&B run was present.

## Preserved

- All final checkpoints, final probe feature caches, metrics, manifests, graphs,
  review bundles, figures, and the shared Waterbirds feature cache.
- Compact W&B exports under `outputs/reference/wandb_exports/`.
- Datasets, current source code and tests, documentation, and Git history.

See `outputs/README.md` for the result catalog and retention policy.

## Code and documentation cleanup

- Removed the obsolete root `linear_probe.py` wrapper; use
  `python -m experiments.spurious_eval.linear_probe`.
- Removed the unused Google Sheets export script and its shell wrapper.
- Removed six Windows-only PowerShell launchers from the Linux cluster copy.
- Removed generated logs from `scripts/output/`.
- Replaced two stale pre-completion plans and the old repository review with
  `docs/README.md` and this cleanup record.
- Replaced 69 study-specific Slurm launchers and 24 shell/JSON configurations
  with `experiments/runner.py`, one manifest and one Slurm adapter.
- Moved the legacy all-runs W&B JSON export to
  `outputs/reference/wandb_exports/`; future filtered exports are written there too.
- Removed the unused CoBalT/spatial and direct-transfer implementations, optional
  graph-search branch, historical report generators and their tests.
- Migrated all 2,477 output files (5,758,325,147 bytes) without loss into
  `outputs/seeds`, `outputs/shared`, `outputs/reports`, and `outputs/reference`.
- Added one artifact-path module and one shared HTML reporting module.
- Removed the unused SAM optimizer and the completed gradient-diagnostic branch.
- Removed no-op linear-probe compatibility flags and the single-choice
  `--method`/unused `--trial` training options.

## Verification

- Python byte-compilation passed for source, scripts, and tests.
- The linear-probe module entry point starts successfully in `grgrie-train`.
- After the architectural reduction, all 46 remaining `unittest` tests passed.
- The suite no longer depends on pytest.
- `git diff --check` passed. Bash syntax validation was unavailable because the
  local WSL Bash service is blocked by Windows permissions.
