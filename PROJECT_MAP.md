# Project map

Storage and synchronization rules are defined in
[`docs/REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md).

## Data flow

```text
training images
  -> frozen SpLiCE dataset cache
  -> concept groups
  -> CoSpRo intervention audit and graph construction
  -> validated sparse teacher graph
  -> graph-aware SimCLR training
  -> frozen encoder linear probe
  -> average and worst-group accuracy
```

## Modules

| Responsibility | Source |
|---|---|
| Sparse decomposition | `splice/model.py`, `splice/splice.py`, `splice/admm.py` |
| Artifact locations and threshold routing | `splice/artifacts.py` |
| Run lifecycle records | `splice/run_recording.py` |
| Frozen SpLiCE dataset cache | `scripts/tools/cache_splice_dataset.py` |
| Concept grouping | `splice/cospro.py`, `scripts/tools/generate_cospro_concept_groups.py` |
| CoSpRo audit and teacher graph | `splice/cospro.py`, `splice/graph_io.py` |
| Complete cache-to-results pipeline | `scripts/run_cospro_pipeline.sh`, `scripts/tools/run_cospro_pipeline.py` |
| Graph sampler and relational KL | `splice/cospro_training.py` |
| Typed training configuration, presets, sweeps | `cospro/config/` |
| Training methods behind one interface | `cospro/methods/` |
| SSL training: epoch loop, callbacks, storage policy | `spur_splice.py`, `cospro/training/`, `experiments/spurious_eval/training/ssl_loop.py` |
| Stable metric names for W&B | `cospro/tracking/` |
| Dataset adapters, loader roles, registry | `experiments/spurious_eval/datasets/` (see [`docs/ADDING_A_DATASET.md`](docs/ADDING_A_DATASET.md)) |
| Linear evaluation | `cospro/evaluation/probe.py`, `experiments/spurious_eval/linear_probe.py` (CLI) |
| Experiment matrix and execution identity | `experiments/runner.py`, `experiments/manifests/` |
| Checked paper registry | `tools/paper/build_paper_results.py`, `paper/paper_results.json` |
| HTML reports | `splice/reporting.py`, `scripts/tools/render_report.py` |
| Grouping and graph diagnostics, dashboard | `cospro/diagnostics/`, `scripts/run_cospro_diagnostics.sbatch` |

The teacher-input flow has three explicit module interfaces: the SpLiCE dataset
cache serializes frozen train-split representations, concept grouping consumes
that cache and serializes groups, and `build_teacher_graph(splice_dataset_cache,
concept_groups, config)` consumes both artifacts for projection geometry, null
controls and sparse graph assembly. Training consumes only validated graphs
through `splice.cospro_training`.

Code, commands, reports and documentation use CoSpRo names. `splice/compat.py` is the one
place that knows the CRP-era names: it keeps run storage names stable, reads the graph
fingerprint of older checkpoints and lists the historical artifact types. The `--crp_*`
options and the `crp_relational` mode stay accepted so the historical `waterbirds_crp`
study remains reproducible.

## Results

- `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/` contains Git-friendly run records.
- `outputs/reports/<study>/results.json` contains aggregate metrics and run provenance.
- `outputs/reports/legacy_analyses/` preserves compact pre-unification analyses.
- `outputs/reference/` contains external and pre-unification registry exports.
- `/scratch/xar68reb/CoSpRo/` contains checkpoints, tensor features, and archive payloads.

Historical source paths embedded in result JSON files are retained as
provenance. New code must obtain destinations through `splice.artifacts`.
The Git-facing root defaults to `outputs/`; `SPUR_SPLICE_OUTPUT_ROOT` or the
runner's `--output-root` selects a packaged tree without rewriting manifests.
Migration normalizes copied paths, and a new run becomes complete only after
`splice.run_recording` verifies every retained artifact attestation.
