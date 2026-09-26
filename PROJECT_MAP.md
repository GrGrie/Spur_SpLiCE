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
| Sparse decomposition | `third_party/splice/model.py`, `third_party/splice/splice.py`, `third_party/splice/admm.py` |
| Artifact locations and threshold routing | `cospro/tracking/artifacts.py` |
| Run lifecycle records | `cospro/tracking/run_recording.py` |
| Frozen SpLiCE dataset cache | `cospro/cli/cache_splice_dataset.py`, `cospro/pipeline/cache.py` |
| Concept dictionaries (bundled and file) | `cospro/pipeline/dictionary.py`, `data/vocab/` |
| Dataset download and verification | `cospro/cli/download_datasets.py`, `scripts/download_datasets.sbatch` |
| Concept grouping | `cospro/pipeline/grouping.py`, `cospro/cli/generate_cospro_concept_groups.py` |
| Neighbour search, audit, selection, teacher graph | `cospro/pipeline/`, `cospro/pipeline/graph_io.py` |
| Complete cache-to-results pipeline | `scripts/run_cospro_pipeline.sh`, `cospro/cli/run_cospro_pipeline.py` |
| Graph sampler and relational KL | `cospro/methods/relational_graph.py` |
| Concept factors, F1 conditioned batches, F2 factor distillation | `cospro/pipeline/concept_factors.py`, `cospro/methods/concept_factors.py`, `cospro/cli/inspect_concept_factors.py` |
| LateTVG pruned second view (combines with any method) | `cospro/training/late_pruning.py`, `experiments/manifests/waterbirds_latetvg.yaml` |
| Typed training configuration, presets, sweeps | `cospro/config/` |
| Training methods behind one interface | `cospro/methods/` |
| SSL training: epoch loop, callbacks, storage policy | `spur_splice.py`, `cospro/training/` |
| Stable metric names for W&B | `cospro/tracking/` |
| Dataset adapters, loader roles, registry | `cospro/data/` (see [`docs/ADDING_A_DATASET.md`](docs/ADDING_A_DATASET.md)) |
| Linear evaluation | `cospro/evaluation/probe.py`, `cospro/cli/linear_probe.py` (CLI) |
| Experiment matrix and execution identity | `experiments/runner.py`, `experiments/manifests/` |
| Checked paper registry | `tools/paper/build_paper_results.py`, `paper/paper_results.json` |
| HTML reports | `cospro/pipeline/html_report.py`, `cospro/cli/render_report.py` |
| Grouping and graph diagnostics, dashboard | `cospro/diagnostics/`, `scripts/run_cospro_diagnostics.sbatch` |

The teacher-input flow has three explicit module interfaces: the SpLiCE dataset
cache serializes frozen train-split representations, concept grouping consumes
that cache and serializes groups, and `build_teacher_graph(splice_dataset_cache,
concept_groups, config)` consumes both artifacts for projection geometry, null
controls and sparse graph assembly. Training consumes only validated graphs
through `cospro.methods.relational_graph`.

Code, commands, reports and documentation use CoSpRo names. `cospro/compat.py` is the one
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
provenance. New code must obtain destinations through `cospro.tracking.artifacts`.
The Git-facing root defaults to `outputs/`; `SPUR_SPLICE_OUTPUT_ROOT` or the
runner's `--output-root` selects a packaged tree without rewriting manifests.
Migration normalizes copied paths, and a new run becomes complete only after
`cospro.tracking.run_recording` verifies every retained artifact attestation.
