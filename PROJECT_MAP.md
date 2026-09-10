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
| SSL training | `spur_splice.py`, `experiments/spurious_eval/training/ssl_loop.py` |
| Dataset adapters | `experiments/spurious_eval/datasets/` |
| Linear evaluation | `experiments/spurious_eval/linear_probe.py` |
| Experiment matrix and execution identity | `experiments/runner.py`, `experiments/manifests/` |
| Checked paper registry | `scripts/tools/build_paper_results.py`, `paper_results.json` |
| HTML reports | `splice/reporting.py`, `scripts/tools/render_report.py` |

The teacher-input flow has three explicit module interfaces: the SpLiCE dataset
cache serializes frozen train-split representations, concept grouping consumes
that cache and serializes groups, and `build_teacher_graph(splice_dataset_cache,
concept_groups, config)` consumes both artifacts for projection geometry, null
controls and sparse graph assembly. Training consumes only validated graphs
through `splice.cospro_training`.

The `splice.crp*` compatibility modules, legacy command wrappers, and serialized
`crp_*` fields remain readable for existing artifacts and runs. New code,
commands, reports, studies, and documentation use CoSpRo consistently.

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
