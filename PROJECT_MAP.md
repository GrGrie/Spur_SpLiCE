# Project map

Storage and synchronization rules are defined in
[`docs/REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md).

## Data flow

```text
training images
  -> frozen OpenCLIP + SpLiCE cache
  -> CRP concept grouping and intervention audit
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
| Frozen feature cache | `scripts/tools/cache_crp_features.py` |
| CRP teacher graph | `splice/crp.py`, `splice/graph_io.py` |
| Graph sampler and relational KL | `splice/crp_training.py` |
| SSL training | `spur_splice.py`, `experiments/spurious_eval/training/ssl_loop.py` |
| Dataset adapters | `experiments/spurious_eval/datasets/` |
| Linear evaluation | `experiments/spurious_eval/linear_probe.py` |
| Experiment matrix and execution identity | `experiments/runner.py`, `experiments/manifests/` |
| Checked paper registry | `scripts/tools/build_paper_results.py`, `paper_results.json` |
| HTML reports | `splice/reporting.py`, `scripts/tools/render_report.py` |

The main CRP interface is `build_teacher_graph(cache, config)`. Its
implementation owns grouping, projection geometry, null controls and sparse
graph assembly. Training consumes only validated graphs through
`splice.crp_training`.

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
