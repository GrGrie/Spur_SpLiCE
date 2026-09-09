# Project map

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
| Artifact locations | `splice/artifacts.py` |
| Frozen feature cache | `scripts/tools/cache_crp_features.py` |
| CRP teacher graph | `splice/crp.py`, `splice/graph_io.py` |
| Graph sampler and relational KL | `splice/crp_training.py` |
| SSL training | `spur_splice.py`, `experiments/spurious_eval/training/ssl_loop.py` |
| Dataset adapters | `experiments/spurious_eval/datasets/` |
| Linear evaluation | `experiments/spurious_eval/linear_probe.py` |
| Experiment matrix | `experiments/runner.py`, `experiments/manifests/` |
| HTML reports | `splice/reporting.py`, `scripts/tools/render_report.py` |

The main CRP interface is `build_teacher_graph(cache, config)`. Its
implementation owns grouping, projection geometry, null controls and sparse
graph assembly. Training consumes only validated graphs through
`splice.crp_training`.

## Results

- `outputs/seeds/seed_01/` through `seed_04/` contain every seed-specific run.
- `outputs/shared/waterbirds/` contains the frozen cache and teacher graphs.
- `outputs/reports/` contains aggregate results and completed-study provenance.
- `outputs/reference/` contains vocabularies and external exports.

Historical source paths embedded in result JSON files are retained as
provenance. New code must obtain destinations through `splice.artifacts`.
