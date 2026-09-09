# Scripts

This directory intentionally has one Slurm adapter.

```bash
sbatch --array=0-19 scripts/run_experiment.sbatch
```

`run_experiment.sbatch` activates the cluster environment and calls
`experiments.runner`. Experiment parameters live in immutable JSON manifests,
not in chains of prepare/array/summary launchers.

Small Python tools remain for operations that are not training matrices:

- `cache_crp_features.py` — build the frozen CRP cache;
- `build_crp_baseline_graphs.py` — build a raw-CLIP graph with anchor support,
  row degrees, weight profiles, confidence, and indegree cap matched to a
  canonical CRP reference graph;
- `download_waterbirds_hf.py` — dataset helper;
- `summarize_crp_audit.py` — concise graph summary;
- `render_report.py` — the single HTML report adapter.

Use `python -m <module> --help` for their interfaces.

Artifact paths default to `outputs/`. Set `SPUR_SPLICE_ARTIFACT_ROOT` or pass
`experiments.runner --artifact-root PATH` when using a packaged tree such as
`outputs/output_cluster`; the runner substitutes `{artifacts}` in manifests.
