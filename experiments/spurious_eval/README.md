# Spurious-correlation evaluation

The supported SSL modes are:

- `none` — ordinary SimCLR;
- `cospro_relational` — CoSpRo graph-aware batches with optional relational KL weight.

`crp_relational` remains accepted as a compatibility alias for historical
manifests and checkpoints.

Run `python -m experiments.spurious_eval.linear_probe --help` for standalone
evaluation. The canonical experiment manifest uses group-balanced `ds_train`
linear fitting and keeps final test evaluation separate.
