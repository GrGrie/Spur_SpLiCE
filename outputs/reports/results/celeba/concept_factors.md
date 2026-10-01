# Concept factors (F1 conditioned batches, F2 factor distillation, F3 concept blocks) on celeba

Validation probe. `val WGA` and `val acc` average the last four periodic probes over the seeds (± standard deviation over seeds; * marks rows that fall back to the last probe). Rows are grouped by study and sorted by val WGA inside it. Compare rows only within one probe protocol (`probe`).

| study | arm | epochs | batch | lr | wd | temp | probe | F1 fraction | F2 weight | targets | merge | min freq | F2 weighting | F2 cross-fit | F3 weight | F3 context | F3 presence | start | groups | factors | seeds | val WGA | val acc | factor expl. var. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| factors_celeba | xfit_norm | 250 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 138 |  (+2 unfinished) |  |  |  |
| factors_celeba | xfit_norm_shuffled | 250 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | shuffled | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 138 |  (+2 unfinished) |  |  |  |

## SimCLR reference on this dataset

| study | arm | epochs | batch | lr | wd | temp | probe | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|
| factors_celeba | simclr | 250 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train |  (+2 unfinished) |  |  |

Run records: `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`.
