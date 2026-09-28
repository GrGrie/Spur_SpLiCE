# Concept factors (F1 conditioned batches, F2 factor distillation, F3 concept blocks) on spur_cifar10

Validation probe. `val WGA` and `val acc` average the last four periodic probes over the seeds (± standard deviation over seeds; * marks rows that fall back to the last probe). Rows are grouped by study and sorted by val WGA inside it. Compare rows only within one probe protocol (`probe`).

| study | arm | epochs | batch | lr | wd | temp | probe | F1 fraction | F2 weight | targets | merge | min freq | F2 weighting | F2 cross-fit | F3 weight | F3 context | F3 presence | start | groups | factors | seeds | val WGA | val acc | factor expl. var. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| factors_spur_cifar10 | f2_std | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 | none | False | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 88 | 1,2 | 32.9 ± 1.6 | 69.5 ± 0.8 | 0.62 |
| factors_spur_cifar10 | f2_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | shuffled | 0.9 | 0.02 | none | False | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 88 | 1,2 | 24.2 ± 0.4 | 58.6 ± 0.7 | 0.02 |
| factors_spur_cifar10 | xfit_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | shuffled | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 88 | 2 (+1 unfinished) | 6.7 | 32.1 |  |
| factors_spur_cifar10 | xfit | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 88 |  (+2 unfinished) |  |  |  |

## SimCLR reference on this dataset

| study | arm | epochs | batch | lr | wd | temp | probe | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|
| factors_spur_cifar10 | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2 | 15.6 ± 1.0 | 59.5 ± 0.2 |

Run records: `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`.
