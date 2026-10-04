# Concept factors (F1 conditioned batches, F2 factor distillation, F3 concept blocks) on spur_cifar10

Validation probe. `val WGA` and `val acc` average the last four periodic probes over the seeds (± standard deviation over seeds; * marks rows that fall back to the last probe). Rows are grouped by study and sorted by val WGA inside it. Compare rows only within one probe protocol (`probe`).

| study | arm | epochs | batch | lr | wd | temp | probe | F1 fraction | F2 weight | targets | merge | min freq | F2 weighting | F2 cross-fit | F3 weight | F3 context | F3 presence | start | groups | factors | seeds | val WGA | val acc | factor expl. var. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| factors_spur_cifar10 | xfit_norm_clippca | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | clip_pca | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 88 | 1,2 | 37.0 ± 0.3 | 70.9 ± 0.1 |  |
| factors_spur_cifar10 | xfit_norm | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 88 | 1,2 | 33.6 ± 2.3 | 69.7 ± 0.2 |  |
| factors_spur_cifar10 | f2_std | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 | none | False | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 88 | 1,2 | 32.9 ± 1.6 | 69.5 ± 0.8 | 0.62 |
| factors_spur_cifar10 | xfit_meaning_w3_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0 | 0.05 | none | True | 0 |  |  | 10 | openimages | 24 | 1,2 | 32.1 ± 9.4 | 63.6 ± 0.9 |  |
| factors_spur_cifar10 | xfit_norm_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 88 | 1,2 | 29.4 ± 2.4 | 68.7 ± 0.1 |  |
| factors_spur_cifar10 | xfit_meaning_w5_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 5 | standardized | 0 | 0.05 | none | True | 0 |  |  | 10 | openimages | 24 | 1,2 | 25.9 ± 3.5 | 63.0 ± 0.5 |  |
| factors_spur_cifar10 | xfit_meaning_w1_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0 | 0.05 | none | True | 0 |  |  | 10 | openimages | 24 | 1,2 | 25.5 ± 1.5 | 65.3 ± 0.3 |  |
| factors_spur_cifar10 | f2_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | shuffled | 0.9 | 0.02 | none | False | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 88 | 1,2 | 24.2 ± 0.4 | 58.6 ± 0.7 | 0.02 |
| factors_spur_cifar10 | xfit_meaning_w10_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 10 | standardized | 0 | 0.05 | none | True | 0 |  |  | 10 | openimages | 24 | 1,2 | 24.1 ± 2.0 | 61.9 ± 0.2 |  |
| factors_spur_cifar10 | xfit_meaning_w10 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 10 | standardized | 0 | 0.05 | none | True | 0 |  |  | 10 | openimages | 24 | 1,2 | 22.7 ± 0.5 | 62.0 ± 1.3 |  |
| factors_spur_cifar10 | xfit_norm_shuffled_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | shuffled | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 88 | 1,2 | 22.3 ± 3.1 | 58.2 ± 1.5 |  |
| factors_spur_cifar10 | xfit_norm_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | shuffled | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 88 | 1,2 | 18.2 ± 3.2 | 57.5 ± 1.8 |  |
| factors_spur_cifar10 | xfit_meaning_w10_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 10 | shuffled | 0 | 0.05 | none | True | 0 |  |  | 10 | openimages | 24 | 1,2 | 16.9 ± 2.7 | 57.5 ± 0.1 |  |
| factors_spur_cifar10 | xfit_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | shuffled | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 88 | 1,2 (+1 unfinished) | 15.3 ± 9.0 | 48.8 ± 14.4 |  |
| factors_spur_cifar10 | xfit | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 88 |  (+2 unfinished) |  |  |  |

## SimCLR reference on this dataset

| study | arm | epochs | batch | lr | wd | temp | probe | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|
| factors_spur_cifar10 | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2 | 15.6 ± 1.0 | 59.5 ± 0.2 |

Run records: `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`.
