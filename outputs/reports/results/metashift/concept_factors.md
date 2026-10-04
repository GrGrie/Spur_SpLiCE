# Concept factors (F1 conditioned batches, F2 factor distillation, F3 concept blocks) on metashift

Validation probe. `val WGA` and `val acc` average the last four periodic probes over the seeds (± standard deviation over seeds; * marks rows that fall back to the last probe). Rows are grouped by study and sorted by val WGA inside it. Compare rows only within one probe protocol (`probe`).

| study | arm | epochs | batch | lr | wd | temp | probe | F1 fraction | F2 weight | targets | merge | min freq | F2 weighting | F2 cross-fit | F3 weight | F3 context | F3 presence | start | groups | factors | seeds | val WGA | val acc | factor expl. var. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| factors_metashift | xfit_meaning_w5_clippca_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 5 | clip_pca | 0 | 0.05 | none | True | 0 |  |  | 10 | laion | 24 | 1,2 | 54.2 ± 5.4 | 65.9 ± 4.5 |  |
| factors_metashift | f2_w3 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0.9 | 0.02 |  |  |  |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 51.4 ± 1.5 | 63.1 ± 2.0 | 0.41 |
| factors_metashift | f2_w3_atyp | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0.9 | 0.02 | atypicality |  |  |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2,3,4 | 50.5 ± 3.8 | 58.8 ± 2.1 | 0.61 |
| factors_metashift | xfit_meaning_w5_response_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 5 | response | 0 | 0.05 | none | True | 0 |  |  | 10 | laion | 24 | 1,2 | 49.8 ± 4.7 | 63.7 ± 0.3 |  |
| factors_metashift | cbc_plain | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 0 | whitened | 0.9 | 0.02 | none |  | 1 | 0 | real | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2,3,4 | 49.7 ± 2.5 | 62.6 ± 0.8 |  |
| factors_metashift | cbc_f2_balanced | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0.9 | 0.02 | balanced |  | 1 | 1 | real | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2,3,4 | 49.4 ± 3.8 | 64.4 ± 2.2 | 0.42 |
| factors_metashift | f2_std_lt | 500 | 256 | 0.05 | 0.001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 |  |  |  |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 48.6 ± 5.9 | 57.4 ± 0.2 | 0.10 |
| factors_metashift | f2_presence | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | presence | 0.9 | 0.02 |  |  |  |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 48.3 ± 7.9 | 57.1 ± 2.1 | 0.15 |
| factors_metashift | f2_white_w3_atyp | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | whitened | 0.9 | 0.02 | atypicality |  |  |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2,3,4 | 48.2 ± 3.9 | 57.6 ± 4.2 | 0.56 |
| factors_metashift | f2_white_w3 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | whitened | 0.9 | 0.02 | none |  |  |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2,3,4 | 47.9 ± 6.4 | 56.9 ± 3.0 | 0.31 |
| factors_metashift | xfit_meaning_w3_residual_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | residual | 0 | 0.05 | none | True | 0 |  |  | 10 | laion | 24 | 1,2 | 47.7 ± 0.2 | 58.6 ± 5.0 |  |
| factors_metashift | xfit_meaning_w3_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | shuffled | 0 | 0.05 | none | True | 0 |  |  | 10 | laion | 24 | 1,2 | 47.6 ± 5.4 | 58.1 ± 2.9 |  |
| factors_metashift | xfit_meaning_w10_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 10 | standardized | 0 | 0.05 | none | True | 0 |  |  | 10 | laion | 24 | 1,2 | 47.6 ± 1.0 | 60.7 ± 1.7 |  |
| factors_metashift | xfit_meaning_w5_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 5 | standardized | 0 | 0.05 | none | True | 0 |  |  | 10 | laion | 24 | 1,2 | 47.2 ± 3.9 | 61.7 ± 0.2 |  |
| factors_metashift | f2_w10 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 10 | standardized | 0.9 | 0.02 |  |  |  |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 47.0 ± 7.1 | 59.7 ± 1.4 | 0.85 |
| factors_metashift | xfit_norm_balanced | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 | balanced | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 46.9 ± 4.4 | 58.5 ± 1.2 |  |
| factors_metashift | xfit_meaning_w3_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0 | 0.05 | none | True | 0 |  |  | 10 | laion | 24 | 1,2 | 46.9 ± 2.0 | 60.3 ± 2.1 |  |
| factors_metashift | xfit_meaning_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | shuffled | 0 | 0.05 | none | True | 0 |  |  | 10 | laion | 24 | 1,2 | 46.7 ± 1.7 | 58.6 ± 1.8 |  |
| factors_metashift | cbc_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 0 | whitened | 0.9 | 0.02 | none |  | 1 | 1 | shuffled | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2,3,4 | 46.4 ± 3.5 | 56.2 ± 2.5 |  |
| factors_metashift | xfit_meaning | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0 | 0.05 | none | True | 0 |  |  | 10 | laion | 24 | 1,2 | 46.0 ± 2.2 | 58.8 ± 3.1 |  |
| factors_metashift | f2_laion | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 |  |  |  |  |  | 10 | laion text_0p8_coactivation_0p3 | 103 | 1,2 | 45.8 ± 1.0 | 57.5 ± 0.5 | 0.18 |
| factors_metashift | xfit_meaning_w10 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 10 | standardized | 0 | 0.05 | none | True | 0 |  |  | 10 | laion | 24 | 1,2 | 45.5 ± 4.4 | 61.0 ± 1.4 |  |
| factors_metashift | f2_start0 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 |  |  |  |  |  | 0 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 45.0 ± 1.7 | 56.6 ± 0.2 | 0.12 |
| factors_metashift | xfit_balanced | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0.9 | 0.02 | balanced | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2,4 (+1 unfinished) | 44.6 ± 5.0 | 54.7 ± 5.9 |  |
| factors_metashift | xfit_meaning_w5_response_residual_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 5 | response_residual | 0 | 0.05 | none | True | 0 |  |  | 10 | laion | 24 | 1,2 | 44.4 ± 4.9 | 57.7 ± 3.4 |  |
| factors_metashift | f2_w3_balanced | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0.9 | 0.02 | balanced |  | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2,3,4 | 44.2 ± 8.1 | 59.0 ± 2.2 | 0.39 |
| factors_metashift | xfit_norm_w3 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 43.9 ± 0.7 | 57.8 ± 4.5 |  |
| factors_metashift | xfit_norm | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 43.9 ± 3.2 | 54.8 ± 0.9 |  |
| factors_metashift | cbc | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 0 | whitened | 0.9 | 0.02 | none |  | 1 | 1 | real | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2,3,4 | 43.8 ± 4.1 | 60.0 ± 1.3 |  |
| factors_metashift | xfit_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | shuffled | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2,3,4 | 43.8 ± 7.8 | 55.1 ± 4.4 |  |
| factors_metashift | xfit | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2,3,4 | 43.5 ± 4.4 | 55.9 ± 1.8 |  |
| factors_metashift | f2_coarse | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.7 | 0.05 |  |  |  |  |  | 10 | openimages text_0p8_coactivation_0p3 | 7 | 1,2 | 43.4 ± 2.5 | 60.6 ± 2.1 | 0.89 |
| factors_metashift | f2_shuffled_w3 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | shuffled | 0.9 | 0.02 | none |  |  |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2,3,4 | 43.3 ± 8.1 | 54.8 ± 3.3 | 0.33 |
| factors_metashift | xfit_meaning_w10_shuffled_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 10 | shuffled | 0 | 0.05 | none | True | 0 |  |  | 10 | laion | 24 | 1,2 | 43.2 ± 3.2 | 52.2 ± 2.8 |  |
| factors_metashift | xfit_meaning_w3 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0 | 0.05 | none | True | 0 |  |  | 10 | laion | 24 | 1,2 | 43.2 ± 4.7 | 59.6 ± 1.2 |  |
| factors_metashift | f2_std | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 |  |  |  |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 43.2 ± 1.2 | 58.2 ± 1.9 | 0.12 |
| factors_metashift | f2_white_w10 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 10 | whitened | 0.9 | 0.02 | none |  |  |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2,3,4 | 42.9 ± 5.5 | 59.5 ± 0.9 | 0.84 |
| factors_metashift | xfit_norm_r10 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 42.5 ± 2.2 | 54.4 ± 2.1 |  |
| factors_metashift | f2_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | shuffled | 0.9 | 0.02 |  |  |  |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 42.2 ± 4.7 | 51.0 ± 0.9 | 0.06 |
| factors_metashift | f2_w3 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0.9 | 0.02 | none |  |  |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 3,4 | 41.8 ± 1.7 | 59.9 ± 1.5 | 0.42 |
| factors_metashift | xfit_norm_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | shuffled | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 41.7 ± 0.0 | 54.4 ± 0.5 |  |

## SimCLR reference on this dataset

| study | arm | epochs | batch | lr | wd | temp | probe | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|
| factors_metashift | simclr_lt | 500 | 256 | 0.05 | 0.001 | 0.05 | logistic/ds_train | 1,2 | 51.4 ± 1.5 | 55.3 ± 0.1 |
| factors_metashift | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 44.8 ± 9.3 | 54.8 ± 4.8 |
| metashift_cospro | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4,5,6,7,8,9,10 | 44.6 ± 6.9 | 54.6 ± 3.8 |

Run records: `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`.
