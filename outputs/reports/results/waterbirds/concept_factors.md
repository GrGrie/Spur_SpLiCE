# Concept factors (F1 conditioned batches, F2 factor distillation, F3 concept blocks) on waterbirds

Validation probe. `val WGA` and `val acc` average the last four periodic probes over the seeds (± standard deviation over seeds; * marks rows that fall back to the last probe). Rows are grouped by study and sorted by val WGA inside it. Compare rows only within one probe protocol (`probe`).

| study | arm | epochs | batch | lr | wd | temp | probe | F1 fraction | F2 weight | targets | merge | min freq | F2 weighting | F2 cross-fit | F3 weight | F3 context | F3 presence | start | groups | factors | seeds | val WGA | val acc | factor expl. var. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| factors_waterbirds | xfit_meaning_w5_response_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 5 | response | 0 | 0.05 | none | True | 0 |  |  | 10 | openimages | 70 | 1,2 | 54.4 ± 3.9 | 65.4 ± 0.3 |  |
| factors_waterbirds | xfit_meaning_w10_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 10 | standardized | 0 | 0.05 | none | True | 0 |  |  | 10 | openimages | 70 | 1,2,3,4 | 51.5 ± 5.5 | 57.5 ± 2.0 |  |
| factors_waterbirds | xfit_meaning_w5_clippca_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 5 | clip_pca | 0 | 0.05 | none | True | 0 |  |  | 10 | openimages | 70 | 1,2 | 51.0 ± 5.9 | 59.4 ± 1.5 |  |
| factors_waterbirds | xfit_meaning_w5_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 5 | standardized | 0 | 0.05 | none | True | 0 |  |  | 10 | openimages | 70 | 1,2,3,4 | 50.4 ± 9.0 | 58.0 ± 3.9 |  |
| factors_waterbirds | xfit_norm_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 217 | 1,2 | 46.6 ± 0.5 | 51.1 ± 1.1 |  |
| factors_waterbirds | xfit_meaning_w3_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0 | 0.05 | none | True | 0 |  |  | 10 | openimages | 70 | 1,2 | 45.7 ± 3.7 | 54.0 ± 2.0 |  |
| factors_waterbirds | f2_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | shuffled | 0.9 | 0.02 | none | False | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 217 | 1,2,3,4 | 45.5 ± 3.5 | 51.8 ± 0.7 | 0.03 |
| factors_waterbirds | xfit_meaning_w5_response_residual_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 5 | response_residual | 0 | 0.05 | none | True | 0 |  |  | 10 | openimages | 70 | 1,2 | 43.9 ± 6.1 | 51.5 ± 1.3 |  |
| factors_waterbirds | xfit_norm_shuffled_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | shuffled | 0.9 | 0.02 | none | True | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 217 | 1,2 | 43.6 ± 2.2 | 50.3 ± 0.2 |  |
| factors_waterbirds | xfit_meaning_w10_residual_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 10 | residual | 0 | 0.05 | none | True | 0 |  |  | 10 | openimages | 70 | 1,2 | 41.7 ± 0.9 | 55.3 ± 2.2 |  |
| factors_waterbirds | f2_std | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 | none | False | 0 |  |  | 10 | openimages text_0p8_coactivation_0p3 | 217 | 1,2,3,4 | 41.6 ± 6.1 | 51.2 ± 1.6 | 0.10 |
| factors_waterbirds | xfit_meaning_w10_shuffled_holdout | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 10 | shuffled | 0 | 0.05 | none | True | 0 |  |  | 10 | openimages | 70 | 1,2,3,4 | 41.4 ± 3.0 | 51.4 ± 2.2 |  |

## SimCLR reference on this dataset

| study | arm | epochs | batch | lr | wd | temp | probe | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|
| crp_followup_controls_v1_seeds34 | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 3,4 | 47.0 ± 0.5* | 53.1 ± 0.6 |
| factors_waterbirds | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 45.8 ± 1.7 | 52.3 ± 1.2 |
| paper_completion_2026_09_08_core | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 45.5 ± 1.7* | 52.1 ± 1.4 |
| waterbirds_cospro | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1 | 45.0 | 50.7 |

Run records: `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`.
