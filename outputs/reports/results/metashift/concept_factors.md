# Concept factors (F1 conditioned batches, F2 factor distillation) on metashift

Validation probe. `val WGA` and `val acc` average the last four periodic probes over the seeds (± standard deviation over seeds; * marks rows that fall back to the last probe). Rows are grouped by study and sorted by val WGA inside it. Compare rows only within one probe protocol (`probe`).

| study | arm | epochs | batch | lr | wd | temp | probe | F1 fraction | F2 weight | targets | merge | min freq | F2 start | groups | factors | seeds | val WGA | val acc | factor expl. var. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| factors_metashift | f2_w3 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 3 | standardized | 0.9 | 0.02 | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 51.4 ± 1.5 | 63.1 ± 2.0 | 0.41 |
| factors_metashift | f2_std_lt | 500 | 256 | 0.05 | 0.001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 48.6 ± 5.9 | 57.4 ± 0.2 | 0.10 |
| factors_metashift | f2_presence | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | presence | 0.9 | 0.02 | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 48.3 ± 7.9 | 57.1 ± 2.1 | 0.15 |
| factors_metashift | f2_w10 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 10 | standardized | 0.9 | 0.02 | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 47.0 ± 7.1 | 59.7 ± 1.4 | 0.85 |
| factors_metashift | f2_laion | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 | 10 | laion text_0p8_coactivation_0p3 | 103 | 1,2 | 45.8 ± 1.0 | 57.5 ± 0.5 | 0.18 |
| factors_metashift | f2_start0 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 | 0 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 45.0 ± 1.7 | 56.6 ± 0.2 | 0.12 |
| factors_metashift | f2_coarse | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.7 | 0.05 | 10 | openimages text_0p8_coactivation_0p3 | 7 | 1,2 | 43.4 ± 2.5 | 60.6 ± 2.1 | 0.89 |
| factors_metashift | f2_std | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | standardized | 0.9 | 0.02 | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 43.2 ± 1.2 | 58.2 ± 1.9 | 0.12 |
| factors_metashift | f2_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 0 | 1 | shuffled | 0.9 | 0.02 | 10 | openimages text_0p8_coactivation_0p3 | 145 | 1,2 | 42.2 ± 4.7 | 51.0 ± 0.9 | 0.06 |

## SimCLR reference on this dataset

| study | arm | epochs | batch | lr | wd | temp | probe | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|
| factors_metashift | simclr_lt | 500 | 256 | 0.05 | 0.001 | 0.05 | logistic/ds_train | 1,2 | 51.4 ± 1.5 | 55.3 ± 0.1 |
| factors_metashift | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2 | 47.7 ± 4.2 | 55.8 ± 0.7 |
| metashift_cospro | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4,5,6,7,8,9,10 | 44.6 ± 6.9 | 54.6 ± 3.8 |

Run records: `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`.
