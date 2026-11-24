# CoSpRo (teacher graph: sampler and relational KL) on metashift

Validation probe. `val WGA` and `val acc` average the last four periodic probes over the seeds (± standard deviation over seeds; * marks rows that fall back to the last probe). Rows are grouped by study and sorted by val WGA inside it. Compare rows only within one probe protocol (`probe`).

| study | arm | epochs | batch | lr | wd | temp | probe | graph | KL weight | KL temp | SimCLR weight | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metashift_cospro | cospro_laion | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | cospro_laion_graph | 0.5 | 0.25 | 1 | 1,2,3,4,5,6,7,8,9,10 | 47.7 ± 4.3 | 55.5 ± 2.5 |
| metashift_cospro | raw_clip_kl | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | raw_clip_graph | 0.5 | 0.25 | 1 | 1,2,3,4,5,6,7,8,9,10 | 45.1 ± 4.9 | 53.6 ± 2.5 |
| metashift_cospro | cospro | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | cospro_graph | 0.5 | 0.25 | 1 | 1,2,3,4,5,6,7,8,9,10 (+1 unfinished) | 42.9 ± 2.7 | 55.2 ± 2.0 |
| metashift_cospro | cospro_laion_gated | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | cospro_laion_gated_graph | 0.5 | 0.25 | 1 | 1,2,3,4,5 | 42.4 ± 3.4 | 54.5 ± 2.5 |
| metashift_cospro_pipeline | cospro | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | groups text_0p82_coactivation_0p35 | 0.5 | 0.25 | 1 | 1 | 45.1 | 56.9 |

## SimCLR reference on this dataset

| study | arm | epochs | batch | lr | wd | temp | probe | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|
| factors_metashift | simclr_lt | 500 | 256 | 0.05 | 0.001 | 0.05 | logistic/ds_train | 1,2 | 51.4 ± 1.5 | 55.3 ± 0.1 |
| factors_metashift | curves_unit_0 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 50.0 ± 5.0 | 61.7 ± 3.8 |
| factors_metashift | curves_unit_3 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 50.0 ± 1.8 | 57.1 ± 1.6 |
| factors_metashift | curves | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 49.8 ± 2.2 | 60.7 ± 3.9 |
| factors_metashift | curves_unit_10 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 49.3 ± 2.5 | 57.3 ± 2.6 |
| factors_metashift | curves_unit_4 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 48.6 ± 2.0 | 55.8 ± 1.9 |
| factors_metashift | curves_unit_2 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 48.5 ± 3.2 | 57.4 ± 2.7 |
| factors_metashift | curves_unit_1 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 48.4 ± 7.3 | 56.5 ± 5.7 |
| factors_metashift | curves_bottom | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 48.0 ± 2.9 | 56.2 ± 2.4 |
| factors_metashift | curves_oracle | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 47.7 ± 4.8 | 56.0 ± 4.7 |
| factors_metashift | curves_unit_5 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 47.7 ± 1.9 | 55.9 ± 2.8 |
| factors_metashift | curves_unit_7 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 47.5 ± 3.6 | 56.1 ± 1.4 |
| factors_metashift | curves_shuffled | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 45.7 ± 3.0 | 55.4 ± 0.3 |
| factors_metashift | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4,5,6 | 44.3 ± 7.3 | 55.2 ± 4.0 |
| factors_metashift | csa_audit | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4,5,6,7,8 | 44.3 ± 6.4 | 54.8 ± 3.6 |
| factors_metashift | curves_unit_11 | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 44.2 ± 4.9 | 53.8 ± 3.4 |
| factors_metashift | curves_random | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 43.1 ± 3.6 | 59.0 ± 2.6 |
| metashift_cospro | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4,5,6,7,8,9,10 | 44.6 ± 6.9 | 54.6 ± 3.8 |

Run records: `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`.
