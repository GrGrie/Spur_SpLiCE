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
| factors_metashift | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 44.8 ± 9.3 | 54.8 ± 4.8 |
| metashift_cospro | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4,5,6,7,8,9,10 | 44.6 ± 6.9 | 54.6 ± 3.8 |

Run records: `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`.
