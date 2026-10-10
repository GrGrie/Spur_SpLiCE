# SimCLR on metashift

Validation probe. `val WGA` and `val acc` average the last four periodic probes over the seeds (± standard deviation over seeds; * marks rows that fall back to the last probe). Rows are grouped by study and sorted by val WGA inside it. Compare rows only within one probe protocol (`probe`).

| study | arm | epochs | batch | lr | wd | temp | probe | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|
| factors_metashift | simclr_lt | 500 | 256 | 0.05 | 0.001 | 0.05 | logistic/ds_train | 1,2 | 51.4 ± 1.5 | 55.3 ± 0.1 |
| factors_metashift | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4,5,6 | 44.3 ± 7.3 | 55.2 ± 4.0 |
| metashift_cospro | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4,5,6,7,8,9,10 | 44.6 ± 6.9 | 54.6 ± 3.8 |

Run records: `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`.
