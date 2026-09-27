# SimCLR on waterbirds

Validation probe. `val WGA` and `val acc` average the last four periodic probes over the seeds (± standard deviation over seeds; * marks rows that fall back to the last probe). Rows are grouped by study and sorted by val WGA inside it. Compare rows only within one probe protocol (`probe`).

| study | arm | epochs | batch | lr | wd | temp | probe | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|
| crp_followup_controls_v1_seeds34 | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 3,4 | 47.0 ± 0.5* | 53.1 ± 0.6 |
| paper_completion_2026_09_08_core | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 45.5 ± 1.7* | 52.1 ± 1.4 |
| waterbirds_cospro | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1 | 45.0 | 50.7 |

Run records: `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`.
