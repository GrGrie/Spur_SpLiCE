# Frozen concept transfer (direct distillation of CLIP/SpLiCE targets) on waterbirds

Validation probe. `val WGA` and `val acc` average the last four periodic probes over the seeds (± standard deviation over seeds; * marks rows that fall back to the last probe). Rows are grouped by study and sorted by val WGA inside it. Compare rows only within one probe protocol (`probe`).

| study | arm | epochs | batch | lr | wd | temp | probe | targets | alpha | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| next_actions_after_transfer_2026_09_07_direct_transfer | raw_distillation | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | raw | 0.1 | 1,3 | 47.7 ± 2.7* | 52.9 ± 0.1 |
| next_actions_after_transfer_2026_09_07_direct_transfer | splice_reconstruction | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | reconstruction | 0.1 | 1,2,3,4 | 46.6 ± 4.5* | 52.9 ± 2.2 |
| next_actions_after_transfer_2026_09_07_direct_transfer | shuffled_reconstruction | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | shuffled_reconstruction | 0.1 | 1,3 | 46.6 ± 0.2* | 51.8 ± 0.3 |
| next_actions_after_transfer_2026_09_07_direct_transfer | matched_simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | raw | 0 | 1,3 | 45.9 ± 2.0* | 51.5 ± 1.7 |
| next_tests_2026_09_07_concept_transfer | matched_simclr | 500 | 128 | 0.01 | 0.0001 | 0.5 | logistic/ds_train | raw | 0 | 1,3 | 44.9 ± 10.8* | 54.1 ± 3.4 |
| next_tests_2026_09_07_concept_transfer | raw_distillation | 500 | 128 | 0.01 | 0.0001 | 0.5 | logistic/ds_train | raw | 0.1 | 1,3 | 39.8 ± 10.0* | 50.9 ± 2.8 |
| next_tests_2026_09_07_concept_transfer | shuffled_reconstruction | 500 | 128 | 0.01 | 0.0001 | 0.5 | logistic/ds_train | shuffled_reconstruction | 0.1 | 1,3 | 39.4 ± 12.1* | 52.0 ± 2.9 |
| next_tests_2026_09_07_concept_transfer | splice_reconstruction | 500 | 128 | 0.01 | 0.0001 | 0.5 | logistic/ds_train | reconstruction | 0.1 | 1,3 | 38.9 ± 21.7* | 52.2 ± 9.3 |

## SimCLR reference on this dataset

| study | arm | epochs | batch | lr | wd | temp | probe | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|
| crp_followup_controls_v1_seeds34 | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 3,4 | 47.0 ± 0.5* | 53.1 ± 0.6 |
| paper_completion_2026_09_08_core | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 45.5 ± 1.7* | 52.1 ± 1.4 |
| waterbirds_cospro | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1 | 45.0 | 50.7 |

Run records: `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`.
