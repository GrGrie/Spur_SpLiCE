# CoSpRo (teacher graph: sampler and relational KL) on waterbirds

Validation probe. `val WGA` and `val acc` average the last four periodic probes over the seeds (± standard deviation over seeds; * marks rows that fall back to the last probe). Rows are grouped by study and sorted by val WGA inside it. Compare rows only within one probe protocol (`probe`).

| study | arm | epochs | batch | lr | wd | temp | probe | graph | KL weight | KL temp | SimCLR weight | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| concept_group_search_v1_ssl | baseline | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | crp_graph | 0.5 | 0.25 | 1 | 1,3 | 48.5 ± 4.3* | 52.5 ± 4.7 |
| concept_group_search_v1_ssl | semantic | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | semantic | 0.5 | 0.25 | 1 | 1,3 | 46.9 ± 1.6* | 50.7 ± 0.1 |
| crp_followup_controls_v1_seeds34 | crp_sampler_only | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | crp_graph | 0 | 0.25 | 1 | 3,4 | 49.0 ± 5.1* | 53.9 ± 5.3 |
| crp_followup_controls_v1_seeds34 | splice_crp_kl | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | crp_graph | 2 | 0.25 | 1 | 3,4 | 43.8 ± 4.0* | 50.8 ± 4.4 |
| crp_followup_controls_v1_seeds34 | raw_clip_kl | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | raw_clip_graph | 2 | 0.25 | 1 | 3,4 | 43.4 ± 6.6* | 50.8 ± 1.0 |
| crp_followup_controls_v1_seeds34 | raw_clip_sampler_only | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | raw_clip_graph | 0 | 0.25 | 1 | 3,4 | 43.2 ± 4.5* | 50.7 ± 0.9 |
| crp_followup_raw_sampler_only_v1_seeds12 | raw_clip_sampler_only | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | raw_clip_graph | 0 | 0.25 | 1 | 1,2 | 45.3 ± 1.3* | 51.3 ± 4.0 |
| crp_lambda05_replication_s34 | splice_crp_kl | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | crp_graph | 0.5 | 0.25 | 1 | 3,4 | 49.2 ± 5.2* | 52.4 ± 4.5 |
| crp_lambda05_replication_s34 | raw_clip_kl | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | raw_clip_graph | 0.5 | 0.25 | 1 | 3,4 | 48.5 ± 3.5* | 53.2 ± 1.4 |
| crp_signal_checks_v1_transfer_lambda_0_2 | raw_clip_kl | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | raw_clip_graph | 0.2 | 0.25 | 1 | 1,2 | 48.4 ± 1.5* | 51.8 ± 2.7 |
| crp_signal_checks_v1_transfer_lambda_0_2 | splice_crp_kl | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | crp_graph | 0.2 | 0.25 | 1 | 1,2 | 44.6 ± 1.1* | 50.3 ± 1.3 |
| crp_signal_checks_v1_transfer_lambda_0_5 | splice_crp_kl | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | crp_graph | 0.5 | 0.25 | 1 | 1,2 | 50.8 ± 1.1* | 54.5 ± 1.8 |
| crp_signal_checks_v1_transfer_lambda_0_5 | raw_clip_kl | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | raw_clip_graph | 0.5 | 0.25 | 1 | 1,2 | 45.4 ± 1.6* | 49.9 ± 0.6 |
| next_actions_after_transfer_2026_09_07_graph_ablation | crp | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | crp_graph | 0.5 | 0.25 | 1 | 1,3 | 48.5 ± 4.3* | 52.5 ± 4.7 |
| next_actions_after_transfer_2026_09_07_graph_ablation | semantic_splice | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | semantic_splice_graph | 0.5 | 0.25 | 1 | 1,2,3,4 | 48.0 ± 4.4* | 53.1 ± 2.2 |
| next_tests_2026_09_07_gradient | crp_lambda05 | 25 | 128 | 0.01 | 0.0001 | 0.5 | logistic/ds_train | crp_graph | 0.5 | 0.1 | 1 | 3 |  |  |
| next_tests_2026_09_07_gradient | crp_lambda2 | 25 | 128 | 0.01 | 0.0001 | 0.5 | logistic/ds_train | crp_graph | 2 | 0.1 | 1 | 3 |  |  |
| next_tests_2026_09_07_gradient | raw_lambda05 | 25 | 128 | 0.01 | 0.0001 | 0.5 | logistic/ds_train | raw_clip_graph | 0.5 | 0.1 | 1 | 3 |  |  |
| paper_completion_2026_09_08_core | splice_crp_kl | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | crp_graph | 0.5 | 0.25 | 1 | 1,2,3,4 | 50.0 ± 3.2* | 53.5 ± 3.1 |
| paper_completion_2026_09_08_core | crp_sampler_only | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | crp_graph | 0 | 0.25 | 1 | 1,2,3,4 | 47.4 ± 4.1* | 52.5 ± 3.6 |
| paper_completion_2026_09_08_core | raw_clip_kl | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | raw_clip_graph | 0.5 | 0.25 | 1 | 1,2,3,4 | 47.0 ± 2.8* | 51.5 ± 2.1 |
| paper_completion_2026_09_08_core | raw_clip_sampler_only | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | raw_clip_graph | 0 | 0.25 | 1 | 1,2,3,4 | 44.3 ± 3.0* | 51.0 ± 2.4 |
| raw_lambda02_replication_s34 | raw_clip_kl | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | raw_clip_graph | 0.2 | 0.25 | 1 | 3,4 | 45.8 ± 3.6* | 51.5 ± 0.1 |
| waterbirds_cospro | cospro | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | crp_graph | 0.5 | 0.25 | 1 | 1 | 51.7 | 55.8 |
| waterbirds_cospro | cospro_laion_gated | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | cospro_laion_gated_graph | 0.5 | 0.25 | 1 | 1,2,3,4 | 47.7 ± 3.3 | 53.7 ± 0.9 |
| waterbirds_cospro | cospro_laion | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | cospro_laion_graph | 0.5 | 0.25 | 1 | 1,2,3,4 (+4 unfinished) | 44.7 ± 6.7 | 50.5 ± 3.4 |

## SimCLR reference on this dataset

| study | arm | epochs | batch | lr | wd | temp | probe | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|
| crp_followup_controls_v1_seeds34 | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 3,4 | 47.0 ± 0.5* | 53.1 ± 0.6 |
| paper_completion_2026_09_08_core | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1,2,3,4 | 45.5 ± 1.7* | 52.1 ± 1.4 |
| waterbirds_cospro | simclr | 500 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | 1 | 45.0 | 50.7 |

Run records: `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`.
