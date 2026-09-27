# CoSpRo (teacher graph: sampler and relational KL) on celeba

Validation probe. `val WGA` and `val acc` average the last four periodic probes over the seeds (± standard deviation over seeds; * marks rows that fall back to the last probe). Rows are grouped by study and sorted by val WGA inside it. Compare rows only within one probe protocol (`probe`).

| study | arm | epochs | batch | lr | wd | temp | probe | graph | KL weight | KL temp | SimCLR weight | seeds | val WGA | val acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| celeba_cospro_pipeline | cospro | 250 | 128 | 0.01 | 0.0001 | 0.05 | logistic/ds_train | groups text_0p82_coactivation_0p35 | 0.5 | 0.25 | 1 | 1 | 81.6 | 88.5 |

Run records: `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`.
