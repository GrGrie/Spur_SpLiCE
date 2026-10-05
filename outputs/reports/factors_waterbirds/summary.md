# factors_waterbirds

Validation probe; `last 4` averages the last four periodic probes. ± is the standard deviation over seeds. `val group acc` lists the mean accuracy of every (class, attribute) group in the dataset's group order. `held-out expl. var.` is the factor variance a ridge regression fitted on the other half of the batch explains, which memorizing images cannot raise.

## waterbirds

| arm | seeds | val WGA (last 4) | val WGA (last) | val acc (last 4) | val group acc | factor expl. var. | held-out expl. var. |
|---|---|---|---|---|---|---|---|
| xfit_meaning_w5_response_holdout | 1,2 | 54.4 ± 3.9 | 54.1 ± 3.2 | 65.4 ± 0.3 | 65 / 67 / 54 / 72 |  | 0.60 |
| spatial_w5_holdout | 1,2,3,4 | 53.0 ± 3.0 | 53.1 ± 3.3 | 60.1 ± 2.5 | 55 / 64 / 59 / 68 |  |  |
| xfit_meaning_w10_holdout | 1,2,3,4 | 51.5 ± 5.5 | 51.6 ± 5.9 | 57.5 ± 2.0 | 52 / 57 / 65 / 73 |  | 0.12 |
| xfit_meaning_w5_clippca_holdout | 1,2 | 51.0 ± 5.9 | 51.3 ± 5.9 | 59.4 ± 1.5 | 51 / 63 / 68 / 68 |  | 0.06 |
| xfit_meaning_w5_holdout | 1,2,3,4 | 50.4 ± 9.0 | 50.6 ± 8.6 | 58.0 ± 3.9 | 51 / 62 / 61 / 66 |  | 0.07 |
| spatial_w5_shuffled_holdout | 1,2,3,4 | 46.7 ± 2.7 | 46.9 ± 2.8 | 52.6 ± 3.5 | 48 / 56 / 54 / 57 |  |  |
| xfit_norm_holdout | 1,2 | 46.6 ± 0.5 | 46.5 ± 0.9 | 51.1 ± 1.1 | 46 / 52 / 52 / 64 |  | -0.01 |
| simclr | 1,2,3,4 | 45.8 ± 1.7 | 45.5 ± 1.7 | 52.3 ± 1.2 | 47 / 54 / 52 / 64 |  |  |
| xfit_meaning_w3_holdout | 1,2 | 45.7 ± 3.7 | 45.6 ± 3.9 | 54.0 ± 2.0 | 46 / 54 / 67 / 69 |  | 0.04 |
| f2_shuffled | 1,2,3,4 | 45.5 ± 3.5 | 45.7 ± 3.4 | 51.8 ± 0.7 | 47 / 54 / 53 / 61 | 0.03 | -0.08 |
| xfit_meaning_w5_response_residual_holdout | 1,2 | 43.9 ± 6.1 | 44.2 ± 6.8 | 51.5 ± 1.3 | 44 / 54 / 57 / 60 |  | 0.11 |
| xfit_norm_shuffled_holdout | 1,2 | 43.6 ± 2.2 | 43.9 ± 2.1 | 50.3 ± 0.2 | 44 / 51 / 58 / 64 |  | -0.03 |
| xfit_meaning_w10_residual_holdout | 1,2 | 41.7 ± 0.9 | 41.9 ± 1.1 | 55.3 ± 2.2 | 42 / 66 / 61 / 61 |  | 0.00 |
| f2_std | 1,2,3,4 | 41.6 ± 6.1 | 41.3 ± 6.1 | 51.2 ± 1.6 | 41 / 53 / 61 / 67 | 0.10 | -0.05 |
| xfit_meaning_w10_shuffled_holdout | 1,2,3,4 | 41.4 ± 3.0 | 41.2 ± 3.1 | 51.4 ± 2.2 | 41 / 55 / 61 / 66 |  | -0.02 |
