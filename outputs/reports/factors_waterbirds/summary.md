# factors_waterbirds

Validation probe; `last 4` averages the last four periodic probes. ± is the standard deviation over seeds. `val group acc` lists the mean accuracy of every (class, attribute) group in the dataset's group order. `held-out expl. var.` is the factor variance a ridge regression fitted on the other half of the batch explains, which memorizing images cannot raise.

## waterbirds

| arm | seeds | val WGA (last 4) | val WGA (last) | val acc (last 4) | val group acc | factor expl. var. | held-out expl. var. |
|---|---|---|---|---|---|---|---|
| comp_w5_holdout | 1,2,3,4,5,6 | 59.1 ± 7.1 | 59.4 ± 6.9 | 64.7 ± 3.4 | 61 / 67 / 68 / 68 |  |  |
| region_w1_holdout | 1,2,3,4 | 55.3 ± 3.1 | 55.6 ± 2.6 | 62.7 ± 2.1 | 62 / 65 / 58 / 62 |  |  |
| xfit_meaning_w5_response_holdout | 1,2 | 54.4 ± 3.9 | 54.1 ± 3.2 | 65.4 ± 0.3 | 65 / 67 / 54 / 72 |  | 0.60 |
| comp_random_w5_holdout | 1,2,3,4,5,6 | 54.2 ± 6.1 | 54.4 ± 6.1 | 62.5 ± 4.0 | 56 / 67 / 65 / 68 |  |  |
| spatial_pca_w5_holdout | 1,2,3,4 | 53.6 ± 1.9 | 53.6 ± 2.0 | 58.1 ± 0.9 | 56 / 58 / 61 / 66 |  |  |
| spatial_w10_holdout | 1,2 | 53.5 ± 4.9 | 53.7 ± 5.1 | 61.8 ± 3.8 | 54 / 67 / 61 / 72 |  |  |
| comp_shuffled_w5_holdout | 1,2,3,4 | 52.9 ± 1.0 | 52.7 ± 1.1 | 57.2 ± 0.8 | 57 / 54 / 59 / 67 |  |  |
| spatial_w5_holdout | 1,2,3,4,5,6 | 52.8 ± 4.7 | 52.7 ± 4.9 | 60.3 ± 2.7 | 54 / 65 / 60 / 67 |  |  |
| comp_context_w5_holdout | 1,2,3,4 | 52.7 ± 2.1 | 53.0 ± 2.6 | 60.0 ± 1.0 | 58 / 62 / 57 / 62 |  |  |
| xfit_meaning_w10_holdout | 1,2,3,4 | 51.5 ± 5.5 | 51.6 ± 5.9 | 57.5 ± 2.0 | 52 / 57 / 65 / 73 |  | 0.12 |
| xfit_meaning_w5_clippca_holdout | 1,2 | 51.0 ± 5.9 | 51.3 ± 5.9 | 59.4 ± 1.5 | 51 / 63 / 68 / 68 |  | 0.06 |
| xfit_meaning_w5_holdout | 1,2,3,4 | 50.4 ± 9.0 | 50.6 ± 8.6 | 58.0 ± 3.9 | 51 / 62 / 61 / 66 |  | 0.07 |
| spatial_w5 | 1,2,3,4,5,6 | 49.5 ± 4.6 | 49.4 ± 4.6 | 59.1 ± 3.7 | 53 / 62 / 58 / 69 |  |  |
| region_w1_shuffled_holdout | 1,2,3,4 | 49.3 ± 1.7 | 49.1 ± 2.1 | 54.6 ± 1.2 | 51 / 58 / 51 / 58 |  |  |
| mix_w5_holdout | 1,2,3,4 | 49.2 ± 2.5 | 48.9 ± 3.0 | 56.0 ± 2.5 | 49 / 58 / 67 / 61 |  |  |
| mix_box_w5_holdout | 1,2,3,4 | 49.0 ± 4.8 | 49.3 ± 4.7 | 56.2 ± 3.3 | 49 / 59 / 65 / 62 |  |  |
| spatial_w5_shuffled_holdout | 1,2,3,4,5,6 | 47.5 ± 2.5 | 47.8 ± 2.7 | 52.8 ± 2.7 | 48 / 56 / 54 / 57 |  |  |
| simclr | 1,2,3,4,5,6 | 46.7 ± 1.9 | 46.4 ± 1.9 | 52.4 ± 1.1 | 48 / 54 / 54 / 63 |  |  |
| xfit_norm_holdout | 1,2 | 46.6 ± 0.5 | 46.5 ± 0.9 | 51.1 ± 1.1 | 46 / 52 / 52 / 64 |  | -0.01 |
| xfit_meaning_w3_holdout | 1,2 | 45.7 ± 3.7 | 45.6 ± 3.9 | 54.0 ± 2.0 | 46 / 54 / 67 / 69 |  | 0.04 |
| f2_shuffled | 1,2,3,4 | 45.5 ± 3.5 | 45.7 ± 3.4 | 51.8 ± 0.7 | 47 / 54 / 53 / 61 | 0.03 | -0.08 |
| mix_shuffled_w5_holdout | 1,2,3,4 | 44.8 ± 3.3 | 44.9 ± 3.3 | 51.6 ± 3.8 | 45 / 52 / 67 / 59 |  |  |
| xfit_meaning_w5_response_residual_holdout | 1,2 | 43.9 ± 6.1 | 44.2 ± 6.8 | 51.5 ± 1.3 | 44 / 54 / 57 / 60 |  | 0.11 |
| xfit_norm_shuffled_holdout | 1,2 | 43.6 ± 2.2 | 43.9 ± 2.1 | 50.3 ± 0.2 | 44 / 51 / 58 / 64 |  | -0.03 |
| xfit_meaning_w10_residual_holdout | 1,2 | 41.7 ± 0.9 | 41.9 ± 1.1 | 55.3 ± 2.2 | 42 / 66 / 61 / 61 |  | 0.00 |
| f2_std | 1,2,3,4 | 41.6 ± 6.1 | 41.3 ± 6.1 | 51.2 ± 1.6 | 41 / 53 / 61 / 67 | 0.10 | -0.05 |
| xfit_meaning_w10_shuffled_holdout | 1,2,3,4 | 41.4 ± 3.0 | 41.2 ± 3.1 | 51.4 ± 2.2 | 41 / 55 / 61 / 66 |  | -0.02 |

## Unfinished runs

- `seed_01/region_w1_shuffled_holdout/24270488`: running
- `seed_01/region_w1_shuffled_holdout/24309457`: running
- `seed_02/region_w1_shuffled_holdout/24270489`: running
- `seed_05/spatial_w5_holdout/24309468`: running
- `seed_06/simclr/24309475`: running
- `seed_06/spatial_w5_holdout/24309469`: running
