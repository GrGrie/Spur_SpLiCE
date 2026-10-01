# factors_waterbirds

Validation probe; `last 4` averages the last four periodic probes. ± is the standard deviation over seeds. `val group acc` lists the mean accuracy of every (class, attribute) group in the dataset's group order. `held-out expl. var.` is the factor variance a ridge regression fitted on the other half of the batch explains, which memorizing images cannot raise.

## waterbirds

| arm | seeds | val WGA (last 4) | val WGA (last) | val acc (last 4) | val group acc | factor expl. var. | held-out expl. var. |
|---|---|---|---|---|---|---|---|
| xfit_norm_holdout | 1,2 | 46.6 ± 0.5 | 46.5 ± 0.9 | 51.1 ± 1.1 | 46 / 52 / 52 / 64 |  | -0.01 |
| simclr | 1,2,3,4 | 45.8 ± 1.7 | 45.5 ± 1.7 | 52.3 ± 1.2 | 47 / 54 / 52 / 64 |  |  |
| f2_shuffled | 1,2,3,4 | 45.5 ± 3.5 | 45.7 ± 3.4 | 51.8 ± 0.7 | 47 / 54 / 53 / 61 | 0.03 | -0.08 |
| xfit_norm_shuffled_holdout | 1,2 | 43.6 ± 2.2 | 43.9 ± 2.1 | 50.3 ± 0.2 | 44 / 51 / 58 / 64 |  | -0.03 |
| f2_std | 1,2,3,4 | 41.6 ± 6.1 | 41.3 ± 6.1 | 51.2 ± 1.6 | 41 / 53 / 61 / 67 | 0.10 | -0.05 |
