# factors_celeba

Validation probe; `last 4` averages the last four periodic probes. ± is the standard deviation over seeds. `val group acc` lists the mean accuracy of every (class, attribute) group in the dataset's group order. `held-out expl. var.` is the factor variance a ridge regression fitted on the other half of the batch explains, which memorizing images cannot raise.

## celeba

| arm | seeds | val WGA (last 4) | val WGA (last) | val acc (last 4) | val group acc | factor expl. var. | held-out expl. var. |
|---|---|---|---|---|---|---|---|
| xfit_norm | 1,2 | 85.9 ± 0.5 | 85.8 ± 1.1 | 88.6 ± 0.3 | 86 / 90 / 93 / 92 |  | 0.07 |
| simclr | 1,2 | 84.9 ± 0.0 | 85.2 ± 0.2 | 87.8 ± 0.5 | 85 / 88 / 94 / 86 |  |  |
| xfit_norm_shuffled | 1,2 | 82.3 ± 3.9 | 82.5 ± 4.0 | 87.7 ± 0.2 | 85 / 88 / 94 / 84 |  | -0.02 |
