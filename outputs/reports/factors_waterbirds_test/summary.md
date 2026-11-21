# factors_waterbirds_test

Locked test: one linear probe on the test split after training (`--final_test`). ± is the standard deviation over seeds; `test group acc` lists every (class, attribute) group in the dataset's order.

## waterbirds

| arm | seeds | test WGA | test acc | test group acc |
|---|---|---|---|---|
| comp_w5_holdout | 1,2,3,4,5,6 | 58.7 ± 6.4 | 64.2 ± 3.0 | 59 / 67 / 69 / 69 |
| spatial_w5_holdout | 1,2,3,4,5,6 | 55.6 ± 3.3 | 61.3 ± 1.3 | 56 / 66 / 61 / 65 |
| simclr | 1,2,3,4,5,6 | 47.4 ± 3.1 | 52.0 ± 1.6 | 49 / 51 / 58 / 60 |
