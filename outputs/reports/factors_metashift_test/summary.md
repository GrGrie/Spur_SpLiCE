# factors_metashift_test

Locked test: one linear probe on the test split after training (`--final_test`). ± is the standard deviation over seeds; `test group acc` lists every (class, attribute) group in the dataset's order.

## metashift

| arm | seeds | test WGA | test acc | test group acc |
|---|---|---|---|---|
| comp_w5_holdout | 1,2,3,4 | 56.2 ± 4.6 | 65.3 ± 1.8 | 67 / 56 / 61 / 76 |
| spatial_w5_holdout | 1,2,3,4 | 54.2 ± 2.0 | 62.5 ± 1.4 | 69 / 57 / 54 / 70 |
| simclr | 1,2,3,4 | 49.0 ± 3.8 | 60.4 ± 1.7 | 66 / 59 / 50 / 66 |
