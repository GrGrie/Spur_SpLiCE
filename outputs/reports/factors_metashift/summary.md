# factors_metashift

Validation probe; `last 4` averages the last four periodic probes. ± is the standard deviation over seeds.

## metashift

| arm | seeds | val WGA (last 4) | val WGA (last) | val acc (last 4) | factor expl. var. |
|---|---|---|---|---|---|
| f2_w3 | 1,2 | 51.4 ± 1.5 | 51.4 ± 2.0 | 63.1 ± 2.0 | 0.41 |
| simclr_lt | 1,2 | 51.4 ± 1.5 | 50.0 ± 2.0 | 55.3 ± 0.1 |  |
| f2_std_lt | 1,2 | 48.6 ± 5.9 | 47.9 ± 6.9 | 57.4 ± 0.2 | 0.10 |
| f2_presence | 1,2 | 48.3 ± 7.9 | 49.3 ± 8.8 | 57.1 ± 2.1 | 0.15 |
| simclr | 1,2 | 47.7 ± 4.2 | 47.2 ± 3.9 | 55.8 ± 0.7 |  |
| f2_w10 | 1,2 | 47.0 ± 7.1 | 45.8 ± 3.9 | 59.7 ± 1.4 | 0.85 |
| f2_laion | 1,2 | 45.8 ± 1.0 | 46.5 ± 1.0 | 57.5 ± 0.5 | 0.18 |
| f2_start0 | 1,2 | 45.0 ± 1.7 | 45.1 ± 2.9 | 56.6 ± 0.2 | 0.12 |
| f2_coarse | 1,2 | 43.4 ± 2.5 | 43.8 ± 2.9 | 60.6 ± 2.1 | 0.89 |
| f2_std | 1,2 | 43.2 ± 1.2 | 42.4 ± 1.0 | 58.2 ± 1.9 | 0.12 |
| f2_shuffled | 1,2 | 42.2 ± 4.7 | 41.7 ± 5.9 | 51.0 ± 0.9 | 0.06 |
