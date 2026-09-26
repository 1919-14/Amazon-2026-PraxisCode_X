# Layer 4 RRF Report

Every run is kept: `runs` in `l4_rrf_report.json` holds the full history.

| run | split | refs | best k | references | elapsed (s) |
|---|---|---|---|---|---|
| `test_all` (latest) | test | all |  | 1732544 | 268.4 |
| `train_train` | train | train | 10 | 100000 | 330.38 |
| `train_val` | train | val | 20 | 30000 | 7.57 |

## Latest run: `test_all`

- countries: `['us', 'india', 'france']` | k-grid: `[10, 20, 40, 60]` | best-n: `10` | cutoffs: `[5, 10, 20, 50]`
- references processed: `1,732,544` | elapsed: `268.4s`
- input coverage: `{'us': 1.0, 'india': 1.0, 'france': 1.0}`

