# Layer 5 Blocking Report

Every run is kept: `runs` in `l5_blocking_report.json` holds the full history.

| run | split | refs | references | avg K | recall | ratio |
|---|---|---|---|---|---|---|
| `test_all` (latest) | test | all | 1732544 | 6.000250498688634 | n/a | 0.9 |
| `train_train` | train | train | 100000 | 50.0 | n/a | 0.0 |
| `train_val` | train | val | 30000 | 6.879266666666667 | 57.34% | 0.7 |

## Latest run: `test_all`

- countries: `['us', 'india', 'france']` | target K band: `[5.5, 7.0]`
- chosen retention ratio: `0.9` | k_min=6 | k_max=12
- references written: `1,732,544` of `1,732,544` expected | coverage: `{'us': 1.0, 'india': 1.0, 'france': 1.0}`
- candidates kept: `10,395,698` | average K: `6.000`
- reduction ratio: `99.999940%` (candidate density `6.02e-07`)
- retrieval-signal sidecar: `candidates_test_all.parquet`
