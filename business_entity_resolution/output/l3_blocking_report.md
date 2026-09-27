# Layer 3 Blocking Report

Every run is kept: `runs` in `l3_blocking_report.json` holds the full history.

| run | split | refs | references | peak RSS (MB) | elapsed (s) |
|---|---|---|---|---|---|
| `legacy_previous_run` | test | all | 259452 |  | 1180.41 |
| `test_all` (latest) | test | all | 1069438 | 1332.6 | 6537.21 |
| `train_train` | train | train | 80000 | 2131.8 | 2262.36 |
| `train_val` | train | val | 30000 | 944.6 | 349.94 |

## Latest run: `test_all`

- countries: `['india', 'france']` | topk: `50` | char channel: `True`
- block size: `2,000,000` | reference block: `125,000` | vocab sample: `300,000`
- references processed: `1,069,438` | elapsed: `6537.2s` | peak RSS: `1333 MB`

- **india**: 809,986 references | 4717565 candidates | blocks `3` | union recall `0.00%`
- **france**: 259,452 references | 1434993 candidates | blocks `1` | union recall `0.00%`
