# Layer 9 Calibration Report

- variant: `a` | folds: `5` | bins: `15`
- chosen calibration: **global**

| option | ECE | Brier | LogLoss | AUC | macro F0.5 |
|---|---|---|---|---|---|
| none | 0.0068 | 0.0421 | 0.1385 | 0.9884 | 0.7937 |
| global | 0.0004 | 0.0420 | 0.1375 | 0.9883 | 0.8050 |
| per_country | 0.0004 | 0.0420 | 0.1375 | 0.9883 | 0.8050 |

Isotonic regression is monotonic, so it preserves ranking and cannot change the best achievable macro F0.5; calibration is adopted only for probability reliability.
