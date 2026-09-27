# Layer 9 Calibration Report

- variant: `a` | folds: `5` | bins: `15`
- chosen calibration: **none**

| option | ECE | Brier | LogLoss | AUC | macro F0.5 |
|---|---|---|---|---|---|
| none | 0.0040 | 0.0228 | 0.0760 | 0.9953 | 0.8416 |
| global | 0.0001 | 0.0227 | 0.0753 | 0.9953 | 0.8519 |
| per_country | 0.0001 | 0.0227 | 0.0753 | 0.9953 | 0.8519 |

Isotonic regression is monotonic, so it preserves ranking and cannot change the best achievable macro F0.5; calibration is adopted only for probability reliability.
