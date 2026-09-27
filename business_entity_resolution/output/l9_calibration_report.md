# Layer 9 Calibration Report

- variant: `a` | folds: `5` | bins: `15`
- chosen calibration: **none**

| option | ECE | Brier | LogLoss | AUC | macro F0.5 |
|---|---|---|---|---|---|
| none | 0.0092 | 0.0244 | 0.0891 | 0.9954 | 0.8772 |
| global | 0.0001 | 0.0227 | 0.0752 | 0.9953 | 0.8517 |
| per_country | 0.0001 | 0.0227 | 0.0752 | 0.9953 | 0.8517 |

Isotonic regression is monotonic, so it preserves ranking and cannot change the best achievable macro F0.5; calibration is adopted only for probability reliability.
