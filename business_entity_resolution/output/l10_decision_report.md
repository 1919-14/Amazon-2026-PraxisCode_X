# Layer 10 Decision Report

- variant: `a` | split: `train`
- thresholds: `tau_match=0.2`, `tau_s=0.4`, `margin=0.05`
- tuned (joint grid search): macro F0.5 = `0.7605`
- open-set references (veto): `0`

## Decisions

- references: `30,000`
- non-empty (match): `25,699`
- singletons (empty): `4,301`
- matched pairs: `54,374`
- applied macro F0.5: `0.7605`

The singleton guard keeps the empty prediction when the top candidate score is below `tau_s`; open-set references additionally require a boosted threshold and a minimum top confidence.
