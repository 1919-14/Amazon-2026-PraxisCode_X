# Layer 10 Decision Report

- variant: `a` | split: `train`
- thresholds: `tau_match=0.2`, `tau_s=0.5`, `margin=0.05`
- tuned (joint grid search): macro F0.5 = `0.8416`
- open-set references (veto): `0`

## Decisions

- references: `100,000`
- non-empty (match): `93,377`
- singletons (empty): `6,623`
- matched pairs: `287,118`
- applied macro F0.5: `0.8875`

The singleton guard keeps the empty prediction when the top candidate score is below `tau_s`; open-set references additionally require a boosted threshold and a minimum top confidence.
