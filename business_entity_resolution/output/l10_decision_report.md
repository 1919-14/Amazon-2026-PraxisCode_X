# Layer 10 Decision Report

- variant: `a` | split: `test`
- thresholds: `tau_match=0.5`, `tau_s=0.1`, `margin=0.05`
- open-set references (veto): `244,182`

## Decisions

- references: `1,617,161`
- non-empty (match): `1,588,601`
- singletons (empty): `28,560`
- matched pairs: `4,571,473`

The singleton guard keeps the empty prediction when the top candidate score is below `tau_s`; open-set references additionally require a boosted threshold and a minimum top confidence.
