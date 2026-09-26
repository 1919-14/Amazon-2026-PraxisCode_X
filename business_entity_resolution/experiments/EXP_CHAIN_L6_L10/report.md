# EXP-CHAIN-L6-L10 — Grouped-OOF Macro F0.5

**Scope:** India training pool, 100,000 reference sample (seed 42), candidate set
from the addr-text blocking of EXP-12A (avg K = 6.287).
**Protocol:** `GroupKFold` on `source1_entity_id` (5 folds), out-of-fold
probabilities, thresholds grid-searched on OOF.

## Headline

| Metric | Value |
|---|---|
| Candidate recall (micro, over true pairs) | **76.36%** |
| Oracle macro F0.5 (perfect classifier within candidates) | **0.9023** |
| L8 Variant A OOF macro F0.5 (raw) | 0.7937 |
| L9 calibrated / L10 applied macro F0.5 | **0.8050** |
| Singleton F0.5 | 0.7834 |
| Non-singleton F0.5 | 0.8062 |

Thresholds: `tau_match = 0.2`, `tau_s = 0.5`, `margin = 0.05`.
Calibration: global isotonic (ECE 0.0068 → 0.0004).
Model: LightGBM, 27 features, 1.10 M variant-A pairs (264,324 pos / 292,006 hard /
547,554 easy), OOF AUC 0.996.

## Leakage audit — Variant B is not a model score

Variant B ("no hard negatives") initially reported **0.9023** with OOF AUC
**1.0000**. Per the brief's rule, ≥0.90 was audited before reporting.

Finding: variant B contains only positives and easy negatives, and L8 evaluates on
`neg_type != "easy"` — so its decision set is **positives only**. Every scored
candidate is a true match, precision is trivially 1.0, and the score measures
candidate recall rather than the matcher.

Decisive confirmation: variant B's 0.9023 is **exactly** the independently computed
oracle macro F0.5 (predict every true candidate). It is the ceiling, not a result.

Guard added in `main_l8.py`: a variant with zero hard negatives in its decision set
is marked `comparable: false` and excluded from winner selection. The valid
headline is therefore **Variant A = 0.7937 raw / 0.8050 calibrated**.

## What the numbers say

- Candidate recall (76.36%) is the binding constraint. Perfect classification
  within the candidates would score 0.9023; the system achieves 0.8050, i.e.
  **89.2% of the achievable ceiling**. The remaining loss is split between
  missing candidates and matcher/decision errors.
- Singleton handling is weaker than matching (0.7834 vs 0.8062) — the singleton
  guard trades some false-empty penalty against false-match penalty.

## Honest caveats

- **India only**, 100k of 706,819 India train references. Not yet confirmed on US.
- Thresholds are tuned on the same OOF scores used to report the metric (mild
  optimism; a nested split would be stricter).
- This is **not** a test score. There is no test ground truth; no test F0.5 is
  computed or claimed.
- Repo-recorded baselines (all-empty 0.0562, uncalibrated pool K≈5.5 0.6305,
  no-singleton-filter ceiling 0.9438) were measured on a different validation
  set, so the comparison is indicative, not controlled.

## Reproduce

```bash
cd business_entity_resolution/src
python main_l6.py --candidates ../output/candidate_pairs_train.tsv
python main_l7.py --pairs ../artifacts/train_pairs/variant_a.parquet --signals-refs train
python main_l7.py --pairs ../artifacts/train_pairs/variant_b.parquet --signals-refs train
python main_l8.py
python main_l9.py --variant a
python main_l10.py --variant a --split train --use-calibrated
```
