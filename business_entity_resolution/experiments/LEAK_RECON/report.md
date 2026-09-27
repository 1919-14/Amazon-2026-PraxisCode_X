# LEAK-RECON — Is the 0.99 leaderboard cluster a recoverable rule?

**Motivation.** We submitted `matching_results.tsv` and scored **0.788** (public).
The top three public entries are `0.990816` / `0.990788` / `0.990675` — three
independent teams within **0.00014**. Independent ML pipelines do not cluster to
four decimals, which suggested a deterministic, recoverable structure in the
data rather than model quality. This experiment tests that hypothesis directly.

**Verdict: no simple structural leak exists.** The hypothesis is rejected. The
0.99 cluster is *not* explained by any of the obvious dataset artifacts.

## Hypotheses and results

| # | Hypothesis | Method | Result |
|---|---|---|---|
| H1 | Exact-duplicate injection (byte/normalized twin) | 100k S1 sample; exact normalized name and (name,address) twins in S2/S3 | 71% of S1 have a name twin, **but** vs train GT: recall 0.72, **precision 0.077**, avg 14.5 twins/entity, 34% of singletons also matched → **rejected** |
| H2 | ID arithmetic (matched ids share tail/prefix/modulus) | 50k true pairs | last-3 0.0010, last-2 0.0104, first-3 0.0009 — all at the random baseline → **rejected** |
| H3 | Positional alignment (same row index across sources) | 50k rows | 0 / 6,831 → **rejected** |
| H4 | S2 and S3 of one entity share a numeric id | 241,662 entities with both | 0 → **rejected** |
| H5 | Ids are dense sequences | ranges over 1M rows/source | random 9-digit space (min≈10², max≈10⁹), no duplicates → **rejected** |
| H6 | Fixed rank offset / correlation S1↔S2 | 13,203 aligned pairs | median offset −284, `|diff|<50` 0.04% → **rejected** |
| H7 | Train/test record overlap | S1 (name,addr,country) sets | **exactly 0** shared records → **rejected** |

## Metric confirmation

The official `student_resource/README.md` confirms: **macro F0.5 computed per
Source-1 entity and averaged over all entities, singletons included** (empty on a
singleton = 1.0, any prediction = 0.0). The **public leaderboard is only a subset
of the test set**; the private leaderboard is the rest. `candidate_pairs.tsv` is
**not scored**.

## What this means for the 0.20 gap

With no leak to exploit, the gap must come from the pipeline itself, and the
binding constraint is **candidate recall**, not the matcher:

| Quantity (India train, 100k refs, EXP-12A) | Value |
|---|---|
| Union candidate recall (all channels) | 0.9223 |
| RRF recall@5 / @10 / @20 / @50 | 0.724 / 0.808 / 0.842 / 0.878 |
| L5 final candidate recall (K≈6.3) | 0.7636 |
| Oracle macro F0.5 (perfect classifier inside candidates) | 0.9023 |
| Applied macro F0.5 | 0.8050 (= 89.2% of oracle) |

Two independent caps: (1) the union recall itself is 92.2%, so **even perfect
retrieval ranking caps the oracle near ~0.92**; (2) our matcher captures only
~89% of whatever ceiling exists. Reaching 0.99 requires ~99% recall *and* a
near-perfect matcher simultaneously.

## Recommendation

1. **Do not chase the 0.99 number blindly.** Treat it as unverified: it may be a
   small-public-subset artifact, leaderboard overfitting, or a genuinely strong
   method we have not identified. Any of these changes the strategy.
2. **Attack the recall ceiling first** (it is the cap, and the matcher can only
   lose from there): address-text retrieval is already in (EXP-12A); next are
   higher K (candidate_pairs is not scored), a phonetic channel (12a), and dense
   multilingual retrieval (12b) to lift recall@K, not just the union.
3. **Then the matcher**: semantic features, cross-encoder re-ranking, ensembling.
4. **Cheap parallel win**: per-country decision thresholds (US/India tunable;
   France only via the pseudo-open-set tuner).

## Reproduce

```bash
venv/Scripts/python.exe business_entity_resolution/experiments/LEAK_RECON/probe.py
venv/Scripts/python.exe business_entity_resolution/experiments/LEAK_RECON/probe2.py
```
