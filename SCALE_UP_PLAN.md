# Scale-Up Plan — 0.788 → 0.98+ (macro F0.5)

Owner: PraxisCode_X · Metric: macro F0.5 per Source-1 entity, singletons credited.
Status: **design accepted, execution starts at P0.**

---

## 0. Diagnosis (measured, not guessed)

### 0.1 The binding constraint is candidate recall, not the matcher

| Quantity (India train, 100k refs, current code) | Value |
|---|---|
| Union candidate recall (channels A/C/D) | 0.9223 |
| RRF recall@5 / @10 / @20 / @50 | 0.724 / 0.808 / 0.842 / **0.878** |
| L5 final candidate recall (K≈6.3) | 0.7636 |
| Oracle macro F0.5 (perfect classifier inside candidates) | 0.9023 |
| Applied macro F0.5 | 0.8050 (= 89% of oracle) |

`candidate_pairs.tsv` is **not scored**, so K is nearly free — the 0.878 recall@50
(not @6) is the relevant number. Even so, the **union recall itself is 92.2%**, which
caps the oracle near 0.92. That is the wall.

### 0.2 What the positives actually look like (1,450 true pairs)

| Signal | Rate |
|---|---|
| name_token_set_ratio ≥ 0.8 | 0.817 |
| house number present & equal | 0.675 |
| name_ratio ≥ 0.8 | 0.650 |
| address_ratio ≥ 0.8 | 0.557 |
| name_ratio ≥ 0.9 | 0.430 |
| exact normalized name | 0.226 |
| exact normalized address | 0.090 |
| **postal code present & equal** | **0.039** |

Noise taxonomy (from real pairs): Devanagari↔Latin transliteration
(`Raj Investments LLP` ↔ `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`), typos
(`Dahlia Ponr`), suffix drift (`Inc` present/absent), address reordering with literal
`null` (`KANSAS CITY, MO, 630 45ND TERRACE, null`), domain-as-name
(`maurewilliamscolombier.com`), corrupted names (`Dréxkor`), accents
(`Lumay Bóral`), house-number suffixes (`1056` vs `1056c` vs `1056-1060`).

Three consequences that invalidate parts of the current design:

1. **Postal codes are unusable** (3.9%). Channel A's `name|postal` key is near-dead;
   `name|house` and house-based keys are the useful exact keys.
2. **No single field exceeds 82% recall**, so recall must come from **fusing name AND
   address** evidence — and the address is the cross-lingual bridge (India addresses
   stay Latin even when the name is Devanagari).
3. **Six state-of-the-art features are missing** from the 27: token_set/sort ratios,
   partial ratio, char-n-gram Jaccard, embedding cosine, and missingness/script
   indicators.

### 0.3 No leak; the target is real

56 teams are above 0.98 → a strong standard approach reaches it. (An earlier
leak hypothesis was tested and rejected: random IDs, no positional/duplicate/overlap
structure — see `experiments/LEAK_RECON/report.md`.)

### 0.4 Target arithmetic

With precision ≈ 1 and per-entity recall `r`, macro F0.5 ≈ `1.25r/(0.25+r)` (upper
bound; singletons reduce it):

| per-entity recall | F0.5 upper bound |
|---|---|
| 0.90 | 0.964 |
| 0.95 | 0.988 |
| 0.97 | 0.994 |

So 0.98 needs **retrieval recall ≈ 0.95–0.97** *and* a matcher with ~0.99 precision on
the retrieved set. Today we have 0.76 recall and 89%-of-ceiling matcher. Both must move.

---

## 1. Architecture v2

```
S1/S2/S3
   │
   ▼  A. Normalization v2  (canonical fields, junk-strip, transliteration flag)
   ▼  B. Blocking v2        (7 channels → RRF, K≈200, block-wise/disk-backed)
   ▼  C. Features v2        (~45 pairwise features incl. semantic + missingness)
   ▼  D. Matcher v2         (LightGBM+CatBoost+XGBoost stack → cross-encoder re-rank)
   ▼  E. Decision v2        (per-country τ + entity-level expected-F0.5 set selection)
   ▼  F. France adapter     (open-set + self-training on test structure)
   ▼  matching_results.tsv
```

### A. Normalization v2 (extend `l2_normalization/`)

- Strip sentinel junk: `null`, `none`, `na`, repeated punctuation.
- Canonical address = order-invariant token multiset keyed by `(house_number_core,
  street_tokens, city, state, country)`; collapse `45ND→45`, `1056-1060→1056`.
- Flag `name_is_domain`, `name_is_transliterated`, `addr_missing`, `name_missing`.
- Keep a **transliteration view**: Devanagari→Latin via `indic-transliteration` (MIT)
  is *allowed* (offline library, not an external lookup) and lifts India recall.

### B. Blocking v2 (extend `l3_l5_blocking/blocked.py`)

Channels, all block-wise and disk-backed, fused with weighted RRF:

| Ch | Key / text | Rationale |
|---|---|---|
| A1 | name_norm + house | high-precision exact key |
| A2 | name_norm + postal | keep, cheap |
| A3 | name_norm | exact |
| C | name token IDF | base lexical |
| D | name char 3–5 gram IDF | typo/abbreviation |
| **E** | **address token IDF** | address-only matches (transliteration) |
| **F** | **address char 3–5 gram IDF** | reordered/typo addresses |
| **G** | **phonetic (double-metaphone) name** | sound-alikes |
| **H** | **dense ANN (FAISS)** name+address, multilingual | semantic/transliteration |

- K raised to **~200** for the matcher (not scored); report a compact blocking set if
  the methodology write-up needs one.
- Gate: **candidate recall ≥ 0.95** (India and US), measured per country.

### C. Features v2 (extend `l6_l8_matching/features.py`)

Per field (name, address): `ratio`, `partial_ratio`, `token_set`, `token_sort`,
`char3_jaccard`, `token_jaccard`, `jaro_winkler`, `levenshtein_norm`, `idf_overlap`,
`embedding_cosine`. Plus sub-field flags: `house_eq`, `street_jaccard`, `city_eq`,
`state_eq`, `postal_eq`, `digit_conflict`, `script_name_eq`, `name_is_domain`,
`addr_missing_xor`, `len_ratios`, `same_country`, retrieval features (RRF, ranks,
agreement, exact-hit). Target ~45 features, all train/serve-parity via the existing
signal sidecar.

### D. Matcher v2 (extend `l6_l8_matching/`, `l11_inference/`)

1. **GBDT stack**: LightGBM + CatBoost + XGBoost, GroupKFold on `s1_id`, OOF.
2. **Cross-encoder re-rank** on the GBDT top-N (N≈20) with
   `BAAI/bge-reranker-v2-m3` (Apache-2.0) or `cross-encoder/mmarco-mMiniLMv2-L12`
   (Apache-2.0). Precision-heavy F0.5 rewards this strongly.
3. **Meta-learner** (logistic regression) over GBDT + cross-encoder scores.

### E. Decision v2 (extend `l10_decision/`)

- Per-country τ_match / τ_s (US, India tunable; France via the pseudo-open-set tuner).
- **Entity-level expected-F0.5 set selection**: using calibrated probabilities, for
  each entity choose the candidate subset maximising expected F0.5 (a *set-level*
  threshold, not a global cut). This directly optimises the scored metric.

### F. France adapter

- France has no train ground truth. Use the open-set policy plus **test-time
  self-training**: block/cluster the France test bucket, derive high-confidence
  pseudo-labels from agreement across channels, and fit a France-specific threshold.

---

## 2. Phased roadmap and gates

| Phase | Deliverable | Gate | Target |
|---|---|---|---|
| **P0** | Reconcile the 0.788 submission; controlled OOF baseline per country | reproduce ±0.005 | 0.79 |
| **P1** | Blocking v2 (channels A1–G, K≈200) | candidate recall ≥ 0.95 | — |
| **P2** | Features v2 + GBDT stack | OOF macro F0.5 ≥ 0.93 | 0.93 |
| **P3** | Decision v2 (per-country + entity-level F0.5) | ≥ 0.95 | 0.95 |
| **P4** | Cross-encoder re-rank + meta-learner | ≥ 0.97 | 0.97 |
| **P5** | Dense channel H + France adapter + full run | private-safe ≥ 0.98 | **0.98+** |

Every phase is measured on held-out OOF, **per country**, never on the public LB.

---

## 3. Compute reality (RTX 4050 6 GB)

- Embeddings: `intfloat/multilingual-e5-small` (MIT, 118 M) or `paraphrase-multilingual-MiniLM-L12-v2` (Apache) — base/large models are too slow for ~12 M texts on 6 GB.
- ANN: FAISS `IVF-PQ` per country bucket (float16), not flat.
- Cross-encoder: only top-N ≈ 20 per entity, batched; `mmarco-mMiniLMv2-L12` fits 6 GB.
- Everything else block-wise and streaming, as the current L3 already is.

## 4. Constraints / integrity

- **No external data lookup** (no geocoding/APIs) — matches the challenge rule. All
  models MIT/Apache-2.0 and ≤ 8 B params.
- No test ground truth: every decision is validated on the train split only.

## 5. Immediate next actions

1. **P0**: reproduce the 0.788 end-to-end; log OOF macro F0.5 per country.
2. **P1a**: add address token/char channels E/F to the blocking engine and measure
   recall@K on India+US.
3. **P1b**: add the phonetic channel G and the `indic-transliteration` view.
