# Amazon ML Challenge 2026: Project Walkthrough & Execution Log 📖

**Team Name:** PraxisCode_X  
**Challenge Track:** Business Entity Resolution  
**Repository:** [1919-14/Amazon-2026-PraxisCode_X](https://github.com/1919-14/Amazon-2026-PraxisCode_X.git)  

---

## 🧭 Overview & Progress Tracker

This document provides a continuous, step-by-step walkthrough of our 14-layer machine learning pipeline. It logs all design decisions, schema contracts, verification results, and exact run commands.

```
[Phase 0: Research & Profiling] ──► [L0: Setup & Audit] ──► [L1: Validation Split] ──► [L2: Normalizer]
            ✅ COMPLETED                    ✅ COMPLETED            ✅ COMPLETED               ⏳ UPCOMING
```

---

## 🚀 Layer 0: Setup, Config, Data Verification (COMPLETED ✅)

### 1. What Was Built
We established the core foundation layer to guarantee data integrity, schema consistency, and zero data leakage:

* [`src/config.py`](business_entity_resolution/src/config.py): Global constants, dynamic dataset path resolution, random seed (`SEED = 42`), and empty grid placeholders for later layers.
* [`src/schemas.py`](business_entity_resolution/src/schemas.py): Type-hinted dataclass contracts (`RawRecord`, `NormRecord`, `CandidateMeta`) preventing schema bugs from propagating.
* [`src/ingest.py`](business_entity_resolution/src/ingest.py): TSV loaders reading all columns as `str` (preserving leading zeros in postal codes), replacing `NaN` with empty strings, asserting exact column headers and entity ID prefixes (`S1-`, `S2-`, `S3-`).
* [`src/audit.py`](business_entity_resolution/src/audit.py): Streaming dataset auditor profiling row counts, missing rates, ground-truth match cardinalities, and verifying pairwise country consistency.
* [`src/utils/cache.py`](business_entity_resolution/src/utils/cache.py): Fast Parquet serialization (`snappy` compression) and artifact subdirectories setup.
* [`src/main_l0.py`](business_entity_resolution/src/main_l0.py): Master Layer 0 entry point.

---

### 2. Execution & Verification Log

#### Command Executed:
```bash
venv\Scripts\python.exe business_entity_resolution/src/main_l0.py
```

#### Verification Checklist Output:
```
============================================================
📋 LAYER 0 VERIFICATION CHECKLIST REPORT
============================================================
1. Row Counts:
   • Train S1: 2,206,821 | S2: 5,034,616 | S3: 5,285,603
   • Test  S1: 1,732,544 | S2: 4,887,273 | S3: 5,082,316

2. Singleton Rate (0 true matches in Train GT): 5.58% (123,247 records)

3. Ground Truth Country Mismatch Rate: 0.0000% (0 / 7,638,365)
   ✅ Country consistency check PASSED (clean partition).

4. Columns with > 50% Missing Values: NONE 
   (All columns well-populated, max missing ~3.3% on S2/S3 address)

5. France in Test but NOT Train: True 
   (Train: ['India', 'US'] | Test: ['France', 'India', 'US'])
============================================================
```

#### Key Discoveries & Strategic Takeaways:
1. **Zero Country Mismatches**: 100% of the 7.64M true matching pairs share the exact same country label. Hard country-partitioned blocking is mathematically safe and eliminates $60-70\%$ of comparisons with zero recall loss.
2. **5.58% Singletons**: 123,247 Source 1 entities have 0 matches. Correctly predicting empty earns a full $1.0$ score, while false merges drop the score to $0.0$.
3. **Open-Set France Challenge**: France appears ONLY in the test set (259k entities). Layer 2 normalizer and Layer 10 decision engine must natively support French address and business structures.

---

---

## ✅ Layer 1: Validation Split & Macro F₀.₅ Scorer (COMPLETED)

```
[Phase 0] ──► [L0] ──► [L1] ──► [L2]
  ✅ Done       ✅ Done  ✅ Done    ⏳ Upcoming
```

### 1. What Was Built

| File | Purpose |
|---|---|
| [`src/l1_validation/__init__.py`](business_entity_resolution/src/l1_validation/__init__.py) | Package marker |
| [`src/l1_validation/split_generator.py`](business_entity_resolution/src/l1_validation/split_generator.py) | Grouped 80/20 split; saves JSON artifacts |
| [`src/l1_validation/metrics.py`](business_entity_resolution/src/l1_validation/metrics.py) | `f_beta`, `precision_recall`, `macro_f05`, `evaluate_with_thresholds` |
| [`src/l1_validation/unit_tests.py`](business_entity_resolution/src/l1_validation/unit_tests.py) | 8 required unit tests |
| [`src/main_l1.py`](business_entity_resolution/src/main_l1.py) | Pipeline entry point |

### 2. Execution Command

```bash
venv\Scripts\python.exe business_entity_resolution/src/main_l1.py
```

### 3. Layer 1 Results

| Metric | Value |
|---|---|
| Total S1 entities | 2,206,821 |
| Train ID count | 1,765,457 (80.00%) |
| Val ID count | 441,364 (20.00%) |
| Actual val fraction | 0.200000 |
| Singleton rate (Full GT) | 5.58% (123,247 entities) |
| Singleton rate (Val GT) | 5.62% (24,809 entities) |
| Avg true matches / non-singleton | **3.67** |

#### Baseline F₀.₅ Scores (Val Set)

| Baseline | F₀.₅ | Interpretation |
|---|---|---|
| All-Empty Predictions | **0.0562** | Minimum viable: correct only for singletons |
| All-Match Dummy Constant | **0.0000** | Worst case: every entity gets a wrong match |
| All-Match (No singleton filter) | **0.9438** | Ceiling with 100% recall but 0 singleton credit |
| Uncalibrated Pool (K≈5.5) | **0.6305** | Target to beat with precision/singleton control |

> **Implication**: Our model must push from **0.6305 → 0.90+** by:
> 1. Precision filtering to remove ~1.8 false candidates/entity
> 2. Singleton guard so $\tau_s$ veto earns 5.62% free points

#### Unit Test Results

```
============================================================
🧪 RUNNING LAYER 1 UNIT TESTS
============================================================
  ✅ PASS: Test 1 — Exact match returns 1.0
  ✅ PASS: Test 2 — Singleton correctly empty returns 1.0
  ✅ PASS: Test 3 — Singleton with false match returns 0.0
  ✅ PASS: Test 4 — Problem statement worked example
  ✅ PASS: Test 5 — Empty gt and empty pred do not crash
  ✅ PASS: Test 6 — Mixed singleton and match
  ✅ PASS: Test 7 — evaluate_with_thresholds basic match
  ✅ PASS: Test 8 — evaluate_with_thresholds singleton path
============================================================
🎉 ALL 8 UNIT TESTS PASSED!
```

#### Memory Usage
- Peak RAM: **1,849 MB** (Ground truth dict with 7.64M pairs held in memory)
- Split generator itself uses ~200 MB (chunked ID read + 2.2M string list)

### 4. Key Decisions

1. **No country stratification in split**: France is test-only. Stratifying would only affect US/India and give false coverage confidence. Pure random grouped split by S1 ID.
2. **JSON over Parquet for ID lists**: Simple, human-readable, zero dependency overhead for ~2.2M string IDs.
3. **`itertuples` in `parse_ground_truth`**: ~30× faster than `iterrows`, critical for 2.2M-row GT loading.

---

---

## ✅ Layer 2: Normalization Engine (COMPLETED)

```
[Phase 0] ──► [L0] ──► [L1] ──► [L2] ──► [L3-L5]
  ✅ Done       ✅ Done  ✅ Done  ✅ Done    ⏳ Upcoming
```

### 1. What Was Built

Layer 2 transforms raw, noisy multi-source records into clean, structured, comparable representations across US, India, and France under strict memory constraints (chunk size 100K rows, streaming to Parquet shards):

| Module | File | Purpose |
|---|---|---|
| Package Init | [`src/l2_normalization/__init__.py`](business_entity_resolution/src/l2_normalization/__init__.py) | Package marker |
| Unicode & Script Utils | [`src/l2_normalization/text_utils.py`](business_entity_resolution/src/l2_normalization/text_utils.py) | NFKC normalization, Latin accent stripping (`réseau` $\rightarrow$ `reseau`), punctuation normalizer (`&` $\rightarrow$ `and`), tokenization, digit run extraction, script detection (Devanagari, Tamil, Telugu, Kannada, Latin) |
| Country Abbreviations | [`src/l2_normalization/abbreviations.py`](business_entity_resolution/src/l2_normalization/abbreviations.py) | Country-aware abbreviation expansion (`ABBREV_US`, `ABBREV_INDIA`, `ABBREV_FRANCE`) & multi-word legal suffix extractor (`extract_legal_suffix` with 5-token greedy search) |
| Address Parser | [`src/l2_normalization/address_parser.py`](business_entity_resolution/src/l2_normalization/address_parser.py) | Regex postal code parser (US 5/9-digit, India 6-digit, France 5-digit), house number parser, state/region matcher, state/European-aware city parser |
| Phonetic Encoder | [`src/l2_normalization/phonetic.py`](business_entity_resolution/src/l2_normalization/phonetic.py) | Latin-only Metaphone encoding via `jellyfish.metaphone` for tokens $\ge 3$ chars |
| Business Stopwords | [`src/l2_normalization/stopwords.py`](business_entity_resolution/src/l2_normalization/stopwords.py) | High-IDF generic term filtering (`remove_stopwords`) for `name_core` |
| Normalization Engine | [`src/l2_normalization/engine.py`](business_entity_resolution/src/l2_normalization/engine.py) | `normalize_record` (full schema mapper) & `normalize_chunk` (100K-row DataFrame transformer) |
| Streaming Pipeline | [`src/l2_normalization/pipeline.py`](business_entity_resolution/src/l2_normalization/pipeline.py) | `process_source` & `process_split` streaming TSVs $\rightarrow$ snappy compressed Parquet shards |
| Entry Point & Verification | [`src/main_l2.py`](business_entity_resolution/src/main_l2.py) | Master runner, sanity spot-checks, NaN assertion, summary exporter |

---

### 2. Execution Command

```bash
venv\Scripts\python.exe business_entity_resolution/src/main_l2.py
```

---

### 3. Layer 2 Results & Summary

#### Shard & Record Breakdown:

| Split | Source | Input File | Shards Written | Rows Processed |
|---|---|---|---|---|
| **Train** | **S1** | `train_source1.tsv` | 23 | 2,206,821 |
| **Train** | **S2** | `train_source2.tsv` | 51 | 5,034,616 |
| **Train** | **S3** | `train_source3.tsv` | 53 | 5,285,603 |
| **Test** | **S1** | `test_source1.tsv` | 18 | 1,732,544 |
| **Test** | **S2** | `test_source2.tsv` | 49 | 4,887,273 |
| **Test** | **S3** | `test_source3.tsv` | 51 | 5,082,316 |
| **TOTAL** | | | **194 shards** | **24,229,173 records** |

- **Peak RAM Usage**: **0.61 GB** (well below 2.0 GB hard limit; under 500 MB per chunk)
- **Zero NaN check**: **PASS** across all parquet columns
- **Compression**: Snappy Parquet (fast I/O for L3 blocking)

---

### 4. Sanity Verification Highlights

```
=================================================================
🔍 SANITY CHECKS REPORT
=================================================================
  ✅ 'Acme Corp' vs 'ACME CORPORATION' → Same name_core? YES ('acme')
  ✅ 'Réseau Hôtel Naïve' → Accents stripped? YES ('reseau hotel naive')
  ✅ France SARL → Suffix extracted? YES ('societe a responsabilite limitee' stripped → core: 'boulangerie dupont')
  ✅ Devanagari Record → script_type: 'devanagari' | name_phonetic: [] (Guarded)
  ✅ US Multi-Digit Address → house: '17560', postal: '78759', state: 'tx', city: 'austin'
  ✅ India Address → postal: '560001', city: 'bangalore', legal_suffix: 'private limited'
  ✅ France Address → postal: '75001', city: 'paris', state: 'paris'
=================================================================
```

---

### 5. Key Architecture & Design Decisions

1. **Greedy Multi-Word Legal Suffix Extraction**: Checks up to 5 trailing tokens joined before single tokens, preventing sub-token truncation (e.g. `limited liability company` takes precedence over `company`, and `societe a responsabilite limitee` takes precedence over `societe`).
2. **Trailing-Match Postal Parser**: Uses `re.findall(pattern, addr)[-1]` to reliably capture the true postal code even when an address begins with a 5-digit house number (e.g. `17560 Ellis Road, Austin, TX 78759`).
3. **European vs US/India City Context Awareness**: Recognizes that French addresses place the city after the 5-digit postal code (`75001 Paris`), whereas US/India addresses place the city before the state/postal code (`Austin, TX 78759`).
4. **Indic Script Protection**: Non-Latin scripts (Devanagari, Tamil, Telugu, Kannada) bypass accent stripping and phonetic encoding to avoid corrupting Unicode characters.
5. **Separation of `name_norm` and `name_core`**: `name_norm` retains all words with abbreviations expanded for exact/soft matching, while `name_core` strips legal suffixes and generic business stopwords for high-recall inverted index blocking in Layer 3.

---

## ✅ Layer 3: Country-Stratified Multi-Channel Blocking (COMPLETED)

```
[Phase 0] ──► [L0] ──► [L1] ──► [L2] ──► [L3] ──► [L4-L5]
  ✅ Done      ✅ Done  ✅ Done  ✅ Done  ✅ Done    ⏳ Upcoming
```

### 1. What Was Built

| Module | File | Purpose |
|---|---|---|
| Package Init | [`src/l3_l5_blocking/__init__.py`](business_entity_resolution/src/l3_l5_blocking/__init__.py) | Package marker + layer map |
| Buckets (L3a) | [`src/l3_l5_blocking/buckets.py`](business_entity_resolution/src/l3_l5_blocking/buckets.py) | Country bucket assignment + normalized parquet loaders (reference / candidate) |
| Channels (L3b-d) | [`src/l3_l5_blocking/channels.py`](business_entity_resolution/src/l3_l5_blocking/channels.py) | `ExactKeyChannel` (A), `SparseTfidfChannel` for word (C) and char 2-4 gram (D) retrieval |
| Engine | [`src/l3_l5_blocking/engine.py`](business_entity_resolution/src/l3_l5_blocking/engine.py) | Per-country index build + batched query, returns candidate `entity_id` lists |
| Recall (L3e) | [`src/l3_l5_blocking/recall.py`](business_entity_resolution/src/l3_l5_blocking/recall.py) | Per-channel + union recall, any-hit coverage, singleton contamination |
| Unit Tests | [`src/l3_l5_blocking/unit_tests.py`](business_entity_resolution/src/l3_l5_blocking/unit_tests.py) | 8 tests (channels A/C/D, country mapping, recall math) |
| Entry Point | [`src/main_l3.py`](business_entity_resolution/src/main_l3.py) | Orchestrates country buckets, writes artifacts + report |

### 2. Execution Command

```bash
# Full validation run (needs enough free RAM for one country bucket at a time)
venv\Scripts\python.exe business_entity_resolution/src/main_l3.py

# Fast smoke test
venv\Scripts\python.exe business_entity_resolution/src/main_l3.py --countries us --max-candidates 50000 --max-refs 2000

# Test-split candidate generation (no ground truth → no recall)
venv\Scripts\python.exe business_entity_resolution/src/main_l3.py --split test --refs all
```

### 3. Design Decisions

1. **Country-serial processing**: one bucket is indexed and queried at a time, then released, so peak memory scales with the largest bucket rather than the full 24M-record corpus.
2. **Channel A specificity ordering**: `name|postal` → `name|house` → `name`, stopping at the first key that matches, with a `max_posting` cap to suppress over-common exact keys.
3. **Channels C/D are recall-first sparse TF-IDF**: `max_df` prunes generic terms/grams (the main memory and noise source), `min_df` prunes hapax terms. Retrieval reuses the fitted vocabulary so query and candidate vectors share a space.
4. **Open-set country handling**: unknown country labels pass through as their own bucket instead of being dropped, keeping the France blind spot and any unseen label valid.

### 4. L3e Recall Validation (real data)

A 15,000-reference US validation sample was scored against a pool of its 55,001 true matches embedded in 150,000 background candidates (top-k = 50):

| Channel | Micro Recall | Any-Hit Recall | Avg Candidates |
|---|---|---|---|
| A (exact key) | 40.20% | 85.32% | 1.9 |
| C (token IDF) | 86.37% | 98.37% | 48.2 |
| D (char 2-4 gram TF-IDF) | 88.40% | 98.73% | 50.0 |
| **UNION** | **92.58%** | **99.22%** | 73.5 |

> Precision is intentionally left to L4 (RRF) and L5 (adaptive truncation); L3 only guarantees the recall ceiling.

---

## ✅ Layer 4: Reciprocal Rank Fusion (COMPLETED)

```
[Phase 0] ──► [L0] ──► [L1] ──► [L2] ──► [L3] ──► [L4] ──► [L5]
  ✅ Done      ✅ Done  ✅ Done  ✅ Done  ✅ Done  ✅ Done    ⏳ Upcoming
```

### 1. What Was Built

| Module | File | Purpose |
|---|---|---|
| RRF core | [`src/l3_l5_blocking/rrf.py`](business_entity_resolution/src/l3_l5_blocking/rrf.py) | `rrf_scores`, `fuse`, `ranked_ids` — weighted RRF with deterministic tie-breaking |
| Ranking recall | [`src/l3_l5_blocking/recall.py`](business_entity_resolution/src/l3_l5_blocking/recall.py) | Added `ranked_recall_counts` / `ranked_recall_at` / `mean_first_hit_rank` |
| Entry Point | [`src/main_l4.py`](business_entity_resolution/src/main_l4.py) | Reads L3 artifacts, tunes `k`, writes fused `l4_*.parquet` + report |

### 2. Execution Command

```bash
# Fuse + tune k on validation (requires L3 artifacts from main_l3.py)
venv\Scripts\python.exe business_entity_resolution/src/main_l4.py

# Fix a single k (no grid search)
venv\Scripts\python.exe business_entity_resolution/src/main_l4.py --k-grid 60

# Test split (no ground truth → fixed k)
venv\Scripts\python.exe business_entity_resolution/src/main_l4.py --split test --refs all
```

### 3. How It Works

$$
\text{RRF}(c) = \sum_{\text{channel}} \frac{w_{\text{channel}}}{k + \text{rank}_{\text{channel}}(c)}
$$

- Fusion only ever **reorders the union** of the channels — a true match is never dropped, so RRF recall equals the L3 union recall (~92.6% on the real-data check). The benefit is **rank quality**: true matches rise toward the top so that L5 can truncate aggressively without losing recall.
- `k` is tuned over `RRF_K_GRID = [10, 20, 40, 60]` using micro `recall@N` (default N = 10, the target $K$ range) on the validation split; ties break toward the smaller `k`.
- Output artifacts (`artifacts/blocking/l4_<split>_<refs>_country=<c>.parquet`) carry the fused order plus fused scores, ready for L5 coarse truncation.

### 4. Verification

- 13/13 package unit tests pass (5 new RRF tests: fusion order, deterministic tie-break, channel weights/zero-weight, batch fusion, ranked recall@N).
- Capped L3→L4 integration run confirmed the fused artifact is written and `recall@50` equals the L3 union recall (sanity: fusion loses nothing).

---

## ✅ Layer 5: Adaptive Candidate Truncation (COMPLETED)

```
[Phase 0] ──► [L0] ──► [L1] ──► [L2] ──► [L3] ──► [L4] ──► [L5] ──► [L6]
  ✅ Done      ✅ Done  ✅ Done  ✅ Done  ✅ Done  ✅ Done  ✅ Done    ⏳ Upcoming
```

### 1. What Was Built

| Module | File | Purpose |
|---|---|---|
| Truncation core | [`src/l3_l5_blocking/truncate.py`](business_entity_resolution/src/l3_l5_blocking/truncate.py) | `coarse_score` (RRF + agreement + exact-key) and `adaptive_truncate` (ratio threshold + bounds) |
| Entry Point | [`src/main_l5.py`](business_entity_resolution/src/main_l5.py) | Tunes ratio, writes `candidate_pairs.tsv` + `blocking_report.md` |

### 2. Execution Commands

```bash
# Validation run (tunes ratio; dev output)
venv\Scripts\python.exe business_entity_resolution/src/main_l5.py

# Test split (fixed ratio; writes the judged deliverable)
venv\Scripts\python.exe business_entity_resolution/src/main_l5.py --split test --refs all

# Fixed ratio, no tuning
venv\Scripts\python.exe business_entity_resolution/src/main_l5.py --no-tune --ratio 0.5
```

### 3. How It Works

**Stage 1 — coarse scoring** (cheap, no pairwise string work):

$$
\text{coarse}(c) = 0.6 \cdot \frac{\text{RRF}(c)}{\max\text{RRF}} + 0.25 \cdot \frac{\text{agreement}(c)}{3} + 0.15 \cdot \mathbb{1}[c \in \text{Channel A}]
$$

**Stage 2 — adaptive truncation**: keep candidates retaining at least `ratio` of the reference's top coarse score, then clamp to `[k_min, k_max]` (default `[1, 12]`). The `ratio` is tuned over `[0.3, 0.5, 0.7, 0.9]` so the average budget lands in the challenge's compactness band `K ∈ [5.5, 7.0]`, maximising recall within that band.

### 4. Deliverables

| Output | Description |
|---|---|
| `output/candidate_pairs.tsv` | Judged blocking deliverable (`source1_entity_id`, `candidate_entity_ids`), tab-separated, one row per S1 entity |
| `output/candidate_pairs_val.tsv` | Same format, for the train validation split during tuning |
| `output/blocking_report.md` / `l5_blocking_report.json` | Recall + average K + reduction ratio report |

### 5. Verification

- 17/17 package unit tests pass (4 new L5 tests: coarse scoring, fallback, ratio truncation, bounds).
- Capped L3→L4→L5 integration run produced avg **K = 6.44** (inside target band), wrote a valid tab-separated `candidate_pairs_val.tsv` (2000 refs + header), and emitted the blocking report.

---

## ✅ Layer 6: Training Pair Construction & Hard-Negative A/B Datasets (COMPLETED)

```
[Phase 0] ──► [L0] ──► [L1] ──► [L2] ──► [L3] ──► [L4] ──► [L5] ──► [L6] ──► [L7]
  ✅ Done      ✅ Done  ✅ Done  ✅ Done  ✅ Done  ✅ Done  ✅ Done  ✅ Done    ⏳ Upcoming
```

### 1. What Was Built

| Module | File | Purpose |
|---|---|---|
| Pair sampler | [`src/l6_l8_matching/pairs.py`](business_entity_resolution/src/l6_l8_matching/pairs.py) | `sample_reference_pairs` (pos / hard / easy), candidate-TSV reader, reservoir easy pool, `variant_b_rows`, `summarize` |
| Unit Tests | [`src/l6_l8_matching/unit_tests.py`](business_entity_resolution/src/l6_l8_matching/unit_tests.py) | 5 tests (TSV parsing, ratios, singletons, easy exclusion, Variant B) |
| Entry Point | [`src/main_l6.py`](business_entity_resolution/src/main_l6.py) | Streams pairs to Parquet (Variant A + B) and writes the sampling report |

### 2. Execution Command

```bash
# Build pairs from a train-split candidate set (L5 --refs train)
venv\Scripts\python.exe business_entity_resolution/src/main_l6.py \
    --candidates output/candidate_pairs_train.tsv

# Small dev run
venv\Scripts\python.exe business_entity_resolution/src/main_l6.py \
    --candidates output/candidate_pairs_train.tsv --max-references 5000
```

### 3. Sampling Design

For each Source 1 reference (`candidate_ids` assumed ranked best-first):
- **Positives** = candidates ∩ ground truth.
- **Hard negatives** = the highest-ranked candidates that are *not* true matches (confusable decoys).
- **Easy negatives** = random same-country S2/S3 records that were neither retrieved nor true (reservoir-sampled pool).

Ratio per reference: **1 positive : 3 hard : 2 easy** (configurable via `L6_POS_HARD_RATIO` / `L6_POS_EASY_RATIO`). Singletons (no true match) receive a fixed `3 hard : 2 easy` so the model learns to predict the empty list — essential for the macro F₀.₅ singleton credit.

### 4. Deliverables

| Output | Description |
|---|---|
| `artifacts/train_pairs/variant_a.parquet` | positives + hard negatives + easy negatives |
| `artifacts/train_pairs/variant_b.parquet` | positives + easy negatives (no hard negatives) |
| `output/l6_pairs_report.json` | Pair counts, achieved ratios, per-country breakdown |

Schema: `s1_id, cand_id, label(int8), neg_type(pos/hard/easy), country, rank(int32)`.

### 5. Verification

- 5/5 L6 unit tests pass.
- Streaming Parquet writer verified (correct schema + dtypes).

> ℹ️ **The model-based A/B test** (train Variant A vs Variant B, compare validation $F_{0.5}$) runs in **L8**, because it requires the L7 pairwise features and the matcher. L6 produces both datasets and the sampling diagnostics that drive it.

---

## ✅ Layer 7: Pairwise Feature Engineering (COMPLETED)

```
[L3] ──► [L4] ──► [L5] ──► [L6] ──► [L7] ──► [L8]
 ✅      ✅      ✅      ✅      ✅      ⏳ Upcoming
```

### 1. What Was Built

| Module | File | Purpose |
|---|---|---|
| Feature functions | [`src/l6_l8_matching/features.py`](business_entity_resolution/src/l6_l8_matching/features.py) | `compute_features` + 4 block functions + `build_idf`; `FEATURE_NAMES` (27) |
| Unit Tests | [`src/l6_l8_matching/unit_tests.py`](business_entity_resolution/src/l6_l8_matching/unit_tests.py) | 5 L7 tests (shape, identical vs different, IDF, retrieval ranks) |
| Entry Point | [`src/main_l7.py`](business_entity_resolution/src/main_l7.py) | Joins L6 pairs with normalized records, computes features, streams Parquet |

### 2. Execution Command

```bash
# Features for the Variant A training pairs
venv\Scripts\python.exe business_entity_resolution/src/main_l7.py \
    --pairs artifacts/train_pairs/variant_a.parquet

# Dev run
venv\Scripts\python.exe business_entity_resolution/src/main_l7.py \
    --pairs artifacts/train_pairs/variant_a.parquet --max-pairs 200000
```

### 3. The 27 Features

| Block | Count | Features |
|---|---|---|
| Name | 8 | `name_ratio`, `name_jaro_winkler`, `name_token_sort`, `name_token_set`, `name_char3_jaccard`, `name_token_jaccard`, `name_idf_overlap`, `name_legal_suffix_match` |
| Address | 9 | `addr_house_exact`, `addr_street_jaccard`, `addr_postal_exact`, `addr_postal_prefix3`, `addr_city_exact`, `addr_state_match`, `addr_token_jaccard`, `addr_char3_jaccard`, `addr_digit_conflict` |
| Retrieval | 6 | `ret_exact_key_hit`, `ret_rrf_score`, `ret_retriever_agreement`, `ret_candidate_rank`, `ret_retrieved`, `ret_rank_inverse` |
| Structural | 4 | `struct_same_country`, `struct_name_len_ratio`, `struct_addr_len_ratio`, `struct_addr_missing_xor` |

String similarity uses `rapidfuzz` (ratio, Jaro-Winkler, token sort/set); set features are Jaccard; `name_idf_overlap` is an IDF-weighted token overlap built from candidate document frequencies. RRF score / channel agreement are joined from L3/L4 artifacts when `--l3-dir`/`--l4-dir` are passed, else 0.

### 4. Verification

- 10/10 package unit tests pass (5 new L7 tests).
- Functional smoke run produced a `(26, 31)` table = 4 meta columns + **27 float32 features**, **0 NaN**.

---

## ✅ Layer 8: GBDT Matcher Training + Hard-Negative A/B (COMPLETED)

```
[L5] ──► [L6] ──► [L7] ──► [L8] ──► [L9]
 ✅      ✅      ✅      ✅      ✅
```

### 1. What Was Built

| Module | File | Purpose |
|---|---|---|
| Training core | [`src/l6_l8_matching/model.py`](business_entity_resolution/src/l6_l8_matching/model.py) | `train_oof` (GroupKFold OOF), `train_full`, `top_importances` |
| Unit Tests | [`src/l6_l8_matching/unit_tests.py`](business_entity_resolution/src/l6_l8_matching/unit_tests.py) | L8 grouped-OOF test |
| Entry Point | [`src/main_l8.py`](business_entity_resolution/src/main_l8.py) | Trains variants, tunes thresholds, exports OOF + model, A/B comparison |

### 2. Execution Command

```bash
# Train + compare both variants
venv\Scripts\python.exe business_entity_resolution/src/main_l8.py

# One variant / dev cap
venv\Scripts\python.exe business_entity_resolution/src/main_l8.py --variant a --max-pairs 500000
```

### 3. Design

- **Grouped folds, not row folds**: pairs sharing an `s1_id` are correlated, so `GroupKFold` on `s1_id` prevents leakage. OOF probabilities are therefore honest and usable for threshold tuning without a separate holdout.
- **Evaluation matches inference**: macro F0.5 is computed on *retrieved* candidates only (easy negatives excluded), so the A/B reflects the real decision problem. The `(tau_match, tau_s)` pair is grid-searched using the official L1 scorer (`evaluate_with_thresholds`).
- **A/B test**: Variant A (with hard negatives) vs Variant B (without) are compared on out-of-fold macro F0.5; the winner is recorded.
- **Exports**: `artifacts/models/oof_<v>.parquet` (for L9 calibration) and `artifacts/models/lgbm_variant_<v>.txt` (for L10/L11 inference).

### 4. Deliverables

| Output | Description |
|---|---|
| `artifacts/models/oof_<v>.parquet` | OOF probability per pair |
| `artifacts/models/lgbm_variant_<v>.txt` | Full-data LightGBM booster |
| `output/l8_model_report.json` | OOF AUC, best macro F0.5, thresholds, importances, A/B winner |

### 5. Verification

- 11/11 package unit tests pass (new grouped-OOF test).
- Functional smoke run trained both variants, produced `(1017, 5)` OOF tables, exported both boosters, and completed the A/B comparison.

> ⚠️ **Windows note**: LightGBM must be imported *before* scikit-learn, otherwise their OpenMP runtimes conflict and `model.fit` raises an access violation. `main_l8.py` imports `lightgbm` first (with a comment) for this reason.

---

## ✅ Layer 9: Conditional Isotonic Calibration (COMPLETED)

```
[L6] ──► [L7] ──► [L8] ──► [L9] ──► [L10]
 ✅      ✅      ✅      ✅      ✅

```

### 1. What Was Built

| Module | File | Purpose |
|---|---|---|
| Calibration core | [`src/l6_l8_matching/calibration.py`](business_entity_resolution/src/l6_l8_matching/calibration.py) | Reliability diagnostics (`brier`, `log_loss`, `reliability_curve`, `ece`), grouped CV isotonic, knot serialization |
| Unit Tests | [`src/l6_l8_matching/unit_tests.py`](business_entity_resolution/src/l6_l8_matching/unit_tests.py) | 4 L9 tests (reliability metrics, ECE extremes, isotonic + serialization, grouped calibration) |
| Entry Point | [`src/main_l9.py`](business_entity_resolution/src/main_l9.py) | Evaluates calibration options, makes the conditional adoption decision, exports artifacts |

### 2. Execution Command

```bash
# Calibrate Variant A on the L8 OOF probabilities
venv\Scripts\python.exe business_entity_resolution/src/main_l9.py --variant a

# Tighten the adoption threshold
venv\Scripts\python.exe business_entity_resolution/src/main_l9.py --variant a --min-ece-improvement 0.01
```

### 3. Design

- **Conditional, not unconditional**: isotonic regression is a **monotonic** map, so it preserves the ranking of the OOF scores. The best macro F0.5 achievable by tuning a threshold is therefore *unchanged* by calibration. Calibration only buys **probability reliability** (Brier / ECE), which matters for per-country decision rules and downstream stacking — never for raw ranking. So we adopt it only when it clearly pays.
- **Three options evaluated** on the retrieved (non-easy) OOF pairs: `none` (raw probabilities), `global` (one calibrator), `per_country` (one calibrator per country bucket).
- **Grouped cross-validation**: calibrators are fit with `GroupKFold` on `s1_id`, so no pair is calibrated by a fit that saw its own entity. Without this the reported ECE would be optimistically biased.
- **Adoption rule**: a calibrator is eligible only if `ece <= baseline_ece - min_ece_improvement` (default `0.005`) **and** `macro_f05 >= baseline_f05 - 1e-6`. Among eligible options the lowest-ECE one wins; otherwise the pipeline keeps `none`. This makes the layer safe by construction.
- **Serialization**: the chosen calibrator is persisted as its isotonic knot (`X_thresholds_` / `y_thresholds_`) list, applied at inference via `np.interp` — no pickling of scikit-learn objects.

### 4. Deliverables

| Output | Description |
|---|---|
| `artifacts/models/oof_calibrated_<v>.parquet` | OOF table with raw and calibrated probabilities |
| `artifacts/models/isotonic_<v>.json` | Chosen calibrator knots (for L10/L11 inference) |
| `output/l9_calibration_report.json` / `.md` | Per-option ECE/Brier/LogLoss/AUC/macro F0.5 and the decision |

### 5. Verification

- **15/15** L6–L9 package unit tests pass (4 new L9 tests).
- Functional smoke run on a miscalibrated OOF table: raw ECE `0.1727` → global `0.0332` / per-country `0.0352` (≈5× reliability gain). Because the sampled macro F0.5 did not improve, the layer correctly chose `none` — the conditional guard behaving as intended.

> ⚠️ **Windows note**: L9 inherits the L8 import-order constraint — LightGBM must be imported before scikit-learn. `calibration.py` imports scikit-learn alone (no conflict), which is why it is imported *after* `model` in the entry point.

---

## ✅ Layer 10: Decision Engine + Singleton Protection (COMPLETED)

```
[L7] ──► [L8] ──► [L9] ──► [L10] ──► [L11]
 ✅      ✅      ✅      ✅      ✅
```

### 1. What Was Built

| Module | File | Purpose |
|---|---|---|
| Decision core | [`src/l10_decision/decision.py`](business_entity_resolution/src/l10_decision/decision.py) | `apply_decision_rule`, `tune_thresholds`, score grouping, submission TSV writer |
| Unit Tests | [`src/l10_decision/unit_tests.py`](business_entity_resolution/src/l10_decision/unit_tests.py) | 11 tests (rule/scorer parity, singleton guard, margin, open-set veto, TSV format) |
| Entry Point | [`src/main_l10.py`](business_entity_resolution/src/main_l10.py) | Loads L9/L8 scores, tunes thresholds, applies the rule, writes `matching_results` + report |

### 2. Execution Command

```bash
# Tune + decide on the validation OOF probabilities
venv\Scripts\python.exe business_entity_resolution/src/main_l10.py --split train

# Decide a scored test candidate set with the L8 tuned thresholds
venv\Scripts\python.exe business_entity_resolution/src/main_l10.py --split test \
    --scores artifacts/scores/test_scores.parquet --candidate-pairs output/candidate_pairs.tsv
```

### 3. Design

- **One rule, two copies that agree**: the official scorer's `evaluate_with_thresholds` is the *evaluation* copy; `apply_decision_rule` is the *production* copy. A unit test asserts they produce identical predictions, so validation thresholds transfer to inference with no reimplementation gap.
- **Joint threshold search**: `(tau_match, tau_s)` are grid searched together against the official L1 macro F0.5, not tuned independently — the two interact through the margin rule.
- **Singleton guard**: when a reference's top candidate score is below `tau_s`, predict the empty list. The scorer grants a correct empty prediction 1.0 for a true singleton, so this guard is where a large fraction of the macro score comes from.
- **Margin rule**: on the match path only candidates within `margin` of the top (and at/above `tau_match`) survive, which keeps near-duplicate source-2/3 records together without dragging in weak candidates.
- **Open-set fallback veto (France)**: countries never seen in training ground truth (France is test-only) get `tau_s` raised by `open_set_boost` and any top score below `veto_min_confidence` is vetoed to empty. Macro F0.5 is precision-weighted, so refusing a low-confidence guess on an unseen country beats risking a false match. Open-set references are detected from the country bucket of the reference record, independent of the split flag.

### 4. Deliverables

| Output | Description |
|---|---|
| `output/matching_results.tsv` | Test-split submission decision (the scored deliverable) |
| `output/matching_results_val.tsv` | Train-split dev decision |
| `output/l10_decision_report.json` / `.md` | Thresholds, tuned F0.5, singleton/match counts, open-set count |

### 5. Verification

- **11/11** L10 unit tests pass.
- Functional smoke run on a synthetic OOF table over real validation references (80 refs, 428 pairs): joint search tuned `tau_match=0.2`, `tau_s=0.5` → macro F0.5 `0.8087`; 75 non-empty / 5 singletons / 119 matched pairs; wrote an 80-row `matching_results_val.tsv` in the exact tab-separated submission format.

> ℹ️ **Inference path**: at real test time the L11 runner scores the L5 `candidate_pairs.tsv` with the exported booster and passes the resulting parquet to this entry point via `--scores`; L10 then needs no ground truth and falls back to the thresholds tuned in L8.

---

## ✅ Layer 10.5: Diagnostic Error Analysis + France Spot-Check (COMPLETED)

### 1. What Was Built

| Module | File | Purpose |
|---|---|---|
| Diagnostics core | [`src/l10_diagnostics/diagnostics.py`](business_entity_resolution/src/l10_diagnostics/diagnostics.py) | Error classification, grouped aggregation, metadata loading, open-set spot-check |
| Unit Tests | [`src/l10_diagnostics/unit_tests.py`](business_entity_resolution/src/l10_diagnostics/unit_tests.py) | 8 tests (error classes, buckets, aggregation, TSV load, spot-check) |
| Entry Point | [`src/main_l10_diagnostics.py`](business_entity_resolution/src/main_l10_diagnostics.py) | Runs the breakdown and writes the report |

### 2. Execution Command

```bash
# Full error breakdown on the validation decisions
venv\Scripts\python.exe business_entity_resolution/src/main_l10_diagnostics.py --split train

# Open-set spot-check on the test decisions (no ground truth)
venv\Scripts\python.exe business_entity_resolution/src/main_l10_diagnostics.py --split test \
    --predictions output/matching_results.tsv --scores artifacts/scores/test_scores.parquet
```

### 3. Design

- **Per-entity error classes first**: macro F0.5 is averaged per Source-1 entity, so the report labels every entity `true_singleton` / `false_positive` / `miss` / `exact` / `partial` and reports each class's count, share, and F0.5 mass. This immediately shows whether loss comes from broken singletons (most costly under a precision-weighted metric), misses, or partial overlaps.
- **Breakdowns the brief asks for**: country, matched-source composition (S2 / S3 / both), name length bucket, address length / missing-address, and script — all computed with the same per-entity averaging so group macro F0.5 sums back to the overall figure.
- **Consistency guard**: the overall macro F0.5 is recomputed with the official L1 scorer and printed alongside, catching any drift between the diagnostic and the metric.
- **France / open-set spot-check**: with no ground truth on test, the report hits the limits of what can be measured — so it surfaces volume (non-empty vs empty), top-score quantiles, and the highest-confidence predictions with their names, for manual review.

### 4. Deliverables

| Output | Description |
|---|---|
| `output/l10_diagnostics_report.json` / `.md` | Error-class table, grouped breakdowns, sanity F0.5, France spot-check samples |

### 5. Verification

- **8/8** L10.5 unit tests pass.
- Functional smoke (train): 400 real validation refs with a constructed error mix — overall macro F0.5 `0.5000` matched the official scorer `0.5000`; breakdowns reported by country (US `0.5098`, India `0.4828`), source composition, and name-length bucket.
- Functional smoke (test): 200 real France references — spot-check reported 67 non-empty / 133 empty and top-score quantiles `p50=0.35, p75=0.67, p100=0.94` with the top-scoring samples listed.

---

## ✅ Layer 11: Test Inference + Submission Validation (COMPLETED)

```
[L8] ──► [L9] ──► [L10] ──► [L11] ──► [L12]
 ✅      ✅      ✅      ✅      ⏳ Upcoming
```

### 1. What Was Built

| Module | File | Purpose |
|---|---|---|
| Scorer | [`src/l11_inference/scorer.py`](business_entity_resolution/src/l11_inference/scorer.py) | Load the L8 booster + optional L9 calibrator, `score_matrix`, `calibrate_scores` |
| Inference engine | [`src/l11_inference/inference.py`](business_entity_resolution/src/l11_inference/inference.py) | Streaming reference-batch scoring, IDF build, submission writers |
| Unit Tests | [`src/l11_inference/unit_tests.py`](business_entity_resolution/src/l11_inference/unit_tests.py) | 7 tests (parsing, writers, scorer, calibrator, streaming inference) |
| Entry Point | [`src/main_l11.py`](business_entity_resolution/src/main_l11.py) | Run inference, write both submission files, invoke the validator |

### 2. Execution Command

```bash
# Full test inference + validation
venv\Scripts\python.exe business_entity_resolution/src/main_l11.py --variant a

# Dev cap (skips full coverage and the validator)
venv\Scripts\python.exe business_entity_resolution/src/main_l11.py --max-references 5000 --no-validate
```

### 3. Design

- **Memory strategy — reference-batch streaming**: the test candidate set is ~10M pairs and each pair only needs its two records while it is scored. Each batch loads its own Source 1 records and candidates, scores them, then discards the records. Only surviving `(candidate, score)` tuples are kept.
- **Lossless pruning**: only candidates with `score >= tau_match` are stored. This is exact because the L10 rule uses `effective_threshold = max(tau_match, top - margin) >= tau_match`, so a candidate below `tau_match` can never be kept.
- **One IDF for every batch**: `name_idf_overlap` needs a document-frequency table; it is built once in a dedicated streaming pass over candidate shards so features are consistent across batches.
- **Complete coverage**: with `all_references=True` the reference list is unioned with every Source-1 id in the split, so entities with no candidates still emit an (empty) row — a validator requirement.
- **No ground truth needed**: thresholds resolve CLI → L10 report → L8 report → defaults, so the same entry point handles validation and the real test run.
- **Calibration is optional and correct**: `load_calibrator` returns `None` when L9 chose `none`; otherwise global or per-country knots are applied via `np.interp`.

### 4. Deliverables

| Output | Description |
|---|---|
| `output/matching_results.tsv` | Final matches (the scored deliverable) |
| `output/candidate_pairs.tsv` | The candidate set fed to the matcher (completeness-checked) |
| `output/l11_inference_report.json` | Config, thresholds, pair counts, validator exit code |

### 5. Verification

- **7/7** L11 unit tests pass; full regression 49/49 across L1/L6–L11.
- Functional smoke on real test records (20 refs, 60 pairs) with a synthetic booster: scored all pairs, wrote both TSVs, open-set detection reported 3 refs.
- A complete-coverage variant of the output **PASSED the official `validate_submission.py`** (20/20 rows, 2 non-empty) — proving the format is submission-safe.
- The `main_l11.py` validator wiring was exercised end-to-end (exit code correctly non-zero when the dev cap omitted required entities).

> ⚠️ **Prerequisite**: a real submission needs the trained booster (`artifacts/models/lgbm_variant_<v>.txt`) and `output/candidate_pairs.tsv` — i.e. the full L3 → L4 → L5 → L6 → L7 → L8 chain on the test split.

