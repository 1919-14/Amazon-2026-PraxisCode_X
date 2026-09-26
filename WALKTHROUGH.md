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

## 🔮 Next Layer: L3–L5 Fast Country-Stratified Blocking & Reciprocal Rank Fusion

**Objective**: Multi-channel candidate generator achieving $\ge 98\%$ recall on the validation set while keeping candidate budget $K \approx 5.5 - 7.0$ per S1 entity.

**Channels to implement**:
- **Channel A (Exact Index)**: `name_core` + `addr_postal` / `addr_house_number` hash buckets.
- **Channel B (High-IDF Token Inverted Index)**: Inverse document frequency token matching over `name_core`.
- **Channel C (Char 2-4 gram TF-IDF)**: Fast cosine/MinHash candidate retrieval.
- **Channel D (Phonetic Block)**: `name_phonetic` + `addr_postal` prefix.
- **Fusion & Truncation (L4-L5)**: Reciprocal Rank Fusion (RRF) + adaptive tiered candidate budget allocation to produce `output/candidate_pairs.tsv`.

