# Amazon ML Challenge 2026: Project Walkthrough & Execution Log 📖

**Team Name:** PraxisCode_X  
**Challenge Track:** Business Entity Resolution  
**Repository:** [1919-14/Amazon-2026-PraxisCode_X](https://github.com/1919-14/Amazon-2026-PraxisCode_X.git)  

---

## 🧭 Overview & Progress Tracker

This document provides a continuous, step-by-step walkthrough of our 14-layer machine learning pipeline. It logs all design decisions, schema contracts, verification results, and exact run commands.

```
[Phase 0: Research & Profiling] ──► [L0: Setup & Audit] ──► [L1: Validation Split] ──► [L2: Normalizer]
            ✅ COMPLETED                    ✅ COMPLETED               ⏳ IN PROGRESS            ⏳ UPCOMING
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

## 🔮 Upcoming Layer: Layer 1 (Validation Split & Scorer)

* **Objective**: Generate a stratified 80/20 train/validation split by Source 1 ID.
* **Deliverables**:
  * `src/l1_validation/split_generator.py`
  * `src/l1_validation/metrics.py` (Official macro $F_{0.5}$ metric)
  * `artifacts/splits/train_ids.parquet` & `artifacts/splits/val_ids.parquet`
