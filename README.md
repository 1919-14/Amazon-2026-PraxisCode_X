# Amazon ML Challenge 2026: Business Entity Resolution 🚀

**Team Name:** PraxisCode_X  
**Challenge Track:** Business Entity Resolution  
**Evaluation Metric:** Macro-Averaged $F_{0.5}$ Score  

---

## 📌 1. Executive Summary

In commercial platforms, business identity data originates from multiple independent, heterogeneous sources with noisy, incomplete, or transliterated fields. This repository provides an end-to-end, high-performance Machine Learning solution designed for the **Amazon ML Challenge 2026**. 

Given business records across three data sources (`Source 1`, `Source 2`, `Source 3`), our pipeline resolves which records across sources represent the same real-world commercial entity.

---

## 🏗️ 2. Core 14-Layer Refined Pipeline Architecture

Our solution implements a disciplined, highly modular **14-Layer Refined Architecture**. Designed for rapid iteration, fast baseline submission, and open-set country adaptation (handling France in test data), it maximizes the precision-heavy **Macro $F_{0.5}$** metric while maintaining an ultra-compact candidate blocking pool ($K \approx 5.5$ per reference entity) to excel in Amazon's scalability audit.

```
┌──────────────────────────────────────────────────────────────────┐
│                    INPUT FILES                                    │
│  train_source1/2/3.tsv, train_ground_truth.tsv,                  │
│  test_source1/2/3.tsv                                            │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
        ╔════════════════════════════════════════════════╗
        ║  L0  SETUP, CONFIG, DATA VERIFICATION          ║
        ║  Load TSVs, audit, seeds, cache dirs, schema   ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L1  TRAIN/VAL SPLIT + F0.5 SCORER             ║
        ║  Plain random split by S1 ID                   ║
        ║  (France = known validation blind spot)        ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L2  NORMALIZATION ENGINE                      ║
        ║  NFKC, accents, legal suffix, address parse,   ║
        ║  digit extraction, script detection            ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L3  COUNTRY-STRATIFIED BLOCKING               ║
        ║                                                ║
        ║  L3a  Country bucket assignment                ║
        ║  L3b  Channel A — Exact-key index              ║
        ║  L3c  Channel C — IDF-weighted token index     ║
        ║  L3d  Channel D — Char TF-IDF + NN             ║
        ║  L3e  Per-channel recall measurement           ║
        ║                                                ║
        ║  (Dense retrieval REMOVED — lives in L12b)     ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L4  RECIPROCAL RANK FUSION (RRF)              ║
        ║  Combine channels into one ranked list per S1  ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L5  ADAPTIVE CANDIDATE TRUNCATION             ║
        ║  Coarse score + tiered budget + recall-vs-K    ║
        ║  → candidate_pairs.tsv                         ║
        ║  → blocking_report.md  [NEW explicit output]   ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L6  TRAINING PAIRS + HARD NEGATIVE A/B TEST   ║
        ║  Variant A: with hard negatives                ║
        ║  Variant B: without hard negatives             ║
        ║  Compare val F0.5, keep winner                 ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L7  PAIRWISE FEATURE ENGINEERING              ║
        ║  Name, address, retrieval, structural features ║
        ║  + feature separation check                    ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L8  MODEL TRAINING (LightGBM)                 ║
        ║  5-fold CV, OOF, class weight, importances     ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L9  CALIBRATION (CONDITIONAL — skip if Tier 1)║
        ║  Isotonic on OOF; only if per-country or       ║
        ║  stacking is used                              ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L10  DECISION ENGINE                          ║
        ║  Joint τ_match + τ_s, margin rule, France veto ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L10.5  ERROR ANALYSIS + FRANCE SPOT-CHECK     ║
        ║  Bucket errors by country/source/name length   ║
        ║  → tells you which stretch feature to add      ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L11  TEST INFERENCE + OUTPUT                  ║
        ║  Run pipeline on test, write both TSVs,        ║
        ║  validate_submission.py → PASS                 ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L11.5  FIRST SUBMISSION ✅ (floor guaranteed) ║
        ║  Upload matching_results.tsv to leaderboard    ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L12  STRETCH (pick ONE based on L10.5)        ║
        ║  12a Phonetic blocking                         ║
        ║  12b Dense retrieval (BGE-M3)                  ║
        ║  12c Cross-encoder + stacking                  ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L12.5  ABLATION LOG                           ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L13  DOCUMENTATION + PACKAGING                ║
        ╚════════════════════════════════════════════════╝
                                 │
                                 ▼
                    output/matching_results.tsv
                    output/candidate_pairs.tsv
                    output/blocking_report.md
```

---

## 📂 3. Modular Layer Directory Structure

To maximize codebase readability, maintainability, and audit compliance, our repository is strictly partitioned into dedicated layer modules under `src/`:

```
business_entity_resolution/
├── DATA_PROFILING_REPORT.md
├── output/                           # Output TSVs & audit reports
│   ├── matching_results.tsv          # Leaderboard submission
│   ├── candidate_pairs.tsv           # Candidates deliverable
│   └── blocking_report.md            # Recall & reduction ratio report
├── models/                           # Saved trained model checkpoints (.pkl / .bin)
│   └── lgbm_model_fold*.pkl
├── cache/                            # Disk cache for intermediate features (parquet)
└── src/                              # All runnable source code
    ├── l0_setup/                     # Layer 0: Setup, audit & data profiling
    │   ├── audit.py
    │   ├── deep_profile.py
    │   └── inspect_samples.py
    ├── l1_validation/                # Layer 1: Stratified split & F0.5 metric scorer
    │   ├── metrics.py
    │   └── split_generator.py
    ├── l2_normalization/             # Layer 2: Multilingual text & address normalizer
    │   └── normalizer.py
    ├── l3_l5_blocking/               # Layers 3-5: Stratified blocking, RRF & adaptive truncation
    │   └── blocking_engine.py
    ├── l6_l8_matching/               # Layers 6-8: Hard negatives, 27+ features & LightGBM CV
    │   ├── feature_engineering.py
    │   └── train_lgbm.py
    ├── l10_decision/                 # Layer 10: Joint thresholding, singleton guard & France veto
    │   └── decision_engine.py
    ├── requirements.txt              # Pinned environment dependencies
    └── main.py                       # Master end-to-end pipeline runner (L0 -> L11)
```

---

## 🛠️ 4. Quickstart & Reproduction Guide

### Environment Setup
```bash
git clone https://github.com/1919-14/Amazon-2026-PraxisCode_X.git
cd Amazon-2026-PraxisCode_X
python -m pip install -r business_entity_resolution/src/requirements.txt
```

### Run End-to-End Pipeline
```bash
python business_entity_resolution/src/main.py
```

### Validate Submission Format
```bash
python DATA SET/student_resource/utils/validate_submission.py \
    --matching business_entity_resolution/output/matching_results.tsv \
    --candidate business_entity_resolution/output/candidate_pairs.tsv \
    --test-dir DATA SET/student_resource/dataset/test
```

---

## ⚖️ 5. Compliance & License

- **Model Frameworks**: PyTorch, LightGBM, CatBoost, Hugging Face Transformers.
- **License**: All models and code libraries utilized are strictly under **Apache 2.0 / MIT Licenses** and within parameter constraints ($\le 8\text{ Billion parameters}$).
- **Data Integrity**: $100\%$ self-contained pipeline with zero external API or geocoding dependencies.
