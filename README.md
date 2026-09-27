# Amazon ML Challenge 2026: Business Entity Resolution 🚀

[![Team](https://img.shields.io/badge/Team-PraxisCode_X-blue.svg?style=for-the-badge)](https://github.com/1919-14/Amazon-2026-PraxisCode_X)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg?style=for-the-badge&logo=python)](https://python.org)
[![Submissions](https://img.shields.io/badge/Format-PASSED_OFFICIAL_VALIDATOR-success.svg?style=for-the-badge)](business_entity_resolution/output/matching_results.tsv)

**Team Name:** PraxisCode X  
**Team Members:**  
- **V S S K Sai Narayana**  
- **Sujeet Jaiswal**  
- **Sujeet Sahni**  

**Challenge Track:** Business Entity Resolution  
**Evaluation Metric:** Macro-Averaged $F_{0.5}$ Score ($\beta = 0.5$)  

---

## 📌 1. Executive Summary

In commercial web platforms, business identity records originate from multiple independent, heterogeneous sources with noisy, incomplete, or multilingual/transliterated fields. This repository contains the complete end-to-end Machine Learning solution developed by **PraxisCode X** for the **Amazon ML Challenge 2026**.

Given business records across three data sources (`Source 1`, `Source 2`, `Source 3`), our pipeline resolves which records across sources represent the same real-world commercial entity. Our solution combines **Multi-Channel Hybrid Candidate Blocking** (High-IDF Sparse Indexing + Character N-grams + Multilingual E5 Dense Embeddings + Phonetic Metaphone Hashing) with a **5-Fold Stacking Meta-Learner (LightGBM + CatBoost + XGBoost)** and an **Adaptive Decision Engine** tuned for zero-shot open-set precision.

---

## 🏗️ 2. Pipeline Architecture

Our solution follows a disciplined, highly modular, and memory-bounded 14-layer architecture:

```
┌──────────────────────────────────────────────────────────────────┐
│                    INPUT DATASETS                                │
│  train_source1/2/3.tsv, train_ground_truth.tsv,                  │
│  test_source1/2/3.tsv  (26.4M+ Total Records)                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
        ╔════════════════════════════════════════════════╗
        ║  L0  SETUP, CONFIG, DATA AUDIT & INGESTION     ║
        ║  Streaming PyArrow chunking, schema validation ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L1  STRATIFIED TRAIN/VAL SPLIT & SCORER       ║
        ║  Source-1 ID split, exact Macro F0.5 scorer    ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L2  MULTILINGUAL NORMALIZATION ENGINE         ║
        ║  NFKC, diacritics, legal suffixes, addresses,  ║
        ║  digit extraction & script detection           ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L3  MULTI-CHANNEL HYBRID BLOCKING             ║
        ║  • Ch A: Exact core name + postal hash         ║
        ║  • Ch C: High-IDF token inverted index         ║
        ║  • Ch D: Character 2-4 gram TF-IDF cosine      ║
        ║  • Ch G: Script-aware Metaphone & Soundex      ║
        ║  • Ch H: Multilingual E5 Dense FAISS IVF-PQ    ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L4  RECIPROCAL RANK FUSION (RRF)              ║
        ║  Multi-channel ranking & signal preservation   ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L5  ADAPTIVE CANDIDATE TRUNCATION             ║
        ║  Compact budget (K ≈ 5.5 - 6.0 per reference)  ║
        ║  → output/candidate_pairs.tsv                  ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L6  TRAINING PAIRS & HARD NEGATIVE SAMPLING   ║
        ║  Positives + informative hard negative decoys  ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L7  PAIRWISE FEATURE ENGINEERING (27+ Signals)║
        ║  String metrics, token Jaccard, address digits ║
        ║  dense cosine sim, phonetic overlap & ratios   ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L8  STACKING ENSEMBLE TRAINING (5-Fold CV)    ║
        ║  Level 1: LightGBM + CatBoost + XGBoost        ║
        ║  Level 2: Calibrated Logistic Regression Meta  ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L9  PROBABILITY CALIBRATION                   ║
        ║  Isotonic calibration on OOF predictions       ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L10  ADAPTIVE DECISION ENGINE & GRAPH CLUSTER ║
        ║  F0.5 threshold optimization, margin rule,     ║
        ║  open-set France singleton veto policy         ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L11  STREAMING TEST INFERENCE & VALIDATION    ║
        ║  Generates final submissions & audit report    ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  FINAL OUTPUT ARTIFACTS                        ║
        ║  • output/matching_results.tsv (1.73M rows)    ║
        ║  • output/candidate_pairs.tsv  (1.73M rows)    ║
        ║  • validate_submission.py → PASS ✅            ║
        ╚════════════════════════════════════════════════╝
```

---

## 🔬 3. Key Technical Innovations

### 1. Multi-Channel Blocking (Recall $\ge 98\%$)
- **Lexical & Character Channels:** Inverted index with IDF term weighting and character 2-4 gram nearest neighbors.
- **Dense Multilingual Semantic Retrieval:** `intfloat/multilingual-e5-small` (118M parameters) generates 384-dimensional embeddings. Indexed using streaming FAISS `IndexIVFPQ` with automated disk cache cleanup, eliminating memory leaks and fitting within Kaggle's 19.5GB disk constraint.
- **Phonetic Channel:** Double Metaphone and Soundex indexing to connect transliterated and phonetically identical names across English and Indian languages.
- **Adaptive Truncation (L5):** Maintains an average of $K \approx 5.5 - 6.0$ candidate pairs per entity, achieving a $>99.95\%$ search space reduction.

### 2. 27+ Engineered Pairwise Features (L7)
- **Name Similarities:** Jaro-Winkler, Levenshtein, Token Sort Ratio, Token Set Ratio, Monge-Elkan, Longest Common Subsequence.
- **Address & Geo Discriminators:** Token Jaccard overlap, house number/building digit exact match vs. conflicting penalty, postal code match boolean.
- **Cross-Modal Retrieval Scores:** RRF reciprocal rank, BM25 score, E5 dense cosine similarity, phonetic token overlap.

### 3. Stacking Ensemble Meta-Learner (L8)
- 5-Fold Stratified Cross-Validation across **LightGBM**, **CatBoost**, and **XGBoost**.
- Level-2 Meta-Classifier dynamically weights tree predictions against non-linear interaction features.

### 4. $F_{0.5}$ Precision Optimization & Open-Set Veto (L10)
- The $F_{0.5}$ metric values Precision over Recall ($\beta = 0.5$).
- Joint tuning of $\tau_{\text{match}} = 0.50$ and singleton threshold $\tau_s = 0.10$ with an ambiguity margin of $0.05$.
- Held-out open-set veto policy prevents false merges on unseen countries (e.g., France test set).

---

## 📂 4. Repository Structure

```
Amazon-2026-PraxisCode_X/
├── README.md                           # Main documentation & architecture guide
├── DOCUMENTATION.md                    # Official competition submission document
├── LICENSE                             # MIT Open-Source License
├── CONTRIBUTING.md                     # Contribution guidelines & setup
└── business_entity_resolution/
    ├── output/
    │   ├── matching_results.tsv        # Primary competition submission (1,732,544 rows)
    │   ├── candidate_pairs.tsv         # Candidate blocking pairs (1,732,544 rows)
    │   ├── l10_decision_report.json    # Decision thresholds & match distribution
    │   └── l11_inference_report.json   # Full inference metrics and timing
    └── src/
        ├── config.py                   # Global directory and pipeline configuration
        ├── schemas.py                  # Frozen PyArrow schemas and contracts
        ├── run_scaled_pipeline.py      # Unified CLI runner for training & inference
        ├── main_l0.py … main_l11.py   # Modular entry points per pipeline layer
        ├── main_phon.py                # Standalone phonetic channel generator
        ├── l1_validation/              # Macro F0.5 scorer and validation split
        ├── l2_normalization/           # NFKC, accents, legal suffixes, addresses
        ├── l3_l5_blocking/             # Inverted index, dense E5 FAISS, phonetic, RRF
        ├── l6_l8_matching/             # Feature extraction, LightGBM, CatBoost, Stacking
        ├── l10_decision/               # F0.5 optimizer, open-set veto, graph clustering
        ├── l11_inference/              # Batch inference engine & validator wrapper
        ├── utils/                      # Coverage checker, report history, cache utils
        └── requirements.txt            # Pinned dependency requirements
```

---

## 🛠️ 5. Quickstart & Reproduction Guide

### 1. Environment Installation
```bash
# Clone the repository
git clone https://github.com/1919-14/Amazon-2026-PraxisCode_X.git
cd Amazon-2026-PraxisCode_X

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r business_entity_resolution/src/requirements.txt
```

### 2. Run Scaled Training & Inference Pipeline
```bash
# Full end-to-end execution across US, India, and France
python business_entity_resolution/src/run_scaled_pipeline.py \
    --mode test \
    --countries us india france \
    --model-kind stack \
    --dense --phon
```

### 3. Verify Submission Format
```bash
python "DATA SET/student_resource/utils/validate_submission.py" \
    --matching "business_entity_resolution/output/matching_results.tsv" \
    --candidate "business_entity_resolution/output/candidate_pairs.tsv" \
    --test-dir "DATA SET/student_resource/dataset/test"
```

**Validator Output:**
```
ML Challenge 2026 — submission validator
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (153495 empty, 1579049 non-empty).
  candidate_pairs.tsv: 1732544 rows (0 empty, 1732544 non-empty).
PASS — no blocking issues found. Safe to submit.
```

---

## ⚖️ 6. Compliance & Constraints

- **Parameter Count:** Model footprint is $< 150\text{M}$ total parameters, well within the competition's $\le 8\text{ Billion}$ parameter limit.
- **Open-Source Licenses:** All base architectures (`intfloat/multilingual-e5-small`, `LightGBM`, `CatBoost`, `XGBoost`, `FAISS`) are MIT / Apache 2.0 compliant.
- **Network Isolation:** $100\%$ offline execution during inference with zero external API calls or web services.
- **Hardware Compliance:** Peak RAM stays $< 14\text{ GB}$ via streaming PyArrow datasets.

---

## 👥 7. Team Details

**Team:** PraxisCode X  
- **V S S K Sai Narayana**
- **Sujeet Jaiswal**
- **Sujeet Sahni**
