# Amazon ML Challenge 2026: Business Entity Resolution 🚀

**Team Name:** PraxisCode_X  
**Challenge Track:** Business Entity Resolution  
**Evaluation Metric:** Macro-Averaged $F_{0.5}$ Score  

---

## 📌 1. Executive Summary

In commercial platforms, business identity data originates from multiple independent, heterogeneous sources with noisy, incomplete, or transliterated fields. This repository provides an end-to-end, high-performance Machine Learning solution designed for the **Amazon ML Challenge 2026**. 

Given business records across three data sources (`Source 1`, `Source 2`, `Source 3`), our pipeline resolves which records across sources represent the same real-world commercial entity.

---

## 🏗️ 2. Core 14-Layer Pipeline Architecture

Our solution implements a disciplined **14-Layer Architecture** engineered specifically to optimize the precision-heavy **Macro $F_{0.5}$** metric while maintaining an ultra-compact candidate blocking pool ($K \approx 5.5$ per reference entity) to excel in Amazon's scalability review.

```
┌──────────────────────────────────────────────────────────────────┐
│                    INPUT TSV DATASETS                            │
│  train_source1.tsv, train_source2.tsv, train_source3.tsv,        │
│  train_ground_truth.tsv, test_source1/2/3.tsv                    │
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
        ║  Stratified split by country, macro F0.5       ║
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
        ║  L3e  Channel E — Dense retrieval (optional)   ║
        ║  L3f  Per-channel recall measurement           ║
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
        ║  → OUTPUT: candidate_pairs.tsv                 ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L6  TRAINING PAIR CONSTRUCTION + HARD NEGS    ║
        ║  Positive / hard neg / easy neg split,         ║
        ║  stratified coarse bands, ablation on/off      ║
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
        ║  L9  CALIBRATION (conditional)                 ║
        ║  Isotonic on OOF; only if per-country or       ║
        ║  stacking is used                              ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L10  DECISION ENGINE                          ║
        ║  Joint τ_match + τ_s tuning, margin rule,      ║
        ║  singleton protection, France fallback         ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L10.5  ERROR ANALYSIS                         ║
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
        ║  L11.5  FIRST SUBMISSION                       ║
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
        ║  Compare val F0.5 with/without stretch         ║
        ╚════════════════════════┬═══════════════════════╝
                                 │
                                 ▼
        ╔════════════════════════════════════════════════╗
        ║  L13  DOCUMENTATION + PACKAGING                ║
        ║  Methodology, README, requirements, ZIP        ║
        ╚════════════════════════════════════════════════╝
                                 │
                                 ▼
                    output/matching_results.tsv
                    output/candidate_pairs.tsv
```

---

## 🔬 3. Detailed Layer Breakdown

### **L0 – L2: Foundation & Data Normalization**
* **L0 Setup & Audit**: Streaming TSV scanner logging schema compliance, record counts, and country distributions.
* **L1 Validation & Scoring**: Stratified train/val split (80/20 on S1 entities) with exact macro $F_{0.5}$ metric computation.
* **L2 Normalization**:
  * Unicode NFKC normalization + accent stripping (`café` $\rightarrow$ `cafe`).
  * Multilingual script detection (Hindi Devanagari, Tamil, Kannada, Telugu).
  * Legal suffix canonicalization (`Pvt Ltd`, `LLC`, `SARL`, `Inc`, `Corp`).
  * Structured address parsing & digit sequence extraction.

### **L3 – L5: Country-Stratified Blocking & Reciprocal Rank Fusion**
* **L3 Country Stratification**: Records only match within the same country (`US`, `India`, `France`). Zero cross-country overhead.
* **Multi-Channel Retrieval**:
  * *L3b Channel A*: Exact normalized name + postal/house hash.
  * *L3c Channel C*: High-IDF token inverted index (filters out stopwords like `Store`, `Enterprises`).
  * *L3d Channel D*: Character 2-4 gram TF-IDF cosine nearest neighbors.
  * *L3e Channel E*: Dense multilingual embeddings (`BGE-M3`).
* **L4 Reciprocal Rank Fusion (RRF)**: Merges channels via $RRF(d) = \sum_{c \in C} \frac{1}{60 + r_c(d)}$.
* **L5 Adaptive Truncation**: Tiered candidate budget targeting $K \approx 5.5 \text{ to } 7.0$ candidates per S1 entity $\rightarrow$ exports `output/candidate_pairs.tsv`.

### **L6 – L9: Training, Feature Engineering & GBDT Matcher**
* **L6 Hard Negative Mining**: Constructs training pairs in 1 Positive : 3 Hard Negatives : 2 Easy Negatives ratio.
* **L7 Pairwise Feature Vector**: 27+ features covering name similarities, structured address digit conflicts, retrieval signals, and structural length ratios.
* **L8 GBDT Matcher**: 5-fold cross-validated LightGBM trained with out-of-fold probability outputs.
* **L9 Calibration**: Isotonic probability calibration on OOF predictions.

### **L10 – L13: Decision Engine, Test Inference & Packaging**
* **L10 Decision Engine**: Joint tuning of match threshold $\tau_{\text{match}}$ and singleton threshold $\tau_s$ to protect the 5.6% singletons.
* **L10.5 Error Analysis**: Diagnostic breakdown of false positive / false negative errors by country, source, and name length.
* **L11 Test Inference**: Generates `matching_results.tsv` and `candidate_pairs.tsv`; passes `utils/validate_submission.py`.
* **L11.5 Submission #1**: Live portal submission.
* **L12 Stretch Iteration**: Targeted enhancement based on error analysis (Phonetic blocking, BGE-M3 dense FAISS, or Multilingual Cross-Encoder).
* **L12.5 Ablation Log**: Quantitative validation of stretch gains.
* **L13 Final Packaging**: Assembles `<team_name>_submission.zip` with runnable code and methodology write-up.

---

## 📂 4. Repository Structure

```
.
├── README.md                           # Main 14-Layer documentation & pipeline architecture
├── .gitignore                          # Git rules excluding large TSV dataset binaries
└── business_entity_resolution/
    ├── DATA_PROFILING_REPORT.md        # Comprehensive 26.4M record profiling report
    ├── output/
    │   ├── audit_summary.json          # Dataset audit counts & missing rates
    │   ├── deep_data_profile.json      # Full statistical breakdown
    │   ├── matching_results.tsv        # Final entity matches (Leaderboard Upload)
    │   └── candidate_pairs.tsv         # Blocking candidate pairs (Audit Evaluation)
    └── src/
        ├── inspect_samples.py          # Data profiling & noise pattern inspector
        ├── audit.py                    # Streaming dataset validation & distribution checker
        ├── deep_profile.py             # Advanced statistical profiling script
        ├── metrics.py                  # Official macro F0.5 metric implementation
        ├── test_core.py                # Core unit tests for metrics and data format
        └── requirements.txt            # Pinned dependencies
```

---

## 🛠️ 5. Quickstart & Reproduction Guide

### Environment Setup
```bash
git clone https://github.com/1919-14/Amazon-2026-PraxisCode_X.git
cd Amazon-2026-PraxisCode_X
python -m pip install -r business_entity_resolution/src/requirements.txt
```

### Validate Submission Format
```bash
python DATA SET/student_resource/utils/validate_submission.py \
    --matching business_entity_resolution/output/matching_results.tsv \
    --candidate business_entity_resolution/output/candidate_pairs.tsv \
    --test-dir DATA SET/student_resource/dataset/test
```

---

## ⚖️ 6. Compliance & License

- **Model Frameworks**: PyTorch, LightGBM, CatBoost, Hugging Face Transformers.
- **License**: All models and code libraries utilized are strictly under **Apache 2.0 / MIT Licenses** and within parameter constraints ($\le 8\text{ Billion parameters}$).
- **Data Integrity**: $100\%$ self-contained pipeline with zero external API or geocoding dependencies.
