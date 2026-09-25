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
                    output/blocking_report.md
```

---

## 🔬 3. Detailed Layer Specifications

### **L0 – L2: Setup, Validation & Normalization**
* **L0 Setup & Audit**: Streaming TSV verification of schema, nulls, record counts, and Unicode scripts.
* **L1 Validation & Scoring**: Stratified train/val split by S1 ID with macro $F_{0.5}$ evaluation. *(Note: France is a known validation blind spot present only in test data!)*
* **L2 Normalization Engine**:
  * Unicode NFKC normalization + accent stripping (`réseau` $\rightarrow$ `reseau`).
  * Legal suffix canonicalization (`Pvt Ltd`, `LLC`, `SARL`, `Inc`, `Corp` $\rightarrow$ canonical form).
  * Structured address parsing (`house_num`, `street`, `city`, `postal`, `state`, `country`) & digit sequence extraction.

### **L3 – L5: Fast Stratified Blocking & Adaptive Candidate Truncation**
* **L3 Country Stratification**: Records are partitioned into independent country buckets (`US`, `India`, `France`). Zero cross-country overhead.
* **Multi-Channel Fast Retrieval**:
  * *L3b Channel A*: Exact normalized name + postal/house hash.
  * *L3c Channel C*: High-IDF token inverted index (filters out stopwords).
  * *L3d Channel D*: Character 2-4 gram TF-IDF nearest neighbors.
  * *(Dense retrieval BGE-M3 is decoupled into L12b stretch for fast initial execution)*.
* **L4 Reciprocal Rank Fusion (RRF)**: Merges sparse channels into a single ranked candidate list per S1 entity.
* **L5 Adaptive Truncation**: Coarse scoring + tiered budget allocation targeting $K \approx 5.5 \text{ to } 7.0$ candidates per entity.
  * **Explicit Outputs**: `output/candidate_pairs.tsv` and `output/blocking_report.md` (for Amazon's candidate compactness review).

### **L6 – L9: Hard Negative A/B Testing, Features & GBDT Training**
* **L6 Training Pair Construction (A/B Test)**:
  * Variant A: Includes hard negativedecoys (top coarse-score non-matches).
  * Variant B: Standard candidate pool sampling.
  * *Evaluates validation $F_{0.5}$ to select the winning pair sampler.*
* **L7 Pairwise Feature Vector**: 27+ features covering string distances, structured address digit conflicts, retrieval RRF ranks, and structural ratios.
* **L8 Model Training**: 5-fold cross-validated LightGBM with out-of-fold probability outputs.
* **L9 Calibration**: Conditional isotonic probability calibration on OOF predictions.

### **L10 – L13: Decision Engine, First Submission & Stretch Iterations**
* **L10 Decision Engine**: Joint tuning of match threshold $\tau_{\text{match}}$ and singleton threshold $\tau_s$ with explicit **France fallback veto rules**.
* **L10.5 Error Analysis & France Spot-Check**: Diagnostic error breakdown by country, source, and string length to pinpoint weaknesses.
* **L11 Test Inference & Output Verification**: Generates `matching_results.tsv`, `candidate_pairs.tsv`, and `blocking_report.md`. Validates schema via `utils/validate_submission.py`.
* **L11.5 First Portal Submission**: Live portal upload securing our baseline leaderboard score.
* **L12 Targeted Stretch Iteration**: Pick ONE high-impact stretch based on L10.5 error analysis:
  * *12a*: Phonetic Blocking (Soundex/Metaphone).
  * *12b*: Dense Retrieval (`BGE-M3` FAISS).
  * *12c*: Multilingual Cross-Encoder (`paraphrase-multilingual-MiniLM-L12-v2`) + Stacking Meta-Learner.
* **L12.5 Ablation Log**: Quantitative validation proving stretch gains on validation $F_{0.5}$.
* **L13 Final Packaging**: Assembles `<team_name>_submission.zip` with runnable source code, `requirements.txt`, and completed `Documentation_template.md`.

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
    │   ├── candidate_pairs.tsv         # Blocking candidate pairs (Audit Evaluation)
    │   └── blocking_report.md          # Candidate recall & reduction ratio report
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
