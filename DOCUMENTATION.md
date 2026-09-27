# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** PraxisCode X  
**Team Members:**  
- V S S K Sai Narayana  
- Sujeet Jaiswal  
- Sujeet Sahni  
**Submission Date:** September 27, 2026  

---

## 1. Executive Summary

We present an end-to-end, highly scalable, and production-grade Business Entity Resolution pipeline designed for the Amazon ML Challenge 2026. Our solution addresses cross-source entity linkage across 26.4M+ business records by combining multi-channel hybrid candidate blocking (High-IDF Token Index, Character N-gram TF-IDF, Multilingual E5 Dense Semantic Embeddings, and Phonetic Metaphone/Soundex encoding) with a 5-Fold Stacking Meta-Ensemble (LightGBM + CatBoost + XGBoost) and an adaptive decision engine with open-set veto policies. The pipeline achieves an outstanding candidate reduction ratio ($K \approx 5.5 - 6.0$ candidate pairs per reference entity) while preserving high recall ($\ge 98\%$) and optimizing specifically for the precision-weighted Macro $F_{0.5}$ metric.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across 26.4 million records revealed major real-world data noise and alignment challenges:
- **Transliteration and Script Variations:** Records across India and US datasets feature mixed-script tokens (Devanagari / Latin) and phonetic spelling permutations (e.g., phonetic variants in merchant names).
- **Address Irregularities & Missing Fields:** 15.2% of postal codes and significant street addresses are noisy, unstandardized, or missing.
- **Legal Entity Suffix Variations:** Heavy variation in corporate suffixes (`Pvt Ltd`, `LLC`, `SARL`, `Inc`, `Corp`, `GmbH`) which dilute string similarity if unnormalized.
- **Open-Set / Unseen Country Generalization:** France appears in test data without any labeled ground truth in the training set, requiring a robust zero-shot open-set rejection and veto policy.
- **High-Penalty False Positives:** The official evaluation metric is Macro $F_{0.5}$ ($\beta = 0.5$), which penalizes false positive merges twice as severely as false negatives.

### 2.2 Solution Strategy
Our architecture follows a layered, decoupled paradigm consisting of:
1. **L0–L2 Normalization & Parsing Engine:** Unicode NFKC normalization, diacritic stripping, legal suffix canonicalization, structured address component extraction, and script detection.
2. **L3–L5 Multi-Channel Hybrid Blocking:** Stratified by country (`US`, `India`, `France`), combining sparse inverted indexes, character TF-IDF, GPU-accelerated FAISS dense multilingual retrieval, and phonetic hash indexes fused via Reciprocal Rank Fusion (RRF).
3. **L6–L9 Feature Extraction & Stacking Ensemble:** 27+ pairwise interaction signals (string distance metrics, token Jaccard, address digit mismatch penalties, dense cosine similarity, phonetic overlap) fed into 5-fold cross-validated LightGBM, CatBoost, and XGBoost models combined via a calibrated Logistic Regression Meta-Learner.
4. **L10–L11 Decision Engine & Inference:** Joint threshold optimization ($\tau_{\text{match}} = 0.5$, $\tau_s = 0.1$, margin $= 0.05$), open-set singleton vetoing for low-confidence clusters, and transitive closure graph clustering.

**Approach Type:** Hybrid (Multi-Channel Blocking + Stacking Gradient Boosted Meta-Classifier + Graph-Based Decision Engine)  
**Core Innovation:** Memory-bounded streaming FAISS dense retrieval (`intfloat/multilingual-e5-small`) unified with phonetic indexing (Metaphone/Soundex) and a 3-way GBDT Stacking Meta-Learner tuned specifically for Macro $F_{0.5}$ open-set precision.

---

## 3. Candidate Generation (Blocking)

To reduce the comparison search space from billions of pairwise combinations down to an ultra-compact candidate pool without losing true matches, we implement a multi-channel blocking architecture:

- **Blocking channels used:**
  - **Channel A (Exact & Postal Hash):** Exact normalized core name + postal code / house number hash.
  - **Channel C (High-IDF Inverted Index):** BM25/TF-IDF token index filtering out high-frequency commercial stop-words (`store`, `traders`, `enterprises`).
  - **Channel D (Character N-gram TF-IDF):** 2-4 character n-gram cosine nearest neighbors for typo tolerance.
  - **Channel G (Phonetic & Metaphone):** Script-aware Double Metaphone and Soundex indexing to bridge pronunciation-based spelling divergences.
  - **Channel H (Dense Semantic Embeddings):** `intfloat/multilingual-e5-small` fp16 embeddings indexed via memory-bounded FAISS `IndexIVFPQ` for cross-lingual semantic alignment.
- **Candidate pairs generated:** $\approx 10.39$ million candidate pairs across $1.73$ million test references ($K \approx 6.0$ pairs per reference).
- **How true matches were preserved:** Multi-channel Reciprocal Rank Fusion (RRF) ensures that if a true match is missed by lexical matching (due to typos or language differences), dense embeddings or phonetic hashing retrieve it. Adaptive tiered truncation retains high-confidence candidates while discarding noisy tail candidates.

---

## 4. Matching Model

### Features Used (27+ Engineered Signals):
- **Name Signals:** Jaro-Winkler distance, Levenshtein distance, Token Sort Ratio, Token Set Ratio, Monge-Elkan similarity, Longest Common Subsequence ratio, Initialism match boolean.
- **Address Signals:** Structured token Jaccard overlap, house/building digit exact match / mismatch penalty, postal code match boolean, city/state equality, sequence edit distance.
- **Retrieval & Semantic Signals:** RRF fused reciprocal rank score, BM25 retrieval score, Dense cosine similarity from E5 embeddings, Phonetic token overlap ratio.
- **Structural Ratios:** Character length difference ratio, word count ratio, legal entity type compatibility.

### Model Type:
- **Level 1 Base Learners:** 
  1. LightGBM (Gradient Boosted Decision Trees, 5-fold CV)
  2. CatBoost (Categorical GBDT, 5-fold CV)
  3. XGBoost (Extreme Gradient Boosting, 5-fold CV)
- **Level 2 Meta-Learner:** Calibrated Stacking Classifier combining Out-Of-Fold probability outputs with non-linear interaction features.

### Threshold Selection Method:
- Grid search optimization on 5-fold Out-Of-Fold validation predictions directly maximizing the competition metric (Macro $F_{0.5}$).
- Decision parameters: $\tau_{\text{match}} = 0.50$, singleton threshold $\tau_s = 0.10$, ambiguity margin $= 0.05$.
- Open-set pseudo-validation veto policy tuned on held-out country distributions to prevent over-merging on unseen France test entities.

---

## 5. Results & Error Analysis

- **$F_{0.5}$ Score (macro validation):** $0.962 - 0.981$ (across 5-fold cross validation with full dense and phonetic features).
- **Candidate Coverage & Reduction:** $100.00\%$ reference coverage with a $>99.95\%$ reduction in total candidate space ($K \approx 6.0$).
- **Submission Validation:** Passed official `validate_submission.py` with 0 blocking errors (1,732,544 rows matched, schema strictly compliant).
- **Common False Positives (Wrong Merges):** Franchise businesses sharing identical brand names located in neighboring postal districts without explicit branch discriminators; mitigated via strict house number and street digit penalty features.
- **Common False Negatives (Missed Matches):** Extreme abbreviatures (e.g. 2-letter acronyms without address tokens); resolved by dense semantic similarity retrieval.

---

## 6. Conclusion

Our solution establishes a resilient, scalable, and memory-bounded framework for resolving business identities at scale across heterogeneous international sources. By fusing multi-modal blocking (sparse, dense, and phonetic) with a stacked gradient-boosted ensemble and $F_{0.5}$-optimized decision boundaries, the system delivers industry-leading entity resolution accuracy while adhering strictly to submission constraints and hardware limits.

---

## Appendix

### A. Code Artefacts & Structure
The complete codebase is located under `business_entity_resolution/` and structured as follows:

```
business_entity_resolution/
├── src/
│   ├── config.py / schemas.py / ingest.py / audit.py   # L0 Environment & Config
│   ├── l1_validation/       # Stratified split & Macro F0.5 evaluation
│   ├── l2_normalization/    # NFKC, accents, legal suffixes, addresses
│   ├── l3_l5_blocking/      # Multi-channel blocking (sparse, dense, phonetic, RRF)
│   ├── l6_l8_matching/      # Pairwise feature extraction, GBDT models, Stacking
│   ├── l10_decision/        # F0.5 optimization, open-set veto, graph clustering
│   ├── l11_inference/       # Streaming scoring, submission generation, validation
│   ├── run_scaled_pipeline.py # Unified end-to-end execution pipeline
│   └── requirements.txt     # Python dependencies
└── output/
    ├── matching_results.tsv # Final competition predictions (1,732,544 rows)
    └── candidate_pairs.tsv  # Blocking candidate pairs
```

#### Entry Point to Reproduce:
To execute the complete end-to-end training and inference pipeline:
```bash
# 1. Install dependencies
pip install -r business_entity_resolution/src/requirements.txt

# 2. Run full scaled pipeline (Dense + Phonetic + Stacking Ensemble)
python business_entity_resolution/src/run_scaled_pipeline.py \
    --mode test \
    --countries us india france \
    --model-kind stack \
    --dense --phon

# 3. Validate submission output
python "DATA SET/student_resource/utils/validate_submission.py" \
    --matching "business_entity_resolution/output/matching_results.tsv" \
    --candidate "business_entity_resolution/output/candidate_pairs.tsv" \
    --test-dir "DATA SET/student_resource/dataset/test"
```

### B. Hardware & Resource Compliance
- **Peak RAM:** $< 14 \text{ GB}$ (Streaming chunk-based pyarrow dataset processing).
- **GPU Acceleration:** CUDA-enabled FAISS for float16 embedding search (under 4GB VRAM peak).
- **Model Parameter Footprint:** $< 150\text{M}$ parameters (100% compliant with the $\le 8\text{B}$ parameter limit).
- **External Dependencies:** Zero external API or geocoding calls; 100% self-contained and reproducible.
