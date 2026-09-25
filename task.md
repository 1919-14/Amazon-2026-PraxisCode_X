# Amazon ML Challenge 2026: Project Task Tracking 📋

**Team Name:** PraxisCode_X  
**Challenge Track:** Business Entity Resolution  
**Repository:** [1919-14/Amazon-2026-PraxisCode_X](https://github.com/1919-14/Amazon-2026-PraxisCode_X.git)  

---

## 📊 Phase 0: Research, Hardware & Data Profiling (COMPLETED ✅)

- [x] **Problem Statement & Rules Alignment**
  - [x] Analyzed macro $F_{0.5}$ metric with $2\times$ precision weight penalty.
  - [x] Analyzed singleton credit ($1.0$ for empty, $0.0$ for false positive).
  - [x] Verified constraints (Apache 2.0 / MIT models, $\le 8\text{B}$ params, zero external lookups).
  - [x] Adapted strategy for new rule: Candidate set size ($K \approx 5.5$) counts towards final ranking tie-breaker.
- [x] **Hardware & Environment Verification**
  - [x] Confirmed PyTorch + CUDA acceleration on NVIDIA GeForce RTX 4050 (6GB VRAM).
  - [x] Installed required packages (`lightgbm`, `xgboost`, `transformers`, `laya`, `rapidfuzz`).
- [x] **Comprehensive Data Profiling**
  - [x] Created streaming dataset auditor (`audit.py`).
  - [x] Analyzed noise patterns in sample pairs (`inspect_samples.py`).
  - [x] Generated deep statistical breakdown on 26.4M records (`deep_profile.py`).
  - [x] Published [`DATA_PROFILING_REPORT.md`](business_entity_resolution/DATA_PROFILING_REPORT.md) documenting ground truth cardinalities, non-Latin Indic scripts (Hindi, Tamil, Kannada), missing addresses, and open-set France test data.
- [x] **Repository Architecture Setup**
  - [x] Created `.gitignore` excluding large dataset TSVs (>100MB).
  - [x] Published 14-Layer Architecture specification in `README.md`.
  - [x] Committed and pushed initial codebase to GitHub `main` branch.

---

## 🏗️ 14-Layer Implementation Checklist

### **[L0 - L2] Foundation & Normalization**
- [x] **L0: Setup, Config & Schema Verification (COMPLETED ✅)**
  - [x] Verified data paths, directory schemas, and system environment.
  - [x] Implemented `src/config.py`, `src/schemas.py`, `src/ingest.py`, `src/audit.py`, `src/utils/cache.py`, `src/main_l0.py`.
  - [x] Generated `output/audit_train.json` and `output/audit_test.json`.
  - [x] Initialized artifact cache directories (`artifacts/normalized/`, `artifacts/blocking/`, etc.).
- [x] **L1: Validation Split & Macro $F_{0.5}$ Evaluator (COMPLETED ✅)**
  - [x] Implemented official macro F0.5 metric in `l1_validation/metrics.py` (`f_beta`, `precision_recall`, `macro_f05`, `evaluate_with_thresholds`).
  - [x] Implemented grouped 80/20 split in `l1_validation/split_generator.py` (seed=42, grouped by S1 ID, no country stratification).
  - [x] Saved split artifacts: `artifacts/splits/train_ids.json` (1,765,457) & `val_ids.json` (441,364).
  - [x] All 8 unit tests PASS in `l1_validation/unit_tests.py` (exit code 0).
  - [x] Ran `src/main_l1.py` — cardinality + baseline report confirmed; Peak RAM: 1,849 MB.
  - [x] Baseline F0.5 established: empty=0.0562 | uncalibrated pool (K≈5.5)=0.6305.
- [ ] **L2: Normalization Engine (`src/l2_normalization/normalizer.py`)**
  - [ ] Unicode NFKC normalization + accent stripping (`réseau` $\rightarrow$ `reseau`).
  - [ ] Multilingual script detection & transliteration flags (Hindi, Tamil, Kannada, Telugu).
  - [ ] Legal entity suffix canonicalization (`Pvt Ltd`, `LLC`, `SARL`, `Inc`, `Corp`).
  - [ ] Structured address parsing (`house_num`, `street`, `city`, `postal`, `state`, `country`).
  - [ ] Digit sequence extraction & URL/domain name cleaner (`maurewilliamscolombier.com` $\rightarrow$ name).

---

### **[L3 - L5] Fast Country-Stratified Blocking & Truncation**
- [ ] **L3: Country-Stratified Multi-Channel Blocking Engine (`src/l3_l5_blocking/`)**
  - [ ] L3a: Country bucket assignment (`US`, `India`, `France`).
  - [ ] L3b: Channel A — Exact normalized name + postal/house hash index.
  - [ ] L3c: Channel C — High-IDF token inverted index (prunes generic stopwords).
  - [ ] L3d: Channel D — Character 2-4 gram TF-IDF nearest neighbors.
  - [ ] L3e: Per-channel recall measurement on validation split.
- [ ] **L4: Reciprocal Rank Fusion (RRF)**
  - [ ] Implement RRF to merge sparse channels into a unified ranked candidate list per S1.
- [ ] **L5: Adaptive Candidate Truncation**
  - [ ] Coarse scoring + tiered candidate budget allocation ($K \approx 5.5 \text{ to } 7.0$).
  - [ ] Generate `output/candidate_pairs.tsv` (judged deliverable).
  - [ ] Generate `output/blocking_report.md` (recall & reduction ratio report).

---

### **[L6 - L9] Hard Negative Mining, Features & GBDT Training**
- [ ] **L6: Training Pair Construction & Hard Negative A/B Test (`src/l6_l8_matching/`)**
  - [ ] Build positive / hard negative / easy negative pair sampler (1 pos : 3 hard neg : 2 easy neg).
  - [ ] A/B test Variant A (with hard negs) vs Variant B (without hard negs) on validation $F_{0.5}$.
- [ ] **L7: Pairwise Feature Engineering Engine**
  - [ ] Name block (8 features: Levenshtein, Jaro-Winkler, Token Sort/Set, Char 3-gram, IDF overlap, legal suffix match).
  - [ ] Address block (9 features: house num exact, street token Jaccard, postal exact, postal 3-digit prefix, city overlap, `digit_string_conflict` boolean).
  - [ ] Retrieval signal block (6 features: exact key hit, RRF rank, retriever agreement count).
  - [ ] Structural block (4 features: same country, length ratios, missing address flags).
- [ ] **L8: GBDT Matcher Model Training**
  - [ ] Implement 5-fold cross-validated LightGBM (`train_lgbm.py`) with out-of-fold probability outputs.
- [ ] **L9: Calibration (Conditional)**
  - [ ] Evaluate conditional isotonic calibration on OOF predictions.

---

### **[L10 - L13] Decision Engine, Submission & Packaging**
- [ ] **L10: Decision Engine & Singleton Protection (`src/l10_decision/`)**
  - [ ] Joint grid search for match threshold $\tau_{\text{match}}$ and singleton threshold $\tau_s$.
  - [ ] Singleton guard: if max candidate score $<\tau_s$, predict empty list.
  - [ ] France fallback veto rule for open-set test entities.
- [ ] **L10.5: Diagnostic Error Analysis & France Spot-Check**
  - [ ] Run error breakdown by country, source file, and name/address length.
- [ ] **L11: Test Inference & Validation**
  - [ ] Execute full pipeline on test set (`test_source1/2/3.tsv`).
  - [ ] Generate `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
  - [ ] Pass `python DATA SET/student_resource/utils/validate_submission.py`.
- [ ] **L11.5: FIRST LEADERBOARD SUBMISSION 🚀**
  - [ ] Upload `matching_results.tsv` to Unstop Portal to lock in baseline score!
- [ ] **L12: Targeted Stretch Enhancements**
  - [ ] Option 12a: Phonetic blocking (Soundex / Double Metaphone).
  - [ ] Option 12b: Dense retrieval (`BGE-M3` FAISS on GPU).
  - [ ] Option 12c: Multilingual Cross-Encoder (`paraphrase-multilingual-MiniLM-L12-v2` / `laya`) + Stacking Meta-Learner.
- [ ] **L12.5: Ablation Log Validation**
  - [ ] Record validation $F_{0.5}$ gains in `ablation_log.md`.
- [ ] **L13: Final Submission Package**
  - [ ] Complete `Documentation_template.md`.
  - [ ] Verify reproducible `README.md` & `requirements.txt`.
  - [ ] Package into `<team_name>_submission.zip`.
