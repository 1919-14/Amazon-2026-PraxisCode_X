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
  - [x] All 10 unit tests PASS in `l1_validation/unit_tests.py` (exit code 0; 2 added in Phase 1: official-formula equivalence + singleton credit).
  - [x] Ran `src/main_l1.py` — cardinality + baseline report confirmed; Peak RAM: 1,849 MB.
  - [x] Baseline F0.5 established: empty=0.0562 | uncalibrated pool (K≈5.5)=0.6305.
- [x] **L2: Normalization Engine (`src/l2_normalization/`) (COMPLETED ✅)**
  - [x] Unicode NFKC normalization + accent stripping (`réseau` $\rightarrow$ `reseau`).
  - [x] Multilingual script detection & transliteration flags (Hindi, Tamil, Kannada, Telugu, Latin).
  - [x] Country-aware legal entity suffix canonicalization (`Pvt Ltd`, `LLC`, `SARL`, `Inc`, `Corp`).
  - [x] Structured address parsing (`addr_house_number`, `addr_postal`, `addr_state`, `addr_city`, `addr_digits`).
  - [x] Phonetic metaphone encoding for Latin names (`jellyfish.metaphone`), guarded for non-Latin.
  - [x] High-IDF business stopwords removal for `name_core`.
  - [x] Streaming 100K-row chunk pipeline to 245 Parquet shards (snappy compression, 4.4 GB on disk, peak RAM < 0.65 GB).

---

### **[L3 - L5] Fast Country-Stratified Blocking & Truncation**
- [x] **L3: Country-Stratified Multi-Channel Blocking Engine (`src/l3_l5_blocking/`) (COMPLETED ✅)**
  - [x] L3a: Country bucket assignment (`US`, `India`, `France`) — `buckets.py`.
  - [x] L3b: Channel A — Exact normalized name + postal/house hash index — `channels.py::ExactKeyChannel`.
  - [x] L3c: Channel C — High-IDF token inverted index (prunes generic tokens) — `channels.py::SparseTfidfChannel`.
  - [x] L3d: Channel D — Character 2-4 gram TF-IDF nearest neighbors — `channels.py::SparseTfidfChannel`.
  - [x] L3e: Per-channel recall measurement on validation split — `recall.py` (union recall 92.6% on a 15K-ref / 205K-pool real-data check).
  - [x] Engine + entry point (`engine.py`, `main_l3.py`) + 8 unit tests PASS.
- [x] **L4: Reciprocal Rank Fusion (RRF) (COMPLETED ✅)**
  - [x] Implemented `rrf.py` — `rrf_scores`, `fuse`, `ranked_ids` (configurable weights + smoothing constant `k`).
  - [x] Merges channels A/C/D into one ranked candidate list per S1, preserving the union (no recall loss).
  - [x] `main_l4.py` reads L3 artifacts, tunes `k` over `RRF_K_GRID` against validation via recall@N, writes fused `l4_*.parquet` for L5.
  - [x] 5 RRF unit tests PASS (total package tests: 13).
- [x] **L5: Adaptive Candidate Truncation (COMPLETED ✅)**
  - [x] Implemented `truncate.py` — `coarse_score` (normalized RRF + channel agreement + exact-key hit) and `adaptive_truncate` (ratio threshold + k_min/k_max clamp).
  - [x] `main_l5.py` tunes the retention ratio to land average K in the `[5.5, 7.0]` band (got avg K = 6.44 on a smoke run).
  - [x] Generates `output/candidate_pairs.tsv` (test) / `candidate_pairs_val.tsv` (dev) — judged deliverable, validated TSV format.
  - [x] Generates `output/blocking_report.md` + `l5_blocking_report.json` (recall & reduction ratio report).
  - [x] 4 L5 unit tests PASS (total package tests: 17).

---

### **[L6 - L9] Hard Negative Mining, Features & GBDT Training**
- [x] **L6: Training Pair Construction & Hard Negative A/B Test (`src/l6_l8_matching/`) (COMPLETED ✅)**
  - [x] Implemented `pairs.py` — positive / hard-negative / easy-negative sampler (1 pos : 3 hard : 2 easy; singletons get fixed negatives) + streaming Parquet writer in `main_l6.py`.
  - [x] Produces `artifacts/train_pairs/variant_a.parquet` (pos+hard+easy) and `variant_b.parquet` (pos+easy) + `output/l6_pairs_report.json`.
  - [x] 5 L6 unit tests PASS; streaming parquet schema/dtypes verified.
  - [x] A/B test Variant A vs B on out-of-fold macro $F_{0.5}$ — executed in L8 (`main_l8.py`).
- [x] **L7: Pairwise Feature Engineering Engine (COMPLETED ✅)**
  - [x] Name block (8): `name_ratio`, `name_jaro_winkler`, `name_token_sort`, `name_token_set`, `name_char3_jaccard`, `name_token_jaccard`, `name_idf_overlap`, `name_legal_suffix_match`.
  - [x] Address block (9): `addr_house_exact`, `addr_street_jaccard`, `addr_postal_exact`, `addr_postal_prefix3`, `addr_city_exact`, `addr_state_match`, `addr_token_jaccard`, `addr_char3_jaccard`, `addr_digit_conflict`.
  - [x] Retrieval block (6): `ret_exact_key_hit`, `ret_rrf_score`, `ret_retriever_agreement`, `ret_candidate_rank`, `ret_retrieved`, `ret_rank_inverse` (RRF/agreement resolved from the L5 signal sidecar, legacy L3/L4 join, or explicit zeros — provenance recorded and checked at L11).
  - [x] Structural block (4): `struct_same_country`, `struct_name_len_ratio`, `struct_addr_len_ratio`, `struct_addr_missing_xor`.
  - [x] Implemented `features.py` + `main_l7.py` (streaming Parquet); 8 L7 unit tests PASS (package total 18; 3 added in Phase 1 for the signal sidecar + retrieval join).
- [x] **L8: GBDT Matcher Model Training (COMPLETED ✅)**
  - [x] Implemented `model.py` — grouped 5-fold LightGBM (`GroupKFold` on `s1_id` prevents entity leakage) with OOF probabilities + `train_full` export.
  - [x] `main_l8.py` trains each variant, tunes `(tau_match, tau_s)` with the L1 scorer on OOF scores, and exports `artifacts/models/oof_<v>.parquet` + `lgbm_variant_<v>.txt`.
  - [x] A/B comparison of Variant A vs B on out-of-fold macro F0.5 → `output/l8_model_report.json`.
  - [x] 1 L8 unit test PASS (total package tests: 11).
- [x] **L9: Calibration (Conditional) (COMPLETED ✅)**
  - [x] Implemented `calibration.py` — reliability diagnostics (`brier`, `log_loss`, `reliability_curve`, `ece`), grouped cross-validated isotonic (`calibrate_grouped`) and knot serialization for inference.
  - [x] `main_l9.py` evaluates `none` / `global` / `per_country` on **retrieved** OOF pairs and adopts a calibrator only when it lowers ECE by `--min-ece-improvement` (default 0.005) **and** does not degrade best macro $F_{0.5}$.
  - [x] Exports `artifacts/models/oof_calibrated_<v>.parquet`, `artifacts/models/isotonic_<v>.json`, `output/l9_calibration_report.json/.md`.
  - [x] 4 L9 unit tests PASS (total package tests: 15).
  - [x] **Key insight**: isotonic regression is monotonic → ranking-preserving → cannot change the best achievable macro $F_{0.5}$; calibration is adopted purely for probability reliability (feeds per-country decision rules / stacking).

---

### **[L10 - L13] Decision Engine, Submission & Packaging**
- [x] **L10: Decision Engine & Singleton Protection (`src/l10_decision/`) (COMPLETED ✅)**
  - [x] `decision.py` — production copy of the scorer's rule (`apply_decision_rule`) so tuned thresholds transfer without a reimplementation gap.
  - [x] Joint grid search for match threshold $\tau_{\text{match}}$ and singleton threshold $\tau_s$ (`tune_thresholds`, grid in `config.TAU_MATCH_GRID` / `TAU_S_GRID`).
  - [x] Singleton guard: if max candidate score $<\tau_s$, predict empty list.
  - [x] France fallback veto rule for open-set test entities: boosted $\tau_s$ (`L10_OPEN_SET_TAU_BOOST`) + minimum-confidence veto (`L10_OPEN_SET_VETO_MIN_CONFIDENCE`).
  - [x] `main_l10.py` consumes L9-calibrated (or L8) OOF scores, applies the rule, writes `output/matching_results[_val].tsv` + `output/l10_decision_report.json/.md`.
  - [x] 11 L10 unit tests PASS (rule-vs-scorer parity, singleton guard, margin/`tau_match` interaction, open-set veto + boost, TSV format).
- [x] **L10.5: Diagnostic Error Analysis & France Spot-Check (COMPLETED ✅)**
  - [x] `l10_diagnostics/diagnostics.py` — per-entity error classes (`true_singleton` / `false_positive` / `miss` / `exact` / `partial`) with F0.5 mass, plus aggregation helpers.
  - [x] Error breakdown by **country**, **matched-source composition** (S2 / S3 / both), **name length**, **address length / missing-address**, and **script**.
  - [x] France spot-check for open-set test entities: volume, top-score quantiles, and the highest-confidence predictions for manual review.
  - [x] `main_l10_diagnostics.py` writes `output/l10_diagnostics_report.json/.md`.
  - [x] 8 L10.5 unit tests PASS; sanity check confirms macro F0.5 matches the official scorer.
- [x] **L11: Test Inference & Validation (`src/l11_inference/`) (CODE COMPLETE ✅ — end-to-end run pending)**
  - [x] `scorer.py` — loads the exported LightGBM booster (`lgbm_variant_<v>.txt`) and the optional L9 isotonic calibrator; LightGBM imported before scikit-learn (Windows OpenMP).
  - [x] `inference.py` — streaming reference-batch inference: reads L5 `candidate_pairs.tsv`, computes the 27 L7 features per pair, scores with the booster, calibrates, and applies the L10 decision engine.
  - [x] Memory-bounded: only candidates at/above `tau_match` are retained (lossless — the L10 margin rule can never keep below `tau_match`); IDF built once in a dedicated streaming pass.
  - [x] `main_l11.py` generates `output/matching_results.tsv` + `output/candidate_pairs.tsv` (full Source-1 coverage) and invokes `DATA SET/student_resource/utils/validate_submission.py`.
  - [x] Thresholds resolve CLI → L10 report → L8 report → defaults; no ground truth needed.
  - [x] Retrieval-signal sidecar read in lockstep with the candidate file (train/serve parity), plus a coverage preflight that blocks partial submissions.
  - [x] Writes the per-pair score table (`artifacts/scores/`) that `main_l10.py --split test --scores` documents.
  - [x] 8 L11 unit tests PASS; smoke proves the output format passes the official validator.
  - [ ] Full end-to-end test run (pending the complete L3→L5→L8 chain + trained booster).

---

## 🛡️ Phase 1: Correctness, Coverage & Memory Hardening (COMPLETED ✅)

A full code audit surfaced seven defects that would have cost the submission
silently. All seven are fixed, regression-tested, and verified on real data:

- [x] **Silent coverage loss (critical)**
  - [x] `utils/coverage.py`: preflight on L4/L5/L11 - missing *or truncated* per-country artifacts become a hard error (exit code 2) naming the country, the reference count at risk and the fix command.
  - [x] Verified on the real workspace: the leftover 5,000-reference US artifact is now rejected (`5,000/264,995 references (1.9%)`) instead of silently emptying ~260k references.
  - [x] L11 additionally checks that the candidate file covers the split's Source-1 set (`--allow-partial-coverage` to override).
  - [x] The guard's own measurement is typo-proof: `--countries` accepts space- or comma-separated buckets, a bucket with zero in-scope references is reported as out of scope rather than as "~0 references at risk", and a request where no bucket exists at all fails with the known-bucket list. Verified on the real workspace (`us,india,france` reports 1,732,544 references at risk in 5.5 s; `usa` is rejected).
- [x] **L3 memory blow-up**
  - [x] `l3_l5_blocking/blocked.py`: block-wise, disk-backed country index (shared sampled vocabulary + IDF, one block resident, streaming rows and recall).
  - [x] Peak RSS now depends on the block size: measured 298 MB on a real capped test-split run versus multi-GB for a full train bucket before.
  - [x] Equivalence test proves the block-wise engine reproduces the classic engine's candidates exactly.
- [x] **Train/serve skew on the retrieval block**
  - [x] `l6_l8_matching/signals.py`: L5 writes a candidate-signal sidecar; L7 and L11 read the same artifact (L11 in lockstep, memory-bounded).
  - [x] L7 records provenance; L8 copies it into the model report; L11 refuses to score with the wrong convention.
- [x] **Unvalidatable France veto**
  - [x] `l10_decision/open_set.py` + `main_l10_open_set.py`: pseudo-open-country grid search of `(boost, veto_min_confidence)` against real ground truth, persisted to `output/l10_open_set_policy.json` and loaded by L10/L11.
  - [x] Also reports the open-set country's top-score quantiles and how many entities each veto threshold would empty.
- [x] **Clobbered reports**
  - [x] `utils/reports.py`: every run is merged into `runs[...]` (L3/L4/L5/L8/L10-open-set/L11) with legacy reports migrated instead of dropped.
- [x] **Dead / duplicated code and stale docs**
  - [x] Deleted `src/metrics.py`, `src/test_core.py` (broken import) and the duplicated nested `business_entity_resolution/WALKTHROUGH.md`.
  - [x] Added an official-formula regression test (problem-statement worked example, 0.714) plus singleton-credit coverage in the L1 suite.
  - [x] `main_l10.py --split test` now actually finds the per-pair score table it documents; README structure/counts corrected.
- [x] **Remaining-chain cost**
  - [x] `--ref-sample` / `--ref-seed` on L3 for a reproducible training pool, `--reuse-existing` on L3/L4 to resume sweeps, and a documented budget plan (France test bucket measured at 1,180 s for 259k refs x 1.43M pool).
  - [x] Suite status: 90 tests green across L1 (10), L3-L5 (22), L6-L9 (18), L10 (14), L10.5 (8), L11 (8) and utils (10).
- [x] **L11.5: FIRST LEADERBOARD SUBMISSION 🚀**
  - [x] Upload `matching_results.tsv` to Unstop Portal to lock in baseline score! (Achieved **0.788** on public leaderboard)
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
