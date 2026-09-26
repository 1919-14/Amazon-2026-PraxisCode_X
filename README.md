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

### **Coverage, parity & memory hardening (verified)**

Seven failure modes found by auditing the pipeline are fixed and regression-tested:

1. **No silent coverage loss** — `utils/coverage.py` runs a preflight on L4/L5/L11:
   every requested country must have a complete upstream artifact, and the output
   must cover every Source-1 entity of the split. A truncated artifact (a dev cap
   left on disk) is detected too, and the message names the country, the reference
   count at stake and the exact command that fixes it. Guards exit with code 2
   unless `--allow-missing-countries` / `--allow-partial-coverage` is passed.
   `--countries` takes space- or comma-separated buckets; a bucket with no in-scope
   references (a typo, or France on the train split) is reported as out of scope
   instead of silently counting as zero references at risk, and a run where *no*
   requested bucket exists fails outright.
2. **Memory-bounded blocking** — Layer 3 is now block-wise and disk-backed
   (`l3_l5_blocking/blocked.py`): one candidate block index is resident at a time,
   references stream past it, and rows are decoded/written in chunks. Peak RSS is a
   function of the block size, not the country bucket (`--block-size`,
   `--ref-block-size`, `--vocab-sample`).
3. **Train/serve signal parity** — L5 writes a retrieval-signal sidecar next to the
   candidate set; L7 and L11 read *the same* artifact (L11 in lockstep, so it never
   holds ~10M pairs), and L11 refuses to score a booster trained with signals when
   the sidecar is missing.
4. **Tuned open-set policy** — `main_l10_open_set.py` holds a *pseudo-open* country
   out of training and grid searches the France veto parameters against real ground
   truth, then previews how many France entities a threshold would empty.
5. **Report history** — `utils/reports.py` merges every run into `runs[...]` so a
   multi-country sweep no longer overwrites the previous country's measurements.
6. **Deterministic retrieval** — the rarest-term selection and the top-k cut are
   canonical under ties, so the blocking artifact is reproducible run to run.
7. **Dead code removed** — the stale `src/metrics.py` / `src/test_core.py` pair and
   the duplicated nested walkthrough are gone; the official F0.5 formula is now
   asserted against the problem statement's worked example in the L1 tests.

### **L3 – L5: Fast Stratified Blocking & Adaptive Candidate Truncation**
* **L3 Country Stratification**: Records are partitioned into independent country buckets (`US`, `India`, `France`). Zero cross-country overhead. Each bucket is indexed block-by-block against a shared, sampled vocabulary so scores stay comparable across blocks and peak memory stays bounded.

#### Training-pool budget (measured)

Blocking cost scales as `references x candidate pool`. Measured: the France test bucket (259,452 references, 1.43M candidate pool) took **1,180 s**. Fully blocking all 2.2M training references would cost roughly ten more hours of L3 alone, so the supported workflow is a deterministic reference sample for the *training* pool:

```bash
python business_entity_resolution/src/main_l3.py --refs train --ref-sample 400000
python business_entity_resolution/src/main_l4.py --refs train
python business_entity_resolution/src/main_l5.py --refs train   # -> candidate_pairs_train.tsv + signal sidecar
python business_entity_resolution/src/main_l6.py --candidates business_entity_resolution/output/candidate_pairs_train.tsv
```

The sample is reproducible (`--ref-seed`) and recorded in the L3 report, and the coverage preflight reads that record so a sampled run is never mistaken for a truncated one. `--reuse-existing` resumes an interrupted multi-country sweep.
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
    │   ├── l*_report.json / .md        # Per-layer reports (history preserved per run)
    │   ├── l10_open_set_policy.json    # Tuned open-set (France) veto policy
    │   ├── matching_results.tsv        # Final entity matches (Leaderboard Upload)
    │   ├── candidate_pairs.tsv         # Blocking candidate pairs (Audit Evaluation)
    │   └── blocking_report.md          # Candidate recall & reduction ratio report
    └── src/
        ├── config.py / schemas.py / ingest.py / audit.py   # L0 foundation
        ├── deep_profile.py / inspect_samples.py            # Phase-0 profiling scripts
        ├── main_l0.py … main_l11.py                        # One entry point per layer
        ├── main_l10_open_set.py                            # Pseudo-open-set policy tuner
        ├── l1_validation/       # Split + official macro F0.5 scorer and threshold eval
        ├── l2_normalization/    # NFKC/accent/legal-suffix/address normalizer
        ├── l3_l5_blocking/      # block-wise engine, channels, RRF, truncation, paths
        ├── l6_l8_matching/      # pair sampling, 27 features, LightGBM, calibration, signals
        ├── l10_decision/        # decision rule, open-set policy tuner
        ├── l10_diagnostics/     # error analysis + France spot-check
        ├── l11_inference/       # streaming scoring, booster/calibrator loading
        ├── utils/               # cache, report history, coverage preflight
        └── requirements.txt     # Pinned dependencies

> Every package ships its own runnable `unit_tests.py` (no pytest needed).
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
