# Contributing to PraxisCode X — Business Entity Resolution

Thank you for your interest in contributing to the **PraxisCode X** Business Entity Resolution solution for the Amazon ML Challenge 2026.

## 👥 Core Team (PraxisCode X)
- **V S S K Sai Narayana**
- **Sujeet Jaiswal**
- **Sujeet Sahni**

---

## 🚀 Development Workflow

1. **Fork or Clone the Repository:**
   ```bash
   git clone https://github.com/1919-14/Amazon-2026-PraxisCode_X.git
   cd Amazon-2026-PraxisCode_X
   ```

2. **Environment Setup:**
   Create a clean Python 3.10+ virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r business_entity_resolution/src/requirements.txt
   ```

3. **Layer-by-Layer Modularity:**
   The architecture is organized into clean, independent layers:
   - `l0_setup`: Configuration, schemas, and dataset ingestion.
   - `l1_validation`: Stratified train/val split & official Macro $F_{0.5}$ scorer.
   - `l2_normalization`: String, legal suffix, and address normalizers.
   - `l3_l5_blocking`: Inverted indexes, character TF-IDF, dense multilingual FAISS retrieval, phonetic hashing, and adaptive candidate truncation.
   - `l6_l8_matching`: Pairwise feature extraction, LightGBM/CatBoost/XGBoost training, and stacking meta-learning.
   - `l10_decision`: $F_{0.5}$ threshold optimization, singleton veto policy, and graph clustering.
   - `l11_inference`: Streaming test inference, validation, and submission formatting.

4. **Testing & Validation:**
   Before submitting changes, ensure unit tests pass:
   ```bash
   python business_entity_resolution/src/l1_validation/unit_tests.py
   python business_entity_resolution/src/l2_normalization/unit_tests.py
   ```

5. **Submission Verification:**
   Validate your generated submission files:
   ```bash
   python "DATA SET/student_resource/utils/validate_submission.py" \
       --matching "business_entity_resolution/output/matching_results.tsv" \
       --candidate "business_entity_resolution/output/candidate_pairs.tsv" \
       --test-dir "DATA SET/student_resource/dataset/test"
   ```

---

## 📜 Code of Conduct & Standards
- Keep scripts memory-bounded (stream datasets in chunks via PyArrow, avoid loading full dataframes into memory).
- Ensure all models strictly comply with the competition rules ($\le 8\text{B}$ parameters, open-source MIT/Apache 2.0 weights).
- Zero external network dependencies during scoring and inference.
