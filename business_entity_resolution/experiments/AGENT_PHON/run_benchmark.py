"""Comprehensive benchmark script for Phonetic & Transliteration Channel on India Train.

Evaluates:
- Exact Canonical Key Channel
- Exact Metaphone Key Channel
- Phonetic Sparse TF-IDF Channel
- Combined Phonetic Channel
- Recall@5, 10, 20, 50, 200
- Union Recall vs train_ground_truth.tsv
- Macro F0.5 Oracle Score:
    def f_beta(pred, true, b=0.5):
        if not true: return 1.0 if not pred else 0.0
        if not pred: return 0.0
        tp = len(pred & true)
        if tp == 0: return 0.0
        P = tp / len(pred); R = tp / len(true); return (1 + b*b)*P*R / (b*b*P + R)
- Breakdown by script_type: Latin vs Devanagari vs Other Indic (Tamil, Telugu, Kannada, Gujarati, Bengali, etc.)
- Saves deliverable parquet: artifacts/blocking/phon_train_100k_country=india.parquet
"""

from __future__ import annotations

import glob
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import jellyfish
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from l2_normalization.transliteration import romanize, detect_indic_script

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Normalization & Canonical Key Logic
# ---------------------------------------------------------------------------

LEGAL_SUFFIX_REGEX = re.compile(
    r"\b("
    r"private\s+limited|pvt\s+ltd|pvt\s+limited|private\s+ltd|private|limited|ltd|pvt|llp|"
    r"limited\s+liability\s+partnership|limited\s+liability\s+company|llc|inc|incorporated|"
    r"corp|corporation|co|company|"
    r"praiveta\s+limiteda|praiveta\s+limited|praiveta|limiteda|praivet\s+limited|praivet|"
    r"praivarr\s+limirrad|praivatu\s+limitada|praivett\s+limitett|"
    r"pra\s+li|pra|li|elaelapi|el\s+el\s+pi|elelbhi|elelpi|el\s+el\s+bhi"
    r")\b",
    re.IGNORECASE,
)

GENERIC_WORDS_REGEX = re.compile(
    r"\b("
    r"enterprises|enterprise|industries|industry|solutions|solution|technologies|technology|"
    r"services|service|associates|associate|international|national|global|india|"
    r"holdings|holding|group|venture|ventures|trading|traders|agency|agencies|"
    r"consultants|consultant|consultancy|consulting|commercial|commercials|"
    r"products|product|systems|system|management|properties|property|"
    r"impex|exports|export|imports|import|logistics|infrastructure|infra"
    r")\b",
    re.IGNORECASE,
)


def extract_canonical_core(name: str, script_hint: str = "latin") -> str:
    """Romanize Indic name (if needed) and strip legal suffixes."""
    if not name:
        return ""
    if script_hint == "latin":
        cleaned = LEGAL_SUFFIX_REGEX.sub(" ", name.lower())
    else:
        rom = romanize(name, script_hint).lower()
        cleaned = LEGAL_SUFFIX_REGEX.sub(" ", rom)
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    return " ".join(tokens)


def extract_phonetic_string(name: str, script_hint: str = "latin") -> str:
    """Return space-separated metaphone codes of core tokens."""
    core = extract_canonical_core(name, script_hint)
    tokens = core.split()
    codes = []
    for tok in tokens:
        if tok.isalpha() and len(tok) >= 2:
            m = jellyfish.metaphone(tok)
            if m:
                codes.append(m)
    return " ".join(codes)


def extract_phonetic_doc(name: str, script_hint: str = "latin") -> str:
    """Return multi-representation phonetic document for TF-IDF inverted index."""
    core = extract_canonical_core(name, script_hint)
    tokens = core.split()
    ph_tokens = []
    for tok in tokens:
        if tok.isalpha() and len(tok) >= 2:
            m = jellyfish.metaphone(tok)
            if m:
                ph_tokens.append(f"M_{m}")
            s = jellyfish.soundex(tok)
            if s:
                ph_tokens.append(f"S_{s}")
    return " ".join(ph_tokens)


# ---------------------------------------------------------------------------
# Metric Functions
# ---------------------------------------------------------------------------

def f_beta(pred: set[str], true: set[str], b: float = 0.5) -> float:
    """Challenge metric: Macro F0.5."""
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    P = tp / len(pred)
    R = tp / len(true)
    return (1.0 + b * b) * P * R / (b * b * P + R)


def compute_oracle_macro_f05(
    reference_ids: Sequence[str],
    predicted_candidates: Sequence[Sequence[str]],
    ground_truth: Mapping[str, set[str]],
    k: int = 50,
) -> float:
    """Compute Oracle Macro F0.5 if a perfect matcher selects top_k ∩ GT."""
    scores = []
    for ref_id, cands in zip(reference_ids, predicted_candidates):
        true_set = ground_truth.get(ref_id, set())
        cand_set = set(cands[:k])
        oracle_pred = cand_set & true_set
        score = f_beta(oracle_pred, true_set, b=0.5)
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def evaluate_recall(
    reference_ids: Sequence[str],
    predicted_candidates: Sequence[Sequence[str]],
    ground_truth: Mapping[str, set[str]],
    cand_script_map: Mapping[str, str],
    cutoffs: Sequence[int] = (5, 10, 20, 50, 200),
) -> dict:
    """Compute recall@k overall and broken down by ground-truth candidate script_type."""
    total_pairs = {"overall": 0, "latin": 0, "devanagari": 0, "other_indic": 0}
    hit_pairs = {cat: {k: 0 for k in cutoffs} for cat in total_pairs}
    
    # Count total ground truth pairs
    for ref_id in reference_ids:
        true_set = ground_truth.get(ref_id, set())
        for cand_id in true_set:
            total_pairs["overall"] += 1
            script = cand_script_map.get(cand_id, "latin")
            if script == "latin":
                total_pairs["latin"] += 1
            elif script == "devanagari":
                total_pairs["devanagari"] += 1
            else:
                total_pairs["other_indic"] += 1

    # Count hits at cutoffs
    for ref_id, cands in zip(reference_ids, predicted_candidates):
        true_set = ground_truth.get(ref_id, set())
        if not true_set:
            continue
        for k in cutoffs:
            cands_k = set(cands[:k])
            hits = cands_k & true_set
            hit_pairs["overall"][k] += len(hits)
            for hit_cand in hits:
                script = cand_script_map.get(hit_cand, "latin")
                if script == "latin":
                    hit_pairs["latin"][k] += 1
                elif script == "devanagari":
                    hit_pairs["devanagari"][k] += 1
                else:
                    hit_pairs["other_indic"][k] += 1

    summary = {}
    for cat in total_pairs:
        tot = total_pairs[cat]
        summary[cat] = {"total_gt_pairs": tot}
        for k in cutoffs:
            h = hit_pairs[cat][k]
            rec = h / tot if tot > 0 else 0.0
            summary[cat][f"recall@{k}"] = rec
            summary[cat][f"hits@{k}"] = h

    return summary


def print_recall_table(res: dict):
    cutoffs = [5, 10, 20, 50, 200]
    cats = [("overall", "Overall"), ("latin", "Latin Candidates"), ("devanagari", "Devanagari Candidates"), ("other_indic", "Other Indic Candidates")]
    
    header = f"{'Category':<25} | {'Pairs':<8} | " + " | ".join(f"R@{k:<3}" for k in cutoffs)
    print("-" * len(header), flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for cat_key, cat_label in cats:
        if cat_key in res:
            tot = res[cat_key]["total_gt_pairs"]
            r_strs = [f"{res[cat_key].get(f'recall@{k}', 0.0)*100:5.2f}%" for k in cutoffs]
            print(f"{cat_label:<25} | {tot:<8} | " + " | ".join(r_strs), flush=True)
    print("-" * len(header), flush=True)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    print("=" * 80, flush=True)
    print("PHONETIC + TRANSLITERATION CHANNEL BENCHMARK (INDIA 100K REFS)", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Ground Truth
    t0 = time.time()
    print("Loading Ground Truth...", flush=True)
    gt = {}
    with open("DATA SET/student_resource/dataset/train/train_ground_truth.tsv", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            if len(p) == 2:
                gt[p[0]] = set(p[1].split(","))
    print(f"Loaded {len(gt):,} ground truth entries in {time.time() - t0:.2f}s", flush=True)

    # 2. Load 100k India Reference Sample
    t0 = time.time()
    print("\nLoading 100k India S1 References...", flush=True)
    l3_art_path = "business_entity_resolution/artifacts/blocking/l3_train_train_country=india.parquet"
    if os.path.exists(l3_art_path):
        ref_ids_table = pq.read_table(l3_art_path, columns=["source1_entity_id"])
        target_ref_ids_list = ref_ids_table["source1_entity_id"].to_pylist()
        target_ref_ids_set = set(target_ref_ids_list)
        print(f"Found {len(target_ref_ids_set):,} target reference IDs from {l3_art_path}", flush=True)
    else:
        target_ref_ids_list = None
        target_ref_ids_set = None

    s1_cols = ["entity_id", "country_norm", "name_raw", "name_norm", "name_core", "addr_norm", "addr_postal", "addr_house_number", "script_type"]
    s1_files = sorted(glob.glob("business_entity_resolution/artifacts/normalized/train_s1/part_*.parquet"))
    
    ref_records = []
    for f in s1_files:
        t = pq.read_table(f, columns=s1_cols)
        df = t.to_pandas()
        df_india = df[df["country_norm"] == "india"]
        if target_ref_ids_set:
            df_india = df_india[df_india["entity_id"].isin(target_ref_ids_set)]
        ref_records.append(df_india)
        if target_ref_ids_set and sum(len(r) for r in ref_records) >= len(target_ref_ids_set):
            break

    ref_df = pd.concat(ref_records, ignore_index=True)
    if target_ref_ids_list:
        ref_df = ref_df.drop_duplicates(subset=["entity_id"]).set_index("entity_id")
        ref_df = ref_df.reindex(target_ref_ids_list).reset_index()
    else:
        ref_df = ref_df.head(100000)

    print(f"Loaded {len(ref_df):,} reference entities in {time.time() - t0:.2f}s", flush=True)

    # 3. Load Candidate Pool (S2 + S3 India records)
    t0 = time.time()
    print("\nLoading Candidate Pool (S2 + S3 India records)...", flush=True)
    cand_cols = ["entity_id", "country_norm", "name_raw", "name_norm", "name_core", "addr_norm", "addr_postal", "addr_house_number", "script_type"]
    
    s2_files = sorted(glob.glob("business_entity_resolution/artifacts/normalized/train_s2/part_*.parquet"))
    s3_files = sorted(glob.glob("business_entity_resolution/artifacts/normalized/train_s3/part_*.parquet"))
    
    # For full coverage, load all candidate shards for India
    print(f"Reading {len(s2_files)} S2 shards and {len(s3_files)} S3 shards...", flush=True)
    cand_records = []
    for idx, f in enumerate(s2_files):
        t = pq.read_table(f, columns=cand_cols)
        df = t.to_pandas()
        cand_records.append(df[df["country_norm"] == "india"])
    for idx, f in enumerate(s3_files):
        t = pq.read_table(f, columns=cand_cols)
        df = t.to_pandas()
        cand_records.append(df[df["country_norm"] == "india"])

    cand_df = pd.concat(cand_records, ignore_index=True).drop_duplicates(subset=["entity_id"])
    print(f"Loaded {len(cand_df):,} total candidates in {time.time() - t0:.2f}s", flush=True)
    print(f"Candidate script distribution:\n{cand_df['script_type'].value_counts()}", flush=True)

    # Create mapping from cand_id -> script_type for recall evaluation
    cand_script_map = dict(zip(cand_df["entity_id"], cand_df["script_type"]))

    # 4. Feature Extraction
    t0 = time.time()
    print("\nPre-computing canonical keys, phonetic strings, and phonetic documents...", flush=True)

    cand_names = cand_df["name_raw"].fillna("").tolist()
    cand_scripts = cand_df["script_type"].fillna("latin").tolist()
    cand_postals = cand_df["addr_postal"].fillna("").tolist()
    cand_houses = cand_df["addr_house_number"].fillna("").tolist()
    cand_ids = cand_df["entity_id"].tolist()

    ref_names = ref_df["name_raw"].fillna("").tolist()
    ref_scripts = ref_df["script_type"].fillna("latin").tolist()
    ref_postals = ref_df["addr_postal"].fillna("").tolist()
    ref_houses = ref_df["addr_house_number"].fillna("").tolist()
    ref_ids = ref_df["entity_id"].tolist()

    # Pre-extract canonical cores and phonetic docs
    cand_cores = [extract_canonical_core(n, s) for n, s in zip(cand_names, cand_scripts)]
    cand_ph_keys = [extract_phonetic_string(n, s) for n, s in zip(cand_names, cand_scripts)]
    cand_ph_docs = [extract_phonetic_doc(n, s) for n, s in zip(cand_names, cand_scripts)]

    ref_cores = [extract_canonical_core(n, s) for n, s in zip(ref_names, ref_scripts)]
    ref_ph_keys = [extract_phonetic_string(n, s) for n, s in zip(ref_names, ref_scripts)]
    ref_ph_docs = [extract_phonetic_doc(n, s) for n, s in zip(ref_names, ref_scripts)]

    print(f"Pre-computation finished in {time.time() - t0:.2f}s", flush=True)

    # 5. Build Indices
    t0 = time.time()
    print("\nBuilding Exact Canonical & Phonetic Inverted Indices...", flush=True)

    by_canonical: dict[str, list[int]] = defaultdict(list)
    by_canonical_postal: dict[str, list[int]] = defaultdict(list)
    by_canonical_house: dict[str, list[int]] = defaultdict(list)

    by_phonetic: dict[str, list[int]] = defaultdict(list)
    by_phonetic_postal: dict[str, list[int]] = defaultdict(list)

    max_post = 2000
    for idx, (core, ph, postal, house) in enumerate(zip(cand_cores, cand_ph_keys, cand_postals, cand_houses)):
        if core:
            if len(by_canonical[core]) < max_post:
                by_canonical[core].append(idx)
            if postal:
                kp = f"{core}|{postal}"
                if len(by_canonical_postal[kp]) < max_post:
                    by_canonical_postal[kp].append(idx)
            if house:
                kh = f"{core}|{house}"
                if len(by_canonical_house[kh]) < max_post:
                    by_canonical_house[kh].append(idx)
        if ph:
            if len(by_phonetic[ph]) < max_post:
                by_phonetic[ph].append(idx)
            if postal:
                kp = f"{ph}|{postal}"
                if len(by_phonetic_postal[kp]) < max_post:
                    by_phonetic_postal[kp].append(idx)

    print(f"Exact indices built in {time.time() - t0:.2f}s (canonical keys: {len(by_canonical):,}, phonetic keys: {len(by_phonetic):,})", flush=True)

    # 6. Build and Query Phonetic Sparse TF-IDF Index
    t0 = time.time()
    print("\nBuilding Phonetic Sparse TF-IDF Index...", flush=True)
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.03,
        sublinear_tf=True,
        dtype=np.float32,
    )
    cand_tfidf_matrix = tfidf.fit_transform(cand_ph_docs).tocsr()
    print(f"Fitted TF-IDF matrix: {cand_tfidf_matrix.shape} with {cand_tfidf_matrix.nnz:,} non-zeros in {time.time() - t0:.2f}s", flush=True)

    t0 = time.time()
    print("Transforming Reference Queries...", flush=True)
    ref_tfidf_matrix = tfidf.transform(ref_ph_docs).tocsr()
    print(f"Transformed references in {time.time() - t0:.2f}s", flush=True)

    # 7. Query Generation
    t0 = time.time()
    print("\nRunning Multi-Channel Phonetic Retrieval for 100,000 References...", flush=True)

    k_retrieve = 200
    cand_doc_matrix_t = cand_tfidf_matrix.T.tocsc()
    n_refs = len(ref_ids)
    batch_size = 5000

    predicted_candidates_phon: list[list[str]] = []
    predicted_scores_phon: list[list[float]] = []

    pred_exact_canonical: list[list[str]] = []
    pred_exact_phonetic: list[list[str]] = []
    pred_sparse_phonetic: list[list[str]] = []

    for b_start in range(0, n_refs, batch_size):
        b_end = min(b_start + batch_size, n_refs)
        
        # Batch Sparse Matrix Multiply
        q_batch = ref_tfidf_matrix[b_start:b_end]
        scores_batch = (q_batch @ cand_doc_matrix_t).tocsr()

        for idx_in_batch, i in enumerate(range(b_start, b_end)):
            core = ref_cores[i]
            ph = ref_ph_keys[i]
            postal = ref_postals[i]
            house = ref_houses[i]

            # 1. Exact Canonical hits
            c_hits = []
            if core:
                if postal and f"{core}|{postal}" in by_canonical_postal:
                    c_hits.extend(by_canonical_postal[f"{core}|{postal}"])
                if house and f"{core}|{house}" in by_canonical_house:
                    c_hits.extend(by_canonical_house[f"{core}|{house}"])
                if core in by_canonical:
                    c_hits.extend(by_canonical[core])

            # 2. Exact Phonetic hits
            p_hits = []
            if ph:
                if postal and f"{ph}|{postal}" in by_phonetic_postal:
                    p_hits.extend(by_phonetic_postal[f"{ph}|{postal}"])
                if ph in by_phonetic:
                    p_hits.extend(by_phonetic[ph])

            # 3. Sparse TF-IDF hits
            row_start = scores_batch.indptr[idx_in_batch]
            row_end = scores_batch.indptr[idx_in_batch + 1]
            s_hits = []
            s_scores = []
            if row_start < row_end:
                cols = scores_batch.indices[row_start:row_end]
                data = scores_batch.data[row_start:row_end]
                if len(cols) <= k_retrieve:
                    order = np.lexsort((cols, -data))
                    s_hits = cols[order].tolist()
                    s_scores = data[order].tolist()
                else:
                    part = np.argpartition(-data, k_retrieve)[:k_retrieve]
                    best_cols = cols[part]
                    best_data = data[part]
                    order = np.lexsort((best_cols, -best_data))
                    s_hits = best_cols[order].tolist()
                    s_scores = best_data[order].tolist()

            # Record per-channel predictions (mapped to entity_ids)
            pred_exact_canonical.append([cand_ids[d] for d in c_hits[:k_retrieve]])
            pred_exact_phonetic.append([cand_ids[d] for d in p_hits[:k_retrieve]])
            pred_sparse_phonetic.append([cand_ids[d] for d in s_hits[:k_retrieve]])

            # Merge and score combined phonetic channel:
            cand_score_map: dict[str, float] = {}
            
            # Add sparse hits
            for d_idx, sc in zip(s_hits, s_scores):
                cid = cand_ids[d_idx]
                cand_score_map[cid] = max(cand_score_map.get(cid, 0.0), float(sc))

            # Add phonetic exact hits
            for d_idx in p_hits[:100]:
                cid = cand_ids[d_idx]
                cand_score_map[cid] = max(cand_score_map.get(cid, 0.0), 2.0)

            # Add canonical exact hits
            for d_idx in c_hits[:100]:
                cid = cand_ids[d_idx]
                cand_score_map[cid] = max(cand_score_map.get(cid, 0.0), 3.0)

            # Rank best-first
            sorted_items = sorted(cand_score_map.items(), key=lambda x: -x[1])[:k_retrieve]
            merged_ids = [k for k, _ in sorted_items]
            merged_scores = [float(v) for _, v in sorted_items]

            predicted_candidates_phon.append(merged_ids)
            predicted_scores_phon.append(merged_scores)

        if (b_end // 20000) > (b_start // 20000) or b_end == n_refs:
            print(f"  Processed {b_end:,} / {n_refs:,} references in {time.time() - t0:.2f}s...", flush=True)

    print(f"Completed retrieval for all {n_refs:,} references in {time.time() - t0:.2f}s!", flush=True)

    # 8. Evaluation & Recall Reporting
    print("\n" + "=" * 80, flush=True)
    print("EVALUATION RESULTS (INDIA 100K REFERENCES)", flush=True)
    print("=" * 80, flush=True)

    print("\n--- 1. Canonical Exact Key Channel ---", flush=True)
    res_canon = evaluate_recall(ref_ids, pred_exact_canonical, gt, cand_script_map)
    print_recall_table(res_canon)

    print("\n--- 2. Phonetic Exact Key Channel ---", flush=True)
    res_phon_exact = evaluate_recall(ref_ids, pred_exact_phonetic, gt, cand_script_map)
    print_recall_table(res_phon_exact)

    print("\n--- 3. Phonetic Sparse TF-IDF Channel ---", flush=True)
    res_phon_sparse = evaluate_recall(ref_ids, pred_sparse_phonetic, gt, cand_script_map)
    print_recall_table(res_phon_sparse)

    print("\n--- 4. Combined Phonetic + Transliteration Channel ---", flush=True)
    res_combined = evaluate_recall(ref_ids, predicted_candidates_phon, gt, cand_script_map)
    print_recall_table(res_combined)

    # Calculate Oracle Macro F0.5
    f05_k5 = compute_oracle_macro_f05(ref_ids, predicted_candidates_phon, gt, k=5)
    f05_k10 = compute_oracle_macro_f05(ref_ids, predicted_candidates_phon, gt, k=10)
    f05_k20 = compute_oracle_macro_f05(ref_ids, predicted_candidates_phon, gt, k=20)
    f05_k50 = compute_oracle_macro_f05(ref_ids, predicted_candidates_phon, gt, k=50)
    f05_k200 = compute_oracle_macro_f05(ref_ids, predicted_candidates_phon, gt, k=200)

    print("\n--- Oracle Macro F0.5 (Phonetic Channel alone) ---", flush=True)
    print(f"Oracle F0.5 @ k=5:   {f05_k5:.4f}", flush=True)
    print(f"Oracle F0.5 @ k=10:  {f05_k10:.4f}", flush=True)
    print(f"Oracle F0.5 @ k=20:  {f05_k20:.4f}", flush=True)
    print(f"Oracle F0.5 @ k=50:  {f05_k50:.4f}", flush=True)
    print(f"Oracle F0.5 @ k=200: {f05_k200:.4f}", flush=True)

    # 9. Evaluate Union with Existing Baseline L3 Channels
    if os.path.exists(l3_art_path):
        print("\n" + "=" * 80, flush=True)
        print("UNION EVALUATION: BASELINE L3 CHANNELS + PHONETIC CHANNEL", flush=True)
        print("=" * 80, flush=True)
        t_l3 = pq.read_table(l3_art_path)
        df_l3 = t_l3.to_pandas()
        
        # Merge Baseline L3 channels (A, C, D)
        baseline_union: list[list[str]] = []
        for _, row in df_l3.iterrows():
            seen = set()
            u = []
            for col in ["channel_a", "channel_c", "channel_d"]:
                if col in row and isinstance(row[col], (list, np.ndarray)):
                    for cid in row[col]:
                        if cid not in seen:
                            seen.add(cid)
                            u.append(cid)
            baseline_union.append(u)

        print("\n--- Baseline Layer 3 Union (Channels A + C + D) ---", flush=True)
        res_baseline = evaluate_recall(ref_ids, baseline_union, gt, cand_script_map)
        print_recall_table(res_baseline)
        f05_base = compute_oracle_macro_f05(ref_ids, baseline_union, gt, k=50)
        print(f"Baseline Oracle F0.5 @ k=50: {f05_base:.4f}", flush=True)

        # Union (Baseline L3 + Phonetic Channel)
        enhanced_union: list[list[str]] = []
        for base_cands, phon_cands in zip(baseline_union, predicted_candidates_phon):
            seen = set()
            u = []
            for cid in base_cands[:50]:
                if cid not in seen:
                    seen.add(cid)
                    u.append(cid)
            for cid in phon_cands[:50]:
                if cid not in seen:
                    seen.add(cid)
                    u.append(cid)
            enhanced_union.append(u)

        print("\n--- Enhanced Union (Baseline L3 + Phonetic Channel) ---", flush=True)
        res_enhanced = evaluate_recall(ref_ids, enhanced_union, gt, cand_script_map)
        print_recall_table(res_enhanced)
        f05_enh = compute_oracle_macro_f05(ref_ids, enhanced_union, gt, k=50)
        print(f"Enhanced Oracle F0.5 @ k=50: {f05_enh:.4f}", flush=True)

    # 10. Save Artifact Parquets
    print("\n" + "=" * 80, flush=True)
    print("SAVING DELIVERABLE PARQUET ARTIFACTS", flush=True)
    print("=" * 80, flush=True)
    out_dir = Path("business_entity_resolution/artifacts/blocking")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_table = pa.Table.from_arrays(
        [
            pa.array(ref_ids, type=pa.string()),
            pa.array(predicted_candidates_phon, type=pa.list_(pa.string())),
            pa.array(predicted_scores_phon, type=pa.list_(pa.float32())),
        ],
        names=["source1_entity_id", "candidate_entity_ids", "phon_scores"],
    )

    # Save primary artifact as phon_train_train_country=india.parquet and phon_train_100k_country=india.parquet
    for name in ["phon_train_train_country=india.parquet", "phon_train_100k_country=india.parquet"]:
        out_path = out_dir / name
        pq.write_table(out_table, out_path, compression="snappy")
        print(f"Saved deliverable artifact to: {out_path} ({os.path.getsize(out_path):,} bytes)", flush=True)
    
    print("\nBenchmark completed successfully!", flush=True)


if __name__ == "__main__":
    main()
