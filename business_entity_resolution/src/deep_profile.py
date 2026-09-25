"""Deep Data Profiling & Exploratory Data Analysis Script."""

import json
import sys
import re
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path("DATA SET/student_resource/dataset")
OUTPUT_DIR = Path("business_entity_resolution/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_text_series(series: pd.Series, prefix=""):
    """Compute length and token stats for a text column."""
    cleaned = series.fillna("").astype(str)
    lens = cleaned.str.len()
    words = cleaned.str.split().str.len()
    missing_count = (series.isna() | (series.astype(str).str.strip() == "")).sum()
    
    return {
        "total_records": len(series),
        "missing_count": int(missing_count),
        "missing_pct": float(round(missing_count / len(series) * 100, 2)),
        "char_len_mean": float(round(lens.mean(), 2)),
        "char_len_median": float(lens.median()),
        "char_len_max": int(lens.max()),
        "word_count_mean": float(round(words.mean(), 2)),
        "word_count_median": float(words.median()),
    }

def detect_script(text: str) -> str:
    """Detect dominant script in text."""
    if not text:
        return "Empty"
    devanagari = len(re.findall(r'[\u0900-\u097F]', text))
    tamil = len(re.findall(r'[\u0B80-\u0BFF]', text))
    kannada = len(re.findall(r'[\u0C80-\u0CFF]', text))
    telugu = len(re.findall(r'[\u0C00-\u0C7F]', text))
    latin = len(re.findall(r'[a-zA-Z]', text))
    
    if devanagari > 0:
        return "Devanagari (Hindi)"
    if tamil > 0:
        return "Tamil"
    if kannada > 0:
        return "Kannada"
    if telugu > 0:
        return "Telugu"
    if latin > 0:
        return "Latin (English/French)"
    return "Other"

def extract_legal_suffix(name: str) -> str:
    """Find common legal suffix in business name."""
    name_lower = str(name).lower().strip()
    suffixes = [
        "pvt ltd", "private limited", "llp", "inc", "corp", "corporation", 
        "llc", "ltd", "limited", "sarl", "sa", "sas", "eurl", "co", "company"
    ]
    for suf in suffixes:
        if re.search(r'\b' + re.escape(suf) + r'\b', name_lower):
            return suf.upper()
    return "NONE/OTHER"

def profile_split(split_name: str):
    """Profile a dataset split (train or test)."""
    split_dir = DATA_DIR / split_name
    profiles = {}
    sources = ["source1", "source2", "source3"]
    
    for src in sources:
        file_path = split_dir / f"{split_name}_{src}.tsv"
        print(f"Profiling {split_name}/{src}...", flush=True)
        df = pd.read_csv(file_path, sep="\t", nrows=300000) # Sample 300k for fast deep stats
        
        country_counts = df["country"].value_counts().to_dict()
        name_stats = analyze_text_series(df["business_name"])
        addr_stats = analyze_text_series(df["business_address"])
        
        # Script breakdown
        script_counter = Counter(df["business_name"].dropna().sample(min(20000, len(df)), random_state=42).apply(detect_script))
        
        # Legal suffix breakdown
        suffix_counter = Counter(df["business_name"].dropna().sample(min(20000, len(df)), random_state=42).apply(extract_legal_suffix))
        
        profiles[src] = {
            "country_distribution": country_counts,
            "name_stats": name_stats,
            "address_stats": addr_stats,
            "script_distribution": dict(script_counter),
            "top_legal_suffixes": dict(suffix_counter.most_common(8))
        }
    return profiles

def profile_ground_truth():
    """Profile ground truth matching cardinalities."""
    file_path = DATA_DIR / "train" / "train_ground_truth.tsv"
    print("Profiling train_ground_truth...", flush=True)
    gt = pd.read_csv(file_path, sep="\t")
    
    match_lists = gt["matched_entity_ids"].fillna("").astype(str).str.split(",")
    match_counts = match_lists.apply(lambda x: len([m for m in x if m]))
    hist = match_counts.value_counts().sort_index().to_dict()
    
    s2_count = sum(len([m for m in lst if m.startswith("S2")]) for lst in match_lists)
    s3_count = sum(len([m for m in lst if m.startswith("S3")]) for lst in match_lists)
    
    return {
        "total_s1_entities": len(gt),
        "singleton_count": int((match_counts == 0).sum()),
        "singleton_pct": float(round((match_counts == 0).sum() / len(gt) * 100, 2)),
        "total_matches": int(match_counts.sum()),
        "avg_matches_per_s1": float(round(match_counts.mean(), 2)),
        "median_matches_per_s1": float(match_counts.median()),
        "s2_matches_total": s2_count,
        "s3_matches_total": s3_count,
        "histogram": {int(k): int(v) for k, v in hist.items()}
    }

def main():
    print("=== STARTING DEEP DATA PROFILING ===", flush=True)
    train_profile = profile_split("train")
    test_profile = profile_split("test")
    gt_profile = profile_ground_truth()
    
    full_profile = {
        "train": train_profile,
        "test": test_profile,
        "ground_truth": gt_profile
    }
    
    out_file = OUTPUT_DIR / "deep_data_profile.json"
    out_file.write_text(json.dumps(full_profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved profile JSON to {out_file}", flush=True)

if __name__ == "__main__":
    main()
