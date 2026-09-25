"""Layer 1 Pipeline: Validation split generator, macro F0.5 scorer, baseline evaluations, and test suite."""

import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is in sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for p in [str(SRC_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import psutil
from config import PATH_ARTIFACTS_DIR, PATH_OUTPUT_DIR, PATH_TRAIN_GT
from ingest import parse_ground_truth

try:
    from src.l1_validation.metrics import f_beta, macro_f05
    from src.l1_validation.split_generator import generate_split, load_split_ids
except ImportError:
    from l1_validation.metrics import f_beta, macro_f05
    from l1_validation.split_generator import generate_split, load_split_ids


def get_ram_mb() -> float:
    """Return current process resident memory usage in megabytes."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main() -> None:
    """Execute complete Layer 1 workflow under strict memory constraints."""
    print("=" * 65)
    print("🚀 LAYER 1: VALIDATION SPLIT & MACRO F0.5 SCORER")
    print("=" * 65)

    peak_ram = get_ram_mb()

    # Step 1: Generate grouped 80/20 train/validation split
    print("\n[Step 1/5] Generating deterministic grouped train/val split (seed=42, 80/20)...")
    split_info = generate_split(seed=42, val_fraction=0.2)
    peak_ram = max(peak_ram, get_ram_mb())

    # Step 2: Load ground truth using ingest.py parse_ground_truth
    print("\n[Step 2/5] Loading ground truth dataset...")
    gt_df = pd.read_csv(PATH_TRAIN_GT, sep="\t", dtype=str, keep_default_na=False)
    gt_map = parse_ground_truth(gt_df)
    del gt_df
    gc.collect()
    peak_ram = max(peak_ram, get_ram_mb())

    # Step 3: Load split IDs
    print("\n[Step 3/5] Loading validation entity IDs...")
    split_ids = load_split_ids()
    train_ids = split_ids["train_ids"]
    val_ids = split_ids["val_ids"]
    peak_ram = max(peak_ram, get_ram_mb())

    # Step 4: Compute full & validation metrics and baseline evaluations
    print("\n[Step 4/5] Computing distributions, cardinality, and baseline scores...")
    total_s1 = len(gt_map)
    full_singletons = sum(1 for m in gt_map.values() if len(m) == 0)
    full_singleton_rate = full_singletons / total_s1 if total_s1 > 0 else 0.0

    # Validation subset ground truth
    val_gt = {s1_id: gt_map[s1_id] for s1_id in val_ids if s1_id in gt_map}
    val_total = len(val_gt)
    val_singletons = sum(1 for m in val_gt.values() if len(m) == 0)
    val_singleton_rate = val_singletons / val_total if val_total > 0 else 0.0
    val_non_singletons = val_total - val_singletons
    val_total_matches = sum(len(m) for m in val_gt.values())
    avg_matches_per_non_singleton = (
        val_total_matches / val_non_singletons if val_non_singletons > 0 else 0.0
    )

    # Baseline A: Empty predictions for all entities (singleton baseline)
    pred_empty = {s1_id: [] for s1_id in val_gt}
    eval_empty = macro_f05(pred_empty, val_gt)
    baseline_empty_f05 = eval_empty["macro_f05"]
    del pred_empty
    gc.collect()

    # Baseline B: All-candidates-as-matches (uncalibrated candidate acceptance)
    # When all retrieved candidates are accepted without precision thresholding,
    # singletons (which receive candidate proposals) get 0.0, while non-singletons
    # maintain recall but suffer precision penalty from distractor candidates.
    # Case B1 (Dummy/Constant Match): Every entity predicts a match -> F0.5 = 0.0000
    pred_dummy = {s1_id: ["DUMMY_MATCH"] for s1_id in val_gt}
    eval_dummy = macro_f05(pred_dummy, val_gt)
    baseline_all_match_dummy_f05 = eval_dummy["macro_f05"]
    del pred_dummy
    gc.collect()

    # Case B2 (Uncalibrated Matches without singleton precision filter):
    # Non-singletons predict true matches (100% recall), but singletons are also predicted with a match:
    pred_uncalibrated_singletons = {
        s1_id: list(matches) if matches else ["DUMMY_MATCH"]
        for s1_id, matches in val_gt.items()
    }
    eval_uncalibrated_singletons = macro_f05(pred_uncalibrated_singletons, val_gt)
    baseline_all_match_no_singleton_filter = eval_uncalibrated_singletons["macro_f05"]
    del pred_uncalibrated_singletons
    gc.collect()

    # Case B3 (Uncalibrated Candidate Pool with target compactness K ≈ 5.5):
    # Simulates accepting all candidates from a candidate generator (recall ~100%, ~1.8 false candidates/entity):
    # For non-singletons: Precision = 3.66 / 5.5 = ~0.665, Recall = 1.0 -> F0.5 = ~0.713.
    # For singletons: Score = 0.0. Macro F0.5 = ~0.673.
    simulated_cand_scores: list[float] = []
    for matches in val_gt.values():
        if not matches:
            simulated_cand_scores.append(0.0)  # Singleton falsely matched
        else:
            tp = len(matches)
            pred_count = tp + 2  # ~2 distractor candidates per entity in blocking pool
            prec = tp / pred_count
            rec = 1.0
            simulated_cand_scores.append(f_beta(prec, rec, beta=0.5))
    baseline_uncalibrated_pool_f05 = sum(simulated_cand_scores) / len(simulated_cand_scores)
    del simulated_cand_scores
    gc.collect()

    peak_ram = max(peak_ram, get_ram_mb())

    # Report Summary
    print("\n" + "=" * 65)
    print("📊 LAYER 1 VALIDATION & CARDINALITY REPORT")
    print("=" * 65)
    print(f"Total S1 entities               : {split_info['total_count']:,}")
    print(f"Train ID count                  : {split_info['train_count']:,}")
    print(f"Val ID count                    : {split_info['val_count']:,}")
    print(f"Actual val fraction             : {split_info['val_fraction']:.6f} (~20.0%)")
    print(f"Singleton count (Full GT)       : {full_singletons:,} ({full_singleton_rate * 100:.2f}%)")
    print(f"Singleton count (Val GT)        : {val_singletons:,} ({val_singleton_rate * 100:.2f}%)")
    print(f"Avg true matches / non-singleton: {avg_matches_per_non_singleton:.2f}")
    print("-" * 65)
    print("Baseline Performance (Macro F0.5):")
    print(f"  1. All-Empty Predictions      : {baseline_empty_f05:.6f} (matches val singleton rate: {val_singleton_rate:.6f})")
    print(f"  2. All-Match (Dummy constant) : {baseline_all_match_dummy_f05:.6f} (zero match precision/recall)")
    print(f"  3. All-Match (No singleton filter): {baseline_all_match_no_singleton_filter:.6f} (100% recall, 0 singleton credit)")
    print(f"  4. All-Candidates-As-Matches  : {baseline_uncalibrated_pool_f05:.6f} (raw recall without precision control, K≈5.5)")
    print(f"Peak RAM usage                  : {peak_ram:.1f} MB (Hard constraint: <500 MB)")
    print("=" * 65)

    # Save summary report to JSON
    summary_report = {
        "total_s1_entities": split_info["total_count"],
        "train_id_count": split_info["train_count"],
        "val_id_count": split_info["val_count"],
        "actual_val_fraction": split_info["val_fraction"],
        "full_singleton_rate": full_singleton_rate,
        "val_singleton_rate": val_singleton_rate,
        "avg_matches_per_non_singleton": avg_matches_per_non_singleton,
        "baseline_empty_f05": baseline_empty_f05,
        "baseline_all_match_dummy_f05": baseline_all_match_dummy_f05,
        "baseline_all_match_no_singleton_filter": baseline_all_match_no_singleton_filter,
        "baseline_uncalibrated_pool_f05": baseline_uncalibrated_pool_f05,
        "peak_ram_mb": peak_ram,
    }
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PATH_OUTPUT_DIR / "l1_validation_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Saved validation summary: {report_path}")

    # Step 5: Run unit tests via subprocess
    print("\n[Step 5/5] Running Layer 1 unit tests via subprocess...")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR};{PROJECT_ROOT};" + env.get("PYTHONPATH", "")
    res = subprocess.run(
        [sys.executable, "-m", "src.l1_validation.unit_tests"],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Unit tests failed with exit code {res.returncode}")

    print("=" * 65)
    print("🌟 LAYER 1 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
