"""Recall@K and oracle macro F0.5 vs K from a cached L4 fused artifact.

No pipeline run: reads the L4 artifact (fused candidate lists) and the train
ground truth, and reports, for each K, how much of the true-match mass survives
and what macro F0.5 a *perfect* matcher would achieve on that candidate set.

Run:
  venv/Scripts/python.exe business_entity_resolution/experiments/SCALE_UP/recall_curve.py \
      --l4 business_entity_resolution/artifacts/blocking/l4_train_train_country=india.parquet
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pyarrow.parquet as pq

csv.field_size_limit(10_000_000)
ROOT = Path(__file__).resolve().parents[3]
DS = ROOT / "DATA SET" / "student_resource" / "dataset"


def load_ground_truth():
    gt = {}
    with open(DS / "train" / "train_ground_truth.tsv", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if len(row) < 2:
                continue
            gt[row[0]] = set(x for x in row[1].split(",") if x)
    return gt


def f_beta(pred: set, true: set, beta: float = 0.5) -> float:
    """Official per-entity F_beta (singleton credit: both empty -> 1.0)."""
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l4", required=True)
    ap.add_argument("--ks", default="6,10,20,50,100,200,500")
    args = ap.parse_args()
    ks = sorted(int(x) for x in args.ks.split(","))

    gt = load_ground_truth()
    print(f"ground truth entities: {len(gt):,}")
    print(f"L4 artifact: {args.l4}")

    # accumulators
    n_refs = 0
    n_singleton = 0
    total_gt_pairs = 0
    hits = {k: 0 for k in ks}
    oracle_sum = {k: 0.0 for k in ks}
    oracle_floor = {k: 0.0 for k in ks}  # singletons forced: empty must stay empty

    pf = pq.ParquetFile(args.l4)
    cols = ["source1_entity_id", "candidate_entity_ids"]
    for batch in pf.iter_batches(batch_size=20000, columns=cols):
        d = batch.to_pydict()
        for s1, cands in zip(d["source1_entity_id"], d["candidate_entity_ids"]):
            s1 = str(s1)
            true = gt.get(s1, set())
            cands = [] if cands is None else [str(c) for c in cands]
            n_refs += 1
            if not true:
                n_singleton += 1
            total_gt_pairs += len(true)
            for k in ks:
                # A perfect matcher keeps exactly the true matches present in the
                # top-K (precision 1). Singletons have no true match, so it predicts
                # empty and earns the singleton credit -- that IS the oracle.
                top_true = set(cands[:k]) & true
                hits[k] += len(top_true)
                oracle_sum[k] += f_beta(top_true, true)

    print(f"\nreferences in artifact: {n_refs:,} | singletons: {n_singleton:,} "
          f"| total GT pairs: {total_gt_pairs:,}")
    print(f"\n{'K':>5} | {'cand_recall':>11} | {'oracle_macroF0.5':>16}")
    print("-" * 42)
    for k in ks:
        rec = hits[k] / max(1, total_gt_pairs)
        ora = oracle_sum[k] / max(1, n_refs)
        print(f"{k:>5} | {rec:>11.4f} | {ora:>16.4f}")


if __name__ == "__main__":
    main()
