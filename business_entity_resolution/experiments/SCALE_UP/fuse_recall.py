"""Union recall + oracle macro F0.5 when extra channels are fused with L4.

Reads the existing fused L4 artifact (channels A/C/D, already RRF-fused) and one
or more extra channel artifacts (phonetic, dense, ...), aligns them by
source1_entity_id, fuses the rankings with RRF, and reports, at several K:

  - candidate recall (micro, over true pairs)
  - oracle macro F0.5 (perfect classifier inside the fused top-K)

This directly answers the only question that matters for the 0.98 target: does an
extra channel lift the recall ceiling, and by how much?

Run:
  venv/Scripts/python.exe business_entity_resolution/experiments/SCALE_UP/fuse_recall.py \
      --l4 artifacts/blocking/l4_train_train_country=india.parquet \
      --extra artifacts/blocking/phon_train_train_country=india.parquet
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pyarrow.parquet as pq

csv.field_size_limit(10_000_000)
ROOT = Path(__file__).resolve().parents[3]
DS = ROOT / "DATA SET" / "student_resource" / "dataset"


def load_ground_truth() -> dict[str, set[str]]:
    gt: dict[str, set[str]] = {}
    with open(DS / "train" / "train_ground_truth.tsv", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if len(row) >= 2:
                gt[row[0]] = set(x for x in row[1].split(",") if x)
    return gt


def f_beta(pred: set, true: set, beta: float = 0.5) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def read_candidates(path: str, col: str = "candidate_entity_ids") -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=20000, columns=["source1_entity_id", col]):
        d = batch.to_pydict()
        for s1, cands in zip(d["source1_entity_id"], d[col]):
            out[str(s1)] = [str(c) for c in (cands or [])]
    return out


def rrf(rank_lists: list[list[str]], k: float = 60.0, weights: list[float] | None = None):
    """Reciprocal rank fusion over several ranked id lists -> ordered id list."""
    weights = weights or [1.0] * len(rank_lists)
    scores: dict[str, float] = {}
    for lst, w in zip(rank_lists, weights):
        if w == 0.0:
            continue
        for rank, cid in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def evaluate(fused: dict[str, list[str]], gt: dict, ks: list[int]) -> None:
    total_pairs = 0
    hits = {k: 0 for k in ks}
    oracle = {k: 0.0 for k in ks}
    n_refs = 0
    for s1, cands in fused.items():
        true = gt.get(s1, set())
        n_refs += 1
        total_pairs += len(true)
        for k in ks:
            top = set(cands[:k])
            top_true = top & true
            hits[k] += len(top_true)
            oracle[k] += f_beta(top_true, true)
    print(f"\nreferences {n_refs:,} | true pairs {total_pairs:,}")
    print(f"\n{'K':>6} | {'cand_recall':>11} | {'oracle_macroF0.5':>16}")
    print("-" * 42)
    for k in ks:
        print(f"{k:>6} | {hits[k]/max(1,total_pairs):>11.4f} | {oracle[k]/max(1,n_refs):>16.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l4", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--l4-weight", type=float, default=1.0)
    ap.add_argument("--extra-weight", type=float, default=1.0)
    ap.add_argument("--ks", default="6,10,20,50,100,200,500")
    args = ap.parse_args()
    ks = sorted(int(x) for x in args.ks.split(","))

    gt = load_ground_truth()
    l4 = read_candidates(args.l4)
    print(f"L4 refs: {len(l4):,}")

    extras = [(p, read_candidates(p)) for p in args.extra]
    for p, d in extras:
        overlap = len(set(d) & set(l4))
        print(f"extra {Path(p).name}: {len(d):,} refs | overlap with L4 {overlap:,}")

    fused: dict[str, list[str]] = {}
    for s1, base in l4.items():
        lists = [base]
        weights = [args.l4_weight]
        for _p, d in extras:
            lists.append(d.get(s1, []))
            weights.append(args.extra_weight)
        fused[s1] = [cid for cid, _ in rrf(lists, weights=weights)]

    label = "L4 only" if not extras else "L4 + " + ",".join(Path(p).name for p, _ in extras)
    print(f"\n=== FUSED ({label}) ===")
    evaluate(fused, gt, ks)


if __name__ == "__main__":
    main()
