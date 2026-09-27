"""Weight sweep for RRF fusion of L4 + extra channels (single process).

Loads ground truth and all candidate artifacts once, then sweeps channel weights
and reports recall@K / oracle macro F0.5, so tuning does not pay the parquet+GT
load cost repeatedly.

Run:
  venv/Scripts/python.exe business_entity_resolution/experiments/SCALE_UP/sweep_weights.py \
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


def fuse_ref(lists, weights, k=60.0):
    scores: dict[str, float] = {}
    for lst, w in zip(lists, weights):
        if w == 0.0:
            continue
        for rank, cid in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    return [c for c, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l4", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--grid", default="0.5,1,1.5,2,3,4,6")
    ap.add_argument("--ks", default="6,10,20,50,100,200,500")
    args = ap.parse_args()
    ks = sorted(int(x) for x in args.ks.split(","))
    grid = [float(x) for x in args.grid.split(",")]

    gt = load_ground_truth()
    l4 = read_candidates(args.l4)
    extras = [read_candidates(p) for p in args.extra]
    refs = list(l4.keys())
    print(f"refs {len(refs):,} | extras {len(extras)} | grid {grid}")

    # Precompute per-ref id lists once.
    base_lists = {s1: l4[s1] for s1 in refs}
    extra_lists = [{s1: d.get(s1, []) for s1 in refs} for d in extras]

    def score_weight(w_l4: float, w_extra: float):
        total_pairs = 0
        n_refs = 0
        hits = {k: 0 for k in ks}
        oracle = {k: 0.0 for k in ks}
        for s1 in refs:
            true = gt.get(s1, set())
            lists = [base_lists[s1]] + [e[s1] for e in extra_lists]
            weights = [w_l4] + [w_extra] * len(extra_lists)
            ordered = fuse_ref(lists, weights)
            n_refs += 1
            total_pairs += len(true)
            for k in ks:
                top_true = set(ordered[:k]) & true
                hits[k] += len(top_true)
                oracle[k] += f_beta(top_true, true)
        recall = {k: hits[k] / max(1, total_pairs) for k in ks}
        ora = {k: oracle[k] / max(1, n_refs) for k in ks}
        return recall, ora

    header = "w_l4".ljust(6) + "".join(f"{'r@'+str(k):>9}" for k in ks) + "  ||" + "".join(f"{'O@'+str(k):>9}" for k in ks)
    print("\n" + header)
    print("-" * len(header))
    for w_l4 in grid:
        recall, ora = score_weight(w_l4, 1.0)
        row = f"{w_l4:<6.1f}" + "".join(f"{recall[k]:>9.4f}" for k in ks) + "  ||" + "".join(f"{ora[k]:>9.4f}" for k in ks)
        print(row, flush=True)


if __name__ == "__main__":
    main()
