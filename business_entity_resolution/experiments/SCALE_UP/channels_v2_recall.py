"""Fast in-memory recall harness for blocking channels (India train pool).

Loads the India train reference sample and candidate pool from the L2 normalized
parquet shards, builds several sparse channels one at a time (so peak memory is
one inverted index), queries them, and reports per-channel recall, union recall,
and the oracle macro F0.5 of the union at several K.

Channels
  C  name_core word TF-IDF
  D  name_core char_wb 3-5 TF-IDF
  E  addr_norm word TF-IDF
  F  addr_norm char_wb 3-5 TF-IDF
  G  name_phonetic word TF-IDF   (metaphone tokens already produced by L2)

Run:
  venv/Scripts/python.exe business_entity_resolution/experiments/SCALE_UP/channels_v2_recall.py \
      --refs 30000 --k 500
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
from l3_l5_blocking.channels import SparseTfidfChannel  # noqa: E402

csv.field_size_limit(10_000_000)
ROOT = Path(__file__).resolve().parents[3]
DS = ROOT / "DATA SET" / "student_resource" / "dataset"
NORM = ROOT / "business_entity_resolution" / "artifacts" / "normalized"


def as_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join(str(x) for x in v if x)
    return str(v)


def load_gt() -> dict:
    gt = {}
    with open(DS / "train" / "train_ground_truth.tsv", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if len(row) >= 2:
                gt[row[0]] = set(x for x in row[1].split(",") if x)
    return gt


def iter_normalized(split: str, source_idx: int, aliases: set):
    cols = ["entity_id", "country_norm", "name_core", "addr_norm", "name_phonetic"]
    for shard in sorted(glob.glob(str(NORM / f"{split}_s{source_idx}" / "part_*.parquet"))):
        df = pd.read_parquet(shard, columns=cols)
        df = df[df["country_norm"].isin(aliases)]
        if df.empty:
            continue
        for eid, nc, an, ph in zip(
            df["entity_id"].astype(str),
            df["name_core"].fillna("").astype(str),
            df["addr_norm"].fillna("").astype(str),
            df["name_phonetic"],
        ):
            yield eid, nc, an, as_text(ph)


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


def run_channel(spec, cand_texts, ref_texts, k, progress=print):
    name, analyzer, ngram, min_df, max_df = spec
    t0 = time.time()
    ch = SparseTfidfChannel(
        name=name, analyzer=analyzer, ngram_range=ngram,
        min_df=min_df, max_df=max_df,
        max_query_terms=25, max_posting_scan=50_000,
    )
    ch.build(cand_texts)
    built = time.time() - t0
    t1 = time.time()
    res = ch.query(ref_texts, k)
    queried = time.time() - t1
    progress(f"    {name}: built {built:.0f}s | queried {queried:.0f}s | "
             f"avg hits {sum(len(x) for x in res)/max(1,len(res)):.1f}")
    del ch
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", type=int, default=30_000)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--countries", default="india")
    ap.add_argument("--max-candidates", type=int, default=1_500_000)
    args = ap.parse_args()

    aliases = {"india"} if args.countries == "india" else {"us", "usa", "united states", "united states of america"}
    gt = load_gt()

    # ---- references: sample India S1 that we have ground truth for ----
    print(f"loading references (up to {args.refs:,}) ...")
    ref_rows = []
    for eid, nc, an, ph in iter_normalized("train", 1, aliases):
        if eid in gt:
            ref_rows.append((eid, nc, an, as_text(ph)))
            if len(ref_rows) >= args.refs:
                break
    ref_ids = [r[0] for r in ref_rows]
    ref_name = [r[1] for r in ref_rows]
    ref_addr = [r[2] for r in ref_rows]
    ref_phon = [r[3] for r in ref_rows]
    print(f"references: {len(ref_ids):,}")

    # ---- pass 1: candidate ids only, so we can pick a pool that keeps EVERY
    # true match of the sampled references plus a bounded set of distractors.
    print("pass 1: collecting India candidate ids ...")
    all_ids = []
    for src_idx in (2, 3):
        for eid, _nc, _an, _ph in iter_normalized("train", src_idx, aliases):
            all_ids.append(eid)
    print(f"India candidates available: {len(all_ids):,}")

    true_ids = set()
    for r in ref_ids:
        true_ids.update(m for m in gt[r] if m.startswith(("S2-", "S3-")))
    rng = np.random.default_rng(args.seed)
    distractor_pool = [eid for eid in all_ids if eid not in true_ids]
    if len(distractor_pool) > args.max_candidates:
        pick = rng.choice(len(distractor_pool), size=args.max_candidates, replace=False)
        distractor_pool = [distractor_pool[i] for i in pick]
    pool = true_ids | set(distractor_pool)
    print(f"pool: {len(pool):,} ({len(true_ids):,} true-match ids + "
          f"{len(distractor_pool):,} distractors)")

    # ---- pass 2: load records for the pool ----
    print("pass 2: loading pool records ...")
    cand_ids, cand_name, cand_addr, cand_phon = [], [], [], []
    for src_idx in (2, 3):
        for eid, nc, an, ph in iter_normalized("train", src_idx, aliases):
            if eid in pool:
                cand_ids.append(eid)
                cand_name.append(nc)
                cand_addr.append(an)
                cand_phon.append(ph)
    print(f"candidates loaded: {len(cand_ids):,}")
    id_to_idx = {eid: i for i, eid in enumerate(cand_ids)}

    true_sets = [{id_to_idx[m] for m in gt[r] if m in id_to_idx} for r in ref_ids]
    total_true = sum(len(s) for s in true_sets)
    print(f"true pairs reachable in pool: {total_true:,} "
          f"(of {sum(len(gt[r]) for r in ref_ids):,})")

    specs = [
        ("C", "word", (1, 1), 2, 0.05),
        ("D", "char_wb", (3, 5), 2, 0.05),
        ("E", "word", (1, 1), 2, 0.20),
        ("F", "char_wb", (3, 5), 2, 0.20),
        ("G", "word", (1, 1), 2, 0.05),
    ]
    text_of = {
        "C": cand_name, "D": cand_name,
        "E": cand_addr, "F": cand_addr,
        "G": cand_phon,
    }
    ref_text_of = {
        "C": ref_name, "D": ref_name,
        "E": ref_addr, "F": ref_addr,
        "G": ref_phon,
    }

    print("\nbuilding + querying channels ...")
    channel_res = {}
    for spec in specs:
        nm = spec[0]
        channel_res[nm] = run_channel(spec, text_of[nm], ref_text_of[nm], args.k)

    # ---- per-channel and union recall ----
    def recall_of(idx_lists):
        hit = sum(len(set(idx_lists[i]) & true_sets[i]) for i in range(len(ref_ids)))
        return hit / max(1, total_true)

    print("\n=== PER-CHANNEL RECALL (micro over true pairs) ===")
    for nm in channel_res:
        print(f"  {nm}: {recall_of(channel_res[nm]):.4f}")

    combined = [set() for _ in ref_ids]
    print("\n=== CUMULATIVE UNION (order: C,D,E,F,G) ===")
    for nm in ["C", "D", "E", "F", "G"]:
        for i, lst in enumerate(channel_res[nm]):
            combined[i].update(lst)
        avg = sum(len(s) for s in combined) / len(combined)
        print(f"  + {nm}: union recall {recall_of(combined):.4f} | avg |union| {avg:.1f}")

    # ---- oracle macro F0.5 of the full union, and of RRF-less truncation ----
    ks = [6, 10, 20, 50, 100, 200, 500]
    print("\n=== ORACLE MACRO F0.5 ON THE UNION (perfect matcher, no ranking) ===")
    oracle = []
    for i, s in enumerate(combined):
        oracle.append(s)
    print(f"{'K':>5} | {'cand_recall':>11} | {'oracle_macroF0.5':>16}")
    for k in ks:
        hit = 0
        fsum = 0.0
        for i, s in enumerate(combined):
            idxs = list(s)[:k]
            top_true = {c for c in idxs if c in true_sets[i]}
            hit += len(top_true)
            fsum += f_beta(top_true, true_sets[i])
        print(f"{k:>5} | {hit/max(1,total_true):>11.4f} | {fsum/len(ref_ids):>16.4f}")


if __name__ == "__main__":
    main()
