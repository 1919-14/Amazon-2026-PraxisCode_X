"""Diagnose MISSED true pairs: what do the candidates we fail to retrieve look like?

Reads the cached L4 India artifact + GT, collects the true pairs absent from the
candidate list, then characteristics of those pairs (script, name/addr similarity,
missing fields). Directs the recall workstream.

Run:
  venv/Scripts/python.exe business_entity_resolution/experiments/SCALE_UP/missing_pairs.py
"""

from __future__ import annotations

import csv
import glob
import random
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

csv.field_size_limit(10_000_000)
ROOT = Path(__file__).resolve().parents[3]
DS = ROOT / "DATA SET" / "student_resource" / "dataset"
NORM = ROOT / "business_entity_resolution" / "artifacts" / "normalized"
L4 = ROOT / "business_entity_resolution" / "artifacts" / "blocking" / "l4_train_train_country=india.parquet"


def load_gt():
    gt = {}
    with open(DS / "train" / "train_ground_truth.tsv", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if len(row) >= 2:
                gt[row[0]] = set(x for x in row[1].split(",") if x)
    return gt


def main():
    gt = load_gt()
    # 1) collect missed (ref, true_cand) pairs from the L4 artifact
    missed = []
    total_true = 0
    total_found = 0
    n_refs = 0
    pf = pq.ParquetFile(str(L4))
    for batch in pf.iter_batches(batch_size=20000,
                                 columns=["source1_entity_id", "candidate_entity_ids"]):
        d = batch.to_pydict()
        for s1, cands in zip(d["source1_entity_id"], d["candidate_entity_ids"]):
            s1 = str(s1)
            true = gt.get(s1, set())
            if not true:
                continue
            cands = set() if cands is None else set(str(c) for c in cands)
            n_refs += 1
            total_true += len(true)
            found = true & cands
            total_found += len(found)
            for m in true - cands:
                missed.append((s1, m))

    print(f"non-singleton refs: {n_refs:,} | true pairs {total_true:,} | "
          f"retrieved {total_found:,} | MISSED {len(missed):,} "
          f"({len(missed)/max(1,total_true):.4f})")

    random.seed(42)
    sample = random.sample(missed, min(4000, len(missed)))
    need = set()
    for a, b in sample:
        need.add(a)
        need.add(b)

    # 2) load normalized records for the sampled ids
    rec = {}
    for src_idx in (1, 2, 3):
        for shard in sorted(glob.glob(str(NORM / f"train_s{src_idx}" / "part_*.parquet"))):
            df = pd.read_parquet(shard, columns=[
                "entity_id", "name_core", "name_norm", "addr_norm",
                "script_type", "is_missing_addr", "is_missing_name"])
            df = df[df["entity_id"].isin(need)]
            if df.empty:
                continue
            for row in df.itertuples(index=False):
                rec[row.entity_id] = row
            if len(rec) >= len(need):
                break
        if len(rec) >= len(need):
            break
    print(f"loaded {len(rec):,}/{len(need):,} records for the missed-pair sample")

    from rapidfuzz import fuzz

    stats = Counter()
    n = 0
    examples = []
    script_pair = Counter()
    for a, b in sample:
        ra, rb = rec.get(a), rec.get(b)
        if ra is None or rb is None:
            continue
        n += 1
        na, nb = str(ra.name_core), str(rb.name_core)
        aa, ab = str(ra.addr_norm), str(rb.addr_norm)
        sa, sb = str(ra.script_type), str(rb.script_type)
        script_pair[(sa, sb)] += 1
        name_ratio = fuzz.token_set_ratio(na, nb) / 100
        addr_ratio = fuzz.token_set_ratio(aa, ab) / 100
        stats["addr_missing_both"] += int(not aa.strip() and not ab.strip())
        stats["addr_missing_one"] += int(bool(aa.strip()) != bool(ab.strip()))
        stats["name_ratio>=0.8"] += int(name_ratio >= 0.8)
        stats["name_ratio<0.5"] += int(name_ratio < 0.5)
        stats["addr_ratio>=0.8"] += int(addr_ratio >= 0.8)
        stats["both_name<0.5_addr<0.5"] += int(name_ratio < 0.5 and addr_ratio < 0.5)
        stats["cross_script"] += int(sa != sb and ("latin" in sa.lower() or "latin" in sb.lower()))
        if len(examples) < 8 and (name_ratio < 0.6):
            examples.append((a, na, aa, sa, b, nb, ab, sb, round(name_ratio, 2), round(addr_ratio, 2)))

    print(f"\n=== MISSED-PAIR CHARACTERISTICS (n={n}) ===")
    for k in ["name_ratio>=0.8", "name_ratio<0.5", "addr_ratio>=0.8",
              "addr_missing_both", "addr_missing_one", "both_name<0.5_addr<0.5",
              "cross_script"]:
        print(f"  {k:28s}: {stats[k]:6,} ({stats[k]/max(1,n):.4f})")

    print("\n=== script_type pairs (top) ===")
    for k, v in script_pair.most_common(8):
        print(f"  {k}: {v}")

    print("\n=== EXAMPLE MISSED PAIRS ===")
    for ex in examples:
        a, na, aa, sa, b, nb, ab, sb, nr, ar = ex
        print(f"\n  {a} [{sa}] {na!r}\n      addr={aa!r}")
        print(f"  {b} [{sb}] {nb!r}\n      addr={ab!r}   (name_tokset={nr}, addr_tokset={ar})")


if __name__ == "__main__":
    main()
