"""Second leak recon pass: cross-source id structure, rank alignment, train/test overlap.

Run: venv/Scripts/python.exe business_entity_resolution/experiments/LEAK_RECON/probe2.py
"""

from __future__ import annotations

import csv
import os
import random
from collections import Counter

csv.field_size_limit(10_000_000)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DS = os.path.join(ROOT, "DATA SET", "student_resource", "dataset")


def rows(path, limit=None):
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for i, row in enumerate(r):
            if limit is not None and i >= limit:
                break
            if len(row) >= 2:
                yield row


def num(eid):
    try:
        return int(eid.split("-")[1])
    except Exception:
        return None


def norm(s):
    out = [ch if (ch.isalnum() or ch.isspace()) else " " for ch in s.lower()]
    return " ".join("".join(out).split())


def h4_cross_source_id(gt, limit=300_000):
    """Do S2 and S3 records of the same entity share a numeric id?"""
    print("\n=== H4 same numeric id across S2/S3 for one entity ===")
    same = 0
    with_both = 0
    checked = 0
    for i, (s1, matches) in enumerate(gt.items()):
        if i >= limit:
            break
        checked += 1
        s2 = [num(m) for m in matches if m.startswith("S2-")]
        s3 = [num(m) for m in matches if m.startswith("S3-")]
        if s2 and s3:
            with_both += 1
            if set(s2) & set(s3):
                same += 1
    print(f"  entities checked: {checked:,} | with both S2 & S3: {with_both:,}")
    print(f"  share a numeric id across S2/S3: {same:,} ({same/max(1,with_both):.5f})")


def h5_id_ranges():
    """ID density/range per source - are they dense sequences or random?"""
    print("\n=== H5 id ranges / density ===")
    for split in ("train", "test"):
        for src in ("source1", "source2", "source3"):
            p = os.path.join(DS, split, f"{split}_{src}.tsv")
            mx = 0
            mn = 10**18
            n = 0
            uniq = set()
            for row in rows(p, limit=1_000_000):
                v = num(row[0])
                if v is None:
                    continue
                n += 1
                mx = max(mx, v)
                mn = min(mn, v)
                if n <= 200_000:
                    uniq.add(v)
            dens_lo = len(uniq) / max(1, (max(uniq) - min(uniq) + 1)) if uniq else 0
            print(f"  {split:5s} {src}: n={n:>9,}  min={mn:<12} max={mx:<12} "
                  f"dup-in-first200k={200_000-len(uniq):,}")


def h6_rank_alignment(gt, n=200_000):
    """Is there a fixed offset / correlation between S1 row index and the true match index?"""
    print("\n=== H6 rank alignment (first 200k rows) ===")
    s1_idx = {}
    for i, row in enumerate(rows(os.path.join(DS, "train", "train_source1.tsv"), n)):
        s1_idx[row[0]] = i
    s2_idx = {}
    for i, row in enumerate(rows(os.path.join(DS, "train", "train_source2.tsv"), n)):
        s2_idx[row[0]] = i

    diffs = []
    neg = 0
    close = 0
    for s1, matches in gt.items():
        if s1 not in s1_idx:
            continue
        for m in matches:
            if m in s2_idx:
                d = s2_idx[m] - s1_idx[s1]
                diffs.append(d)
                if d < 0:
                    neg += 1
                if abs(d) < 50:
                    close += 1
    if not diffs:
        print("  no aligned pairs")
        return
    diffs.sort()
    med = diffs[len(diffs) // 2]
    print(f"  aligned pairs: {len(diffs):,}")
    print(f"  median (idx_S2 - idx_S1) = {med:,} | frac diff<0 = {neg/len(diffs):.4f} | "
          f"frac |diff|<50 = {close/len(diffs):.4f}")
    print(f"  [if order were random, frac |diff|<50 ~ 0.00025 and median ~0]")


def h7_train_test_overlap():
    """Do TEST records also appear in TRAIN (same normalized name+address)?"""
    print("\n=== H7 train/test record overlap (S1) ===")
    train_keys = set()
    for row in rows(os.path.join(DS, "train", "train_source1.tsv")):
        train_keys.add((norm(row[1]), norm(row[2]), row[3].strip().lower()))
    test_keys = set()
    for row in rows(os.path.join(DS, "test", "test_source1.tsv")):
        test_keys.add((norm(row[1]), norm(row[2]), row[3].strip().lower()))
    inter = train_keys & test_keys
    print(f"  train S1 unique (name,addr,country): {len(train_keys):,}")
    print(f"  test  S1 unique (name,addr,country): {len(test_keys):,}")
    print(f"  overlap: {len(inter):,} ({len(inter)/max(1,len(test_keys)):.5f} of test)")


def main():
    gt = {}
    for i, row in enumerate(rows(os.path.join(DS, "train", "train_ground_truth.tsv"))):
        ids = [x for x in row[1].split(",") if x]
        gt[row[0]] = set(ids)
    print(f"loaded GT: {len(gt):,} S1 entities")
    h4_cross_source_id(gt)
    h6_rank_alignment(gt)
    del gt
    h5_id_ranges()
    h7_train_test_overlap()


if __name__ == "__main__":
    main()
