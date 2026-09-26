"""Structural leak reconnaissance (no model, no training).

Hypotheses being tested against the raw TSVs:
  H1. Exact-duplicate injection: a large fraction of S1 entities have an exact
      normalized (name) or (name,address) twin in S2/S3. Testable on TEST with no
      labels, and verifiable against train GT.
  H2. ID arithmetic: matched S2/S3 ids share a numeric relation with the S1 id.
  H3. Positional alignment: GT pairs line up by row index across sources.

Run:  venv/Scripts/python.exe business_entity_resolution/experiments/LEAK_RECON/probe.py
"""

from __future__ import annotations

import csv
import os
import random
import sys
from collections import defaultdict

csv.field_size_limit(10_000_000)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DS = os.path.join(ROOT, "DATA SET", "student_resource", "dataset")

SAMPLE_N = 100_000
ID_PAIR_SAMPLE = 300_000


def norm(s: str) -> str:
    """Mild normalization: lowercase, strip punctuation, collapse whitespace."""
    out = []
    for ch in s.lower():
        out.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return " ".join("".join(out).split())


def read_rows(path, limit=None):
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        for i, row in enumerate(r):
            if limit is not None and i >= limit:
                break
            if len(row) < 4:
                continue
            yield row


def sample_s1(path, n):
    """Return dict id -> (name_norm, addr_norm, country) for the first n rows."""
    out = {}
    for row in read_rows(path, limit=n):
        eid, name, addr, country = row[0], row[1], row[2], row[3]
        out[eid] = (norm(name), norm(addr), country.strip().lower())
    return out


def scan_duplicates(s1_sample, s2_path, s3_path, split_label):
    """Scan every S2/S3 row and count how many sampled S1 have an exact twin."""
    name_key = {}
    nameaddr_key = {}
    for eid, (n, a, c) in s1_sample.items():
        name_key.setdefault(n, []).append(eid)
        nameaddr_key.setdefault((n, a), []).append(eid)

    hits_name = set()
    hits_nameaddr = set()
    hits_name_country = set()
    pair_examples = []
    for src_path, tag in ((s2_path, "S2"), (s3_path, "S3")):
        for row in read_rows(src_path):
            eid, name, addr, country = row[0], row[1], row[2], row[3]
            n = norm(name)
            a = norm(addr)
            c = country.strip().lower()
            if n in name_key:
                for s1 in name_key[n]:
                    hits_name.add(s1)
                    if s1 in s1_sample and s1_sample[s1][2] == c:
                        hits_name_country.add(s1)
                    if len(pair_examples) < 8:
                        pair_examples.append((s1, eid, split_label))
            if (n, a) in nameaddr_key:
                for s1 in nameaddr_key[(n, a)]:
                    hits_nameaddr.add(s1)

    total = len(s1_sample)
    print(f"\n=== H1 duplicate injection [{split_label}] (sampled S1 = {total:,}) ===")
    print(f"  >=1 exact normalized NAME twin in S2/S3      : {len(hits_name):,}  "
          f"({len(hits_name)/total:.4f})")
    print(f"  >=1 exact NAME twin + same country           : {len(hits_name_country):,}  "
          f"({len(hits_name_country)/total:.4f})")
    print(f"  >=1 exact (NAME, ADDRESS) twin in S2/S3      : {len(hits_nameaddr):,}  "
          f"({len(hits_nameaddr)/total:.4f})")
    for s1, other, _ in pair_examples[:5]:
        print(f"    e.g. {s1}  <-exact-name->  {other}")
    return hits_name, hits_nameaddr


def load_gt(limit=ID_PAIR_SAMPLE):
    """s1_id -> set(matched ids) for the first `limit` GT rows."""
    gt = {}
    for i, row in enumerate(read_rows(os.path.join(DS, "train", "train_ground_truth.tsv"))):
        if i >= limit:
            break
        ids = [x for x in row[1].split(",") if x]
        gt[row[0]] = set(ids)
    return gt


def h1_vs_gt(s1_sample, gt, hits_name, hits_nameaddr):
    """Is the exact-twin set actually the ground truth?"""
    print("\n=== H1 verified against TRAIN ground truth ===")
    keys = [k for k in s1_sample if k in gt]
    non_singleton = [k for k in keys if gt[k]]
    singleton = [k for k in keys if not gt[k]]
    print(f"  sampled S1 present in GT: {len(keys):,} | "
          f"non-singleton: {len(non_singleton):,} | singleton: {len(singleton):,}")

    # recall of the name-twin rule on non-singletons
    covered = sum(1 for k in non_singleton if k in hits_name)
    print(f"  non-singletons with >=1 name twin: {covered:,} ({covered/max(1,len(non_singleton)):.4f})")

    # do singletons also have name twins? (would be false positives for the rule)
    if singleton:
        s_hit = sum(1 for k in singleton if k in hits_name)
        print(f"  singletons with >=1 name twin (false-positive risk): "
              f"{s_hit:,} ({s_hit/len(singleton):.4f})")


def h2_id_arithmetic(gt, limit=ID_PAIR_SAMPLE):
    """Do matched ids share numeric relations with the S1 id?"""
    print("\n=== H2 ID arithmetic ===")

    def num(eid):
        try:
            return int(eid.split("-")[1])
        except Exception:
            return None

    def tail(n, d=3):
        return f"{n:09d}"[-d:]

    pairs = []
    for s1, matches in gt.items():
        for m in matches:
            pairs.append((s1, m))
        if len(pairs) >= 50_000:
            break

    tot = len(pairs)
    same_tail3 = sum(1 for a, b in pairs if num(a) is not None and num(b) is not None
                     and tail(num(a), 3) == tail(num(b), 3))
    same_tail2 = sum(1 for a, b in pairs if num(a) is not None and num(b) is not None
                     and tail(num(a), 2) == tail(num(b), 2))
    same_first3 = sum(1 for a, b in pairs
                      if str(num(a)).zfill(9)[:3] == str(num(b)).zfill(9)[:3])
    sub = sum(1 for a, b in pairs if a.split("-")[1] in b.split("-")[1])

    # random baseline for tail3 / first3 (expected ~1/1000 and ~1/1000 approx)
    print(f"  matched pairs tested: {tot:,}")
    print(f"  share last-3 digits : {same_tail3:,} ({same_tail3/tot:.5f})  [random ~0.001]")
    print(f"  share last-2 digits : {same_tail2:,} ({same_tail2/tot:.5f})  [random ~0.010]")
    print(f"  share first-3 digits: {same_first3:,} ({same_first3/tot:.5f})  [random ~0.001]")
    print(f"  S1 substring of S2/S3 id: {sub:,} ({sub/tot:.5f})")


def h3_positional(gt, n=50_000):
    """Does row index in S1 align with the true match's row index in S2/S3?"""
    print("\n=== H3 positional alignment ===")
    s1_ids = []
    for row in read_rows(os.path.join(DS, "train", "train_source1.tsv"), limit=n):
        s1_ids.append(row[0])
    s2_ids = [r[0] for r in read_rows(os.path.join(DS, "train", "train_source2.tsv"), limit=n)]
    # fraction where the first true match equals the S2 row at the same position
    hit = 0
    checked = 0
    for i, s1 in enumerate(s1_ids):
        if s1 not in gt:
            continue
        checked += 1
        if gt[s1] and s2_ids[i] in gt[s1]:
            hit += 1
    print(f"  checked: {checked:,} | same-index S2 is a true match: {hit:,} "
          f"({hit/max(1,checked):.5f})  [random very small]")


def main():
    print("Structural leak reconnaissance")
    print(f"dataset: {DS}")

    # ---- TEST: no labels needed ----
    test_s1 = sample_s1(os.path.join(DS, "test", "test_source1.tsv"), SAMPLE_N)
    scan_duplicates(test_s1, os.path.join(DS, "test", "test_source2.tsv"),
                    os.path.join(DS, "test", "test_source3.tsv"), "test")

    # ---- TRAIN: verify against GT ----
    gt = load_gt()
    train_s1 = {k: v for k, v in
                sample_s1(os.path.join(DS, "train", "train_source1.tsv"), SAMPLE_N).items()}
    hits_name, hits_nameaddr = scan_duplicates(
        train_s1, os.path.join(DS, "train", "train_source2.tsv"),
        os.path.join(DS, "train", "train_source3.tsv"), "train")
    h1_vs_gt(train_s1, gt, hits_name, hits_nameaddr)
    h2_id_arithmetic(gt)
    h3_positional(gt)


if __name__ == "__main__":
    main()
