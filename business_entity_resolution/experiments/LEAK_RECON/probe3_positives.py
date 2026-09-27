"""Inspect real true-match pairs: how noisy are the positives, and what signal separates them?

Run: venv/Scripts/python.exe business_entity_resolution/experiments/LEAK_RECON/probe3_positives.py
"""

from __future__ import annotations

import csv
import os
import re
from collections import Counter

csv.field_size_limit(10_000_000)
try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DS = os.path.join(ROOT, "DATA SET", "student_resource", "dataset")

N_S1 = 400


def rows(path, limit=None, min_cols=4):
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for i, row in enumerate(r):
            if limit is not None and i >= limit:
                break
            if len(row) >= min_cols:
                yield row


def norm(s):
    out = [ch if (ch.isalnum() or ch.isspace()) else " " for ch in s.lower()]
    return " ".join("".join(out).split())


POSTAL = re.compile(r"\b(\d{5,6})\b")
NUMRUN = re.compile(r"\b\d+\b")


def house(addr):
    m = NUMRUN.search(addr)
    return m.group(0) if m else ""


def postal(addr):
    m = POSTAL.findall(addr)
    return m[-1] if m else ""


def main():
    try:
        from rapidfuzz import fuzz
        have_rf = True
    except Exception:
        have_rf = False
        print("rapidfuzz not available; lexical ratios skipped")

    # 1. sample S1 that have matches
    gt = {}
    for row in rows(os.path.join(DS, "train", "train_ground_truth.tsv"), min_cols=2):
        matches = [x for x in row[1].split(",") if x]
        if len(matches) >= 1:
            gt[row[0]] = matches
            if len(gt) >= N_S1:
                break
    print(f"sampled {len(gt)} S1 with >=1 match")

    need = set(gt.keys())
    for ms in gt.values():
        need.update(ms)

    # 2. collect records
    rec = {}
    for split in ("train",):
        for src in ("source1", "source2", "source3"):
            p = os.path.join(DS, split, f"{split}_{src}.tsv")
            for row in rows(p):
                if row[0] in need:
                    rec[row[0]] = (row[1], row[2], row[3])
            if len(rec) >= len(need):
                break
    print(f"collected {len(rec)}/{len(need)} records")

    # 3. eyeball examples
    print("\n=== EXAMPLE TRUE PAIRS ===")
    for i, (s1, ms) in enumerate(gt.items()):
        if i >= 6:
            break
        n1, a1, c1 = rec.get(s1, ("?", "?", "?"))
        print(f"\nS1 {s1} [{c1}] name={n1!r}\n            addr={a1!r}")
        for m in ms[:4]:
            n2, a2, c2 = rec.get(m, ("?", "?", "?"))
            print(f"   {m} [{c2}] name={n2!r}\n              addr={a2!r}")

    # 4. similarity stats over positives
    stats = Counter()
    n_pairs = 0
    for s1, ms in gt.items():
        if s1 not in rec:
            continue
        n1, a1, c1 = rec[s1]
        for m in ms:
            if m not in rec:
                continue
            n2, a2, c2 = rec[m]
            n_pairs += 1
            nn1, nn2 = norm(n1), norm(n2)
            aa1, aa2 = norm(a1), norm(a2)
            stats["exact_raw_name"] += int(n1 == n2)
            stats["exact_norm_name"] += int(nn1 == nn2)
            stats["exact_norm_addr"] += int(aa1 == aa2)
            stats["house_match"] += int(house(a1) != "" and house(a1) == house(a2))
            stats["postal_match"] += int(postal(a1) != "" and postal(a1) == postal(a2))
            if have_rf:
                stats["name_ratio>=0.9"] += int(fuzz.ratio(nn1, nn2) >= 90)
                stats["name_ratio>=0.8"] += int(fuzz.ratio(nn1, nn2) >= 80)
                stats["addr_ratio>=0.8"] += int(fuzz.ratio(aa1, aa2) >= 80)
                stats["name_token_set>=0.8"] += int(fuzz.token_set_ratio(nn1, nn2) >= 80)
    print(f"\n=== POSITIVE-PAIR SIGNAL RATES (n={n_pairs}) ===")
    for k in ["exact_raw_name", "exact_norm_name", "name_ratio>=0.9", "name_ratio>=0.8",
              "name_token_set>=0.8", "exact_norm_addr", "addr_ratio>=0.8",
              "house_match", "postal_match"]:
        if k in stats:
            print(f"  {k:22s}: {stats[k]:7,} ({stats[k]/max(1,n_pairs):.4f})")

    # 5. how many true matches per entity
    sizes = [len(ms) for ms in gt.values()]
    print(f"\n  true matches per entity: mean={sum(sizes)/len(sizes):.2f} "
          f"min={min(sizes)} max={max(sizes)}")

    # 6. source composition
    comp = Counter()
    for ms in gt.values():
        s2 = sum(1 for m in ms if m.startswith("S2-"))
        s3 = sum(1 for m in ms if m.startswith("S3-"))
        comp[("both" if s2 and s3 else ("S2only" if s2 else "S3only"))] += 1
    print(f"  match source composition: {dict(comp)}")


if __name__ == "__main__":
    main()
