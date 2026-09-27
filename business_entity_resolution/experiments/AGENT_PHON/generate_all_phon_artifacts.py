"""Generate phonetic blocking artifacts for all missing splits and countries.

Writes:
  business_entity_resolution/artifacts/blocking/phon_<split>_<refs>_country=<c>.parquet

for every (split, refs, country) tuple that has a corresponding L3 artifact
but no phon artifact yet.

Schema of each output parquet:
  source1_entity_id   : string
  candidate_entity_ids: list<string>   (ranked best-first, up to k_retrieve=200)
  phon_scores         : list<float32>
"""

from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from l3_l5_blocking.phonetic_channel import PhoneticChannel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = ROOT / "artifacts" / "blocking"
NORM_DIR  = ROOT / "artifacts" / "normalized"

# Discover existing L3 artifacts and what phon artifacts are still needed
l3_files = sorted(ARTIFACTS.glob("l3_*.parquet"))
needed = []
for l3p in l3_files:
    stem = l3p.stem  # e.g. l3_train_val_country=us
    parts = stem.split("_", 1)   # ['l3', 'train_val_country=us']
    rest = parts[1]              # 'train_val_country=us'
    phon_name = "phon_" + rest + ".parquet"
    phon_path = ARTIFACTS / phon_name
    if not phon_path.exists():
        needed.append((l3p, phon_path, rest))
    else:
        print(f"  SKIP (exists): {phon_path.name}")

if not needed:
    print("All phon artifacts already exist. Nothing to do.")
    sys.exit(0)

print(f"\nWill generate {len(needed)} artifact(s):")
for _, phon_path, rest in needed:
    print(f"  -> {phon_path.name}")


def load_s1_for_country(l3_table: pa.Table, split: str, country: str) -> pd.DataFrame:
    """Load S1 reference records for the given country, filtered to ref IDs in l3_table."""
    ref_ids = set(l3_table["source1_entity_id"].to_pylist())

    # split names: 'train_train', 'train_val', 'test_all'
    if split.startswith("train"):
        s1_src = "train_s1"
    else:
        s1_src = "test_s1"

    cols = ["entity_id", "country_norm", "name_raw", "script_type", "addr_postal", "addr_norm"]
    files = sorted((NORM_DIR / s1_src).glob("part_*.parquet"))
    records = []
    for f in files:
        t = pq.read_table(f, columns=cols)
        df = t.to_pandas()
        df = df[(df["country_norm"] == country) & (df["entity_id"].isin(ref_ids))]
        records.append(df)
        if sum(len(r) for r in records) >= len(ref_ids):
            break
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.concat(records, ignore_index=True).drop_duplicates("entity_id")


def load_candidates_for_country(split: str, country: str) -> pd.DataFrame:
    """Load S2+S3 candidate pool for the given country and split."""
    if split.startswith("train"):
        s2_src, s3_src = "train_s2", "train_s3"
    else:
        s2_src, s3_src = "test_s2", "test_s3"

    cols = ["entity_id", "country_norm", "name_raw", "script_type", "addr_postal", "addr_norm"]
    records = []
    for src_name in [s2_src, s3_src]:
        src_dir = NORM_DIR / src_name
        if not src_dir.exists():
            continue
        for f in sorted(src_dir.glob("part_*.parquet")):
            t = pq.read_table(f, columns=cols)
            df = t.to_pandas()
            records.append(df[df["country_norm"] == country])
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.concat(records, ignore_index=True).drop_duplicates("entity_id")


for l3_path, phon_path, rest in needed:
    country_part = rest.split("country=")[1]
    split_part   = rest.split("_country=")[0]   # e.g. 'train_val'

    print(f"\n{'='*70}")
    print(f"Generating: {phon_path.name}")
    print(f"  split={split_part}  country={country_part}")

    t0 = time.time()

    # Load L3 table to get ordered reference IDs
    l3_table = pq.read_table(l3_path, columns=["source1_entity_id"])
    ref_ids_ordered = l3_table["source1_entity_id"].to_pylist()
    print(f"  References: {len(ref_ids_ordered):,}")

    # Load S1 reference records
    ref_df = load_s1_for_country(l3_table, split_part, country_part)
    if ref_df.empty:
        print(f"  WARNING: no S1 records found for country={country_part}, split={split_part}. Skipping.")
        continue

    # Reorder to match l3 ordering
    ref_df = ref_df.set_index("entity_id").reindex(ref_ids_ordered).reset_index()
    ref_df["name_raw"]    = ref_df["name_raw"].fillna("")
    ref_df["script_type"] = ref_df["script_type"].fillna("latin")
    ref_df["addr_postal"] = ref_df["addr_postal"].fillna("")
    ref_df["addr_norm"]   = ref_df["addr_norm"].fillna("")
    print(f"  Loaded {len(ref_df):,} S1 records in {time.time()-t0:.1f}s")

    # Load candidate pool
    t1 = time.time()
    cand_df = load_candidates_for_country(split_part, country_part)
    if cand_df.empty:
        print(f"  WARNING: no candidate records for country={country_part}. Skipping.")
        continue
    cand_df["name_raw"]    = cand_df["name_raw"].fillna("")
    cand_df["script_type"] = cand_df["script_type"].fillna("latin")
    cand_df["addr_postal"] = cand_df["addr_postal"].fillna("")
    cand_df["addr_norm"]   = cand_df["addr_norm"].fillna("")
    print(f"  Loaded {len(cand_df):,} candidates in {time.time()-t1:.1f}s")

    # Build PhoneticChannel
    t2 = time.time()
    ch = PhoneticChannel(max_posting=2000, k_retrieve=200)
    ch.build(
        cand_ids    = cand_df["entity_id"].tolist(),
        cand_names  = cand_df["name_raw"].tolist(),
        cand_scripts= cand_df["script_type"].tolist(),
        cand_postals= cand_df["addr_postal"].tolist(),
        cand_addrs  = cand_df["addr_norm"].tolist(),
    )
    print(f"  Channel built in {time.time()-t2:.1f}s")

    # Query batch
    t3 = time.time()
    result_table = ch.query_batch(
        ref_ids    = ref_df["entity_id"].tolist(),
        ref_names  = ref_df["name_raw"].tolist(),
        ref_scripts= ref_df["script_type"].tolist(),
        ref_postals= ref_df["addr_postal"].tolist(),
        ref_addrs  = ref_df["addr_norm"].tolist(),
        batch_size = 5000,
        verbose    = True,
    )
    print(f"  Queries done in {time.time()-t3:.1f}s")

    # Save
    pq.write_table(result_table, phon_path, compression="snappy")
    sz = phon_path.stat().st_size
    print(f"  Saved {phon_path.name} ({sz:,} bytes) in {time.time()-t0:.1f}s total")

print(f"\n{'='*70}")
print("All phonetic blocking artifacts generated successfully.")
