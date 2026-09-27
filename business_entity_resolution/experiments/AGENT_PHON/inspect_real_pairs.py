"""Find and inspect real ground truth pairs with cross-script matches in India."""

import sys
import glob
import pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

# Load ground truth
gt = {}
with open("DATA SET/student_resource/dataset/train/train_ground_truth.tsv", encoding="utf-8") as f:
    header = f.readline()
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) == 2:
            gt[parts[0]] = set(parts[1].split(","))

print(f"Total ground truth entries: {len(gt)}")

# Load a sample of S1, S2, S3 India records
s1_files = glob.glob("business_entity_resolution/artifacts/normalized/train_s1/*.parquet")[:3]
s2_files = glob.glob("business_entity_resolution/artifacts/normalized/train_s2/*.parquet")[:3]
s3_files = glob.glob("business_entity_resolution/artifacts/normalized/train_s3/*.parquet")[:3]

cols = ["entity_id", "country_norm", "name_raw", "name_norm", "name_core", "script_type", "addr_norm"]
s1_df = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in s1_files])
s2_df = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in s2_files])
s3_df = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in s3_files])

s1_india = s1_df[s1_df["country_norm"] == "india"].set_index("entity_id")
s2_india = s2_df[s2_df["country_norm"] == "india"].set_index("entity_id")
s3_india = s3_df[s3_df["country_norm"] == "india"].set_index("entity_id")

print(f"Loaded India S1: {len(s1_india)}, S2: {len(s2_india)}, S3: {len(s3_india)}")

# Find cross-script matches
non_latin_pairs = []
for s1_id, row1 in s1_india.iterrows():
    if s1_id in gt:
        for cand_id in gt[s1_id]:
            if cand_id in s2_india.index:
                row2 = s2_india.loc[cand_id]
                if row2["script_type"] != "latin":
                    non_latin_pairs.append((s1_id, row1, cand_id, row2))
            elif cand_id in s3_india.index:
                row3 = s3_india.loc[cand_id]
                if row3["script_type"] != "latin":
                    non_latin_pairs.append((s1_id, row1, cand_id, row3))

print(f"Found {len(non_latin_pairs)} non-latin ground truth pairs in this slice!")

for i, (s1_id, r1, c_id, r2) in enumerate(non_latin_pairs[:15]):
    print(f"\n--- Pair {i+1} [{r2['script_type']}] ---")
    print(f"S1 ({s1_id}): {r1['name_raw']} | core: {r1['name_core']}")
    print(f"Cand ({c_id}): {r2['name_raw']} | core: {r2['name_core']}")
    print(f"Addr S1: {r1['addr_norm']}")
    print(f"Addr Cand: {r2['addr_norm']}")
