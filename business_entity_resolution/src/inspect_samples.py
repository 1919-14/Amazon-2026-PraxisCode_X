"""Inspect matched pairs from training data."""

import sys
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

data_dir = Path("DATA SET/student_resource/dataset/train")

gt = pd.read_csv(data_dir / "train_ground_truth.tsv", sep="\t", nrows=200)
gt_matched = gt[gt["matched_entity_ids"].fillna("").str.len() > 0].head(8)

target_s1_ids = set(gt_matched["source1_entity_id"])
target_m_ids = set([
    m for row in gt_matched["matched_entity_ids"]
    for m in str(row).split(",") if m
])

print(f"Tracking {len(target_s1_ids)} S1 entities and {len(target_m_ids)} matching S2/S3 entities...")

s1_df = pd.read_csv(data_dir / "train_source1.tsv", sep="\t")
s1_subset = s1_df[s1_df["entity_id"].isin(target_s1_ids)].set_index("entity_id")

s2_matches = []
for chunk in pd.read_csv(data_dir / "train_source2.tsv", sep="\t", chunksize=500000):
    m = chunk[chunk["entity_id"].isin(target_m_ids)]
    if not m.empty:
        s2_matches.append(m)
s2_df = pd.concat(s2_matches).set_index("entity_id") if s2_matches else pd.DataFrame()

s3_matches = []
for chunk in pd.read_csv(data_dir / "train_source3.tsv", sep="\t", chunksize=500000):
    m = chunk[chunk["entity_id"].isin(target_m_ids)]
    if not m.empty:
        s3_matches.append(m)
s3_df = pd.concat(s3_matches).set_index("entity_id") if s3_matches else pd.DataFrame()

for _, row in gt_matched.iterrows():
    s1_id = row["source1_entity_id"]
    m_ids = str(row["matched_entity_ids"]).split(",")
    print("=" * 70)
    if s1_id in s1_subset.index:
        s1 = s1_subset.loc[s1_id]
        print(f"REFERENCE [S1] {s1_id}:")
        print(f"   Name   : {s1['business_name']}")
        print(f"   Address: {s1['business_address']}")
        print(f"   Country: {s1['country']}")
    print("MATCHES:")
    for mid in m_ids:
        if not mid:
            continue
        if mid.startswith("S2") and mid in s2_df.index:
            rec = s2_df.loc[mid]
            print(f"   [{mid} (S2)]: {rec['business_name']}  |  {rec['business_address']}")
        elif mid.startswith("S3") and mid in s3_df.index:
            rec = s3_df.loc[mid]
            print(f"   [{mid} (S3)]: {rec['business_name']}  |  {rec['business_address']}")
        else:
            print(f"   [{mid}]: not in chunks")
