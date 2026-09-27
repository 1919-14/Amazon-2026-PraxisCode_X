import pandas as pd
import glob
import json

def load_first_shard(split, source):
    path = sorted(glob.glob(f"artifacts/normalized/{split}_s{source}/part_*.parquet"))[0]
    return pd.read_parquet(path)

def dump(rec, label):
    print(f"\n=== {label} ===")
    for k, v in rec.items():
        if isinstance(v, list):
            print(f"  {k}: {v[:5]}{'...' if len(v) > 5 else ''}")
        else:
            print(f"  {k}: {repr(v)}")

# --- Check 1: Train S1 sample ---
s1 = load_first_shard("train", 1)
dump(s1.iloc[0].to_dict(), "Train S1 first record")

# --- Check 2: Train S2 sample ---
s2 = load_first_shard("train", 2)
dump(s2.iloc[0].to_dict(), "Train S2 first record")

# --- Check 3: France sample from test S2 ---
print("\nScanning test S2 for a France record...")
found_fr = None
for f in sorted(glob.glob("artifacts/normalized/test_s2/part_*.parquet"))[:5]:
    df = pd.read_parquet(f)
    fr = df[df["country_norm"] == "france"]
    if len(fr) > 0:
        found_fr = fr.iloc[0].to_dict()
        break
if found_fr:
    dump(found_fr, "France record (test_s2)")
else:
    print("  ⚠️ No France records found in first 5 shards")

# --- Check 4: Non-Latin script sample ---
print("\nScanning train S2 for a non-Latin script record...")
found_indic = None
for f in sorted(glob.glob("artifacts/normalized/train_s2/part_*.parquet"))[:10]:
    df = pd.read_parquet(f)
    indic = df[df["script_type"] != "latin"]
    if len(indic) > 0:
        found_indic = indic.iloc[0].to_dict()
        break
if found_indic:
    dump(found_indic, f"Non-Latin record (script_type={found_indic['script_type']})")
    print(f"  phonetic empty? {len(found_indic['name_phonetic']) == 0}")
else:
    print("  ⚠️ No non-Latin records found in first 10 shards")

# --- Check 5: Acme Corp equivalence ---
print("\n=== Acme Corp equivalence check ===")
targets = {"acme corp", "acme corporation", "ACME CORP.", "Acme Pvt Ltd"}
cores = {}
for f in sorted(glob.glob("artifacts/normalized/train_s1/part_*.parquet"))[:5]:
    df = pd.read_parquet(f)
    for _, r in df.iterrows():
        if r["name_raw"] in targets:
            cores[r["name_raw"]] = r["name_core"]
            if len(cores) >= 4:
                break
    if len(cores) >= 4:
        break

if cores:
    for raw, core in cores.items():
        print(f"  {raw!r:25} -> core={core!r}")
    unique_cores = set(cores.values())
    print(f"  Unique name_core values: {unique_cores}")
    print(f"  All variants collapse to same core? {len(unique_cores) == 1}")
else:
    print("  ⚠️ None of the test names found in first 5 shards")

# --- Check 6: NaN and field sanity ---
print("\n=== NaN / schema sanity ===")
df = pd.read_parquet(sorted(glob.glob("artifacts/normalized/train_s1/part_*.parquet"))[0])
print(f"  Rows in first shard: {len(df)}")
print(f"  Columns: {len(df.columns)}")
print(f"  Any NaN? {df.isna().sum().sum()}")

# --- Check 7: Postal extraction sanity ---
print("\n=== Postal extraction sample (US records) ===")
us = df[df["country_norm"] == "us"].head(5)
for _, r in us.iterrows():
    print(f"  {r['entity_id']}: postal={r['addr_postal']!r} state={r['addr_state']!r} city={r['addr_city']!r}")

print("\n=== Summary counts ===")
print(f"  script_type values in first train_s1 shard: {df['script_type'].value_counts().to_dict()}")
print(f"  country_norm values in first train_s1 shard: {df['country_norm'].value_counts().to_dict()}")