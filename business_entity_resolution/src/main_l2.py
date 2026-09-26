"""Layer 2 entry point: run normalization pipeline + verification + sanity checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for p in [str(SRC_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import psutil

from config import PATH_ARTIFACTS_DIR, PATH_OUTPUT_DIR
from l2_normalization.engine import normalize_record
from l2_normalization.pipeline import process_split


def get_ram_gb() -> float:
    """Return current process RSS in GB."""
    return psutil.Process().memory_info().rss / (1024 ** 3)


def verify_no_nan(shard_path: Path) -> bool:
    """Return True if shard has zero NaN values."""
    df = pd.read_parquet(shard_path)
    total_na = df.isna().sum().sum()
    return total_na == 0


def print_record(label: str, rec: dict) -> None:
    """Pretty-print a normalized record."""
    print(f"\n  ── {label} ──")
    fields = [
        "entity_id", "source", "country_raw", "country_norm",
        "name_raw", "name_norm", "name_core", "name_legal_suffix",
        "name_phonetic", "addr_raw", "addr_norm",
        "addr_house_number", "addr_postal", "addr_state", "addr_city",
        "addr_digits", "script_type",
        "is_missing_name", "is_missing_addr",
    ]
    for f in fields:
        v = rec.get(f, "N/A")
        print(f"    {f:<22}: {v}")


def run_sanity_checks(all_results: dict) -> None:
    """Run spot-checks and sanity assertions."""
    print("\n" + "=" * 65)
    print("🔍 SANITY CHECKS")
    print("=" * 65)

    # Check 1: "Acme Corp" vs "ACME CORPORATION" → same name_core?
    row_a = {"entity_id": "S1-0", "business_name": "Acme Corp.",
              "business_address": "", "country": "US"}
    row_b = {"entity_id": "S1-1", "business_name": "ACME CORPORATION",
              "business_address": "", "country": "US"}
    rec_a = normalize_record(row_a)
    rec_b = normalize_record(row_b)
    same_core = rec_a["name_core"] == rec_b["name_core"]
    print(f"\n  ✅ 'Acme Corp' name_core    : '{rec_a['name_core']}'")
    print(f"  ✅ 'ACME CORPORATION' core  : '{rec_b['name_core']}'")
    print(f"  → Same name_core? {'✅ YES' if same_core else '❌ NO'}")

    # Check 2: French accent stripping "réseau"
    row_c = {"entity_id": "S2-0", "business_name": "Réseau Hôtel Naïve",
              "business_address": "12 avenue de la Paix", "country": "France"}
    rec_c = normalize_record(row_c)
    accent_stripped = "reseau" in rec_c["name_norm"]
    print(f"\n  ✅ 'Réseau Hôtel Naïve' name_norm: '{rec_c['name_norm']}'")
    print(f"  → 'réseau' → 'reseau' stripped? {'✅ YES' if accent_stripped else '❌ NO'}")

    # Check 3: France SARL legal suffix extraction
    row_fr = {"entity_id": "S2-2", "business_name": "Boulangerie Dupont SARL",
              "business_address": "12 rue de la Paix 75001 Paris", "country": "France"}
    rec_fr = normalize_record(row_fr)
    fr_suffix_ok = rec_fr["name_legal_suffix"] == "societe a responsabilite limitee"
    print(f"\n  ✅ France SARL:")
    print(f"    name_core        : '{rec_fr['name_core']}'")
    print(f"    name_legal_suffix: '{rec_fr['name_legal_suffix']}'")
    print(f"    addr_postal      : '{rec_fr['addr_postal']}'")
    print(f"    addr_city        : '{rec_fr['addr_city']}'")
    print(f"  → Suffix extracted? {'✅ YES' if fr_suffix_ok else '❌ NO'}")

    # Check 4: Devanagari record
    devanagari_name = "अमेज़न इंडिया प्राइवेट लिमिटेड"
    row_d = {"entity_id": "S3-0", "business_name": devanagari_name,
              "business_address": "Mumbai", "country": "India"}
    rec_d = normalize_record(row_d)
    is_devanagari = rec_d["script_type"] == "devanagari"
    no_phonetic = rec_d["name_phonetic"] == []
    print(f"\n  ✅ Devanagari record:")
    print(f"    script_type: {rec_d['script_type']}")
    print(f"    name_phonetic: {rec_d['name_phonetic']}")
    print(f"  → script_type == 'devanagari'? {'✅ YES' if is_devanagari else '❌ NO'}")
    print(f"  → name_phonetic empty?         {'✅ YES' if no_phonetic else '❌ NO'}")

    # Check 5: NaN-free first shard
    shard_path = PATH_ARTIFACTS_DIR / "normalized" / "train_s1" / "part_00000.parquet"
    if shard_path.exists():
        ok = verify_no_nan(shard_path)
        print(f"\n  → NaN check train_s1 part_00000: {'✅ PASS' if ok else '❌ FAIL (NaN found)'}")

    # Check 6: US postal code parsing (with 5-digit house number)
    row_us = {"entity_id": "S1-2", "business_name": "Test Corp",
              "business_address": "17560 Ellis Road, Austin, TX 78759", "country": "US"}
    rec_us = normalize_record(row_us)
    print(f"\n  ✅ US postal parse:")
    print(f"    addr_postal : '{rec_us['addr_postal']}'  (expect: '78759')")
    print(f"    addr_state  : '{rec_us['addr_state']}'   (expect: 'tx')")
    print(f"    addr_city   : '{rec_us['addr_city']}'    (expect: 'austin')")
    print(f"    addr_house_number: '{rec_us['addr_house_number']}' (expect: '17560')")

    # Check 7: India postal & suffix
    row_in = {"entity_id": "S2-1", "business_name": "Tech Pvt Ltd",
              "business_address": "45 MG Road Bangalore Karnataka 560001", "country": "India"}
    rec_in = normalize_record(row_in)
    print(f"\n  ✅ India postal parse:")
    print(f"    addr_postal      : '{rec_in['addr_postal']}' (expect: '560001')")
    print(f"    addr_city        : '{rec_in['addr_city']}'   (expect: 'bangalore')")
    print(f"    name_legal_suffix: '{rec_in['name_legal_suffix']}' (expect: 'private limited')")


def load_sample_rows(split: str, source_idx: int, n: int = 3) -> list[dict]:
    """Load first N rows from first shard of a split/source combination."""
    shard = PATH_ARTIFACTS_DIR / "normalized" / f"{split}_s{source_idx}" / "part_00000.parquet"
    if not shard.exists():
        return []
    df = pd.read_parquet(shard).head(n)
    return df.to_dict(orient="records")


def find_france_sample(split: str = "test") -> dict | None:
    """Find a France record from test_s2 first shard."""
    shard = PATH_ARTIFACTS_DIR / "normalized" / f"{split}_s2" / "part_00000.parquet"
    if not shard.exists():
        return None
    df = pd.read_parquet(shard)
    france_rows = df[df["country_norm"] == "france"]
    if france_rows.empty:
        return None
    return france_rows.iloc[0].to_dict()


def find_nonlatin_sample() -> dict | None:
    """Find first non-Latin record from train_s1 or train_s2."""
    for split_src in [("train", 1), ("train", 2), ("train", 3)]:
        shard = PATH_ARTIFACTS_DIR / "normalized" / f"{split_src[0]}_s{split_src[1]}" / "part_00000.parquet"
        if not shard.exists():
            continue
        df = pd.read_parquet(shard)
        non_latin = df[df["script_type"] != "latin"]
        if not non_latin.empty:
            return non_latin.iloc[0].to_dict()
    return None


def main() -> None:
    """Run full Layer 2 pipeline and verification."""
    print("=" * 65)
    print("🚀 LAYER 2: NORMALIZATION ENGINE")
    print("=" * 65)

    peak_ram = get_ram_gb()
    all_results: dict[str, dict] = {}

    # Step 1: Process train split
    train_results = process_split("train")
    all_results["train"] = train_results
    peak_ram = max(peak_ram, get_ram_gb())

    # Step 2: Process test split
    test_results = process_split("test")
    all_results["test"] = test_results
    peak_ram = max(peak_ram, get_ram_gb())

    # Step 3: Print shard/row summary
    print("\n" + "=" * 65)
    print("📊 LAYER 2 SUMMARY")
    print("=" * 65)
    for split_name, res in all_results.items():
        for src, stats in res.items():
            print(f"  {split_name} {src}: {stats['shards']} shards, {stats['rows']:,} rows")

    # Step 4: Sample records
    print("\n" + "=" * 65)
    print("📋 SAMPLE NORMALIZED RECORDS")
    print("=" * 65)

    train_s1_samples = load_sample_rows("train", 1, 3)
    for i, rec in enumerate(train_s1_samples):
        print_record(f"train_s1 row {i + 1}", rec)

    france_sample = find_france_sample("test")
    if france_sample:
        print_record("France sample (test_s2)", france_sample)
    else:
        print("\n  ⚠️  No France sample found in test_s2 part_00000 (may be in later shards)")

    nonlatin = find_nonlatin_sample()
    if nonlatin:
        print_record("Non-Latin sample", nonlatin)
    else:
        print("\n  ⚠️  No non-Latin sample found in first shards")

    # Step 5: Sanity checks
    run_sanity_checks(all_results)

    # Step 6: Save summary JSON
    summary = {
        "train": {src: stats for src, stats in all_results.get("train", {}).items()},
        "test": {src: stats for src, stats in all_results.get("test", {}).items()},
        "peak_ram_gb": round(peak_ram, 3),
    }
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PATH_OUTPUT_DIR / "l2_normalization_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Peak RAM: {peak_ram:.2f} GB")
    print(f"  Summary saved: {out_path}")
    print("\n" + "=" * 65)
    print("🌟 LAYER 2 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
