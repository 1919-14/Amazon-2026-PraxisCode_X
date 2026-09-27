"""Test core extraction and phonetic alignment on 100+ real cross-script pairs."""

import sys
import glob
import re
import unicodedata
import jellyfish
import pandas as pd
import pyarrow.parquet as pq
from indic_transliteration import sanscript

sys.stdout.reconfigure(encoding="utf-8")

UNICODE_RANGES = [
    (0x0900, 0x097F, "devanagari", sanscript.DEVANAGARI),
    (0x0980, 0x09FF, "bengali", sanscript.BENGALI),
    (0x0A00, 0x0A7F, "gurmukhi", sanscript.GURMUKHI),
    (0x0A80, 0x0AFF, "gujarati", sanscript.GUJARATI),
    (0x0B00, 0x0B7F, "oriya", sanscript.ORIYA),
    (0x0B80, 0x0BFF, "tamil", sanscript.TAMIL),
    (0x0C00, 0x0C7F, "telugu", sanscript.TELUGU),
    (0x0C80, 0x0CFF, "kannada", sanscript.KANNADA),
    (0x0D00, 0x0D7F, "malayalam", sanscript.MALAYALAM),
]

INDIC_LEGAL_SUFFIX_PATTERNS = [
    r"\b(private\s+limited|pvt\s+ltd|pvt\s+limited|private\s+ltd|private|limited|ltd|pvt|llp|limited\s+liability\s+partnership)\b",
    r"\b(praiveta\s+limiteda|praiveta\s+limited|praiveta|limiteda|praivet\s+limited|praivet|praivarr\s+limirrad)\b",
    r"\b(pra\s+li|pra|li|elaelapi|el\s+el\s+pi|elelbhi|elelpi)\b",
    r"\b(corporation|corp|incorporated|inc|company|co)\b",
]

def detect_script(text: str):
    for ch in text:
        cp = ord(ch)
        for start, end, name, scheme in UNICODE_RANGES:
            if start <= cp <= end:
                return name, scheme
    return "latin", None

def romanize(text: str, script_hint: str | None = None) -> str:
    if not text:
        return ""
    s_name, s_scheme = detect_script(text)
    if not s_scheme:
        # Already Latin
        return text
    try:
        # Use OPTITRANS or ITRANS for natural English phonetic mapping
        rom = sanscript.transliterate(text, s_scheme, sanscript.OPTITRANS)
        nfd = unicodedata.normalize("NFD", rom)
        clean = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        clean = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in clean)
        return " ".join(clean.lower().split())
    except Exception:
        return text

def canonical_core_name(text: str) -> str:
    """Return cleaned core name: romanized, legal suffixes removed, stripped."""
    rom = romanize(text).lower()
    for pat in INDIC_LEGAL_SUFFIX_PATTERNS:
        rom = re.sub(pat, " ", rom)
    tokens = [t for t in rom.split() if len(t) >= 2]
    return " ".join(tokens)

def phonetic_keys(text: str) -> list[str]:
    """Return metaphone codes for words in the canonical name."""
    core = canonical_core_name(text)
    codes = []
    for tok in core.split():
        if tok.isalpha():
            m = jellyfish.metaphone(tok)
            if m:
                codes.append(m)
    return codes

# Load ground truth
gt = {}
with open("DATA SET/student_resource/dataset/train/train_ground_truth.tsv", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split("\t")
        if len(p) == 2:
            gt[p[0]] = set(p[1].split(","))

cols = ["entity_id", "country_norm", "name_raw", "name_norm", "name_core", "script_type"]
s1_files = glob.glob("business_entity_resolution/artifacts/normalized/train_s1/*.parquet")[:2]
s2_files = glob.glob("business_entity_resolution/artifacts/normalized/train_s2/*.parquet")[:2]
s3_files = glob.glob("business_entity_resolution/artifacts/normalized/train_s3/*.parquet")[:2]

s1_df = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in s1_files])
s2_df = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in s2_files])
s3_df = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in s3_files])

s1_dict = s1_df[s1_df["country_norm"] == "india"].set_index("entity_id").to_dict(orient="index")
s2_dict = s2_df[s2_df["country_norm"] == "india"].set_index("entity_id").to_dict(orient="index")
s3_dict = s3_df[s3_df["country_norm"] == "india"].set_index("entity_id").to_dict(orient="index")

cand_pool = {**s2_dict, **s3_dict}

# Find non-Latin GT pairs
test_pairs = []
for s1_id, r1 in s1_dict.items():
    if s1_id in gt:
        for c_id in gt[s1_id]:
            if c_id in cand_pool:
                r2 = cand_pool[c_id]
                if r2["script_type"] != "latin":
                    test_pairs.append((s1_id, r1, c_id, r2))

print(f"Total non-Latin pairs for testing: {len(test_pairs)}")

exact_core_hits = 0
phonetic_any_hits = 0
phonetic_all_hits = 0
jaccard_scores = []

for s1_id, r1, c_id, r2 in test_pairs[:200]:
    s1_core = canonical_core_name(r1["name_raw"])
    c_core = canonical_core_name(r2["name_raw"])
    
    s1_ph = phonetic_keys(r1["name_raw"])
    c_ph = phonetic_keys(r2["name_raw"])
    
    if s1_core == c_core and s1_core:
        exact_core_hits += 1
    
    s1_set = set(s1_ph)
    c_set = set(c_ph)
    if s1_set and c_set:
        inter = len(s1_set & c_set)
        union = len(s1_set | c_set)
        jaccard = inter / union
        jaccard_scores.append(jaccard)
        if inter > 0:
            phonetic_any_hits += 1
        if inter == min(len(s1_set), len(c_set)):
            phonetic_all_hits += 1
    else:
        jaccard_scores.append(0.0)

n = min(len(test_pairs), 200)
print(f"Tested {n} non-Latin pairs:")
print(f"  Exact Canonical Core Hits: {exact_core_hits} / {n} ({exact_core_hits / n * 100:.1f}%)")
print(f"  Phonetic (Any Token Metaphone Hit): {phonetic_any_hits} / {n} ({phonetic_any_hits / n * 100:.1f}%)")
print(f"  Phonetic (All Core Token Hit): {phonetic_all_hits} / {n} ({phonetic_all_hits / n * 100:.1f}%)")
print(f"  Mean Metaphone Jaccard: {sum(jaccard_scores) / len(jaccard_scores):.3f}")

print("\nSample alignments:")
for s1_id, r1, c_id, r2 in test_pairs[:10]:
    s1_core = canonical_core_name(r1["name_raw"])
    c_core = canonical_core_name(r2["name_raw"])
    s1_ph = phonetic_keys(r1["name_raw"])
    c_ph = phonetic_keys(r2["name_raw"])
    print(f"S1:   {r1['name_raw']:<35} -> core: {s1_core:<25} -> phon: {s1_ph}")
    print(f"Cand: {r2['name_raw']:<35} -> core: {c_core:<25} -> phon: {c_ph}")
    print(f"  Overlap: {set(s1_ph) & set(c_ph)}")
    print("-" * 70)
