"""Prototype testing of phonetic and canonical key channels on India train data."""

import sys
import glob
import re
import time
import unicodedata
import jellyfish
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from indic_transliteration import sanscript
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout.reconfigure(encoding="utf-8")

# 1. Transliteration & Normalization
UNICODE_SCRIPT_RANGES = [
    (0x0900, 0x097F, "devanagari", "DEVANAGARI"),
    (0x0980, 0x09FF, "bengali", "BENGALI"),
    (0x0A00, 0x0A7F, "gurmukhi", "GURMUKHI"),
    (0x0A80, 0x0AFF, "gujarati", "GUJARATI"),
    (0x0B00, 0x0B7F, "oriya", "ORIYA"),
    (0x0B80, 0x0BFF, "tamil", "TAMIL"),
    (0x0C00, 0x0C7F, "telugu", "TELUGU"),
    (0x0C80, 0x0CFF, "kannada", "KANNADA"),
    (0x0D00, 0x0D7F, "malayalam", "MALAYALAM"),
]

SCRIPT_MAP = {
    "devanagari": sanscript.DEVANAGARI,
    "tamil": sanscript.TAMIL,
    "telugu": sanscript.TELUGU,
    "kannada": sanscript.KANNADA,
    "gujarati": sanscript.GUJARATI,
    "bengali": sanscript.BENGALI,
    "malayalam": sanscript.MALAYALAM,
    "oriya": sanscript.ORIYA,
    "gurmukhi": sanscript.GURMUKHI,
}

LEGAL_SUFFIX_REGEX = re.compile(
    r"\b("
    r"private\s+limited|pvt\s+ltd|pvt\s+limited|private\s+ltd|private|limited|ltd|pvt|llp|"
    r"limited\s+liability\s+partnership|limited\s+liability\s+company|llc|inc|incorporated|"
    r"corp|corporation|co|company|"
    r"praiveta\s+limiteda|praiveta\s+limited|praiveta|limiteda|praivet\s+limited|praivet|"
    r"praivarr\s+limirrad|praivatu\s+limitada|praivett\s+limitett|"
    r"pra\s+li|pra|li|elaelapi|el\s+el\s+pi|elelbhi|elelpi|el\s+el\s+bhi"
    r")\b",
    re.IGNORECASE,
)

GENERIC_TERMS_REGEX = re.compile(
    r"\b(enterprises|enterprise|industries|industry|solutions|technologies|technology|"
    r"services|service|associates|associates|international|national|global|india|"
    r"holdings|holding|group|venture|ventures|trading|traders|agency|agencies|"
    r"consultants|consultancy|consulting|commercial|commercials)\b",
    re.IGNORECASE,
)

def romanize_text(text: str, script_hint: str = "latin") -> str:
    if not text or script_hint == "latin":
        return text or ""
    
    src_scheme = SCRIPT_MAP.get(script_hint)
    if not src_scheme:
        for ch in text:
            cp = ord(ch)
            for start, end, name, attr in UNICODE_SCRIPT_RANGES:
                if start <= cp <= end:
                    src_scheme = getattr(sanscript, attr, name)
                    break
            if src_scheme:
                break
                
    if not src_scheme:
        return text

    try:
        rom = sanscript.transliterate(text, src_scheme, sanscript.OPTITRANS)
        nfd = unicodedata.normalize("NFD", rom)
        clean = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        clean = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in clean)
        return re.sub(r"\s+", " ", clean).strip().lower()
    except Exception:
        return text

def extract_canonical_core(name: str, script_hint: str = "latin") -> str:
    rom = romanize_text(name, script_hint).lower()
    rom = LEGAL_SUFFIX_REGEX.sub(" ", rom)
    tokens = [t for t in rom.split() if len(t) >= 2]
    return " ".join(tokens)

def extract_strong_core(name: str, script_hint: str = "latin") -> str:
    """Core with both legal suffixes and generic business words removed."""
    core = extract_canonical_core(name, script_hint)
    strong = GENERIC_TERMS_REGEX.sub(" ", core)
    tokens = [t for t in strong.split() if len(t) >= 2]
    return " ".join(tokens) if tokens else core

def get_phonetic_tokens(name: str, script_hint: str = "latin") -> list[str]:
    core = extract_canonical_core(name, script_hint)
    tokens = core.split()
    codes = []
    for tok in tokens:
        if tok.isalpha() and len(tok) >= 2:
            m = jellyfish.metaphone(tok)
            if m:
                codes.append(f"M_{m}")
            s = jellyfish.soundex(tok)
            if s:
                codes.append(f"S_{s}")
    return codes

def get_phonetic_string(name: str, script_hint: str = "latin") -> str:
    return " ".join(get_phonetic_tokens(name, script_hint))

def get_metaphone_key(name: str, script_hint: str = "latin") -> str:
    core = extract_canonical_core(name, script_hint)
    codes = [jellyfish.metaphone(tok) for tok in core.split() if tok.isalpha() and len(tok) >= 2]
    codes = [c for c in codes if c]
    return "_".join(codes)

print("Helper functions initialized.")
