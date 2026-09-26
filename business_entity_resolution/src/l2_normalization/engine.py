"""Core normalization engine: per-record normalization and chunk-level DataFrame processing."""

from __future__ import annotations

import pandas as pd

from l2_normalization.abbreviations import expand_tokens, extract_legal_suffix
from l2_normalization.address_parser import (
    parse_city,
    parse_house_number,
    parse_postal,
    parse_state,
)
from l2_normalization.phonetic import phonetic_encode
from l2_normalization.stopwords import remove_stopwords
from l2_normalization.text_utils import (
    detect_script,
    extract_digit_runs,
    lowercase,
    nfkc_normalize,
    normalize_punctuation,
    strip_accents,
    tokenize,
)

# Output column order — must match the parquet schema exactly
OUTPUT_COLUMNS: list[str] = [
    "entity_id", "source", "country_raw", "country_norm",
    "name_raw", "name_norm", "name_tokens", "name_phonetic",
    "name_legal_suffix", "name_core",
    "addr_raw", "addr_norm", "addr_tokens",
    "addr_house_number", "addr_postal", "addr_state", "addr_city",
    "addr_digits",
    "script_type",
    "is_missing_name", "is_missing_addr",
    "is_missing_postal", "is_missing_state", "is_missing_city",
]


def _infer_source(entity_id: str) -> str:
    """Infer source label from entity_id prefix."""
    if entity_id.startswith("S1-"):
        return "S1"
    if entity_id.startswith("S2-"):
        return "S2"
    if entity_id.startswith("S3-"):
        return "S3"
    return "UNKNOWN"


def _normalize_country(country: str) -> str:
    """Lowercase and strip the country string; normalize common variants."""
    cn = country.strip().lower()
    # Map common country variants to canonical form
    _MAP = {
        "united states": "us",
        "united states of america": "us",
        "usa": "us",
        "u.s.": "us",
        "u.s.a.": "us",
        "us": "us",
        "india": "india",
        "in": "india",
        "france": "france",
        "fr": "france",
    }
    return _MAP.get(cn, cn)


def normalize_record(row: dict) -> dict:
    """Normalize one raw record dict into the full parquet output schema dict."""
    entity_id: str = str(row.get("entity_id", ""))
    name_raw: str = str(row.get("business_name", ""))
    addr_raw: str = str(row.get("business_address", ""))
    country_raw: str = str(row.get("country", ""))

    source = _infer_source(entity_id)
    country_norm = _normalize_country(country_raw)

    # ------------------------------------------------------------------
    # NAME NORMALIZATION
    # ------------------------------------------------------------------
    is_missing_name = name_raw.strip() == ""

    # Step 1: Unicode normalization
    name_clean = nfkc_normalize(name_raw)

    # Step 2: Detect script on raw text (before lowercasing destroys casing info)
    script_type = detect_script(name_clean)

    # Step 3: Strip accents only for Latin script
    if script_type == "latin":
        name_clean = strip_accents(name_clean)

    # Step 4: Lowercase + punctuation normalization
    name_clean = lowercase(normalize_punctuation(name_clean))

    # Step 5: Tokenize
    name_tokens_raw = tokenize(name_clean)

    # Step 6: Expand abbreviations
    name_tokens_expanded = expand_tokens(name_tokens_raw, country_norm)

    # Step 7: Re-tokenize expanded tokens (multi-word expansions may contain spaces)
    name_tokens_flat: list[str] = []
    for t in name_tokens_expanded:
        name_tokens_flat.extend(tokenize(t))

    # Step 8: Extract legal suffix from the flat expanded token list
    core_tokens_with_sw, name_legal_suffix = extract_legal_suffix(
        name_tokens_flat, country_norm
    )

    # Step 9: Remove stopwords from core
    core_tokens = remove_stopwords(core_tokens_with_sw)

    # Step 10: Build output name strings
    name_norm = " ".join(name_tokens_flat)          # full normalized, no suffix removed
    name_core = " ".join(core_tokens)               # stopwords removed, suffix removed

    # Step 11: Phonetic codes
    name_phonetic = phonetic_encode(name_tokens_flat, script_type)

    # ------------------------------------------------------------------
    # ADDRESS NORMALIZATION
    # ------------------------------------------------------------------
    is_missing_addr = addr_raw.strip() == ""

    addr_clean = nfkc_normalize(addr_raw)
    if script_type == "latin":   # use same script classification as name
        addr_clean = strip_accents(addr_clean)
    addr_clean = lowercase(normalize_punctuation(addr_clean))
    addr_tokens = tokenize(addr_clean)
    addr_norm = " ".join(addr_tokens)

    addr_postal = parse_postal(addr_norm, country_norm)
    addr_house_number = parse_house_number(addr_norm)
    addr_state = parse_state(addr_tokens, country_norm)
    addr_city = parse_city(addr_tokens, addr_postal, addr_state)
    addr_digits = extract_digit_runs(addr_raw)

    is_missing_postal = addr_postal == ""
    is_missing_state = addr_state == ""
    is_missing_city = addr_city == ""

    return {
        "entity_id": entity_id,
        "source": source,
        "country_raw": country_raw,
        "country_norm": country_norm,
        "name_raw": name_raw,
        "name_norm": name_norm,
        "name_tokens": name_tokens_flat,
        "name_phonetic": name_phonetic,
        "name_legal_suffix": name_legal_suffix,
        "name_core": name_core,
        "addr_raw": addr_raw,
        "addr_norm": addr_norm,
        "addr_tokens": addr_tokens,
        "addr_house_number": addr_house_number,
        "addr_postal": addr_postal,
        "addr_state": addr_state,
        "addr_city": addr_city,
        "addr_digits": addr_digits,
        "script_type": script_type,
        "is_missing_name": is_missing_name,
        "is_missing_addr": is_missing_addr,
        "is_missing_postal": is_missing_postal,
        "is_missing_state": is_missing_state,
        "is_missing_city": is_missing_city,
    }


def normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a 100K-row chunk and return a DataFrame with the parquet schema."""
    records = [normalize_record(row) for row in df.to_dict(orient="records")]
    out_df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    # Ensure no NaN: fill string columns with "" and list columns with []
    for col in out_df.columns:
        if out_df[col].dtype == object:
            out_df[col] = out_df[col].apply(
                lambda v: v if v is not None else ([] if col in {
                    "name_tokens", "name_phonetic", "addr_tokens", "addr_digits"
                } else "")
            )
    return out_df
