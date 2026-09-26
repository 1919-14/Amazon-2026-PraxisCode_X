"""Low-level Unicode text utilities for normalization, tokenization, and script detection."""

import re
import unicodedata


def nfkc_normalize(s: str) -> str:
    """Apply Unicode NFKC normalization (resolves compatibility equivalents)."""
    return unicodedata.normalize("NFKC", s)


def strip_accents(s: str) -> str:
    """Strip combining accent marks from Latin text only (NFD decompose → remove Mn category)."""
    nfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def lowercase(s: str) -> str:
    """Lowercase a string using Python built-in."""
    return s.lower()


def normalize_punctuation(s: str) -> str:
    """Expand '&', remove non-alphanumeric chars, collapse whitespace."""
    s = s.replace("&", " and ")
    # Remove all chars that are not alphanumeric, whitespace, or accented Latin letters
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    # Collapse runs of whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokenize(s: str) -> list[str]:
    """Split a normalized string on whitespace and return non-empty tokens."""
    return [tok for tok in s.split() if tok]


def extract_digit_runs(s: str) -> list[str]:
    """Return all maximal digit runs, deduplicated while preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for match in re.finditer(r"\d+", s):
        d = match.group()
        if d not in seen:
            seen.add(d)
            result.append(d)
    return result


def detect_script(s: str) -> str:
    """Return dominant script: 'devanagari'|'tamil'|'telugu'|'kannada'|'latin'|'other'.

    Indic scripts take priority over Latin for mixed strings.
    """
    has_latin = False
    for ch in s:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return "devanagari"
        if 0x0B80 <= cp <= 0x0BFF:
            return "tamil"
        if 0x0C00 <= cp <= 0x0C7F:
            return "telugu"
        if 0x0C80 <= cp <= 0x0CFF:
            return "kannada"
        if 0x0041 <= cp <= 0x024F:
            has_latin = True
    return "latin" if has_latin else "other"
