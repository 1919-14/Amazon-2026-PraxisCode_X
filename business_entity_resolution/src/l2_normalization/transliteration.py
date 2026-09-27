"""Indic transliteration and script-aware romanization.

Converts Indic scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali,
Malayalam, Oriya, Gurmukhi) into clean, phonetic Latin representations using
the `indic-transliteration` package.

Scheme Details:
--------------
- Primary Scheme: `sanscript.OPTITRANS` (falls back to `sanscript.ITRANS`).
  OPTITRANS produces standard, intuitive English phonetic transliterations
  (e.g., 'shakti', 'investments', 'enterprises', 'builders') that align closely
  with English business name conventions and Jellyfish phonetic encoders (Metaphone/Soundex).
- Diacritic Normalization: Unicode NFD decomposition followed by removal of
  combining marks (category 'Mn') to ensure clean ASCII strings.
- Latin text is detected and left untouched (zero overhead for Latin records).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

try:
    from indic_transliteration import sanscript

    _INDIC_TRANSLITERATION_AVAILABLE = True
except ImportError:  # pragma: no cover
    sanscript = None  # type: ignore[assignment]
    _INDIC_TRANSLITERATION_AVAILABLE = False


# Map from script_type name to sanscript scheme constant
SCRIPT_TO_SANSCRIPT = {
    "devanagari": getattr(sanscript, "DEVANAGARI", "devanagari"),
    "tamil": getattr(sanscript, "TAMIL", "tamil"),
    "telugu": getattr(sanscript, "TELUGU", "telugu"),
    "kannada": getattr(sanscript, "KANNADA", "kannada"),
    "gujarati": getattr(sanscript, "GUJARATI", "gujarati"),
    "bengali": getattr(sanscript, "BENGALI", "bengali"),
    "malayalam": getattr(sanscript, "MALAYALAM", "malayalam"),
    "oriya": getattr(sanscript, "ORIYA", "oriya"),
    "gurmukhi": getattr(sanscript, "GURMUKHI", "gurmukhi"),
}

# Unicode codepoint ranges for quick script detection
UNICODE_SCRIPT_RANGES: list[tuple[int, int, str, str]] = [
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


def detect_indic_script(text: str) -> tuple[str, Optional[str]]:
    """Detect if text contains Indic characters and return (script_name, sanscript_scheme).

    Returns ("latin", None) if no Indic characters are found.
    """
    if not text:
        return "latin", None
    for ch in text:
        cp = ord(ch)
        for start, end, name, scheme_attr in UNICODE_SCRIPT_RANGES:
            if start <= cp <= end:
                scheme = getattr(sanscript, scheme_attr, name) if sanscript else name
                return name, scheme
    return "latin", None


def romanize(text: str, script_hint: Optional[str] = None) -> str:
    """Romanize Indic scripts to clean ASCII Latin text.

    Args:
        text: Input string (business name or address).
        script_hint: Optional script name hint (e.g. from normalized parquet 'script_type').
                     If 'latin', text is returned as-is immediately.

    Returns:
        Romanized string in lowercased ASCII format with punctuation cleaned,
        or original text if already Latin or transliteration is unavailable.
    """
    if not text:
        return ""

    if script_hint == "latin":
        return text

    if not _INDIC_TRANSLITERATION_AVAILABLE or sanscript is None:
        return text

    # Determine source scheme
    source_scheme = None
    if script_hint and script_hint in SCRIPT_TO_SANSCRIPT:
        source_scheme = SCRIPT_TO_SANSCRIPT[script_hint]
    else:
        _, source_scheme = detect_indic_script(text)

    if source_scheme is None:
        # Pure Latin text
        return text

    try:
        # Use OPTITRANS for standard phonetic transliteration
        target_scheme = sanscript.OPTITRANS
        transliterated = sanscript.transliterate(text, source_scheme, target_scheme)
        
        # Decompose Unicode and strip combining diacritical marks
        nfd = unicodedata.normalize("NFD", transliterated)
        ascii_text = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        
        # Replace non-alphanumeric characters with spaces (& -> and, rest -> space)
        prepped = ascii_text.replace("&", " and ")
        cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in prepped)
        
        # Collapse whitespace and lowercase
        return re.sub(r"\s+", " ", cleaned).strip().lower()
    except Exception:
        # Fallback gracefully to original text
        return text


def cleaned_chars(s: str) -> str:
    """Helper to clean residual characters from transliteration."""
    return s.replace("&", " and ")
