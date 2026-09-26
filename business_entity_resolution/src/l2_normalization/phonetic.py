"""Phonetic encoding for Latin-script business name tokens using jellyfish.metaphone."""

from __future__ import annotations

try:
    import jellyfish
    _ENCODER = "jellyfish"
except ImportError:
    jellyfish = None  # type: ignore[assignment]
    _ENCODER = "none"


def phonetic_encode(tokens: list[str], script_type: str) -> list[str]:
    """Return deduplicated metaphone codes for Latin tokens of length >= 3.

    Returns empty list for non-Latin scripts (Devanagari, Tamil, Telugu, Kannada).
    Falls back to empty list if jellyfish is unavailable.
    """
    if script_type != "latin":
        return []
    if jellyfish is None:
        return []

    seen: set[str] = set()
    result: list[str] = []
    for tok in tokens:
        # Only encode purely alphabetic tokens of length >= 3
        if len(tok) >= 3 and tok.isalpha():
            code = jellyfish.metaphone(tok)
            if code and code not in seen:
                seen.add(code)
                result.append(code)
    return result
