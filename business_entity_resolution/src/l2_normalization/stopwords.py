"""Generic high-IDF business stopwords to remove from name core tokens."""

from __future__ import annotations

BUSINESS_STOPWORDS: set[str] = {
    "the", "and", "of", "a", "an", "for", "to", "in", "on", "at", "by",
    "services", "solutions", "enterprises", "group", "holdings",
    "international", "global", "systems", "technologies",
}


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords from token list; fall back to original tokens if result is empty."""
    filtered = [t for t in tokens if t not in BUSINESS_STOPWORDS]
    return filtered if filtered else list(tokens)
