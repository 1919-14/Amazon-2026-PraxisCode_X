"""Structured address field parser: postal codes, house numbers, states, and cities."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Postal code regex patterns per country
# ---------------------------------------------------------------------------

POSTAL_PATTERNS: dict[str, str] = {
    "us": r"\b\d{5}(?:-\d{4})?\b",
    "india": r"\b\d{6}\b",
    "france": r"\b\d{5}\b",
}

# ---------------------------------------------------------------------------
# State / region reference sets
# ---------------------------------------------------------------------------

US_STATES: set[str] = {
    # 2-letter abbreviations
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
    "dc",
    # Full names (lowercased)
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming", "district of columbia",
}

INDIA_STATES: set[str] = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand",
    "karnataka", "kerala", "madhya pradesh", "maharashtra", "manipur",
    "meghalaya", "mizoram", "nagaland", "odisha", "punjab", "rajasthan",
    "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "west bengal", "delhi", "jammu and kashmir", "ladakh",
    "chandigarh", "puducherry", "andaman and nicobar islands",
    "dadra and nagar haveli and daman and diu", "lakshadweep",
    # Common abbreviations
    "ap", "up", "mp", "wb", "tn", "mh", "ka", "gj", "rj", "pb",
    "hr", "hp", "jk", "uk", "jh", "od", "as", "br",
}

FRANCE_REGIONS: set[str] = {
    "auvergne-rhone-alpes", "bourgogne-franche-comte", "bretagne",
    "centre-val de loire", "corse", "grand est", "hauts-de-france",
    "ile-de-france", "normandie", "nouvelle-aquitaine", "occitanie",
    "pays de la loire", "provence-alpes-cote d'azur",
    # Common city / department names that may appear as state
    "paris", "lyon", "marseille", "toulouse", "nice", "nantes",
    "strasbourg", "montpellier", "bordeaux", "lille",
}

_COUNTRY_STATES: dict[str, set[str]] = {
    "us": US_STATES,
    "india": INDIA_STATES,
    "france": FRANCE_REGIONS,
}


def parse_postal(addr_norm: str, country_norm: str) -> str:
    """Extract matching postal code for the given country. Returns '' if none.

    Takes the last match to avoid confusing leading 5-digit house numbers with US postal codes.
    """
    pattern = POSTAL_PATTERNS.get(country_norm.lower())
    if not pattern:
        return ""
    matches = re.findall(pattern, addr_norm)
    return matches[-1] if matches else ""


def parse_house_number(addr_norm: str) -> str:
    """Return the first purely numeric or N-M format token as house number."""
    tokens = addr_norm.split()
    for tok in tokens:
        if re.fullmatch(r"\d+(?:-\d+)?", tok):
            return tok
    return ""


def parse_state(addr_tokens: list[str], country_norm: str) -> str:
    """Return last token matching the country's state set (case-insensitive)."""
    states = _COUNTRY_STATES.get(country_norm.lower(), set())
    if not states:
        return ""
    for tok in reversed(addr_tokens):
        if tok.lower() in states:
            return tok
    return ""


def parse_city(addr_tokens: list[str], addr_postal: str, addr_state: str = "") -> str:
    """Return city token deduced from postal code / state context, or else last alphabetic token."""
    if addr_postal and addr_tokens:
        for i, tok in enumerate(addr_tokens):
            if tok == addr_postal:
                # If there is a token after postal and it is alphabetic (French / European format)
                if i + 1 < len(addr_tokens) and addr_tokens[i + 1].isalpha():
                    return addr_tokens[i + 1]
                # In US/India format: token before postal might be state
                if i > 0:
                    prev = addr_tokens[i - 1]
                    if addr_state and prev == addr_state.lower() and i > 1 and addr_tokens[i - 2].isalpha():
                        return addr_tokens[i - 2]
                    return prev
    # Fallback: last alphabetic token that is not the state
    if addr_state:
        for tok in reversed(addr_tokens):
            if tok.isalpha() and tok.lower() != addr_state.lower():
                return tok
    for tok in reversed(addr_tokens):
        if tok.isalpha():
            return tok
    return ""
