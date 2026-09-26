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
    # Official region names (hyphen-containing ones stored post-normalization)
    "auvergne rhone alpes",   # from Auvergne-Rhône-Alpes
    "bourgogne franche comte",  # from Bourgogne-Franche-Comté
    "bretagne",
    "centre val de loire",    # from Centre-Val de Loire
    "corse",
    "grand est",
    "hauts de france",        # from Hauts-de-France
    "ile de france",          # from Île-de-France
    "normandie",
    "nouvelle aquitaine",     # from Nouvelle-Aquitaine
    "occitanie",
    "pays de la loire",
    "provence alpes cote d azur",  # from Provence-Alpes-Côte d'Azur
    # Major cities that commonly appear as region-level identifiers
    "paris", "lyon", "marseille", "toulouse", "nice", "nantes",
    "strasbourg", "montpellier", "bordeaux", "lille",
}

_COUNTRY_STATES: dict[str, set[str]] = {
    "us": US_STATES,
    "india": INDIA_STATES,
    "france": FRANCE_REGIONS,
}

# Street-type tokens that terminate a backwards city scan. Commas are stripped
# during normalization, so without this the city would swallow the street type
# (e.g. 'Street, Springfield, IL' -> 'street springfield').
_STREET_TYPES: set[str] = {
    "street", "st", "drive", "dr", "road", "rd", "avenue", "ave", "av",
    "boulevard", "blvd", "bd", "lane", "ln", "way", "court", "ct",
    "place", "pl", "highway", "hwy", "suite", "ste", "apartment", "apt",
    "building", "bldg", "parkway", "pkwy", "circle", "cir", "terrace",
    "ter", "square", "sq", "trail", "trl", "loop", "route", "rt",
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


def _match_multi_token(tokens: list[str], ref_set: set[str], max_tokens: int) -> tuple[int, str]:
    """Try to match the last max_tokens joined tokens against ref_set.

    Returns (matched_token_count, matched_string) or (0, '') if no match.
    """
    for k in range(min(max_tokens, len(tokens)), 0, -1):
        candidate = " ".join(tokens[-k:])
        if candidate in ref_set:
            return k, candidate
    return 0, ""


def parse_address_components(
    addr_tokens: list[str], country_norm: str
) -> tuple[str, str]:
    """Return (state, city) from a tokenized normalized address.

    Strategy:
    - France: scan the last 1-3 tokens (joined) against FRANCE_REGIONS. The
      city is the token(s) just before the state.
    - US: the last single token that is a US state abbreviation/name. City
      is up to 2 alphabetic tokens immediately before the state, stopping at
      the first street-type token (captures 'high point' / 'new york' without
      swallowing 'drive' / 'street').
    - India: last 1-2 tokens against INDIA_STATES. City is the last alphabetic
      token before the state.
    - Fallback: state='', city = last alphabetic token.
    """
    cn = country_norm.lower()
    alpha_tokens = [t for t in addr_tokens if t.isalpha()]

    if cn == "france":
        # Match region at tail (up to 3 tokens, hyphenated regions normalised to spaces)
        k, state = _match_multi_token(addr_tokens, FRANCE_REGIONS, 3)
        if k:
            remaining = addr_tokens[:-k]
            # City = last alphabetic token in remaining that is not a stopword
            city = ""
            for tok in reversed(remaining):
                if tok.isalpha():
                    city = tok
                    break
            return state, city
        # No region match — no state, city = last alphabetic token
        city = alpha_tokens[-1] if alpha_tokens else ""
        return "", city

    elif cn == "us":
        # US: state is the last token matching US_STATES
        state_idx = -1
        state = ""
        for i in range(len(addr_tokens) - 1, -1, -1):
            if addr_tokens[i].lower() in US_STATES:
                state_idx = i
                state = addr_tokens[i]
                break
        if state_idx > 0:
            # City: alphabetic tokens immediately preceding the state, stopping at
            # the first street-type token. This captures multi-word cities
            # ('high point', 'los angeles') without swallowing the street type.
            city_parts: list[str] = []
            j = state_idx - 1
            while j >= 0 and addr_tokens[j].isalpha() and len(city_parts) < 2:
                if addr_tokens[j].lower() in _STREET_TYPES:
                    break
                city_parts.insert(0, addr_tokens[j])
                j -= 1
            city = " ".join(city_parts) if city_parts else ""
            return state, city
        # Fallback
        city = alpha_tokens[-1] if alpha_tokens else ""
        return "", city

    elif cn == "india":
        # India: match last 1-2 tokens against INDIA_STATES
        k, state = _match_multi_token(addr_tokens, INDIA_STATES, 2)
        if k:
            remaining = addr_tokens[:-k]
            city = ""
            for tok in reversed(remaining):
                if tok.isalpha():
                    city = tok
                    break
            return state, city
        city = alpha_tokens[-1] if alpha_tokens else ""
        return "", city

    else:
        # Unknown country — no state, city = last alphabetic token
        city = alpha_tokens[-1] if alpha_tokens else ""
        return "", city


def parse_state(addr_tokens: list[str], country_norm: str) -> str:
    """Return state string (backward-compat wrapper around parse_address_components)."""
    state, _ = parse_address_components(addr_tokens, country_norm)
    return state


def parse_city(addr_tokens: list[str], addr_postal: str, addr_state: str = "") -> str:
    """Return city string; addr_postal used only for European post-code-then-city format.

    For the common path (US/India/France), the coordinated parse_address_components
    already computed city correctly, so this function is called only as a fallback
    from engine.py when addr_state is already known.
    """
    if addr_postal and addr_tokens:
        for i, tok in enumerate(addr_tokens):
            if tok == addr_postal:
                # French / European: postal then city
                if i + 1 < len(addr_tokens) and addr_tokens[i + 1].isalpha():
                    return addr_tokens[i + 1]
    # Fallback: last alphabetic non-state token
    for tok in reversed(addr_tokens):
        if tok.isalpha() and (not addr_state or tok.lower() != addr_state.lower()):
            return tok
    return ""

