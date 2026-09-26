"""Country-specific abbreviation expansion tables and legal suffix extraction."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Abbreviation dictionaries (token → expanded form, already lowercased)
# ---------------------------------------------------------------------------

ABBREV_US: dict[str, str] = {
    "corp": "corporation",
    "inc": "incorporated",
    "llc": "limited liability company",
    "co": "company",
    "ltd": "limited",
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "hwy": "highway",
}

ABBREV_INDIA: dict[str, str] = {
    "pvt": "private",
    "ltd": "limited",
    "llp": "limited liability partnership",
    "rd": "road",
    "st": "street",
    "opp": "opposite",
    "nr": "near",
    "nagar": "nagar",
    "colony": "colony",
}

ABBREV_FRANCE: dict[str, str] = {
    "sarl": "societe a responsabilite limitee",
    "sas": "societe par actions simplifiee",
    "sa": "societe anonyme",
    "eurl": "entreprise unipersonnelle a responsabilite limitee",
    "sci": "societe civile immobiliere",
    "av": "avenue",
    "bd": "boulevard",
    "ste": "sainte",
    "st": "saint",
}

# Map from country_norm (lowercased country string) → abbreviation dict
_COUNTRY_ABBREV: dict[str, dict[str, str]] = {
    "us": ABBREV_US,
    "india": ABBREV_INDIA,
    "france": ABBREV_FRANCE,
}

# ---------------------------------------------------------------------------
# Legal suffix tables per country
# ---------------------------------------------------------------------------

LEGAL_SUFFIXES: dict[str, set[str]] = {
    "us": {
        "corporation",
        "incorporated",
        "limited liability company",
        "company",
        "limited",
    },
    "india": {
        "private limited",
        "private",
        "limited",
        "limited liability partnership",
    },
    "france": {
        "societe a responsabilite limitee",
        "societe par actions simplifiee",
        "societe anonyme",
        "entreprise unipersonnelle a responsabilite limitee",
        "entreprise unipersonnelle",
        "societe civile immobiliere",
    },
}


def _get_abbrev(country_norm: str) -> dict[str, str]:
    """Return abbreviation dict for country; empty dict for unknown countries."""
    return _COUNTRY_ABBREV.get(country_norm.lower(), {})


def expand_tokens(tokens: list[str], country_norm: str) -> list[str]:
    """Expand abbreviated tokens using the country-specific abbreviation table.

    Multi-word expansions are inserted as a single string element (not re-split),
    so downstream logic treats them as one logical unit.
    """
    abbrev = _get_abbrev(country_norm)
    result: list[str] = []
    for tok in tokens:
        expanded = abbrev.get(tok, tok)
        result.append(expanded)
    return result


def extract_legal_suffix(
    tokens: list[str], country_norm: str
) -> tuple[list[str], str]:
    """Remove trailing legal suffix token(s) and return (core_tokens, suffix_str).

    Checks from longest candidate suffix (up to 5 tokens) down to 1 token to ensure
    multi-word suffixes like 'limited liability company' take precedence over 'company'.
    """
    suffixes = LEGAL_SUFFIXES.get(country_norm.lower(), set())
    if not tokens or not suffixes:
        return tokens, ""

    max_len = min(len(tokens), 5)
    for k in range(max_len, 0, -1):
        candidate = " ".join(tokens[-k:])
        if candidate in suffixes:
            return tokens[:-k], candidate

    return tokens, ""
