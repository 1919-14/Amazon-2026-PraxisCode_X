"""L7: pairwise feature engineering for the entity-resolution matcher.

45 features across four blocks, computed from a normalized Source 1 record and a
normalized candidate record (plus the retrieval metadata attached to the pair):

Name block (14)
  name_ratio, name_jaro_winkler, name_token_sort, name_token_set,
  name_char3_jaccard, name_token_jaccard, name_idf_overlap, name_legal_suffix_match,
  name_partial_ratio, name_char5_jaccard, name_script_eq, name_missing_xor,
  name_is_domain_xor, name_phonetic_jaccard
Address block (13)
  addr_house_exact, addr_street_jaccard, addr_postal_exact, addr_postal_prefix3,
  addr_city_exact, addr_state_match, addr_token_jaccard, addr_char3_jaccard,
  addr_digit_conflict, addr_partial_ratio, addr_house_prefix, addr_house_conflict,
  addr_city_token_jaccard
Retrieval block (10)
  ret_exact_key_hit, ret_rrf_score, ret_retriever_agreement, ret_candidate_rank,
  ret_retrieved, ret_rank_inverse, ret_phon_score, ret_phon_hit, ret_dense_score,
  ret_dense_hit
Structural block (8)
  struct_same_country, struct_name_len_ratio, struct_addr_len_ratio,
  struct_addr_missing_xor, struct_addr_missing_both, struct_name_missing_both,
  struct_name_script_latin_xor, struct_addr_digit_count_ratio

All features are numeric. Similarity features are in ``[0, 1]``. The extra-channel
retrieval features (``ret_phon_*`` / ``ret_dense_*``) come from the optional
extra-signal sidecar and default to 0 when it is absent, so training and inference
stay in lockstep.
"""

from __future__ import annotations

import math
import re
from typing import Mapping, Optional, Sequence

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES: list[str] = [
    # name block (14)
    "name_ratio",
    "name_jaro_winkler",
    "name_token_sort",
    "name_token_set",
    "name_char3_jaccard",
    "name_token_jaccard",
    "name_idf_overlap",
    "name_legal_suffix_match",
    "name_partial_ratio",
    "name_char5_jaccard",
    "name_script_eq",
    "name_missing_xor",
    "name_is_domain_xor",
    "name_phonetic_jaccard",
    # address block (13)
    "addr_house_exact",
    "addr_street_jaccard",
    "addr_postal_exact",
    "addr_postal_prefix3",
    "addr_city_exact",
    "addr_state_match",
    "addr_token_jaccard",
    "addr_char3_jaccard",
    "addr_digit_conflict",
    "addr_partial_ratio",
    "addr_house_prefix",
    "addr_house_conflict",
    "addr_city_token_jaccard",
    # retrieval block (10)
    "ret_exact_key_hit",
    "ret_rrf_score",
    "ret_retriever_agreement",
    "ret_candidate_rank",
    "ret_retrieved",
    "ret_rank_inverse",
    "ret_phon_score",
    "ret_phon_hit",
    "ret_dense_score",
    "ret_dense_hit",
    # structural block (8)
    "struct_same_country",
    "struct_name_len_ratio",
    "struct_addr_len_ratio",
    "struct_addr_missing_xor",
    "struct_addr_missing_both",
    "struct_name_missing_both",
    "struct_name_script_latin_xor",
    "struct_addr_digit_count_ratio",
]

N_FEATURES = len(FEATURE_NAMES)

# A name that looks like a registered domain rather than a business name. Seen in
# the real data ("maurewilliamscolombier.com" used as ``name``).
_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9-]*(\.[a-z0-9-]+){1,3}$")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two sets (0.0 when both are empty)."""
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _as_list(value) -> list:
    """Coerce a possibly-numpy/None value into a plain list (safe for arrays)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    """Character n-gram set (falls back to the whole string if shorter than n)."""
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _tokens(record: Mapping) -> set[str]:
    return set(_as_list(record.get("name_tokens")))


def _safe_ratio(score: float) -> float:
    """rapidfuzz 0-100 score to 0-1."""
    return score / 100.0


def _len_ratio(a: str, b: str) -> float:
    """min/max character-length ratio (0.0 if either is empty)."""
    la, lb = len(a or ""), len(b or "")
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _idf_overlap(a_tokens: set[str], b_tokens: set[str], idf: Mapping[str, float]) -> float:
    """IDF-weighted token overlap: sum(idf on intersection) / sum(idf on union)."""
    union = a_tokens | b_tokens
    if not union:
        return 0.0
    default = idf.get("__default__", 1.0)
    denom = sum(idf.get(t, default) for t in union)
    if denom <= 0:
        return 0.0
    num = sum(idf.get(t, default) for t in (a_tokens & b_tokens))
    return num / denom


def _street_tokens(record: Mapping) -> set[str]:
    """Address tokens excluding the house number, postal code, state and city."""
    tokens = set(_as_list(record.get("addr_tokens")))
    for key in ("addr_house_number", "addr_postal", "addr_state"):
        value = (record.get(key) or "").strip()
        if value:
            tokens.discard(value)
    city = (record.get("addr_city") or "").strip()
    if city:
        for tok in city.split():
            tokens.discard(tok)
    return tokens


def _is_domain(name: str) -> bool:
    """True when the normalized name looks like a domain name."""
    if not name or " " in name:
        return False
    return bool(_DOMAIN_RE.match(name.lower()))


def _digits_only(value: str) -> str:
    """Leading digit run of a house number ("1056c" -> "1056")."""
    match = re.match(r"^(\d+)", value or "")
    return match.group(1) if match else ""


def _script(record: Mapping) -> str:
    """Normalized script tag ("latin" when absent)."""
    return str(record.get("script_type") or "latin").lower()


def _is_missing(record: Mapping, key: str) -> bool:
    """Missingness flag for a field, tolerant of the flag being absent."""
    if key in record and record.get(key) is not None:
        return bool(record.get(key))
    return not bool((record.get("name_core") if key == "is_missing_name" else record.get("addr_norm")) or "")


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------

def compute_name_features(s1: Mapping, cand: Mapping, idf: Mapping[str, float]) -> list[float]:
    """14 name-similarity / name-shape features."""
    a = s1.get("name_core") or ""
    b = cand.get("name_core") or ""
    a_tokens = _tokens(s1) or set(a.split())
    b_tokens = _tokens(cand) or set(b.split())

    suffix_a = s1.get("name_legal_suffix") or ""
    suffix_b = cand.get("name_legal_suffix") or ""
    suffix_match = 1.0 if (suffix_a and suffix_b and suffix_a == suffix_b) else 0.0

    # Phonetic token overlap (only meaningful for Latin/metaphone tokens; 0 when a
    # record has no phonetic view, which is exactly the non-Latin case handled by
    # the transliteration channel).
    phon_a = set(_as_list(s1.get("name_phonetic")))
    phon_b = set(_as_list(cand.get("name_phonetic")))
    phon_jaccard = _jaccard(phon_a, phon_b) if (phon_a or phon_b) else 0.0

    missing_a = _is_missing(s1, "is_missing_name")
    missing_b = _is_missing(cand, "is_missing_name")

    return [
        _safe_ratio(fuzz.ratio(a, b)),
        float(JaroWinkler.similarity(a, b)) if (a or b) else 0.0,
        _safe_ratio(fuzz.token_sort_ratio(a, b)),
        _safe_ratio(fuzz.token_set_ratio(a, b)),
        _jaccard(_char_ngrams(a), _char_ngrams(b)),
        _jaccard(a_tokens, b_tokens),
        _idf_overlap(a_tokens, b_tokens, idf),
        suffix_match,
        _safe_ratio(fuzz.partial_ratio(a, b)),
        _jaccard(_char_ngrams(a, 5), _char_ngrams(b, 5)),
        1.0 if _script(s1) == _script(cand) else 0.0,
        1.0 if missing_a != missing_b else 0.0,
        1.0 if _is_domain(a) != _is_domain(b) else 0.0,
        phon_jaccard,
    ]


def compute_address_features(s1: Mapping, cand: Mapping) -> list[float]:
    """13 address-similarity features."""
    house_a = (s1.get("addr_house_number") or "").strip()
    house_b = (cand.get("addr_house_number") or "").strip()
    house_exact = 1.0 if (house_a and house_b and house_a == house_b) else 0.0
    house_core_a = _digits_only(house_a)
    house_core_b = _digits_only(house_b)
    house_prefix = (
        1.0
        if (house_core_a and house_core_b and house_core_a[:3] == house_core_b[:3])
        else 0.0
    )
    house_conflict = 1.0 if (house_core_a and house_core_b and house_core_a != house_core_b) else 0.0

    postal_a = (s1.get("addr_postal") or "").strip()
    postal_b = (cand.get("addr_postal") or "").strip()
    postal_exact = 1.0 if (postal_a and postal_b and postal_a == postal_b) else 0.0
    postal_prefix3 = (
        1.0 if (len(postal_a) >= 3 and len(postal_b) >= 3 and postal_a[:3] == postal_b[:3]) else 0.0
    )

    city_a = (s1.get("addr_city") or "").strip()
    city_b = (cand.get("addr_city") or "").strip()
    state_a = (s1.get("addr_state") or "").strip()
    state_b = (cand.get("addr_state") or "").strip()

    city_exact = 1.0 if (city_a and city_b and city_a == city_b) else 0.0
    state_match = 1.0 if (state_a and state_b and state_a == state_b) else 0.0

    digits_a = set(_as_list(s1.get("addr_digits")))
    digits_b = set(_as_list(cand.get("addr_digits")))
    digit_conflict = 1.0 if (digits_a and digits_b and not (digits_a & digits_b)) else 0.0

    addr_a = s1.get("addr_norm") or ""
    addr_b = cand.get("addr_norm") or ""

    return [
        house_exact,
        _jaccard(_street_tokens(s1), _street_tokens(cand)),
        postal_exact,
        postal_prefix3,
        city_exact,
        state_match,
        _jaccard(set(_as_list(s1.get("addr_tokens"))), set(_as_list(cand.get("addr_tokens")))),
        _jaccard(_char_ngrams(addr_a), _char_ngrams(addr_b)),
        digit_conflict,
        _safe_ratio(fuzz.partial_ratio(addr_a, addr_b)),
        house_prefix,
        house_conflict,
        _jaccard(set(city_a.split()), set(city_b.split())),
    ]


def compute_retrieval_features(
    s1: Mapping,
    cand: Mapping,
    pair: Mapping,
    signals: Optional[Mapping[str, float]] = None,
) -> list[float]:
    """10 retrieval-signal features.

    ``pair`` supplies ``rank`` (candidate-list position, -1 for easy negatives).
    ``signals`` optionally supplies ``rrf_score``, ``channel_agreement`` and the
    extra-channel scores ``phon_score`` / ``dense_score``; each defaults to 0 when
    unavailable so a model trained without them scores identically at inference.
    """
    rank = int(pair.get("rank", -1))
    retrieved = 1.0 if rank >= 0 else 0.0
    rank_inverse = 1.0 / (rank + 1.0) if rank >= 0 else 0.0

    name_a = (s1.get("name_core") or "").strip()
    name_b = (cand.get("name_core") or "").strip()
    exact_key_hit = 1.0 if (name_a and name_b and name_a == name_b) else 0.0

    sig = signals or {}
    rrf_score = float(sig.get("rrf_score", 0.0) or 0.0)
    agreement = float(sig.get("channel_agreement", 0.0) or 0.0)
    phon_score = float(sig.get("phon_score", 0.0) or 0.0)
    dense_score = max(0.0, float(sig.get("dense_score", 0.0) or 0.0))

    return [
        exact_key_hit,
        rrf_score,
        agreement,
        float(rank),
        retrieved,
        rank_inverse,
        phon_score,
        1.0 if phon_score > 0.0 else 0.0,
        dense_score,
        1.0 if dense_score > 0.0 else 0.0,
    ]


def compute_structural_features(s1: Mapping, cand: Mapping) -> list[float]:
    """8 structural / missingness features."""
    same_country = 1.0 if s1.get("country_norm") == cand.get("country_norm") else 0.0
    missing_a = bool(s1.get("is_missing_addr"))
    missing_b = bool(cand.get("is_missing_addr"))
    name_missing_a = _is_missing(s1, "is_missing_name")
    name_missing_b = _is_missing(cand, "is_missing_name")
    script_a = _script(s1)
    script_b = _script(cand)
    digits_a = len(_as_list(s1.get("addr_digits")))
    digits_b = len(_as_list(cand.get("addr_digits")))
    digit_count_ratio = (
        min(digits_a, digits_b) / max(digits_a, digits_b) if (digits_a or digits_b) else 0.0
    )
    return [
        same_country,
        _len_ratio(s1.get("name_core") or "", cand.get("name_core") or ""),
        _len_ratio(s1.get("addr_norm") or "", cand.get("addr_norm") or ""),
        1.0 if missing_a != missing_b else 0.0,
        1.0 if (missing_a and missing_b) else 0.0,
        1.0 if (name_missing_a and name_missing_b) else 0.0,
        1.0 if ((script_a == "latin") != (script_b == "latin")) else 0.0,
        digit_count_ratio,
    ]


def compute_features(
    s1: Mapping,
    cand: Mapping,
    pair: Mapping,
    idf: Optional[Mapping[str, float]] = None,
    signals: Optional[Mapping[str, float]] = None,
) -> list[float]:
    """Compute the full 45-feature vector in ``FEATURE_NAMES`` order."""
    idf = idf or {}
    return (
        compute_name_features(s1, cand, idf)
        + compute_address_features(s1, cand)
        + compute_retrieval_features(s1, cand, pair, signals)
        + compute_structural_features(s1, cand)
    )


def build_idf(token_document_counts: Mapping[str, int], n_documents: int) -> dict[str, float]:
    """Convert token document frequencies into smoothed IDF weights.

    ``idf = log((N + 1) / (df + 1)) + 1``. A ``__default__`` entry gives the
    weight for unseen tokens.
    """
    n = max(1, n_documents)
    idf = {
        token: math.log((n + 1) / (df + 1)) + 1.0
        for token, df in token_document_counts.items()
    }
    idf["__default__"] = math.log((n + 1) / 1.0) + 1.0
    return idf
