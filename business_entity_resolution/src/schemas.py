"""Shared dataclass schema contracts for raw records, normalized records, and candidate metadata."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class RawRecord:
    """Raw unprocessed record read directly from source TSV files."""
    entity_id: str
    business_name: str
    business_address: str
    country: str
    source: str


@dataclass(slots=True)
class NormRecord:
    """Canonical normalized record representation after Layer 2 preprocessing."""
    entity_id: str
    source: str
    country_norm: str
    name_raw: str
    name_norm: str
    name_tokens: list[str] = field(default_factory=list)
    name_phonetic: list[str] = field(default_factory=list)
    name_legal_suffix: Optional[str] = None
    addr_raw: str = ""
    addr_norm: str = ""
    addr_tokens: list[str] = field(default_factory=list)
    addr_house_number: Optional[str] = None
    addr_postal: Optional[str] = None
    addr_state: Optional[str] = None
    addr_city: Optional[str] = None
    addr_digits: list[str] = field(default_factory=list)
    script_type: str = "Latin"
    is_missing_name: bool = False
    is_missing_addr: bool = False
    is_missing_postal: bool = False


@dataclass(slots=True)
class CandidateMeta:
    """Candidate pair metadata and retrieval rank signals produced by Layer 3-5 blocking."""
    s1_id: str
    cand_id: str
    channels_fired: list[str] = field(default_factory=list)
    per_channel_rank: dict[str, int] = field(default_factory=dict)
    rrf_score: float = 0.0
