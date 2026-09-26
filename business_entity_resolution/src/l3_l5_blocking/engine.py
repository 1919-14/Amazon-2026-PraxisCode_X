"""L3 blocking engine: build channel indices per country and retrieve candidates.

The engine is intentionally country-serial. For each country bucket it loads the
reference records and the candidate pool, builds the three channels, queries
them and returns candidate ``entity_id`` lists. Processing one country at a time
keeps peak memory proportional to that bucket rather than the full dataset.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (
    L3_A_MAX_POSTING,
    L3_CHANNEL_MAX_POSTING_SCAN,
    L3_CHANNEL_MAX_QUERY_TERMS,
    L3_CHANNEL_TOPK,
    L3_C_MAX_DF_FRAC,
    L3_C_MIN_DF,
    L3_D_MAX_DF_FRAC,
    L3_D_MIN_DF,
    L3_D_NGRAM_RANGE,
    L3_ENABLE_CHAR_CHANNEL,
    L3_QUERY_BATCH,
)
from l3_l5_blocking.channels import ExactKeyChannel, SparseTfidfChannel

CHANNEL_A = "A"
CHANNEL_C = "C"
CHANNEL_D = "D"


@dataclass
class CountryBlockingResult:
    """Candidate lists produced for one country bucket."""

    country: str
    reference_ids: list[str] = field(default_factory=list)
    channels: dict[str, list[list[str]]] = field(default_factory=dict)
    n_candidates: int = 0
    n_references: int = 0


def _as_str_list(series: pd.Series) -> list[str]:
    """Convert a DataFrame column to a null-safe list of strings."""
    return series.fillna("").astype(str).tolist()


def build_channels(
    candidate_df: pd.DataFrame,
    enable_char: bool = True,
) -> dict[str, object]:
    """Build channels A, C and (optionally) D over the candidate pool."""
    names = _as_str_list(candidate_df["name_core"])
    postals = _as_str_list(candidate_df["addr_postal"])
    houses = _as_str_list(candidate_df["addr_house_number"])

    channel_a = ExactKeyChannel(max_posting=L3_A_MAX_POSTING)
    channel_a.build(names, postals, houses)

    channel_c = SparseTfidfChannel(
        name=CHANNEL_C,
        analyzer="word",
        ngram_range=(1, 1),
        min_df=L3_C_MIN_DF,
        max_df=L3_C_MAX_DF_FRAC,
        max_query_terms=L3_CHANNEL_MAX_QUERY_TERMS,
        max_posting_scan=L3_CHANNEL_MAX_POSTING_SCAN,
    )
    channel_c.build(names)

    channels: dict[str, object] = {CHANNEL_A: channel_a, CHANNEL_C: channel_c}

    if enable_char:
        channel_d = SparseTfidfChannel(
            name=CHANNEL_D,
            analyzer="char_wb",
            ngram_range=L3_D_NGRAM_RANGE,
            min_df=L3_D_MIN_DF,
            max_df=L3_D_MAX_DF_FRAC,
            max_query_terms=L3_CHANNEL_MAX_QUERY_TERMS,
            max_posting_scan=L3_CHANNEL_MAX_POSTING_SCAN,
        )
        channel_d.build(names)
        channels[CHANNEL_D] = channel_d

    return channels


def run_country_blocking(
    country: str,
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    topk: int = L3_CHANNEL_TOPK,
    enable_char: bool = True,
) -> CountryBlockingResult:
    """Build channels for one country and retrieve candidates for its references."""
    reference_df = reference_df.reset_index(drop=True)
    candidate_df = candidate_df.reset_index(drop=True)

    reference_ids = _as_str_list(reference_df["entity_id"])
    candidate_ids = _as_str_list(candidate_df["entity_id"])

    result = CountryBlockingResult(
        country=country,
        reference_ids=reference_ids,
        n_candidates=len(candidate_ids),
        n_references=len(reference_ids),
    )

    if not reference_ids or not candidate_ids:
        result.channels = {CHANNEL_A: [], CHANNEL_C: [], CHANNEL_D: []}
        return result

    channels = build_channels(candidate_df, enable_char=enable_char)

    ref_names = _as_str_list(reference_df["name_core"])
    ref_postals = _as_str_list(reference_df["addr_postal"])
    ref_houses = _as_str_list(reference_df["addr_house_number"])

    # Channel A: per-record exact key lookup.
    channel_a = channels[CHANNEL_A]
    assert isinstance(channel_a, ExactKeyChannel)
    a_indices = [
        channel_a.query(name, postal, house, topk)
        for name, postal, house in zip(ref_names, ref_postals, ref_houses)
    ]

    # Channels C and D: batched sparse retrieval.
    channel_c = channels[CHANNEL_C]
    assert isinstance(channel_c, SparseTfidfChannel)
    c_indices = channel_c.query(ref_names, topk, batch_size=L3_QUERY_BATCH)

    channel_d = channels.get(CHANNEL_D)
    if isinstance(channel_d, SparseTfidfChannel):
        d_indices = channel_d.query(ref_names, topk, batch_size=L3_QUERY_BATCH)
    else:
        d_indices = [[] for _ in ref_names]

    def to_ids(indices_per_ref: list[list[int]]) -> list[list[str]]:
        return [[candidate_ids[i] for i in indices] for indices in indices_per_ref]

    result.channels = {
        CHANNEL_A: to_ids(a_indices),
        CHANNEL_C: to_ids(c_indices),
        CHANNEL_D: to_ids(d_indices),
    }

    # Release the heavy indices before the next country bucket is loaded.
    del channels
    gc.collect()
    return result
