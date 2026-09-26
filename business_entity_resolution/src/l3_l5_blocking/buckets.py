"""L3a: Country-stratified bucket assignment and normalized record loading.

Reads the snappy-compressed Parquet shards produced by Layer 2
(``artifacts/normalized/<split>_s<idx>/part_*.parquet``) and partitions records
into independent country buckets so that blocking never crosses countries.

Country is handled as an open set: the official dataset only contains US and
India in train (plus France in test), but unknown labels are preserved as their
own bucket rather than dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Ensure src/ is importable when this module is run/imported directly.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import PATH_ARTIFACTS_DIR, L3_COUNTRIES

# Canonical bucket -> set of raw ``country_norm`` values that map to it.
COUNTRY_ALIASES: dict[str, set[str]] = {
    "us": {
        "us",
        "usa",
        "united states",
        "united states of america",
        "u.s.",
        "u.s.a.",
    },
    "india": {"india", "in"},
    "france": {"france", "fr"},
}

# Columns required for blocking. Kept minimal to bound memory while indexing.
CANDIDATE_COLUMNS: list[str] = [
    "entity_id",
    "country_norm",
    "name_core",
    "name_norm",
    "addr_postal",
    "addr_house_number",
]
REFERENCE_COLUMNS: list[str] = list(CANDIDATE_COLUMNS)


def assign_country(country_norm: str) -> str:
    """Map a raw country label to its canonical bucket (or itself if unknown)."""
    value = (country_norm or "").strip().lower()
    for bucket, aliases in COUNTRY_ALIASES.items():
        if value in aliases:
            return bucket
    return value or "other"


def aliases_for(country: str) -> set[str]:
    """Return the raw label set belonging to a canonical bucket."""
    return COUNTRY_ALIASES.get(country, {country})


def iter_source_shards(split: str, source_idx: int) -> list[Path]:
    """Return the sorted parquet shard paths for one split/source, if present."""
    directory = PATH_ARTIFACTS_DIR / "normalized" / f"{split}_s{source_idx}"
    if not directory.exists():
        return []
    return sorted(directory.glob("part_*.parquet"))


def count_parquet_rows(paths: Iterable[Path]) -> int:
    """Count rows across parquet shards using footer metadata (no data loaded)."""
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(str(path)).metadata.num_rows for path in paths)


def dataset_candidate_rows(split: str) -> int:
    """Total Source 2 + Source 3 row count for a split (the blocking denominator)."""
    total = 0
    for source_idx in (2, 3):
        total += count_parquet_rows(iter_source_shards(split, source_idx))
    return total


def load_country_candidates(
    split: str,
    country: str,
    max_candidates: Optional[int] = None,
) -> pd.DataFrame:
    """Load all Source 2 + Source 3 records belonging to one country bucket.

    ``max_candidates`` optionally caps the number of loaded records (useful for
    smoke tests); ``None`` loads everything.
    """
    aliases = aliases_for(country)
    frames: list[pd.DataFrame] = []
    total = 0

    for source_idx in (2, 3):
        for shard in iter_source_shards(split, source_idx):
            df = pd.read_parquet(shard, columns=CANDIDATE_COLUMNS)
            df = df[df["country_norm"].isin(aliases)]
            if df.empty:
                continue
            if max_candidates is not None and total + len(df) > max_candidates:
                df = df.iloc[: max_candidates - total]
            frames.append(df)
            total += len(df)
            if max_candidates is not None and total >= max_candidates:
                break
        if max_candidates is not None and total >= max_candidates:
            break

    if not frames:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _iter_blocks(
    split: str,
    country: str,
    source_indices: tuple[int, ...],
    columns: list[str],
    block_size: int,
    ref_ids: Optional[Iterable[str]] = None,
    max_rows: Optional[int] = None,
):
    """Yield bounded DataFrames of one country bucket, block by block.

    Keeps peak memory at one block instead of one country: the streamed frames
    are concatenated only when a block is complete and the buffer is released
    before the next block is read.
    """
    aliases = aliases_for(country)
    wanted = set(ref_ids) if ref_ids is not None else None
    buffer: list[pd.DataFrame] = []
    buffered = 0
    total = 0

    for source_idx in source_indices:
        for shard in iter_source_shards(split, source_idx):
            df = pd.read_parquet(shard, columns=columns)
            if wanted is not None:
                df = df[df["entity_id"].isin(wanted)]
            df = df[df["country_norm"].isin(aliases)]
            if df.empty:
                continue
            if max_rows is not None:
                remaining = max_rows - total
                if remaining <= 0:
                    return
                if len(df) > remaining:
                    df = df.iloc[:remaining]
            for start in range(0, len(df), block_size):
                chunk = df.iloc[start : start + block_size]
                remaining_space = block_size - buffered
                head = chunk.iloc[:remaining_space]
                buffer.append(head)
                total += len(head)
                buffered += len(head)
                if buffered >= block_size:
                    yield pd.concat(buffer, ignore_index=True)
                    buffer = []
                    buffered = 0
                tail = chunk.iloc[remaining_space:]
                if len(tail):
                    buffer.append(tail)
                    total += len(tail)
                    buffered += len(tail)
            if max_rows is not None and total >= max_rows:
                break
        if max_rows is not None and total >= max_rows:
            break

    if buffer:
        yield pd.concat(buffer, ignore_index=True)


def iter_country_candidate_blocks(
    split: str,
    country: str,
    block_size: int,
    max_candidates: Optional[int] = None,
):
    """Yield the Source 2 + Source 3 candidate pool of one country in blocks."""
    yield from _iter_blocks(
        split,
        country,
        (2, 3),
        CANDIDATE_COLUMNS,
        block_size,
        max_rows=max_candidates,
    )


def iter_country_reference_blocks(
    split: str,
    country: str,
    block_size: int,
    ref_ids: Optional[Iterable[str]] = None,
    max_refs: Optional[int] = None,
):
    """Yield the Source 1 references of one country in blocks."""
    yield from _iter_blocks(
        split,
        country,
        (1,),
        REFERENCE_COLUMNS,
        block_size,
        ref_ids=ref_ids,
        max_rows=max_refs,
    )


def load_references(
    split: str,
    country: str,
    ref_ids: Optional[Iterable[str]] = None,
    max_refs: Optional[int] = None,
) -> pd.DataFrame:
    """Load Source 1 reference records within one country bucket.

    ``ref_ids`` restricts the load to a specific reference set; ``None`` loads
    every Source 1 record in the country bucket.
    """
    aliases = aliases_for(country)
    wanted = set(ref_ids) if ref_ids is not None else None
    frames: list[pd.DataFrame] = []

    for shard in iter_source_shards(split, 1):
        df = pd.read_parquet(shard, columns=REFERENCE_COLUMNS)
        if wanted is not None:
            df = df[df["entity_id"].isin(wanted)]
        df = df[df["country_norm"].isin(aliases)]
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=REFERENCE_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    if max_refs is not None and len(out) > max_refs:
        out = out.iloc[:max_refs]
    return out
