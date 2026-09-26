"""L11: streaming inference over the L5 candidate set.

Reads the candidate pairs the blocking stage produced, computes the same 27 L7
features for every pair, scores them with the exported L8 booster, applies the
optional L9 calibrator, and hands the per-reference candidate scores to the L10
decision engine.

Memory strategy
---------------
The test candidate set is ~10M pairs and the two records behind each pair are
only needed while that pair is scored, so the layer streams over **reference
batches**: each batch loads just its own Source 1 records and candidates from the
normalized shards, scores them, and keeps only the surviving ``(candidate,
score)`` tuples. Only candidates at or above ``tau_match`` are retained, which is
lossless: the L10 margin rule can never keep a candidate below ``tau_match``
(``effective_threshold = max(tau_match, top - margin) >= tau_match``).

The IDF table used by ``name_idf_overlap`` is built once in a dedicated streaming
pass over the candidate records so every batch sees the same weights.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd

from config import L10_OPEN_SET_COUNTRIES, L10_SEEN_COUNTRIES
from l3_l5_blocking.buckets import assign_country, iter_source_shards
from l6_l8_matching.features import FEATURE_NAMES, build_idf, compute_features
from l10_decision.decision import apply_decision_rule, write_id_list_tsv
from l11_inference.scorer import calibrate_scores, score_matrix
from main_l6 import load_reference_countries

# Columns read from the normalized parquet shards for feature computation.
RECORD_COLUMNS: list[str] = [
    "entity_id",
    "country_norm",
    "name_core",
    "name_norm",
    "name_tokens",
    "name_legal_suffix",
    "addr_norm",
    "addr_tokens",
    "addr_house_number",
    "addr_postal",
    "addr_state",
    "addr_city",
    "addr_digits",
    "is_missing_addr",
]

LIST_FIELDS = ("name_tokens", "addr_tokens", "addr_digits")

MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def read_candidate_pairs(path: str | Path) -> tuple[list[str], dict[str, list[str]]]:
    """Load a ``candidate_pairs.tsv`` into an ordered reference list and a map.

    Rows with an empty candidate column are preserved so the reference still
    receives an (empty) prediction row.
    """
    pair_path = Path(path)
    if not pair_path.exists():
        raise FileNotFoundError(f"candidate pairs not found: {pair_path}")

    refs: list[str] = []
    cand_map: dict[str, list[str]] = {}
    with pair_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is not None and header[0].strip().lower() != "source1_entity_id":
            raise ValueError(f"unexpected header in {pair_path}: {header}")
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            raw = row[1] if len(row) > 1 else ""
            cand_map[s1_id] = [c for c in raw.split(",") if c]
            refs.append(s1_id)
    return refs, cand_map


def all_reference_ids(split: str) -> list[str]:
    """Every Source 1 entity id present in the split's normalized shards."""
    ids: set[str] = set()
    for shard in iter_source_shards(split, 1):
        df = pd.read_parquet(shard, columns=["entity_id"])
        ids.update(df["entity_id"].tolist())
    return sorted(ids)


def load_records(
    split: str,
    source_indices: Iterable[int],
    ids: set[str],
    max_shards: Optional[int] = None,
) -> dict[str, dict]:
    """Load normalized records for ``ids`` from the requested sources."""
    wanted = set(ids)
    lookup: dict[str, dict] = {}
    for source_idx in source_indices:
        shards = iter_source_shards(split, source_idx)
        if max_shards is not None:
            shards = shards[:max_shards]
        for shard in shards:
            df = pd.read_parquet(shard, columns=RECORD_COLUMNS)
            df = df[df["entity_id"].isin(wanted)]
            if df.empty:
                continue
            for record in df.to_dict(orient="records"):
                for field in LIST_FIELDS:
                    raw = record.get(field)
                    record[field] = list(raw) if raw is not None else []
                lookup[record["entity_id"]] = record
    return lookup


def build_inference_idf(
    split: str,
    sources: tuple[int, ...] = (2, 3),
    max_shards: Optional[int] = None,
) -> dict[str, float]:
    """Build the IDF table from candidate name-token document frequencies."""
    token_df: Counter[str] = Counter()
    n_docs = 0
    for source_idx in sources:
        shards = iter_source_shards(split, source_idx)
        if max_shards is not None:
            shards = shards[:max_shards]
        for shard in shards:
            df = pd.read_parquet(shard, columns=["name_tokens"])
            for tokens in df["name_tokens"].tolist():
                n_docs += 1
                if tokens is not None and len(tokens):
                    token_df.update(set(tokens))
    return build_idf(token_df, n_docs)


# ---------------------------------------------------------------------------
# Streaming inference
# ---------------------------------------------------------------------------

def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_inference(
    split: str,
    candidate_pairs_path: str | Path,
    booster,
    idf: Optional[dict[str, float]] = None,
    calibrator: Optional[dict] = None,
    tau_match: float = 0.5,
    tau_s: float = 0.3,
    margin: float = 0.05,
    seen_countries: Iterable[str] = L10_SEEN_COUNTRIES,
    open_set_countries: Iterable[str] = L10_OPEN_SET_COUNTRIES,
    open_set_boost: float = 0.10,
    veto_min_confidence: float = 0.50,
    all_references: bool = True,
    ref_batch_size: int = 50_000,
    max_references: Optional[int] = None,
    max_shards: Optional[int] = None,
    idf_max_shards: Optional[int] = None,
    progress=print,
) -> tuple[dict[str, list[str]], dict]:
    """Score the candidate set and return ``(predictions, stats)``.

    Args:
        split: dataset split the normalized shards come from (``test`` or ``train``).
        candidate_pairs_path: the L5 ``candidate_pairs.tsv``.
        booster: a loaded LightGBM booster.
        idf: precomputed IDF table; built from the candidate records when omitted.
        calibrator: optional L9 calibrator payload.
        tau_match, tau_s, margin: decision thresholds forwarded to L10.
        seen_countries / open_set_countries / open_set_boost / veto_min_confidence:
            open-set veto controls.
        all_references: when true (and no ``max_references`` cap) every Source 1
            entity in the split receives a row, including those with no candidates.
        ref_batch_size: references scored per streaming batch.
        max_references: dev cap on references scored (disables ``all_references``).
        max_shards: dev cap on shards read per source during record loading.
        idf_max_shards: dev cap on shards read while building the IDF.

    Returns:
        ``(predictions, stats)`` where predictions maps each reference to matched
        candidate ids.
    """
    refs_file, cand_map = read_candidate_pairs(candidate_pairs_path)

    if max_references is not None:
        refs_file = refs_file[:max_references]
        cand_map = {ref: cand_map[ref] for ref in refs_file}

    if all_references and max_references is None:
        refs = sorted(set(refs_file) | set(all_reference_ids(split)))
    else:
        refs = list(refs_file)

    scorable = [ref for ref in refs_file if cand_map.get(ref)]
    progress(f"  references: {len(refs):,} | scorable: {len(scorable):,}")

    if idf is None:
        progress("  building IDF from candidate records ...")
        idf = build_inference_idf(split, max_shards=idf_max_shards)

    # Accumulate only candidates at/above tau_match: the L10 margin rule can never
    # keep a candidate below it, so this is lossless and bounds memory.
    kept: dict[str, list[tuple[str, float]]] = {}
    n_scored = 0
    n_missing_record = 0

    for batch_idx, batch in enumerate(_chunks(scorable, ref_batch_size), start=1):
        batch_set = set(batch)
        cand_ids: set[str] = set()
        for ref in batch:
            cand_ids.update(cand_map[ref])

        s1_lookup = load_records(split, (1,), batch_set, max_shards=max_shards)
        cand_lookup = load_records(split, (2, 3), cand_ids, max_shards=max_shards)

        rows: list[list[float]] = []
        row_refs: list[str] = []
        row_cands: list[str] = []
        row_countries: list[str] = []

        for ref in batch:
            s1 = s1_lookup.get(ref)
            if s1 is None:
                n_missing_record += len(cand_map[ref])
                continue
            country = assign_country(s1.get("country_norm", "other"))
            for rank, cand_id in enumerate(cand_map[ref]):
                cand = cand_lookup.get(cand_id)
                if cand is None:
                    n_missing_record += 1
                    continue
                pair = {"rank": rank}
                features = compute_features(s1, cand, pair, idf, None)
                rows.append(features)
                row_refs.append(ref)
                row_cands.append(cand_id)
                row_countries.append(country)

        if rows:
            probs = score_matrix(booster, np.asarray(rows, dtype=np.float32))
            probs = calibrate_scores(probs, calibrator, row_countries if calibrator else None)
            for ref, cand_id, prob in zip(row_refs, row_cands, probs):
                if prob >= tau_match:
                    kept.setdefault(ref, []).append((cand_id, float(prob)))
            n_scored += len(rows)

        if batch_idx % 5 == 0 or batch_idx == 1:
            progress(f"    batch {batch_idx} | scored {n_scored:,} pairs | refs {len(kept):,}")

    # Country per reference drives the open-set veto.
    ref_country = load_reference_countries(split, set(refs))
    seen = set(seen_countries)
    open_set_configured = set(open_set_countries)
    open_set_ids = {
        ref
        for ref in refs
        if ref_country.get(ref, "other") in open_set_configured
        or ref_country.get(ref, "other") not in seen
    }

    predictions = apply_decision_rule(
        kept,
        refs,
        tau_match=tau_match,
        tau_s=tau_s,
        margin=margin,
        open_set_ids=open_set_ids,
        open_set_boost=open_set_boost,
        veto_min_confidence=veto_min_confidence,
    )

    stats = {
        "references": len(refs),
        "scorable_references": len(scorable),
        "pairs_scored": n_scored,
        "pairs_missing_record": n_missing_record,
        "open_set_references": len(open_set_ids),
        "non_empty": sum(1 for ids in predictions.values() if ids),
        "matched_pairs": sum(len(ids) for ids in predictions.values()),
    }
    return predictions, stats


def write_predictions(
    path: str | Path,
    predictions: dict[str, list[str]],
    reference_ids: list[str],
) -> int:
    """Write ``matching_results.tsv`` (one row per reference)."""
    return write_id_list_tsv(Path(path), predictions, reference_ids, MATCHING_HEADER)


def write_candidates(
    path: str | Path,
    candidate_map: dict[str, list[str]],
    reference_ids: list[str],
) -> int:
    """Write ``candidate_pairs.tsv`` (one row per reference)."""
    return write_id_list_tsv(Path(path), candidate_map, reference_ids, CANDIDATE_HEADER)


def feature_names() -> list[str]:
    """Exposed so callers can assert model/feature ordering alignment."""
    return list(FEATURE_NAMES)
