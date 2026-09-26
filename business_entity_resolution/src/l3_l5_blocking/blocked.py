"""L3 memory-bounded blocking: a block-wise, disk-backed country index.

Why this exists
---------------
The first implementation built all three channels for an entire country bucket and
returned every candidate list for every reference in one object. Two things then
scale with the *whole bucket*:

* the TF-IDF matrices: ``char_wb`` 2-4 grams on the train US pool (6.2M names) is
  on the order of 300M nonzeros - several GB with weights and indices;
* the result itself: 1.3M references x 3 channels x top-50 is ~200M individual id
  strings, far more memory than the matrices.

Both blow past the 2 GB budget that a competing box will actually have, so the
engine now works in blocks:

1. **Vocabulary sample** - fit one shared vocabulary + IDF per sparse channel from a
   bounded sample of the pool, so every block scores in the same space and scores
   are comparable across blocks (this is what makes exact top-k merging possible).
2. **Block index build** - stream the candidate pool in blocks; for each block build
   channel A plus the two sparse inverted indices, persist them to a scratch
   directory, and release the block. Peak memory = one block index.
3. **Reference sweep** - stream reference blocks past the persisted index blocks,
   keeping only a compact ``(n_refs, topk)`` int32 key / float32 score state per
   channel. Peak memory = one reference block's state + one candidate block index.
4. **Streaming output** - rows are decoded and written in sub-chunks, and recall is
   accumulated from the same streamed rows, so nothing ever holds all of a country.

Scratch layout (removed after a successful run unless ``--keep-scratch``)::

    artifacts/blocking/_scratch/<split>_<refs>_country=<c>/
        meta.json            block sizes/offsets, vocabulary sizes, flags
        a_00000.pkl          channel A hash maps for block 0
        c_00000.npz          channel C inverted index for block 0 (scipy NPZ)
        d_00000.npz          channel D inverted index for block 0
        ids_00000.npy        candidate entity ids of block 0 (UTF-32 fixed width)
"""

from __future__ import annotations

import gc
import json
import pickle
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse as sp

from config import (
    L3_A_MAX_POSTING,
    L3_BLOCK_SIZE,
    L3_C_MAX_DF_FRAC,
    L3_C_MIN_DF,
    L3_CHANNEL_MAX_POSTING_SCAN,
    L3_CHANNEL_MAX_QUERY_TERMS,
    L3_D_MAX_DF_FRAC,
    L3_D_MIN_DF,
    L3_D_NGRAM_RANGE,
    L3_VOCAB_SAMPLE,
)
from l3_l5_blocking.buckets import (
    iter_country_candidate_blocks,
    iter_country_reference_blocks,
)
from l3_l5_blocking.channels import ExactKeyChannel, SparseTfidfChannel

CHANNEL_A = "A"
CHANNEL_C = "C"
CHANNEL_D = "D"
ALL_CHANNELS = (CHANNEL_A, CHANNEL_C, CHANNEL_D)

META_FILE = "meta.json"


def _as_str_list(series: pd.Series) -> list[str]:
    """Convert a DataFrame column to a null-safe list of strings."""
    return series.fillna("").astype(str).tolist()


def make_word_channel() -> SparseTfidfChannel:
    """Channel C: word-level TF-IDF with the configured pruning."""
    return SparseTfidfChannel(
        name=CHANNEL_C,
        analyzer="word",
        ngram_range=(1, 1),
        min_df=L3_C_MIN_DF,
        max_df=L3_C_MAX_DF_FRAC,
        max_query_terms=L3_CHANNEL_MAX_QUERY_TERMS,
        max_posting_scan=L3_CHANNEL_MAX_POSTING_SCAN,
    )


def make_char_channel() -> SparseTfidfChannel:
    """Channel D: character 2-4 gram TF-IDF with the configured pruning."""
    return SparseTfidfChannel(
        name=CHANNEL_D,
        analyzer="char_wb",
        ngram_range=L3_D_NGRAM_RANGE,
        min_df=L3_D_MIN_DF,
        max_df=L3_D_MAX_DF_FRAC,
        max_query_terms=L3_CHANNEL_MAX_QUERY_TERMS,
        max_posting_scan=L3_CHANNEL_MAX_POSTING_SCAN,
    )


@dataclass
class CountryIndexMeta:
    """Location and shape of a persisted country index."""

    country: str
    split: str
    refs: str
    scratch_dir: str
    block_size: int
    enable_char: bool
    n_candidates: int
    block_sizes: list[int] = field(default_factory=list)
    block_offsets: list[int] = field(default_factory=list)
    vocab_size_c: int = 0
    vocab_size_d: int = 0
    sample_size: int = 0

    @property
    def n_blocks(self) -> int:
        """Number of candidate blocks persisted."""
        return len(self.block_sizes)

    def path(self, name: str) -> Path:
        """Resolve a file inside the scratch directory."""
        return Path(self.scratch_dir) / name

    def save(self) -> Path:
        """Persist the metadata (called once the build finishes)."""
        path = self.path(META_FILE)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, scratch_dir: str | Path) -> "CountryIndexMeta":
        """Load metadata from a scratch directory."""
        payload = json.loads((Path(scratch_dir) / META_FILE).read_text(encoding="utf-8"))
        return cls(**payload)

    def cleanup(self) -> None:
        """Delete the scratch directory."""
        shutil.rmtree(self.scratch_dir, ignore_errors=True)


def build_country_index(
    split: str,
    country: str,
    *,
    refs: str,
    scratch_dir: str | Path,
    block_size: int = L3_BLOCK_SIZE,
    vocab_sample: int = L3_VOCAB_SAMPLE,
    enable_char: bool = True,
    max_candidates: Optional[int] = None,
    progress=print,
) -> CountryIndexMeta:
    """Build and persist a block-wise index for one country bucket.

    Args:
        split: dataset split the normalized shards come from.
        country: canonical country bucket.
        refs: run scope, recorded in the metadata for reference.
        scratch_dir: destination for the persisted blocks.
        block_size: candidates per block (bounds peak memory).
        vocab_sample: documents sampled to fit the shared vocabulary + IDF.
        enable_char: build channel D as well.
        max_candidates / progress: development caps and progress logging.

    Returns:
        The metadata needed to query the persisted index.
    """
    scratch = Path(scratch_dir)
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)

    meta = CountryIndexMeta(
        country=country,
        split=split,
        refs=refs,
        scratch_dir=str(scratch),
        block_size=int(block_size),
        enable_char=bool(enable_char),
        n_candidates=0,
    )

    # ------------------------------------------------------------------
    # Pass 1: shared vocabulary + IDF from a bounded sample
    # ------------------------------------------------------------------
    sample: list[str] = []
    for block in iter_country_candidate_blocks(split, country, block_size, max_candidates):
        if len(sample) >= vocab_sample:
            break
        names = _as_str_list(block["name_core"])
        sample.extend(names[: vocab_sample - len(sample)])

    meta.sample_size = len(sample)
    if not sample:
        progress("  ⚠️  no candidates for this bucket — empty index")
        meta.save()
        return meta

    channel_c = make_word_channel()
    channel_d = make_char_channel() if enable_char else None
    meta.vocab_size_c = channel_c.fit_vocab(sample)
    _save_vocab(meta, CHANNEL_C, channel_c)
    if channel_d is not None:
        meta.vocab_size_d = channel_d.fit_vocab(sample)
        _save_vocab(meta, CHANNEL_D, channel_d)
    progress(
        f"  vocabulary sample: {len(sample):,} docs | |C|={meta.vocab_size_c:,}"
        + (f" | |D|={meta.vocab_size_d:,}" if channel_d is not None else "")
    )
    del sample
    gc.collect()

    # ------------------------------------------------------------------
    # Pass 2: build + persist one block at a time
    # ------------------------------------------------------------------
    offset = 0
    for block_idx, block in enumerate(
        iter_country_candidate_blocks(split, country, block_size, max_candidates)
    ):
        names = _as_str_list(block["name_core"])
        postals = _as_str_list(block["addr_postal"])
        houses = _as_str_list(block["addr_house_number"])
        ids = _as_str_list(block["entity_id"])

        channel_a = ExactKeyChannel(max_posting=L3_A_MAX_POSTING)
        channel_a.build(names, postals, houses)
        with meta.path(f"a_{block_idx:05d}.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "by_name": channel_a.by_name,
                    "by_name_postal": channel_a.by_name_postal,
                    "by_name_house": channel_a.by_name_house,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        del channel_a

        channel_c.build_with_vocab(names)
        if channel_c.inverted is not None:
            sp.save_npz(meta.path(f"c_{block_idx:05d}.npz"), channel_c.inverted)
        if channel_d is not None:
            channel_d.build_with_vocab(names)
            if channel_d.inverted is not None:
                sp.save_npz(meta.path(f"d_{block_idx:05d}.npz"), channel_d.inverted)

        np.save(meta.path(f"ids_{block_idx:05d}.npy"), np.asarray(ids, dtype=np.str_))

        meta.block_sizes.append(len(ids))
        meta.block_offsets.append(offset)
        offset += len(ids)
        meta.n_candidates += len(ids)
        progress(f"    block {block_idx + 1:3d} | {len(ids):>8,} candidates | total {offset:>9,}")

        del names, postals, houses, ids, block
        gc.collect()

    meta.save()
    return meta


# ----------------------------------------------------------------------
# Loading persisted blocks
# ----------------------------------------------------------------------

def _save_vocab(meta: CountryIndexMeta, channel: str, index: SparseTfidfChannel) -> None:
    """Persist the shared vocabulary, IDF and term df for one channel."""
    if index.vocabulary_ is None or index.idf_ is None:
        return
    term_df = index.global_term_df
    if term_df is None:
        term_df = np.zeros(len(index.vocabulary_), dtype=np.int64)
    np.savez(
        meta.path(f"vocab_{channel.lower()}.npz"),
        vocabulary=np.asarray(index.vocabulary_, dtype=object),
        idf=index.idf_,
        term_df=np.asarray(term_df, dtype=np.int64),
    )


def load_block_exact_index(meta: CountryIndexMeta, block_idx: int) -> ExactKeyChannel:
    """Load channel A for one block."""
    with meta.path(f"a_{block_idx:05d}.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    channel = ExactKeyChannel(max_posting=L3_A_MAX_POSTING)
    channel.by_name = payload["by_name"]
    channel.by_name_postal = payload["by_name_postal"]
    channel.by_name_house = payload["by_name_house"]
    return channel


def load_block_sparse_index(
    meta: CountryIndexMeta,
    block_idx: int,
    channel: str,
) -> Optional[SparseTfidfChannel]:
    """Rebuild a sparse channel around one block's persisted inverted index."""
    path = meta.path(f"{channel.lower()}_{block_idx:05d}.npz")
    if not path.exists():
        return None
    builder = make_word_channel if channel == CHANNEL_C else make_char_channel
    index = builder()
    # The shared vocabulary/IDF live with the *build-time* channel objects; reload
    # them from the vocabulary file saved alongside the metadata.
    vocab_path = meta.path(f"vocab_{channel.lower()}.npz")
    if not vocab_path.exists():
        return None
    payload = np.load(vocab_path, allow_pickle=True)
    index.vocabulary_ = [str(term) for term in payload["vocabulary"]]
    index.idf_ = payload["idf"].astype(np.float32)
    index.global_term_df = payload["term_df"].astype(np.int64)
    index._vocab_index = {term: i for i, term in enumerate(index.vocabulary_)}
    from sklearn.feature_extraction.text import CountVectorizer

    index._counter = CountVectorizer(
        analyzer=index.analyzer,
        ngram_range=index.ngram_range,
        vocabulary=index._vocab_index,
        dtype=np.float32,
    )
    index.inverted = sp.load_npz(path).tocsc()
    index.term_df = index.global_term_df
    return index


def load_block_ids(meta: CountryIndexMeta, block_idx: int):
    """Memory-mapped candidate ids of one block (order == block-local doc index)."""
    return np.load(meta.path(f"ids_{block_idx:05d}.npy"), mmap_mode="r")


# ----------------------------------------------------------------------
# Merging per-block top-k
# ----------------------------------------------------------------------

def _merge_topk(
    cur_ref: np.ndarray,
    cur_key: np.ndarray,
    cur_score: np.ndarray,
    new_ref: np.ndarray,
    new_key: np.ndarray,
    new_score: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge a new block's hits into the running per-reference top-k state.

    Ordering is ``(reference, score desc, global key asc)``: score first because
    that is the retrieval signal, then the key so that equal scores resolve
    deterministically (an earlier block wins, and within a block the lower doc
    index wins).
    """
    if len(new_ref) == 0:
        return cur_ref, cur_key, cur_score

    ref = np.concatenate([cur_ref, new_ref]) if len(cur_ref) else new_ref
    key = np.concatenate([cur_key, new_key]) if len(cur_key) else new_key
    score = np.concatenate([cur_score, new_score]) if len(cur_score) else new_score

    order = np.lexsort((key, -score, ref))
    ref_sorted = ref[order]
    key_sorted = key[order]
    score_sorted = score[order]

    unique_refs, first_index = np.unique(ref_sorted, return_index=True)
    rank = np.arange(len(ref_sorted), dtype=np.int64)
    rank -= first_index[np.searchsorted(unique_refs, ref_sorted)]
    keep = rank < k
    return ref_sorted[keep], key_sorted[keep], score_sorted[keep]


@dataclass
class TopKResult:
    """Merged per-reference candidate keys/scores for one reference block."""

    reference_ids: list[str]
    keys: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]
    channels: tuple[str, ...] = ALL_CHANNELS

    def __len__(self) -> int:
        return len(self.reference_ids)


def _group_bounds(flat_ref: np.ndarray, n_refs: int) -> tuple[np.ndarray, np.ndarray]:
    """Row and in-group column indices for a flat, reference-grouped array."""
    counts = np.bincount(flat_ref, minlength=n_refs)
    ends = np.cumsum(counts)
    starts = ends - counts
    rows = np.repeat(np.arange(n_refs), counts)
    columns = np.arange(len(flat_ref)) - np.repeat(starts, counts)
    return rows, columns


def query_reference_block(
    meta: CountryIndexMeta,
    reference_df: pd.DataFrame,
    *,
    topk: int,
    progress=None,
) -> TopKResult:
    """Query every persisted index block for one block of references.

    Returns compact key/score matrices (``-1`` marks an empty slot) so the caller
    can stream them to disk without materialising id strings for the whole bucket.
    """
    reference_df = reference_df.reset_index(drop=True)
    reference_ids = _as_str_list(reference_df["entity_id"])
    names = _as_str_list(reference_df["name_core"])
    postals = _as_str_list(reference_df["addr_postal"])
    houses = _as_str_list(reference_df["addr_house_number"])
    n_refs = len(reference_ids)

    empty = (
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float32),
    )
    state: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        channel: empty for channel in ALL_CHANNELS
    }

    for block_idx in range(meta.n_blocks):
        block_offset = meta.block_offsets[block_idx]

        # ---------------- Channel A (exact key, tiered) ----------------
        exact = load_block_exact_index(meta, block_idx)
        a_ref: list[int] = []
        a_key: list[int] = []
        a_score: list[float] = []
        for row, (name, postal, house) in enumerate(zip(names, postals, houses)):
            hits, tiers = exact.query_tiered(name, postal, house, topk)
            for local_doc, tier in zip(hits, tiers):
                a_ref.append(row)
                a_key.append(block_offset + local_doc)
                a_score.append(float(tier))
        del exact
        state[CHANNEL_A] = _merge_topk(
            *state[CHANNEL_A],
            np.asarray(a_ref, dtype=np.int32),
            np.asarray(a_key, dtype=np.int32),
            np.asarray(a_score, dtype=np.float32),
            topk,
        )

        # ---------------- Channels C and D (sparse TF-IDF) ----------------
        for channel in (CHANNEL_C, CHANNEL_D):
            index = load_block_sparse_index(meta, block_idx, channel)
            if index is None:
                continue
            results = index.query_with_scores(names, topk)
            refs: list[int] = []
            keys: list[int] = []
            scores: list[float] = []
            for row, (hits, hit_scores) in enumerate(results):
                for local_doc, score in zip(hits, hit_scores):
                    refs.append(row)
                    keys.append(block_offset + local_doc)
                    scores.append(score)
            del index, results
            state[channel] = _merge_topk(
                *state[channel],
                np.asarray(refs, dtype=np.int32),
                np.asarray(keys, dtype=np.int32),
                np.asarray(scores, dtype=np.float32),
                topk,
            )
            gc.collect()

        if progress is not None and (block_idx + 1) % 4 == 0:
            progress(f"      merged {block_idx + 1}/{meta.n_blocks} index blocks")
        gc.collect()

    keys_out: dict[str, np.ndarray] = {}
    scores_out: dict[str, np.ndarray] = {}
    for channel in ALL_CHANNELS:
        flat_ref, flat_key, flat_score = state[channel]
        if channel == CHANNEL_A and len(flat_ref):
            flat_ref, flat_key, flat_score = _keep_best_tier(flat_ref, flat_key, flat_score)
        keys_out[channel], scores_out[channel] = _to_matrices(
            flat_ref, flat_key, flat_score, n_refs, topk
        )

    return TopKResult(reference_ids=reference_ids, keys=keys_out, scores=scores_out)


def _keep_best_tier(
    flat_ref: np.ndarray,
    flat_key: np.ndarray,
    flat_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only the strongest exact-key tier per reference.

    Channel A queries ``name|postal``, then ``name|house``, then ``name`` and stops
    at the first key that matches, so a reference whose postal key fired never
    contributes bare-name candidates. The block-wise engine sees the tiers in
    separate blocks, so it recovers the same semantics afterwards: if any block
    produced a stronger key for a reference, its weaker hits are dropped. Without
    this the candidate set would silently grow (hurting the compactness criterion)
    and no longer match the classic engine.
    """
    best = np.zeros(int(flat_ref.max()) + 1, dtype=np.float32)
    np.maximum.at(best, flat_ref, flat_score)
    keep = flat_score >= best[flat_ref]
    return flat_ref[keep], flat_key[keep], flat_score[keep]


def _to_matrices(
    flat_ref: np.ndarray,
    flat_key: np.ndarray,
    flat_score: np.ndarray,
    n_refs: int,
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn flat grouped arrays back into ``(n_refs, topk)`` matrices."""
    keys = np.full((n_refs, topk), -1, dtype=np.int32)
    scores = np.zeros((n_refs, topk), dtype=np.float32)
    if len(flat_ref):
        rows, columns = _group_bounds(flat_ref, n_refs)
        keys[rows, columns] = flat_key
        scores[rows, columns] = flat_score
    return keys, scores


def decode_ids(meta: CountryIndexMeta, keys: np.ndarray) -> list[list[str]]:
    """Decode a ``(n_refs, topk)`` key matrix into per-reference id lists.

    Keys are grouped by source block so only the needed id arrays are touched;
    empty slots (key ``-1``) are dropped, preserving best-first order.
    """
    out = np.empty(keys.shape, dtype=object)
    out[:] = ""
    for block_idx in range(meta.n_blocks):
        block_offset = meta.block_offsets[block_idx]
        block_size = meta.block_sizes[block_idx]
        mask = (keys >= block_offset) & (keys < block_offset + block_size)
        if not mask.any():
            continue
        local = (keys[mask] - block_offset).astype(np.int64)
        ids = load_block_ids(meta, block_idx)
        out[mask] = ids[local]
    return [[value for value in row.tolist() if value] for row in out]
