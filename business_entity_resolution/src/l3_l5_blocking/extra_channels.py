"""L4x: fuse extra candidate channels (phonetic G, dense H) with the L4 union.

The lexical L4 artifact already fuses channels A/C/D. The phonetic and dense
retrieval workstreams each write their own candidate artifact with the same
per-reference scope. RRF is the natural fusion: it only ever *adds* candidates
(channels never remove evidence), so the union recall can only go up while the
matcher recovers precision downstream.

Both the L4 artifact and the extra artifacts are ordered by the same Layer 3
reference order, so this module streams them in lockstep by reference batch and
verifies alignment instead of holding the whole country bucket (the US test
bucket alone is ~660k references x ~200 candidates, which does not fit in RAM).

Artifacts produced:
  artifacts/blocking/l4x_<split>_<refs>_country=<c>.parquet
with the same schema as L4 (candidate_entity_ids, fused_scores), so L5/L7/L11
consume it unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Score column name per known extra channel artifact.
EXTRA_SCORE_COLUMNS = {
    "phon": "phon_scores",
    "dense": "dense_scores",
    "addr": "addr_scores",
}

RRF_DEFAULT_K: float = 60.0

FUSED_SCHEMA = pa.schema(
    [
        ("source1_entity_id", pa.string()),
        ("country", pa.string()),
        ("candidate_entity_ids", pa.list_(pa.string())),
        ("fused_scores", pa.list_(pa.float64())),
    ]
)


def rrf_fuse_ref(
    l4_ids: Sequence[str],
    extra_ids: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: float = RRF_DEFAULT_K,
) -> list[tuple[str, float]]:
    """RRF-fuse one reference's L4 ranking with several extra-channel rankings.

    Args:
        l4_ids: L4 candidate ids, best first.
        extra_ids: one ranked id list per extra channel.
        weights: optional per-channel weight (L4 first, then the extras).
        k: RRF smoothing constant.

    Returns:
        ``(candidate_id, fused_score)`` sorted by score desc, ties by id.
    """
    channel_lists = [l4_ids, *extra_ids]
    if weights is None:
        weights = [1.0] * len(channel_lists)
    weights = list(weights) + [1.0] * (len(channel_lists) - len(weights))

    scores: dict[str, float] = {}
    for lst, weight in zip(channel_lists, weights):
        if weight == 0.0:
            continue
        for rank, cand in enumerate(lst, start=1):
            scores[cand] = scores.get(cand, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def iter_extra_batches(
    path: str | Path,
    batch_size: int,
    score_column: str | None = None,
) -> Iterator[tuple[list[str], list[list[str]], list[list[float]]]]:
    """Stream an extra-channel artifact in reference batches.

    Yields ``(reference_ids, candidate_lists, score_lists)`` for each batch.
    """
    parquet_file = pq.ParquetFile(str(path))
    columns = list(parquet_file.schema_arrow.names)
    score_col = score_column
    if score_col is None:
        for candidate in EXTRA_SCORE_COLUMNS.values():
            if candidate in columns:
                score_col = candidate
                break
    use_cols = ["source1_entity_id", "candidate_entity_ids"]
    if score_col and score_col in columns:
        use_cols.append(score_col)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=use_cols):
        data = batch.to_pydict()
        refs = [str(v) for v in data["source1_entity_id"]]
        cands = [
            [] if v is None else [str(c) for c in v]
            for v in data["candidate_entity_ids"]
        ]
        scores = []
        if score_col and score_col in data:
            scores = [
                [] if v is None else [float(x) for x in v]
                for v in data[score_col]
            ]
        else:
            scores = [[] for _ in refs]
        yield refs, cands, scores


def fuse_country_extra(
    l4_path: str | Path,
    extra_paths: Sequence[str | Path],
    out_path: str | Path,
    country: str,
    weights: Sequence[float] | None = None,
    batch_size: int = 10_000,
    rrf_k: float = RRF_DEFAULT_K,
    topn: int | None = None,
    progress=print,
) -> int:
    """Stream-fuse a country's L4 artifact with extra channels; return row count.

    Raises ``ValueError`` if an extra artifact's reference order drifts from the
    L4 artifact, so a mis-scoped artifact fails loudly instead of corrupting the
    union silently.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    l4_file = pq.ParquetFile(str(l4_path))

    extras = [pq.ParquetFile(str(p)) for p in extra_paths]
    extra_iters = [
        pf.iter_batches(batch_size=batch_size, columns=["source1_entity_id", "candidate_entity_ids"])
        for pf in extras
    ]

    rows = 0
    writer = pq.ParquetWriter(str(out), FUSED_SCHEMA)
    try:
        for l4_batch in l4_file.iter_batches(
            batch_size=batch_size,
            columns=["source1_entity_id", "candidate_entity_ids"],
        ):
            data = l4_batch.to_pydict()
            refs = [str(v) for v in data["source1_entity_id"]]
            l4_cands = [
                [] if v is None else [str(c) for c in v] for v in data["candidate_entity_ids"]
            ]

            per_extra_ids: list[list[list[str]]] = [[] for _ in extras]
            for extra_idx, it in enumerate(extra_iters):
                try:
                    eb = next(it)
                except StopIteration:
                    raise ValueError(
                        f"extra artifact {extra_paths[extra_idx]} ended early at row {rows}"
                    ) from None
                ed = eb.to_pydict()
                erefs = [str(v) for v in ed["source1_entity_id"]]
                if erefs != refs:
                    raise ValueError(
                        f"extra artifact {extra_paths[extra_idx]} is misaligned at row "
                        f"{rows}: reference order differs from {Path(l4_path).name}"
                    )
                ec = [
                    [] if v is None else [str(c) for c in v]
                    for v in ed["candidate_entity_ids"]
                ]
                per_extra_ids[extra_idx] = ec

            fused_ids: list[list[str]] = []
            fused_scores: list[list[float]] = []
            for i in range(len(refs)):
                ordered = rrf_fuse_ref(
                    l4_cands[i],
                    [per_extra_ids[e][i] for e in range(len(extras))],
                    weights=weights,
                    k=rrf_k,
                )
                if topn is not None:
                    ordered = ordered[:topn]
                fused_ids.append([c for c, _ in ordered])
                fused_scores.append([float(s) for _, s in ordered])

            writer.write_table(
                pa.table(
                    {
                        "source1_entity_id": refs,
                        "country": [country] * len(refs),
                        "candidate_entity_ids": fused_ids,
                        "fused_scores": fused_scores,
                    },
                    schema=FUSED_SCHEMA,
                )
            )
            rows += len(refs)
            if rows % (batch_size * 10) == 0:
                progress(f"    fused {rows:,} references")

        # Every extra artifact must be exhausted too, otherwise it was longer than L4.
        for extra_idx, it in enumerate(extra_iters):
            if next(it, None) is not None:
                raise ValueError(
                    f"extra artifact {extra_paths[extra_idx]} has more rows than "
                    f"{Path(l4_path).name} — reference scopes differ"
                )
    finally:
        writer.close()
    return rows
