"""L4: Reciprocal Rank Fusion (RRF) over the L3 blocking channels.

RRF merges several independently ranked candidate lists into a single ranking
without needing comparable score scales. A candidate's fused score is

    RRF(c) = sum_over_channels  weight_channel / (k + rank_channel(c))

where ``rank`` is 1-based and ``k`` is a smoothing constant (60 in the original
paper). Candidates absent from a channel simply contribute nothing, so channels
only ever *add* evidence: fusion never discards a candidate, it only reorders
the union. Precision loss is recovered later by L5 truncation and the matcher.

Reference: Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion
Outperforms Condorcet and Individual Rank Learning Methods".
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

# Standard smoothing constant.
RRF_DEFAULT_K: float = 60.0


def rrf_scores(
    channel_lists: Mapping[str, Sequence[str]],
    k: float = RRF_DEFAULT_K,
    weights: Optional[Mapping[str, float]] = None,
) -> list[tuple[str, float]]:
    """Fuse one reference entity's per-channel candidate lists.

    Args:
        channel_lists: channel name -> ranked candidate ids (best first).
        k: RRF smoothing constant.
        weights: optional per-channel weight (default 1.0 each).

    Returns:
        ``(candidate_id, fused_score)`` pairs sorted by score desc, ties broken
        by candidate id ascending for deterministic output.
    """
    scores: dict[str, float] = {}
    for channel, candidates in channel_lists.items():
        weight = 1.0 if weights is None else float(weights.get(channel, 1.0))
        if weight == 0.0:
            continue
        for rank, candidate in enumerate(candidates, start=1):
            scores[candidate] = scores.get(candidate, 0.0) + weight / (k + rank)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def fuse(
    channel_candidates: Mapping[str, Sequence[Sequence[str]]],
    k: float = RRF_DEFAULT_K,
    weights: Optional[Mapping[str, float]] = None,
) -> list[list[tuple[str, float]]]:
    """Fuse candidate lists for many reference entities at once.

    Args:
        channel_candidates: channel name -> list (indexed by reference) of
            ranked candidate-id lists.
        k: RRF smoothing constant.
        weights: optional per-channel weights.

    Returns:
        For each reference, the fused ``(candidate_id, score)`` list.
    """
    n_references = max((len(v) for v in channel_candidates.values()), default=0)
    fused: list[list[tuple[str, float]]] = []
    for i in range(n_references):
        per_channel = {
            channel: (candidates[i] if i < len(candidates) else [])
            for channel, candidates in channel_candidates.items()
        }
        fused.append(rrf_scores(per_channel, k=k, weights=weights))
    return fused


def ranked_ids(
    fused: Sequence[Sequence[tuple[str, float]]],
) -> list[list[str]]:
    """Drop the scores and keep only the fused candidate-id ordering."""
    return [[candidate for candidate, _ in row] for row in fused]
