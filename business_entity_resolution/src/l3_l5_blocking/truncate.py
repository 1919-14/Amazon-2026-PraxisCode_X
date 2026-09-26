"""L5: coarse scoring + adaptive candidate truncation.

Layer 4 produces a fused, recall-preserving ranking that is deliberately
generous (up to ~150 candidates per reference). Layer 5 turns that into a
compact candidate set of roughly ``K ~= 5.5 - 7.0`` candidates per Source 1
entity for Amazon's candidate-compactness audit, while keeping recall as high
as the budget allows.

Two stages:
  1. **Coarse scoring** re-ranks the fused list with cheap signals that need no
     pairwise string work: the normalised RRF score, how many channels proposed
     the candidate (agreement), and whether an exact blocking key fired.
  2. **Adaptive truncation** keeps candidates whose coarse score retains at
     least ``ratio`` of the reference's top score, clamps the result to
     ``[k_min, k_max]``, and is tuned so the average budget lands in the target
     compactness band.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from config import (
    L5_WEIGHT_AGREE,
    L5_WEIGHT_EXACT,
    L5_WEIGHT_RRF,
)

CHANNEL_A = "A"


def coarse_score(
    fused: Sequence[tuple[str, float]],
    channel_lists: Optional[Mapping[str, Sequence[str]]] = None,
    w_rrf: float = L5_WEIGHT_RRF,
    w_agree: float = L5_WEIGHT_AGREE,
    w_exact: float = L5_WEIGHT_EXACT,
) -> list[tuple[str, float]]:
    """Re-rank one reference's fused candidates with cheap coarse signals.

    Args:
        fused: L4 output, ``(candidate_id, rrf_score)`` best-first.
        channel_lists: optional ``channel -> ranked candidate ids`` for the same
            reference (from the L3 artifact); enables agreement/exact signals.

    Returns:
        ``(candidate_id, coarse_score)`` sorted by score desc, ties broken by
        the original fused position (deterministic).
    """
    if not fused:
        return []

    max_rrf = max(score for _, score in fused)
    if max_rrf <= 0.0:
        max_rrf = 1.0

    agreement: dict[str, int] = {}
    exact: set[str] = set()
    n_channels = 1
    if channel_lists:
        n_channels = max(1, len(channel_lists))
        for channel, candidates in channel_lists.items():
            for candidate in candidates:
                agreement[candidate] = agreement.get(candidate, 0) + 1
        exact = set(channel_lists.get(CHANNEL_A, ()))

    scored: list[tuple[str, float, int]] = []
    for position, (candidate, rrf) in enumerate(fused):
        score = w_rrf * (rrf / max_rrf)
        if channel_lists:
            score += w_agree * (agreement.get(candidate, 0) / n_channels)
            if candidate in exact:
                score += w_exact
        scored.append((candidate, score, position))

    scored.sort(key=lambda item: (-item[1], item[2]))
    return [(candidate, score) for candidate, score, _ in scored]


def adaptive_truncate(
    scored: Sequence[tuple[str, float]],
    ratio: float,
    k_min: int = 1,
    k_max: int = 12,
) -> list[str]:
    """Keep the prefix of a coarse-scored list within ``ratio`` of its top score.

    Args:
        scored: ``(candidate_id, coarse_score)`` sorted best-first.
        ratio: minimum fraction of the top score required to survive.
        k_min: minimum kept when any candidate exists (unless all scores are 0).
        k_max: hard cap on kept candidates.

    Returns:
        Kept candidate ids in coarse order.
    """
    if not scored:
        return []

    top = scored[0][1]
    if top <= 0.0:
        kept = [candidate for candidate, _ in scored]
    else:
        threshold = ratio * top
        kept = [candidate for candidate, score in scored if score >= threshold]

    if len(kept) > k_max:
        kept = kept[:k_max]
    if len(kept) < k_min:
        kept = [candidate for candidate, _ in scored[:k_min]]
    return kept
