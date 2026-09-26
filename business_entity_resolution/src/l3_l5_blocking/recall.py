"""L3e: per-channel blocking recall and reduction measurement.

Given the candidate lists produced by each channel for a set of reference
entities, compute recall against the training ground truth, any-hit coverage,
average candidate count and singleton contamination.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def channel_stats(
    reference_ids: Sequence[str],
    predicted: Sequence[Sequence[str]],
    ground_truth: Mapping[str, set[str]],
) -> dict:
    """Compute recall-oriented statistics for one channel.

    Args:
        reference_ids: ordered S1 entity ids aligned with ``predicted``.
        predicted: candidate id list (best-first) per reference entity.
        ground_truth: S1 id -> set of true matching S2/S3 ids.

    Returns a dict with micro recall over all true pairs, the fraction of
    non-singleton references with at least one true candidate present, the
    average candidate count per reference, and singleton contamination.
    """
    total_gt_pairs = 0
    hit_gt_pairs = 0
    refs_with_gt = 0
    refs_with_any_hit = 0
    singleton_refs = 0
    singleton_with_candidates = 0
    total_candidates = 0

    for s1_id, cands in zip(reference_ids, predicted):
        total_candidates += len(cands)
        truth = ground_truth.get(s1_id)

        if truth is None:
            continue

        if not truth:
            singleton_refs += 1
            if cands:
                singleton_with_candidates += 1
            continue

        refs_with_gt += 1
        total_gt_pairs += len(truth)
        hits = len(set(cands) & truth)
        hit_gt_pairs += hits
        if hits > 0:
            refs_with_any_hit += 1

    n_refs = len(reference_ids)
    return {
        "n_references": n_refs,
        "n_references_with_gt": refs_with_gt,
        "total_gt_pairs": total_gt_pairs,
        "hit_gt_pairs": hit_gt_pairs,
        "micro_recall": (hit_gt_pairs / total_gt_pairs) if total_gt_pairs else 0.0,
        "any_hit_recall": (refs_with_any_hit / refs_with_gt) if refs_with_gt else 0.0,
        "avg_candidates_per_reference": (total_candidates / n_refs) if n_refs else 0.0,
        "singleton_references": singleton_refs,
        "singleton_candidate_rate": (
            singleton_with_candidates / singleton_refs if singleton_refs else 0.0
        ),
    }


def union_stats(
    reference_ids: Sequence[str],
    per_channel: Mapping[str, Sequence[Sequence[str]]],
    ground_truth: Mapping[str, set[str]],
) -> dict:
    """Compute recall of the union of all channels (the L3 recall ceiling)."""
    merged: list[list[str]] = []
    for i in range(len(reference_ids)):
        seen: set[str] = set()
        merged_ids: list[str] = []
        for cands in per_channel.values():
            for cand in cands[i]:
                if cand not in seen:
                    seen.add(cand)
                    merged_ids.append(cand)
        merged.append(merged_ids)
    return channel_stats(reference_ids, merged, ground_truth)


def ranked_recall_counts(
    reference_ids: Sequence[str],
    ranked: Sequence[Sequence[str]],
    ground_truth: Mapping[str, set[str]],
    cutoffs: Sequence[int] = (5, 10, 20, 50),
) -> tuple[dict[int, int], int]:
    """Raw true-pair hit counts at each cutoff plus the total true-pair count.

    Returning counts (rather than fractions) lets callers aggregate across
    country buckets with correct weighting.
    """
    total_gt_pairs = 0
    hit_at = {cut: 0 for cut in cutoffs}

    for s1_id, ranked_ids in zip(reference_ids, ranked):
        truth = ground_truth.get(s1_id)
        if not truth:
            continue
        total_gt_pairs += len(truth)
        for cut in cutoffs:
            hit_at[cut] += len(set(ranked_ids[:cut]) & truth)

    return hit_at, total_gt_pairs


def ranked_recall_at(
    reference_ids: Sequence[str],
    ranked: Sequence[Sequence[str]],
    ground_truth: Mapping[str, set[str]],
    cutoffs: Sequence[int] = (5, 10, 20, 50),
) -> dict[str, float]:
    """Micro recall over true pairs when keeping only the top-N ranked candidates.

    Used to measure how well a ranking (e.g. RRF fusion) positions true matches,
    since fusion reorders the union but does not drop candidates.
    """
    hit_at, total_gt_pairs = ranked_recall_counts(reference_ids, ranked, ground_truth, cutoffs)
    if total_gt_pairs == 0:
        return {f"recall@{cut}": 0.0 for cut in cutoffs}
    return {f"recall@{cut}": hit_at[cut] / total_gt_pairs for cut in cutoffs}


def mean_first_hit_rank(
    reference_ids: Sequence[str],
    ranked: Sequence[Sequence[str]],
    ground_truth: Mapping[str, set[str]],
) -> float:
    """Average rank (1-based) of the first true match, ignoring references with none."""
    total = 0.0
    counted = 0
    for s1_id, ranked_ids in zip(reference_ids, ranked):
        truth = ground_truth.get(s1_id)
        if not truth:
            continue
        for rank, candidate in enumerate(ranked_ids, start=1):
            if candidate in truth:
                total += rank
                counted += 1
                break
    return (total / counted) if counted else 0.0


def format_report(stats_by_channel: Mapping[str, dict]) -> str:
    """Render a compact human-readable report for console output."""
    lines = [
        f"{'channel':<10}{'recall':>10}{'any-hit':>10}{'avg cands':>12}{'singleton%':>12}",
        "-" * 54,
    ]
    for name, stats in stats_by_channel.items():
        lines.append(
            f"{name:<10}"
            f"{stats['micro_recall'] * 100:>9.2f}%"
            f"{stats['any_hit_recall'] * 100:>9.2f}%"
            f"{stats['avg_candidates_per_reference']:>12.2f}"
            f"{stats['singleton_candidate_rate'] * 100:>11.2f}%"
        )
    return "\n".join(lines)
