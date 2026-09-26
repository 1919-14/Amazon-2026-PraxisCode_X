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
        "n_references_with_any_hit": refs_with_any_hit,
        "total_gt_pairs": total_gt_pairs,
        "hit_gt_pairs": hit_gt_pairs,
        "micro_recall": (hit_gt_pairs / total_gt_pairs) if total_gt_pairs else 0.0,
        "any_hit_recall": (refs_with_any_hit / refs_with_gt) if refs_with_gt else 0.0,
        "avg_candidates_per_reference": (total_candidates / n_refs) if n_refs else 0.0,
        "total_candidates": total_candidates,
        "singleton_references": singleton_refs,
        "singleton_with_candidates": singleton_with_candidates,
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


class _Counters:
    """Additive version of the counters :func:`channel_stats` computes."""

    __slots__ = (
        "n_references",
        "total_gt_pairs",
        "hit_gt_pairs",
        "refs_with_gt",
        "refs_with_any_hit",
        "singleton_refs",
        "singleton_with_candidates",
        "total_candidates",
    )

    def __init__(self) -> None:
        self.n_references = 0
        self.total_gt_pairs = 0
        self.hit_gt_pairs = 0
        self.refs_with_gt = 0
        self.refs_with_any_hit = 0
        self.singleton_refs = 0
        self.singleton_with_candidates = 0
        self.total_candidates = 0

    def add(
        self,
        reference_ids: Sequence[str],
        predicted: Sequence[Sequence[str]],
        ground_truth: Mapping[str, set[str]],
    ) -> None:
        """Accumulate one batch of predictions (streaming equivalent of channel_stats)."""
        self.n_references += len(reference_ids)
        for s1_id, cands in zip(reference_ids, predicted):
            self.total_candidates += len(cands)
            truth = ground_truth.get(s1_id)
            if truth is None:
                continue
            if not truth:
                self.singleton_refs += 1
                if cands:
                    self.singleton_with_candidates += 1
                continue
            self.refs_with_gt += 1
            self.total_gt_pairs += len(truth)
            hits = len(set(cands) & truth)
            self.hit_gt_pairs += hits
            if hits > 0:
                self.refs_with_any_hit += 1

    def stats(self) -> dict:
        """Render the same dict shape as :func:`channel_stats`."""
        return {
            "n_references": self.n_references,
            "n_references_with_gt": self.refs_with_gt,
            "n_references_with_any_hit": self.refs_with_any_hit,
            "total_gt_pairs": self.total_gt_pairs,
            "hit_gt_pairs": self.hit_gt_pairs,
            "micro_recall": (self.hit_gt_pairs / self.total_gt_pairs) if self.total_gt_pairs else 0.0,
            "any_hit_recall": (self.refs_with_any_hit / self.refs_with_gt) if self.refs_with_gt else 0.0,
            "avg_candidates_per_reference": (
                self.total_candidates / self.n_references if self.n_references else 0.0
            ),
            "total_candidates": self.total_candidates,
            "singleton_references": self.singleton_refs,
            "singleton_with_candidates": self.singleton_with_candidates,
            "singleton_candidate_rate": (
                self.singleton_with_candidates / self.singleton_refs if self.singleton_refs else 0.0
            ),
        }


class RecallAccumulator:
    """Streaming recall measurement over reference batches.

    The block-wise engine never holds a whole country's candidate lists, so recall
    is accumulated as rows stream past instead of being computed from complete
    lists. Results are identical to :func:`channel_stats` / :func:`union_stats`
    over the concatenated batches (asserted by the unit tests).
    """

    def __init__(self, channels: Sequence[str] = ("A", "C", "D")) -> None:
        self.channels = tuple(channels)
        self._per_channel = {channel: _Counters() for channel in self.channels}
        self._union = _Counters()

    def add_batch(
        self,
        reference_ids: Sequence[str],
        per_channel: Mapping[str, Sequence[Sequence[str]]],
        ground_truth: Mapping[str, set[str]],
    ) -> None:
        """Accumulate one batch of per-channel candidate lists."""
        merged: list[list[str]] = []
        for index in range(len(reference_ids)):
            seen: set[str] = set()
            merged_ids: list[str] = []
            for channel in self.channels:
                candidates = per_channel.get(channel) or []
                if index >= len(candidates):
                    continue
                for candidate in candidates[index]:
                    if candidate not in seen:
                        seen.add(candidate)
                        merged_ids.append(candidate)
            merged.append(merged_ids)

        for channel in self.channels:
            candidates = per_channel.get(channel) or [[] for _ in reference_ids]
            self._per_channel[channel].add(reference_ids, candidates, ground_truth)
        self._union.add(reference_ids, merged, ground_truth)

    def channel_stats(self) -> dict[str, dict]:
        """Per-channel stats (same keys as :func:`channel_stats`)."""
        return {channel: counters.stats() for channel, counters in self._per_channel.items()}

    def union_stats(self) -> dict:
        """Union-of-channels stats (same keys as :func:`union_stats`)."""
        return self._union.stats()

    def stats(self) -> dict[str, dict]:
        """Per-channel stats plus a ``UNION`` entry, ready for ``format_report``."""
        stats = self.channel_stats()
        stats["UNION"] = self.union_stats()
        return stats


COUNTER_KEYS = (
    "n_references",
    "n_references_with_gt",
    "n_references_with_any_hit",
    "total_gt_pairs",
    "hit_gt_pairs",
    "total_candidates",
    "singleton_references",
    "singleton_with_candidates",
)


def aggregate_stats(per_country_stats: Mapping[str, Mapping[str, dict]]) -> dict[str, dict]:
    """Combine per-country channel stats into overall stats.

    Exact (not an average of averages): the raw counters are summed and the rates
    recomputed, so a large country bucket is weighted as one.
    """
    totals: dict[str, dict[str, int]] = {}
    for country_stats in per_country_stats.values():
        for channel, stats in country_stats.items():
            bucket = totals.setdefault(channel, {key: 0 for key in COUNTER_KEYS})
            for key in COUNTER_KEYS:
                bucket[key] += int(stats.get(key, 0))

    overall: dict[str, dict] = {}
    for channel, counts in totals.items():
        n_refs = counts["n_references"]
        overall[channel] = {
            **counts,
            "micro_recall": (
                counts["hit_gt_pairs"] / counts["total_gt_pairs"]
                if counts["total_gt_pairs"]
                else 0.0
            ),
            "any_hit_recall": (
                counts["n_references_with_any_hit"] / counts["n_references_with_gt"]
                if counts["n_references_with_gt"]
                else 0.0
            ),
            "avg_candidates_per_reference": (
                counts["total_candidates"] / n_refs if n_refs else 0.0
            ),
            "singleton_candidate_rate": (
                counts["singleton_with_candidates"] / counts["singleton_references"]
                if counts["singleton_references"]
                else 0.0
            ),
        }
    return overall


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
