"""Layer 3: unit tests for blocking channels and recall measurement."""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src/ is on the path whether run as a module or a script.
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from l3_l5_blocking.buckets import assign_country
from l3_l5_blocking.channels import ExactKeyChannel, SparseTfidfChannel
from l3_l5_blocking.recall import (
    channel_stats,
    ranked_recall_at,
    ranked_recall_counts,
    union_stats,
)
from l3_l5_blocking.rrf import fuse, ranked_ids, rrf_scores
from l3_l5_blocking.truncate import adaptive_truncate, coarse_score


def test_exact_key_channel_priority() -> None:
    """Channel A should prefer name+postal matches over bare name matches."""
    channel = ExactKeyChannel(max_posting=10)
    channel.build(
        names=["acme corp", "acme corp", "acme corp"],
        postals=["10001", "20002", ""],
        houses=["1", "2", ""],
    )
    # Exact name + postal narrows to the single matching candidate.
    assert channel.query("acme corp", "10001", "1", k=10) == [0]
    # Without postal, the house key narrows to candidate 1.
    assert channel.query("acme corp", "", "2", k=10) == [1]
    # Bare name falls back to all postings.
    assert sorted(channel.query("acme corp", "", "", k=10)) == [0, 1, 2]
    # Unknown name yields nothing.
    assert channel.query("globex", "10001", "", k=10) == []


def test_exact_key_channel_max_posting() -> None:
    """Channel A should cap overly common keys at ``max_posting``."""
    channel = ExactKeyChannel(max_posting=2)
    channel.build(names=["same"] * 5, postals=[""] * 5, houses=[""] * 5)
    assert channel.query("same", "", "", k=10) == [0, 1]


def test_word_channel_retrieves_shared_tokens() -> None:
    """Channel C should retrieve candidates sharing high-IDF tokens."""
    channel = SparseTfidfChannel(
        name="C", analyzer="word", ngram_range=(1, 1), min_df=1, max_df=1.0
    )
    channel.build(["acme corporation", "globex corporation", "acme limited"])
    results = channel.query(["acme"], k=5)
    assert results[0] and set(results[0]).issuperset({0, 2})
    assert 1 not in results[0]


def test_char_channel_tolerates_typo() -> None:
    """Channel D should surface the closest string despite a spelling variant."""
    channel = SparseTfidfChannel(
        name="D", analyzer="char_wb", ngram_range=(2, 4), min_df=1, max_df=1.0
    )
    docs = ["acme corporation", "globex corporation", "totally different"]
    channel.build(docs)
    results = channel.query(["acme corporaton"], k=1)
    assert results[0] == [0]


def test_empty_channel_query() -> None:
    """Querying an empty/unbuilt channel must not crash."""
    channel = SparseTfidfChannel(
        name="C", analyzer="word", ngram_range=(1, 1), min_df=1, max_df=1.0
    )
    channel.build([])
    assert channel.query(["acme"], k=5) == [[]]


def test_country_assignment() -> None:
    """Country labels map to canonical buckets, unknown labels pass through."""
    assert assign_country("US") == "us"
    assert assign_country("United States") == "us"
    assert assign_country("India") == "india"
    assert assign_country("FR") == "france"
    assert assign_country("Brazil") == "brazil"
    assert assign_country("") == "other"


def test_channel_stats_recall() -> None:
    """Recall, any-hit, and singleton contamination are computed correctly."""
    refs = ["S1-1", "S1-2", "S1-3"]
    predicted = [["S2-a", "S2-x"], ["S2-y"], ["S2-z"]]
    gt = {"S1-1": {"S2-a", "S3-b"}, "S1-2": {"S2-q"}, "S1-3": set()}
    stats = channel_stats(refs, predicted, gt)
    assert stats["total_gt_pairs"] == 3
    assert stats["hit_gt_pairs"] == 1
    assert abs(stats["micro_recall"] - 1 / 3) < 1e-9
    assert abs(stats["any_hit_recall"] - 0.5) < 1e-9
    assert stats["singleton_references"] == 1
    assert stats["singleton_candidate_rate"] == 1.0


def test_union_stats() -> None:
    """The union of channels should recover candidates split across channels."""
    refs = ["S1-1"]
    per_channel = {"A": [["S2-a"]], "C": [["S2-b"]], "D": [[]]}
    gt = {"S1-1": {"S2-a", "S2-b"}}
    stats = union_stats(refs, per_channel, gt)
    assert stats["micro_recall"] == 1.0
    assert stats["avg_candidates_per_reference"] == 2.0


def test_rrf_scores_basic() -> None:
    """A candidate ranked highly by multiple channels should win fusion."""
    fused = rrf_scores(
        {"A": ["x", "a"], "C": ["y", "x"]},
        k=60,
    )
    order = [candidate for candidate, _ in fused]
    # x appears rank1(A) + rank2(C); y only rank1(C); a only rank2(A).
    assert order[0] == "x"
    assert order[1] == "y"
    assert order[2] == "a"
    # Scores are strictly descending.
    assert fused[0][1] > fused[1][1] > fused[2][1]


def test_rrf_tie_break_is_deterministic() -> None:
    """Equal-scoring candidates are ordered by id ascending."""
    fused = rrf_scores({"A": ["b"], "C": ["a"]}, k=60)
    assert [candidate for candidate, _ in fused] == ["a", "b"]


def test_rrf_weights_and_zero() -> None:
    """Weights scale contributions and a zero weight disables a channel."""
    fused = rrf_scores({"A": ["x"], "C": ["y"]}, k=60, weights={"A": 2.0, "C": 0.0})
    assert [candidate for candidate, _ in fused] == ["x"]


def test_rrf_fuse_many() -> None:
    """fuse() aligns per-reference channel lists and ranked_ids strips scores."""
    channel_candidates = {
        "A": [["x"], ["p"]],
        "C": [["y"], ["p", "q"]],
    }
    fused = fuse(channel_candidates, k=60)
    ids = ranked_ids(fused)
    assert len(ids) == 2
    assert ids[0][0] == "x"
    assert ids[1][0] == "p"  # p matched by both channels for ref 2


def test_ranked_recall_at_cutoff() -> None:
    """Recall@N reflects how early true matches are ranked."""
    refs = ["S1-1", "S1-2"]
    ranked = [["X", "Y", "S2-a"], ["S2-b"]]
    gt = {"S1-1": {"S2-a"}, "S1-2": {"S2-b"}}
    counts, total = ranked_recall_counts(refs, ranked, gt, cutoffs=(1, 3))
    assert total == 2
    assert counts[1] == 1  # only S2-b is at rank 1
    assert counts[3] == 2
    stats = ranked_recall_at(refs, ranked, gt, cutoffs=(1, 3))
    assert stats["recall@1"] == 0.5
    assert stats["recall@3"] == 1.0


def test_coarse_score_prefers_agreement_and_exact() -> None:
    """A candidate matched by more channels and by the exact key should rank first."""
    fused = [("a", 0.9), ("b", 0.8), ("c", 0.1)]
    channel_lists = {"A": ["b"], "C": ["b", "a"], "D": []}
    scored = coarse_score(fused, channel_lists)
    assert scored[0][0] == "b"
    # ordering is strictly descending
    assert scored[0][1] > scored[1][1] > scored[2][1]


def test_coarse_score_without_channels() -> None:
    """With no channel context the coarse order follows the fused order."""
    fused = [("a", 0.9), ("b", 0.8)]
    scored = coarse_score(fused, None)
    assert [c for c, _ in scored] == ["a", "b"]


def test_adaptive_truncate_ratio() -> None:
    """The retention ratio sets how far down the coarse list we keep."""
    scored = [("x", 1.0), ("y", 0.6), ("z", 0.2)]
    assert adaptive_truncate(scored, ratio=0.5, k_min=1, k_max=12) == ["x", "y"]
    assert adaptive_truncate(scored, ratio=0.8, k_min=1, k_max=12) == ["x"]


def test_adaptive_truncate_bounds() -> None:
    """k_min floors and k_max caps the kept set."""
    scored = [("x", 1.0), ("y", 0.9), ("z", 0.8)]
    assert adaptive_truncate(scored, ratio=0.95, k_min=1, k_max=12) == ["x"]
    assert adaptive_truncate(scored, ratio=0.95, k_min=2, k_max=12) == ["x", "y"]
    assert adaptive_truncate(scored, ratio=0.1, k_min=1, k_max=2) == ["x", "y"]
    assert adaptive_truncate([], ratio=0.5) == []


def run_all_tests() -> bool:
    """Run every test and report pass/fail."""
    tests = [
        ("ExactKeyChannel priority", test_exact_key_channel_priority),
        ("ExactKeyChannel max_posting", test_exact_key_channel_max_posting),
        ("Channel C shared tokens", test_word_channel_retrieves_shared_tokens),
        ("Channel D typo tolerance", test_char_channel_tolerates_typo),
        ("Empty channel query", test_empty_channel_query),
        ("Country assignment", test_country_assignment),
        ("Channel recall stats", test_channel_stats_recall),
        ("Union stats", test_union_stats),
        ("RRF basic fusion", test_rrf_scores_basic),
        ("RRF deterministic tie-break", test_rrf_tie_break_is_deterministic),
        ("RRF weights", test_rrf_weights_and_zero),
        ("RRF batch fusion", test_rrf_fuse_many),
        ("RRF ranked recall@N", test_ranked_recall_at_cutoff),
        ("L5 coarse scoring", test_coarse_score_prefers_agreement_and_exact),
        ("L5 coarse score fallback", test_coarse_score_without_channels),
        ("L5 adaptive ratio", test_adaptive_truncate_ratio),
        ("L5 adaptive bounds", test_adaptive_truncate_bounds),
    ]

    print("\n" + "=" * 60)
    print("🧪 RUNNING LAYER 3-5 UNIT TESTS")
    print("=" * 60)
    all_passed = True
    for name, func in tests:
        try:
            func()
            print(f"  ✅ PASS: {name}")
        except Exception as exc:  # noqa: BLE001 - report any failure
            print(f"  ❌ FAIL: {name} -> {exc}")
            all_passed = False
    print("=" * 60)
    print(f"🎉 ALL {len(tests)} TESTS PASSED!\n" if all_passed else "💥 SOME TESTS FAILED!\n")
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
