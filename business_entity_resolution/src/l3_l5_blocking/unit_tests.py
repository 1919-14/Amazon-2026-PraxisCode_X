"""Layer 3: unit tests for blocking channels and recall measurement."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src/ is on the path whether run as a module or a script.
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from l3_l5_blocking import blocked, engine
from l3_l5_blocking.buckets import assign_country
from l3_l5_blocking.channels import ExactKeyChannel, SparseTfidfChannel
from l3_l5_blocking.recall import (
    RecallAccumulator,
    aggregate_stats,
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


def _synthetic_pool(n: int = 30) -> pd.DataFrame:
    """Deterministic synthetic candidate pool for the blocked-engine comparison."""
    base = [
        "acme corporation",
        "acme corp",
        "globex corporation",
        "globex limited",
        "initech solutions",
        "wayne enterprises",
        "stark industries",
        "umbrella corporation",
        "cyberdyne systems",
        "tyrell corporation",
    ]
    rows = []
    for i in range(n):
        name = base[i % len(base)]
        rows.append(
            {
                "entity_id": f"S2-{i:04d}",
                "country_norm": "us",
                "name_core": name,
                "name_norm": name,
                "addr_postal": f"{10000 + (i % 7)}",
                "addr_house_number": str(i % 13),
            }
        )
    return pd.DataFrame(rows)


def _synthetic_references() -> pd.DataFrame:
    """Deterministic synthetic Source 1 references (exact, typo and unrelated names)."""
    specs = [
        ("acme corporation", "10000", "1"),
        ("globex corporation", "10001", "2"),
        ("initech solutions", "10004", "5"),
        ("stark industrias", "10006", "7"),
        ("completely unrelated firm", "99999", "0"),
    ]
    return pd.DataFrame(
        [
            {
                "entity_id": f"S1-{i:04d}",
                "country_norm": "us",
                "name_core": name,
                "name_norm": name,
                "addr_postal": postal,
                "addr_house_number": house,
            }
            for i, (name, postal, house) in enumerate(specs)
        ]
    )


def test_shared_vocab_block_build_matches_classic() -> None:
    """Block-wise TF-IDF (shared vocab) must equal the classic whole-pool build."""
    docs = [
        "acme corporation",
        "globex corporation",
        "acme limited",
        "initech solutions",
        "wayne enterprises",
    ]
    params = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=1, max_df=1.0)
    classic = SparseTfidfChannel("D", **params)
    classic.build(docs)

    blockwise = SparseTfidfChannel("D", **params)
    blockwise.fit_vocab(docs)
    blockwise.build_with_vocab(docs)

    assert classic.inverted.shape == blockwise.inverted.shape
    assert np.allclose(classic.inverted.toarray(), blockwise.inverted.toarray(), atol=1e-5)

    queries = ["acme corporaton", "globex"]
    classic_ids = classic.query(queries, k=3)
    blockwise_ids = blockwise.query(queries, k=3)
    assert classic_ids == blockwise_ids
    classic_scored = classic.query_with_scores(queries, k=3)
    for (ids_a, scores_a), (ids_b, scores_b) in zip(classic_scored, blockwise.query_with_scores(queries, k=3)):
        assert ids_a == ids_b
        assert np.allclose(scores_a, scores_b, atol=1e-5)


def test_merge_topk_keeps_best_per_reference() -> None:
    """Merging two blocks keeps each reference's best k in (score desc, key asc) order."""
    cur = blocked._merge_topk(
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float32),
        np.array([0, 0, 1], dtype=np.int32),
        np.array([10, 11, 20], dtype=np.int32),
        np.array([0.5, 0.4, 0.9], dtype=np.float32),
        2,
    )
    ref, key, score = blocked._merge_topk(
        *cur,
        np.array([0, 0, 2], dtype=np.int32),
        np.array([12, 13, 21], dtype=np.int32),
        np.array([0.7, 0.45, 0.95], dtype=np.float32),
        2,
    )
    assert ref.tolist() == [0, 0, 1, 2]
    assert key.tolist() == [12, 10, 20, 21]
    assert np.allclose(score, [0.7, 0.5, 0.9, 0.95], atol=1e-6)

    # Equal scores resolve by ascending key, so the result is order-stable too.
    tied = blocked._merge_topk(
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float32),
        np.array([0, 0], dtype=np.int32),
        np.array([9, 3], dtype=np.int32),
        np.array([1.0, 1.0], dtype=np.float32),
        2,
    )
    assert tied[1].tolist() == [3, 9]


def test_blocked_engine_matches_classic_engine() -> None:
    """The block-wise engine must reproduce the classic single-index results."""
    candidates = _synthetic_pool(30)
    references = _synthetic_references()

    # Loosen corpus-size-dependent pruning so both engines keep the same vocabulary.
    patched = {
        (module, name): getattr(module, name)
        for module in (engine, blocked)
        for name in ("L3_C_MIN_DF", "L3_C_MAX_DF_FRAC", "L3_D_MIN_DF", "L3_D_MAX_DF_FRAC")
    }
    for module in (engine, blocked):
        module.L3_C_MIN_DF = 1
        module.L3_C_MAX_DF_FRAC = 1.0
        module.L3_D_MIN_DF = 1
        module.L3_D_MAX_DF_FRAC = 1.0

    blocks = [candidates.iloc[i : i + 10].reset_index(drop=True) for i in range(0, 30, 10)]
    original_iter = blocked.iter_country_candidate_blocks
    blocked.iter_country_candidate_blocks = (  # type: ignore[assignment]
        lambda split, country, block_size, max_candidates=None: iter(blocks)
    )
    try:
        classic = engine.run_country_blocking("us", references, candidates, topk=5, enable_char=True)
        with tempfile.TemporaryDirectory() as tmp:
            meta = blocked.build_country_index(
                "test",
                "us",
                refs="all",
                scratch_dir=Path(tmp) / "scratch",
                block_size=10,
                vocab_sample=1000,
                enable_char=True,
                progress=lambda *a, **k: None,
            )
            result = blocked.query_reference_block(meta, references, topk=5)
            decoded = {
                channel: blocked.decode_ids(meta, result.keys[channel])
                for channel in ("A", "C", "D")
            }
            assert meta.n_blocks == 3
            assert meta.n_candidates == 30
            for channel in ("A", "C", "D"):
                assert decoded[channel] == classic.channels[channel], channel
            assert meta.path("meta.json").exists()
            meta.cleanup()
            assert not Path(meta.scratch_dir).exists()
    finally:
        blocked.iter_country_candidate_blocks = original_iter
        for (module, name), value in patched.items():
            setattr(module, name, value)


def test_address_text_does_not_change_channel_a() -> None:
    """Widening the sparse text must not disturb channel A's exact-key lookups.

    Channel A is keyed on the bare name, so a query built from
    ``name_core + addr_norm`` silently matches nothing. This test pins that
    invariant: the exact channel is identical with and without ``include_address``,
    and it is non-empty (the regression produced an empty channel A).
    """
    names = ["acme corporation", "globex corporation", "initech solutions"]
    candidates = pd.DataFrame(
        {
            "entity_id": [f"S2-{i}" for i in range(3)],
            "country_norm": "us",
            "name_core": names,
            "name_norm": names,
            "addr_norm": ["1 market street san francisco", "2 oak avenue austin", "3 pine road boston"],
            "addr_postal": ["94103", "73301", "02108"],
            "addr_house_number": ["1", "2", "3"],
        }
    )
    references = pd.DataFrame(
        {
            "entity_id": ["S1-0", "S1-1", "S1-2"],
            "country_norm": "us",
            "name_core": ["acme corporation", "globex corporation", "initech solutions"],
            "name_norm": ["acme corporation", "globex corporation", "initech solutions"],
            "addr_norm": ["1 market street san francisco", "9 elsewhere", "3 pine road boston"],
            "addr_postal": ["94103", "99999", "02108"],
            "addr_house_number": ["1", "9", "3"],
        }
    )

    patched = {
        name: getattr(blocked, name) for name in ("L3_C_MIN_DF", "L3_C_MAX_DF_FRAC", "L3_D_MIN_DF", "L3_D_MAX_DF_FRAC")
    }
    blocked.L3_C_MIN_DF = 1
    blocked.L3_C_MAX_DF_FRAC = 1.0
    blocked.L3_D_MIN_DF = 1
    blocked.L3_D_MAX_DF_FRAC = 1.0

    blocks = [candidates.reset_index(drop=True)]
    original_iter = blocked.iter_country_candidate_blocks
    blocked.iter_country_candidate_blocks = (  # type: ignore[assignment]
        lambda split, country, block_size, max_candidates=None: iter(blocks)
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            def build(include_address: bool):
                meta = blocked.build_country_index(
                    "test",
                    "us",
                    refs="all",
                    scratch_dir=Path(tmp) / f"scratch_{include_address}",
                    block_size=10,
                    vocab_sample=100,
                    enable_char=True,
                    include_address=include_address,
                    progress=lambda *a, **k: None,
                )
                result = blocked.query_reference_block(meta, references, topk=5)
                decoded = {
                    channel: blocked.decode_ids(meta, result.keys[channel]) for channel in ("A", "C", "D")
                }
                meta.cleanup()
                return decoded

            name_only = build(False)
            widened = build(True)

            assert name_only["A"] == widened["A"], "channel A must ignore the widened text"
            assert widened["A"][0] == ["S2-0"], widened["A"]
            # The widened text must still retrieve the identical name+address first.
            assert widened["C"][0][0] == "S2-0", widened["C"]
    finally:
        blocked.iter_country_candidate_blocks = original_iter
        for name, value in patched.items():
            setattr(blocked, name, value)


def test_recall_accumulator_matches_batch_stats() -> None:
    """Streaming recall must equal the whole-list measurement, split into batches."""
    refs = ["S1-1", "S1-2", "S1-3", "S1-4"]
    per_channel = {
        "A": [["S2-a"], ["S2-b"], [], ["S2-x"]],
        "C": [["S2-a", "S2-c"], ["S2-q"], ["S2-r"], []],
        "D": [[], ["S2-c"], ["S2-r"], ["S2-y"]],
    }
    gt = {"S1-1": {"S2-a", "S3-b"}, "S1-2": {"S2-c"}, "S1-3": set(), "S1-4": {"S2-z"}}

    accumulator = RecallAccumulator()
    accumulator.add_batch(
        refs[:2], {channel: values[:2] for channel, values in per_channel.items()}, gt
    )
    accumulator.add_batch(
        refs[2:], {channel: values[2:] for channel, values in per_channel.items()}, gt
    )

    stats = accumulator.stats()
    for channel in ("A", "C", "D"):
        assert stats[channel] == channel_stats(refs, per_channel[channel], gt), channel
    assert stats["UNION"] == union_stats(refs, per_channel, gt)

    aggregated = aggregate_stats({"country_a": stats, "country_b": {}})
    for channel in ("A", "C", "D", "UNION"):
        assert aggregated[channel]["hit_gt_pairs"] == stats[channel]["hit_gt_pairs"]
        assert aggregated[channel]["micro_recall"] == stats[channel]["micro_recall"]


def test_aggregate_stats_sums_countries_exactly() -> None:
    """Aggregating per-country stats equals measuring the concatenated batches."""
    first = ["S1-1", "S1-2"]
    second = ["S1-3"]
    first_candidates = [["S2-a"], ["S2-b", "S2-c"]]
    second_candidates = [["S2-q"]]
    gt = {"S1-1": {"S2-a"}, "S1-2": {"S2-b"}, "S1-3": {"S2-z"}}

    per_country = {
        "us": {"C": channel_stats(first, first_candidates, gt)},
        "india": {"C": channel_stats(second, second_candidates, gt)},
    }
    combined = aggregate_stats(per_country)
    direct = channel_stats(first + second, first_candidates + second_candidates, gt)
    assert combined["C"]["micro_recall"] == direct["micro_recall"]
    assert combined["C"]["hit_gt_pairs"] == direct["hit_gt_pairs"]
    assert combined["C"]["total_candidates"] == direct["total_candidates"]
    assert combined["C"]["avg_candidates_per_reference"] == direct["avg_candidates_per_reference"]


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
        ("Shared-vocab block build matches classic", test_shared_vocab_block_build_matches_classic),
        ("Top-k merge bounds + ordering", test_merge_topk_keeps_best_per_reference),
        ("Blocked engine matches classic engine", test_blocked_engine_matches_classic_engine),
        ("Address text preserves channel A", test_address_text_does_not_change_channel_a),
        ("Recall accumulator matches batch stats", test_recall_accumulator_matches_batch_stats),
        ("Aggregate stats sum countries", test_aggregate_stats_sums_countries_exactly),
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
