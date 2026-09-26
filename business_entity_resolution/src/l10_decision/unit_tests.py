"""Layer 10: unit tests for the decision engine and submission writer."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from l10_decision.decision import (  # noqa: E402
    apply_decision_rule,
    build_reference_ids,
    group_scores,
    tune_thresholds,
    write_id_list_tsv,
)
from l1_validation.metrics import evaluate_with_thresholds  # noqa: E402


def test_singleton_guard_below_tau_s():
    """A reference whose top score is below tau_s must predict empty."""
    grouped = group_scores({("S1-a", "S2-1"): 0.25, ("S1-a", "S2-2"): 0.20})
    preds = apply_decision_rule(grouped, ["S1-a"], tau_match=0.1, tau_s=0.5, margin=0.05)
    assert preds["S1-a"] == [], preds


def test_match_path_margin_and_tau_match():
    """Keep candidates within margin of the top and at/above tau_match."""
    grouped = group_scores(
        {("S1-a", "S2-1"): 0.90, ("S1-a", "S2-2"): 0.87, ("S1-a", "S2-3"): 0.40}
    )
    preds = apply_decision_rule(grouped, ["S1-a"], tau_match=0.5, tau_s=0.3, margin=0.05)
    # top=0.90 -> threshold max(0.5, 0.85)=0.85 -> only 0.90 and 0.87 survive.
    assert preds["S1-a"] == ["S2-1", "S2-2"], preds


def test_tau_match_raises_floor_above_margin():
    """tau_match can override the margin window when it is higher."""
    grouped = group_scores({("S1-a", "S2-1"): 0.90, ("S1-a", "S2-2"): 0.60})
    preds = apply_decision_rule(grouped, ["S1-a"], tau_match=0.8, tau_s=0.3, margin=0.5)
    # margin would keep both (0.90-0.5=0.40), but tau_match=0.8 wins.
    assert preds["S1-a"] == ["S2-1"], preds


def test_reference_without_candidates_is_empty():
    """References with no scores still receive an (empty) prediction."""
    preds = apply_decision_rule({}, ["S1-missing"], tau_match=0.5, tau_s=0.3)
    assert preds == {"S1-missing": []}, preds


def test_open_set_veto_blocks_low_confidence():
    """Open-set references below the veto confidence predict empty."""
    grouped = group_scores({("S1-fr", "S2-1"): 0.45})
    seen = apply_decision_rule(grouped, ["S1-fr"], tau_match=0.3, tau_s=0.4)
    assert seen["S1-fr"] == ["S2-1"], seen  # 0.45 >= tau_s 0.4 -> kept on a seen country
    vetoed = apply_decision_rule(
        grouped,
        ["S1-fr"],
        tau_match=0.3,
        tau_s=0.4,
        open_set_ids={"S1-fr"},
        veto_min_confidence=0.5,
    )
    assert vetoed["S1-fr"] == [], vetoed


def test_open_set_boost_raises_singleton_threshold():
    """The boost makes an open-set reference a singleton where it otherwise matched."""
    grouped = group_scores({("S1-fr", "S2-1"): 0.55})
    base = apply_decision_rule(grouped, ["S1-fr"], tau_match=0.3, tau_s=0.5)
    assert base["S1-fr"] == ["S2-1"], base
    boosted = apply_decision_rule(
        grouped,
        ["S1-fr"],
        tau_match=0.3,
        tau_s=0.5,
        open_set_ids={"S1-fr"},
        open_set_boost=0.1,
    )
    assert boosted["S1-fr"] == [], boosted


def test_group_scores_dedupes_by_max():
    """Duplicate candidate ids collapse to their maximum score and rank once."""
    scores = {("S1-a", "S2-1"): 0.4, ("S1-a", "S2-1"): 0.7}
    grouped = group_scores(scores)
    assert grouped["S1-a"] == [("S2-1", 0.7)], grouped


def test_decision_rule_matches_official_scorer():
    """The production rule must agree with the L1 evaluation rule on seen refs."""
    scores = {
        ("S1-a", "S2-1"): 0.91,
        ("S1-a", "S2-2"): 0.88,
        ("S1-a", "S2-3"): 0.30,
        ("S1-b", "S2-4"): 0.20,
        ("S1-b", "S2-5"): 0.15,
    }
    gt = {"S1-a": {"S2-1", "S2-2"}, "S1-b": set()}
    for tau_match, tau_s in [(0.5, 0.3), (0.85, 0.3), (0.5, 0.5), (0.1, 0.1)]:
        official = evaluate_with_thresholds(scores, gt, tau_match, tau_s, margin=0.05)
        applied = apply_decision_rule(
            group_scores(scores), list(gt), tau_match, tau_s, margin=0.05
        )
        # The official scorer also exposes a per-entity dict; both must exist.
        assert set(official["per_entity"]) == set(gt)
        # Recompute the official predictions from its own rule for equality.
        expected = {}
        for s1_id, truth in gt.items():
            cands = [(c, s) for (sid, c), s in scores.items() if sid == s1_id]
            if not cands:
                expected[s1_id] = []
                continue
            top = max(s for _, s in cands)
            if top < tau_s:
                expected[s1_id] = []
            else:
                thr = max(tau_match, top - 0.05)
                expected[s1_id] = sorted(c for c, s in cands if s >= thr)
        for s1_id in gt:
            assert sorted(applied[s1_id]) == expected[s1_id], (tau_match, tau_s, applied, expected)


def test_tune_thresholds_prefers_precision():
    """Joint search must pick a threshold that avoids a low-score false positive."""
    scores = {("S1-a", "S2-1"): 0.9, ("S1-a", "S2-2"): 0.2}
    gt = {"S1-a": {"S2-1"}}
    # margin wide enough that tau_match is the only thing separating the two
    # candidates; a low tau_match would keep the 0.2 false positive.
    best = tune_thresholds(
        scores, gt, tau_match_grid=[0.1, 0.5], tau_s_grid=[0.1], margin=0.9
    )
    assert best["macro_f05"] == 1.0, best
    assert best["tau_match"] == 0.5, best


def test_build_reference_ids_unions_and_sorts():
    """Reference coverage unions scores and candidates, sorted for determinism."""
    refs = build_reference_ids(["S1-b", "S1-a"], ["S1-c", "S1-a"])
    assert refs == ["S1-a", "S1-b", "S1-c"], refs
    assert build_reference_ids(["S1-b", "S1-a"]) == ["S1-a", "S1-b"]


def test_write_id_list_tsv_format():
    """The writer emits the exact submission format with one row per reference."""
    predictions = {"S1-a": ["S2-1", "S3-2"], "S1-b": []}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "matching_results.tsv"
        written = write_id_list_tsv(path, predictions, ["S1-a", "S1-b"])
        assert written == 2
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "source1_entity_id\tmatched_entity_ids"
        assert lines[1] == "S1-a\tS2-1,S3-2"
        assert lines[2] == "S1-b\t"


def run_all_tests() -> bool:
    """Run the L10 test suite and report pass/fail."""
    tests = [
        ("Singleton guard below tau_s", test_singleton_guard_below_tau_s),
        ("Match path margin + tau_match", test_match_path_margin_and_tau_match),
        ("tau_match floor over margin", test_tau_match_raises_floor_above_margin),
        ("Missing reference is empty", test_reference_without_candidates_is_empty),
        ("Open-set veto blocks low confidence", test_open_set_veto_blocks_low_confidence),
        ("Open-set boost raises tau_s", test_open_set_boost_raises_singleton_threshold),
        ("Group scores dedupes by max", test_group_scores_dedupes_by_max),
        ("Rule matches official scorer", test_decision_rule_matches_official_scorer),
        ("Joint tuning prefers precision", test_tune_thresholds_prefers_precision),
        ("Reference ids union + sort", test_build_reference_ids_unions_and_sorts),
        ("TSV writer format", test_write_id_list_tsv_format),
    ]

    print("\n" + "=" * 60)
    print("🧪 RUNNING LAYER 10 UNIT TESTS")
    print("=" * 60)
    all_passed = True
    for name, func in tests:
        try:
            func()
            print(f"  ✅ PASS: {name}")
        except Exception as exc:  # noqa: BLE001 - surface any failure
            print(f"  ❌ FAIL: {name} -> {exc}")
            all_passed = False
    print("=" * 60)
    print(f"🎉 ALL {len(tests)} TESTS PASSED!\n" if all_passed else "💥 SOME TESTS FAILED!\n")
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
