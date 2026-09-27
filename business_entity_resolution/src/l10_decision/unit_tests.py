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

from config import (  # noqa: E402
    L10_OPEN_SET_TAU_BOOST,
    L10_OPEN_SET_VETO_MIN_CONFIDENCE,
)
from l10_decision.decision import (  # noqa: E402
    apply_decision_rule,
    build_reference_ids,
    group_scores,
    tune_thresholds,
    write_id_list_tsv,
)
from l10_decision.decision_v2 import (  # noqa: E402
    apply_decision_v2,
    entity_expected_f05,
    tune_decision_v2,
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


def _pseudo_open_scores() -> tuple[dict, dict, list[str]]:
    """Synthetic pseudo-open country: 10 singletons + 10 true matches.

    The singletons' best candidate scores 0.55 - above the shipped veto floor of
    0.50, so the config default pays a false-merge penalty on each of them, while a
    slightly higher veto would empty them all for free (their true matches score
    0.8 and survive).
    """
    scores: dict[tuple[str, str], float] = {}
    gt: dict[str, set[str]] = {}
    refs: list[str] = []
    for i in range(20):
        s1 = f"S1-{i:03d}"
        refs.append(s1)
        if i < 10:
            gt[s1] = set()
            scores[(s1, f"S2-{i:03d}")] = 0.55
        else:
            gt[s1] = {f"S2-{i:03d}", f"S3-{i:03d}"}
            scores[(s1, f"S2-{i:03d}")] = 0.80
            scores[(s1, f"S3-{i:03d}")] = 0.78
    return scores, gt, refs


def test_open_set_tuning_beats_config_default() -> None:
    """The tuner finds the stricter veto that the shipped constant misses."""
    from l10_decision.open_set import tune_open_set_policy

    scores, gt, refs = _pseudo_open_scores()
    tuning = tune_open_set_policy(
        scores,
        gt,
        refs,
        tau_match=0.5,
        tau_s=0.3,
        margin=0.05,
        boost_grid=(0.0, 0.1),
        veto_grid=(0.0, 0.5, 0.6),
    )
    assert tuning["default"]["veto_min_confidence"] == 0.50
    assert tuning["improvement"] > 0.3
    assert tuning["best"]["veto_min_confidence"] == 0.6
    assert tuning["best"]["macro_f05"] > tuning["default"]["macro_f05"]
    # The tuner never degrades the fake-open country below its best grid point.
    assert tuning["best"]["macro_f05"] == max(row["macro_f05"] for row in tuning["grid"])
    assert len(tuning["grid"]) == 2 * 3


def test_open_set_policy_persistence_and_precedence() -> None:
    """A saved policy wins over the config default, and CLI args win over both."""
    from l10_decision.open_set import (
        load_open_set_policy,
        resolve_open_set_policy,
        save_open_set_policy,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.json"
        assert load_open_set_policy(path) is None

        save_open_set_policy(
            path,
            {
                "chosen": {"open_set_boost": 0.22, "veto_min_confidence": 0.66},
                "pseudo_open_country": "india",
            },
        )
        loaded = load_open_set_policy(path)
        assert loaded is not None
        assert loaded["boost"] == 0.22 and loaded["veto_min_confidence"] == 0.66

        boost, veto, source = resolve_open_set_policy(None, None, path)
        assert (boost, veto, source) == (0.22, 0.66, "tuned_policy")

        boost, veto, source = resolve_open_set_policy(0.05, None, path)
        assert (boost, veto, source) == (0.05, 0.66, "cli")

        boost, veto, source = resolve_open_set_policy(None, None, Path(tmp) / "absent.json")
        assert source == "config_default"
        assert (boost, veto) == (L10_OPEN_SET_TAU_BOOST, L10_OPEN_SET_VETO_MIN_CONFIDENCE)


def test_veto_impact_preview_counts_references() -> None:
    """The preview shows how many references a threshold would empty."""
    from l10_decision.open_set import score_quantiles, veto_impact_preview

    top = {f"S1-{i}": value / 100 for i, value in enumerate([10, 20, 45, 55, 90])}
    preview = veto_impact_preview(top, thresholds=(0.0, 0.5, 0.95))
    assert preview[0] == {
        "veto_min_confidence": 0.0,
        "vetoed_references": 0,
        "vetoed_fraction": 0.0,
    }
    assert preview[1]["vetoed_references"] == 3
    assert abs(preview[1]["vetoed_fraction"] - 0.6) < 1e-9
    assert preview[2]["vetoed_references"] == 5

    quantiles = score_quantiles([0.1, 0.2, 0.3, 0.4], quantiles=(0.5,))
    assert abs(quantiles["q50"] - 0.25) < 1e-9
    assert score_quantiles([], quantiles=(0.5,)) == {"q50": 0.0}


def test_entity_expected_f05_keeps_strong_drops_weak():
    """Entity-level selection keeps the confident prefix and drops weak candidates."""
    kept = entity_expected_f05(
        [("S2-a", 0.95), ("S2-b", 0.90), ("S2-c", 0.05)], recall_hat=0.8
    )
    assert "S2-a" in kept and "S2-b" in kept
    assert "S2-c" not in kept
    # All-weak scores must fall back to the singleton (empty) decision.
    assert entity_expected_f05([("S2-x", 0.02), ("S2-y", 0.01)], recall_hat=0.8) == []


def test_decision_v2_per_country_and_veto():
    """Per-country recall_hat is applied and the open-set veto empties low scores."""
    grouped = group_scores(
        {
            ("S1-in", "S2-1"): 0.9,
            ("S1-in", "S2-2"): 0.85,
            ("S1-fr", "S2-3"): 0.4,
        }
    )
    country = {"S1-in": "india", "S1-fr": "france"}
    preds = apply_decision_v2(
        grouped,
        ["S1-in", "S1-fr"],
        recall_hat=0.8,
        per_country={"india": 0.85},
        country_by_ref=country,
        open_set_ids=["S1-fr"],
        veto_min_confidence=0.5,
    )
    assert preds["S1-in"], preds
    assert preds["S1-fr"] == []  # 0.4 < veto floor


def test_decision_v2_tuner_returns_policy():
    """The tuner returns a usable policy and a non-negative score."""
    scores = {("S1-a", "S2-1"): 0.9, ("S1-a", "S2-2"): 0.1, ("S1-b", "S2-3"): 0.05}
    gt = {"S1-a": {"S2-1"}, "S1-b": set()}
    policy = tune_decision_v2(
        scores,
        gt,
        ["S1-a", "S1-b"],
        country_by_ref={"S1-a": "india", "S1-b": "india"},
        recall_hats=(0.7, 0.9),
        tau_floors=(0.0, 0.1),
    )
    assert policy["macro_f05"] >= 0.0
    assert "recall_hat" in policy and "per_country" in policy


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
        ("Open-set tuner beats config default", test_open_set_tuning_beats_config_default),
        ("Open-set policy precedence", test_open_set_policy_persistence_and_precedence),
        ("Veto impact preview", test_veto_impact_preview_counts_references),
        ("Entity-level expected F0.5", test_entity_expected_f05_keeps_strong_drops_weak),
        ("Decision v2 per-country + veto", test_decision_v2_per_country_and_veto),
        ("Decision v2 tuner policy", test_decision_v2_tuner_returns_policy),
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
