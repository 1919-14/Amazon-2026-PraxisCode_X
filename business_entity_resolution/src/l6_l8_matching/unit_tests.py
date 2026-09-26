"""Layer 6: unit tests for pair sampling and candidate-pairs parsing."""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from l6_l8_matching.features import (
    FEATURE_NAMES,
    build_idf,
    compute_features,
    compute_retrieval_features,
)
# model.py loads LightGBM before scikit-learn; keep this import before calibration
# (sklearn) to avoid the Windows OpenMP import-order conflict.
from l6_l8_matching.model import train_oof
from l6_l8_matching.calibration import (
    apply_isotonic,
    apply_serialized,
    brier,
    calibrate_grouped,
    ece,
    fit_isotonic,
    log_loss,
    serialize_isotonic,
)
from l6_l8_matching.pairs import (
    EASY,
    HARD,
    POSITIVE,
    SamplingConfig,
    iter_candidate_pairs,
    sample_reference_pairs,
    summarize,
    variant_b_rows,
)


def _record(**overrides) -> dict:
    """Minimal normalized record with sensible defaults for feature tests."""
    base = {
        "entity_id": "S1-x",
        "country_norm": "us",
        "name_core": "acme corporation",
        "name_norm": "acme corporation",
        "name_tokens": ["acme", "corporation"],
        "name_legal_suffix": "corporation",
        "addr_norm": "100 main street springfield il 62704",
        "addr_tokens": ["100", "main", "street", "springfield", "il", "62704"],
        "addr_house_number": "100",
        "addr_postal": "62704",
        "addr_state": "il",
        "addr_city": "springfield",
        "addr_digits": ["100", "62704"],
        "is_missing_addr": False,
    }
    base.update(overrides)
    return base


def test_iter_candidate_pairs() -> None:
    """The TSV reader skips the header and handles empty candidate fields."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "candidate_pairs.tsv"
        path.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-1\tS2-a,S2-b\n"
            "S1-2\t\n"
            "S1-3\tS3-z\n",
            encoding="utf-8",
        )
        rows = list(iter_candidate_pairs(path))
        assert rows == [("S1-1", ["S2-a", "S2-b"]), ("S1-2", []), ("S1-3", ["S3-z"])]


def test_sampling_ratio_for_matched_reference() -> None:
    """1 positive : 3 hard : 2 easy is honoured for a matched reference."""
    rng = random.Random(0)
    config = SamplingConfig(pos_hard=3, pos_easy=2, singleton_hard=3, singleton_easy=2)
    samples = sample_reference_pairs(
        "S1-1",
        candidate_ids=["S2-a", "S2-x", "S2-y", "S2-z"],
        ground_truth={"S2-a"},
        country="us",
        easy_pool=[f"S2-e{i}" for i in range(100)],
        rng=rng,
        config=config,
    )
    types = [s.neg_type for s in samples]
    assert types.count(POSITIVE) == 1
    assert types.count(HARD) == 3
    assert types.count(EASY) == 2
    # Hard negatives keep the ranked order.
    hard_ids = [s.cand_id for s in samples if s.neg_type == HARD]
    assert hard_ids == ["S2-x", "S2-y", "S2-z"]


def test_sampling_for_singleton() -> None:
    """A singleton still receives a fixed number of negatives."""
    rng = random.Random(1)
    config = SamplingConfig(pos_hard=3, pos_easy=2, singleton_hard=3, singleton_easy=2)
    samples = sample_reference_pairs(
        "S1-9",
        candidate_ids=["S2-x", "S2-y"],
        ground_truth=set(),
        country="india",
        easy_pool=[f"S3-e{i}" for i in range(50)],
        rng=rng,
        config=config,
    )
    assert sum(1 for s in samples if s.neg_type == POSITIVE) == 0
    assert sum(1 for s in samples if s.neg_type == HARD) == 2  # capped by availability
    assert sum(1 for s in samples if s.neg_type == EASY) == 2


def test_easy_negatives_exclude_candidates_and_truth() -> None:
    """Easy negatives must never coincide with a candidate or a true match."""
    rng = random.Random(2)
    candidates = ["S2-a", "S2-b"]
    truth = {"S2-a"}
    # Pool deliberately stuffed with the forbidden ids.
    pool = ["S2-a", "S2-b"] * 20 + ["S2-safe-1", "S2-safe-2", "S2-safe-3"]
    samples = sample_reference_pairs(
        "S1-1", candidates, truth, "us", pool, rng, SamplingConfig()
    )
    easy = [s.cand_id for s in samples if s.neg_type == EASY]
    assert easy
    assert not (set(easy) & (set(candidates) | truth))


def test_variant_b_removes_hard_negatives() -> None:
    """Variant B keeps positives + easy negatives only."""
    rng = random.Random(3)
    samples = sample_reference_pairs(
        "S1-1", ["S2-a", "S2-x"], {"S2-a"}, "us", [f"S2-e{i}" for i in range(20)], rng
    )
    b_rows = variant_b_rows(samples)
    assert all(s.neg_type != HARD for s in b_rows)
    assert any(s.neg_type == POSITIVE for s in b_rows)
    assert summarize(samples)["hard_negatives"] == 1
    assert summarize(b_rows)["hard_negatives"] == 0


def test_feature_vector_shape() -> None:
    """There are exactly 27 features and compute_features returns all of them."""
    assert len(FEATURE_NAMES) == 27
    features = compute_features(_record(), _record(), {"rank": 0}, build_idf({}, 10))
    assert len(features) == 27
    assert all(isinstance(v, float) for v in features)


def test_identical_records_high_similarity() -> None:
    """Identical records should score maximally on the key similarity features."""
    features = dict(zip(FEATURE_NAMES, compute_features(_record(), _record(), {"rank": 0}, build_idf({}, 1))))
    assert features["name_ratio"] == 1.0
    assert features["name_jaro_winkler"] == 1.0
    assert features["name_char3_jaccard"] == 1.0
    assert features["name_token_jaccard"] == 1.0
    assert features["name_legal_suffix_match"] == 1.0
    assert features["addr_house_exact"] == 1.0
    assert features["addr_postal_exact"] == 1.0
    assert features["addr_city_exact"] == 1.0
    assert features["addr_state_match"] == 1.0
    assert features["ret_exact_key_hit"] == 1.0
    assert features["struct_same_country"] == 1.0
    assert features["struct_addr_missing_xor"] == 0.0


def test_different_records_flag_conflicts() -> None:
    """Different addresses should surface digit conflict and no postal match."""
    other = _record(
        name_core="globex limited",
        name_tokens=["globex", "limited"],
        name_legal_suffix="limited",
        addr_norm="55 oak avenue boston ma 02108",
        addr_tokens=["55", "oak", "avenue", "boston", "ma", "02108"],
        addr_house_number="55",
        addr_postal="02108",
        addr_state="ma",
        addr_city="boston",
        addr_digits=["55", "02108"],
    )
    features = dict(zip(FEATURE_NAMES, compute_features(_record(), other, {"rank": 1}, build_idf({}, 1))))
    assert features["name_ratio"] < 0.6
    assert features["addr_postal_exact"] == 0.0
    assert features["addr_postal_prefix3"] == 0.0
    assert features["addr_digit_conflict"] == 1.0
    assert features["addr_state_match"] == 0.0
    assert features["struct_same_country"] == 1.0


def test_idf_overlap_weights_rare_tokens() -> None:
    """IDF overlap lies in [0,1] and unseen-token default is applied."""
    idf = build_idf({"acme": 5, "corporation": 100}, 200)
    assert "__default__" in idf
    features = dict(zip(FEATURE_NAMES, compute_features(_record(), _record(), {"rank": 0}, idf)))
    assert 0.0 <= features["name_idf_overlap"] <= 1.0


def test_retrieval_features_rank() -> None:
    """Retrieved/rank-inverse depend on the candidate rank."""
    easy = compute_retrieval_features(_record(), _record(), {"rank": -1})
    assert easy[3] == -1.0 and easy[4] == 0.0 and easy[5] == 0.0
    hard = compute_retrieval_features(_record(), _record(), {"rank": 2})
    assert hard[4] == 1.0
    assert abs(hard[5] - 1.0 / 3.0) < 1e-9
    assert hard[0] == 1.0  # exact key hit (identical name_core)


def test_train_oof_grouped() -> None:
    """Grouped OOF returns one probability per row and separates the classes."""
    rng = np.random.default_rng(0)
    rows = []
    for group in range(12):
        for j in range(30):
            positive = rng.random() < 0.5
            rows.append(
                {
                    "s1_id": f"S1-{group}",
                    "cand_id": f"S2-{group}-{j}",
                    "label": int(positive),
                    "neg_type": "pos" if positive else "hard",
                    "f1": float(rng.normal(1.5 if positive else -1.5, 0.5)),
                    "f2": float(rng.normal(1.5 if positive else -1.5, 0.5)),
                }
            )
    df = pd.DataFrame(rows)
    oof, models, importances = train_oof(df, ["f1", "f2"], n_folds=3, seed=0)
    assert len(oof) == len(df)
    assert ((oof >= 0.0) & (oof <= 1.0)).all()
    assert len(models) == 3
    assert set(importances) == {"f1", "f2"}
    # Strongly separable data: positive label should have larger mean probability.
    assert oof[df["label"].to_numpy() == 1].mean() > oof[df["label"].to_numpy() == 0].mean()


def test_reliability_metrics_basic() -> None:
    """Brier and log-loss match their closed-form values on simple inputs."""
    assert brier([1.0, 0.0], [1, 0]) == 0.0
    assert abs(brier([0.5, 0.5], [1, 0]) - 0.25) < 1e-12
    assert abs(log_loss([0.5, 0.5], [1, 0]) - 0.6931471805599453) < 1e-9


def test_ece_extremes() -> None:
    """Perfect predictions give ECE 0; confidently wrong predictions give high ECE."""
    assert ece(np.array([0.0, 1.0]), np.array([0, 1])) < 1e-9
    assert abs(ece(np.array([0.9, 0.9]), np.array([0, 0])) - 0.9) < 1e-9
    assert ece(np.array([]), np.array([])) == 0.0


def test_isotonic_monotonic_and_serialization() -> None:
    """Isotonic output is monotonic in [0,1] and survives serialization."""
    rng = np.random.default_rng(1)
    probs = rng.random(400)
    labels = (rng.random(400) < probs**0.4).astype(int)  # deliberately under-confident
    iso = fit_isotonic(probs, labels)
    preds = apply_isotonic(iso, probs)
    assert preds.min() >= 0.0 and preds.max() <= 1.0
    assert np.all(np.diff(iso.X_thresholds_) > 0)
    assert np.all(np.diff(iso.y_thresholds_) >= 0)
    knots = serialize_isotonic(iso)
    restored = apply_serialized(knots, probs)
    assert np.allclose(restored, preds, atol=1e-6)


def test_calibrate_grouped_range() -> None:
    """Grouped calibration returns one value per row, all within [0,1]."""
    rng = np.random.default_rng(2)
    probs = rng.random(240)
    labels = (rng.random(240) < probs).astype(int)
    groups = np.array([f"g{i % 12}" for i in range(240)])
    calibrated = calibrate_grouped(probs, labels, groups, n_folds=3, seed=0)
    assert len(calibrated) == len(probs)
    assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0


def run_all_tests() -> bool:
    """Run every test and report pass/fail."""
    tests = [
        ("Candidate TSV parsing", test_iter_candidate_pairs),
        ("Sampling ratio (matched)", test_sampling_ratio_for_matched_reference),
        ("Sampling for singleton", test_sampling_for_singleton),
        ("Easy negatives excluded", test_easy_negatives_exclude_candidates_and_truth),
        ("Variant B drops hard negatives", test_variant_b_removes_hard_negatives),
        ("Feature vector shape (27)", test_feature_vector_shape),
        ("Identical records high sim", test_identical_records_high_similarity),
        ("Different records flag conflicts", test_different_records_flag_conflicts),
        ("IDF overlap weights", test_idf_overlap_weights_rare_tokens),
        ("Retrieval rank features", test_retrieval_features_rank),
        ("L8 grouped OOF training", test_train_oof_grouped),
        ("L9 reliability metrics", test_reliability_metrics_basic),
        ("L9 ECE extremes", test_ece_extremes),
        ("L9 isotonic + serialization", test_isotonic_monotonic_and_serialization),
        ("L9 grouped calibration", test_calibrate_grouped_range),
    ]

    print("\n" + "=" * 60)
    print("🧪 RUNNING LAYER 6-9 UNIT TESTS")
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
