"""Layer 10.5: unit tests for the diagnostic error analysis."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from l10_diagnostics.diagnostics import (  # noqa: E402
    aggregate,
    classify_entity,
    error_focus,
    france_spot_check,
    group_metrics,
    length_bucket,
    load_predictions_tsv,
    source_composition,
)


def test_classify_entity_classes():
    """Every error class is assigned with the correct F0.5."""
    assert classify_entity([], [])["error_class"] == "true_singleton"
    assert classify_entity([], [])["f05"] == 1.0

    fp = classify_entity(["S2-1"], [])
    assert fp["error_class"] == "false_positive" and fp["f05"] == 0.0

    miss = classify_entity([], ["S2-1"])
    assert miss["error_class"] == "miss" and miss["f05"] == 0.0

    exact = classify_entity(["S2-1", "S3-2"], {"S2-1", "S3-2"})
    assert exact["error_class"] == "exact" and exact["f05"] == 1.0

    partial = classify_entity(["S2-1", "S2-2"], {"S2-1"})
    assert partial["error_class"] == "partial"
    # precision 0.5, recall 1.0 -> F0.5 = 1.25*0.5*1 / (0.25*0.5 + 1)
    assert abs(partial["f05"] - (1.25 * 0.5 / 1.125)) < 1e-9, partial


def test_length_bucket_boundaries():
    """Length buckets use the documented ranges."""
    assert length_bucket(0) == "0"
    assert length_bucket(1) == "1-10"
    assert length_bucket(10) == "1-10"
    assert length_bucket(11) == "11-20"
    assert length_bucket(21) == "21-40"
    assert length_bucket(41) == "41+"


def test_source_composition():
    """Composition distinguishes S2, S3, mixed and empty."""
    assert source_composition([]) == "none"
    assert source_composition(["S2-1"]) == "s2"
    assert source_composition(["S3-1"]) == "s3"
    assert source_composition(["S2-1", "S3-2"]) == "s2+s3"


def test_aggregate_macro_and_counts():
    """Aggregation averages per-entity F0.5 and counts classes."""
    records = [
        {"error_class": "true_singleton", "f05": 1.0, "precision": 1.0, "recall": 1.0},
        {"error_class": "false_positive", "f05": 0.0, "precision": 0.0, "recall": 0.0},
        {"error_class": "miss", "f05": 0.0, "precision": 0.0, "recall": 0.0},
        {"error_class": "exact", "f05": 1.0, "precision": 1.0, "recall": 1.0},
    ]
    stats = aggregate(records)
    assert stats["n"] == 4
    assert abs(stats["macro_f05"] - 0.5) < 1e-9, stats
    assert stats["error_classes"]["true_singleton"] == 1
    assert stats["error_classes"]["false_positive"] == 1
    assert stats["error_classes"]["exact"] == 1
    assert stats["n_singletons"] == 1


def test_group_metrics_orders_by_size():
    """Groups are returned largest-first."""
    records = [
        {"country": "us", "error_class": "exact", "f05": 1.0, "precision": 1.0, "recall": 1.0},
        {"country": "us", "error_class": "miss", "f05": 0.0, "precision": 0.0, "recall": 0.0},
        {"country": "india", "error_class": "exact", "f05": 1.0, "precision": 1.0, "recall": 1.0},
    ]
    grouped = group_metrics(records, "country")
    assert list(grouped) == ["us", "india"], grouped
    assert grouped["us"]["n"] == 2 and grouped["india"]["n"] == 1


def test_error_focus_mass():
    """Error focus reports counts, fractions and F0.5 mass per class."""
    records = [
        {"error_class": "exact", "f05": 1.0},
        {"error_class": "miss", "f05": 0.0},
    ]
    focus = error_focus(records)
    assert focus["counts"]["exact"] == 1
    assert abs(focus["fraction"]["exact"] - 0.5) < 1e-9
    assert abs(focus["f05_mass"]["exact"] - 1.0) < 1e-9


def test_load_predictions_tsv_roundtrip():
    """The TSV loader keeps empty rows as empty lists."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "matching_results.tsv"
        path.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-a\tS2-1,S3-2\n"
            "S1-b\t\n",
            encoding="utf-8",
        )
        preds = load_predictions_tsv(path)
        assert preds == {"S1-a": ["S2-1", "S3-2"], "S1-b": []}, preds


def test_france_spot_check_summary():
    """Spot-check summarises volume and orders samples by top score."""
    records = [
        {"s1_id": "S1-a", "pred": ["S2-1"], "top_score": 0.4, "meta": {"name_core": "alpha"}},
        {"s1_id": "S1-b", "pred": [], "top_score": 0.2, "meta": {}},
        {"s1_id": "S1-c", "pred": ["S2-9", "S3-8"], "top_score": 0.9,
         "meta": {"name_core": "gamma", "is_missing_addr": True}},
    ]
    spot = france_spot_check(records, top_n=2)
    assert spot["n"] == 3
    assert spot["n_non_empty"] == 2 and spot["n_empty"] == 1
    assert abs(spot["mean_pred_len"] - 1.0) < 1e-9
    # Highest-confidence non-empty sample first.
    assert [s["s1_id"] for s in spot["samples"]] == ["S1-c", "S1-a"], spot["samples"]
    assert spot["top_score_quantiles"]["p100"] == 0.9


def run_all_tests() -> bool:
    """Run the L10.5 test suite and report pass/fail."""
    tests = [
        ("Classify entity classes", test_classify_entity_classes),
        ("Length bucket boundaries", test_length_bucket_boundaries),
        ("Source composition", test_source_composition),
        ("Aggregate macro + counts", test_aggregate_macro_and_counts),
        ("Group metrics ordering", test_group_metrics_orders_by_size),
        ("Error focus mass", test_error_focus_mass),
        ("Predictions TSV roundtrip", test_load_predictions_tsv_roundtrip),
        ("France spot-check summary", test_france_spot_check_summary),
    ]

    print("\n" + "=" * 60)
    print("🧪 RUNNING LAYER 10.5 UNIT TESTS")
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
