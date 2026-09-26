"""Layer 1: Unit tests for macro F0.5 scoring, singleton penalties, and threshold evaluation."""

import sys
from pathlib import Path

# Add project root and src/ to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Support both `from src.l1_validation.metrics` and `from l1_validation.metrics`
try:
    from src.l1_validation.metrics import evaluate_with_thresholds, macro_f05
except ImportError:
    from l1_validation.metrics import evaluate_with_thresholds, macro_f05


def run_all_tests() -> bool:
    """Run all eight unit tests and return True if all pass, False otherwise."""
    tests = []

    # Test 1: Exact match returns 1.0
    def test_1():
        gt = {"S1-1": {"S2-1", "S3-1"}}
        pred = {"S1-1": ["S2-1", "S3-1"]}
        assert macro_f05(pred, gt)["macro_f05"] == 1.0

    tests.append(("Test 1 — Exact match returns 1.0", test_1))

    # Test 2: Singleton correctly empty returns 1.0
    def test_2():
        gt = {"S1-1": set()}
        pred = {"S1-1": []}
        assert macro_f05(pred, gt)["macro_f05"] == 1.0

    tests.append(("Test 2 — Singleton correctly empty returns 1.0", test_2))

    # Test 3: Singleton with false match returns 0.0
    def test_3():
        gt = {"S1-1": set()}
        pred = {"S1-1": ["S2-1"]}
        assert macro_f05(pred, gt)["macro_f05"] == 0.0

    tests.append(("Test 3 — Singleton with false match returns 0.0", test_3))

    # Test 4: Problem statement worked example
    def test_4():
        gt = {"S1-1": {"S2-47", "S3-812"}}
        pred = {"S1-1": ["S2-47", "S2-193", "S3-812"]}
        result = macro_f05(pred, gt)
        assert abs(result["macro_f05"] - 0.7142857) < 1e-5

    tests.append(("Test 4 — Problem statement worked example", test_4))

    # Test 5: Empty gt and empty pred do not crash
    def test_5():
        gt = {}
        pred = {}
        result = macro_f05(pred, gt)
        assert isinstance(result["macro_f05"], float)

    tests.append(("Test 5 — Empty gt and empty pred do not crash", test_5))

    # Test 6: Mixed singleton and match
    def test_6():
        gt = {"S1-1": set(), "S1-2": {"S2-1"}}
        pred = {"S1-1": [], "S1-2": ["S2-1"]}
        assert macro_f05(pred, gt)["macro_f05"] == 1.0

    tests.append(("Test 6 — Mixed singleton and match", test_6))

    # Test 7: evaluate_with_thresholds basic match
    def test_7():
        scores = {("S1-1", "S2-1"): 0.9, ("S1-1", "S2-2"): 0.3}
        gt = {"S1-1": {"S2-1"}}
        result = evaluate_with_thresholds(scores, gt, tau_match=0.7, tau_s=0.5)
        assert result["macro_f05"] == 1.0

    tests.append(("Test 7 — evaluate_with_thresholds basic match", test_7))

    # Test 8: evaluate_with_thresholds singleton path
    def test_8():
        scores = {("S1-1", "S2-1"): 0.4}
        gt = {"S1-1": set()}
        result = evaluate_with_thresholds(scores, gt, tau_match=0.7, tau_s=0.5)
        assert result["macro_f05"] == 1.0

    tests.append(("Test 8 — evaluate_with_thresholds singleton path", test_8))

    def test_9():
        # The official problem statement gives
        #   F_0.5 = (1.25 * P * R) / (0.25 * P + R)
        # and its worked example (pred 3 ids, gt 2 ids, 2 true positives) scores
        # 0.714. The closed form used by the per-entity scorer must agree exactly,
        # so a change to either formulation cannot silently drift apart.
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        gt = {"S2-00047", "S3-00812"}
        precision = len(pred & gt) / len(pred)
        recall = len(pred & gt) / len(gt)
        official = (1.25 * precision * recall) / (0.25 * precision + recall)
        closed_form = 5.0 * len(pred & gt) / (len(gt) + 4 * len(pred))
        assert abs(official - 0.714) < 0.0005, official
        assert abs(official - closed_form) < 1e-12

        result = macro_f05({"S1-00001": sorted(pred)}, {"S1-00001": gt})
        assert abs(result["macro_f05"] - 0.714) < 0.0005, result["macro_f05"]
        assert result["n_singletons"] == 0
        assert abs(result["match_f05"] - result["macro_f05"]) < 1e-12

    tests.append(("Test 9 — official F0.5 formula equivalence", test_9))

    def test_10():
        # Singletons score 1.0 when predicted empty and 0.0 otherwise, and they are
        # part of the macro average (a whole test set of France-like entities is
        # only scored correctly if this holds).
        gt = {"S1-1": set(), "S1-2": set(), "S1-3": {"S2-x"}}
        result = macro_f05({"S1-1": [], "S1-2": ["S2-y"], "S1-3": ["S2-x"]}, gt)
        assert result["n_singletons"] == 2
        assert result["singleton_f05"] == 0.5
        assert result["match_f05"] == 1.0
        assert abs(result["macro_f05"] - (1.0 + 0.0 + 1.0) / 3) < 1e-12

    tests.append(("Test 10 — singleton credit inside the macro average", test_10))

    all_passed = True
    print("\n" + "=" * 60)
    print("🧪 RUNNING LAYER 1 UNIT TESTS")
    print("=" * 60)
    for name, test_func in tests:
        try:
            test_func()
            print(f"  ✅ PASS: {name}")
        except Exception as e:
            print(f"  ❌ FAIL: {name} -> {e}")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print(f"🎉 ALL {len(tests)} UNIT TESTS PASSED!\n")
    else:
        print("💥 SOME UNIT TESTS FAILED!\n")
    return all_passed


if __name__ == "__main__":
    passed = run_all_tests()
    sys.exit(0 if passed else 1)
