"""Layer 11: unit tests for the scorer and the streaming inference engine."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from l11_inference import inference  # noqa: E402
from l11_inference.scorer import calibrate_scores, load_booster, load_calibrator, score_matrix  # noqa: E402


class _FakeBooster:
    """Minimal booster stand-in: returns preset probabilities in row order."""

    def __init__(self, probs):
        self.probs = list(probs)

    def predict(self, features):
        n = len(features)
        return np.asarray(self.probs[:n], dtype=np.float64)


def _s1_record(entity_id: str, country: str, name: str) -> dict:
    return {
        "entity_id": entity_id,
        "country_norm": country,
        "name_core": name,
        "name_norm": name,
        "name_tokens": name.split(),
        "name_legal_suffix": "",
        "addr_norm": "100 main st springfield il 62704",
        "addr_tokens": ["100", "main", "st", "springfield", "il", "62704"],
        "addr_house_number": "100",
        "addr_postal": "62704",
        "addr_state": "il",
        "addr_city": "springfield",
        "addr_digits": ["100"],
        "is_missing_addr": False,
    }


def test_read_candidate_pairs_parsing():
    """Candidate TSV parsing keeps rows with empty candidate lists."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate_pairs.tsv"
        path.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-a\tS2-1,S3-2\n"
            "S1-b\t\n",
            encoding="utf-8",
        )
        refs, cand_map = inference.read_candidate_pairs(path)
        assert refs == ["S1-a", "S1-b"], refs
        assert cand_map == {"S1-a": ["S2-1", "S3-2"], "S1-b": []}, cand_map


def test_write_predictions_and_candidates_format():
    """Both submission files use the tab format with one row per reference."""
    with tempfile.TemporaryDirectory() as tmp:
        match_path = Path(tmp) / "matching_results.tsv"
        cand_path = Path(tmp) / "candidate_pairs.tsv"
        preds = {"S1-a": ["S2-1"], "S1-b": []}
        cands = {"S1-a": ["S2-1", "S3-2"], "S1-b": []}
        assert inference.write_predictions(match_path, preds, ["S1-a", "S1-b"]) == 2
        assert inference.write_candidates(cand_path, cands, ["S1-a", "S1-b"]) == 2
        assert match_path.read_text(encoding="utf-8").splitlines() == [
            "source1_entity_id\tmatched_entity_ids",
            "S1-a\tS2-1",
            "S1-b\t",
        ]
        assert cand_path.read_text(encoding="utf-8").splitlines() == [
            "source1_entity_id\tcandidate_entity_ids",
            "S1-a\tS2-1,S3-2",
            "S1-b\t",
        ]


def test_score_matrix_shapes_and_empty():
    """score_matrix returns probabilities and handles the empty case."""
    booster = _FakeBooster([0.7, 0.2])
    out = score_matrix(booster, np.zeros((2, 27), dtype=np.float32))
    assert list(out) == [0.7, 0.2], out
    assert score_matrix(booster, np.zeros((0, 27), dtype=np.float32)).shape == (0,)


def test_load_booster_missing_file():
    """A missing model file raises a clear error."""
    try:
        load_booster("does-not-exist.txt")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_load_calibrator_semantics():
    """chosen='none' disables calibration; global/per_country return knots."""
    with tempfile.TemporaryDirectory() as tmp:
        none_path = Path(tmp) / "none.json"
        none_path.write_text(json.dumps({"chosen": "none"}), encoding="utf-8")
        assert load_calibrator(none_path) is None

        global_path = Path(tmp) / "global.json"
        global_path.write_text(
            json.dumps({"chosen": "global", "global": {"x_thresholds": [0, 1], "y_thresholds": [0, 1]}}),
            encoding="utf-8",
        )
        loaded = load_calibrator(global_path)
        assert loaded["mode"] == "global"

        assert load_calibrator(Path(tmp) / "absent.json") is None
        assert load_calibrator(None) is None


def test_calibrate_scores_global_and_per_country():
    """Calibration applies globally or per country and is identity when None."""
    probs = np.array([0.0, 0.5, 1.0])
    assert np.allclose(calibrate_scores(probs, None), probs)

    identity = {"mode": "global", "knots": {"x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 1.0]}}
    assert np.allclose(calibrate_scores(probs, identity), probs)

    per_country = {
        "mode": "per_country",
        "knots": {"france": {"x_thresholds": [0.0, 1.0], "y_thresholds": [0.5, 0.5]}},
    }
    countries = ["france", "us", "us"]
    out = calibrate_scores(probs, per_country, countries)
    assert abs(out[0] - 0.5) < 1e-9, out  # france rows flattened
    assert abs(out[2] - 1.0) < 1e-9, out  # us rows untouched


def test_run_inference_end_to_end():
    """Streaming inference applies singleton guard and open-set veto."""
    with tempfile.TemporaryDirectory() as tmp:
        pair_path = Path(tmp) / "candidate_pairs.tsv"
        pair_path.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-a\tS2-1,S2-2\n"
            "S1-b\tS2-3\n",
            encoding="utf-8",
        )

        s1_records = {"S1-a": _s1_record("S1-a", "us", "alpha"), "S1-b": _s1_record("S1-b", "france", "beta")}
        cand_records = {f"S2-{i}": _s1_record(f"S2-{i}", "us", f"cand {i}") for i in (1, 2, 3)}

        # Patch the IO boundaries; the decision logic stays real. Restore after.
        originals = (inference.build_inference_idf, inference.load_records, inference.load_reference_countries)
        inference.build_inference_idf = lambda *a, **k: {}
        inference.load_reference_countries = lambda split, ids: {"S1-a": "us", "S1-b": "france"}

        def fake_load_records(split, sources, ids, max_shards=None):
            if tuple(sources) == (1,):
                return {r: s1_records[r] for r in ids if r in s1_records}
            return {c: cand_records[c] for c in ids if c in cand_records}

        inference.load_records = fake_load_records
        try:
            # Row order: S1-a->S2-1, S1-a->S2-2, S1-b->S2-3.
            booster = _FakeBooster([0.95, 0.40, 0.40])
            predictions, stats = inference.run_inference(
                split="test",
                candidate_pairs_path=pair_path,
                booster=booster,
                idf={},
                tau_match=0.2,
                tau_s=0.3,
                margin=0.05,
                open_set_countries=("france",),
                seen_countries=("us",),
                veto_min_confidence=0.5,
                all_references=False,
                ref_batch_size=10,
                progress=lambda *a, **k: None,
            )
        finally:
            (
                inference.build_inference_idf,
                inference.load_records,
                inference.load_reference_countries,
            ) = originals

        assert predictions["S1-a"] == ["S2-1"], predictions  # margin keeps only the top
        assert predictions["S1-b"] == [], predictions  # open-set veto (0.40 < 0.50)
        assert stats["non_empty"] == 1 and stats["matched_pairs"] == 1, stats
        assert stats["pairs_scored"] == 3, stats


def test_run_inference_signals_and_score_table():
    """The sidecar is read in lockstep and the per-pair score table is written."""
    from l6_l8_matching.signals import MODE_SIDECAR, SignalWriter

    with tempfile.TemporaryDirectory() as tmp:
        pair_path = Path(tmp) / "candidate_pairs.tsv"
        pair_path.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-a\tS2-1,S2-2\n"
            "S1-b\tS2-3\n",
            encoding="utf-8",
        )
        signals_path = Path(tmp) / "signals.parquet"
        writer = SignalWriter(signals_path, chunk=2)
        writer.add("S1-a", ["S2-1", "S2-2"], [0.5, 0.25], [1.0, 0.5])
        writer.add("S1-b", ["S2-3"], [0.1], [0.3333])
        writer.close()
        scores_out = Path(tmp) / "scores.parquet"

        s1_records = {"S1-a": _s1_record("S1-a", "us", "alpha"), "S1-b": _s1_record("S1-b", "us", "beta")}
        cand_records = {f"S2-{i}": _s1_record(f"S2-{i}", "us", f"cand {i}") for i in (1, 2, 3)}

        originals = (inference.build_inference_idf, inference.load_records, inference.load_reference_countries)
        inference.build_inference_idf = lambda *a, **k: {}
        inference.load_reference_countries = lambda split, ids: {key: "us" for key in ids}

        def fake_load_records(split, sources, ids, max_shards=None):
            if tuple(sources) == (1,):
                return {r: s1_records[r] for r in ids if r in s1_records}
            return {c: cand_records[c] for c in ids if c in cand_records}

        inference.load_records = fake_load_records
        try:
            predictions, stats = inference.run_inference(
                split="test",
                candidate_pairs_path=pair_path,
                booster=_FakeBooster([0.95, 0.40, 0.40]),
                idf={},
                tau_match=0.2,
                tau_s=0.3,
                margin=0.05,
                all_references=False,
                ref_batch_size=10,
                signals_path=signals_path,
                scores_out=scores_out,
                progress=lambda *a, **k: None,
            )

            # A sidecar that does not line up with the candidate file must fail loudly.
            bad_path = Path(tmp) / "misaligned.parquet"
            bad = SignalWriter(bad_path)
            bad.add("S1-b", ["S2-3"], [0.1], [0.33])
            bad.close()
            try:
                inference.run_inference(
                    split="test",
                    candidate_pairs_path=pair_path,
                    booster=_FakeBooster([0.95, 0.40, 0.40]),
                    idf={},
                    tau_match=0.2,
                    tau_s=0.3,
                    all_references=False,
                    ref_batch_size=10,
                    signals_path=bad_path,
                    progress=lambda *a, **k: None,
                )
                raise AssertionError("expected a signal misalignment error")
            except ValueError as exc:
                assert "out of sync" in str(exc)
        finally:
            (
                inference.build_inference_idf,
                inference.load_records,
                inference.load_reference_countries,
            ) = originals

        assert stats["signals_mode"] == MODE_SIDECAR
        assert stats["signals_rows"] == 3
        assert predictions["S1-a"] == ["S2-1"]
        # Every candidate at/above tau_match is stored (3 of 3 here). Candidates
        # below it are dropped, which is lossless: the L10 margin rule can never
        # keep a candidate below tau_match, so main_l10 can recompute the exact
        # same decision from this table.
        assert stats["scores_written"] == 3
        assert scores_out.exists()
        import pandas as pd

        scores = pd.read_parquet(scores_out)
        assert list(scores.columns) == ["s1_id", "cand_id", "prob", "country"]
        assert scores.iloc[0]["s1_id"] == "S1-a"
        assert scores.iloc[0]["cand_id"] == "S2-1"
        assert abs(float(scores.iloc[0]["prob"]) - 0.95) < 1e-6
        assert scores.iloc[0]["country"] == "us"


def run_all_tests() -> bool:
    """Run the L11 test suite and report pass/fail."""
    tests = [
        ("Read candidate pairs", test_read_candidate_pairs_parsing),
        ("Write submission TSVs", test_write_predictions_and_candidates_format),
        ("score_matrix shapes", test_score_matrix_shapes_and_empty),
        ("Booster missing file", test_load_booster_missing_file),
        ("Calibrator load semantics", test_load_calibrator_semantics),
        ("Calibrate global/per-country", test_calibrate_scores_global_and_per_country),
        ("Streaming inference end-to-end", test_run_inference_end_to_end),
        ("Signals lockstep + score table", test_run_inference_signals_and_score_table),
    ]

    print("\n" + "=" * 60)
    print("🧪 RUNNING LAYER 11 UNIT TESTS")
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
