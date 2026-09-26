"""Unit tests for the shared utilities: report history + coverage preflight."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pyarrow as pa
import pyarrow.parquet as pq

from utils import coverage
from utils.coverage import (
    CoverageError,
    check_country_artifacts,
    check_reference_coverage,
    normalize_countries,
    parquet_rows,
)
from utils.reports import load_json_report, merge_json_report, render_markdown_table


def test_merge_json_report_keeps_history() -> None:
    """Each run is appended under `runs` while the newest payload stays on top."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.json"
        merge_json_report(path, "train_val", {"config": {"split": "train"}, "n": 1})
        merged = merge_json_report(path, "test_all", {"config": {"split": "test"}, "n": 2})

        assert merged["latest_run"] == "test_all"
        assert merged["run_keys"] == ["test_all", "train_val"]
        assert merged["runs"]["train_val"]["n"] == 1
        # The newest payload is mirrored at the top level for existing readers.
        assert merged["config"]["split"] == "test"
        assert merged["n"] == 2
        reloaded = load_json_report(path)
        assert reloaded is not None and set(reloaded["runs"]) == {"train_val", "test_all"}
        assert reloaded["runs"]["train_val"]["config"]["split"] == "train"


def test_merge_json_report_survives_unreadable_file() -> None:
    """A corrupt report is replaced rather than crashing the run."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.json"
        path.write_text("{not json", encoding="utf-8")
        merged = merge_json_report(path, "run", {"ok": True})
        assert merged["runs"]["run"]["ok"] is True


def test_render_markdown_table_marks_latest() -> None:
    """The markdown history table renders one row per run and flags the latest."""
    runs = {
        "a": {"config": {"split": "train"}, "n_references": 10},
        "b": {"config": {"split": "test"}, "n_references": 20},
    }
    text = render_markdown_table(
        runs,
        {"split": "split", "references": "n_references"},
        lambda key, run: {"split": (run.get("config") or {}).get("split")},
        latest="b",
    )
    assert "| run | split | references |" in text
    assert "`a` | train | 10" in text
    assert "`b` (latest) | test | 20" in text


def _write_artifact(path: Path, rows: int) -> None:
    """Write a minimal parquet artifact with ``rows`` rows."""
    schema = pa.schema([("source1_entity_id", pa.string()), ("country", pa.string())])
    table = pa.table(
        {
            "source1_entity_id": [f"S1-{i}" for i in range(rows)],
            "country": ["us"] * rows,
        },
        schema=schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_check_country_artifacts_missing_raises() -> None:
    """A missing country artifact is a hard error that names the references at risk."""
    with tempfile.TemporaryDirectory() as tmp:
        original = coverage.expected_artifact_paths
        coverage.expected_artifact_paths = lambda layer, split, refs, countries: {
            "us": Path(tmp) / "l4_us.parquet",
            "india": Path(tmp) / "l4_india.parquet",
        }
        # 'us' exists and is complete; only 'india' is missing.
        _write_artifact(Path(tmp) / "l4_us.parquet", rows=100)
        try:
            try:
                check_country_artifacts(
                    "l4",
                    "test",
                    "all",
                    ["us", "india"],
                    expected_references={"us": 100, "india": 250},
                )
                raise AssertionError("expected CoverageError")
            except CoverageError as exc:
                assert "india" in str(exc) and "250" in str(exc)
                assert "--allow-missing-countries" in str(exc)

            report = check_country_artifacts(
                "l4",
                "test",
                "all",
                ["us", "india"],
                allow_missing=True,
                expected_references={"us": 100, "india": 250},
            )
            assert report["ok"] is False
            assert report["missing"] == ["india"]
        finally:
            coverage.expected_artifact_paths = original


def test_check_country_artifacts_detects_truncated_artifact() -> None:
    """An artifact left behind by a capped smoke run is rejected, not trusted."""
    with tempfile.TemporaryDirectory() as tmp:
        original = coverage.expected_artifact_paths
        artifact = Path(tmp) / "l3_us.parquet"
        _write_artifact(artifact, rows=3)
        coverage.expected_artifact_paths = lambda layer, split, refs, countries: {"us": artifact}
        try:
            try:
                check_country_artifacts(
                    "l3",
                    "train",
                    "val",
                    ["us"],
                    expected_references={"us": 1000},
                    verify_counts=True,
                )
                raise AssertionError("expected CoverageError")
            except CoverageError as exc:
                assert "truncated" in str(exc)
                assert "0.3%" in str(exc)

            report = check_country_artifacts(
                "l3",
                "train",
                "val",
                ["us"],
                allow_missing=True,
                expected_references={"us": 1000},
                verify_counts=True,
            )
            assert report["truncated"] == ["us"]
            assert report["artifact_rows"] == {"us": 3}
        finally:
            coverage.expected_artifact_paths = original


def test_normalize_countries_accepts_both_syntaxes() -> None:
    """`--countries us,india` must not become one bogus bucket name."""
    assert normalize_countries(["us", "india"]) == ["us", "india"]
    assert normalize_countries(["us,india"]) == ["us", "india"]
    assert normalize_countries([" US , India ", "us"]) == ["us", "india"]
    assert normalize_countries([]) == []


def test_check_country_artifacts_handles_out_of_scope_bucket() -> None:
    """A bucket with zero in-scope references is skipped, not treated as a gap."""
    with tempfile.TemporaryDirectory() as tmp:
        original = coverage.expected_artifact_paths
        coverage.expected_artifact_paths = lambda layer, split, refs, countries: {
            country: Path(tmp) / f"l4_{country}.parquet" for country in countries
        }
        _write_artifact(Path(tmp) / "l4_us.parquet", rows=100)
        try:
            report = check_country_artifacts(
                "l4",
                "train",
                "val",
                ["us", "france"],
                expected_references={"us": 100, "france": 0},
            )
            assert report["ok"] is True
            assert report["missing"] == []
            assert report["out_of_scope"] == ["france"]
            assert "france" in report["note"]

            # A typo mixed with a genuinely missing bucket is still surfaced.
            try:
                check_country_artifacts(
                    "l4",
                    "train",
                    "val",
                    ["india", "usa"],
                    expected_references={"india": 100, "usa": 0},
                )
                raise AssertionError("expected CoverageError")
            except CoverageError as exc:
                assert "missing L4 artifact(s)" in str(exc)
                assert "india" in str(exc)
                assert "usa" in str(exc)

            # Nothing in scope at all means the run itself is misconfigured.
            try:
                check_country_artifacts(
                    "l4",
                    "train",
                    "val",
                    ["usa", "uk"],
                    expected_references={"usa": 0, "uk": 0},
                )
                raise AssertionError("expected CoverageError")
            except CoverageError as exc:
                assert "nothing to" in str(exc)
                assert "usa" in str(exc) and "--countries us india" in str(exc)
        finally:
            coverage.expected_artifact_paths = original


def test_parquet_rows_reads_footer() -> None:
    """row counts come from the footer, so no data is loaded."""
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "x.parquet"
        _write_artifact(artifact, rows=7)
        assert parquet_rows(artifact) == 7
        assert parquet_rows(Path(tmp) / "missing.parquet") == 0


def test_check_reference_coverage_reports_gaps() -> None:
    """Coverage is measured per country against the split's Source-1 set."""
    original = coverage.iter_reference_country_pairs
    coverage.iter_reference_country_pairs = lambda split, refs, max_shards=None: iter(
        [
            ("S1-1", "us"),
            ("S1-2", "us"),
            ("S1-3", "india"),
            ("S1-4", "france"),
        ]
    )
    try:
        try:
            check_reference_coverage("test", "all", ["S1-1", "S1-2"])
            raise AssertionError("expected CoverageError")
        except CoverageError as exc:
            assert "50.00%" in str(exc)
            assert "india" in str(exc) and "france" in str(exc)
            assert "--allow-partial-coverage" in str(exc)

        report = check_reference_coverage(
            "test", "all", ["S1-1", "S1-2", "S1-3"], allow_partial=True
        )
        assert report["expected_total"] == 4
        assert report["missing_total"] == 1
        assert report["missing_by_country"] == {"france": 1}
        assert abs(report["coverage"] - 0.75) < 1e-9

        full = check_reference_coverage(
            "test", "all", ["S1-1", "S1-2", "S1-3", "S1-4"], allow_partial=False
        )
        assert full["ok"] is True and full["missing_total"] == 0
    finally:
        coverage.iter_reference_country_pairs = original


def test_planned_reference_counts_honours_sampled_run() -> None:
    """A deliberately sampled L3 run is not mistaken for a truncated artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "l3_blocking_report.json"
        merge_json_report(
            report_path,
            "train_val",
            {
                "config": {"split": "train", "refs": "val", "ref_sample": 5000, "max_refs": None},
                "per_country": {"us": {"references": 4000}, "india": {"references": 1000}},
            },
        )
        planned = coverage.planned_reference_counts("train", "val", ["us", "india"], report_path)
        assert planned == {"us": 4000, "india": 1000}


def run_all_tests() -> bool:
    """Run the utility test suite and report pass/fail."""
    tests = [
        ("Report history merge", test_merge_json_report_keeps_history),
        ("Report survives corrupt file", test_merge_json_report_survives_unreadable_file),
        ("Markdown history table", test_render_markdown_table_marks_latest),
        ("Missing country artifact raises", test_check_country_artifacts_missing_raises),
        ("Truncated artifact detected", test_check_country_artifacts_detects_truncated_artifact),
        ("Country argument normalisation", test_normalize_countries_accepts_both_syntaxes),
        ("Out-of-scope country bucket", test_check_country_artifacts_handles_out_of_scope_bucket),
        ("Parquet footer row count", test_parquet_rows_reads_footer),
        ("Reference coverage gaps", test_check_reference_coverage_reports_gaps),
        ("Planned counts honour sampling", test_planned_reference_counts_honours_sampled_run),
    ]

    print("\n" + "=" * 60)
    print("🧪 RUNNING UTILS UNIT TESTS")
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
