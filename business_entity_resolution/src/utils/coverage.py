"""Coverage preflight for the blocking chain (L3-L5) and inference (L11).

Silent coverage loss is the most dangerous failure mode in this pipeline:

* Layer 4/5 skipped a country whose upstream artifact was missing, printed a
  warning, and still wrote a candidate file covering only the countries it had;
* Layer 11 then passed ``all_references=True``, which backfilled every unseen
  Source-1 entity with an *empty* prediction.

The result was a format-valid submission that scores ~0 on everything it never
saw. These helpers turn that into an error that names the missing country, the
number of references at stake and the exact command that fixes it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

from config import L3_COUNTRIES, PATH_OUTPUT_DIR
from l3_l5_blocking.artifacts import l3_artifact_path, l4_artifact_path
from l3_l5_blocking.buckets import assign_country, iter_source_shards

ARTIFACT_LAYERS = {"l3": l3_artifact_path, "l4": l4_artifact_path}


class CoverageError(RuntimeError):
    """Raised when a run would silently produce an incomplete result."""


def parquet_rows(path: Path) -> int:
    """Row count of a parquet artifact from its footer (no data loaded)."""
    import pyarrow.parquet as pq

    try:
        return int(pq.ParquetFile(str(path)).metadata.num_rows)
    except Exception:  # noqa: BLE001 - an unreadable artifact counts as empty
        return 0


def scope_reference_ids(refs: str) -> Optional[set[str]]:
    """Return the reference-id filter for a run scope (``val``/``train``/``all``).

    ``None`` means "every Source 1 entity in the split".
    """
    if refs == "all":
        return None
    from l1_validation.split_generator import load_split_ids

    split_ids = load_split_ids()
    key = "val_ids" if refs == "val" else "train_ids"
    return set(split_ids[key])


def iter_reference_country_pairs(split: str, refs: str = "all", max_shards: int | None = None):
    """Yield ``(entity_id, country_bucket)`` for every in-scope Source 1 entity.

    Streaming, so callers can count, sample or audit millions of references without
    holding them all.
    """
    wanted = scope_reference_ids(refs)
    shards = iter_source_shards(split, 1)
    if max_shards is not None:
        shards = shards[:max_shards]
    for shard in shards:
        df = pd.read_parquet(shard, columns=["entity_id", "country_norm"])
        if wanted is not None:
            df = df[df["entity_id"].isin(wanted)]
        if df.empty:
            continue
        for entity_id, country in zip(df["entity_id"].tolist(), df["country_norm"].tolist()):
            yield entity_id, assign_country(str(country))


def count_references_by_country(
    split: str,
    refs: str = "all",
    max_shards: Optional[int] = None,
) -> Counter:
    """Count Source 1 entities per country bucket for a split/scope."""
    counts: Counter = Counter()
    for _, country in iter_reference_country_pairs(split, refs, max_shards):
        counts[country] += 1
    return counts


def planned_reference_counts(
    split: str,
    refs: str,
    countries: Sequence[str],
    report_path: Optional[Path] = None,
) -> dict[str, int]:
    """Per-country reference counts a downstream layer should expect.

    Honours a reference-sampled or capped Layer 3 run by reading what that run
    actually processed from the L3 report history; otherwise the counts come from
    the dataset itself. This is what lets the coverage guard distinguish "L3 ran on
    a deliberate sample" from "L3 produced a truncated artifact".
    """
    from utils.reports import load_json_report

    path = report_path or (PATH_OUTPUT_DIR / "l3_blocking_report.json")
    run = ((load_json_report(path) or {}).get("runs") or {}).get(f"{split}_{refs}") or {}
    config = run.get("config") or {}
    per_country = run.get("per_country") or {}
    deliberate_cap = config.get("ref_sample") is not None or config.get("max_refs") is not None
    if deliberate_cap and per_country:
        return {
            country: int((per_country.get(country) or {}).get("references") or 0)
            for country in countries
        }
    base = count_references_by_country(split, refs)
    return {country: int(base.get(country, 0)) for country in countries}


def count_candidates_by_country(split: str, max_shards: Optional[int] = None) -> Counter:
    """Count Source 2 + Source 3 records per country bucket for a split."""
    counts: Counter = Counter()
    for source_idx in (2, 3):
        shards = iter_source_shards(split, source_idx)
        if max_shards is not None:
            shards = shards[:max_shards]
        for shard in shards:
            df = pd.read_parquet(shard, columns=["country_norm"])
            counts.update(
                assign_country(str(country)) for country in df["country_norm"].tolist()
            )
    return counts


def missing_normalized_sources(split: str, sources: Sequence[int] = (1, 2, 3)) -> list[int]:
    """Return the source indices of a split that have no normalized shards."""
    return [idx for idx in sources if not iter_source_shards(split, idx)]


def check_normalized_shards(
    split: str,
    sources: Sequence[int] = (1, 2, 3),
    *,
    allow_missing: bool = False,
) -> dict:
    """Fail when a split is missing normalized shards for a required source."""
    missing = missing_normalized_sources(split, sources)
    report = {"split": split, "missing_sources": missing, "ok": not missing}
    if missing and not allow_missing:
        raise CoverageError(
            f"split '{split}' has no normalized shards for source(s) {missing}. "
            f"Run: python business_entity_resolution/src/main_l2.py"
        )
    return report


def normalize_countries(values: Sequence[str]) -> list[str]:
    """Canonicalise a ``--countries`` argument into an ordered, unique list.

    Accepts both space-separated (``--countries us india``) and comma-separated
    (``--countries us,india``) input, so a comma-joined value can no longer reach a
    guard as one bogus country name whose expected reference count is silently 0.
    """
    normalized: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            country = part.strip().lower()
            if country and country not in normalized:
                normalized.append(country)
    return normalized


def expected_artifact_paths(layer: str, split: str, refs: str, countries: Sequence[str]) -> dict[str, Path]:
    """Map each country bucket to the artifact path a layer is expected to have written."""
    if layer not in ARTIFACT_LAYERS:
        raise ValueError(f"unknown layer '{layer}' (expected one of {sorted(ARTIFACT_LAYERS)})")
    factory = ARTIFACT_LAYERS[layer]
    return {country: factory(split, refs, country) for country in countries}


def check_country_artifacts(
    layer: str,
    split: str,
    refs: str,
    countries: Sequence[str],
    *,
    allow_missing: bool = False,
    expected_references: Optional[Mapping[str, int]] = None,
    verify_counts: bool = False,
    min_coverage: float = 0.999,
    remediation: Optional[str] = None,
) -> dict:
    """Fail when a layer's per-country artifacts are missing or truncated.

    Two distinct failure modes are caught:

    * **missing** - no artifact for a country bucket at all (the original silent gap);
    * **truncated** - the artifact exists but covers far fewer references than it
      should, e.g. a smoke run written with ``--max-refs 2000`` left on disk.
      Enabled with ``verify_counts``.

    Args:
        layer: ``"l3"`` or ``"l4"``.
        split / refs / countries: the run's scope.
        allow_missing: keep going (and record the gap) instead of raising.
        expected_references: ``country -> n_references`` expectation
            (see :func:`planned_reference_counts`).
        verify_counts: compare artifact row counts against the expectation.
        min_coverage: minimum fraction of expected references per country.
        remediation: optional command hint shown in the error.

    Returns:
        ``{layer, split, refs, countries, present, missing, coverage, ok}``
    """
    countries = normalize_countries(countries)
    paths = expected_artifact_paths(layer, split, refs, countries)
    present = [country for country, path in paths.items() if path.exists()]
    absent = [country for country, path in paths.items() if not path.exists()]

    counts_known = expected_references is not None
    if expected_references is None:
        try:
            expected_references = count_references_by_country(split, refs)
            counts_known = True
        except Exception:  # noqa: BLE001 - the count is only for the error message
            expected_references = {}

    def _expected(country: str) -> Optional[int]:
        """Planned reference count, or ``None`` when the plan is unknown."""
        if not counts_known or expected_references is None or country not in expected_references:
            return None
        return int(expected_references[country])

    # A bucket the plan says has zero in-scope references (e.g. france on the train
    # split) cannot produce or need an artifact - it is out of scope, not missing.
    out_of_scope = [country for country in absent if _expected(country) == 0]
    missing = [country for country in absent if country not in out_of_scope]
    missing_references: dict[str, int] = {
        country: int(_expected(country) or 0) for country in missing
    }

    artifact_rows: dict[str, int] = {}
    coverage_by_country: dict[str, float] = {}
    truncated: list[str] = []
    for country in present:
        rows = parquet_rows(paths[country])
        artifact_rows[country] = rows
        expected = _expected(country)
        coverage_by_country[country] = (rows / expected) if expected else 1.0
        if verify_counts and expected and coverage_by_country[country] < min_coverage:
            truncated.append(country)

    report = {
        "layer": layer,
        "split": split,
        "refs": refs,
        "countries": list(countries),
        "present": present,
        "missing": missing,
        "out_of_scope": out_of_scope,
        "missing_references": missing_references,
        "artifact_rows": artifact_rows,
        "coverage": coverage_by_country,
        "truncated": truncated,
        "ok": not missing and not truncated,
    }

    if countries and not present and not missing and out_of_scope and not allow_missing:
        valid = ", ".join(L3_COUNTRIES)
        raise CoverageError(
            f"none of the requested country bucket(s) {', '.join(out_of_scope)} have any "
            f"in-scope references in the {split}/{refs} scope, so there is nothing to "
            f"block/score. This is usually a typo or the wrong split; buckets are "
            f"space- or comma-separated (e.g. --countries us india). Known buckets: "
            f"{valid}."
        )

    if truncated and not allow_missing:
        detail = ", ".join(
            f"{country}: {artifact_rows[country]:,}/"
            f"{int(_expected(country) or 0):,} references "
            f"({coverage_by_country[country] * 100:.1f}%)"
            for country in truncated
        )
        hint = remediation or (
            f"re-run main_l3.py --split {split} --refs {refs} for the affected buckets"
        )
        raise CoverageError(
            f"{layer.upper()} artifact(s) are truncated: {detail}. That looks like a "
            f"development cap (--max-refs / --max-candidates) left on disk; the rest of "
            f"those references would be submitted as empty predictions. Fix: {hint} — "
            f"or pass --allow-missing-countries to proceed anyway."
        )

    if missing and not allow_missing:
        at_risk = sum(missing_references.values())
        extra = (
            f" Also ignoring {', '.join(out_of_scope)}: no in-scope references "
            "(check the spelling / the split)."
            if out_of_scope
            else ""
        )
        detail = ", ".join(
            f"{country} (~{missing_references.get(country, 0):,} references)"
            for country in missing
        )
        hint = remediation or (
            f"run main_l3.py --split {split} --refs {refs} first"
            if layer == "l4"
            else f"run main_l4.py --split {split} --refs {refs} first"
        )
        raise CoverageError(
            f"missing {layer.upper()} artifact(s) for {detail}. Writing an output now "
            f"would cover only {present or 'nothing'} and leave ~{at_risk:,} references "
            f"unmatched (they would be emitted as empty predictions). Fix: {hint} "
            f"— or pass --allow-missing-countries to proceed anyway.{extra}"
        )

    if out_of_scope:
        report["note"] = (
            f"no in-scope references for {', '.join(out_of_scope)} in this scope; "
            "skipped (expected, not a coverage gap)"
        )
    return report


def check_reference_coverage(
    split: str,
    refs: str,
    present_ids: Iterable[str],
    *,
    allow_partial: bool = False,
    remediation: Optional[str] = None,
) -> dict:
    """Measure how much of a split's Source-1 set an output file actually covers.

    Args:
        split: ``train`` or ``test``.
        refs: run scope of the *reference* set (``all``/``val``/``train``).
        present_ids: reference ids the output file contains.
        allow_partial: record the gap instead of raising.
        remediation: optional command hint shown in the error.

    Returns:
        ``{expected_by_country, present, missing_by_country, expected_total,
        missing_total, coverage, ok}``
    """
    present = set(str(value) for value in present_ids)
    expected: Counter = Counter()
    missing: Counter = Counter()

    for entity_id, bucket in iter_reference_country_pairs(split, refs):
        expected[bucket] += 1
        if entity_id not in present:
            missing[bucket] += 1

    expected_total = sum(expected.values())
    missing_total = sum(missing.values())
    report = {
        "split": split,
        "refs": refs,
        "expected_by_country": dict(sorted(expected.items())),
        "missing_by_country": {key: value for key, value in sorted(missing.items()) if value},
        "expected_total": expected_total,
        "present": len(present),
        "missing_total": missing_total,
        "coverage": ((expected_total - missing_total) / expected_total) if expected_total else 1.0,
        "ok": missing_total == 0,
    }

    if missing_total and not allow_partial:
        detail = ", ".join(
            f"{country}: {count:,}" for country, count in sorted(missing.items()) if count
        )
        hint = remediation or "re-run the blocking chain for every country bucket"
        raise CoverageError(
            f"output covers {report['coverage'] * 100:.2f}% of the '{split}' Source-1 set "
            f"({missing_total:,} references missing -> {detail}). Those entities would be "
            f"submitted as empty predictions. Fix: {hint} — or pass "
            f"--allow-partial-coverage to proceed anyway."
        )
    return report
