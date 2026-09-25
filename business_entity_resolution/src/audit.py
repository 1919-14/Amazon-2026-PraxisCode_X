"""Stream the provided TSV files without loading the full dataset into RAM."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


RECORD_FIELDS = ["entity_id", "business_name", "business_address", "country"]
LABEL_FIELDS = ["source1_entity_id", "matched_entity_ids"]


def audit_file(path: Path, limit: int = 0) -> dict[str, Any]:
    """Count records, missing fields, scripts, countries, and label cardinalities."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    is_labels = path.name == "train_ground_truth.tsv"
    expected_fields = LABEL_FIELDS if is_labels else RECORD_FIELDS
    countries: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    cardinality: Counter[int] = Counter()
    prefix_counts: Counter[str] = Counter()
    non_ascii: Counter[str] = Counter()
    records = 0
    bad_prefixes = 0
    repeated_ids_in_lists = 0
    expected_prefix = "S1-" if is_labels else "S" + path.stem[-1] + "-"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected_fields:
            raise ValueError(f"{path}: unexpected header {reader.fieldnames}")
        for record in reader:
            if None in record or any(value is None for value in record.values()):
                raise ValueError(f"{path}: malformed record near line {reader.line_num}")
            records += 1
            for field in expected_fields:
                if not record[field].strip():
                    missing[field] += 1
            reference = record[expected_fields[0]]
            bad_prefixes += not reference.startswith(expected_prefix)
            if is_labels:
                matches = record["matched_entity_ids"].split(",") if record["matched_entity_ids"] else []
                cardinality[len(matches)] += 1
                repeated_ids_in_lists += len(matches) != len(set(matches))
                for match in matches:
                    prefix_counts[match.split("-", 1)[0]] += 1
            else:
                countries[record["country"]] += 1
                for field in ("business_name", "business_address"):
                    non_ascii[field] += not record[field].isascii()
            if limit and records >= limit:
                break
    result: dict[str, Any] = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "records_scanned": records,
        "scan_limit": limit or None,
        "missing_or_blank": dict(missing),
        "bad_reference_prefixes": bad_prefixes,
    }
    if is_labels:
        result.update({
            "match_count_histogram": dict(sorted(cardinality.items())),
            "positive_pairs": sum(count * frequency for count, frequency in cardinality.items()),
            "singleton_fraction": cardinality[0] / records if records else None,
            "all_empty_macro_f05": cardinality[0] / records if records else None,
            "target_prefix_counts": dict(prefix_counts),
            "rows_with_duplicate_match_ids": repeated_ids_in_lists,
        })
    else:
        result.update({"countries": dict(countries), "non_ascii_rows": dict(non_ascii)})
    return result


def main() -> None:
    """Write a reproducible JSON profile; report progress on stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Rows per file; 0 scans all rows")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    results = []
    for split in ("train", "test"):
        names = [f"{split}_source{source}.tsv" for source in (1, 2, 3)]
        if split == "train":
            names.append("train_ground_truth.tsv")
        for name in names:
            print(f"Scanning {split}/{name}", file=sys.stderr, flush=True)
            results.append(audit_file(args.data_dir / split / name, args.limit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "scope": "Full scan" if not args.limit else "Prefix sample, not a random sample",
        "limitations": "Does not check global ID uniqueness, label referential integrity, or train/test overlap.",
        "files": results,
    }, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()