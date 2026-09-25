"""Data quality auditing, schema consistency, and ground truth cardinality verification."""

import json
from pathlib import Path
from typing import Any
import pandas as pd

from config import PATH_OUTPUT_DIR


def audit(df_dict: dict[str, Any], split_name: str) -> dict[str, Any]:
    """Compute and log data quality metrics, missing value rates, GT distributions, and country consistency."""
    report: dict[str, Any] = {
        "split": split_name,
        "files": {},
        "missing_pct": {},
        "country_distribution": {},
    }

    # Audit individual source files
    for src_name in ["s1", "s2", "s3"]:
        df: pd.DataFrame = df_dict[src_name]
        row_count = len(df)
        report["files"][src_name] = row_count
        report["country_distribution"][src_name] = df["country"].value_counts().to_dict()

        # Missing percentage per column (empty string)
        missing_dict: dict[str, float] = {}
        for col in df.columns:
            missing_count = (df[col].astype(str).str.strip() == "").sum()
            missing_dict[col] = float(round((missing_count / row_count) * 100, 4)) if row_count > 0 else 0.0
        report["missing_pct"][src_name] = missing_dict

    # Audit ground truth if present
    gt_df = df_dict.get("gt")
    if gt_df is not None:
        gt_rows = len(gt_df)
        report["files"]["gt"] = gt_rows

        match_lists = gt_df["matched_entity_ids"].astype(str).apply(
            lambda x: [m.strip() for m in x.split(",") if m.strip()]
        )
        match_counts = match_lists.apply(len)

        c0 = int((match_counts == 0).sum())
        c1 = int((match_counts == 1).sum())
        c2_plus = int((match_counts >= 2).sum())
        total_matches = int(match_counts.sum())
        singleton_rate = float(round(c0 / gt_rows, 6)) if gt_rows > 0 else 0.0

        report["ground_truth"] = {
            "total_records": gt_rows,
            "total_positive_pairs": total_matches,
            "singleton_count": c0,
            "singleton_rate": singleton_rate,
            "single_match_count": c1,
            "multi_match_count": c2_plus,
            "match_count_summary": {
                "0": c0,
                "1": c1,
                "2+": c2_plus,
            },
        }

        # Country Consistency Check
        s1_country_map = dict(zip(df_dict["s1"]["entity_id"], df_dict["s1"]["country"]))
        s2_country_map = dict(zip(df_dict["s2"]["entity_id"], df_dict["s2"]["country"]))
        s3_country_map = dict(zip(df_dict["s3"]["entity_id"], df_dict["s3"]["country"]))

        total_checked_pairs = 0
        mismatched_pairs = 0

        for s1_id, matches in zip(gt_df["source1_entity_id"], match_lists):
            s1_country = s1_country_map.get(s1_id)
            for m in matches:
                total_checked_pairs += 1
                cand_country = s2_country_map.get(m) if m.startswith("S2-") else s3_country_map.get(m)
                if cand_country is not None and s1_country != cand_country:
                    mismatched_pairs += 1

        mismatch_rate = float(round(mismatched_pairs / total_checked_pairs, 6)) if total_checked_pairs > 0 else 0.0
        report["ground_truth"]["country_consistency"] = {
            "pairs_checked": total_checked_pairs,
            "mismatched_pairs": mismatched_pairs,
            "mismatch_rate": mismatch_rate,
            "flag_high_mismatch": bool(mismatch_rate > 0.05),
        }

    # Save JSON report
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PATH_OUTPUT_DIR / f"audit_{split_name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print Console Summary
    print(f"\n{'=' * 60}")
    print(f"📊 DATA AUDIT SUMMARY: {split_name.upper()}")
    print(f"{'=' * 60}")
    print(f"File Row Counts: {report['files']}")
    print(f"Country Distribution: {report['country_distribution']}")
    print(f"Missing Field Rates (%): {report['missing_pct']}")

    if "ground_truth" in report:
        gt_rep = report["ground_truth"]
        print("\nGround Truth Linkage Stats:")
        print(f"  • Total S1 Records: {gt_rep['total_records']:,}")
        print(f"  • Total Positive Pairs: {gt_rep['total_positive_pairs']:,}")
        print(f"  • Singletons (0 matches): {gt_rep['singleton_count']:,} ({gt_rep['singleton_rate'] * 100:.2f}%)")
        print(f"  • 1 Match: {gt_rep['single_match_count']:,} | 2+ Matches: {gt_rep['multi_match_count']:,}")
        cc = gt_rep["country_consistency"]
        print(f"  • Country Mismatch Rate: {cc['mismatch_rate'] * 100:.4f}% ({cc['mismatched_pairs']} / {cc['pairs_checked']})")
        if cc["flag_high_mismatch"]:
            print("  ⚠️ WARNING: Country mismatch rate is above 5%!")
        else:
            print("  ✅ Country consistency check PASSED (clean partition).")

    print(f"Saved audit report: {out_path}")
    print(f"{'=' * 60}\n")
    return report