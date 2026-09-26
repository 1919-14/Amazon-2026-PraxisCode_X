"""Layer 10.5 entry point: diagnostic error analysis & open-set spot-check.

Explains the Layer 10 decision's loss:

  * per-entity error classes (true singleton / false positive / miss / exact /
    partial) with their share of count and F0.5 mass;
  * breakdowns by **country**, **matched-source composition** (S2 / S3 / both),
    **name length**, **address length / missing-address**, and **script**;
  * an open-set (France) **spot-check** — volume, confidence quantiles and the
    top-scoring predictions for manual review, since no ground truth exists.

    output/l10_diagnostics_report.json / .md

Usage
-----
python business_entity_resolution/src/main_l10_diagnostics.py --split train
python business_entity_resolution/src/main_l10_diagnostics.py --split test \\
    --predictions output/matching_results.tsv --scores artifacts/scores/test_scores.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for _p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from config import (
    L10_OPEN_SET_COUNTRIES,
    L10_SEEN_COUNTRIES,
    PATH_OUTPUT_DIR,
)
from l10_diagnostics.diagnostics import (
    aggregate,
    classify_entity,
    error_focus,
    france_spot_check,
    group_metrics,
    length_bucket,
    load_predictions_tsv,
    load_reference_metadata,
    source_composition,
)

from l3_l5_blocking.buckets import assign_country
from main_l3 import load_ground_truth

try:
    from src.l1_validation.metrics import macro_f05 as official_macro_f05
except ImportError:  # pragma: no cover
    from l1_validation.metrics import macro_f05 as official_macro_f05


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Layer 10.5 run."""
    parser = argparse.ArgumentParser(description="Layer 10.5 diagnostic error analysis")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Decision TSV (default: matching_results_val.tsv on train, matching_results.tsv on test).",
    )
    parser.add_argument("--scores", type=str, default=None, help="Optional score parquet for top scores.")
    parser.add_argument("--prob-col", type=str, default="oof_prob")
    parser.add_argument("--top-n-france", type=int, default=10)
    parser.add_argument("--top-groups", type=int, default=25)
    parser.add_argument("--seen-countries", nargs="+", default=list(L10_SEEN_COUNTRIES))
    parser.add_argument("--open-set-countries", nargs="+", default=list(L10_OPEN_SET_COUNTRIES))
    return parser.parse_args()


def default_predictions_path(split: str) -> Path:
    """Pick the decision file for the requested split."""
    name = "matching_results.tsv" if split == "test" else "matching_results_val.tsv"
    return PATH_OUTPUT_DIR / name


def load_top_scores(path: Path, prob_col: str, references: set[str]) -> dict[str, float]:
    """Return the maximum score per reference from a score parquet."""
    df = pd.read_parquet(path)
    if prob_col not in df.columns:
        for candidate in ("prob", "oof_prob", "calibrated_prob", "score"):
            if candidate in df.columns:
                prob_col = candidate
                break
    df = df[df["s1_id"].isin(references)]
    if df.empty:
        return {}
    top = df.groupby("s1_id")[prob_col].max()
    return {str(s1): float(v) for s1, v in top.items()}


def main() -> None:
    """Run Layer 10.5 diagnostics end to end."""
    args = parse_args()
    predictions_path = Path(args.predictions) if args.predictions else default_predictions_path(args.split)

    print("=" * 65)
    print("🚀 LAYER 10.5: DIAGNOSTIC ERROR ANALYSIS + OPEN-SET SPOT-CHECK")
    print("=" * 65)
    print(f"  split={args.split} | predictions={predictions_path}")

    if not predictions_path.exists():
        print(f"  ❌ predictions not found: {predictions_path}")
        print("     run main_l10.py first to produce the decision file.")
        raise SystemExit(1)

    predictions = load_predictions_tsv(predictions_path)
    reference_ids = set(predictions)
    print(f"  references: {len(reference_ids):,}")

    gt_map = load_ground_truth(reference_ids) if args.split == "train" else {}
    metadata = load_reference_metadata(args.split, reference_ids)
    print(f"  metadata loaded: {len(metadata):,} | ground truth: {len(gt_map):,}")

    top_scores: dict[str, float] = {}
    if args.scores:
        top_scores = load_top_scores(Path(args.scores), args.prob_col, reference_ids)
        print(f"  top scores loaded: {len(top_scores):,}")

    seen = set(args.seen_countries)
    open_set_configured = set(args.open_set_countries)

    # ------------------------------------------------------------------
    # Build per-entity diagnostic records.
    # ------------------------------------------------------------------
    records: list[dict] = []
    open_set_records: list[dict] = []
    for s1_id in reference_ids:
        pred = predictions.get(s1_id, [])
        meta = metadata.get(s1_id, {})
        country = assign_country(meta.get("country_norm", "other")) if meta else "other"
        is_open = country in open_set_configured or country not in seen

        if is_open:
            open_set_records.append(
                {"s1_id": s1_id, "pred": pred, "top_score": top_scores.get(s1_id), "meta": meta}
            )

        if not gt_map:
            continue

        gt = gt_map.get(s1_id, set())
        outcome = classify_entity(pred, gt)
        name_core = meta.get("name_core") or ""
        addr_norm = meta.get("addr_norm") or ""
        missing_name = bool(meta.get("is_missing_name", not name_core))
        missing_addr = bool(meta.get("is_missing_addr", not addr_norm))

        records.append(
            {
                **outcome,
                "country": country,
                "is_open_set": is_open,
                "gt_source": source_composition(gt),
                "pred_source": source_composition(pred),
                "name_bucket": "missing" if missing_name else length_bucket(len(name_core)),
                "addr_bucket": "missing" if missing_addr else length_bucket(len(addr_norm)),
                "script": str(meta.get("script_type", "unknown")),
                "pred_len": len(pred),
                "gt_len": len(gt),
            }
        )

    spot = france_spot_check(open_set_records, top_n=args.top_n_france)

    # ------------------------------------------------------------------
    # Overall + grouped summaries (train split only).
    # ------------------------------------------------------------------
    overall = aggregate(records) if records else None
    sanity = None
    if records:
        gt_only = {s1: gt_map[s1] for s1 in reference_ids if s1 in gt_map}
        sanity = official_macro_f05(predictions, gt_only)["macro_f05"]

    dimensions = ("country", "gt_source", "pred_source", "name_bucket", "addr_bucket", "script")
    grouped = {
        dim: group_metrics(records, dim, top_n=args.top_groups) for dim in dimensions
    } if records else {}
    focus = error_focus(records) if records else None

    # ------------------------------------------------------------------
    # Console report.
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("📊 LAYER 10.5 DIAGNOSTIC REPORT")
    print("=" * 65)
    if records and overall:
        print(f"  entities scored : {overall['n']:,}")
        print(f"  macro F0.5      : {overall['macro_f05']:.4f} (sanity vs scorer: {sanity:.4f})")
        print(f"  singleton F0.5  : {overall['singleton_f05']:.4f} over {overall['n_singletons']:,}")
        print(f"  match F0.5      : {overall['match_f05']:.4f}")
        print("\n  error classes (count / share / F0.5 mass):")
        for cls in ("true_singleton", "false_positive", "miss", "exact", "partial"):
            count = focus["counts"][cls]
            share = focus["fraction"][cls]
            mass = focus["f05_mass"][cls]
            print(f"    {cls:<15} {count:>9,}  {share * 100:>6.2f}%  {mass:>8.2f}")
        print("\n  by country:")
        for country, stats in grouped.get("country", {}).items():
            fp = stats["error_classes"]["false_positive"]
            miss = stats["error_classes"]["miss"]
            print(
                f"    {country:<10} n={stats['n']:>7,} F0.5={stats['macro_f05']:.4f} "
                f"FP={fp:,} miss={miss:,}"
            )
        print("\n  by matched-source composition (pred):")
        for comp, stats in grouped.get("pred_source", {}).items():
            print(f"    {comp:<8} n={stats['n']:>7,} F0.5={stats['macro_f05']:.4f}")
        print("\n  by name length bucket:")
        for bucket, stats in grouped.get("name_bucket", {}).items():
            print(f"    {bucket:<8} n={stats['n']:>7,} F0.5={stats['macro_f05']:.4f}")
    else:
        print("  no ground truth available — error analysis skipped (test split)")

    print("\n  open-set spot-check:")
    print(f"    references   : {spot['n']:,}")
    print(f"    non-empty    : {spot['n_non_empty']:,} | empty: {spot['n_empty']:,}")
    if spot["top_score_quantiles"]:
        q = spot["top_score_quantiles"]
        print(f"    top-score pct: p50={q.get('p50')} p75={q.get('p75')} p100={q.get('p100')}")
    for sample in spot["samples"][:5]:
        print(f"      {sample['s1_id']} score={sample['top_score']} n={sample['n_matched']} → {sample['matched']}")

    # ------------------------------------------------------------------
    # Persist report.
    # ------------------------------------------------------------------
    report = {
        "config": {
            "split": args.split,
            "predictions": str(predictions_path),
            "scores": args.scores,
            "top_n_france": args.top_n_france,
            "seen_countries": args.seen_countries,
            "open_set_countries": args.open_set_countries,
        },
        "references": len(reference_ids),
        "overall": overall,
        "macro_f05_sanity": sanity,
        "error_focus": focus,
        "grouped": grouped,
        "open_set_spot_check": spot,
    }
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PATH_OUTPUT_DIR / "l10_diagnostics_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = PATH_OUTPUT_DIR / "l10_diagnostics_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Layer 10.5 Diagnostic Report\n\n")
        f.write(f"- split: `{args.split}` | predictions: `{predictions_path}`\n")
        f.write(f"- references: `{len(reference_ids):,}`\n\n")
        if records and overall:
            f.write("## Overall\n\n")
            f.write(f"- macro F0.5: `{overall['macro_f05']:.4f}` (scorer: `{sanity:.4f}`)\n")
            f.write(f"- singleton F0.5: `{overall['singleton_f05']:.4f}` over `{overall['n_singletons']:,}`\n")
            f.write(f"- match F0.5: `{overall['match_f05']:.4f}`\n\n")
            f.write("## Error classes\n\n")
            f.write("| class | count | share | F0.5 mass |\n|---|---|---|---|\n")
            for cls in ("true_singleton", "false_positive", "miss", "exact", "partial"):
                f.write(
                    f"| {cls} | {focus['counts'][cls]:,} | {focus['fraction'][cls] * 100:.2f}% | "
                    f"{focus['f05_mass'][cls]:.2f} |\n"
                )
            f.write("\n## Breakdown by country\n\n")
            f.write("| country | n | macro F0.5 | false positives | misses |\n|---|---|---|---|---|\n")
            for country, stats in grouped.get("country", {}).items():
                f.write(
                    f"| {country} | {stats['n']:,} | {stats['macro_f05']:.4f} | "
                    f"{stats['error_classes']['false_positive']:,} | "
                    f"{stats['error_classes']['miss']:,} |\n"
                )
            f.write("\n## Breakdown by matched-source composition (pred)\n\n")
            f.write("| composition | n | macro F0.5 |\n|---|---|---|\n")
            for comp, stats in grouped.get("pred_source", {}).items():
                f.write(f"| {comp} | {stats['n']:,} | {stats['macro_f05']:.4f} |\n")
            f.write("\n## Breakdown by name length\n\n")
            f.write("| bucket | n | macro F0.5 |\n|---|---|---|\n")
            for bucket, stats in grouped.get("name_bucket", {}).items():
                f.write(f"| {bucket} | {stats['n']:,} | {stats['macro_f05']:.4f} |\n")
            f.write("\n## Breakdown by address length\n\n")
            f.write("| bucket | n | macro F0.5 |\n|---|---|---|\n")
            for bucket, stats in grouped.get("addr_bucket", {}).items():
                f.write(f"| {bucket} | {stats['n']:,} | {stats['macro_f05']:.4f} |\n")
        f.write("\n## Open-set (France) spot-check\n\n")
        f.write(f"- references: `{spot['n']:,}`\n")
        f.write(f"- non-empty: `{spot['n_non_empty']:,}` | empty: `{spot['n_empty']:,}`\n")
        if spot["top_score_quantiles"]:
            f.write(f"- top-score quantiles: `{spot['top_score_quantiles']}`\n")
        f.write("\n| s1_id | top score | n matched | matched (first 5) | name |\n|---|---|---|---|---|\n")
        for sample in spot["samples"]:
            f.write(
                f"| {sample['s1_id']} | {sample['top_score']} | {sample['n_matched']} | "
                f"{', '.join(sample['matched'])} | {sample['name_core']} |\n"
            )

    print(f"\n  saved report: {report_path}")
    print(f"  saved report: {md_path}")
    print("=" * 65)
    print("🌟 LAYER 10.5 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
