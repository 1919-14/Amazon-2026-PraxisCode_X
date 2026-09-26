"""Layer 3 entry point: country-stratified multi-channel blocking.

Builds channels A/C/D over the candidate pool (S2+S3), retrieves candidates for
the requested reference set (default: the Layer 1 validation split), measures
per-channel and union recall, and writes blocking artifacts + a report.

Usage examples
--------------
# Full validation run (all countries, all candidates):
python business_entity_resolution/src/main_l3.py

# Fast smoke test on a small slice:
python business_entity_resolution/src/main_l3.py --countries us --max-candidates 50000 --max-refs 2000

# Test split candidate generation (no ground truth, no recall):
python business_entity_resolution/src/main_l3.py --split test --refs all
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

# Ensure src/ and project root are importable.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for _p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from config import (
    L3_CHANNEL_TOPK,
    L3_COUNTRIES,
    L3_ENABLE_CHAR_CHANNEL,
    L3_MAX_CANDIDATES,
    L3_MAX_REFS,
    PATH_ARTIFACTS_DIR,
    PATH_OUTPUT_DIR,
    PATH_TRAIN_GT,
)
from ingest import EXPECTED_GT_COLUMNS
from l3_l5_blocking.buckets import load_country_candidates, load_references
from l3_l5_blocking.engine import CHANNEL_A, CHANNEL_C, CHANNEL_D, run_country_blocking
from l3_l5_blocking.recall import channel_stats, format_report, union_stats

try:
    from src.l1_validation.split_generator import load_split_ids
except ImportError:  # pragma: no cover - path fallback
    from l1_validation.split_generator import load_split_ids


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Layer 3 run."""
    parser = argparse.ArgumentParser(description="Layer 3 country-stratified blocking")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--refs",
        choices=["val", "train", "all"],
        default="val",
        help="Reference set to block for (train split only; test always uses all).",
    )
    parser.add_argument("--countries", nargs="+", default=list(L3_COUNTRIES))
    parser.add_argument("--topk", type=int, default=L3_CHANNEL_TOPK)
    parser.add_argument("--max-candidates", type=int, default=L3_MAX_CANDIDATES)
    parser.add_argument("--max-refs", type=int, default=L3_MAX_REFS)
    parser.add_argument("--no-char", action="store_true", help="Disable channel D")
    return parser.parse_args()


def resolve_reference_ids(args: argparse.Namespace) -> set[str] | None:
    """Return the reference id filter for the requested split/refs combination."""
    if args.split == "test" or args.refs == "all":
        return None

    split_ids = load_split_ids()
    key = "val_ids" if args.refs == "val" else "train_ids"
    return set(split_ids[key])


def load_ground_truth(valid_ids: set[str] | None) -> dict[str, set[str]]:
    """Load the ground-truth map, optionally restricted to a reference id set.

    Reads the TSV in chunks so only the requested references are retained in
    memory (the full map holds 2.2M keys).
    """
    if not PATH_TRAIN_GT.exists():
        return {}

    gt_map: dict[str, set[str]] = {}
    for chunk in pd.read_csv(
        PATH_TRAIN_GT,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=EXPECTED_GT_COLUMNS,
        chunksize=500_000,
    ):
        if valid_ids is not None:
            chunk = chunk[chunk["source1_entity_id"].isin(valid_ids)]
        for s1_id, raw in chunk.itertuples(index=False, name=None):
            gt_map[s1_id] = {m.strip() for m in raw.split(",") if m.strip()} if raw else set()
    return gt_map


def save_country_candidates(result, split: str, refs: str) -> Path:
    """Persist per-channel candidate lists for one country to Parquet."""
    out_dir = PATH_ARTIFACTS_DIR / "blocking"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"l3_{split}_{refs}_country={result.country}.parquet"
    df = pd.DataFrame(
        {
            "source1_entity_id": result.reference_ids,
            "country": result.country,
            "channel_a": result.channels.get(CHANNEL_A, []),
            "channel_c": result.channels.get(CHANNEL_C, []),
            "channel_d": result.channels.get(CHANNEL_D, []),
        }
    )
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    return out_path


def main() -> None:
    """Run Layer 3 blocking end to end."""
    args = parse_args()
    enable_char = L3_ENABLE_CHAR_CHANNEL and not args.no_char

    print("=" * 65)
    print("🚀 LAYER 3: COUNTRY-STRATIFIED MULTI-CHANNEL BLOCKING")
    print("=" * 65)
    print(f"  split={args.split} | refs={args.refs} | countries={args.countries}")
    print(f"  topk={args.topk} | char_channel={enable_char}")
    if args.max_candidates is not None or args.max_refs is not None:
        print(f"  ⚠️  DEV CAPS: max_candidates={args.max_candidates} max_refs={args.max_refs}")

    ref_ids = resolve_reference_ids(args)
    gt_map = load_ground_truth(ref_ids) if args.split == "train" else {}
    has_ground_truth = bool(gt_map)

    all_reference_ids: list[str] = []
    per_channel_overall: dict[str, list[list[str]]] = {CHANNEL_A: [], CHANNEL_C: [], CHANNEL_D: []}
    per_country_report: dict[str, dict] = {}
    t0 = time.time()

    for country in args.countries:
        print(f"\n{'─' * 65}\n🌍 Country bucket: {country}\n{'─' * 65}")
        reference_df = load_references(args.split, country, ref_ids, max_refs=args.max_refs)
        candidate_df = load_country_candidates(args.split, country, max_candidates=args.max_candidates)
        print(f"  references: {len(reference_df):,} | candidates: {len(candidate_df):,}")

        if reference_df.empty:
            print("  ⚠️  no reference records for this bucket — skipping")
            continue

        result = run_country_blocking(
            country=country,
            reference_df=reference_df,
            candidate_df=candidate_df,
            topk=args.topk,
            enable_char=enable_char,
        )

        del reference_df, candidate_df
        gc.collect()

        out_path = save_country_candidates(result, args.split, args.refs)
        print(f"  saved candidates: {out_path}")

        all_reference_ids.extend(result.reference_ids)
        for channel_name, cands in result.channels.items():
            per_channel_overall.setdefault(channel_name, []).extend(cands)

        if has_ground_truth:
            country_stats = {
                name: channel_stats(result.reference_ids, cands, gt_map)
                for name, cands in result.channels.items()
            }
            country_stats["UNION"] = union_stats(result.reference_ids, result.channels, gt_map)
            per_country_report[country] = country_stats
            print("\n" + format_report(country_stats))

        del result
        gc.collect()

    # ------------------------------------------------------------------
    # Overall report
    # ------------------------------------------------------------------
    overall_report: dict[str, dict] = {}
    if has_ground_truth and all_reference_ids:
        for name, cands in per_channel_overall.items():
            overall_report[name] = channel_stats(all_reference_ids, cands, gt_map)
        overall_report["UNION"] = union_stats(all_reference_ids, per_channel_overall, gt_map)

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print("📊 LAYER 3 BLOCKING REPORT (overall)")
    print("=" * 65)
    if overall_report:
        print(format_report(overall_report))
    else:
        print("  (no ground truth available for reference set — candidate generation only)")
    print(f"\n  references processed : {len(all_reference_ids):,}")
    print(f"  elapsed              : {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Persist report
    # ------------------------------------------------------------------
    report = {
        "config": {
            "split": args.split,
            "refs": args.refs,
            "countries": args.countries,
            "topk": args.topk,
            "char_channel": enable_char,
            "max_candidates": args.max_candidates,
            "max_refs": args.max_refs,
        },
        "overall": overall_report,
        "per_country": per_country_report,
        "n_references": len(all_reference_ids),
        "elapsed_seconds": round(elapsed, 2),
    }

    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PATH_OUTPUT_DIR / "l3_blocking_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = PATH_OUTPUT_DIR / "l3_blocking_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Layer 3 Blocking Report\n\n")
        f.write(f"- split: `{args.split}` | refs: `{args.refs}` | countries: `{args.countries}`\n")
        f.write(f"- topk: `{args.topk}` | char channel: `{enable_char}`\n")
        f.write(f"- references processed: `{len(all_reference_ids):,}` | elapsed: `{elapsed:.1f}s`\n\n")
        if overall_report:
            f.write("## Overall per-channel recall\n\n")
            f.write(format_report(overall_report))
            f.write("\n")

    print(f"\n  saved report: {report_path}")
    print(f"  saved report: {md_path}")
    print("=" * 65)
    print("🌟 LAYER 3 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
