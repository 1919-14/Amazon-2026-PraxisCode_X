"""Layer 3 entry point: country-stratified, block-wise multi-channel blocking.

Builds channels A/C/D over the candidate pool (S2+S3), retrieves candidates for the
requested reference set (default: the Layer 1 validation split), measures per-channel
and union recall, and writes blocking artifacts + a report.

Memory
------
The engine is block-wise and disk-backed (see ``l3_l5_blocking/blocked.py``): one
candidate block index is resident at a time, references stream past it, and rows
are decoded and written in chunks. Peak memory no longer scales with the country
bucket, so the train buckets (6.2M / 4.1M candidates) fit the 2 GB budget.

Usage
-----
# Validation run (all countries, all candidates, tuned for the recall report):
python business_entity_resolution/src/main_l3.py

# Fast smoke test on a small slice:
python business_entity_resolution/src/main_l3.py --countries us --max-candidates 50000 --max-refs 2000

# Test-split candidate generation (no ground truth -> no recall):
python business_entity_resolution/src/main_l3.py --split test --refs all

# Affordably-sized training pool: block a deterministic reference sample first.
python business_entity_resolution/src/main_l3.py --refs train --ref-sample 400000
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
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from config import (
    L3_BLOCK_SIZE,
    L3_CHANNEL_TOPK,
    L3_COUNTRIES,
    L3_ENABLE_CHAR_CHANNEL,
    L3_KEEP_SCRATCH,
    L3_MAX_CANDIDATES,
    L3_MAX_REFS,
    L3_MEMORY_BUDGET_MB,
    L3_REF_BLOCK_SIZE,
    L3_VOCAB_SAMPLE,
    PATH_OUTPUT_DIR,
    PATH_TRAIN_GT,
    SEED,
)
from ingest import EXPECTED_GT_COLUMNS
from l3_l5_blocking.artifacts import (
    blocking_scratch_dir,
    l3_artifact_path,
    l3_artifact_run_key,
)
from l3_l5_blocking.blocked import (
    CHANNEL_A,
    CHANNEL_C,
    CHANNEL_D,
    build_country_index,
    decode_ids,
    query_reference_block,
)
from l3_l5_blocking.buckets import aliases_for, iter_country_reference_blocks, iter_source_shards
from l3_l5_blocking.recall import RecallAccumulator, aggregate_stats, format_report
from utils.coverage import (
    CoverageError,
    check_normalized_shards,
    count_references_by_country,
    normalize_countries,
)
from utils.reports import merge_json_report, render_markdown_table, write_text

try:
    from src.l1_validation.split_generator import load_split_ids
except ImportError:  # pragma: no cover - path fallback
    from l1_validation.split_generator import load_split_ids

L3_SCHEMA = pa.schema(
    [
        ("source1_entity_id", pa.string()),
        ("country", pa.string()),
        ("channel_a", pa.list_(pa.string())),
        ("channel_c", pa.list_(pa.string())),
        ("channel_d", pa.list_(pa.string())),
    ]
)

ROW_CHUNK = 20_000


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
    parser.add_argument(
        "--block-size",
        type=int,
        default=L3_BLOCK_SIZE,
        help="Candidates per block (bounds peak memory).",
    )
    parser.add_argument("--ref-block-size", type=int, default=L3_REF_BLOCK_SIZE)
    parser.add_argument(
        "--vocab-sample",
        type=int,
        default=L3_VOCAB_SAMPLE,
        help="Documents sampled to fit the shared vocabulary + IDF.",
    )
    parser.add_argument(
        "--ref-sample",
        type=int,
        default=None,
        help="Block a deterministic random sample of each country's references "
        "(the affordable way to build a training pool).",
    )
    parser.add_argument("--ref-seed", type=int, default=SEED)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Keep artifacts already on disk (resume an interrupted multi-country run).",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        default=L3_KEEP_SCRATCH,
        help="Keep the per-country block index after the run.",
    )
    parser.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="Proceed even when a country bucket is empty or normalized shards are missing.",
    )
    args = parser.parse_args()
    # Accept both `--countries us india` and `--countries us,india`.
    args.countries = normalize_countries(args.countries)
    return args


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


def country_reference_ids(split: str, country: str, scope_ids: set[str] | None) -> list[str]:
    """Every Source 1 id of one country bucket (used to draw a reference sample)."""
    aliases = aliases_for(country)
    ids: list[str] = []
    for shard in iter_source_shards(split, 1):
        df = pd.read_parquet(shard, columns=["entity_id", "country_norm"])
        if scope_ids is not None:
            df = df[df["entity_id"].isin(scope_ids)]
        df = df[df["country_norm"].isin(aliases)]
        if not df.empty:
            ids.extend(df["entity_id"].tolist())
    return ids


def sample_references(ids: list[str], size: int, seed: int) -> list[str]:
    """Deterministically sample ``size`` reference ids (order-independent of shards)."""
    import random

    if size >= len(ids):
        return sorted(ids)
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(ids), size))


def peak_rss_mb() -> float:
    """Current process peak RSS in MB (0.0 when psutil is unavailable)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001 - metrics must never break a run
        return 0.0


def iter_result_rows(meta, result, chunk: int = ROW_CHUNK):
    """Yield ``(reference_ids, per_channel_lists)`` chunks from a query result."""
    total = len(result)
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        reference_ids = result.reference_ids[start:end]
        per_channel = {
            channel: decode_ids(meta, result.keys[channel][start:end])
            for channel in result.channels
        }
        yield reference_ids, per_channel


def iter_artifact_rows(path: Path, chunk: int = ROW_CHUNK):
    """Stream an existing L3 artifact as ``(reference_ids, per_channel)`` chunks."""
    parquet_file = pq.ParquetFile(str(path))
    columns = ["source1_entity_id", "channel_a", "channel_c", "channel_d"]
    for batch in parquet_file.iter_batches(batch_size=chunk, columns=columns):
        data = batch.to_pydict()
        yield (
            data["source1_entity_id"],
            {
                CHANNEL_A: data["channel_a"],
                CHANNEL_C: data["channel_c"],
                CHANNEL_D: data["channel_d"],
            },
        )


def main() -> None:
    """Run Layer 3 blocking end to end."""
    args = parse_args()
    enable_char = L3_ENABLE_CHAR_CHANNEL and not args.no_char

    print("=" * 65)
    print("🚀 LAYER 3: COUNTRY-STRATIFIED BLOCK-WISE BLOCKING")
    print("=" * 65)
    print(f"  split={args.split} | refs={args.refs} | countries={args.countries}")
    print(f"  topk={args.topk} | char_channel={enable_char}")
    print(
        f"  block_size={args.block_size:,} | ref_block_size={args.ref_block_size:,} | "
        f"vocab_sample={args.vocab_sample:,}"
    )
    if args.max_candidates is not None or args.max_refs is not None:
        print(f"  ⚠️  DEV CAPS: max_candidates={args.max_candidates} max_refs={args.max_refs}")
    if args.ref_sample is not None:
        print(f"  reference sample: {args.ref_sample:,} per country (seed={args.ref_seed})")

    check_normalized_shards(args.split, allow_missing=args.allow_missing_data)

    ref_ids = resolve_reference_ids(args)
    gt_map = load_ground_truth(ref_ids) if args.split == "train" else {}
    has_ground_truth = bool(gt_map)

    expected_by_country: dict[str, int] = {}
    if args.ref_sample is None:
        try:
            expected_by_country = dict(count_references_by_country(args.split, args.refs))
        except Exception as exc:  # noqa: BLE001 - a count is not worth failing a run
            print(f"  (could not pre-count references: {exc})")

    per_country_stats: dict[str, dict] = {}
    per_country_summary: dict[str, dict] = {}
    overall_accumulator = RecallAccumulator()
    peak_mb = 0.0
    t0 = time.time()

    for country in args.countries:
        print(f"\n{'─' * 65}\n🌍 Country bucket: {country}\n{'─' * 65}")
        artifact_path = l3_artifact_path(args.split, args.refs, country)

        if args.reuse_existing and artifact_path.exists():
            rows = pq.ParquetFile(str(artifact_path)).metadata.num_rows
            print(f"  ♻️  reusing existing artifact ({rows:,} references): {artifact_path}")
            accumulator = RecallAccumulator()
            for reference_ids, per_channel in iter_artifact_rows(artifact_path):
                if has_ground_truth:
                    accumulator.add_batch(reference_ids, per_channel, gt_map)
                    overall_accumulator.add_batch(reference_ids, per_channel, gt_map)
            stats = accumulator.stats() if has_ground_truth else {}
            if stats:
                per_country_stats[country] = stats
                print("\n" + format_report(stats))
            per_country_summary[country] = {
                "references": rows,
                "candidates": None,
                "reused": True,
                "artifact": str(artifact_path),
                "recall": stats.get("UNION", {}).get("micro_recall") if stats else None,
            }
            del accumulator
            gc.collect()
            continue

        # --------------------------------------------------------------
        # Reference set (optionally a deterministic sample)
        # --------------------------------------------------------------
        country_scope_ids = ref_ids
        if args.ref_sample is not None:
            all_ids = country_reference_ids(args.split, country, ref_ids)
            if not all_ids:
                if args.allow_missing_data:
                    print("  ⚠️  no reference records for this bucket — skipping")
                    continue
                raise CoverageError(
                    f"country bucket '{country}' has no Source 1 references in split "
                    f"'{args.split}' (scope '{args.refs}'). Pass --allow-missing-data to skip it."
                )
            sampled = sample_references(all_ids, args.ref_sample, args.ref_seed)
            print(f"  reference sample: {len(sampled):,} of {len(all_ids):,} in this bucket")
            country_scope_ids = set(sampled)
            expected_by_country[country] = len(sampled)
            del all_ids
            gc.collect()

        # --------------------------------------------------------------
        # Build the block index (one candidate block resident at a time)
        # --------------------------------------------------------------
        scratch = blocking_scratch_dir(args.split, args.refs, country)
        meta = build_country_index(
            args.split,
            country,
            refs=args.refs,
            scratch_dir=scratch,
            block_size=args.block_size,
            vocab_sample=args.vocab_sample,
            enable_char=enable_char,
            max_candidates=args.max_candidates,
            progress=print,
        )
        peak_mb = max(peak_mb, peak_rss_mb())

        if meta.n_candidates == 0 or meta.n_blocks == 0:
            # The index build created a scratch dir even though there is nothing to
            # block, so remove it before bailing out (or the sweep leaves litter).
            meta.cleanup()
            if args.allow_missing_data:
                print("  ⚠️  no candidates for this bucket — skipping")
                continue
            raise CoverageError(
                f"country bucket '{country}' has no Source 2/3 candidates in split "
                f"'{args.split}'. Pass --allow-missing-data to skip it."
            )

        # --------------------------------------------------------------
        # Stream references past the persisted index, writing rows as we go
        # --------------------------------------------------------------
        accumulator = RecallAccumulator()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(str(artifact_path), L3_SCHEMA)
        n_references = 0
        n_written = 0

        reference_blocks = iter_country_reference_blocks(
            args.split,
            country,
            args.ref_block_size,
            ref_ids=country_scope_ids,
            max_refs=args.max_refs,
        )
        try:
            for block_idx, reference_df in enumerate(reference_blocks, start=1):
                result = query_reference_block(
                    meta, reference_df, topk=args.topk, progress=print
                )
                del reference_df
                gc.collect()

                for reference_ids, per_channel in iter_result_rows(meta, result):
                    writer.write_table(
                        pa.table(
                            {
                                "source1_entity_id": reference_ids,
                                "country": [country] * len(reference_ids),
                                "channel_a": per_channel[CHANNEL_A],
                                "channel_c": per_channel[CHANNEL_C],
                                "channel_d": per_channel[CHANNEL_D],
                            },
                            schema=L3_SCHEMA,
                        )
                    )
                    n_written += len(reference_ids)
                    if has_ground_truth:
                        accumulator.add_batch(reference_ids, per_channel, gt_map)
                        overall_accumulator.add_batch(reference_ids, per_channel, gt_map)

                n_references += len(result)
                print(
                    f"    reference block {block_idx} | {len(result):>7,} refs | "
                    f"written {n_written:>9,} | peak RSS {peak_rss_mb():.0f} MB"
                )
                peak_mb = max(peak_mb, peak_rss_mb())
                del result
                gc.collect()
        finally:
            writer.close()

        stats = accumulator.stats() if has_ground_truth else {}
        if stats:
            per_country_stats[country] = stats
            print("\n" + format_report(stats))
        per_country_summary[country] = {
            "references": n_written,
            "expected_references": expected_by_country.get(country),
            "candidates": meta.n_candidates,
            "blocks": meta.n_blocks,
            "vocab_size_c": meta.vocab_size_c,
            "vocab_size_d": meta.vocab_size_d,
            "reused": False,
            "artifact": str(artifact_path),
            "artifact_mb": round(artifact_path.stat().st_size / (1024 * 1024), 1),
            "recall": stats.get("UNION", {}).get("micro_recall") if stats else None,
        }

        del accumulator
        if not args.keep_scratch:
            meta.cleanup()
        del meta
        gc.collect()

    # ------------------------------------------------------------------
    # Overall report
    # ------------------------------------------------------------------
    overall_report = aggregate_stats(per_country_stats) if has_ground_truth else {}
    elapsed = time.time() - t0
    total_references = sum(int(summary.get("references") or 0) for summary in per_country_summary.values())

    print("\n" + "=" * 65)
    print("📊 LAYER 3 BLOCKING REPORT (overall)")
    print("=" * 65)
    if overall_report:
        print(format_report(overall_report))
    else:
        print("  (no ground truth available for reference set — candidate generation only)")
    print(f"\n  references processed : {total_references:,}")
    print(f"  elapsed              : {elapsed:.1f}s")
    print(f"  peak RSS             : {peak_mb:.0f} MB (budget {L3_MEMORY_BUDGET_MB} MB)")
    if peak_mb > L3_MEMORY_BUDGET_MB:
        print("  ⚠️  peak memory exceeded the budget — lower --block-size / --ref-block-size")

    # ------------------------------------------------------------------
    # Persist reports (merged: previous runs are never overwritten)
    # ------------------------------------------------------------------
    run_key = f"{args.split}_{args.refs}"
    payload = {
        "config": {
            "split": args.split,
            "refs": args.refs,
            "countries": args.countries,
            "topk": args.topk,
            "char_channel": enable_char,
            "max_candidates": args.max_candidates,
            "max_refs": args.max_refs,
            "block_size": args.block_size,
            "ref_block_size": args.ref_block_size,
            "vocab_sample": args.vocab_sample,
            "ref_sample": args.ref_sample,
            "ref_seed": args.ref_seed,
        },
        "per_country": per_country_summary,
        "overall": overall_report,
        "n_references": total_references,
        "peak_rss_mb": round(peak_mb, 1),
        "memory_budget_mb": L3_MEMORY_BUDGET_MB,
        "elapsed_seconds": round(elapsed, 2),
        "run": run_key,
    }
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = merge_json_report(PATH_OUTPUT_DIR / "l3_blocking_report.json", run_key, payload)

    md_path = PATH_OUTPUT_DIR / "l3_blocking_report.md"
    runs = merged.get("runs", {})
    lines = [
        "# Layer 3 Blocking Report",
        "",
        "Every run is kept: `runs` in `l3_blocking_report.json` holds the full history.",
        "",
        render_markdown_table(
            runs,
            {
                "split": "split",
                "refs": "refs",
                "references": "n_references",
                "peak RSS (MB)": "peak_rss_mb",
                "elapsed (s)": "elapsed_seconds",
            },
            lambda key, run: {
                "split": (run.get("config") or {}).get("split"),
                "refs": (run.get("config") or {}).get("refs"),
            },
            latest=merged.get("latest_run"),
        ),
        "",
        f"## Latest run: `{run_key}`",
        "",
        f"- countries: `{args.countries}` | topk: `{args.topk}` | char channel: `{enable_char}`",
        f"- block size: `{args.block_size:,}` | reference block: `{args.ref_block_size:,}` | "
        f"vocab sample: `{args.vocab_sample:,}`",
        f"- references processed: `{total_references:,}` | elapsed: `{elapsed:.1f}s` | "
        f"peak RSS: `{peak_mb:.0f} MB`",
        "",
    ]
    for country, summary in per_country_summary.items():
        lines.append(
            f"- **{country}**: {summary['references']:,} references | "
            f"{summary['candidates'] if summary['candidates'] is not None else 'n/a'} candidates | "
            f"blocks `{summary.get('blocks', 'n/a')}` | "
            f"union recall `{(summary.get('recall') or 0) * 100:.2f}%`"
            + (" (reused)" if summary.get("reused") else "")
        )
    if overall_report:
        lines.extend(["", "## Overall per-channel recall", "", "```", format_report(overall_report), "```"])
    write_text(md_path, "\n".join(lines) + "\n")

    print(f"\n  saved report: {PATH_OUTPUT_DIR / 'l3_blocking_report.json'} (history preserved)")
    print(f"  saved report: {md_path}")
    print("=" * 65)
    print("🌟 LAYER 3 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except CoverageError as error:
        print(f"\n❌ COVERAGE GUARD: {error}\n")
        raise SystemExit(2) from None
