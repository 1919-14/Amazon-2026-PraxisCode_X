"""Layer 4 entry point: Reciprocal Rank Fusion over the L3 blocking channels.

Reads the per-country candidate artifacts written by Layer 3
(``artifacts/blocking/l3_<split>_<refs>_country=<c>.parquet``), fuses channels
A/C/D with RRF, optionally tunes the smoothing constant ``k`` against the
validation split using recall@N cutoffs, and writes fused candidate artifacts
for Layer 5:

    artifacts/blocking/l4_<split>_<refs>_country=<c>.parquet
    output/l4_rrf_report.json / .md

Memory
------
A full country bucket is far too large to hold in Python objects: the US test
bucket is 663k references x ~370 candidates x 3 channels, i.e. ~700M id strings
(>10 GB), and the earlier implementation loaded exactly that. The layer therefore
streams the L3 artifact in reference batches: the k-grid tuning pass accumulates
only integer hit counts, and the write pass emits each batch's rows before
releasing it. Peak memory is one batch (~``--ref-batch`` references), not the
country. The tuning pass re-reads the artifact (two streaming passes) because the
chosen ``k`` has to be known before any output row can be encoded.

Usage
-----
# Fuse with the best k on validation:
python business_entity_resolution/src/main_l4.py

# Fix a single k (no grid search):
python business_entity_resolution/src/main_l4.py --k-grid 60

# Test split (no ground truth -> fixed k):
python business_entity_resolution/src/main_l4.py --split test --refs all
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for _p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from config import (
    L3_COUNTRIES,
    L4_REF_BATCH,
    PATH_OUTPUT_DIR,
    RRF_K_GRID,
)
from l3_l5_blocking.artifacts import l3_artifact_path, l4_artifact_path
from l3_l5_blocking.engine import CHANNEL_A, CHANNEL_C, CHANNEL_D
from l3_l5_blocking.recall import ranked_recall_counts
from l3_l5_blocking.rrf import fuse, ranked_ids
from utils.coverage import (
    CoverageError,
    check_country_artifacts,
    normalize_countries,
    planned_reference_counts,
)
from utils.reports import merge_json_report, render_markdown_table, write_text

# Reuse Layer 3 helpers so ground-truth loading and reference selection stay identical.
from main_l3 import load_ground_truth, resolve_reference_ids

CHANNEL_COLUMNS = {
    "channel_a": CHANNEL_A,
    "channel_c": CHANNEL_C,
    "channel_d": CHANNEL_D,
}

L4_SCHEMA = pa.schema(
    [
        ("source1_entity_id", pa.string()),
        ("country", pa.string()),
        ("candidate_entity_ids", pa.list_(pa.string())),
        ("fused_scores", pa.list_(pa.float64())),
    ]
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Layer 4 run."""
    parser = argparse.ArgumentParser(description="Layer 4 Reciprocal Rank Fusion")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--refs", choices=["val", "train", "all"], default="val")
    parser.add_argument("--countries", nargs="+", default=list(L3_COUNTRIES))
    parser.add_argument(
        "--k-grid",
        nargs="+",
        type=int,
        default=list(RRF_K_GRID),
        help="RRF smoothing constants to evaluate.",
    )
    parser.add_argument(
        "--best-n",
        type=int,
        default=10,
        help="Recall@N cutoff used to pick the best k.",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        default=[5, 10, 20, 50],
        help="Recall@N cutoffs reported for each k.",
    )
    parser.add_argument(
        "--ref-batch",
        type=int,
        default=L4_REF_BATCH,
        help="References per streaming batch (bounds peak memory).",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Keep fused artifacts already on disk (resume an interrupted run).",
    )
    parser.add_argument(
        "--allow-missing-countries",
        action="store_true",
        help="Fuse the countries that exist even when others are missing (default: fail).",
    )
    args = parser.parse_args()
    # Accept both `--countries us india` and `--countries us,india`.
    args.countries = normalize_countries(args.countries)
    return args


# Path helpers live in l3_l5_blocking.artifacts (single source of truth); they are
# re-exported here because Layer 5 imports them from this module.
__all__ = ["l3_artifact_path", "l4_artifact_path", "main"]


def _to_list(value) -> list[str]:
    """Normalise a parquet list cell to a plain list of strings."""
    if value is None:
        return []
    return [str(v) for v in value]


def load_l3_artifact(path: Path) -> tuple[list[str], dict[str, list[list[str]]]]:
    """Load channel candidate lists from a Layer 3 parquet artifact.

    Kept for callers that genuinely need a whole artifact in memory (small
    buckets, tests). Bulk layers should prefer :func:`iter_l3_artifact_batches`.
    """
    df = pd.read_parquet(path, columns=["source1_entity_id"] + list(CHANNEL_COLUMNS))
    reference_ids = [str(v) for v in df["source1_entity_id"].tolist()]
    channels: dict[str, list[list[str]]] = {}
    for column, channel in CHANNEL_COLUMNS.items():
        channels[channel] = [_to_list(v) for v in df[column].tolist()]
    return reference_ids, channels


def iter_l3_artifact_batches(
    path: Path,
    batch_size: int,
):
    """Yield ``(reference_ids, channels)`` streaming batches from an L3 artifact.

    ``channels`` maps a channel name to a list of candidate-id lists aligned with
    ``reference_ids`` for *this batch only*, so nothing larger than the batch is
    ever resident.
    """
    parquet_file = pq.ParquetFile(str(path))
    columns = ["source1_entity_id"] + list(CHANNEL_COLUMNS)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        data = batch.to_pydict()
        reference_ids = [str(v) for v in data["source1_entity_id"]]
        channels = {
            channel: [_to_list(v) for v in data[column]]
            for column, channel in CHANNEL_COLUMNS.items()
        }
        yield reference_ids, channels


def _write_fused_batch(
    writer: pq.ParquetWriter,
    reference_ids: list[str],
    fused,
    country: str,
) -> None:
    """Append one batch of fused candidates to the output artifact."""
    writer.write_table(
        pa.table(
            {
                "source1_entity_id": reference_ids,
                "country": [country] * len(reference_ids),
                "candidate_entity_ids": [[candidate for candidate, _ in row] for row in fused],
                "fused_scores": [[float(score) for _, score in row] for row in fused],
            },
            schema=L4_SCHEMA,
        )
    )


def main() -> None:
    """Run Layer 4 RRF fusion end to end."""
    args = parse_args()
    k_grid = sorted(set(args.k_grid))
    cutoffs = sorted(set(args.cutoffs))

    print("=" * 65)
    print("🚀 LAYER 4: RECIPROCAL RANK FUSION")
    print("=" * 65)
    print(f"  split={args.split} | refs={args.refs} | countries={args.countries}")
    print(f"  k-grid={k_grid} | best-n={args.best_n} | cutoffs={cutoffs}")
    print(f"  ref-batch={args.ref_batch:,} (streamed, bounds memory)")

    coverage = check_country_artifacts(
        "l3",
        args.split,
        args.refs,
        args.countries,
        allow_missing=args.allow_missing_countries,
        expected_references=planned_reference_counts(args.split, args.refs, args.countries),
        verify_counts=True,
        remediation=f"run main_l3.py --split {args.split} --refs {args.refs} first",
    )
    if coverage["coverage"]:
        print(
            "  input coverage: "
            + " | ".join(
                f"{country} {share * 100:.1f}%"
                for country, share in sorted(coverage["coverage"].items())
            )
        )

    ref_ids = resolve_reference_ids(args)
    gt_map = load_ground_truth(ref_ids) if args.split == "train" else {}
    has_ground_truth = bool(gt_map)

    # Accumulators for the k-grid evaluation (weighted by true-pair counts).
    grid_hits: dict[int, dict[int, int]] = {k: {cut: 0 for cut in cutoffs} for k in k_grid}
    grid_total_pairs = 0

    per_country_report: dict[str, dict] = {}
    total_references = 0
    t0 = time.time()

    for country in args.countries:
        source_path = l3_artifact_path(args.split, args.refs, country)
        print(f"\n{'─' * 65}\n🌍 Country bucket: {country}\n{'─' * 65}")

        if not source_path.exists():
            print(f"  ⚠️  missing L3 artifact: {source_path}")
            print("      run main_l3.py for this country first — skipping")
            continue

        out_path = l4_artifact_path(args.split, args.refs, country)
        if args.reuse_existing and out_path.exists():
            reused_rows = pq.ParquetFile(str(out_path)).metadata.num_rows
            print(
                f"  ♻️  reusing existing fused artifact ({reused_rows:,} references) — "
                "skipped from the k-grid report"
            )
            per_country_report[country] = {
                "chosen_k": None,
                "k_grid": {},
                "reused": True,
                "references": int(reused_rows),
            }
            continue

        # ------------------------------------------------------------------
        # Pass 1: tune k (streamed; accumulates counts only)
        # ------------------------------------------------------------------
        best_k = k_grid[0]
        if has_ground_truth:
            country_stats: dict[str, dict] = {}
            best_score = -1.0
            total_pairs = 0
            n_refs = 0
            # Per-country hit counts, so a later country's stats cannot inherit an
            # earlier country's hits (grid_hits is the cross-country total).
            country_hits: dict[int, dict[int, int]] = {
                k: {cut: 0 for cut in cutoffs} for k in k_grid
            }

            for reference_ids, channels in iter_l3_artifact_batches(source_path, args.ref_batch):
                n_refs += len(reference_ids)
                batch_total_pairs = 0
                for k in k_grid:
                    fused = fuse(channels, k=k)
                    ranked = ranked_ids(fused)
                    hits, batch_total_pairs = ranked_recall_counts(
                        reference_ids, ranked, gt_map, cutoffs
                    )
                    for cut in cutoffs:
                        country_hits[k][cut] += hits[cut]
                        grid_hits[k][cut] += hits[cut]
                    del ranked, fused

                # The true-pair count is independent of k, so it is added once per
                # batch (adding it inside the k loop would inflate the denominator
                # per k and deflate every recall figure).
                total_pairs += batch_total_pairs
                del channels
                gc.collect()

            total_references += n_refs

            for k in k_grid:
                k_stats = {
                    f"recall@{cut}": (country_hits[k][cut] / total_pairs if total_pairs else 0.0)
                    for cut in cutoffs
                }
                country_stats[f"k={k}"] = k_stats
                score = k_stats.get(f"recall@{args.best_n}", 0.0)
                if score > best_score:
                    best_score = score
                    best_k = k

            grid_total_pairs += total_pairs
            per_country_report[country] = {"chosen_k": best_k, "k_grid": country_stats}
            print(f"  references: {n_refs:,} | artifact: {source_path.name}")
            print(f"  chosen k = {best_k}")
            for cut in cutoffs:
                print(
                    f"    recall@{cut:<3} = "
                    f"{country_stats[f'k={best_k}'][f'recall@{cut}'] * 100:6.2f}%"
                )
        else:
            print(f"  no ground truth — using fixed k = {best_k}")
            per_country_report[country] = {"chosen_k": best_k, "k_grid": {}}

        # ------------------------------------------------------------------
        # Pass 2: write the fused artifact (streamed)
        # ------------------------------------------------------------------
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(str(out_path), L4_SCHEMA)
        written = 0
        try:
            for reference_ids, channels in iter_l3_artifact_batches(source_path, args.ref_batch):
                fused = fuse(channels, k=best_k)
                _write_fused_batch(writer, reference_ids, fused, country)
                written += len(reference_ids)
                del fused, channels
                gc.collect()
        finally:
            writer.close()

        if not has_ground_truth:
            total_references += written
        print(f"  saved fused candidates: {out_path} ({written:,} references)")

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Overall k-grid report
    # ------------------------------------------------------------------
    overall_grid: dict[str, dict] = {}
    best_overall_k: int | None = None
    if has_ground_truth and grid_total_pairs:
        for k in k_grid:
            overall_grid[f"k={k}"] = {
                f"recall@{cut}": (grid_hits[k][cut] / grid_total_pairs) for cut in cutoffs
            }
        best_overall_k = max(
            k_grid, key=lambda k: overall_grid[f"k={k}"].get(f"recall@{args.best_n}", 0.0)
        )

    print("\n" + "=" * 65)
    print("📊 LAYER 4 RRF REPORT (overall)")
    print("=" * 65)
    if overall_grid:
        header = "k".ljust(8) + "".join(f"{'recall@' + str(c):>12}" for c in cutoffs)
        print(header)
        print("-" * len(header))
        for k in k_grid:
            row = f"{k:<8}" + "".join(
                f"{overall_grid[f'k={k}'][f'recall@{c}'] * 100:>11.2f}%" for c in cutoffs
            )
            marker = "  ← best" if k == best_overall_k else ""
            print(row + marker)
    else:
        print("  (no ground truth — fusion only)")

    print(f"\n  references processed : {total_references:,}")
    print(f"  elapsed              : {elapsed:.1f}s")

    report = {
        "config": {
            "split": args.split,
            "refs": args.refs,
            "countries": args.countries,
            "k_grid": k_grid,
            "best_n": args.best_n,
            "cutoffs": cutoffs,
            "ref_batch": args.ref_batch,
        },
        "overall_k_grid": overall_grid,
        "best_k": best_overall_k,
        "per_country": per_country_report,
        "n_references": total_references,
        "elapsed_seconds": round(elapsed, 2),
    }

    run_key = f"{args.split}_{args.refs}"
    report["coverage"] = coverage
    report["run"] = run_key
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PATH_OUTPUT_DIR / "l4_rrf_report.json"
    merged = merge_json_report(report_path, run_key, report)

    md_path = PATH_OUTPUT_DIR / "l4_rrf_report.md"
    lines = [
        "# Layer 4 RRF Report",
        "",
        "Every run is kept: `runs` in `l4_rrf_report.json` holds the full history.",
        "",
        render_markdown_table(
            merged.get("runs", {}),
            {
                "split": "split",
                "refs": "refs",
                "best k": "best_k",
                "references": "n_references",
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
        f"- countries: `{args.countries}` | k-grid: `{k_grid}` | best-n: `{args.best_n}` | cutoffs: `{cutoffs}`",
        f"- references processed: `{total_references:,}` | elapsed: `{elapsed:.1f}s`",
        f"- input coverage: `{coverage['coverage']}`",
        "",
    ]
    if overall_grid:
        lines.append(f"## Overall recall by k (best k = {best_overall_k})")
        lines.append("")
        lines.append("| k | " + " | ".join(f"recall@{c}" for c in cutoffs) + " |")
        lines.append("|" + "---|" * (len(cutoffs) + 1))
        for k in k_grid:
            cells = " | ".join(
                f"{overall_grid[f'k={k}'][f'recall@{c}'] * 100:.2f}%" for c in cutoffs
            )
            lines.append(f"| {k} | {cells} |")
    write_text(md_path, "\n".join(lines) + "\n")

    print(f"\n  saved report: {report_path} (history preserved)")
    print(f"  saved report: {md_path}")
    print("=" * 65)
    print("🌟 LAYER 4 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except CoverageError as error:
        print(f"\n❌ COVERAGE GUARD: {error}\n")
        raise SystemExit(2) from None
