"""Layer 5 entry point: adaptive candidate truncation.

Consumes the Layer 4 fused artifacts, applies coarse scoring + adaptive
truncation, tunes the score-ratio so the average candidate budget lands in the
challenge's target compactness band (K ~= 5.5 - 7.0), and emits the judged
deliverable plus a blocking report:

    output/candidate_pairs.tsv   (test split; the model's inference input)
    output/candidate_pairs_val.tsv  (train split dev output)
    output/blocking_report.md

Usage
-----
# Validation run (tunes ratio, writes dev candidate pairs + report):
python business_entity_resolution/src/main_l5.py

# Test split (fixed ratio from config; writes the deliverable):
python business_entity_resolution/src/main_l5.py --split test --refs all

# Disable tuning and use the configured ratio:
python business_entity_resolution/src/main_l5.py --no-tune --ratio 0.5
"""

from __future__ import annotations

import argparse
import csv
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

import pandas as pd

from config import (
    L3_COUNTRIES,
    L5_COARSE_RATIO,
    L5_K_MAX,
    L5_K_MAX_TARGET,
    L5_K_MIN,
    L5_K_MIN_TARGET,
    L5_K_TARGET,
    L5_RATIO_GRID,
    PATH_ARTIFACTS_DIR,
    PATH_OUTPUT_DIR,
)
from l3_l5_blocking.buckets import dataset_candidate_rows
from l3_l5_blocking.truncate import adaptive_truncate, coarse_score

# Reuse Layer 3/4 artifact helpers so paths and loading stay consistent.
from main_l3 import load_ground_truth, resolve_reference_ids
from main_l4 import l3_artifact_path, l4_artifact_path, load_l3_artifact


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Layer 5 run."""
    parser = argparse.ArgumentParser(description="Layer 5 adaptive candidate truncation")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--refs", choices=["val", "train", "all"], default="val")
    parser.add_argument("--countries", nargs="+", default=list(L3_COUNTRIES))
    parser.add_argument(
        "--ratio-grid",
        nargs="+",
        type=float,
        default=list(L5_RATIO_GRID),
        help="Coarse-score retention ratios to tune over.",
    )
    parser.add_argument("--ratio", type=float, default=L5_COARSE_RATIO, help="Fixed ratio fallback.")
    parser.add_argument("--k-min", type=int, default=L5_K_MIN)
    parser.add_argument("--k-max", type=int, default=L5_K_MAX)
    parser.add_argument("--no-tune", action="store_true", help="Skip ratio tuning (use --ratio).")
    parser.add_argument("--output", type=str, default=None, help="Override candidate_pairs.tsv path.")
    return parser.parse_args()


def default_output_path(args: argparse.Namespace) -> Path:
    """Pick the candidate pairs destination based on the split/ref set."""
    if args.output:
        return Path(args.output)
    if args.split == "test":
        return PATH_OUTPUT_DIR / "candidate_pairs.tsv"
    return PATH_OUTPUT_DIR / f"candidate_pairs_{args.refs}.tsv"


def load_l4_artifact(path: Path) -> tuple[list[str], list[list[tuple[str, float]]]]:
    """Load fused candidate ids + scores from a Layer 4 parquet artifact."""
    df = pd.read_parquet(path, columns=["source1_entity_id", "candidate_entity_ids", "fused_scores"])
    reference_ids = [str(v) for v in df["source1_entity_id"].tolist()]
    fused: list[list[tuple[str, float]]] = []
    for candidates, scores in zip(df["candidate_entity_ids"].tolist(), df["fused_scores"].tolist()):
        candidates = [] if candidates is None else candidates
        scores = [] if scores is None else scores
        fused.append([(str(c), float(s)) for c, s in zip(candidates, scores)])
    return reference_ids, fused


def choose_ratio(
    ratios: list[float],
    kept_counts: dict[float, int],
    hits: dict[float, int],
    n_references: int,
    total_gt_pairs: int,
    has_ground_truth: bool,
) -> float:
    """Pick the retention ratio that best honours the target compactness band.

    Prefers ratios whose average budget falls inside ``[L5_K_MIN_TARGET,
    L5_K_MAX_TARGET]`` and, among those, the one with the highest recall. If no
    ratio lands in the band, pick the ratio whose average is closest to
    ``L5_K_TARGET``.
    """
    averages = {r: (kept_counts[r] / n_references if n_references else 0.0) for r in ratios}

    if has_ground_truth and total_gt_pairs:
        in_band = [r for r in ratios if L5_K_MIN_TARGET <= averages[r] <= L5_K_MAX_TARGET]
        if in_band:
            return max(in_band, key=lambda r: (hits[r] / total_gt_pairs, -averages[r]))
        return min(ratios, key=lambda r: abs(averages[r] - L5_K_TARGET))

    in_band = [r for r in ratios if L5_K_MIN_TARGET <= averages[r] <= L5_K_MAX_TARGET]
    if in_band:
        return min(in_band, key=lambda r: abs(averages[r] - L5_K_TARGET))
    return min(ratios, key=lambda r: abs(averages[r] - L5_K_TARGET))


def main() -> None:
    """Run Layer 5 truncation end to end."""
    args = parse_args()
    ratios = sorted(set(args.ratio_grid))
    out_path = default_output_path(args)

    print("=" * 65)
    print("🚀 LAYER 5: ADAPTIVE CANDIDATE TRUNCATION")
    print("=" * 65)
    print(f"  split={args.split} | refs={args.refs} | countries={args.countries}")
    print(f"  ratio-grid={ratios} | k_min={args.k_min} | k_max={args.k_max}")
    print(f"  target K band = [{L5_K_MIN_TARGET}, {L5_K_MAX_TARGET}]")

    ref_ids = resolve_reference_ids(args)
    gt_map = load_ground_truth(ref_ids) if args.split == "train" else {}
    has_ground_truth = bool(gt_map)

    kept_counts = {r: 0 for r in ratios}
    hits = {r: 0 for r in ratios}
    kept_non_singleton = {r: 0 for r in ratios}
    n_references = 0
    n_non_singleton = 0
    total_gt_pairs = 0
    missing_artifacts: list[str] = []
    recall = 0.0
    avg_k = 0.0
    t0 = time.time()

    # ------------------------------------------------------------------
    # Pass 1: tune the retention ratio (only when ground truth is available)
    # ------------------------------------------------------------------
    tune_ratios = ratios if (has_ground_truth and not args.no_tune) else [args.ratio]
    if has_ground_truth and not args.no_tune:
        for country in args.countries:
            l4_path = l4_artifact_path(args.split, args.refs, country)
            l3_path = l3_artifact_path(args.split, args.refs, country)
            if not l4_path.exists():
                missing_artifacts.append(country)
                continue

            reference_ids, fused_lists = load_l4_artifact(l4_path)
            l3_ids, l3_channels = load_l3_artifact(l3_path) if l3_path.exists() else ([], {})
            aligned = l3_ids == reference_ids

            for i, (s1_id, fused) in enumerate(zip(reference_ids, fused_lists)):
                if aligned:
                    channel_lists = {ch: l3_channels[ch][i] for ch in l3_channels}
                else:
                    channel_lists = None
                scored = coarse_score(fused, channel_lists)
                truth = gt_map.get(s1_id)

                n_references += 1
                if truth:
                    n_non_singleton += 1
                    total_gt_pairs += len(truth)

                for ratio in tune_ratios:
                    kept = adaptive_truncate(scored, ratio, args.k_min, args.k_max)
                    kept_counts[ratio] += len(kept)
                    if truth:
                        kept_non_singleton[ratio] += len(kept)
                        hits[ratio] += len(set(kept) & truth)

            del fused_lists, l3_channels
            gc.collect()

        chosen_ratio = choose_ratio(
            tune_ratios, kept_counts, hits, n_references, total_gt_pairs, has_ground_truth
        )
        avg_k = kept_counts[chosen_ratio] / n_references if n_references else 0.0
        recall = hits[chosen_ratio] / total_gt_pairs if total_gt_pairs else 0.0
        print(f"\n  tuned ratio = {chosen_ratio} | avg K = {avg_k:.3f} | recall = {recall * 100:.2f}%")
    else:
        chosen_ratio = args.ratio
        print(f"\n  fixed ratio = {chosen_ratio} (no tuning)")

    # ------------------------------------------------------------------
    # Pass 2: write the truncated candidate set
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_kept = 0
    rows_written = 0

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for country in args.countries:
            l4_path = l4_artifact_path(args.split, args.refs, country)
            l3_path = l3_artifact_path(args.split, args.refs, country)
            if not l4_path.exists():
                if country not in missing_artifacts:
                    missing_artifacts.append(country)
                continue

            reference_ids, fused_lists = load_l4_artifact(l4_path)
            l3_ids, l3_channels = load_l3_artifact(l3_path) if l3_path.exists() else ([], {})
            aligned = l3_ids == reference_ids

            for i, (s1_id, fused) in enumerate(zip(reference_ids, fused_lists)):
                channel_lists = (
                    {ch: l3_channels[ch][i] for ch in l3_channels} if aligned else None
                )
                kept = adaptive_truncate(
                    coarse_score(fused, channel_lists), chosen_ratio, args.k_min, args.k_max
                )
                writer.writerow([s1_id, ",".join(kept)])
                total_kept += len(kept)
                rows_written += 1

            del fused_lists, l3_channels
            gc.collect()

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    n_pool = dataset_candidate_rows(args.split)
    cross_product = rows_written * n_pool
    density = (total_kept / cross_product) if cross_product else 0.0
    avg_k_final = total_kept / rows_written if rows_written else 0.0

    print("\n" + "=" * 65)
    print("📊 LAYER 5 BLOCKING REPORT")
    print("=" * 65)
    print(f"  chosen ratio        : {chosen_ratio}")
    print(f"  references written  : {rows_written:,}")
    print(f"  candidates kept     : {total_kept:,}")
    print(f"  avg K per reference : {avg_k_final:.3f}")
    if has_ground_truth and not args.no_tune:
        print(f"  candidate recall    : {recall * 100:.2f}% (micro, over true pairs)")
    print(f"  reduction ratio     : {(1 - density) * 100:.6f}% (density {density:.2e})")
    if missing_artifacts:
        print(f"  ⚠️  missing artifacts for: {missing_artifacts} — run main_l3.py/main_l4.py first")
    print(f"  output              : {out_path}")
    print(f"  elapsed             : {elapsed:.1f}s")

    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_json = PATH_OUTPUT_DIR / "l5_blocking_report.json"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "split": args.split,
                    "refs": args.refs,
                    "countries": args.countries,
                    "ratio_grid": tune_ratios,
                    "chosen_ratio": chosen_ratio,
                    "k_min": args.k_min,
                    "k_max": args.k_max,
                },
                "references": rows_written,
                "candidates_kept": total_kept,
                "avg_k": avg_k_final,
                "candidate_recall": (recall if (has_ground_truth and not args.no_tune) else None),
                "reduction_ratio": 1 - density,
                "missing_artifacts": missing_artifacts,
                "elapsed_seconds": round(elapsed, 2),
            },
            f,
            indent=2,
        )

    report_md = PATH_OUTPUT_DIR / "blocking_report.md"
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# Layer 5 Blocking Report\n\n")
        f.write(f"- split: `{args.split}` | refs: `{args.refs}` | countries: `{args.countries}`\n")
        f.write(f"- target K band: `[{L5_K_MIN_TARGET}, {L5_K_MAX_TARGET}]`\n")
        f.write(f"- chosen retention ratio: `{chosen_ratio}` | k_min={args.k_min} | k_max={args.k_max}\n\n")
        f.write("## Results\n\n")
        f.write(f"- references written: `{rows_written:,}`\n")
        f.write(f"- candidates kept: `{total_kept:,}`\n")
        f.write(f"- average K per reference: `{avg_k_final:.3f}`\n")
        if has_ground_truth and not args.no_tune:
            f.write(f"- candidate recall (micro, true pairs): `{recall * 100:.2f}%`\n")
        f.write(f"- reduction ratio: `{(1 - density) * 100:.6f}%` (candidate density `{density:.2e}`)\n")
        if missing_artifacts:
            f.write(f"- ⚠️ missing artifacts for: `{missing_artifacts}`\n")

    print(f"\n  saved report: {report_json}")
    print(f"  saved report: {report_md}")
    print("=" * 65)
    print("🌟 LAYER 5 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
