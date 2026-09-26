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
from l6_l8_matching.signals import SignalWriter, signals_path
from l3_l5_blocking.buckets import dataset_candidate_rows
from l3_l5_blocking.truncate import adaptive_truncate, coarse_score
from utils.coverage import (
    CoverageError,
    check_country_artifacts,
    normalize_countries,
    planned_reference_counts,
)
from utils.reports import merge_json_report, render_markdown_table, write_text

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
    parser.add_argument(
        "--allow-missing-countries",
        action="store_true",
        help="Write the candidate set even when a country bucket's L4 artifact is missing "
        "(those references would be submitted as empty predictions).",
    )
    parser.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help="Accept an output that does not cover every expected reference.",
    )
    parser.add_argument(
        "--no-signals",
        action="store_true",
        help="Skip the retrieval-signal sidecar (training and inference then both use zeros).",
    )
    args = parser.parse_args()
    # Accept both `--countries us india` and `--countries us,india`.
    args.countries = normalize_countries(args.countries)
    return args


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

    planned = planned_reference_counts(args.split, args.refs, args.countries)
    coverage = check_country_artifacts(
        "l4",
        args.split,
        args.refs,
        args.countries,
        allow_missing=args.allow_missing_countries,
        expected_references=planned,
        verify_counts=True,
        remediation=f"run main_l4.py --split {args.split} --refs {args.refs} first",
    )
    print(
        "  input coverage: "
        + " | ".join(
            f"{country} {share * 100:.1f}%" for country, share in sorted(coverage["coverage"].items())
        )
        if coverage["coverage"]
        else "  input coverage: (no artifacts verified)"
    )
    expected_rows = sum(
        int(planned.get(country, 0)) for country in coverage["present"]
    )

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
    written_by_country: dict[str, int] = {}
    kept_by_country: dict[str, int] = {}

    # Retrieval signals are written next to the candidate set in exactly the same
    # order, so Layer 7 (training) and Layer 11 (inference) both read the *same*
    # values for ret_rrf_score / ret_retriever_agreement instead of one side
    # silently scoring zeros.
    signal_writer = None
    signal_file = None
    if not args.no_signals:
        signal_file = signals_path(args.split, args.refs)
        signal_writer = SignalWriter(signal_file)

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
                written_by_country[country] = written_by_country.get(country, 0) + 1
                kept_by_country[country] = kept_by_country.get(country, 0) + len(kept)

                if signal_writer is not None:
                    rrf_by_candidate = dict(fused)
                    agreement: dict[str, int] = {}
                    if channel_lists:
                        for candidates in channel_lists.values():
                            for candidate in candidates:
                                agreement[candidate] = agreement.get(candidate, 0) + 1
                    n_channels = max(1, len(channel_lists)) if channel_lists else 1
                    signal_writer.add(
                        s1_id,
                        kept,
                        [rrf_by_candidate.get(candidate, 0.0) for candidate in kept],
                        [agreement.get(candidate, 0) / n_channels for candidate in kept],
                    )

            del fused_lists, l3_channels
            gc.collect()

    if signal_writer is not None:
        signal_writer.close()

    # ------------------------------------------------------------------
    # Output coverage verification: every country that was blocked must appear
    # in full. A short count means the artifact was truncated somewhere upstream.
    # ------------------------------------------------------------------
    coverage_by_country: dict[str, float] = {}
    for country in coverage["present"]:
        expected = int(planned.get(country, 0))
        written = written_by_country.get(country, 0)
        coverage_by_country[country] = (written / expected) if expected else 1.0
    short = [
        country
        for country, share in coverage_by_country.items()
        if share < 1.0
    ]
    observed_rows = sum(written_by_country.values())
    if short and not args.allow_partial_coverage:
        detail = ", ".join(
            f"{country}: {written_by_country.get(country, 0):,}/{int(planned.get(country, 0)):,}"
            for country in short
        )
        raise CoverageError(
            f"candidate set covers {observed_rows:,} references but {short} are short "
            f"({detail}). Those Source-1 entities would be submitted as empty "
            f"predictions. Fix: re-run the L3/L4 chain for the affected buckets, or "
            f"pass --allow-partial-coverage."
        )

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

    run_key = f"{args.split}_{args.refs}"
    payload = {
        "config": {
            "split": args.split,
            "refs": args.refs,
            "countries": args.countries,
            "ratio_grid": tune_ratios,
            "chosen_ratio": chosen_ratio,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "signals": str(signal_file) if signal_file is not None else None,
        },
        "references": rows_written,
        "expected_references": expected_rows,
        "coverage": coverage_by_country,
        "candidates_kept": total_kept,
        "avg_k": avg_k_final,
        "candidate_recall": (recall if (has_ground_truth and not args.no_tune) else None),
        "reduction_ratio": 1 - density,
        "missing_artifacts": missing_artifacts,
        "run": run_key,
        "elapsed_seconds": round(elapsed, 2),
    }
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_json = PATH_OUTPUT_DIR / "l5_blocking_report.json"
    merged = merge_json_report(report_json, run_key, payload)

    report_md = PATH_OUTPUT_DIR / "blocking_report.md"
    lines = [
        "# Layer 5 Blocking Report",
        "",
        "Every run is kept: `runs` in `l5_blocking_report.json` holds the full history.",
        "",
        render_markdown_table(
            merged.get("runs", {}),
            {
                "split": "split",
                "refs": "refs",
                "references": "references",
                "avg K": "avg_k",
                "recall": "candidate_recall",
                "ratio": "chosen_ratio",
            },
            lambda key, run: {
                "split": (run.get("config") or {}).get("split"),
                "refs": (run.get("config") or {}).get("refs"),
                "ratio": (run.get("config") or {}).get("chosen_ratio"),
                "recall": (
                    f"{(run['candidate_recall'] * 100):.2f}%"
                    if run.get("candidate_recall") is not None
                    else "n/a"
                ),
            },
            latest=merged.get("latest_run"),
        ),
        "",
        f"## Latest run: `{run_key}`",
        "",
        f"- countries: `{args.countries}` | target K band: `[{L5_K_MIN_TARGET}, {L5_K_MAX_TARGET}]`",
        f"- chosen retention ratio: `{chosen_ratio}` | k_min={args.k_min} | k_max={args.k_max}",
        f"- references written: `{rows_written:,}` of `{expected_rows:,}` expected | "
        f"coverage: `{coverage_by_country}`",
        f"- candidates kept: `{total_kept:,}` | average K: `{avg_k_final:.3f}`",
    ]
    if has_ground_truth and not args.no_tune:
        lines.append(f"- candidate recall (micro, true pairs): `{recall * 100:.2f}%`")
    lines.append(f"- reduction ratio: `{(1 - density) * 100:.6f}%` (candidate density `{density:.2e}`)")
    lines.append(
        f"- retrieval-signal sidecar: `{signal_file.name if signal_file else 'disabled'}`"
    )
    if missing_artifacts:
        lines.append(f"- ⚠️ missing artifacts for: `{missing_artifacts}`")
    write_text(report_md, "\n".join(lines) + "\n")

    print(f"\n  saved report: {report_json} (history preserved)")
    print(f"  saved report: {report_md}")
    if signal_file is not None:
        print(f"  saved signals: {signal_file} ({signal_writer.rows:,} candidate rows)")
    print("=" * 65)
    print("🌟 LAYER 5 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except CoverageError as error:
        print(f"\n❌ COVERAGE GUARD: {error}\n")
        raise SystemExit(2) from None
