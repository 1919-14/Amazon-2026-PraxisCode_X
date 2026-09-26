"""Layer 11 entry point: test inference, submission generation & validation.

The end-to-end inference run:

  1. load the L5 ``candidate_pairs.tsv`` (the exact set the matcher scores);
  2. stream the 27 L7 features per pair over reference batches;
  3. score with the L8 booster, optionally corrected by the L9 calibrator;
  4. apply the L10 decision engine (singleton guard + open-set veto);
  5. write ``output/matching_results.tsv`` and ``output/candidate_pairs.tsv``;
  6. run the official ``validate_submission.py``.

No ground truth is needed: thresholds come from the L10 report, then the L8
report, then conservative defaults.

Usage
-----
python business_entity_resolution/src/main_l11.py --variant a
python business_entity_resolution/src/main_l11.py --variant a --max-references 5000 --no-validate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for _p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import (
    DATASET_DIR,
    L10_MARGIN,
    L10_OPEN_SET_COUNTRIES,
    L10_OPEN_SET_TAU_BOOST,
    L10_OPEN_SET_VETO_MIN_CONFIDENCE,
    L10_SEEN_COUNTRIES,
    PATH_ARTIFACTS_DIR,
    PATH_OUTPUT_DIR,
    WORKSPACE_ROOT,
)
from l11_inference.inference import (
    read_candidate_pairs,
    run_inference,
    write_candidates,
    write_predictions,
)
from l11_inference.scorer import load_booster, load_calibrator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Layer 11 run."""
    parser = argparse.ArgumentParser(description="Layer 11 test inference")
    parser.add_argument("--variant", choices=["a", "b"], default="a")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument(
        "--candidate-pairs",
        type=str,
        default=str(PATH_OUTPUT_DIR / "candidate_pairs.tsv"),
        help="L5 candidate pairs TSV (input).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Booster path (default: artifacts/models/lgbm_variant_<variant>.txt).",
    )
    parser.add_argument(
        "--calibrator",
        type=str,
        default=None,
        help="L9 calibrator JSON (default: artifacts/models/isotonic_<variant>.json).",
    )
    parser.add_argument("--no-calibration", action="store_true", help="Ignore the L9 calibrator.")
    parser.add_argument("--tau-match", type=float, default=None)
    parser.add_argument("--tau-s", type=float, default=None)
    parser.add_argument("--margin", type=float, default=L10_MARGIN)
    parser.add_argument("--seen-countries", nargs="+", default=list(L10_SEEN_COUNTRIES))
    parser.add_argument("--open-set-countries", nargs="+", default=list(L10_OPEN_SET_COUNTRIES))
    parser.add_argument("--open-set-boost", type=float, default=L10_OPEN_SET_TAU_BOOST)
    parser.add_argument("--veto-min-confidence", type=float, default=L10_OPEN_SET_VETO_MIN_CONFIDENCE)
    parser.add_argument("--ref-batch-size", type=int, default=50_000)
    parser.add_argument("--max-references", type=int, default=None, help="Dev cap (disables full coverage).")
    parser.add_argument("--max-shards", type=int, default=None, help="Dev cap on shards per source.")
    parser.add_argument("--idf-max-shards", type=int, default=None, help="Dev cap for the IDF pass.")
    parser.add_argument(
        "--matching-out",
        type=str,
        default=None,
        help="Output matching TSV (default: output/matching_results[_val].tsv).",
    )
    parser.add_argument(
        "--candidate-out",
        type=str,
        default=None,
        help="Output candidate TSV (default: output/candidate_pairs[_val].tsv).",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip the official validator.")
    return parser.parse_args()


def default_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve model, calibrator, matching and candidate output paths."""
    models_dir = PATH_ARTIFACTS_DIR / "models"
    model_path = Path(args.model) if args.model else models_dir / f"lgbm_variant_{args.variant}.txt"
    calibrator_path = (
        Path(args.calibrator) if args.calibrator else models_dir / f"isotonic_{args.variant}.json"
    )

    suffix = "" if args.split == "test" else "_val"
    matching_path = (
        Path(args.matching_out)
        if args.matching_out
        else PATH_OUTPUT_DIR / f"matching_results{suffix}.tsv"
    )
    candidate_path = (
        Path(args.candidate_out)
        if args.candidate_out
        else PATH_OUTPUT_DIR / f"candidate_pairs{suffix}.tsv"
    )
    return model_path, calibrator_path, matching_path, candidate_path


def resolve_thresholds(args: argparse.Namespace) -> tuple[float, float, str]:
    """Pick (tau_match, tau_s): CLI → L10 report → L8 report → defaults."""
    if args.tau_match is not None and args.tau_s is not None:
        return args.tau_match, args.tau_s, "cli"

    l10_report = PATH_OUTPUT_DIR / "l10_decision_report.json"
    if l10_report.exists():
        try:
            config = json.loads(l10_report.read_text(encoding="utf-8")).get("config", {})
            if config.get("tau_match") is not None and config.get("tau_s") is not None:
                return float(config["tau_match"]), float(config["tau_s"]), "l10_report"
        except (OSError, json.JSONDecodeError):
            pass

    l8_report = PATH_OUTPUT_DIR / "l8_model_report.json"
    if l8_report.exists():
        try:
            entry = json.loads(l8_report.read_text(encoding="utf-8")).get("variants", {}).get(args.variant)
            if entry:
                return float(entry.get("tau_match", 0.5)), float(entry.get("tau_s", 0.3)), "l8_report"
        except (OSError, json.JSONDecodeError):
            pass

    tau_match = args.tau_match if args.tau_match is not None else 0.5
    tau_s = args.tau_s if args.tau_s is not None else 0.3
    return tau_match, tau_s, "default"


def run_validator(matching_path: Path, candidate_path: Path) -> int:
    """Invoke the official submission validator; return its exit code."""
    validator = WORKSPACE_ROOT / "DATA SET" / "student_resource" / "utils" / "validate_submission.py"
    test_dir = DATASET_DIR / "test"
    if not validator.exists():
        print(f"  ⚠️  validator not found: {validator}")
        return 1
    cmd = [
        sys.executable,
        str(validator),
        "--matching",
        str(matching_path),
        "--candidate",
        str(candidate_path),
        "--test-dir",
        str(test_dir),
    ]
    print("  running validator:", " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    return subprocess.run(cmd).returncode


def main() -> None:
    """Run Layer 11 inference end to end."""
    args = parse_args()
    model_path, calibrator_path, matching_path, candidate_path = default_paths(args)
    candidate_in = Path(args.candidate_pairs)

    print("=" * 65)
    print("🚀 LAYER 11: TEST INFERENCE + SUBMISSION GENERATION")
    print("=" * 65)
    print(f"  variant={args.variant} | split={args.split}")
    print(f"  candidates: {candidate_in}")
    print(f"  model     : {model_path}")

    if not model_path.exists():
        print(f"  ❌ model not found: {model_path}")
        print("     run main_l8.py first to train and export the matcher.")
        raise SystemExit(1)
    if not candidate_in.exists():
        print(f"  ❌ candidate pairs not found: {candidate_in}")
        print("     run main_l5.py --split test --refs all first.")
        raise SystemExit(1)

    tau_match, tau_s, source = resolve_thresholds(args)
    print(f"  thresholds: tau_match={tau_match} tau_s={tau_s} margin={args.margin} (from {source})")

    booster = load_booster(model_path)
    calibrator = None
    if not args.no_calibration:
        calibrator = load_calibrator(calibrator_path)
    print(f"  calibrator: {calibrator['mode'] if calibrator else 'none'}")

    # Load candidates once so the map can be re-emitted (completeness-checked).
    refs_file, cand_map = read_candidate_pairs(candidate_in)

    predictions, stats = run_inference(
        split=args.split,
        candidate_pairs_path=candidate_in,
        booster=booster,
        calibrator=calibrator,
        tau_match=tau_match,
        tau_s=tau_s,
        margin=args.margin,
        seen_countries=args.seen_countries,
        open_set_countries=args.open_set_countries,
        open_set_boost=args.open_set_boost,
        veto_min_confidence=args.veto_min_confidence,
        all_references=True,
        ref_batch_size=args.ref_batch_size,
        max_references=args.max_references,
        max_shards=args.max_shards,
        idf_max_shards=args.idf_max_shards,
    )

    # Emit with the same (complete) reference ordering as the predictions.
    reference_ids = list(predictions)
    matching_path.parent.mkdir(parents=True, exist_ok=True)
    n_match = write_predictions(matching_path, predictions, reference_ids)
    n_cand = write_candidates(candidate_path, cand_map, reference_ids)

    print("\n" + "=" * 65)
    print("📊 LAYER 11 INFERENCE REPORT")
    print("=" * 65)
    print(f"  references written : {n_match:,}")
    print(f"  candidates written : {n_cand:,}")
    print(f"  pairs scored       : {stats['pairs_scored']:,}")
    print(f"  pairs missing rec. : {stats['pairs_missing_record']:,}")
    print(f"  non-empty (match)  : {stats['non_empty']:,}")
    print(f"  matched pairs      : {stats['matched_pairs']:,}")
    print(f"  open-set refs      : {stats['open_set_references']:,}")
    print(f"  saved: {matching_path}")
    print(f"  saved: {candidate_path}")

    report = {
        "config": {
            "variant": args.variant,
            "split": args.split,
            "model": str(model_path),
            "calibrator": calibrator["mode"] if calibrator else "none",
            "candidate_pairs": str(candidate_in),
            "tau_match": tau_match,
            "tau_s": tau_s,
            "margin": args.margin,
            "threshold_source": source,
            "max_references": args.max_references,
        },
        "stats": stats,
        "outputs": {"matching": str(matching_path), "candidate": str(candidate_path)},
    }
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PATH_OUTPUT_DIR / "l11_inference_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  saved report: {report_path}")

    if not args.no_validate:
        code = run_validator(matching_path, candidate_path)
        report["validation_exit_code"] = code
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  validator exit code: {code}")
        if code != 0:
            print("  ⚠️  validator reported issues — see output above")

    print("=" * 65)
    print("🌟 LAYER 11 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
